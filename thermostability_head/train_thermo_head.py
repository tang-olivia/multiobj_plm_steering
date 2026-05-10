import gc
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import esm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

CSV_PATH = "tm_sequence_dataset.csv"
TRAIN_CSV = "train_split.csv"
TEST_CSV = "test_split.csv"
TRAIN_CACHE = "pooled_train.pt"
TEST_CACHE = "pooled_test.pt"
ESM_NAME = "esm2_t33_650M_UR50D"
ESM_LAYER = 33
EMBED_DIM = 1280

# ESM2 forward pass is the bottleneck; this batch is for the precompute step
# only. Training on cached features uses HEAD_BATCH_SIZE below.
ESM_BATCH_SIZE = 8

HEAD_BATCH_SIZE = 256
EPOCHS = 50
PEAK_LR = 5e-4
WEIGHT_DECAY = 1e-2
WARMUP_FRACTION = 0.05
MAX_LEN = 1022  # ESM2 has 1024 positions, with CLS + EOS leaving 1022 for residues
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_seqs(batch):
    seqs, indices = zip(*batch)
    return list(seqs), torch.tensor(indices, dtype=torch.long)


class SequenceIndexDataset(Dataset):
    """Yields (sequence, original_index) pairs for the precompute pass."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][:MAX_LEN]
        return seq, idx


class ThermoHead(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, pooled):
        # pooled: (B, E) — mean-pooled features over real residues
        x = self.dense(pooled)
        x = F.gelu(x)
        x = self.layer_norm(x)
        return self.out_proj(x).squeeze(-1)

    @torch.no_grad()
    def init_from_esm_lm_head(self, lm_head):
        self.dense.weight.data.copy_(lm_head.dense.weight.data)
        self.dense.bias.data.copy_(lm_head.dense.bias.data)
        self.layer_norm.weight.data.copy_(lm_head.layer_norm.weight.data)
        self.layer_norm.bias.data.copy_(lm_head.layer_norm.bias.data)


@torch.inference_mode()
def precompute_pooled_feats(
    sequences,
    esm_model,
    alphabet,
    batch_converter,
    cache_path,
    desc,
    use_cache: bool = True,
):
    cache_meta = {
        "n": len(sequences),
        "max_len": MAX_LEN,
        "esm": ESM_NAME,
        "layer": ESM_LAYER,
        "embed_dim": EMBED_DIM,
    }

    if use_cache:
        if cache_path is None:
            raise ValueError("cache_path is required when use_cache=True")
        cache_file = Path(cache_path)
        if cache_file.exists():
            blob = torch.load(cache_file, map_location="cpu", weights_only=False)
            if blob.get("meta") == cache_meta:
                print(f"[{desc}] Loaded {len(sequences)} cached pooled features from {cache_path}")
                return blob["pooled"]
            print(
                f"[{desc}] Cache {cache_path} exists but metadata mismatch "
                f"(got {blob.get('meta')}, want {cache_meta}); recomputing."
            )

    if use_cache:
        print(f"[{desc}] Cache miss; running ESM2 over {len(sequences)} sequences...")
    else:
        print(f"[{desc}] Running ESM2 over {len(sequences)} sequences (caching disabled)...")
    pooled = torch.empty(len(sequences), EMBED_DIM, dtype=torch.float32)

    dataset = SequenceIndexDataset(sequences)
    loader = DataLoader(
        dataset,
        batch_size=ESM_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_seqs,
    )

    use_bf16 = (
        DEVICE == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for chunk_seqs, chunk_idx in tqdm(loader, desc=f"Precompute {desc}"):
        data = [(str(i), s) for i, s in enumerate(chunk_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(DEVICE == "cuda")):
            out = esm_model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)
        reps = out["representations"][ESM_LAYER].float()

        valid = (
            (tokens != alphabet.padding_idx)
            & (tokens != alphabet.cls_idx)
            & (tokens != alphabet.eos_idx)
        ).unsqueeze(-1).float()

        denom = valid.sum(dim=1).clamp(min=1.0)
        chunk_pooled = (reps * valid).sum(dim=1) / denom
        pooled[chunk_idx] = chunk_pooled.cpu()

    if use_cache:
        cache_file = Path(cache_path)
        torch.save({"pooled": pooled, "meta": cache_meta}, cache_file)
        print(f"[{desc}] Saved {len(sequences)} pooled features to {cache_path}")
    return pooled


def evaluate(loader, head, tm_mean, tm_std):
    head.eval()
    preds_all = []
    y_all = []

    with torch.no_grad():
        for pooled, y in loader:
            pooled = pooled.to(DEVICE)
            y = y.to(DEVICE)

            preds_norm = head(pooled)
            preds = preds_norm * tm_std + tm_mean

            preds_all.extend(preds.cpu().numpy())
            y_all.extend(y.cpu().numpy())

    preds_all = pd.Series(preds_all)
    y_all = pd.Series(y_all)

    mse = ((preds_all - y_all) ** 2).mean()
    mae = (preds_all - y_all).abs().mean()
    spearman = spearmanr(preds_all, y_all).correlation
    pearson = pearsonr(preds_all, y_all)[0]

    return mse, mae, spearman, pearson


def build_param_groups(module, weight_decay):
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "layer_norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print("Train Tm stats:")
    print(train_df["median_Tm"].describe())
    print("Test Tm stats:")
    print(test_df["median_Tm"].describe())

    tm_mean = torch.tensor(train_df["median_Tm"].mean(), dtype=torch.float32).to(DEVICE)
    tm_std = torch.tensor(train_df["median_Tm"].std(), dtype=torch.float32).to(DEVICE)

    train_y = torch.tensor(train_df["median_Tm"].values, dtype=torch.float32)
    test_y = torch.tensor(test_df["median_Tm"].values, dtype=torch.float32)

    print("Loading ESM2...")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    esm_model = esm_model.to(DEVICE)
    esm_model.eval()
    # Match the paper repo's `load_esm2_model` (utils/esm2_utils.py): with
    # token_dropout enabled, ESM2 rescales activations by (1 - 0.12) /
    # (1 - mask_ratio) at every forward, even when no mask tokens are
    # present. Disabling it gives clean, untouched layer-33 features.
    esm_model.token_dropout = False

    for p in esm_model.parameters():
        p.requires_grad = False

    train_feats = precompute_pooled_feats(
        train_df["sequence"].tolist(),
        esm_model,
        alphabet,
        batch_converter,
        TRAIN_CACHE,
        desc="train",
    )
    test_feats = precompute_pooled_feats(
        test_df["sequence"].tolist(),
        esm_model,
        alphabet,
        batch_converter,
        TEST_CACHE,
        desc="test",
    )

    head = ThermoHead(embed_dim=EMBED_DIM).to(DEVICE)
    head.init_from_esm_lm_head(esm_model.lm_head)
    print("Initialized head dense + layer_norm from esm_model.lm_head")

    # ESM2 is no longer needed: features are cached and the head is warm-started.
    # Free ~2.5 GB of GPU memory so the head training has room to breathe.
    del esm_model, alphabet, batch_converter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move all pooled features to GPU once. With ~28k * 1280 * 4 bytes
    # that's only ~143 MB — trivial compared to ESM2's footprint, and it
    # eliminates per-step host->device copies.
    train_feats = train_feats.to(DEVICE)
    test_feats = test_feats.to(DEVICE)
    train_y_dev = train_y.to(DEVICE)
    test_y_dev = test_y.to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(train_feats, train_y_dev),
        batch_size=HEAD_BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_feats, test_y_dev),
        batch_size=HEAD_BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        build_param_groups(head, WEIGHT_DECAY),
        lr=PEAK_LR,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=PEAK_LR,
        total_steps=total_steps,
        pct_start=WARMUP_FRACTION,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e3,
    )
    loss_fn = nn.MSELoss()

    best_spearman = -float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(EPOCHS):
        head.train()
        total_loss = 0.0
        n_batches = 0

        for pooled, y in train_loader:
            y_norm = (y - tm_mean) / tm_std

            preds_norm = head(pooled)
            loss = loss_fn(preds_norm, y_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        mse, mae, spearman, pearson = evaluate(test_loader, head, tm_mean, tm_std)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1}: "
            f"lr={current_lr:.2e}, "
            f"train_loss={avg_loss:.4f}, "
            f"test_mse={mse:.4f}, "
            f"test_mae={mae:.4f}, "
            f"spearman={spearman:.4f}, "
            f"pearson={pearson:.4f}"
        )

        if spearman is not None and spearman > best_spearman:
            best_spearman = spearman
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    print(f"Best epoch: {best_epoch} with Spearman {best_spearman:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)

    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "tm_mean": tm_mean.item(),
            "tm_std": tm_std.item(),
            "best_epoch": best_epoch,
            "best_spearman": best_spearman,
        },
        "thermostability_head.pt",
    )

    print("Saved model to thermostability_head.pt")


if __name__ == "__main__":
    main()
