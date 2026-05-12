"""Construct ESM2 steering vectors for thermostability and structural plausibility.

Reads the four steering set CSVs in steering_sets/, length-stratified-samples 100
sequences from each (20 per bin x 5 bins matching the PDF spec: 100-130, 130-160,
160-190, 190-220, 220-256), runs them through ESM2-650M, mean-pools over residue
tokens (excluding BOS, EOS, and padding), then takes mean(pos) - mean(neg) at
every hidden state to produce per-layer steering vectors.

Outputs (in --out_dir):
  v_thermo.pt                        # mean(pos_thermo) - mean(neg_thermo), shape [L+1, H]
  v_struct_toward_plausible.pt       # mean(pos_struct) - mean(neg_struct), shape [L+1, H]
  v_struct_toward_implausible.pt     # negation (matches PDF h' = h + a*v_thermo - b*v_struct)
  <set>_per_seq_activations.pt       # [n_seqs, L+1, H], one per set
  <set>_mean_activations.pt          # [L+1, H], one per set
  <set>_samples.csv                  # the chosen sequences
  meta.json                          # config + per-layer diagnostics

Uses bfloat16 on the model forward (no overflow, ~fp16 speed on A100). fp16
causes NaN/Inf in ESM2 attention on longer sequences.

Note: the rest of the pipeline (orthogonalization.py, generate_with_steering.py,
layer_analysis.py) loads ESM2 via fair-esm. This script uses HuggingFace
transformers.EsmModel for convenience; the hidden states are equivalent up to
numerical precision.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SETS_DIR = PROJECT_ROOT / "steering_sets"
DEFAULT_OUT_DIR = PROJECT_ROOT / "steering_vectors"

LENGTH_BINS = [(100, 130), (130, 160), (160, 190), (190, 220), (220, 256)]
SETS = {
    "thermo_pos": "positive_thermostability.csv",
    "thermo_neg": "negative_thermostability.csv",
    "struct_pos": "positive_structural_plausibility.csv",
    "struct_neg": "negative_structural_plausibility.csv",
}


def length_stratified_sample(df: pd.DataFrame, bins, per_bin: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    L = df["sequence"].str.len().to_numpy()
    chosen = []
    for i, (lo, hi) in enumerate(bins):
        m = (L >= lo) & (L <= hi) if i == len(bins) - 1 else (L >= lo) & (L < hi)
        pool = np.flatnonzero(m)
        if len(pool) < per_bin:
            raise ValueError(f"bin [{lo},{hi}] has only {len(pool)} sequences, need {per_bin}")
        chosen.extend(rng.choice(pool, size=per_bin, replace=False).tolist())
    return df.iloc[chosen].reset_index(drop=True)


@torch.no_grad()
def mean_pool_per_seq(sequences, model, tokenizer, device, batch_size: int):
    out = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        toks = {k: v.to(device) for k, v in toks.items()}
        res = model(**toks, output_hidden_states=True, return_dict=True)
        hs = torch.stack(res.hidden_states, dim=1)  # [B, L+1, T, H]

        attn = toks["attention_mask"]  # [B, T] — 1 for real (incl BOS/EOS), 0 for pad
        residue_mask = attn.clone()
        residue_mask[:, 0] = 0  # drop BOS
        eos_idx = attn.sum(dim=1) - 1  # last real-token position == EOS
        residue_mask[torch.arange(attn.size(0), device=device), eos_idx] = 0  # drop EOS

        m = residue_mask.unsqueeze(1).unsqueeze(-1).to(hs.dtype)  # [B, 1, T, 1]
        summed = (hs * m).sum(dim=2)  # [B, L+1, H]
        counts = residue_mask.sum(dim=1).clamp(min=1).view(-1, 1, 1).to(hs.dtype)
        pooled = summed / counts  # [B, L+1, H]
        out.append(pooled.float().cpu())
        print(f"    {min(start + batch_size, len(sequences))}/{len(sequences)}")
    return torch.cat(out, dim=0)  # [n_seqs, L+1, H]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--sets_dir", default=str(DEFAULT_SETS_DIR),
                   help="Directory containing the four steering-set CSVs (defaults to <repo>/steering_sets).")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR),
                   help="Where to write the vectors and per-set artifacts (defaults to <repo>/steering_vectors).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per_bin", type=int, default=20, help="Sequences per length bin (5 bins -> 100 per set).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                   help="Model forward precision. bf16 is safe on A100; fp16 is NOT supported (overflow in ESM2 attention).")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sets_dir = Path(args.sets_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"loading {args.model} ({args.dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).eval().to(device)
    n_layers = model.config.num_hidden_layers
    H = model.config.hidden_size
    print(f"loaded: {n_layers} layers, hidden_size={H}, dtype={next(model.parameters()).dtype}")

    set_means = {}
    for name, csv in SETS.items():
        path = sets_dir / csv
        print(f"\n[{name}] {path}")
        df = pd.read_csv(path)
        sampled = length_stratified_sample(df, LENGTH_BINS, args.per_bin, args.seed)
        per_seq = mean_pool_per_seq(sampled["sequence"].tolist(), model, tokenizer, device, args.batch_size)
        set_means[name] = per_seq.mean(dim=0)
        torch.save(per_seq, out_dir / f"{name}_per_seq_activations.pt")
        torch.save(set_means[name], out_dir / f"{name}_mean_activations.pt")
        sampled.assign(seq_len=sampled["sequence"].str.len()).to_csv(
            out_dir / f"{name}_samples.csv", index=False
        )
        print(f"  per_seq shape: {tuple(per_seq.shape)}")

    v_thermo = set_means["thermo_pos"] - set_means["thermo_neg"]
    v_struct_toward_plausible = set_means["struct_pos"] - set_means["struct_neg"]
    v_struct_toward_implausible = -v_struct_toward_plausible

    for nm, v in [("v_thermo", v_thermo), ("v_struct_toward_plausible", v_struct_toward_plausible)]:
        bad = (~torch.isfinite(v)).sum().item()
        if bad:
            print(f"WARNING: {nm} has {bad} non-finite entries — check --dtype")

    torch.save(v_thermo, out_dir / "v_thermo.pt")
    torch.save(v_struct_toward_plausible, out_dir / "v_struct_toward_plausible.pt")
    torch.save(v_struct_toward_implausible, out_dir / "v_struct_toward_implausible.pt")

    diag = []
    print(f"\n  {'layer':>5}  {'||v_thermo||':>14}  {'||v_struct||':>14}  {'cos':>7}")
    for l in range(v_thermo.shape[0]):
        nt = v_thermo[l].norm().item()
        ns = v_struct_toward_plausible[l].norm().item()
        cos = torch.nn.functional.cosine_similarity(
            v_thermo[l].unsqueeze(0), v_struct_toward_plausible[l].unsqueeze(0)
        ).item()
        diag.append({"layer": l, "norm_v_thermo": nt, "norm_v_struct": ns, "cos_thermo_struct": cos})
        print(f"  {l:>5}  {nt:>14.4f}  {ns:>14.4f}  {cos:>+7.3f}")

    meta = {
        "model": args.model,
        "n_hidden_states": int(v_thermo.shape[0]),
        "hidden_size": int(v_thermo.shape[1]),
        "length_bins": LENGTH_BINS,
        "per_bin": args.per_bin,
        "n_per_set": args.per_bin * len(LENGTH_BINS),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "pooling": "mean over residue tokens (excluding BOS, EOS, and padding)",
        "precision": str(dtype),
        "vector_definitions": {
            "v_thermo": "mean(pos_thermo) - mean(neg_thermo)",
            "v_struct_toward_plausible": "mean(pos_struct) - mean(neg_struct)",
            "v_struct_toward_implausible": "-v_struct_toward_plausible (matches PDF h' = h + a*v_thermo - b*v_struct)",
        },
        "layer_diagnostics": diag,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
