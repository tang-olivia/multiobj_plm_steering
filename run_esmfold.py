from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.loss import compute_tm


_CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWYX")


def clean_sequence(seq: str) -> str:
    seq = seq.upper().replace("*", "").replace("-", "").replace(" ", "")
    return "".join(c if c in _CANONICAL_AAS else "X" for c in seq)


def load_inputs(pkl_path: str) -> List[Tuple[str, str]]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    out: List[Tuple[str, str]] = []
    for cid, rec in data.items():
        seq = clean_sequence(rec["sequence"])
        if 1 <= len(seq):
            out.append((cid, seq))
    return out


def load_already_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["cluster_id"])
            except Exception:
                continue
    return done


def make_batches(
    items: List[Tuple[str, str]],
    max_tokens: int,
    batch_cap: int,
) -> Iterable[List[Tuple[str, str]]]:
    """Greedy length-bucketed packing.

    Sort by descending length and accumulate into a batch until either:
      (a) the batch hits `batch_cap` samples, or
      (b) `(len(batch) + 1) * max_len_in_batch` would exceed `max_tokens`.

    Sorting longest-first means OOMs (if any) happen on the very first batch,
    which is great for fail-fast behaviour while tuning --max-tokens.
    """
    items_sorted = sorted(items, key=lambda x: -len(x[1]))
    batch: List[Tuple[str, str]] = []
    cur_max_len = 0
    for cid, seq in items_sorted:
        new_max = max(cur_max_len, len(seq))
        full = len(batch) + 1 > batch_cap
        too_many_tokens = (len(batch) + 1) * new_max > max_tokens
        if batch and (full or too_many_tokens):
            yield batch
            batch, cur_max_len, new_max = [], 0, len(seq)
        batch.append((cid, seq))
        cur_max_len = new_max
    if batch:
        yield batch


@torch.inference_mode()
def fold_batch(model, tokenizer, batch, device) -> List[dict]:
    cids, seqs = zip(*batch)
    toks = tokenizer(
        list(seqs),
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    toks = {k: v.to(device, non_blocking=True) for k, v in toks.items()}

    out = model(**toks)
    plddt = out["plddt"]                       
    mask = toks["attention_mask"].bool()      
    ca_plddt = plddt[..., 1]                   
    ptm_logits = out.get("ptm_logits", None)  

    lengths = mask.sum(dim=1).tolist()

    results: List[dict] = []
    for i, cid in enumerate(cids):
        L_i = int(lengths[i])
        per_res = ca_plddt[i, :L_i].float().cpu().numpy()

        if ptm_logits is None:
            ptm_i = None
        else:
            no_bins = ptm_logits.shape[-1]
            logits_i = ptm_logits[i : i + 1, :L_i, :L_i] 
            ptm_i = float(compute_tm(logits_i, max_bin=31, no_bins=no_bins))

        results.append(
            {
                "cluster_id": cid,
                "length": L_i,
                "mean_plddt": float(per_res.mean()),
                "ptm": ptm_i,
                "per_res_plddt": np.round(per_res, 3).tolist(),
            }
        )
    return results


def fold_batch_with_oom_recovery(model, tokenizer, batch, device) -> Tuple[List[dict], int]:
    """Run fold_batch; on CUDA OOM, recursively split in half. Returns (results, n_failed)."""
    try:
        return fold_batch(model, tokenizer, batch, device), 0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            cid, seq = batch[0]
            print(f"\n[!] OOM on single seq {cid} (len={len(seq)}); skipping")
            return [], 1
        mid = len(batch) // 2
        left, l_fail = fold_batch_with_oom_recovery(model, tokenizer, batch[:mid], device)
        right, r_fail = fold_batch_with_oom_recovery(model, tokenizer, batch[mid:], device)
        return left + right, l_fail + r_fail


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input",
        default="/home/ubuntu/68610_project/lysozyme_uniref50/lysozyme_uniref50_400_res.pkl",
    )
    p.add_argument(
        "--output",
        default="/home/ubuntu/68610_project/lysozyme_uniref50/lysozyme_plddt.jsonl",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Per-batch budget: batch_size * max_seq_len_in_batch must stay <= this. "
             "1024 is conservative; 2048 is great on a 40 GB+ card.",
    )
    p.add_argument("--batch-cap", type=int, default=8, help="Hard cap on samples per batch.")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Folding-trunk attention chunk size. Lower = less VRAM, slower. Try 128/64/32/16.",
    )
    p.add_argument("--cache-dir", default=None, help="HF cache dir for the model weights.")
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Keep the ESM trunk in fp32 (debug only -- much slower and more VRAM).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of sequences to fold (for smoke tests).",
    )
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading {args.input}")
    items = load_inputs(args.input)
    print(f"      {len(items):,} sequences after sanitization")

    print(f"[2/4] checkpoint check: {out_path}")
    done = load_already_done(out_path)
    if done:
        print(f"      skipping {len(done):,} already-folded sequences")
    items = [(cid, seq) for cid, seq in items if cid not in done]
    if args.limit is not None:
        items = items[: args.limit]
    print(f"      to fold: {len(items):,}")
    if not items:
        print("      nothing to do, exiting.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible; ESMFold on CPU is not viable for 21k sequences.")
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[3/4] loading ESMFold to {device}")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=args.cache_dir)
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()
    if not args.no_fp16:
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(args.chunk_size)

    print(f"[4/4] folding {len(items):,} sequences")
    batches = list(make_batches(items, max_tokens=args.max_tokens, batch_cap=args.batch_cap))
    print(f"      grouped into {len(batches):,} batches")

    n_done = n_fail = 0
    t0 = time.time()
    with out_path.open("a") as f, tqdm(total=len(items), unit="seq") as pbar:
        for batch in batches:
            bs = len(batch)
            max_l = max(len(s) for _, s in batch)
            results, n_failed = fold_batch_with_oom_recovery(model, tokenizer, batch, device)
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n_done += len(results)
            n_fail += n_failed
            pbar.update(bs)
            elapsed = time.time() - t0
            pbar.set_postfix(
                bs=bs,
                max_l=max_l,
                fail=n_fail,
                rate=f"{n_done / max(elapsed, 1e-6):.2f} seq/s",
            )

    elapsed_min = (time.time() - t0) / 60
    print(f"[done] folded {n_done:,} sequences ({n_fail:,} failed) in {elapsed_min:.1f} min")
    print(f"       output: {out_path}")


if __name__ == "__main__":
    main()
