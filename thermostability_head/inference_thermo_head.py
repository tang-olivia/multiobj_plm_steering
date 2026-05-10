#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import esm
import pandas as pd
import torch

from train_thermo_head import ThermoHead, precompute_pooled_feats, EMBED_DIM, DEVICE

NLP_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = NLP_DIR / "thermostability_head.pt"


def _read_sequences(path: Path, sequence_column: str | None) -> list[str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        if not sequence_column:
            raise ValueError("For CSV input, pass --sequence-column (e.g. sequence).")
        df = pd.read_csv(path)
        if sequence_column not in df.columns:
            raise ValueError(
                f"Column {sequence_column!r} not in {path}; have {list(df.columns)}"
            )
        seqs = df[sequence_column].astype(str).str.strip().tolist()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        seqs = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith(">"):
                raise ValueError(
                    f"{path}: FASTA not supported here; use one sequence per line or CSV."
                )
            seqs.append(s)

    out = [s for s in seqs if s]
    if not out:
        raise ValueError(f"No sequences read from {path}")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict Tm (°C) with ESM2 pooled features + trained ThermoHead.",
    )
    p.add_argument(
        "-i",
        "--sequences",
        type=Path,
        required=True,
        help="Input: .txt (one sequence per line) or .csv (use --sequence-column).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV with columns: sequence, Tm (°C).",
    )
    p.add_argument(
        "--sequence-column",
        type=str,
        default=None,
        help="For CSV input: column name containing amino-acid sequences.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT,
        help=f"ThermoHead checkpoint (default: {DEFAULT_CKPT})",
    )
    p.add_argument(
        "--desc",
        type=str,
        default="infer",
        help="Label for precompute_pooled_feats logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    sequences_path: Path = args.sequences
    output_path: Path = args.output
    ckpt_path: Path = args.checkpoint.expanduser().resolve()

    sequences = _read_sequences(sequences_path, args.sequence_column)

    if not ckpt_path.is_file():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(DEVICE).eval()
    esm_model.token_dropout = False
    for p in esm_model.parameters():
        p.requires_grad = False

    pooled = precompute_pooled_feats(
        sequences,
        esm_model,
        alphabet,
        batch_converter,
        cache_path=None,
        desc=args.desc,
        use_cache=False,
    )

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    tm_mean = torch.tensor(ckpt["tm_mean"], dtype=torch.float32)
    tm_std = torch.tensor(ckpt["tm_std"], dtype=torch.float32)

    head = ThermoHead(embed_dim=EMBED_DIM).to(DEVICE)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    pooled = pooled.to(DEVICE)
    with torch.no_grad():
        preds_norm = head(pooled)
        preds_tm = preds_norm * tm_std.to(DEVICE) + tm_mean.to(DEVICE)

    out_df = pd.DataFrame(
        {
            "sequence": sequences,
            "Tm": preds_tm.cpu().numpy().astype(float),
        }
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {len(out_df)} predictions to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
