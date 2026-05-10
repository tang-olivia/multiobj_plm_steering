from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
STEERING_PLMS_ROOT = PROJECT_ROOT / "Steering-PLMs"
if str(STEERING_PLMS_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_PLMS_ROOT))

from module.steerable_esm2 import steering_forward  # noqa: E402
from utils.esm2_utils import generate_sequences, load_esm2_model  # noqa: E402

DEFAULT_NO_ORTHO = PROJECT_ROOT / "steering_joint" / "no_ortho"
DEFAULT_STRUCT_PERP = PROJECT_ROOT / "steering_joint" / "struct_perp_wrt_thermo"
DEFAULT_THERMO_ONLY = PROJECT_ROOT / "steering_vectors" / "v_thermo.pt"

REGIMES = ("none", "thermo_only", "joint_no_ortho", "joint_struct_perp")

N_TRUNK_LAYERS = 33

def _load_vec(path: Path) -> torch.Tensor:
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        t = torch.load(path, map_location="cpu")
    except Exception:
        t = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(t).__name__}")
    return t.float()


def align_steering_to_trunk_rows(v: torch.Tensor) -> torch.Tensor:
    if v.ndim != 2:
        raise ValueError(f"Steering tensor must be 2-D, got {v.shape}")
    if v.shape[0] == 34:
        return v[1:].contiguous()
    if v.shape[0] == 33:
        return v.contiguous()
    raise ValueError(
        f"Expected 33 or 34 layers in steering tensor, got first dim {v.shape[0]}"
    )


def build_steering_tensor(
    regime: str,
    device: torch.device,
    alpha: float,
    beta: float,
    no_ortho_dir: Path,
    struct_perp_dir: Path,
    thermo_only_path: Path,
) -> torch.Tensor | None:
    if regime == "none":
        return None

    if regime == "thermo_only":
        v_t = _load_vec(thermo_only_path)
        v_trunk = align_steering_to_trunk_rows(v_t)
        return (alpha * v_trunk).to(device)

    if regime == "joint_no_ortho":
        v_t = _load_vec(no_ortho_dir / "v_thermo.pt")
        v_s = _load_vec(no_ortho_dir / "v_struct_toward_plausible.pt")
        v_trunk = alpha * align_steering_to_trunk_rows(v_t) + beta * align_steering_to_trunk_rows(
            v_s
        )
        return v_trunk.to(device)

    if regime == "joint_struct_perp":
        v_t = _load_vec(struct_perp_dir / "v_thermo.pt")
        v_p = _load_vec(struct_perp_dir / "v_struct_perp.pt")
        v_trunk = alpha * align_steering_to_trunk_rows(v_t) + beta * align_steering_to_trunk_rows(
            v_p
        )
        return v_trunk.to(device)

    raise ValueError(f"Unknown regime: {regime}")


def parse_steer_layers(spec: str) -> list[int] | None:
    s = spec.strip().lower()
    if s in ("", "all", "*"):
        return None

    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            lo, hi = int(a.strip()), int(b.strip())
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))

    for i in out:
        if i < 0 or i >= N_TRUNK_LAYERS:
            raise ValueError(
                f"steer_layers: index {i} out of trunk range [0, {N_TRUNK_LAYERS - 1}]"
            )
    return sorted(out)


def mask_steering_to_layers(
    steering: torch.Tensor, active_trunk_layers: list[int] | None
) -> torch.Tensor:
    if active_trunk_layers is None:
        return steering
    if steering.shape[0] != N_TRUNK_LAYERS:
        raise ValueError(
            f"Expected steering trunk shape ({N_TRUNK_LAYERS}, *), got {steering.shape}"
        )
    mask = torch.zeros(N_TRUNK_LAYERS, device=steering.device, dtype=steering.dtype)
    for i in active_trunk_layers:
        mask[i] = 1.0
    return steering * mask.unsqueeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ESM2 iterative generation with four steering regimes."
    )
    p.add_argument(
        "--regime",
        type=str,
        required=True,
        choices=REGIMES,
        help="Steering mode: none | thermo_only | joint_no_ortho | joint_struct_perp",
    )
    p.add_argument(
        "--ref_data_path",
        type=str,
        required=True,
        help="CSV with a `sequence` column (e.g. therm_easy.csv).",
    )
    p.add_argument("--n", type=int, default=1000, help="Number of sequences to generate.")
    p.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output CSV path (columns: sequence, regime, ref_index, reference_sequence).",
    )
    p.add_argument("--model", type=str, default="650M", help="ESM2 size: 150M, 650M, 3B")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scale for thermostability direction (all steered regimes).",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Scale for structural direction (joint_no_ortho and joint_struct_perp only).",
    )
    p.add_argument("--mask_ratio", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument(
        "--no_ortho_dir",
        type=str,
        default=str(DEFAULT_NO_ORTHO),
        help="Bundle from orthogonalization.py (v_thermo + v_struct_toward_plausible).",
    )
    p.add_argument(
        "--struct_perp_dir",
        type=str,
        default=str(DEFAULT_STRUCT_PERP),
        help="Bundle with v_thermo + v_struct_perp.",
    )
    p.add_argument(
        "--thermo_only_vector",
        type=str,
        default=str(DEFAULT_THERMO_ONLY),
        help="Path to v_thermo.pt for thermo_only (34 or 33 rows).",
    )
    p.add_argument(
        "--dump_config",
        type=str,
        default=None,
        help="If set, write a JSON record of all arguments and resolved paths here.",
    )
    p.add_argument(
        "--steer_layers",
        type=str,
        default="all",
        help=(
            "Trunk indices 0-32 that receive steering (see module docstring). "
            "Default ``all``. Examples: ``24-32``, ``30``, ``1,24-26,30``. "
            "Ignored when regime=none."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device == "cuda":
        print("CUDA not available; using CPU.")

    ref_path = Path(args.ref_data_path)
    df = pd.read_csv(ref_path)
    if "sequence" not in df.columns:
        raise ValueError(f"CSV {ref_path} must contain column 'sequence'")
    ref_seqs = df["sequence"].astype(str).tolist()

    no_ortho_dir = Path(args.no_ortho_dir)
    struct_perp_dir = Path(args.struct_perp_dir)
    thermo_only_path = Path(args.thermo_only_vector)

    active_layers = parse_steer_layers(args.steer_layers)
    if args.regime == "none" and active_layers is not None:
        print("Note: regime=none ignores --steer_layers.")

    steering_vectors = build_steering_tensor(
        regime=args.regime,
        device=device,
        alpha=args.alpha,
        beta=args.beta,
        no_ortho_dir=no_ortho_dir,
        struct_perp_dir=struct_perp_dir,
        thermo_only_path=thermo_only_path,
    )
    if steering_vectors is not None:
        steering_vectors = mask_steering_to_layers(steering_vectors, active_layers)

    if args.dump_config:
        cfg = {
            "regime": args.regime,
            "ref_data_path": str(ref_path.resolve()),
            "n": args.n,
            "model": args.model,
            "device": str(device),
            "alpha": args.alpha,
            "beta": args.beta,
            **(
                {"beta_note": "ignored for regimes none and thermo_only"}
                if args.regime in ("none", "thermo_only")
                else {}
            ),
            "mask_ratio": args.mask_ratio,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "no_ortho_dir": str(no_ortho_dir.resolve()),
            "struct_perp_dir": str(struct_perp_dir.resolve()),
            "thermo_only_vector": str(thermo_only_path.resolve()),
            "steering_trunk_shape": list(steering_vectors.shape)
            if steering_vectors is not None
            else None,
            "steer_layers_raw": args.steer_layers,
            "steer_trunk_indices": active_layers
            if active_layers is not None
            else list(range(N_TRUNK_LAYERS)),
        }
        Path(args.dump_config).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_config, "w") as f:
            json.dump(cfg, f, indent=2)

    model, alphabet = load_esm2_model(args.model, device=str(device))
    model.steering_forward = types.MethodType(steering_forward, model)
    batch_converter = alphabet.get_batch_converter()

    gen_seqs = []
    ref_indices = []
    ref_used = []
    for i in tqdm(range(args.n), desc=f"generate [{args.regime}]"):
        ref_i = i % len(ref_seqs)
        seq = ref_seqs[ref_i]
        _, _, seq_token = batch_converter([("protein", seq)])
        seq_token = seq_token.to(device)
        new_seq = generate_sequences(
            seq_token,
            model,
            steering_vectors,
            args.mask_ratio,
            alphabet,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        gen_seqs.append(new_seq)
        ref_indices.append(ref_i)
        ref_used.append(seq)

    out_df = pd.DataFrame(
        {
            "sequence": gen_seqs,
            "regime": [args.regime] * len(gen_seqs),
            "ref_index": ref_indices,
            "reference_sequence": ref_used,
        }
    )
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")

    seq_txt_path = out_path.with_suffix('.txt')
    with open(seq_txt_path, "w") as seq_txt_file:
        for seq in out_df["sequence"]:
            seq_txt_file.write(f"{seq}\n")
    print(f"Wrote sequences, one per line, to {seq_txt_path}")


if __name__ == "__main__":
    main()
