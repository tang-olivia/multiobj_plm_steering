from __future__ import annotations

import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
STEERING_DIR = PROJECT_ROOT / "steering_vectors"
OUT_ROOT = PROJECT_ROOT / "steering_joint"
NO_ORTHO_DIR = OUT_ROOT / "no_ortho"
PERP_DIR = OUT_ROOT / "struct_perp_wrt_thermo"

V_THERMO_NAME = "v_thermo.pt"
V_STRUCT_NAME = "v_struct_toward_plausible.pt"
V_STRUCT_PERP_NAME = "v_struct_perp.pt"


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(obj).__name__}")
    return obj.float()


def orthogonalize_struct_wrt_thermo(
    v_thermo: torch.Tensor,
    v_struct: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Return v_struct with per-layer projection onto v_thermo removed."""
    if v_thermo.shape != v_struct.shape:
        raise ValueError(
            f"Shape mismatch: v_thermo {v_thermo.shape} vs v_struct {v_struct.shape}"
        )

    v_t = v_thermo.clone()
    v_s = v_struct.clone()

    norm_t = v_t.norm(dim=-1, keepdim=True).clamp(min=eps)
    u_t = v_t / norm_t

    proj_coef = (v_s * u_t).sum(dim=-1, keepdim=True)
    v_perp = v_s - proj_coef * u_t

    # Diagnostics (per layer)
    nt = v_t.norm(dim=-1)
    ns = v_s.norm(dim=-1)
    np_ = v_perp.norm(dim=-1)
    cos_ts = (v_t * v_s).sum(dim=-1) / (nt * ns).clamp(min=eps)

    meta = {
        "n_layers": int(v_t.shape[0]),
        "hidden_size": int(v_t.shape[1]),
        "per_layer": [],
    }
    for ell in range(v_t.shape[0]):
        meta["per_layer"].append(
            {
                "layer": ell,
                "cos_thermo_struct": float(cos_ts[ell].item()),
                "norm_v_thermo": float(nt[ell].item()),
                "norm_v_struct": float(ns[ell].item()),
                "norm_v_struct_perp": float(np_[ell].item()),
                "frac_perp_of_struct_norm": float(
                    (np_[ell] / ns[ell].clamp(min=eps)).item()
                ),
            }
        )

    return v_perp, meta


def save_bundle(
    out_dir: Path,
    v_thermo: torch.Tensor,
    v_second: torch.Tensor,
    second_filename: str,
    extra_meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(v_thermo.contiguous(), out_dir / V_THERMO_NAME)
    torch.save(v_second.contiguous(), out_dir / second_filename)

    manifest = {
        "source_dir": str(STEERING_DIR.resolve()),
        "v_thermo_file": V_THERMO_NAME,
        "second_vector_file": second_filename,
        "steering_recipe": extra_meta.get("steering_recipe", ""),
        **{k: v for k, v in extra_meta.items() if k != "steering_recipe"},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    thermo_path = STEERING_DIR / V_THERMO_NAME
    struct_path = STEERING_DIR / V_STRUCT_NAME

    if not thermo_path.is_file() or not struct_path.is_file():
        raise FileNotFoundError(
            f"Missing {thermo_path} or {struct_path}. "
            "Expected pre-computed vectors under steering_vectors/."
        )

    v_thermo = _load_tensor(thermo_path)
    v_struct = _load_tensor(struct_path)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Bundle A: no orthogonalization (explicit copies for a stable path) ---
    save_bundle(
        NO_ORTHO_DIR,
        v_thermo,
        v_struct,
        V_STRUCT_NAME,
        extra_meta={
            "steering_recipe": (
                "Additive steering with independent thermostability and structural "
                "plausibility directions (no orthogonalization)."
            ),
            "second_vector_role": "v_struct_toward_plausible (same as source)",
            "inference": "h_tilde = h + alpha * v_thermo + beta * v_struct; "
            "then rescale to ||h|| per token (paper recipe).",
        },
    )

    # --- Bundle B: structural direction orthogonal to thermostability ---
    v_struct_perp, ortho_diag = orthogonalize_struct_wrt_thermo(v_thermo, v_struct)
    save_bundle(
        PERP_DIR,
        v_thermo,
        v_struct_perp,
        V_STRUCT_PERP_NAME,
        extra_meta={
            "steering_recipe": (
                "Structural steering vector is Gram–Schmidt residual of "
                "v_struct_toward_plausible w.r.t. unit v_thermo per layer."
            ),
            "second_vector_role": "v_struct_perp",
            "inference": "h_tilde = h + alpha * v_thermo + beta * v_struct_perp; "
            "then rescale to ||h|| per token (paper recipe).",
            "orthogonalization": ortho_diag,
        },
    )

    # Root summary for quick comparison
    summary = {
        "no_ortho_dir": str(NO_ORTHO_DIR.resolve()),
        "struct_perp_dir": str(PERP_DIR.resolve()),
        "global_cosine_stats": {
            "mean_cos_thermo_struct": float(
                sum(p["cos_thermo_struct"] for p in ortho_diag["per_layer"])
                / len(ortho_diag["per_layer"])
            ),
        },
    }
    with open(OUT_ROOT / "orthogonalization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote no-ortho bundle → {NO_ORTHO_DIR}")
    print(f"Wrote struct⟂thermo bundle → {PERP_DIR}")
    print(f"Summary → {OUT_ROOT / 'orthogonalization_summary.json'}")


if __name__ == "__main__":
    main()
