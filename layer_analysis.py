import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path("/home/ubuntu/68610_project")
STEERING_DIR = PROJECT_DIR / "steering_vectors"
FIGURE_PATH = PROJECT_DIR / "figures" / "layer_classifier_accuracy.png"
RESULTS_PATH = PROJECT_DIR / "layer_classifier_accuracy.json"

PROPERTIES = {
    "thermo": "Thermostability",
    "struct": "Structural plausibility",
}

N_FOLDS = 5
RANDOM_STATE = 42

LOGREG_C = 0.1
MAX_ITER = 5000


def load_activations(prop):
    pos_path = STEERING_DIR / f"{prop}_pos_per_seq_activations.pt"
    neg_path = STEERING_DIR / f"{prop}_neg_per_seq_activations.pt"
    pos = torch.load(pos_path, map_location="cpu", weights_only=False)
    neg = torch.load(neg_path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(pos) or not torch.is_tensor(neg):
        raise TypeError(
            f"Expected raw tensors at {pos_path} and {neg_path}, got "
            f"{type(pos).__name__} and {type(neg).__name__}"
        )
    return pos.numpy().astype(np.float32), neg.numpy().astype(np.float32)


def layer_sweep(pos, neg):
    n_pos, n_layers, _ = pos.shape
    n_neg = neg.shape[0]

    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    accuracies = np.zeros(n_layers, dtype=np.float64)
    for layer in range(n_layers):
        X = np.concatenate([pos[:, layer, :], neg[:, layer, :]], axis=0)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=LOGREG_C,
                penalty="l2",
                solver="lbfgs",
                max_iter=MAX_ITER,
                random_state=RANDOM_STATE,
            ),
        )
        scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        accuracies[layer] = scores.mean()

    return accuracies


def plot_results(results):
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for prop, pretty in PROPERTIES.items():
        accs = results[prop]["accuracies"]
        best = results[prop]["best_layer"]
        line = ax.plot(range(len(accs)), accs, marker="o", label=pretty)[0]
        ax.axvline(best, linestyle="--", color=line.get_color(), alpha=0.5)

    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xlabel("Hidden state index (0 = embedding, 33 = LM-head input)")
    ax.set_ylabel(f"{N_FOLDS}-fold CV accuracy")
    ax.set_title("Per-layer linear separability: pos vs. neg")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(range(0, 34, 2))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {FIGURE_PATH}")


def main():
    results = {}
    for prop, pretty in PROPERTIES.items():
        print(f"=== {pretty} ({prop}) ===")
        pos, neg = load_activations(prop)
        print(f"  pos: {pos.shape}    neg: {neg.shape}")

        accs = layer_sweep(pos, neg)
        best = int(np.argmax(accs))
        print(f"  best layer:    {best}  (acc {accs[best]:.4f})")
        print(f"  layer 0 (emb): {accs[0]:.4f}")
        print(f"  layer 33 (lm): {accs[-1]:.4f}")
        print(f"  layers >= 0.80 acc: {[i for i, a in enumerate(accs) if a >= 0.80]}")
        print(f"  layers >= 0.90 acc: {[i for i, a in enumerate(accs) if a >= 0.90]}")
        print()

        results[prop] = {
            "pretty_name": pretty,
            "accuracies": accs.tolist(),
            "best_layer": best,
            "best_accuracy": float(accs[best]),
            "n_pos": int(pos.shape[0]),
            "n_neg": int(neg.shape[0]),
        }

    results["_meta"] = {
        "n_folds": N_FOLDS,
        "logreg_C": LOGREG_C,
        "max_iter": MAX_ITER,
        "random_state": RANDOM_STATE,
        "scaling": "per-feature StandardScaler within each fold",
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")

    plot_results(results)


if __name__ == "__main__":
    main()
