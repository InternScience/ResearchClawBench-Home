"""
Main experiment: train DIDS-MFL and baselines for binary + multiclass with multiple seeds.
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from train_utils import train_one

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "outputs")

CONFIGS = {
    "didsmfl":      dict(n_epochs=8, batch_size=2048, alpha=0.05, beta=0.05, lr=1e-3),
    "ablate_sd":    dict(n_epochs=8, batch_size=2048, alpha=0.0,  beta=0.05, lr=1e-3),
    "ablate_rd":    dict(n_epochs=8, batch_size=2048, alpha=0.05, beta=0.0,  lr=1e-3),
    "ablate_ms":    dict(n_epochs=8, batch_size=2048, alpha=0.05, beta=0.05, lr=1e-3),
    "egraphsage":   dict(n_epochs=8, batch_size=2048, alpha=0.0,  beta=0.0,  lr=1e-3),
    "mlp":          dict(n_epochs=8, batch_size=4096, alpha=0.0,  beta=0.0,  lr=1e-3),
}

SEEDS = [0, 1, 2]


def main():
    summary = {}
    per_attack = {}
    histories = {}
    for kind, cfg in CONFIGS.items():
        run_metrics = []
        run_attack = []
        run_hist = []
        for seed in SEEDS:
            t0 = time.time()
            model, hist, test = train_one(kind, seed=seed, verbose=True, **cfg)
            print(f"=== {kind} seed={seed} done in {time.time()-t0:.1f}s ===")
            scalars = {k: float(v) for k, v in test.items()
                       if not isinstance(v, (dict, np.ndarray))}
            run_metrics.append(scalars)
            run_attack.append(test["per_attack_f1"])
            run_hist.append(hist)
            # save embeddings and predictions for primary model only on seed 0
            if kind == "didsmfl" and seed == 0:
                np.savez_compressed(os.path.join(OUT, "didsmfl_seed0_test.npz"),
                                    preds_b=test["preds_b"], preds_m=test["preds_m"],
                                    prob_pos=test["prob_pos"],
                                    y_b=test["y_b"], y_a=test["y_a"])
        summary[kind] = {
            "runs": run_metrics,
            "mean": {k: float(np.mean([r[k] for r in run_metrics])) for k in run_metrics[0]},
            "std": {k: float(np.std([r[k] for r in run_metrics])) for k in run_metrics[0]},
        }
        # mean per-attack F1
        cls_keys = list(run_attack[0].keys())
        per_attack[kind] = {
            "mean": {c: float(np.mean([r[c] for r in run_attack])) for c in cls_keys},
            "std": {c: float(np.std([r[c] for r in run_attack])) for c in cls_keys},
        }
        histories[kind] = run_hist

    with open(os.path.join(OUT, "main_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "per_attack_results.json"), "w") as f:
        json.dump(per_attack, f, indent=2)
    with open(os.path.join(OUT, "histories.json"), "w") as f:
        json.dump(histories, f, indent=2)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k:14s} bin_f1={v['mean']['bin_f1']:.4f}±{v['std']['bin_f1']:.4f} "
              f"AUC={v['mean']['bin_auc']:.4f}±{v['std']['bin_auc']:.4f} "
              f"multi_macro_f1={v['mean']['multi_macro_f1']:.4f}±{v['std']['multi_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
