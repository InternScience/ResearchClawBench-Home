"""
Few-shot scenario: simulate that for selected minority attack classes we only
have k labeled training examples. We mask out all training rows of those
classes EXCEPT k random ones per class.

We report:
  - multi-class per-class F1 (very hard with k <= 10)
  - binary detection recall on test instances of those rare classes (the
    practically important quantity: did we still flag this attack?)
  - overall binary F1 / AUC for context
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import f1_score

from train_utils import train_one, load_data

ATTACK_NAMES = {
    0: "Analysis", 1: "Backdoor", 3: "DoS", 4: "Exploits", 5: "Fuzzers",
    6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms",
    2: "Benign",
}

# Few-shot target classes (rare / hard)
FS_CLASSES = [0, 1, 5, 8, 9]  # Analysis, Backdoor, Fuzzers, Shellcode, Worms

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "outputs")


def make_fewshot_mask(attack, idx_tr, k, seed=0):
    rng = np.random.default_rng(seed)
    keep = np.ones(len(attack), dtype=bool)
    for c in FS_CLASSES:
        c_idx = np.where(attack[idx_tr] == c)[0]
        abs_idx = idx_tr[c_idx]
        if len(abs_idx) <= k:
            continue
        chosen = rng.choice(abs_idx, size=k, replace=False)
        keep[abs_idx] = False
        keep[chosen] = True
    return keep


def main():
    data = load_data()
    attack = data["attack"]
    idx_tr = data["idx_tr"]
    out = {}
    for kind in ["didsmfl", "egraphsage", "mlp"]:
        out[kind] = {}
        for k in [1, 5, 10, 50]:
            t0 = time.time()
            mask = make_fewshot_mask(attack, idx_tr, k, seed=0)
            model, hist, te = train_one(
                kind, seed=0, n_epochs=6, batch_size=2048,
                alpha=0.05, beta=0.05, lr=1e-3,
                verbose=False, mask_train=mask,
            )
            per = te["per_attack_f1"]
            mean_f1 = float(np.mean([per[str(c)] for c in FS_CLASSES]))

            yb, pb, ya = te["y_b"], te["preds_b"], te["y_a"]
            per_recall = {}
            for c in FS_CLASSES:
                sel = (ya == c)
                per_recall[str(c)] = float((pb[sel] == 1).mean()) if sel.sum() else float("nan")
            mean_recall = float(np.nanmean(list(per_recall.values())))

            res = {
                "k": k,
                "per_class_f1_fewshot": {str(c): per[str(c)] for c in FS_CLASSES},
                "fewshot_mean_f1": mean_f1,
                "per_class_binary_recall": per_recall,
                "fewshot_mean_binary_recall": mean_recall,
                "bin_f1": te["bin_f1"],
                "multi_macro_f1": te["multi_macro_f1"],
                "bin_auc": te["bin_auc"],
            }
            print(f"[{kind:11s} k={k:3d}] mean_fs_recall={mean_recall:.4f} "
                  f"mean_fs_f1={mean_f1:.4f} bin_f1={te['bin_f1']:.4f} "
                  f"({time.time()-t0:.1f}s)")
            out[kind][str(k)] = res

    with open(os.path.join(OUT, "fewshot_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
