"""
Unknown attack scenario (Leave-One-Attack-Out, LOAO).
For each attack class c (excluding benign=2), train DIDS-MFL and the
E-GraphSAGE-style baseline on the train split with the supervised loss
masking out class c (i.e. the model never sees attack c in training).
At test time, evaluate binary detection F1 specifically on flows of attack c,
i.e. how well the model flags this previously-unseen attack as attack.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from train_utils import train_one, load_data, remap_nodes, evaluate, chunked_batches

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "outputs")

# attack id 2 is benign; 0,1,3,4,5,6,7,8,9 are attacks
ATTACK_IDS = [0, 1, 3, 4, 5, 6, 7, 8, 9]
ATTACK_NAMES = {
    0: "Analysis", 1: "Backdoor", 3: "DoS", 4: "Exploits", 5: "Fuzzers",
    6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms",
    2: "Benign",
}


def run_loao(model_kind, hide_attack, n_epochs=6, batch_size=2048, seed=0):
    """Train with mask: drop training rows where attack == hide_attack.
    Evaluate test rows whose attack == hide_attack -- their binary label is
    1 (attack), and we measure recall (i.e., detection F1).
    """
    data = load_data()
    src, dst, t = data["src"], data["dst"], data["t"]
    msg, label, attack = data["msg"], data["label"], data["attack"]
    idx_tr, idx_va, idx_te = data["idx_tr"], data["idx_va"], data["idx_te"]

    # build mask: True where we keep this training row's supervised loss
    keep_mask = np.ones(len(label), dtype=bool)
    keep_mask[attack == hide_attack] = False

    # We pass mask_train sized to all rows; train_one will index by batch
    model, hist, te = train_one(
        model_kind, seed=seed, n_epochs=n_epochs, batch_size=batch_size,
        alpha=0.05, beta=0.05, lr=1e-3, verbose=False,
        mask_train=keep_mask,
    )

    # Recover predictions on test rows of the hidden attack
    yb = te["y_b"]
    ya = te["y_a"]
    pb = te["preds_b"]
    pp = te["prob_pos"]

    sel = (ya == hide_attack)
    n_sel = int(sel.sum())
    if n_sel == 0:
        return None
    # Detection performance for hidden attack: recall (TPR) and F1 vs benign in mixed test
    detect_recall = float((pb[sel] == 1).mean())  # how many of these hidden attacks were flagged

    # Combined task: binary task across all test data, but attack labels for the
    # hidden class were never seen. Binary F1 on the *combined* test set is meaningful too.
    bin_f1_combined = float(f1_score(yb, pb))
    bin_auc_combined = float(roc_auc_score(yb, pp))

    # F1 on the held-out attack treated as 1 vs benign-only baseline
    # (simulate "just this attack vs benign" subset)
    sel_or_benign = (ya == hide_attack) | (ya == 2)
    f1_subset = float(f1_score(yb[sel_or_benign], pb[sel_or_benign], zero_division=0))

    return {
        "n_test_attack": n_sel,
        "detect_recall": detect_recall,
        "bin_f1_combined_test": bin_f1_combined,
        "bin_auc_combined_test": bin_auc_combined,
        "f1_subset_attack_vs_benign": f1_subset,
    }


def main():
    out = {}
    for kind in ["didsmfl", "egraphsage", "mlp"]:
        out[kind] = {}
        for aid in ATTACK_IDS:
            t0 = time.time()
            try:
                res = run_loao(kind, aid, n_epochs=6, batch_size=2048, seed=0)
            except Exception as e:
                print("ERR", kind, aid, e)
                res = {"error": str(e)}
            res = res or {}
            res["attack_name"] = ATTACK_NAMES[aid]
            print(f"[{kind}] hide={ATTACK_NAMES[aid]:14s} "
                  f"recall={res.get('detect_recall', float('nan')):.3f} "
                  f"f1_subset={res.get('f1_subset_attack_vs_benign', float('nan')):.3f} "
                  f"({time.time()-t0:.1f}s)")
            out[kind][str(aid)] = res

    with open(os.path.join(OUT, "unknown_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
