"""
Data preparation for NF-UNSW-NB15-v2 (TemporalData).
Splits the temporal flow stream into train/val/test by time, normalizes msg
features, and emits compact numpy arrays for downstream models.
"""
import os
import json
import numpy as np
import torch

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(WORK, "data", "NF-UNSW-NB15-v2_3d.pt")
OUT = os.path.join(WORK, "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    d = torch.load(DATA, weights_only=False, map_location="cpu")
    src = d["src"].numpy().astype(np.int64)
    dst = d["dst"].numpy().astype(np.int64)
    t = d["t"].numpy().astype(np.int64)
    msg = d["msg"].numpy().astype(np.float32)
    label = d["label"].numpy().astype(np.int64)
    attack = d["attack"].numpy().astype(np.int64)
    dt = d["dt"].numpy().astype(np.float32)

    # Sort by time to be safe
    order = np.argsort(t, kind="stable")
    src, dst, t, msg, label, attack, dt = (
        src[order], dst[order], t[order], msg[order], label[order], attack[order], dt[order]
    )

    n = len(label)
    # Time-based 60/20/20 split
    n_tr = int(0.6 * n)
    n_va = int(0.2 * n)
    idx_tr = np.arange(0, n_tr)
    idx_va = np.arange(n_tr, n_tr + n_va)
    idx_te = np.arange(n_tr + n_va, n)

    # Normalize msg using train statistics
    mu = msg[idx_tr].mean(0)
    sd = msg[idx_tr].std(0) + 1e-6
    msg_z = (msg - mu) / sd

    np.savez_compressed(
        os.path.join(OUT, "data_clean.npz"),
        src=src, dst=dst, t=t, msg=msg_z.astype(np.float32),
        label=label, attack=attack, dt=dt,
        idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te,
        mu=mu.astype(np.float32), sd=sd.astype(np.float32),
    )

    # Stats
    from collections import Counter
    overall = Counter(attack.tolist())
    train_attack = Counter(attack[idx_tr].tolist())
    test_attack = Counter(attack[idx_te].tolist())

    stats = {
        "n_flows": int(n),
        "n_features": int(msg.shape[1]),
        "n_unique_src": int(np.unique(src).shape[0]),
        "n_unique_dst": int(np.unique(dst).shape[0]),
        "t_min": int(t.min()),
        "t_max": int(t.max()),
        "n_train": int(len(idx_tr)),
        "n_val": int(len(idx_va)),
        "n_test": int(len(idx_te)),
        "binary_label_overall": dict((str(k), int(v)) for k, v in Counter(label.tolist()).items()),
        "attack_dist_overall": dict((str(k), int(v)) for k, v in overall.items()),
        "attack_dist_train": dict((str(k), int(v)) for k, v in train_attack.items()),
        "attack_dist_test": dict((str(k), int(v)) for k, v in test_attack.items()),
    }
    with open(os.path.join(OUT, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
