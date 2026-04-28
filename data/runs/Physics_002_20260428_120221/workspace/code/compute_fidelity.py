"""
RCS fidelity estimation on arbitrary-geometry random circuits.

For each experiment (N, d, r, kind) we compute one fidelity estimate
with bootstrap uncertainty from the verification subset:

* XEB  (linear cross-entropy, Eq. used by Sycamore / Zuchongzhi RCS):
       F_XEB = (D / N_s) * sum_{s in measured} p_ideal(s)  -  1
       D = 2^N
       The data files match a small verifiable bitstring set
       (~20 outcomes per instance).
* MB   (Mirror Benchmarking probability of correct return):
       F_MB = (# samples == ideal_target) / N_s
* Transport_1QRB :
       F_Transport = (# samples == ideal_target) / N_s

All bitstring keys are stored as either tuple-strings "(0, 1, ...)"
or plain bitstrings.  We normalise both representations.

Uncertainty: per-instance bootstrap (1000 resamples) of the shot list,
then aggregate across instances r as mean +/- SEM.

Outputs:
    outputs/per_instance_fidelity.csv
    outputs/aggregated_fidelity.csv
"""

import os
import re
import json
import glob
import math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "data", "results")
AMP_DIR = os.path.join(ROOT, "data", "amplitudes")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------- helpers -------------------------------------- #

_TUPLE_RE = re.compile(r"\d")


def normalize_key(k):
    """Return a canonical bitstring like '0110...' from either
    a tuple-string '(0, 1, 1, 0, ...)' or a plain bitstring."""
    if not isinstance(k, str):
        if isinstance(k, (list, tuple)):
            return "".join(str(int(x)) for x in k)
        return str(k)
    if "(" in k:
        return "".join(_TUPLE_RE.findall(k))
    return k


def load_counts(path):
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        out[normalize_key(k)] = int(v)
    return out


def load_amplitudes(path):
    """Return dict bitstring -> p_ideal (real, non-negative)."""
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        bk = normalize_key(k)
        if isinstance(v, str):
            # complex amplitude string
            try:
                z = complex(v)
                p = abs(z) ** 2
            except Exception:
                p = float(v)
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            z = complex(v[0], v[1])
            p = abs(z) ** 2
        else:
            # real probability or amplitude magnitude
            try:
                vv = float(v)
                p = vv
            except Exception:
                p = float("nan")
        out[bk] = p
    return out


def load_ideal_bitstring(path):
    raw = json.load(open(path))
    return normalize_key(raw)


# -------------------------- fidelity estimators -------------------------- #


def xeb_fidelity(counts, p_ideal, N, n_boot=1000, rng=None):
    """Linear XEB on the verifiable subset.

    F_XEB = D * <p_ideal(s)> - 1, average over experimental samples.
    """
    rng = rng or np.random.default_rng(0)
    keys = sorted(set(counts.keys()) & set(p_ideal.keys()))
    if not keys:
        return None
    # build shot list of p_ideal values (one entry per measured shot)
    shot_p = []
    for k in keys:
        c = counts[k]
        p = p_ideal[k]
        shot_p.extend([p] * c)
    shot_p = np.asarray(shot_p, dtype=np.float64)
    n_s = shot_p.size
    if n_s == 0:
        return None
    D = 2 ** N
    f_point = D * shot_p.mean() - 1.0
    # bootstrap
    if n_s > 1:
        idx = rng.integers(0, n_s, size=(n_boot, n_s))
        boots = D * shot_p[idx].mean(axis=1) - 1.0
        f_std = float(boots.std(ddof=1))
    else:
        f_std = float("nan")
    return {
        "fidelity": float(f_point),
        "fidelity_std": f_std,
        "n_shots": int(n_s),
        "n_matched_keys": len(keys),
    }


def return_prob(counts, ideal_bitstring, n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    target = ideal_bitstring
    n_s = sum(counts.values())
    if n_s == 0:
        return None
    n_correct = counts.get(target, 0)
    p_point = n_correct / n_s
    # bootstrap on a flattened shot list of 0/1 indicators
    indicators = []
    for k, c in counts.items():
        indicators.extend([1 if k == target else 0] * c)
    indicators = np.asarray(indicators, dtype=np.float64)
    if n_s > 1:
        idx = rng.integers(0, n_s, size=(n_boot, n_s))
        boots = indicators[idx].mean(axis=1)
        p_std = float(boots.std(ddof=1))
    else:
        p_std = float("nan")
    return {
        "fidelity": float(p_point),
        "fidelity_std": p_std,
        "n_shots": int(n_s),
        "n_correct": int(n_correct),
    }


# -------------------------- file walking --------------------------------- #


FILE_RE = re.compile(
    r"N(?P<N>\d+)_d(?P<d>\d+)_r(?P<r>\d+)_(?P<kind>XEB|MB|Transport_1QRB)_counts\.json$"
)


def walk_dataset():
    rng = np.random.default_rng(123)
    rows = []
    groups = sorted(os.listdir(RES_DIR))
    for group in groups:
        g_dir = os.path.join(RES_DIR, group)
        if not os.path.isdir(g_dir):
            continue
        for sub in sorted(os.listdir(g_dir)):
            s_dir = os.path.join(g_dir, sub)
            if not os.path.isdir(s_dir):
                continue
            for fname in sorted(os.listdir(s_dir)):
                m = FILE_RE.match(fname)
                if not m:
                    continue
                N = int(m["N"])
                d = int(m["d"])
                r = int(m["r"])
                kind = m["kind"]
                cpath = os.path.join(s_dir, fname)
                counts = load_counts(cpath)

                row = {
                    "group": group,
                    "subset": sub,
                    "N": N,
                    "d": d,
                    "r": r,
                    "kind": kind,
                    "n_shots": sum(counts.values()),
                }

                if kind == "XEB":
                    apath = os.path.join(
                        AMP_DIR, group, sub,
                        f"N{N}_d{d}_r{r}_XEB_amplitudes.json",
                    )
                    if os.path.isfile(apath):
                        p_ideal = load_amplitudes(apath)
                        res = xeb_fidelity(counts, p_ideal, N, rng=rng)
                        if res:
                            row.update(res)
                            row["estimator"] = "XEB"
                            rows.append(row)
                            continue
                    # no amplitudes: skip (cannot compute XEB)
                    row["estimator"] = "XEB"
                    row["fidelity"] = float("nan")
                    row["fidelity_std"] = float("nan")
                    row["n_matched_keys"] = 0
                    rows.append(row)
                else:
                    # MB or Transport_1QRB
                    bpath = os.path.join(
                        s_dir,
                        fname.replace("_counts.json", "_ideal_bitstring.json"),
                    )
                    if os.path.isfile(bpath):
                        target = load_ideal_bitstring(bpath)
                        res = return_prob(counts, target, rng=rng)
                        if res:
                            row.update(res)
                            row["estimator"] = kind
                            rows.append(row)
                            continue
                    row["estimator"] = kind
                    row["fidelity"] = float("nan")
                    row["fidelity_std"] = float("nan")
                    row["n_correct"] = 0
                    rows.append(row)
    return pd.DataFrame(rows)


# -------------------------- aggregation ---------------------------------- #


def aggregate(df):
    out = (
        df.dropna(subset=["fidelity"])
          .groupby(["group", "estimator", "N", "d"], as_index=False)
          .agg(
              n_instances=("r", "nunique"),
              fidelity_mean=("fidelity", "mean"),
              fidelity_sem=("fidelity",
                            lambda s: float(np.std(s, ddof=1) / math.sqrt(len(s))
                                             if len(s) > 1 else float('nan'))),
              fidelity_std=("fidelity", lambda s: float(np.std(s, ddof=1)) if len(s)>1 else float('nan')),
              fidelity_p25=("fidelity", lambda s: float(np.percentile(s, 25))),
              fidelity_p75=("fidelity", lambda s: float(np.percentile(s, 75))),
              fidelity_min=("fidelity", "min"),
              fidelity_max=("fidelity", "max"),
              avg_per_instance_std=("fidelity_std", "mean"),
              total_shots=("n_shots", "sum"),
          )
          .sort_values(["estimator", "group", "N", "d"])
    )
    return out


def main():
    df = walk_dataset()
    df.to_csv(os.path.join(OUT_DIR, "per_instance_fidelity.csv"), index=False)
    print(f"per-instance rows: {len(df)}")

    agg = aggregate(df)
    agg.to_csv(os.path.join(OUT_DIR, "aggregated_fidelity.csv"), index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
