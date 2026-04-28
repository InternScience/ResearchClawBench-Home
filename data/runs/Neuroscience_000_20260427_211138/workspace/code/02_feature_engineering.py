"""
02_feature_engineering.py
Engineer SimBA-style frame-level features from the 8 body parts × 2 mice pose
data. Output: outputs/engineered_features.csv (frames x N features) plus
outputs/feature_inventory.json describing each feature group.

Feature groups:
- Pose probabilities (16 raw + 2 means).
- Pairwise distances within mouse (28 per mouse * 2 = 56)
- Inter-mouse distances between equivalent body parts (8) and full inter-mouse
  pairwise distances between centers/noses/tail bases (16 selected).
- Velocity magnitudes per body part (16) and accelerations (16).
- Convex-hull "size" proxy: bounding-box width, height, diagonal, per mouse (3*2).
- Body angle: nose -> tail_base direction in radians and its angular velocity.
- Distance between the two animals' centers (1) and its first/second derivatives.
- Rolling means and stds (window=15 ≈ 0.5 s at 30 fps) over a small subset of
  the most informative kinematic features.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = ROOT / "outputs"

feat = pd.read_csv(DATA / "Together_1_features_extracted.csv")
tgt  = pd.read_csv(DATA / "Together_1_targets_inserted.csv")
assert (feat.shape[0] == tgt.shape[0])

BODYPARTS = ["Nose", "Ear_left", "Ear_right", "Center", "Lat_left",
             "Lat_right", "Tail_base", "Tail_end"]
ANIMALS = [1, 2]

def col(bp, a, kind):
    return f"{bp}_{a}_{kind}"

def xy(df, bp, a):
    return df[col(bp, a, "x")].values, df[col(bp, a, "y")].values

X = {}  # feature dict

# 1) Probabilities
p_means = {a: [] for a in ANIMALS}
for a in ANIMALS:
    for bp in BODYPARTS:
        p = feat[col(bp, a, "p")].values
        X[f"prob_{bp}_{a}"] = p
        p_means[a].append(p)
for a in ANIMALS:
    X[f"prob_mean_animal_{a}"] = np.mean(p_means[a], axis=0)

# 2) Within-animal pairwise distances
def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

for a in ANIMALS:
    coords = {bp: xy(feat, bp, a) for bp in BODYPARTS}
    for i in range(len(BODYPARTS)):
        for j in range(i + 1, len(BODYPARTS)):
            bp_i, bp_j = BODYPARTS[i], BODYPARTS[j]
            x1, y1 = coords[bp_i]
            x2, y2 = coords[bp_j]
            X[f"d_{bp_i}_{bp_j}_a{a}"] = dist(x1, y1, x2, y2)

# 3) Inter-animal distances between equivalent body parts
for bp in BODYPARTS:
    x1, y1 = xy(feat, bp, 1)
    x2, y2 = xy(feat, bp, 2)
    X[f"d_inter_{bp}"] = dist(x1, y1, x2, y2)

# 4) Inter-animal cross-distances (Nose1<->Tail_base2 etc.) — proxies of social posture
for bp_a in ["Nose", "Center", "Tail_base"]:
    for bp_b in ["Nose", "Center", "Tail_base"]:
        if bp_a == bp_b:
            continue
        x1, y1 = xy(feat, bp_a, 1)
        x2, y2 = xy(feat, bp_b, 2)
        X[f"d_cross_{bp_a}1_{bp_b}2"] = dist(x1, y1, x2, y2)

# 5) Velocity & acceleration magnitudes per bodypart
for a in ANIMALS:
    for bp in BODYPARTS:
        x, y = xy(feat, bp, a)
        dx = np.gradient(x); dy = np.gradient(y)
        v = np.sqrt(dx**2 + dy**2)
        ax = np.gradient(dx); ay = np.gradient(dy)
        acc = np.sqrt(ax**2 + ay**2)
        X[f"vel_{bp}_a{a}"] = v
        X[f"acc_{bp}_a{a}"] = acc

# 6) Bounding-box size per animal
for a in ANIMALS:
    xs = np.stack([feat[col(bp, a, "x")].values for bp in BODYPARTS], axis=1)
    ys = np.stack([feat[col(bp, a, "y")].values for bp in BODYPARTS], axis=1)
    bw = xs.max(axis=1) - xs.min(axis=1)
    bh = ys.max(axis=1) - ys.min(axis=1)
    X[f"bbox_w_a{a}"] = bw
    X[f"bbox_h_a{a}"] = bh
    X[f"bbox_diag_a{a}"] = np.sqrt(bw**2 + bh**2)

# 7) Body angle (nose - tail_base) and angular velocity
for a in ANIMALS:
    nx, ny = xy(feat, "Nose", a)
    tx, ty = xy(feat, "Tail_base", a)
    ang = np.arctan2(ny - ty, nx - tx)
    X[f"angle_a{a}"] = ang
    # unwrap then derivative
    ang_u = np.unwrap(ang)
    X[f"angvel_a{a}"] = np.gradient(ang_u)

# 8) Inter-animal center distance derivatives & relative motion
cx1, cy1 = xy(feat, "Center", 1)
cx2, cy2 = xy(feat, "Center", 2)
inter_d = dist(cx1, cy1, cx2, cy2)
X["inter_center_dist"] = inter_d
X["inter_center_dist_d1"] = np.gradient(inter_d)
X["inter_center_dist_d2"] = np.gradient(np.gradient(inter_d))

# Relative angle of mouse2 w.r.t. mouse1 heading
heading_dx = (feat[col("Nose",1,"x")].values - feat[col("Tail_base",1,"x")].values)
heading_dy = (feat[col("Nose",1,"y")].values - feat[col("Tail_base",1,"y")].values)
to_other_dx = cx2 - cx1
to_other_dy = cy2 - cy1
def signed_angle(a_dx, a_dy, b_dx, b_dy):
    a = np.arctan2(a_dy, a_dx)
    b = np.arctan2(b_dy, b_dx)
    diff = b - a
    return np.arctan2(np.sin(diff), np.cos(diff))
X["rel_angle_1_to_2"] = signed_angle(heading_dx, heading_dy, to_other_dx, to_other_dy)

# Build dataframe
df = pd.DataFrame(X)

# 9) Rolling stats (window = 15 frames) on a curated kinematic subset
WINDOW = 15
roll_keys = [
    "inter_center_dist", "inter_center_dist_d1",
    "vel_Center_a1", "vel_Center_a2",
    "vel_Nose_a1",   "vel_Nose_a2",
    "d_inter_Nose",  "d_inter_Center",  "d_inter_Tail_base",
    "rel_angle_1_to_2",
    "angvel_a1", "angvel_a2",
]
for k in roll_keys:
    s = df[k]
    df[f"{k}_rmean{WINDOW}"] = s.rolling(WINDOW, min_periods=1, center=True).mean()
    df[f"{k}_rstd{WINDOW}"]  = s.rolling(WINDOW, min_periods=1, center=True).std().fillna(0.0)

# Replace any NaN/inf with finite values
df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Attach labels for convenience (these will be split off before training)
df["__Attack__"] = tgt["Attack"].values
df["__Sniffing__"] = tgt["Sniffing"].values

print("Engineered feature matrix:", df.shape)
df.to_csv(OUT / "engineered_features.csv", index=False)

inventory = {
    "n_frames": int(df.shape[0]),
    "n_features": int(df.shape[1] - 2),
    "groups": {
        "probabilities": [c for c in df.columns if c.startswith("prob_")],
        "within_distances": [c for c in df.columns if c.startswith("d_") and "_a" in c[-3:]],
        "inter_distances":  [c for c in df.columns if c.startswith("d_inter_")],
        "cross_distances":  [c for c in df.columns if c.startswith("d_cross_")],
        "velocities":       [c for c in df.columns if c.startswith("vel_")],
        "accelerations":    [c for c in df.columns if c.startswith("acc_")],
        "bbox":             [c for c in df.columns if c.startswith("bbox_")],
        "angles":           [c for c in df.columns if c.startswith("angle_") or c.startswith("angvel_") or c.startswith("rel_angle")],
        "inter_center":     [c for c in df.columns if c.startswith("inter_center")],
        "rolling":          [c for c in df.columns if c.endswith(f"_rmean{WINDOW}") or c.endswith(f"_rstd{WINDOW}")],
    },
}
with open(OUT / "feature_inventory.json", "w") as f:
    json.dump({"summary":{k: len(v) for k, v in inventory["groups"].items()},
               "n_features": inventory["n_features"],
               "n_frames": inventory["n_frames"]}, f, indent=2)
print("Saved outputs/engineered_features.csv  &  outputs/feature_inventory.json")
