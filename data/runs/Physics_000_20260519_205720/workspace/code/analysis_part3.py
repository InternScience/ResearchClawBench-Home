import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

os.makedirs("report/images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

with open("outputs/parsed_data.json", "r") as f:
    d = json.load(f)

# Growth results: 18 rows, every 6 rows is one trajectory
gr = np.array(d["growth_results"])
gr_df = pd.DataFrame(gr, columns=["step", "category", "avg_mismatch"])
# split into trajectories (6 points each)
trajectories = [gr_df.iloc[i*6:(i+1)*6] for i in range(3)]

fig, ax = plt.subplots(figsize=(6,5))
colors = ['tab:blue', 'tab:green', 'tab:red']
labels = ['Seed 1: Na13 + Na (MC)', 'Seed 2: Na13@Rb32 + Rb (Ch1)', 'Seed 3: Ag13 + Cu (MC→Ch1)']
for traj, color, label in zip(trajectories, colors, labels):
    ax.plot(traj["step"], traj["avg_mismatch"], 'o-', color=color, label=label)
ax.set_xlabel('Simulation step')
ax.set_ylabel('Average size mismatch')
ax.set_title('Growth Trajectories: Average Mismatch vs Step')
ax.legend(fontsize=8)
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig("report/images/fig6_growth_trajectories.png", dpi=300)
plt.close(fig)
gr_df.to_csv("outputs/growth_results.csv", index=False)

# Path selection stats
path_stats = d["path_selection_stats"]
path_df = pd.DataFrame(path_stats, columns=["path_type", "count"])
path_df.to_csv("outputs/path_selection_stats.csv", index=False)

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(path_df["path_type"], path_df["count"], color=['steelblue','coral','gold','mediumpurple'])
ax.set_ylabel('Count')
ax.set_title('Path Selection Statistics from Growth Simulations')
ax.set_xticklabels(path_df["path_type"], rotation=15, ha='right')
fig.tight_layout()
fig.savefig("report/images/fig7_path_stats.png", dpi=300)
plt.close(fig)

# Predict stable multi-shell structures using mismatch matrix and optimal ranges
radii = {el: r for el, r in d["atomic_radii"]}
optimal = d["optimal_mismatch_ranges"]  # list of [inner_cat, outer_cat, min, max]

predictions = []
for el1, r1 in radii.items():
    for el2, r2 in radii.items():
        if el1 == el2:
            continue
        sm = (r2 - r1) / r1
        for inner, outer, smin, smax in optimal:
            if smin <= abs(sm) <= smax:
                predictions.append({
                    "core": el1,
                    "shell": el2,
                    "core_cat": inner,
                    "shell_cat": outer,
                    "mismatch": round(sm, 4),
                    "in_range": True
                })

pred_df = pd.DataFrame(predictions)
pred_df.to_csv("outputs/predicted_stable_structures.csv", index=False)

# Figure 8: mismatch compliance chart (bar of predicted pairs)
if not pred_df.empty:
    pred_df["pair"] = pred_df["core"] + "→" + pred_df["shell"]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(pred_df["pair"], pred_df["mismatch"], color='teal')
    ax.set_xlabel('Predicted size mismatch')
    ax.set_title('Predicted Stable Core→Shell Pairs within Optimal Mismatch Ranges')
    fig.tight_layout()
    fig.savefig("report/images/fig8_predicted_pairs.png", dpi=300)
    plt.close(fig)

print("Part3 done: fig6, fig7, fig8 saved. Predicted pairs:", len(pred_df))
