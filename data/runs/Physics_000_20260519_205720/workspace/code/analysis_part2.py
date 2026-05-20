import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

os.makedirs("report/images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

with open("outputs/parsed_data.json", "r") as f:
    d = json.load(f)

# Build atomic radii dict
radii = {el: r for el, r in d["atomic_radii"]}
elements = list(radii.keys())

# Compute pairwise size mismatch matrix: (r_j - r_i)/r_i
mismatch_mat = np.zeros((len(elements), len(elements)))
for i, ei in enumerate(elements):
    for j, ej in enumerate(elements):
        mismatch_mat[i, j] = (radii[ej] - radii[ei]) / radii[ei]

mismatch_df = pd.DataFrame(mismatch_mat, index=elements, columns=elements)
mismatch_df.to_csv("outputs/size_mismatch_matrix.csv")

# Figure 3: heatmap with optimal range annotations
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(mismatch_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0, linewidths=.5, ax=ax, vmin=-0.4, vmax=0.4)
ax.set_title('Pairwise Size Mismatch Matrix $\delta = (r_j - r_i)/r_i$')
fig.tight_layout()
fig.savefig("report/images/fig3_size_mismatch_heatmap.png", dpi=300)
plt.close(fig)

# Figure 4: Shell energy landscape
shell_energies = d["shell_energies"]
se_df = pd.DataFrame(shell_energies, columns=["shell_index", "chiral_category", "relative_energy"])
se_df.to_csv("outputs/shell_energies.csv", index=False)
fig, ax = plt.subplots(figsize=(6,5))
for cat in se_df["chiral_category"].unique():
    sub = se_df[se_df["chiral_category"] == cat]
    ax.plot(sub["shell_index"], sub["relative_energy"], 'o-', label=cat, color=d["shell_colors"].get(cat, None))
ax.set_xlabel('Shell index')
ax.set_ylabel('Relative energy (normalized units)')
ax.set_title('Shell Energy Landscape by Chiral Category')
ax.legend(title="Chiral category")
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig("report/images/fig4_shell_energy.png", dpi=300)
plt.close(fig)

# Figure 5: Experimental validation parity plot
exp = np.array(d["experimental_points"])
exp_df = pd.DataFrame(exp, columns=["T_i", "T_j", "measured_sm", "theoretical_sm"])
exp_df.to_csv("outputs/experimental_validation.csv", index=False)
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(exp_df["theoretical_sm"], exp_df["measured_sm"], c='tab:green', s=80, edgecolors='k')
# Add 1:1 line
lims = [min(exp_df["theoretical_sm"].min(), exp_df["measured_sm"].min()) - 0.01,
        max(exp_df["theoretical_sm"].max(), exp_df["measured_sm"].max()) + 0.01]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Theoretical size mismatch')
ax.set_ylabel('Measured size mismatch')
ax.set_title('Experimental Validation Parity Plot')
# annotate points
for _, row in exp_df.iterrows():
    ax.annotate(f"T={int(row['T_i'])}→{int(row['T_j'])}", (row["theoretical_sm"], row["measured_sm"]), textcoords="offset points", xytext=(5,5), fontsize=8)
fig.tight_layout()
fig.savefig("report/images/fig5_experimental_parity.png", dpi=300)
plt.close(fig)

# Compute RMSE and R2
from sklearn.metrics import r2_score, mean_squared_error
rmse = np.sqrt(mean_squared_error(exp_df["theoretical_sm"], exp_df["measured_sm"]))
r2 = r2_score(exp_df["theoretical_sm"], exp_df["measured_sm"])
print(f"RMSE={rmse:.5f}, R2={r2:.4f}")

# Save metrics
metrics = {"RMSE": rmse, "R2": r2}
pd.DataFrame([metrics]).to_csv("outputs/validation_metrics.csv", index=False)

print("Part2 done: fig3, fig4, fig5 saved.")
