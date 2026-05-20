import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

os.makedirs("report/images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

with open("outputs/parsed_data.json", "r") as f:
    d = json.load(f)

# Markov transition inference from path selection stats
path_df = pd.DataFrame(d["path_selection_stats"], columns=["path_type", "count"])
total = path_df["count"].sum()
path_df["probability"] = path_df["count"] / total
path_df.to_csv("outputs/path_probabilities.csv", index=False)

# Simplified transition matrix among chiral categories based on mismatch_params and path probabilities
# We use mismatch_params to infer transitions and weight by path probability weights
mismatch_params = pd.DataFrame(d["mismatch_params"], columns=["inner_shell", "outer_shell", "inner_cat", "outer_cat", "mismatch"])
mismatch_params.to_csv("outputs/mismatch_params.csv", index=False)

# Transition matrix rows = inner_cat, cols = outer_cat
cats = list(d["chiral_labels"])
Tmat = np.zeros((len(cats), len(cats)))
for _, row in mismatch_params.iterrows():
    i = cats.index(row["inner_cat"])
    j = cats.index(row["outer_cat"])
    Tmat[i, j] += 1
# Normalize rows
row_sums = Tmat.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
Tmat_norm = Tmat / row_sums

Tmat_df = pd.DataFrame(Tmat_norm, index=cats, columns=cats)
Tmat_df.to_csv("outputs/transition_matrix.csv")

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(Tmat_df.values, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(cats)))
ax.set_yticks(np.arange(len(cats)))
ax.set_xticklabels(cats)
ax.set_yticklabels(cats)
ax.set_xlabel('Outer shell chiral category')
ax.set_ylabel('Inner shell chiral category')
ax.set_title('Inferred Chiral Category Transition Matrix')
for i in range(len(cats)):
    for j in range(len(cats)):
        ax.text(j, i, f"{Tmat_df.values[i,j]:.2f}", ha="center", va="center", color="black" if Tmat_df.values[i,j] < 0.5 else "white")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("report/images/fig9_transition_matrix.png", dpi=300)
plt.close(fig)

# Lennard-Jones potential plot for a representative pair
lj = pd.DataFrame(d["lj_parameters"], columns=["pair", "epsilon", "sigma"])
fig, ax = plt.subplots(figsize=(6,4))
for _, row in lj.iterrows():
    r = np.linspace(0.8*row["sigma"], 3*row["sigma"], 200)
    eps = row["epsilon"]
    sig = row["sigma"]
    U = 4*eps*((sig/r)**12 - (sig/r)**6)
    ax.plot(r, U, label=row["pair"])
ax.set_xlabel('Interatomic distance $r$ (Å)')
ax.set_ylabel('Potential energy $U(r)$')
ax.set_title('Lennard-Jones Potentials for Atomic Pairs')
ax.set_ylim(-1.5, 2.0)
ax.legend(fontsize=7, ncol=2)
ax.axhline(0, color='k', linewidth=0.5)
fig.tight_layout()
fig.savefig("report/images/fig10_lj_potentials.png", dpi=300)
plt.close(fig)

# Summary table of multicomponent clusters
mc = pd.DataFrame(d["multicomponent_clusters"], columns=["cluster", "core_elem", "shell_elem", "core_cat", "shell_cat"])
mc.to_csv("outputs/multicomponent_clusters.csv", index=False)

print("Part4 done: fig9, fig10 saved.")
