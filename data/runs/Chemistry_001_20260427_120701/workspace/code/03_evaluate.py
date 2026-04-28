"""
Evaluate predicted complex against the 2L3R reference.

Metrics:
    - Cα RMSD after Kabsch alignment (protein backbone)
    - Symmetry-aware Hungarian-matched ligand RMSD
    - Combined complex RMSD

Also produces:
    - structural overlay figure (predicted vs. reference)
    - denoising trajectory figure
    - diffusion noise schedule figure
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")
os.makedirs(IMG, exist_ok=True)


# ---------------------------------------------------------------------------
# RMSD helpers
# ---------------------------------------------------------------------------
def kabsch(P: np.ndarray, Q: np.ndarray):
    """Return optimal rotation R and translation t such that R P + t ≈ Q."""
    P = P - P.mean(0)
    Q0 = Q.mean(0)
    Qc = Q - Q0
    H = P.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, Q0


def kabsch_align(P, Q):
    """Align P onto Q (least-squares). Returns aligned P_aligned."""
    Pc = P - P.mean(0)
    R, t = kabsch(Pc, Q)
    return Pc @ R.T + t


def rmsd(A, B):
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))


def hungarian_rmsd(P_pred, P_ref, elements):
    """
    Symmetry-aware ligand RMSD via element-restricted Hungarian assignment.

    Atoms can only match other atoms of the same element. After Kabsch
    alignment we solve a Hungarian assignment problem on the squared-distance
    matrix and report the root-mean-square distance under the optimal
    permutation.
    """
    P = kabsch_align(P_pred, P_ref)
    n = len(elements)
    cost = np.full((n, n), 1e8)
    elements = np.asarray(elements)
    for el in np.unique(elements):
        idx = np.where(elements == el)[0]
        sub = ((P[idx][:, None, :] - P_ref[idx][None, :, :]) ** 2).sum(-1)
        for i, gi in enumerate(idx):
            for j, gj in enumerate(idx):
                cost[gi, gj] = sub[i, j]
    row, col = linear_sum_assignment(cost)
    matched = cost[row, col].sum() / n
    return float(np.sqrt(matched)), col


# ---------------------------------------------------------------------------
# Load prediction
# ---------------------------------------------------------------------------
pred = np.load(os.path.join(OUT, "prediction.npz"), allow_pickle=True)
x_pred = pred["x_pred"]
x0     = pred["x0"]
traj   = pred["traj"]
prot_n = int(pred["protein_n"])
lig_n  = int(pred["ligand_n"])
data_scale = float(pred["data_scale"])

ref_protein  = x0[:prot_n]
ref_ligand   = x0[prot_n:]
pred_protein = x_pred[:prot_n]
pred_ligand  = x_pred[prot_n:]

# Load ligand element list
parsed = np.load(os.path.join(OUT, "parsed_2l3r.npz"), allow_pickle=True)
lig_elem = list(parsed["lig_heavy_elem"])

# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
# Cα RMSD with Kabsch
pred_protein_aligned = kabsch_align(pred_protein, ref_protein)
ca_rmsd = rmsd(pred_protein_aligned, ref_protein)

# Ligand symmetry-aware Hungarian RMSD
lig_rmsd_hungarian, perm = hungarian_rmsd(pred_ligand, ref_ligand, lig_elem)

# Naive ligand RMSD (no permutation)
pred_ligand_aligned = kabsch_align(pred_ligand, ref_ligand)
lig_rmsd_naive = rmsd(pred_ligand_aligned, ref_ligand)

# Combined complex RMSD (joint Kabsch alignment over all atoms)
all_pred_aligned = kabsch_align(x_pred, x0)
complex_rmsd = rmsd(all_pred_aligned, x0)

# Random baseline for sanity check (sample uniform inside ground-truth bbox)
rng = np.random.default_rng(0)
def random_baseline_rmsd(coords_ref, n_runs=20):
    bbox_lo = coords_ref.min(0); bbox_hi = coords_ref.max(0)
    vals = []
    for _ in range(n_runs):
        rnd = rng.uniform(bbox_lo, bbox_hi, size=coords_ref.shape)
        vals.append(rmsd(kabsch_align(rnd, coords_ref), coords_ref))
    return float(np.mean(vals)), float(np.std(vals))

random_protein_rmsd, _ = random_baseline_rmsd(ref_protein)
random_ligand_rmsd, _  = random_baseline_rmsd(ref_ligand)

metrics = {
    "ca_rmsd_after_kabsch_A": ca_rmsd,
    "ligand_rmsd_naive_kabsch_A": lig_rmsd_naive,
    "ligand_rmsd_hungarian_A": lig_rmsd_hungarian,
    "complex_rmsd_after_kabsch_A": complex_rmsd,
    "random_baseline_protein_rmsd_A": random_protein_rmsd,
    "random_baseline_ligand_rmsd_A": random_ligand_rmsd,
    "improvement_protein_vs_random": float(random_protein_rmsd / max(ca_rmsd, 1e-6)),
    "improvement_ligand_vs_random": float(random_ligand_rmsd / max(lig_rmsd_hungarian, 1e-6)),
    "n_protein_tokens": prot_n,
    "n_ligand_tokens": lig_n,
    "data_scale_used_during_diffusion_A": data_scale,
}
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Figure: structural overlay (predicted vs. reference)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(ref_protein[:, 0], ref_protein[:, 1], ref_protein[:, 2],
         color="#1f77b4", lw=2.5, label="reference Cα", zorder=3)
ax1.scatter(ref_ligand[:, 0], ref_ligand[:, 1], ref_ligand[:, 2],
            color="#ff7f0e", s=22, label="reference FK506", zorder=4)
ax1.plot(pred_protein_aligned[:, 0], pred_protein_aligned[:, 1],
         pred_protein_aligned[:, 2], color="#2ca02c", lw=1.6,
         linestyle="--", label="predicted Cα", zorder=2)
ax1.scatter(pred_ligand_aligned[:, 0], pred_ligand_aligned[:, 1],
            pred_ligand_aligned[:, 2], color="#d62728", s=22,
            marker="^", label="predicted ligand", zorder=2)
ax1.set_title(f"Structural overlay (Cα-RMSD = {ca_rmsd:.2f} Å)")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
ax1.legend(fontsize=8, loc="upper left")

ax2 = fig.add_subplot(1, 2, 2)
labels = ["random\nbaseline\n(protein)", "model\nprediction\n(protein)",
          "random\nbaseline\n(ligand)", "model prediction\n(ligand,\nHungarian)"]
vals = [random_protein_rmsd, ca_rmsd, random_ligand_rmsd, lig_rmsd_hungarian]
colors = ["#888","#1f77b4","#bbb","#ff7f0e"]
ax2.bar(labels, vals, color=colors)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)
ax2.set_ylabel("RMSD (Å)")
ax2.set_title("Prediction RMSD vs. random-coordinate baseline")
ax2.tick_params(axis='x', labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "structural_overlay.png"), dpi=140,
            bbox_inches="tight")
plt.close()
print("Saved structural_overlay.png")


# ---------------------------------------------------------------------------
# Figure: denoising trajectory
# ---------------------------------------------------------------------------
T = traj.shape[0]
ncol = 4
nrow = (T + ncol - 1) // ncol
fig = plt.figure(figsize=(4 * ncol, 3.5 * nrow))
for i in range(T):
    ax = fig.add_subplot(nrow, ncol, i + 1, projection='3d')
    pi_p = traj[i, :prot_n]
    pi_l = traj[i, prot_n:]
    ax.plot(pi_p[:, 0], pi_p[:, 1], pi_p[:, 2], color="#1f77b4", lw=1.0)
    ax.scatter(pi_l[:, 0], pi_l[:, 1], pi_l[:, 2], color="#ff7f0e", s=8)
    pct = int(round((1 - i / max(T - 1, 1)) * 100))
    ax.set_title(f"step {i+1}/{T}  (~{pct}% noise)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
plt.suptitle("Diffusion denoising trajectory: random noise → predicted "
             "FKBP12 / FK506 complex", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "denoising_trajectory.png"), dpi=130,
            bbox_inches="tight")
plt.close()
print("Saved denoising_trajectory.png")


# ---------------------------------------------------------------------------
# Figure: diffusion noise schedule
# ---------------------------------------------------------------------------
import torch
sys.path.append(os.path.join(ROOT, "code"))
from framework import cosine_alpha_bar
T = 100
ab = cosine_alpha_bar(T).numpy()
beta = 1 - ab[1:] / ab[:-1]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(np.arange(T+1), ab, color="#1f77b4")
axes[0].set_xlabel("timestep t"); axes[0].set_ylabel(r"$\bar\alpha_t$")
axes[0].set_title("Cosine schedule cumulative product")
axes[0].grid(alpha=0.3)

axes[1].plot(np.arange(1, T+1), beta, color="#d62728")
axes[1].set_xlabel("timestep t"); axes[1].set_ylabel(r"$\beta_t$")
axes[1].set_title("Per-step noise rate")
axes[1].grid(alpha=0.3)

axes[2].plot(np.arange(T+1), np.sqrt(ab), label=r"$\sqrt{\bar\alpha_t}$ "
             "(signal)", color="#1f77b4")
axes[2].plot(np.arange(T+1), np.sqrt(1 - ab), label=r"$\sqrt{1-\bar\alpha_t}$ "
             "(noise)", color="#d62728")
axes[2].set_xlabel("timestep t"); axes[2].set_ylabel("magnitude")
axes[2].set_title("Signal vs. noise scaling under DDPM forward process")
axes[2].grid(alpha=0.3); axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG, "diffusion_schedule.png"), dpi=140,
            bbox_inches="tight")
plt.close()
print("Saved diffusion_schedule.png")
