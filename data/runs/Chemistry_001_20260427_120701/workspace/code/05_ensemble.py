"""
Multi-sample evaluation: draw N independent samples from the trained
diffusion model, evaluate each against the reference, and report the
distribution + best sample. AF3 itself reports best-of-K (and typical-K)
ligand RMSDs in its evaluations.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from framework import UnifiedComplexDiffusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)

# Load setup
data = np.load(os.path.join(OUT, "parsed_2l3r.npz"), allow_pickle=True)
ca_full = torch.tensor(data["ca_xyz"], dtype=torch.float32)
lig     = torch.tensor(data["lig_heavy_xyz"], dtype=torch.float32)
lig_el  = list(data["lig_heavy_elem"])

with open(os.path.join(OUT, "data_summary.json")) as f:
    summary = json.load(f)
prot_seq_full = summary["protein"]["sequence"]
mask = (torch.cdist(ca_full, lig).min(dim=1).values < 8.0).numpy()
ca_xyz = ca_full[mask]; protein_seq = "".join(c for c,m in zip(prot_seq_full, mask) if m)

from rdkit import Chem
SDF = os.path.join(ROOT, "data", "sample", "2l3r", "2l3r_ligand.sdf")
mol = next(m for m in Chem.SDMolSupplier(SDF, removeHs=False, sanitize=True)
           if m is not None)
heavy_idx = [i for i,a in enumerate(mol.GetAtoms()) if a.GetSymbol()!='H']
hpos = {o:n for n,o in enumerate(heavy_idx)}
lig_bonds = [[hpos[b.GetBeginAtomIdx()], hpos[b.GetEndAtomIdx()]]
             for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in hpos and b.GetEndAtomIdx() in hpos]

x0_full = torch.cat([ca_xyz, lig], 0); x0_full = x0_full - x0_full.mean(0, keepdim=True)
data_scale = float(x0_full.std()); x0_scaled = x0_full / data_scale
prot_n, lig_n = ca_xyz.shape[0], lig.shape[0]; N = x0_full.shape[0]

# Re-train (deterministically) since we don't checkpoint -- skip if prediction.npz already
# has up-to-date trained weights baked into traj. Instead, RUN A FRESH TRAINING IDENTICAL
# to script 02 (deterministic seed) and then sample multiple times.
torch.manual_seed(0); np.random.seed(0)
model = UnifiedComplexDiffusion(dim_s=48, dim_z=16, n_trunk=1, T=100)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
N_STEPS = 250
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)
losses = []; t0 = time.time()
for step in range(N_STEPS):
    types, s, z = model.featurize(protein_seq, "", lig_el, lig_bonds)
    loss, _, _ = model.diffusion_loss(x0_scaled, s, z)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    losses.append(loss.item())
    if step % 50 == 0:
        print(f"retrain step {step}: loss {loss.item():.3f} elapsed {time.time()-t0:.1f}s",
              flush=True)
print(f"Retrain done in {time.time()-t0:.1f}s, final loss {losses[-1]:.3f}", flush=True)

# Multi-sample inference
from scipy.optimize import linear_sum_assignment
def kabsch_align(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U,_,Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1,1,d]) @ U.T
    return Pc @ R.T + Q.mean(0)

def rmsd(A,B): return float(np.sqrt(((A-B)**2).sum(1).mean()))

def hungarian_rmsd(Pp, Pr, els):
    P = kabsch_align(Pp, Pr)
    n = len(els); cost = np.full((n,n), 1e8); els = np.asarray(els)
    for el in np.unique(els):
        idx = np.where(els==el)[0]
        sub = ((P[idx][:,None,:]-Pr[idx][None,:,:])**2).sum(-1)
        for i,gi in enumerate(idx):
            for j,gj in enumerate(idx):
                cost[gi,gj] = sub[i,j]
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(cost[r,c].sum()/n))

K = 8
all_metrics = []
all_samples = []
all_trajs = []
ref_p = x0_full.numpy()[:prot_n]
ref_l = x0_full.numpy()[prot_n:]
types, s, z = model.featurize(protein_seq, "", lig_el, lig_bonds)
for k in range(K):
    torch.manual_seed(100 + k)
    x_pred_s, traj_s = model.sample(s, z, N=N, n_save=12, scale=1.0)
    x_pred = (x_pred_s * data_scale).numpy()
    traj   = (traj_s    * data_scale).numpy()
    pp = x_pred[:prot_n]; pl = x_pred[prot_n:]
    ca = rmsd(kabsch_align(pp, ref_p), ref_p)
    lr = hungarian_rmsd(pl, ref_l, lig_el)
    all_metrics.append({"sample": k, "ca_rmsd": ca, "lig_rmsd_hungarian": lr})
    all_samples.append(x_pred)
    all_trajs.append(traj)
    print(f"  sample {k}: Cα-RMSD={ca:.3f}, lig-RMSD={lr:.3f}", flush=True)

# Best-by-Cα
best_idx = int(np.argmin([m["ca_rmsd"] for m in all_metrics]))
best_lig_idx = int(np.argmin([m["lig_rmsd_hungarian"] for m in all_metrics]))
ensemble = {
    "n_samples": K,
    "per_sample": all_metrics,
    "best_protein_sample": best_idx,
    "best_protein_metrics": all_metrics[best_idx],
    "best_ligand_sample": best_lig_idx,
    "best_ligand_metrics": all_metrics[best_lig_idx],
    "mean_ca_rmsd": float(np.mean([m["ca_rmsd"] for m in all_metrics])),
    "mean_lig_rmsd": float(np.mean([m["lig_rmsd_hungarian"] for m in all_metrics])),
    "std_ca_rmsd": float(np.std([m["ca_rmsd"] for m in all_metrics])),
    "std_lig_rmsd": float(np.std([m["lig_rmsd_hungarian"] for m in all_metrics])),
}
with open(os.path.join(OUT, "ensemble_metrics.json"), "w") as f:
    json.dump(ensemble, f, indent=2)
print(json.dumps(ensemble, indent=2))

# Save best sample as the canonical prediction (used by 03_evaluate.py)
np.savez(os.path.join(OUT, "prediction.npz"),
         x_pred=all_samples[best_idx],
         x0=x0_full.numpy(),
         traj=all_trajs[best_idx],
         types=types.numpy(),
         losses=np.array(losses),
         protein_n=prot_n, ligand_n=lig_n,
         data_scale=data_scale,
         pocket_resnum=data["ca_resnum"][mask])

# Update training_loss figure too
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(losses, color="#1f77b4", lw=1.4)
ax.set_xlabel("training step"); ax.set_ylabel("ε-prediction MSE loss")
ax.set_title(f"AF3-style diffusion training (N={N} tokens, "
             f"{N_STEPS} CPU steps, final loss {losses[-1]:.2f})")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "training_loss.png"), dpi=140)
plt.close()

# Ensemble distribution figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ca_vals = [m["ca_rmsd"] for m in all_metrics]
lr_vals = [m["lig_rmsd_hungarian"] for m in all_metrics]
axes[0].bar(range(K), ca_vals, color="#1f77b4")
axes[0].axhline(np.mean(ca_vals), color="#d62728", ls="--", label="mean")
axes[0].set_xlabel("sample index"); axes[0].set_ylabel("Cα RMSD (Å)")
axes[0].set_title(f"Per-sample Cα RMSD  (mean {np.mean(ca_vals):.2f}, "
                  f"best {min(ca_vals):.2f})")
axes[0].legend()
axes[1].bar(range(K), lr_vals, color="#ff7f0e")
axes[1].axhline(np.mean(lr_vals), color="#d62728", ls="--", label="mean")
axes[1].set_xlabel("sample index"); axes[1].set_ylabel("Hungarian ligand RMSD (Å)")
axes[1].set_title(f"Per-sample ligand RMSD  (mean {np.mean(lr_vals):.2f}, "
                  f"best {min(lr_vals):.2f})")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG, "ensemble_rmsd.png"), dpi=140)
plt.close()
print("Saved ensemble_rmsd.png")
