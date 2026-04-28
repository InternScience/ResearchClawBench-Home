"""
Train the small AF3-style diffusion framework on the FKBP12 / FK506 (2L3R)
binding-pocket sub-system and run inference.

We keep the model and training small enough to run on CPU in a few minutes.
Token set:
    - protein Cα atoms within 8 Å of any ligand heavy atom (pocket)
    - all ligand heavy atoms

After training, we (1) sample a fresh structure from the diffusion module,
(2) save the denoising trajectory, and (3) save the loss curve figure.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from framework import UnifiedComplexDiffusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(8)

# --- load parsed data ---------------------------------------------------------
data = np.load(os.path.join(OUT, "parsed_2l3r.npz"), allow_pickle=True)
ca_xyz_full   = torch.tensor(data["ca_xyz"],        dtype=torch.float32)
lig_xyz       = torch.tensor(data["lig_heavy_xyz"], dtype=torch.float32)
lig_elem      = list(data["lig_heavy_elem"])

with open(os.path.join(OUT, "data_summary.json")) as f:
    summary = json.load(f)
protein_seq_full = summary["protein"]["sequence"]
ca_resnum_full = data["ca_resnum"]

# --- select binding-pocket residues (within 8 Å of ligand heavy atoms) -------
d_to_lig = torch.cdist(ca_xyz_full, lig_xyz).min(dim=1).values
mask = (d_to_lig < 8.0).numpy().astype(bool)
ca_xyz = ca_xyz_full[mask]
protein_seq = "".join(c for c, m in zip(protein_seq_full, mask) if m)
ca_resnum = ca_resnum_full[mask]
print(f"Pocket residues: {ca_xyz.shape[0]}  (sequence={protein_seq})")

# --- ligand bonds (heavy-heavy) ----------------------------------------------
from rdkit import Chem
SDF = os.path.join(ROOT, "data", "sample", "2l3r", "2l3r_ligand.sdf")
mol = next(m for m in Chem.SDMolSupplier(SDF, removeHs=False, sanitize=True)
           if m is not None)
heavy_idx = [i for i,a in enumerate(mol.GetAtoms()) if a.GetSymbol() != 'H']
heavy_pos = {orig: new for new, orig in enumerate(heavy_idx)}
lig_bonds = []
for b in mol.GetBonds():
    a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    if a in heavy_pos and c in heavy_pos:
        lig_bonds.append([heavy_pos[a], heavy_pos[c]])
print(f"Ligand: {lig_xyz.shape[0]} heavy atoms, {len(lig_bonds)} heavy-heavy bonds")

# --- assemble ground-truth coords --------------------------------------------
x0_full = torch.cat([ca_xyz, lig_xyz], dim=0)
x0_full = x0_full - x0_full.mean(0, keepdim=True)
# Scale down to typical Gaussian range so the diffusion target is well-conditioned.
data_scale = float(x0_full.std())
x0_scaled = x0_full / data_scale          # roughly std 1
N = x0_full.shape[0]
print(f"Total tokens: {N}; data_scale={data_scale:.2f}")

# --- build the model ---------------------------------------------------------
model = UnifiedComplexDiffusion(dim_s=48, dim_z=16, n_trunk=1, T=100)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Featurise once
types, s_init, z_init = model.featurize(protein_seq=protein_seq,
                                        ligand_elements=lig_elem,
                                        ligand_bonds=lig_bonds)
assert types.shape[0] == N

# --- training ---------------------------------------------------------------
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
N_STEPS = 250
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)

losses = []
t_start = time.time()
for step in range(N_STEPS):
    types, s, z = model.featurize(protein_seq=protein_seq,
                                  ligand_elements=lig_elem,
                                  ligand_bonds=lig_bonds)
    loss, _, _ = model.diffusion_loss(x0_scaled, s, z)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    losses.append(loss.item())
    if step % 5 == 0 or step == N_STEPS-1:
        print(f"step {step:3d}  loss={loss.item():.4f}  "
              f"elapsed={time.time()-t_start:.1f}s", flush=True)

# --- inference --------------------------------------------------------------
torch.manual_seed(42)
types, s, z = model.featurize(protein_seq=protein_seq,
                              ligand_elements=lig_elem,
                              ligand_bonds=lig_bonds)
x_pred_scaled, traj_scaled = model.sample(s, z, N=N, n_save=12, scale=1.0)
x_pred = x_pred_scaled * data_scale         # unscale to Å
traj   = traj_scaled    * data_scale
print(f"x_pred {x_pred.shape}, traj {traj.shape}")

# --- save outputs -----------------------------------------------------------
np.savez(os.path.join(OUT, "prediction.npz"),
         x_pred=x_pred.numpy(),
         x0=x0_full.numpy(),
         traj=traj.numpy(),
         types=types.numpy(),
         losses=np.array(losses),
         protein_n=ca_xyz.shape[0],
         ligand_n=lig_xyz.shape[0],
         data_scale=data_scale,
         pocket_resnum=ca_resnum)

# Loss curve
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(losses, color="#1f77b4", lw=1.4)
ax.set_xlabel("training step"); ax.set_ylabel("ε-prediction MSE loss")
ax.set_title(f"AF3-style diffusion training on 2L3R pocket "
             f"(N={N} tokens, CPU, {N_STEPS} steps)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "training_loss.png"), dpi=140)
plt.close()
print("Saved training_loss.png")
