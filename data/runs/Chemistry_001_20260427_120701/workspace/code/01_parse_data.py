"""
Parse 2L3R protein PDB and FK506 ligand SDF, summarise them, and
save a data-overview figure.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDB = os.path.join(ROOT, "data", "sample", "2l3r", "2l3r_protein.pdb")
SDF = os.path.join(ROOT, "data", "sample", "2l3r", "2l3r_ligand.sdf")
OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report", "images")
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

# --- Parse protein PDB (CA atoms only, full residue list) ---------------------
THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

ca_xyz = []
ca_resname = []
ca_resnum = []
all_atom_xyz = []
all_atom_name = []
all_atom_resnum = []
all_atom_resname = []

with open(PDB) as fh:
    for line in fh:
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        resnum = int(line[22:26])
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        all_atom_xyz.append([x,y,z])
        all_atom_name.append(atom_name)
        all_atom_resnum.append(resnum)
        all_atom_resname.append(resname)
        if atom_name == "CA":
            ca_xyz.append([x,y,z])
            ca_resname.append(resname)
            ca_resnum.append(resnum)

ca_xyz = np.array(ca_xyz)
all_atom_xyz = np.array(all_atom_xyz)
sequence = "".join(THREE_TO_ONE.get(r, "X") for r in ca_resname)
print(f"Protein: {len(ca_resname)} residues, {len(all_atom_xyz)} atoms")
print(f"Sequence: {sequence}")

# --- Parse ligand SDF ---------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import AllChem

suppl = Chem.SDMolSupplier(SDF, removeHs=False, sanitize=True)
mol = None
for m in suppl:
    if m is not None:
        mol = m
        break
if mol is None:
    # Fallback: parse manually if RDKit fails
    raise RuntimeError("RDKit could not parse SDF")

n_atoms = mol.GetNumAtoms()
n_heavy = mol.GetNumHeavyAtoms()
n_bonds = mol.GetNumBonds()

# Get coords
conf = mol.GetConformer()
lig_xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])
lig_elements = [a.GetSymbol() for a in mol.GetAtoms()]
heavy_idx = [i for i,a in enumerate(mol.GetAtoms()) if a.GetSymbol() != 'H']
lig_heavy_xyz = lig_xyz[heavy_idx]
lig_heavy_elem = [lig_elements[i] for i in heavy_idx]

print(f"Ligand: {n_atoms} atoms ({n_heavy} heavy), {n_bonds} bonds")
print(f"SMILES: {Chem.MolToSmiles(mol)[:120]}")

# Save processed data
data_summary = {
    "protein": {
        "n_residues": len(ca_resname),
        "n_atoms_total": int(len(all_atom_xyz)),
        "sequence": sequence,
        "ca_centroid": ca_xyz.mean(axis=0).tolist(),
        "radius_of_gyration": float(np.sqrt(((ca_xyz - ca_xyz.mean(axis=0))**2).sum(axis=1).mean()))
    },
    "ligand": {
        "n_atoms": int(n_atoms),
        "n_heavy_atoms": int(n_heavy),
        "n_bonds": int(n_bonds),
        "smiles": Chem.MolToSmiles(mol),
        "elements_heavy": lig_heavy_elem,
        "centroid": lig_heavy_xyz.mean(axis=0).tolist(),
        "max_extent": float((lig_heavy_xyz.max(axis=0) - lig_heavy_xyz.min(axis=0)).max())
    },
    "complex": {
        "ca_to_ligand_min_dist": float(np.linalg.norm(
            ca_xyz[:,None,:] - lig_heavy_xyz[None,:,:], axis=-1).min()),
        "ca_to_ligand_centroid_dist": float(np.linalg.norm(
            ca_xyz.mean(axis=0) - lig_heavy_xyz.mean(axis=0)))
    }
}
with open(os.path.join(OUT, "data_summary.json"), "w") as f:
    json.dump(data_summary, f, indent=2)

# Save numpy arrays for later modules
np.savez(os.path.join(OUT, "parsed_2l3r.npz"),
         ca_xyz=ca_xyz, ca_resnum=np.array(ca_resnum),
         lig_xyz=lig_xyz, lig_elements=np.array(lig_elements, dtype=object),
         lig_heavy_xyz=lig_heavy_xyz, lig_heavy_elem=np.array(lig_heavy_elem, dtype=object))

# --- Data-overview figure -----------------------------------------------------
fig = plt.figure(figsize=(13, 9))

ax1 = fig.add_subplot(2, 3, 1)
ax1.bar(["protein\nresidues", "protein\nall atoms", "ligand\nheavy atoms",
         "ligand\nall atoms", "ligand\nbonds"],
        [len(ca_resname), len(all_atom_xyz), n_heavy, n_atoms, n_bonds],
        color=["#1f77b4","#3a89c9","#ff7f0e","#ffa54a","#888"])
ax1.set_title("Token / atom counts in the 2L3R complex")
ax1.set_ylabel("count")
ax1.tick_params(axis='x', labelsize=8)

ax2 = fig.add_subplot(2, 3, 2)
from collections import Counter
elem_counts = Counter(lig_heavy_elem)
ax2.bar(list(elem_counts.keys()), list(elem_counts.values()),
        color=["#2ca02c","#1f77b4","#d62728","#9467bd"][:len(elem_counts)])
ax2.set_title("Ligand heavy-atom composition")
ax2.set_ylabel("count")

ax3 = fig.add_subplot(2, 3, 3)
ca_to_lig = np.linalg.norm(ca_xyz[:,None,:] - lig_heavy_xyz[None,:,:],
                           axis=-1).min(axis=1)
ax3.plot(ca_resnum, ca_to_lig, color="#444")
ax3.fill_between(ca_resnum, ca_to_lig, alpha=0.3, color="#444")
ax3.axhline(5.0, color="red", ls="--", label="5 Å contact threshold")
ax3.set_xlabel("residue number")
ax3.set_ylabel("min Cα–ligand distance (Å)")
ax3.set_title("Protein–ligand contact profile")
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.plot(ca_xyz[:,0], ca_xyz[:,1], ca_xyz[:,2], color="#1f77b4", lw=1.5,
         label="protein backbone (Cα)")
ax4.scatter(lig_heavy_xyz[:,0], lig_heavy_xyz[:,1], lig_heavy_xyz[:,2],
            color="#ff7f0e", s=20, label="FK506 heavy atoms")
ax4.set_title("3-D view of the FKBP12 / FK506 complex")
ax4.set_xlabel("x (Å)"); ax4.set_ylabel("y (Å)"); ax4.set_zlabel("z (Å)")
ax4.legend(fontsize=8, loc="upper left")

ax5 = fig.add_subplot(2, 3, 5)
# residue-residue distance map
dmap = np.linalg.norm(ca_xyz[:,None,:]-ca_xyz[None,:,:],axis=-1)
im = ax5.imshow(dmap, cmap="viridis", aspect="auto")
ax5.set_title("Cα–Cα distance map (Å)")
ax5.set_xlabel("residue index"); ax5.set_ylabel("residue index")
plt.colorbar(im, ax=ax5, fraction=0.046)

ax6 = fig.add_subplot(2, 3, 6)
# ligand bond-graph 2D projection (use first two PCs)
mu = lig_heavy_xyz.mean(0); X = lig_heavy_xyz - mu
U, S, Vt = np.linalg.svd(X, full_matrices=False)
proj = X @ Vt.T[:, :2]
for b in mol.GetBonds():
    a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    if a in heavy_idx and c in heavy_idx:
        ai = heavy_idx.index(a); ci = heavy_idx.index(c)
        ax6.plot([proj[ai,0], proj[ci,0]], [proj[ai,1], proj[ci,1]],
                 color="grey", lw=0.8, zorder=1)
elem_color = {"C":"#222","O":"#d62728","N":"#1f77b4","P":"#9467bd","S":"#bcbd22"}
for i, e in enumerate(lig_heavy_elem):
    ax6.scatter(proj[i,0], proj[i,1], color=elem_color.get(e,"#888"),
                s=60, zorder=2, edgecolor="white")
ax6.set_aspect("equal"); ax6.set_title("FK506 heavy-atom graph (PCA-2D)")
ax6.set_xlabel("PC1 (Å)"); ax6.set_ylabel("PC2 (Å)")

plt.tight_layout()
plt.savefig(os.path.join(IMG, "data_overview.png"), dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {os.path.join(IMG, 'data_overview.png')}")
print(json.dumps(data_summary, indent=2)[:600])
