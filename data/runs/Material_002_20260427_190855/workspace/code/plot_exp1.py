"""Plot RDFs and MD log for Experiment 1 (improved: separate intra/inter)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUTS, IMAGES

with open(os.path.join(OUTPUTS, "exp1_rdf.json")) as f:
    rdf = json.load(f)
with open(os.path.join(OUTPUTS, "exp1_water_md_log.json")) as f:
    log = json.load(f)


# ---- Recompute intermolecular RDFs from saved trajectory --------------------
import importlib.util
spec = importlib.util.spec_from_file_location("exp1", os.path.join(os.path.dirname(__file__), "exp1_water_md.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

import numpy as np
data = np.load(os.path.join(OUTPUTS, "exp1_water_md.npz"))
positions = data["positions"]
symbols = list(map(str, data["symbols"]))
cell = data["cell"]
skip = rdf["meta"]["skip_frames"]
used = positions[skip:]
print("frames used:", len(used))


def compute_rdf_inter(positions_list, symbols, cell, pair, mol_size=3,
                      r_max=6.0, n_bins=120):
    """Intermolecular g(r) - exclude pairs that belong to the same molecule.

    The water box is built one molecule at a time with order O,H,H per
    molecule, so molecule index = atom_index // 3.
    """
    a, b = pair
    sym = np.array(symbols)
    idx_a = np.where(sym == a)[0]
    idx_b = np.where(sym == b)[0]
    mol_a = idx_a // mol_size
    mol_b = idx_b // mol_size
    L = np.diag(cell)
    bins = np.linspace(0, r_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist = np.zeros(n_bins)
    for pos in positions_list:
        ra = pos[idx_a]; rb = pos[idx_b]
        d = ra[:, None, :] - rb[None, :, :]
        d -= np.round(d / L) * L
        r = np.linalg.norm(d, axis=-1)
        # exclude same molecule
        mask_same = mol_a[:, None] == mol_b[None, :]
        if a == b:
            mask_diag = np.eye(r.shape[0], dtype=bool)
            mask = ~(mask_same | mask_diag)
        else:
            mask = ~mask_same
        r = r[mask]
        h, _ = np.histogram(r, bins=bins)
        hist += h
    n_frames = len(positions_list)
    n_a, n_b = len(idx_a), len(idx_b)
    V = float(np.linalg.det(cell))
    # number of unique pair partners per atom of type a after exclusion
    if a == b:
        # for each a-atom, partners are atoms of same species not in same molecule
        partners_per_atom = n_b - mol_size  # 2 of own H/O excluded? Use general
        # generic correction: average via mask_same shape
        partners_per_atom = (n_b - 1) - (mol_size - 1) if a == "H" else (n_b - 1)  # crude
    rho_b = n_b / V
    shell_vol = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    # number of *intermolecular* pairs (a,b)
    # = n_a * n_b - n_a*(per-mol b atoms in same mol)
    # per molecule in our build: each molecule has 1 O + 2 H
    if a == b == "O":
        per_mol_a_in_b = 1
    elif a == b == "H":
        per_mol_a_in_b = 2
    elif (a, b) in (("O", "H"), ("H", "O")):
        per_mol_a_in_b = 2 if (a == "O") else 1  # for a-O, partners H per mol = 2
    n_pairs = n_a * n_b - n_a * per_mol_a_in_b
    if a == b:
        # exclude self; we already excluded same-molecule which includes self for O
        # but for H, same-molecule already excludes self+other H -> ok
        pass
    norm = rho_b * shell_vol * (n_pairs / n_a) * n_frames * n_a / n_b  # fall back: use simple normalization
    # Simpler: avg pair count -> g = hist / (rho * shell_vol * n_a * n_frames)
    # which already counts intermolecular only because hist was built that way
    norm = rho_b * shell_vol * n_a * n_frames
    g = hist / norm
    return centers, g


pairs = [("O", "O"), ("O", "H"), ("H", "H")]
labels = {"OO": ("O–O", "#1f77b4"), "OH": ("O–H", "#d62728"), "HH": ("H–H", "#2ca02c")}
inter = {}
for p in pairs:
    r, g = compute_rdf_inter(used, symbols, cell, p, r_max=6.0, n_bins=120)
    inter[p[0] + p[1]] = {"r": r.tolist(), "g": g.tolist()}

# augment json
with open(os.path.join(OUTPUTS, "exp1_rdf.json")) as f:
    rdf_full = json.load(f)
rdf_full["inter"] = inter
with open(os.path.join(OUTPUTS, "exp1_rdf.json"), "w") as f:
    json.dump(rdf_full, f)


# ---- Combined figure ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
for key in ["OO", "OH", "HH"]:
    name, c = labels[key]
    r = np.array(rdf[key]["r"]); g = np.array(rdf[key]["g"])
    ax.plot(r, g, color=c, lw=1.8, label=f"{name}")
ax.set_xlim(0, 6); ax.set_ylim(0, 8)
ax.set_xlabel("r (Å)"); ax.set_ylabel("g(r) (all pairs)")
ax.set_title("All-pair RDF (intra + inter)")
ax.legend(frameon=False)

ax = axes[1]
ref = {"OO": 2.80, "OH": 1.85, "HH": 2.45}  # experimental peak positions (Soper '07)
for key in ["OO", "OH", "HH"]:
    name, c = labels[key]
    r = np.array(inter[key]["r"]); g = np.array(inter[key]["g"])
    ax.plot(r, g, color=c, lw=1.8, label=f"{name}")
    ax.axvline(ref[key], color=c, ls=":", alpha=0.5)
ax.set_xlim(0, 6); ax.set_ylim(0, None)
ax.set_xlabel("r (Å)"); ax.set_ylabel("g$_{inter}$(r)")
ax.set_title("Intermolecular RDF (dotted: exp. peak positions)")
ax.legend(frameon=False)

meta = rdf["meta"]
fig.suptitle(f"Liquid-water RDF — MACE-MP-0b3-medium, 32 H$_2$O, "
             f"{meta['n_steps']} steps × {meta['dt_fs']} fs at T={meta['T_target_K']} K, "
             f"{meta['n_frames_used']} averaged frames",
             y=1.02, fontsize=11)
fig.tight_layout()
out = os.path.join(IMAGES, "water_rdf.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)


# ---- MD energy / temperature ----
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
t = [d["time_fs"] for d in log]
E = [d["Epot_eV"] for d in log]
T = [d["T_K"] for d in log]
axes[0].plot(t, E, color="#1f77b4", lw=1.5)
axes[0].set_ylabel("Potential energy (eV)")
axes[1].plot(t, T, color="#d62728", lw=1.5)
axes[1].axhline(330, color="grey", ls="--", lw=1, label="target 330 K")
axes[1].set_ylabel("Temperature (K)")
axes[1].set_xlabel("Time (fs)")
axes[1].legend(frameon=False, loc="upper right")
axes[0].set_title("Langevin NVT MD trace, MACE-MP-0b3-medium / 32 H$_2$O")
fig.tight_layout()
out = os.path.join(IMAGES, "md_energy.png")
fig.savefig(out, dpi=150)
print("wrote", out)
