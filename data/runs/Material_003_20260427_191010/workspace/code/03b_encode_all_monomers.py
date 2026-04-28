"""
Phase 5b: encode every unique vitrimer monomer (acid + epoxide) using the
trained graph VAE. The training set in 03_train_graph_vae.py was a stratified
8000-monomer subsample for speed; here we run the *trained* encoder on the
full 15396 unique-monomer corpus so downstream property modelling and
pair-level predictions can use all 8424 vitrimer pairs.

Outputs:
  - outputs/vae_latents_all.npz      (z, smiles, kind)
"""
import os, re
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0); np.random.seed(0)
DEV = torch.device("cpu")

ckpt = torch.load(os.path.join(ROOT, "outputs/vae_state.pt"),
                  map_location=DEV, weights_only=False)
vocab = ckpt["vocab"]; V = len(vocab)
ATOM_FDIM = ckpt["atom_fdim"]; BOND_FDIM = ckpt["bond_fdim"]
Z_DIM = ckpt["z_dim"]

# Recreate model classes with the *currently used* hyperparameters
class GINLayer(nn.Module):
    def __init__(self, d, edge_dim):
        super().__init__()
        self.edge_lin = nn.Linear(edge_dim, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.eps = nn.Parameter(torch.tensor(0.0))
    def forward(self, h, src, dst, ef):
        m = h[src] + self.edge_lin(ef)
        agg = torch.zeros_like(h); agg.index_add_(0, dst, m)
        return self.mlp((1 + self.eps) * h + agg)

class GraphEncoder(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden=64, n_layers=3, z_dim=64):
        super().__init__()
        self.in_lin = nn.Linear(atom_dim, hidden)
        self.layers = nn.ModuleList(GINLayer(hidden, edge_dim) for _ in range(n_layers))
        self.read = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.fc_mu = nn.Linear(hidden, z_dim); self.fc_lv = nn.Linear(hidden, z_dim)
    def forward(self, x, src, dst, ef, batch):
        h = F.relu(self.in_lin(x))
        for l in self.layers: h = h + F.relu(l(h, src, dst, ef))
        n_g = int(batch.max().item()) + 1
        g = torch.zeros(n_g, h.shape[1]); g.index_add_(0, batch, h)
        g = self.read(g)
        return self.fc_mu(g), self.fc_lv(g)

class GRUDecoder(nn.Module):
    def __init__(self, V, z_dim, emb=32, hidden=128, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.z2h = nn.Linear(z_dim, hidden)
        self.gru = nn.GRU(emb + z_dim, hidden, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden, V)
        self.n_layers = n_layers; self.hidden = hidden; self.z_dim = z_dim

class GraphVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = GraphEncoder(ATOM_FDIM, BOND_FDIM, z_dim=Z_DIM)
        self.dec = GRUDecoder(V, Z_DIM)
        self.z_dim = Z_DIM

vae = GraphVAE().to(DEV)
vae.load_state_dict(ckpt["state_dict"])
vae.eval()

# Featurization (must match 03_train_graph_vae.py)
ATOM_LIST = [6, 7, 8, 9, 16, 17, 35, 53, 14, 15]
HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
       Chem.rdchem.HybridizationType.SP3D2]
def atom_features(atom):
    z = atom.GetAtomicNum()
    feat = [int(z == zz) for zz in ATOM_LIST]
    feat.append(int(z not in ATOM_LIST))
    feat.append(atom.GetDegree() / 6.0)
    feat.append(atom.GetFormalCharge() / 2.0)
    feat.append(int(atom.GetIsAromatic()))
    feat.append(int(atom.IsInRing()))
    feat.extend([int(atom.GetHybridization() == h) for h in HYB])
    feat.append(atom.GetTotalNumHs() / 4.0)
    return feat
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
def bond_features(bond):
    feat = [int(bond.GetBondType() == bt) for bt in BOND_TYPES]
    feat.append(int(bond.IsInRing()))
    feat.append(int(bond.GetIsConjugated()))
    return feat

def mol_to_graph(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    n = m.GetNumAtoms()
    x = np.zeros((n, ATOM_FDIM), dtype=np.float32)
    for i, a in enumerate(m.GetAtoms()): x[i] = atom_features(a)
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]; dst += [j, i]; ef += [f, f]
    if not src: src, dst, ef = [0], [0], [[0]*BOND_FDIM]
    return (x, np.asarray(src,dtype=np.int64), np.asarray(dst,dtype=np.int64),
            np.asarray(ef, dtype=np.float32))

vit = pd.read_csv(os.path.join(ROOT, "outputs/vitrimer_calibrated_tg.csv"))

def canon(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m, canonical=True) if m else None
acids   = vit["acid"].map(canon).dropna().unique().tolist()
epoxes  = vit["epoxide"].map(canon).dropna().unique().tolist()
all_smi = sorted(set(acids + epoxes))
kind = ["acid" if s in set(acids) and s not in set(epoxes) else
        ("epoxide" if s in set(epoxes) and s not in set(acids) else "both")
        for s in all_smi]
print("unique monomers:", len(all_smi))

graphs = []
keep = []
for s in all_smi:
    g = mol_to_graph(s)
    if g is None: continue
    graphs.append(g); keep.append(s)
all_smi = keep
print("graph-encodable monomers:", len(all_smi))

def collate(batch_idx):
    xs, srcs, dsts, efs, batch = [], [], [], [], []
    offset = 0
    for k, i in enumerate(batch_idx):
        x, src, dst, ef = graphs[i]
        xs.append(x); srcs.append(src+offset); dsts.append(dst+offset); efs.append(ef)
        batch.append(np.full(x.shape[0], k, dtype=np.int64))
        offset += x.shape[0]
    return (torch.from_numpy(np.concatenate(xs)),
            torch.from_numpy(np.concatenate(srcs)),
            torch.from_numpy(np.concatenate(dsts)),
            torch.from_numpy(np.concatenate(efs)),
            torch.from_numpy(np.concatenate(batch)))

BATCH = 256
N = len(all_smi)
mus = np.zeros((N, Z_DIM), dtype=np.float32)
with torch.no_grad():
    for s in range(0, N, BATCH):
        bi = list(range(s, min(s+BATCH, N)))
        x, src, dst, ef, batch = collate(bi)
        mu, _ = vae.enc(x, src, dst, ef, batch)
        mus[s:s+len(bi)] = mu.numpy()

# rebuild kind
acid_set, epox_set = set(acids), set(epoxes)
def kind_of(s):
    if s in acid_set and s in epox_set: return "both"
    return "acid" if s in acid_set else "epoxide"
kinds = np.array([kind_of(s) for s in all_smi])

np.savez(os.path.join(ROOT, "outputs/vae_latents_all.npz"),
         z=mus, smiles=np.array(all_smi), kind=kinds)
print("Saved", N, "monomer latents")
