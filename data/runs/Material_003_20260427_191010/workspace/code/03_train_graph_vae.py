"""
Phase 5: Graph variational autoencoder for vitrimer monomers
============================================================

Architecture (CPU-friendly, no torch_geometric):
  * Encoder: a small message-passing graph net (GIN-style) operating on RDKit
    atom-level graphs. Atom features: one-hot atomic number (subset),
    degree, formal charge, aromatic flag, in-ring flag, hybridization.
    Bond features used as messages: single/double/triple/aromatic + in-ring.
    K message-passing rounds with sum aggregation, followed by a sum readout.
    Two heads produce the variational mean and log-variance of a 64-dim
    Gaussian latent.
  * Decoder: GRU SMILES decoder conditioned on z; teacher-forced cross-entropy
    over a small character vocabulary built from the training SMILES.
  * Loss: reconstruction (token-wise CE) + KL with a beta schedule.

This intentionally keeps the encoder over molecular *graphs* (the named "graph
VAE" requirement) while using a SMILES-string decoder for tractability — a
common strategy in chemistry VAEs (Gomez-Bombarelli ChemVAE, Jin JT-VAE
ablations).

Trains on the union of unique acid + epoxide SMILES (about 1.2-1.5e4 unique
monomers).  Saves model and a sample of latent vectors for downstream
inverse-design steps.

Outputs:
  - outputs/vae_state.pt        : model weights + vocabulary + config
  - outputs/vae_train_log.csv   : per-epoch losses
  - outputs/vae_latents.npz     : latent vectors of all training molecules
  - report/images/fig_vae_training.png
"""
from __future__ import annotations
import os, json, math, random, time, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0); np.random.seed(0); random.seed(0)
DEV = torch.device("cpu")

# -------------------------------------------------------------------------
# 1. Build molecule corpus
# -------------------------------------------------------------------------
vit = pd.read_csv(os.path.join(ROOT, "outputs/vitrimer_calibrated_tg.csv"))
acids   = vit["acid"].astype(str).tolist()
epoxes  = vit["epoxide"].astype(str).tolist()
tg_vals = vit["tg_calibrated"].values.astype(np.float32)

mols = pd.DataFrame({"smiles": acids + epoxes,
                     "kind":   ["acid"]*len(acids) + ["epoxide"]*len(epoxes),
                     "tg":     np.concatenate([tg_vals, tg_vals])})
mols = mols.drop_duplicates("smiles").reset_index(drop=True)
# Subsample to keep CPU training tractable; stratify by kind so both acid and
# epoxide chemistries are represented in the VAE corpus.
_n = min(8000, len(mols))
mols = (mols.groupby("kind", group_keys=False)
            .apply(lambda d: d.sample(n=int(round(_n * len(d) / len(mols))),
                                      random_state=0))
            .reset_index(drop=True))

print(f"Unique monomers: {len(mols)} "
      f"(acids unique: {mols.kind.eq('acid').sum()}, "
      f"epoxides unique: {mols.kind.eq('epoxide').sum()})")

# Canonicalize SMILES
def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None
mols["smiles"] = mols["smiles"].map(canon)
mols = mols.dropna(subset=["smiles"]).reset_index(drop=True)
print("After canonicalisation:", len(mols))

# -------------------------------------------------------------------------
# 2. Vocabulary for SMILES decoder (BPE-free char-level with multichar
#     handling for two-letter atoms and bracketed atoms)
# -------------------------------------------------------------------------
import re
TOKEN_RE = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Se|@@?|%\d{2}|.)")
def tokenize(smi):
    return TOKEN_RE.findall(smi)

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
counts = {}
for s in mols["smiles"]:
    for t in tokenize(s):
        counts[t] = counts.get(t, 0) + 1
vocab = [PAD, BOS, EOS] + sorted(counts.keys())
stoi = {t: i for i, t in enumerate(vocab)}
itos = vocab
V = len(vocab)
print("Vocab size:", V)

MAX_LEN = max(len(tokenize(s)) for s in mols["smiles"]) + 2
print("Max SMILES tokens (incl bos/eos):", MAX_LEN)

# Pre-encode SMILES targets
def encode_smiles(s):
    toks = [BOS] + tokenize(s) + [EOS]
    return [stoi[t] for t in toks]
smi_tokens = [encode_smiles(s) for s in mols["smiles"]]
smi_lens   = [len(t) for t in smi_tokens]

# -------------------------------------------------------------------------
# 3. RDKit molecule -> graph tensor
# -------------------------------------------------------------------------
ATOM_LIST = [6, 7, 8, 9, 16, 17, 35, 53, 14, 15]   # C, N, O, F, S, Cl, Br, I, Si, P
HYB = [Chem.rdchem.HybridizationType.SP,
       Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3,
       Chem.rdchem.HybridizationType.SP3D,
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

ATOM_FDIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]

def bond_features(bond):
    feat = [int(bond.GetBondType() == bt) for bt in BOND_TYPES]
    feat.append(int(bond.IsInRing()))
    feat.append(int(bond.GetIsConjugated()))
    return feat
BOND_FDIM = 4 + 2

def mol_to_graph(smi):
    m = Chem.MolFromSmiles(smi)
    n = m.GetNumAtoms()
    x = np.zeros((n, ATOM_FDIM), dtype=np.float32)
    for i, a in enumerate(m.GetAtoms()):
        x[i] = atom_features(a)
    # build edge index w/ both directions
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]; dst += [j, i]; ef += [f, f]
    if len(src) == 0:
        src, dst, ef = [0], [0], [[0]*BOND_FDIM]
    return (x,
            np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
            np.asarray(ef, dtype=np.float32))

print("Pre-computing molecular graphs ...")
graphs = [mol_to_graph(s) for s in mols["smiles"]]

# -------------------------------------------------------------------------
# 4. Batching: concatenate graphs into one big graph w/ batch index
# -------------------------------------------------------------------------
def collate(batch_idx):
    xs, srcs, dsts, efs, batch, tok = [], [], [], [], [], []
    offset = 0
    for k, i in enumerate(batch_idx):
        x, s, d, e = graphs[i]
        xs.append(x); srcs.append(s + offset); dsts.append(d + offset); efs.append(e)
        batch.append(np.full(x.shape[0], k, dtype=np.int64))
        offset += x.shape[0]
        tok.append(smi_tokens[i])
    x = torch.from_numpy(np.concatenate(xs))
    src = torch.from_numpy(np.concatenate(srcs))
    dst = torch.from_numpy(np.concatenate(dsts))
    ef  = torch.from_numpy(np.concatenate(efs))
    batch = torch.from_numpy(np.concatenate(batch))
    # pad SMILES tokens
    L = max(len(t) for t in tok)
    tk = torch.zeros(len(tok), L, dtype=torch.long)
    lens = torch.zeros(len(tok), dtype=torch.long)
    for k, t in enumerate(tok):
        tk[k, :len(t)] = torch.tensor(t, dtype=torch.long)
        lens[k] = len(t)
    return x, src, dst, ef, batch, tk, lens

# -------------------------------------------------------------------------
# 5. Model
# -------------------------------------------------------------------------
class GINLayer(nn.Module):
    def __init__(self, d, edge_dim):
        super().__init__()
        self.edge_lin = nn.Linear(edge_dim, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.eps = nn.Parameter(torch.tensor(0.0))
    def forward(self, h, src, dst, ef):
        m = h[src] + self.edge_lin(ef)
        # sum-aggregate to dst
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        return self.mlp((1 + self.eps) * h + agg)

class GraphEncoder(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden=128, n_layers=4, z_dim=64):
        super().__init__()
        self.in_lin = nn.Linear(atom_dim, hidden)
        self.layers = nn.ModuleList(GINLayer(hidden, edge_dim) for _ in range(n_layers))
        self.read = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_lv = nn.Linear(hidden, z_dim)
    def forward(self, x, src, dst, ef, batch):
        h = F.relu(self.in_lin(x))
        for l in self.layers:
            h = h + F.relu(l(h, src, dst, ef))   # residual
        # sum readout per molecule
        n_g = int(batch.max().item()) + 1
        g = torch.zeros(n_g, h.shape[1], device=h.device)
        g.index_add_(0, batch, h)
        g = self.read(g)
        return self.fc_mu(g), self.fc_lv(g)

class GRUDecoder(nn.Module):
    def __init__(self, V, z_dim, emb=64, hidden=256, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.z2h = nn.Linear(z_dim, hidden)
        self.gru = nn.GRU(emb + z_dim, hidden, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden, V)
        self.n_layers = n_layers
        self.hidden = hidden
        self.z_dim = z_dim
    def forward(self, z, tk):
        # teacher forcing
        B, L = tk.shape
        e = self.emb(tk[:, :-1])               # (B, L-1, emb)
        zt = z.unsqueeze(1).expand(-1, L-1, -1)
        x = torch.cat([e, zt], dim=-1)
        h0 = torch.tanh(self.z2h(z)).unsqueeze(0).expand(self.n_layers, B, self.hidden).contiguous()
        out, _ = self.gru(x, h0)
        return self.out(out)
    @torch.no_grad()
    def sample(self, z, max_len, greedy=True):
        B = z.shape[0]
        h = torch.tanh(self.z2h(z)).unsqueeze(0).expand(self.n_layers, B, self.hidden).contiguous()
        cur = torch.full((B, 1), 1, dtype=torch.long, device=z.device)  # BOS=1
        seqs = [cur]
        finished = torch.zeros(B, dtype=torch.bool, device=z.device)
        for _ in range(max_len-1):
            e = self.emb(cur)
            x = torch.cat([e, z.unsqueeze(1)], dim=-1)
            o, h = self.gru(x, h)
            logits = self.out(o[:, -1])
            if greedy:
                tok = logits.argmax(-1, keepdim=True)
            else:
                tok = torch.multinomial(F.softmax(logits, -1), 1)
            seqs.append(tok)
            finished |= tok.squeeze(-1).eq(2)   # EOS=2
            cur = tok
            if finished.all(): break
        return torch.cat(seqs, dim=1)

class GraphVAE(nn.Module):
    def __init__(self, atom_dim, edge_dim, V, z_dim=64):
        super().__init__()
        self.enc = GraphEncoder(atom_dim, edge_dim, hidden=64, n_layers=3, z_dim=z_dim)
        self.dec = GRUDecoder(V, z_dim, emb=32, hidden=128, n_layers=1)
        self.z_dim = z_dim
    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()
    def forward(self, x, src, dst, ef, batch, tk):
        mu, lv = self.enc(x, src, dst, ef, batch)
        z = self.reparam(mu, lv)
        logits = self.dec(z, tk)
        return logits, mu, lv

# -------------------------------------------------------------------------
# 6. Training loop
# -------------------------------------------------------------------------
N = len(mols)
idx_all = np.arange(N); rng = np.random.RandomState(0); rng.shuffle(idx_all)
n_val = max(200, N // 20)
val_idx, train_idx = idx_all[:n_val], idx_all[n_val:]

model = GraphVAE(ATOM_FDIM, BOND_FDIM, V, z_dim=64).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
crit = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

EPOCHS = 30
BATCH = 256
def beta_for(ep):
    return min(0.05, 0.005 * ep)   # gentle KL warm-up

logs = []
t0 = time.time()
for ep in range(1, EPOCHS+1):
    model.train()
    rng.shuffle(train_idx)
    rec_loss_tot, kl_tot, ntoks = 0.0, 0.0, 0
    for s in range(0, len(train_idx), BATCH):
        bi = train_idx[s:s+BATCH]
        x, src, dst, ef, batch, tk, lens = collate(bi)
        x, src, dst, ef, batch, tk = (t.to(DEV) for t in (x, src, dst, ef, batch, tk))
        logits, mu, lv = model(x, src, dst, ef, batch, tk)
        # CE on positions 1..L (predict tk[:,1:])
        rec = crit(logits.reshape(-1, V), tk[:, 1:].reshape(-1))
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum()
        beta = beta_for(ep)
        loss = (rec + beta * kl) / x.shape[0]   # avg per atom-graph
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        n_t = (tk[:, 1:] != 0).sum().item()
        rec_loss_tot += rec.item(); kl_tot += kl.item(); ntoks += n_t
    # val
    model.eval()
    with torch.no_grad():
        vrec, vkl, vtok, vcorr = 0.0, 0.0, 0, 0
        for s in range(0, len(val_idx), BATCH):
            bi = val_idx[s:s+BATCH]
            x, src, dst, ef, batch, tk, lens = collate(bi)
            x, src, dst, ef, batch, tk = (t.to(DEV) for t in (x, src, dst, ef, batch, tk))
            logits, mu, lv = model(x, src, dst, ef, batch, tk)
            vrec += crit(logits.reshape(-1, V), tk[:, 1:].reshape(-1)).item()
            vkl  += -0.5*(1+lv-mu.pow(2)-lv.exp()).sum().item()
            mask = tk[:, 1:] != 0
            pred = logits.argmax(-1)
            vcorr += ((pred == tk[:, 1:]) & mask).sum().item()
            vtok  += mask.sum().item()
    log = dict(epoch=ep,
               train_rec_per_tok=rec_loss_tot/ntoks,
               train_kl=kl_tot/len(train_idx),
               val_rec_per_tok=vrec/vtok,
               val_token_acc=vcorr/vtok,
               beta=beta_for(ep),
               minutes=(time.time()-t0)/60)
    logs.append(log)
    print(json.dumps(log))

pd.DataFrame(logs).to_csv(os.path.join(ROOT, "outputs/vae_train_log.csv"), index=False)

# -------------------------------------------------------------------------
# 7. Encode all molecules and persist artifacts
# -------------------------------------------------------------------------
model.eval()
mus = np.zeros((N, model.z_dim), dtype=np.float32)
with torch.no_grad():
    for s in range(0, N, BATCH):
        bi = np.arange(s, min(s+BATCH, N))
        x, src, dst, ef, batch, tk, lens = collate(bi)
        x, src, dst, ef, batch = (t.to(DEV) for t in (x, src, dst, ef, batch))
        mu, _ = model.enc(x, src, dst, ef, batch)
        mus[s:s+len(bi)] = mu.cpu().numpy()

np.savez(os.path.join(ROOT, "outputs/vae_latents.npz"),
         z=mus, smiles=mols["smiles"].values, kind=mols["kind"].values, tg=mols["tg"].values)

torch.save({"state_dict": model.state_dict(),
            "vocab": vocab, "atom_fdim": ATOM_FDIM, "bond_fdim": BOND_FDIM,
            "max_len": int(MAX_LEN), "z_dim": int(model.z_dim)},
           os.path.join(ROOT, "outputs/vae_state.pt"))

# -------------------------------------------------------------------------
# 8. Training figure
# -------------------------------------------------------------------------
df = pd.DataFrame(logs)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(df.epoch, df.train_rec_per_tok, '-o', label="train rec/tok")
axes[0].plot(df.epoch, df.val_rec_per_tok,   '-o', label="val rec/tok")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("CE per token (nats)"); axes[0].legend()
axes[0].set_title("Reconstruction CE (lower is better)")
axes[1].plot(df.epoch, df.val_token_acc, '-o', color='#2ca02c')
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("token-level accuracy")
axes[1].set_title("Validation token accuracy")
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "report/images/fig_vae_training.png"), dpi=150)
plt.close(fig)
print("Wrote VAE artifacts and training figure.")
