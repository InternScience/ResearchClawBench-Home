"""
Phase 6: Pair-level Tg predictor + inverse design of vitrimer (acid, epoxide)
pairs.

We concatenate the encoder mean of an acid latent and an epoxide latent and
train a small MLP to predict the GP-calibrated pair Tg. Inverse design then
optimises (z_acid, z_epoxide) jointly to hit a target Tg, decodes them back
to SMILES, validates with RDKit, checks novelty, and re-scores with the
predictor.

Outputs:
  - outputs/pair_tg_predictor.pt
  - outputs/pair_predictor_metrics.json
  - outputs/designed_candidates.csv          (raw candidates)
  - outputs/designed_candidates_top.csv      (top-10 novel valid per target)
  - report/images/fig_latent_property_pred.png
  - report/images/fig_inverse_design.png
"""
import os, json, math, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(2); np.random.seed(2); random.seed(2)
DEV = torch.device("cpu")

# -------------------------------------------------------------------------
# 1. Load all-monomer latents and pair-level Tg labels
# -------------------------------------------------------------------------
lat = np.load(os.path.join(ROOT, "outputs/vae_latents_all.npz"), allow_pickle=True)
Z_ALL = lat["z"]
SMI_ALL = lat["smiles"]
smi2idx = {s: i for i, s in enumerate(SMI_ALL.tolist())}

vit = pd.read_csv(os.path.join(ROOT, "outputs/vitrimer_calibrated_tg.csv"))
def canon(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m, canonical=True) if m else None
vit["acid_c"]    = vit["acid"].map(canon)
vit["epoxide_c"] = vit["epoxide"].map(canon)

idx_a = vit["acid_c"].map(smi2idx).values
idx_e = vit["epoxide_c"].map(smi2idx).values
mask = pd.notna(idx_a) & pd.notna(idx_e)
print(f"pairs with both monomers encoded: {int(mask.sum())}/{len(vit)}")
vit = vit[mask].reset_index(drop=True)
idx_a = idx_a[mask].astype(int); idx_e = idx_e[mask].astype(int)

Z_a = Z_ALL[idx_a]; Z_e = Z_ALL[idx_e]
X = np.concatenate([Z_a, Z_e], axis=1).astype(np.float32)
y = vit["tg_calibrated"].values.astype(np.float32)
print("Pair design matrix:", X.shape, "Tg range:", float(y.min()), float(y.max()))

# -------------------------------------------------------------------------
# 2. Train predictor with held-out test set
# -------------------------------------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=0)

class MLP(nn.Module):
    def __init__(self, d, h=256, h2=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h, h2), nn.ReLU(),
                                 nn.Linear(h2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

mlp = MLP(X.shape[1]).to(DEV)
opt = torch.optim.Adam(mlp.parameters(), lr=2e-3, weight_decay=1e-5)
XT = torch.from_numpy(Xtr).to(DEV); yT = torch.from_numpy(ytr).to(DEV)
XV = torch.from_numpy(Xte).to(DEV); yV = torch.from_numpy(yte).to(DEV)

best_state, best_rmse = None, 1e9
for epoch in range(800):
    mlp.train()
    perm = torch.randperm(len(XT))
    for s in range(0, len(XT), 512):
        bi = perm[s:s+512]
        pred = mlp(XT[bi])
        loss = F.mse_loss(pred, yT[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 25 == 0 or epoch == 799:
        mlp.eval()
        with torch.no_grad():
            yp = mlp(XV).cpu().numpy()
        rmse = math.sqrt(mean_squared_error(yte, yp))
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}

mlp.load_state_dict(best_state)
mlp.eval()
with torch.no_grad():
    yp_te = mlp(XV).cpu().numpy()
    yp_tr = mlp(XT).cpu().numpy()

metrics = dict(train_n=int(len(Xtr)), test_n=int(len(Xte)),
               test_r2=float(r2_score(yte, yp_te)),
               test_rmse=float(math.sqrt(mean_squared_error(yte, yp_te))),
               test_mae=float(mean_absolute_error(yte, yp_te)),
               train_r2=float(r2_score(ytr, yp_tr)))
print("Pair Tg predictor:", metrics)
torch.save(mlp.state_dict(), os.path.join(ROOT, "outputs/pair_tg_predictor.pt"))
json.dump(metrics, open(os.path.join(ROOT, "outputs/pair_predictor_metrics.json"), "w"), indent=2)

# Figure: parity & latent PCA --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
ax.scatter(yte, yp_te, s=8, alpha=0.4, color="#1f77b4")
lo, hi = float(min(yte.min(), yp_te.min())), float(max(yte.max(), yp_te.max()))
ax.plot([lo, hi], [lo, hi], 'k--')
ax.set_xlabel("GP-calibrated pair Tg [K]")
ax.set_ylabel("MLP prediction from latents [K]")
ax.set_title(f"Pair-level Tg prediction\n(test R²={metrics['test_r2']:.3f}, "
             f"MAE={metrics['test_mae']:.1f} K)")

ax = axes[1]
P = PCA(n_components=2).fit_transform(X[:5000])
sc = ax.scatter(P[:,0], P[:,1], c=y[:5000], s=4, cmap="viridis",
                vmin=np.percentile(y, 2), vmax=np.percentile(y, 98))
plt.colorbar(sc, ax=ax, label="calibrated Tg [K]")
ax.set_title("Pair latent space (PCA-2D), coloured by Tg")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "report/images/fig_latent_property_pred.png"), dpi=150)
plt.close(fig)

# -------------------------------------------------------------------------
# 3. Reload VAE decoder
# -------------------------------------------------------------------------
ckpt = torch.load(os.path.join(ROOT, "outputs/vae_state.pt"),
                  map_location=DEV, weights_only=False)
vocab = ckpt["vocab"]; V = len(vocab); MAX_LEN = ckpt["max_len"]
ATOM_FDIM = ckpt["atom_fdim"]; BOND_FDIM = ckpt["bond_fdim"]; Z_DIM = ckpt["z_dim"]

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
    @torch.no_grad()
    def sample(self, z, max_len, greedy=True, temp=1.0):
        B = z.shape[0]
        h = torch.tanh(self.z2h(z)).unsqueeze(0).expand(self.n_layers, B, self.hidden).contiguous()
        cur = torch.full((B, 1), 1, dtype=torch.long)
        seqs = [cur]; finished = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_len-1):
            e = self.emb(cur)
            x = torch.cat([e, z.unsqueeze(1)], dim=-1)
            o, h = self.gru(x, h)
            logits = self.out(o[:, -1]) / temp
            tok = logits.argmax(-1, keepdim=True) if greedy \
                  else torch.multinomial(F.softmax(logits, -1), 1)
            seqs.append(tok)
            finished |= tok.squeeze(-1).eq(2)
            cur = tok
            if finished.all(): break
        return torch.cat(seqs, dim=1)

class GraphVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = GraphEncoder(ATOM_FDIM, BOND_FDIM, z_dim=Z_DIM)
        self.dec = GRUDecoder(V, Z_DIM)

vae = GraphVAE().to(DEV)
vae.load_state_dict(ckpt["state_dict"])
vae.eval()

def decode(z_batch, greedy=True, temp=1.0):
    seqs = vae.dec.sample(z_batch, MAX_LEN, greedy=greedy, temp=temp)
    out = []
    for row in seqs.tolist():
        toks = []
        for t in row[1:]:
            if t == 2: break
            if t == 0: continue
            toks.append(vocab[t])
        out.append("".join(toks))
    return out

train_smi = set(SMI_ALL.tolist())
train_pairs = set(zip(vit["acid_c"], vit["epoxide_c"]))

def smiles_canon_or_none(s):
    if not s: return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None

# featurization helpers (must match 03_train_graph_vae.py)
ATOM_LIST = [6, 7, 8, 9, 16, 17, 35, 53, 14, 15]
HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
       Chem.rdchem.HybridizationType.SP3D2]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
def atom_features(a):
    z = a.GetAtomicNum()
    f = [int(z == zz) for zz in ATOM_LIST]; f.append(int(z not in ATOM_LIST))
    f.append(a.GetDegree()/6.0); f.append(a.GetFormalCharge()/2.0)
    f.append(int(a.GetIsAromatic())); f.append(int(a.IsInRing()))
    f.extend([int(a.GetHybridization()==h) for h in HYB])
    f.append(a.GetTotalNumHs()/4.0); return f
def bond_features(b):
    f = [int(b.GetBondType()==bt) for bt in BOND_TYPES]
    f.append(int(b.IsInRing())); f.append(int(b.GetIsConjugated())); return f
def encode_new(smi):
    if smi in smi2idx: return Z_ALL[smi2idx[smi]]
    m = Chem.MolFromSmiles(smi)
    if m is None: return np.zeros(Z_DIM, dtype=np.float32)
    n = m.GetNumAtoms()
    x = np.zeros((n, ATOM_FDIM), dtype=np.float32)
    for i, a in enumerate(m.GetAtoms()): x[i] = atom_features(a)
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bond_features(b)
        src += [i,j]; dst += [j,i]; ef += [f,f]
    if not src: src,dst,ef = [0],[0],[[0]*BOND_FDIM]
    with torch.no_grad():
        mu, _ = vae.enc(torch.from_numpy(x),
                        torch.tensor(src), torch.tensor(dst),
                        torch.tensor(np.asarray(ef, dtype=np.float32)),
                        torch.zeros(n, dtype=torch.long))
    return mu.numpy()[0]
def encode_pair(ca, ce):
    return np.concatenate([encode_new(ca), encode_new(ce)], axis=-1)

# -------------------------------------------------------------------------
# 4. Latent-space inverse design
# -------------------------------------------------------------------------
TARGETS = [350.0, 400.0, 450.0]
N_PER_TARGET = 200
N_OPT = 250
LR = 0.05

def anchors_for(target, n):
    j = np.argsort(np.abs(y - target))[:n]
    return X[j].copy()

def latent_optim(target, n=N_PER_TARGET, jitter=0.6):
    init = anchors_for(target, n) + jitter * np.random.randn(n, X.shape[1]).astype(np.float32)
    z = torch.tensor(init, requires_grad=True, dtype=torch.float32)
    optz = torch.optim.Adam([z], lr=LR)
    for _ in range(N_OPT):
        pred = mlp(z)
        loss = ((pred - target)**2).mean() + 1e-3 * (z**2).sum(-1).mean()
        optz.zero_grad(); loss.backward(); optz.step()
    with torch.no_grad():
        pred = mlp(z).cpu().numpy()
    return z.detach(), pred

def split_pair(z_pair):
    return z_pair[:, :Z_DIM], z_pair[:, Z_DIM:]

records = []
for tgt in TARGETS:
    z_pair, pred = latent_optim(tgt)
    za, ze = split_pair(z_pair)
    smi_a = decode(za, greedy=True)
    smi_e = decode(ze, greedy=True)
    smi_a_s = decode(za, greedy=False, temp=0.8)
    smi_e_s = decode(ze, greedy=False, temp=0.8)
    for k in range(z_pair.shape[0]):
        for tag, sa, se in [("greedy", smi_a[k], smi_e[k]),
                            ("sample", smi_a_s[k], smi_e_s[k])]:
            ca = smiles_canon_or_none(sa); ce = smiles_canon_or_none(se)
            valid = (ca is not None) and (ce is not None)
            novel_pair = valid and ((ca, ce) not in train_pairs)
            novel_mono = valid and ((ca not in train_smi) or (ce not in train_smi))
            records.append(dict(
                target_Tg=float(tgt), decoder=tag, anchor_idx=k,
                acid_decoded=sa, epoxide_decoded=se,
                acid_canonical=ca, epoxide_canonical=ce,
                valid=valid, novel_pair=novel_pair,
                novel_monomer=novel_mono,
                pred_tg=float(pred[k]),
            ))

cand = pd.DataFrame(records)
cand.to_csv(os.path.join(ROOT, "outputs/designed_candidates.csv"), index=False)

print("Validity per (target, decoder):")
print(cand.groupby(["target_Tg","decoder"])["valid"].mean())
print("\nNovel pair counts:")
print(cand.groupby(["target_Tg","decoder"])["novel_pair"].sum())
print("\nNovel monomer counts:")
print(cand.groupby(["target_Tg","decoder"])["novel_monomer"].sum())

# Re-encode all valid candidate pairs and re-score
valid = cand[cand.valid].copy().reset_index(drop=True)
new_X = np.stack([encode_pair(r.acid_canonical, r.epoxide_canonical) for _, r in valid.iterrows()])
with torch.no_grad():
    valid["pred_tg_reencoded"] = mlp(torch.from_numpy(new_X.astype(np.float32))).cpu().numpy()
valid["abs_err_reencoded"] = (valid.pred_tg_reencoded - valid.target_Tg).abs()

top_novel = valid[valid.novel_pair].copy()
top_novel = top_novel.sort_values(["target_Tg","abs_err_reencoded"]).groupby("target_Tg").head(10)
top_novel.to_csv(os.path.join(ROOT, "outputs/designed_candidates_top.csv"), index=False)
print("\nTop novel candidates (re-encoded re-scored):")
print(top_novel[["target_Tg","pred_tg_reencoded","acid_canonical","epoxide_canonical"]]
      .to_string(max_colwidth=70))

# -------------------------------------------------------------------------
# 5. Inverse-design figure
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
for tgt, c in zip(TARGETS, ["#1f77b4","#2ca02c","#d62728"]):
    sub = valid[valid.target_Tg == tgt]
    ax.hist(sub.pred_tg_reencoded, bins=25, alpha=0.55, color=c, label=f"target {tgt:.0f} K")
    ax.axvline(tgt, color=c, ls="--", lw=1.2)
ax.set_xlabel("Re-encoded MLP-predicted Tg [K] (valid candidates)")
ax.set_ylabel("count")
ax.set_title("Inverse design: predicted Tg of decoded molecules")
ax.legend()

ax = axes[1]
val_rate  = cand.groupby("target_Tg")["valid"].mean()
nov_pair  = cand.groupby("target_Tg")["novel_pair"].mean()
nov_mono  = cand.groupby("target_Tg")["novel_monomer"].mean()
xs = np.arange(len(TARGETS))
w = 0.27
ax.bar(xs - w, val_rate.values, w, label="valid SMILES pair", color="#1f77b4")
ax.bar(xs    , nov_pair.values, w, label="valid & novel pair", color="#ff7f0e")
ax.bar(xs + w, nov_mono.values, w, label="valid w/ ≥1 novel monomer", color="#2ca02c")
ax.set_xticks(xs); ax.set_xticklabels([f"{t:.0f} K" for t in TARGETS])
ax.set_ylabel("fraction"); ax.set_ylim(0, 1.05)
ax.set_title("Validity & novelty"); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "report/images/fig_inverse_design.png"), dpi=150)
plt.close(fig)
print("Done.")
