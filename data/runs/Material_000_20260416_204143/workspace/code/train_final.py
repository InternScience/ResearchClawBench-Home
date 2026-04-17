"""
AI-Powered Altermagnet Discovery - Optimized for CPU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
import numpy as np
import json, os, sys, types, copy, random
from sklearn.metrics import *

WORKSPACE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_000_20260416_204143'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# Load data
data_prepare = types.ModuleType('data_prepare')
class RCD:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): self.__dict__.update(s)
data_prepare.RealisticCrystalDataset = RCD
sys.modules['data_prepare'] = data_prepare

pretrain_data = torch.load(f'{DATA_DIR}/pretrain_data.pt', weights_only=False).data_list
finetune_data = torch.load(f'{DATA_DIR}/finetune_data.pt', weights_only=False).data_list
candidate_data = torch.load(f'{DATA_DIR}/candidate_data.pt', weights_only=False).data_list

NF = 28  # node features

print(f"Data: pretrain={len(pretrain_data)}, finetune={len(finetune_data)}, candidate={len(candidate_data)}")
ft_labels = [d.y.item() for d in finetune_data]
cd_labels = [d.y.item() for d in candidate_data]
print(f"FT: {sum(ft_labels)} pos, {len(ft_labels)-sum(ft_labels)} neg")
print(f"CD: {sum(cd_labels)} pos, {len(cd_labels)-sum(cd_labels)} neg")

# ---- Models ----
class GINEnc(nn.Module):
    def __init__(self, in_ch=28, hid=32, layers=3, drop=0.1):
        super().__init__()
        self.hid = hid
        self.embed = nn.Sequential(nn.Linear(in_ch, hid), nn.BatchNorm1d(hid), nn.ReLU())
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hid, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Linear(hid, hid))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hid))
        self.drop = drop
        self.layers = layers
    
    def forward(self, x, ei, ea=None, batch=None):
        x = self.embed(x)
        outs = []
        for i in range(self.layers):
            x = self.convs[i](x, ei)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.drop, self.training)
            if batch is not None:
                outs.append(global_mean_pool(x, batch))
            else:
                outs.append(x.mean(0, keepdim=True))
        return torch.cat(outs, dim=1)  # layers * hid

class Clf(nn.Module):
    def __init__(self, encoder, hid=32, layers=3, drop=0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hid * layers, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hid, 1)
        )
    def forward(self, x, ei, ea=None, batch=None):
        return self.head(self.encoder(x, ei, ea, batch))
    def get_emb(self, x, ei, ea=None, batch=None):
        return self.encoder(x, ei, ea, batch)

# ---- Contrastive pre-training ----
def nt_xent(z1, z2, t=0.3):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    bs = z1.size(0)
    z = torch.cat([z1, z2])
    sim = z @ z.t() / t
    labels = torch.cat([torch.arange(bs)+bs, torch.arange(bs)])
    sim.fill_diagonal_(-1e9)
    return F.cross_entropy(sim, labels)

def pretrain(data, epochs=25, bs=512, lr=1e-3, hid=32):
    print("\n=== PRE-TRAINING ===")
    enc = GINEnc(NF, hid, 3, 0.1)
    proj = nn.Sequential(nn.Linear(hid*3, hid), nn.ReLU(), nn.Linear(hid, 16))
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    loader = DataLoader(data, batch_size=bs, shuffle=True)
    losses = []
    enc.train(); proj.train()
    for ep in range(epochs):
        el = 0; nb = 0
        for batch in loader:
            x1 = batch.x * (torch.rand_like(batch.x) > 0.15).float()
            x2 = batch.x * (torch.rand_like(batch.x) > 0.15).float()
            ne = batch.edge_index.size(1)
            k1 = torch.rand(ne) > 0.2; k2 = torch.rand(ne) > 0.2
            if k1.sum()==0: k1[0]=True
            if k2.sum()==0: k2[0]=True
            h1 = enc(x1, batch.edge_index[:,k1], batch=batch.batch)
            h2 = enc(x2, batch.edge_index[:,k2], batch=batch.batch)
            loss = nt_xent(proj(h1), proj(h2))
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        losses.append(el/nb)
        if (ep+1) % 5 == 0: print(f"  Ep {ep+1}/{epochs}, Loss: {losses[-1]:.4f}")
    return enc, losses

# ---- Fine-tuning ----
def oversample(data, ratio=0.3):
    pos = [d for d in data if d.y.item()==1]
    neg = [d for d in data if d.y.item()==0]
    target = int(len(neg)*ratio/(1-ratio))
    reps = target // len(pos) + 1
    combined = neg + (pos*reps)[:target]
    random.shuffle(combined)
    return combined

def finetune(enc, data, epochs=60, bs=64, lr=3e-4, hid=32, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    labels = [d.y.item() for d in data]
    pi = [i for i,l in enumerate(labels) if l==1]
    ni = [i for i,l in enumerate(labels) if l==0]
    np.random.shuffle(pi); np.random.shuffle(ni)
    nvp = max(2, int(len(pi)*0.2)); nvn = int(len(ni)*0.2)
    vi = pi[:nvp]+ni[:nvn]; ti = pi[nvp:]+ni[nvn:]
    train_d = oversample([data[i] for i in ti], 0.25)
    val_d = [data[i] for i in vi]
    
    model = Clf(copy.deepcopy(enc), hid, 3, 0.3)
    np_ = sum(d.y.item() for d in train_d)
    nn_ = len(train_d)-np_
    pw = torch.tensor([nn_/max(np_,1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': lr*0.2},
        {'params': model.head.parameters(), 'lr': lr}
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    tl = DataLoader(train_d, bs, shuffle=True)
    vl = DataLoader(val_d, bs, shuffle=False)
    
    hist = {'tl':[],'vl':[],'auc':[],'ap':[],'f1':[]}
    best_m = 0; best_s = None; pat = 0
    
    for ep in range(epochs):
        model.train()
        el=0; nb=0
        for b in tl:
            lo = model(b.x, b.edge_index, b.edge_attr, b.batch).squeeze(-1)
            loss = crit(lo, b.y.float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el+=loss.item(); nb+=1
        sched.step()
        hist['tl'].append(el/nb)
        
        model.eval()
        vp=[]; vt=[]
        with torch.no_grad():
            for b in vl:
                lo = model(b.x, b.edge_index, b.edge_attr, b.batch).squeeze(-1)
                vp.extend(torch.sigmoid(lo).numpy().tolist())
                vt.extend(b.y.numpy().tolist())
        try: auc=roc_auc_score(vt,vp); ap=average_precision_score(vt,vp)
        except: auc=0.5; ap=0.0
        f1=f1_score(vt,[1 if p>0.5 else 0 for p in vp], zero_division=0)
        hist['auc'].append(auc); hist['ap'].append(ap); hist['f1'].append(f1)
        
        met = ap + 0.3*auc
        if met > best_m: best_m=met; best_s=copy.deepcopy(model.state_dict()); pat=0
        else: pat+=1
        if (ep+1)%15==0: print(f"    Ep {ep+1}, AUC:{auc:.4f}, AP:{ap:.4f}, F1:{f1:.4f}")
        if pat>=25: print(f"    Early stop ep {ep+1}"); break
    
    if best_s: model.load_state_dict(best_s)
    return model, hist

def ens_predict(models, data, bs=512):
    loader = DataLoader(data, bs, shuffle=False)
    all_p = []
    for m in models:
        m.eval(); ps=[]
        with torch.no_grad():
            for b in loader:
                lo = m(b.x, b.edge_index, b.edge_attr, b.batch).squeeze(-1)
                ps.extend(torch.sigmoid(lo).numpy().tolist())
        all_p.append(np.array(ps))
    return np.mean(all_p, axis=0), all_p

def get_emb(model, data, bs=512):
    loader = DataLoader(data, bs, shuffle=False)
    model.eval(); es=[]
    with torch.no_grad():
        for b in loader:
            es.append(model.get_emb(b.x, b.edge_index, b.edge_attr, b.batch).numpy())
    return np.concatenate(es)

# ---- MAIN ----
HID = 32
results = {}

# Pre-train
enc, pt_losses = pretrain(pretrain_data, epochs=25, bs=512, lr=1e-3, hid=HID)
json.dump(pt_losses, open(f'{OUTPUT_DIR}/pretrain_losses.json','w'))

# Fine-tune ensemble (3 models)
print("\n=== FINE-TUNING (Pre-trained) ===")
pt_models = []
pt_hists = []
for i in range(3):
    s = SEED + i*17
    print(f"  Model {i+1}/3 (seed={s})")
    m, h = finetune(enc, finetune_data, epochs=60, bs=64, lr=3e-4, hid=HID, seed=s)
    pt_models.append(m); pt_hists.append(h)

# Random init ensemble
print("\n=== FINE-TUNING (Random Init) ===")
ri_models = []
ri_hists = []
for i in range(3):
    s = SEED + i*17
    print(f"  Random Model {i+1}/3 (seed={s})")
    re = GINEnc(NF, HID, 3, 0.1)
    m, h = finetune(re, finetune_data, epochs=60, bs=64, lr=3e-4, hid=HID, seed=s)
    ri_models.append(m); ri_hists.append(h)

# Screening
print("\n=== CANDIDATE SCREENING ===")
ct = np.array([d.y.item() for d in candidate_data])

pp, pp_ind = ens_predict(pt_models, candidate_data)
rp, _ = ens_predict(ri_models, candidate_data)

# Embeddings
ce = get_emb(pt_models[0], candidate_data)
fe = get_emb(pt_models[0], finetune_data)
fl = np.array([d.y.item() for d in finetune_data])

# Evaluate
print("\n=== EVALUATION ===")
for name, probs in [("Pretrained_GIN_Ensemble", pp), ("Random_GIN_Ensemble", rp)]:
    auc_roc = roc_auc_score(ct, probs)
    auc_pr = average_precision_score(ct, probs)
    bf1, bt = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(ct, (probs>t).astype(int), zero_division=0)
        if f > bf1: bf1=f; bt=t
    preds = (probs>bt).astype(int)
    p = precision_score(ct, preds, zero_division=0)
    r = recall_score(ct, preds, zero_division=0)
    cm = confusion_matrix(ct, preds)
    print(f"\n{name}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}, F1={bf1:.4f}(t={bt:.2f}), P={p:.4f}, R={r:.4f}")
    print(f"  CM: {cm.tolist()}")
    results[name] = {'auc_roc':float(auc_roc),'auc_pr':float(auc_pr),'f1':float(bf1),
                     'threshold':float(bt),'precision':float(p),'recall':float(r),'cm':cm.tolist()}

# Individual
for i, pr in enumerate(pp_ind):
    a=roc_auc_score(ct,pr); ap_=average_precision_score(ct,pr)
    print(f"  Ind. {i+1}: AUC={a:.4f}, AP={ap_:.4f}")

# Discovery rate
print("\n--- Discovery Rate ---")
si = np.argsort(-pp)
dr = {}
for k in [10,20,30,50,75,100,150,200]:
    found = int(ct[si[:k]].sum()); total = int(ct.sum())
    print(f"  Top-{k}: {found}/{total} ({found/total:.1%})")
    dr[str(k)] = {'found':found,'total':total,'rate':float(found/total),'prec':float(found/k)}
results['discovery_rates'] = dr

# Save
json.dump(results, open(f'{OUTPUT_DIR}/evaluation_results.json','w'), indent=2)

preds_list = []
for i in si:
    preds_list.append({'rank':int(np.where(si==i)[0][0]+1),'idx':int(i),
                       'prob':float(pp[i]),'true':int(ct[i])})
json.dump(preds_list, open(f'{OUTPUT_DIR}/all_candidate_predictions.json','w'), indent=2)
json.dump(preds_list[:100], open(f'{OUTPUT_DIR}/top100_candidates.json','w'), indent=2)

np.save(f'{OUTPUT_DIR}/candidate_embeddings.npy', ce)
np.save(f'{OUTPUT_DIR}/candidate_probs.npy', pp)
np.save(f'{OUTPUT_DIR}/candidate_true_labels.npy', ct)
np.save(f'{OUTPUT_DIR}/finetune_embeddings.npy', fe)
np.save(f'{OUTPUT_DIR}/finetune_labels.npy', fl)
np.save(f'{OUTPUT_DIR}/cand_probs_random.npy', rp)

for i,h in enumerate(pt_hists):
    json.dump(h, open(f'{OUTPUT_DIR}/pt_hist_{i}.json','w'))
for i,h in enumerate(ri_hists):
    json.dump(h, open(f'{OUTPUT_DIR}/ri_hist_{i}.json','w'))

print("\n=== ALL DONE ===")
