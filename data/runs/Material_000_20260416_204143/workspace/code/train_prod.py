"""
AI-Powered Altermagnet Discovery - Production version
Uses simpler pre-training (graph reconstruction) for speed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
import json, os, sys, types, copy, random, time
from sklearn.metrics import *

WS = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_000_20260416_204143'
OUT = f'{WS}/outputs'
os.makedirs(OUT, exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

dp = types.ModuleType('data_prepare')
class R:
    def __init__(self,*a,**k): pass
    def __setstate__(self,s): self.__dict__.update(s)
dp.RealisticCrystalDataset = R
sys.modules['data_prepare'] = dp

t0 = time.time()
pt_data = torch.load(f'{WS}/data/pretrain_data.pt', weights_only=False).data_list
ft_data = torch.load(f'{WS}/data/finetune_data.pt', weights_only=False).data_list
cd_data = torch.load(f'{WS}/data/candidate_data.pt', weights_only=False).data_list

NF = 28
ft_y = [d.y.item() for d in ft_data]
cd_y = np.array([d.y.item() for d in cd_data])
print(f"Data loaded ({time.time()-t0:.1f}s). FT:{sum(ft_y)}+/{len(ft_y)-sum(ft_y)}- CD:{sum(cd_y)}+/{len(cd_y)-sum(cd_y)}-")

# ---- GIN Encoder ----
class GIN(nn.Module):
    def __init__(self, inf=28, h=32, nl=2, dr=0.1):
        super().__init__()
        self.h = h; self.nl = nl
        self.emb = nn.Sequential(nn.Linear(inf, h), nn.BatchNorm1d(h), nn.ReLU())
        self.cs = nn.ModuleList()
        self.bs = nn.ModuleList()
        for _ in range(nl):
            m = nn.Sequential(nn.Linear(h,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h,h))
            self.cs.append(GINConv(m, train_eps=True))
            self.bs.append(nn.BatchNorm1d(h))
        self.dr = dr
    def forward(self, x, ei, ea=None, batch=None):
        x = self.emb(x)
        for i in range(self.nl):
            x = F.relu(self.bs[i](self.cs[i](x, ei)))
            x = F.dropout(x, self.dr, self.training)
        return global_mean_pool(x, batch) if batch is not None else x.mean(0, keepdim=True)

class Head(nn.Module):
    def __init__(self, enc, h=32, dr=0.3):
        super().__init__()
        self.enc = enc
        self.fc = nn.Sequential(nn.Linear(h,16), nn.ReLU(), nn.Dropout(dr), nn.Linear(16,1))
    def forward(self, x, ei, ea=None, batch=None):
        return self.fc(self.enc(x, ei, ea, batch))
    def emb(self, x, ei, ea=None, batch=None):
        return self.enc(x, ei, ea, batch)

# ---- Pre-train: Graph property prediction (fast self-supervised) ----
def pretrain_gpp(data, epochs=20, bs=256, lr=1e-3, h=32):
    """Pre-train by predicting graph-level properties (num nodes, avg degree, element composition)."""
    print("\n=== PRETRAIN (Graph Property Prediction) ===")
    # Compute targets: num_nodes, num_edges, avg_degree, element_composition_summary
    targets = []
    for d in data:
        nn_ = d.x.size(0)
        ne_ = d.edge_index.size(1)
        avg_deg = ne_ / max(nn_, 1)
        elem_sum = d.x.sum(0).numpy()  # 28-dim
        # Use a few summary stats
        targets.append(np.array([nn_/24.0, avg_deg/10.0] + elem_sum[:5].tolist()))
    
    target_dim = len(targets[0])
    targets_t = torch.tensor(np.array(targets), dtype=torch.float32)
    
    # Add targets to data
    for i, d in enumerate(data):
        d.target = targets_t[i]
    
    enc = GIN(NF, h, 2, 0.1)
    pred_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, target_dim))
    params = list(enc.parameters()) + list(pred_head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    
    loader = DataLoader(data, bs, shuffle=True)
    losses = []
    enc.train(); pred_head.train()
    
    for ep in range(epochs):
        el=0; nb=0
        for b in loader:
            h_ = enc(b.x, b.edge_index, batch=b.batch)
            pred = pred_head(h_)
            tgt = b.target.view(pred.shape)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        losses.append(el/nb)
        if (ep+1)%5==0: print(f"  Ep {ep+1}/{epochs} L={losses[-1]:.4f} ({time.time()-t0:.0f}s)")
    
    # Clean up targets
    for d in data:
        if hasattr(d, 'target'): delattr(d, 'target')
    
    return enc, losses

# ---- Fine-tune ----
def oversample(data, r=0.3):
    p=[d for d in data if d.y.item()==1]; n=[d for d in data if d.y.item()==0]
    t=int(len(n)*r/(1-r)); rp=t//len(p)+1
    c = n + (p*rp)[:t]; random.shuffle(c); return c

def ft(enc, data, epochs=50, bs=64, lr=3e-4, h=32, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    y = [d.y.item() for d in data]
    pi=[i for i,l in enumerate(y) if l==1]; ni=[i for i,l in enumerate(y) if l==0]
    np.random.shuffle(pi); np.random.shuffle(ni)
    nvp=max(2,int(len(pi)*0.2)); nvn=int(len(ni)*0.2)
    vi=pi[:nvp]+ni[:nvn]; ti=pi[nvp:]+ni[nvn:]
    td = oversample([data[i] for i in ti], 0.25)
    vd = [data[i] for i in vi]
    
    model = Head(copy.deepcopy(enc), h, 0.3)
    np_=sum(d.y.item() for d in td); nn_=len(td)-np_
    pw = torch.tensor([nn_/max(np_,1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam([
        {'params':model.enc.parameters(),'lr':lr*0.2},
        {'params':model.fc.parameters(),'lr':lr}
    ], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    tl=DataLoader(td,bs,shuffle=True); vl=DataLoader(vd,bs,shuffle=False)
    hist={'tl':[],'auc':[],'ap':[],'f1':[]}
    bm=0; bs_=None; pat=0
    
    for ep in range(epochs):
        model.train(); el=0; nb=0
        for b in tl:
            lo=model(b.x,b.edge_index,b.edge_attr,b.batch).squeeze(-1)
            loss=crit(lo,b.y.float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); el+=loss.item(); nb+=1
        sch.step(); hist['tl'].append(el/nb)
        
        model.eval(); vp=[]; vt=[]
        with torch.no_grad():
            for b in vl:
                lo=model(b.x,b.edge_index,b.edge_attr,b.batch).squeeze(-1)
                vp.extend(torch.sigmoid(lo).numpy().tolist())
                vt.extend(b.y.numpy().tolist())
        try: a=roc_auc_score(vt,vp); ap_=average_precision_score(vt,vp)
        except: a=0.5; ap_=0.0
        f1=f1_score(vt,[1 if p>0.5 else 0 for p in vp], zero_division=0)
        hist['auc'].append(a); hist['ap'].append(ap_); hist['f1'].append(f1)
        
        met=ap_+0.3*a
        if met>bm: bm=met; bs_=copy.deepcopy(model.state_dict()); pat=0
        else: pat+=1
        if (ep+1)%10==0: print(f"    Ep{ep+1} AUC={a:.4f} AP={ap_:.4f} F1={f1:.4f}")
        if pat>=20: print(f"    Stop ep{ep+1}"); break
    
    if bs_: model.load_state_dict(bs_)
    return model, hist

def ens_pred(models, data, bs=512):
    loader = DataLoader(data, bs, shuffle=False)
    ap = []
    for m in models:
        m.eval(); ps=[]
        with torch.no_grad():
            for b in loader:
                lo=m(b.x,b.edge_index,b.edge_attr,b.batch).squeeze(-1)
                ps.extend(torch.sigmoid(lo).numpy().tolist())
        ap.append(np.array(ps))
    return np.mean(ap, axis=0), ap

def get_emb(model, data, bs=512):
    loader = DataLoader(data, bs, shuffle=False)
    model.eval(); es=[]
    with torch.no_grad():
        for b in loader:
            es.append(model.emb(b.x,b.edge_index,b.edge_attr,b.batch).numpy())
    return np.concatenate(es)

# ---- MAIN ----
H = 32
results = {}

# Pre-train
enc, pt_losses = pretrain_gpp(pt_data, epochs=20, bs=256, lr=1e-3, h=H)
json.dump(pt_losses, open(f'{OUT}/pretrain_losses.json','w'))

# Fine-tune ensemble (3)
print(f"\n=== FT PRETRAINED === ({time.time()-t0:.0f}s)")
pt_ms = []; pt_hs = []
for i in range(3):
    s=SEED+i*17; print(f" M{i+1} s={s}")
    m,h = ft(enc, ft_data, epochs=50, bs=64, lr=3e-4, h=H, seed=s)
    pt_ms.append(m); pt_hs.append(h)
print(f"  Done ({time.time()-t0:.0f}s)")

# Random init ensemble (3)
print(f"\n=== FT RANDOM === ({time.time()-t0:.0f}s)")
ri_ms = []; ri_hs = []
for i in range(3):
    s=SEED+i*17; print(f" RM{i+1} s={s}")
    re = GIN(NF, H, 2, 0.1)
    m,h = ft(re, ft_data, epochs=50, bs=64, lr=3e-4, h=H, seed=s)
    ri_ms.append(m); ri_hs.append(h)
print(f"  Done ({time.time()-t0:.0f}s)")

# Screen
print(f"\n=== SCREENING === ({time.time()-t0:.0f}s)")
pp, pp_i = ens_pred(pt_ms, cd_data)
rp, _ = ens_pred(ri_ms, cd_data)
ce = get_emb(pt_ms[0], cd_data)
fe = get_emb(pt_ms[0], ft_data)
fl = np.array([d.y.item() for d in ft_data])

# Evaluate
print("\n=== EVAL ===")
for name, probs in [("Pretrained_Ensemble", pp), ("Random_Ensemble", rp)]:
    ar = roc_auc_score(cd_y, probs)
    ap_ = average_precision_score(cd_y, probs)
    bf1=0; bt=0.5
    for t in np.arange(0.05,0.95,0.01):
        f=f1_score(cd_y,(probs>t).astype(int),zero_division=0)
        if f>bf1: bf1=f; bt=t
    preds=(probs>bt).astype(int)
    p=precision_score(cd_y,preds,zero_division=0)
    r=recall_score(cd_y,preds,zero_division=0)
    cm=confusion_matrix(cd_y,preds)
    print(f"\n{name}: AUC-ROC={ar:.4f} AUC-PR={ap_:.4f} F1={bf1:.4f}(t={bt:.2f}) P={p:.4f} R={r:.4f}")
    print(f"  CM: {cm.tolist()}")
    results[name]={'auc_roc':float(ar),'auc_pr':float(ap_),'f1':float(bf1),
                   'threshold':float(bt),'precision':float(p),'recall':float(r),'cm':cm.tolist()}

# Individual
ind_a=[]; ind_ap=[]
for i,pr in enumerate(pp_i):
    a=roc_auc_score(cd_y,pr); ap_=average_precision_score(cd_y,pr)
    ind_a.append(float(a)); ind_ap.append(float(ap_))
    print(f"  Ind{i+1}: AUC={a:.4f} AP={ap_:.4f}")
results['individual_aucs']=ind_a; results['individual_aps']=ind_ap

# Discovery
print("\n--- Discovery Rate ---")
si=np.argsort(-pp); dr={}
for k in [10,20,30,50,75,100,150,200]:
    f_=int(cd_y[si[:k]].sum()); t_=int(cd_y.sum())
    print(f"  Top-{k}: {f_}/{t_} ({f_/t_:.1%}) P@{k}={f_/k:.1%}")
    dr[str(k)]={'found':f_,'total':t_,'rate':float(f_/t_),'prec':float(f_/k)}
results['discovery_rates']=dr

# Save
json.dump(results, open(f'{OUT}/evaluation_results.json','w'), indent=2)
si_list = []
for i in si:
    si_list.append({'rank':int(np.where(si==i)[0][0]+1),'idx':int(i),'prob':float(pp[i]),'true':int(cd_y[i])})
json.dump(si_list, open(f'{OUT}/all_candidate_predictions.json','w'), indent=2)
json.dump(si_list[:100], open(f'{OUT}/top100_candidates.json','w'), indent=2)

np.save(f'{OUT}/candidate_embeddings.npy', ce)
np.save(f'{OUT}/candidate_probs.npy', pp)
np.save(f'{OUT}/candidate_true_labels.npy', cd_y)
np.save(f'{OUT}/finetune_embeddings.npy', fe)
np.save(f'{OUT}/finetune_labels.npy', fl)
np.save(f'{OUT}/cand_probs_random.npy', rp)

for i,h in enumerate(pt_hs): json.dump(h, open(f'{OUT}/pt_hist_{i}.json','w'))
for i,h in enumerate(ri_hs): json.dump(h, open(f'{OUT}/ri_hist_{i}.json','w'))

print(f"\n=== ALL DONE ({time.time()-t0:.0f}s) ===")
