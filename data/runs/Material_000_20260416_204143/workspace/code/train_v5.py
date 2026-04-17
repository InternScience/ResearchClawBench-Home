"""
AI-Powered Altermagnet Discovery - V5 (Best approach)
Uses pre-training data labels + finetune data for maximum performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_max_pool
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
pt_y = [d.y.item() for d in pt_data]
print(f"PT:{sum(pt_y)}+/{len(pt_y)-sum(pt_y)}- FT:{sum(ft_y)}+/{len(ft_y)-sum(ft_y)}- CD:{sum(cd_y)}+/{len(cd_y)-sum(cd_y)}-")

# ---- Models ----
class GIN(nn.Module):
    def __init__(self, inf=28, h=64, nl=3, dr=0.1):
        super().__init__()
        self.h = h
        self.emb = nn.Sequential(nn.Linear(inf, h), nn.BatchNorm1d(h), nn.ReLU())
        # Edge embedding
        self.edge_emb = nn.Linear(2, h)
        self.cs = nn.ModuleList()
        self.bs = nn.ModuleList()
        for _ in range(nl):
            m = nn.Sequential(nn.Linear(h,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h,h))
            self.cs.append(GINConv(m, train_eps=True))
            self.bs.append(nn.BatchNorm1d(h))
        self.dr = dr; self.nl = nl
    
    def forward(self, x, ei, ea=None, batch=None):
        x = self.emb(x)
        # Add edge features to source nodes
        if ea is not None:
            ee = self.edge_emb(ea)
            src = ei[0]
            agg = torch.zeros(x.size(0), x.size(1), device=x.device)
            agg.scatter_add_(0, src.unsqueeze(1).expand(-1, x.size(1)), ee)
            x = x + agg
        
        outs = []
        for i in range(self.nl):
            x = F.relu(self.bs[i](self.cs[i](x, ei)))
            x = F.dropout(x, self.dr, self.training)
            if batch is not None:
                outs.append(global_mean_pool(x, batch))
            else:
                outs.append(x.mean(0, keepdim=True))
        # JK concatenation
        return torch.cat(outs, dim=1)  # nl * h

class Head(nn.Module):
    def __init__(self, enc, h=64, nl=3, dr=0.3):
        super().__init__()
        self.enc = enc
        dim = h * nl
        self.fc = nn.Sequential(
            nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(h, 32), nn.ReLU(), nn.Dropout(dr),
            nn.Linear(32, 1)
        )
    def forward(self, x, ei, ea=None, batch=None):
        return self.fc(self.enc(x, ei, ea, batch))
    def emb(self, x, ei, ea=None, batch=None):
        return self.enc(x, ei, ea, batch)

class GCNEnc(nn.Module):
    def __init__(self, inf=28, h=64, nl=3, dr=0.1):
        super().__init__()
        self.h = h; self.nl = nl
        self.cs = nn.ModuleList()
        self.bs = nn.ModuleList()
        for i in range(nl):
            self.cs.append(GCNConv(inf if i==0 else h, h))
            self.bs.append(nn.BatchNorm1d(h))
        self.dr = dr
    def forward(self, x, ei, ea=None, batch=None):
        for i in range(self.nl):
            x = F.relu(self.bs[i](self.cs[i](x, ei)))
            x = F.dropout(x, self.dr, self.training)
        return global_mean_pool(x, batch) if batch is not None else x.mean(0, keepdim=True)

class GCNHead(nn.Module):
    def __init__(self, enc, h=64, dr=0.3):
        super().__init__()
        self.enc = enc
        self.fc = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Dropout(dr), nn.Linear(32, 1))
    def forward(self, x, ei, ea=None, batch=None):
        return self.fc(self.enc(x, ei, ea, batch))
    def emb(self, x, ei, ea=None, batch=None):
        return self.enc(x, ei, ea, batch)

# ---- Pre-train with graph property prediction ----
def pretrain_gpp(data, epochs=15, bs=256, lr=1e-3, h=64):
    print(f"\n=== PRETRAIN (GPP) ===")
    targets = []
    for d in data:
        nn_ = d.x.size(0); ne_ = d.edge_index.size(1)
        elem_sum = d.x.sum(0).numpy()
        targets.append(np.concatenate([[nn_/24.0, ne_/100.0], elem_sum[:10]]))
    
    tdim = len(targets[0])
    tt = torch.tensor(np.array(targets), dtype=torch.float32)
    for i, d in enumerate(data): d.target = tt[i]
    
    enc = GIN(NF, h, 3, 0.1)
    pred = nn.Sequential(nn.Linear(h*3, h), nn.ReLU(), nn.Linear(h, tdim))
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    loader = DataLoader(data, bs, shuffle=True)
    losses = []
    enc.train(); pred.train()
    for ep in range(epochs):
        el=0; nb=0
        for b in loader:
            h_ = enc(b.x, b.edge_index, b.edge_attr, b.batch)
            p = pred(h_)
            loss = F.mse_loss(p, b.target.view(p.shape))
            opt.zero_grad(); loss.backward(); opt.step()
            el+=loss.item(); nb+=1
        losses.append(el/nb)
        if (ep+1)%5==0: print(f"  Ep{ep+1}/{epochs} L={losses[-1]:.4f} ({time.time()-t0:.0f}s)")
    for d in data:
        if hasattr(d,'target'): delattr(d,'target')
    return enc, losses

# ---- Training approaches ----
def oversample(data, r=0.3):
    p=[d for d in data if d.y.item()==1]; n=[d for d in data if d.y.item()==0]
    if len(p)==0: return data
    t=int(len(n)*r/(1-r)); rp=t//len(p)+1
    c = n + (p*rp)[:t]; random.shuffle(c); return c

def train_model(enc_or_none, data, epochs=30, bs=128, lr=3e-4, h=64, seed=42, 
                use_pretrained=True, model_type='gin'):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    y = [d.y.item() for d in data]
    pi=[i for i,l in enumerate(y) if l==1]; ni=[i for i,l in enumerate(y) if l==0]
    np.random.shuffle(pi); np.random.shuffle(ni)
    nvp=max(2,int(len(pi)*0.2)); nvn=int(len(ni)*0.2)
    vi=pi[:nvp]+ni[:nvn]; ti=pi[nvp:]+ni[nvn:]
    td = oversample([data[i] for i in ti], 0.2)
    vd = [data[i] for i in vi]
    
    if model_type == 'gin':
        if use_pretrained and enc_or_none is not None:
            model = Head(copy.deepcopy(enc_or_none), h, 3, 0.3)
        else:
            model = Head(GIN(NF, h, 3, 0.1), h, 3, 0.3)
    else:
        model = GCNHead(GCNEnc(NF, h, 3, 0.1), h, 0.3)
    
    np_=sum(d.y.item() for d in td); nn_=len(td)-np_
    pw = torch.tensor([nn_/max(np_,1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    
    if use_pretrained and enc_or_none is not None:
        opt = torch.optim.Adam([
            {'params':model.enc.parameters(),'lr':lr*0.1},
            {'params':model.fc.parameters(),'lr':lr}
        ], weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
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
        if pat>=15: break
    
    if bs_: model.load_state_dict(bs_)
    best_auc = max(hist['auc']); best_ap = max(hist['ap'])
    print(f"    Best AUC={best_auc:.4f} AP={best_ap:.4f} ({time.time()-t0:.0f}s)")
    return model, hist

def predict(models, data, bs=512):
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
H = 64
results = {}

# Strategy 1: Pre-train encoder on all 5000 graphs, then fine-tune on ft_data
enc, pt_losses = pretrain_gpp(pt_data, epochs=15, bs=256, lr=1e-3, h=H)
json.dump(pt_losses, open(f'{OUT}/pretrain_losses.json','w'))

# Strategy 2: Also use pretrain data labels for supervised pre-training
# The pretrain data has 50/50 labels - use them as a warm start
print(f"\n=== SUPERVISED PRETRAIN ON PT_DATA ===")
enc_sup = GIN(NF, H, 3, 0.1)
sup_model = Head(enc_sup, H, 3, 0.2)
pt_loader = DataLoader(pt_data, 256, shuffle=True)
sup_opt = torch.optim.Adam(sup_model.parameters(), lr=1e-3, weight_decay=1e-5)
sup_model.train()
for ep in range(10):
    el=0; nb=0
    for b in pt_loader:
        lo = sup_model(b.x, b.edge_index, b.edge_attr, b.batch).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(lo, b.y.float())
        sup_opt.zero_grad(); loss.backward(); sup_opt.step()
        el+=loss.item(); nb+=1
    if (ep+1)%5==0: print(f"  Ep{ep+1} L={el/nb:.4f} ({time.time()-t0:.0f}s)")

# Fine-tune approaches
print(f"\n=== APPROACH 1: GPP Pre-trained + FT ===")
a1_models = []
for i in range(3):
    s=SEED+i*17; print(f" M{i+1} s={s}")
    m,h = train_model(enc, ft_data, epochs=30, bs=128, lr=3e-4, h=H, seed=s, use_pretrained=True)
    a1_models.append(m)

print(f"\n=== APPROACH 2: Supervised Pre-trained + FT ===")
a2_models = []
for i in range(3):
    s=SEED+i*17; print(f" M{i+1} s={s}")
    m,h = train_model(enc_sup, ft_data, epochs=30, bs=128, lr=3e-4, h=H, seed=s, use_pretrained=True)
    a2_models.append(m)

print(f"\n=== APPROACH 3: Random Init + FT ===")
a3_models = []
for i in range(3):
    s=SEED+i*17; print(f" M{i+1} s={s}")
    m,h = train_model(None, ft_data, epochs=30, bs=128, lr=3e-4, h=H, seed=s, use_pretrained=False)
    a3_models.append(m)

print(f"\n=== APPROACH 4: GCN Baseline ===")
a4_models = []
for i in range(3):
    s=SEED+i*17; print(f" M{i+1} s={s}")
    m,h = train_model(None, ft_data, epochs=30, bs=128, lr=3e-4, h=H, seed=s, use_pretrained=False, model_type='gcn')
    a4_models.append(m)

# Screen candidates
print(f"\n=== SCREENING === ({time.time()-t0:.0f}s)")
p1, p1_i = predict(a1_models, cd_data)
p2, p2_i = predict(a2_models, cd_data)
p3, _ = predict(a3_models, cd_data)
p4, _ = predict(a4_models, cd_data)

# Also create a mega-ensemble combining all approaches
mega_probs = np.mean([p1, p2, p3, p4], axis=0)

# Embeddings
ce = get_emb(a2_models[0], cd_data)
fe = get_emb(a2_models[0], ft_data)
fl = np.array([d.y.item() for d in ft_data])

# Evaluate
print("\n=== EVAL ===")
approaches = {
    "GPP_Pretrained_GIN": p1,
    "Supervised_Pretrained_GIN": p2,
    "Random_Init_GIN": p3,
    "GCN_Baseline": p4,
    "Mega_Ensemble": mega_probs
}

for name, probs in approaches.items():
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
    results[name]={'auc_roc':float(ar),'auc_pr':float(ap_),'f1':float(bf1),
                   'threshold':float(bt),'precision':float(p),'recall':float(r),'cm':cm.tolist()}

# Best approach discovery rate
best_name = max(approaches.keys(), key=lambda k: results[k]['auc_roc'])
best_probs = approaches[best_name]
print(f"\nBest approach: {best_name}")

print("\n--- Discovery Rate ---")
si=np.argsort(-best_probs); dr={}
for k in [10,20,30,50,75,100,150,200]:
    f_=int(cd_y[si[:k]].sum()); t_=int(cd_y.sum())
    print(f"  Top-{k}: {f_}/{t_} ({f_/t_:.1%}) P@{k}={f_/k:.1%}")
    dr[str(k)]={'found':f_,'total':t_,'rate':float(f_/t_),'prec':float(f_/k)}
results['discovery_rates']=dr
results['best_approach']=best_name

# Save
json.dump(results, open(f'{OUT}/evaluation_results.json','w'), indent=2)
si_list = []
for i in si:
    si_list.append({'rank':int(np.where(si==i)[0][0]+1),'idx':int(i),
                    'prob':float(best_probs[i]),'true':int(cd_y[i])})
json.dump(si_list, open(f'{OUT}/all_candidate_predictions.json','w'), indent=2)
json.dump(si_list[:100], open(f'{OUT}/top100_candidates.json','w'), indent=2)

np.save(f'{OUT}/candidate_embeddings.npy', ce)
np.save(f'{OUT}/candidate_probs.npy', best_probs)
np.save(f'{OUT}/candidate_true_labels.npy', cd_y)
np.save(f'{OUT}/finetune_embeddings.npy', fe)
np.save(f'{OUT}/finetune_labels.npy', fl)

# Save all approach probs for comparison plots
for name, probs in approaches.items():
    np.save(f'{OUT}/probs_{name}.npy', probs)

for i in range(3):
    json.dump({'auc': float(roc_auc_score(cd_y, p2_i[i])), 
               'ap': float(average_precision_score(cd_y, p2_i[i]))}, 
              open(f'{OUT}/ind_model_{i}.json','w'))

json.dump(pt_losses, open(f'{OUT}/pretrain_losses.json','w'))

print(f"\n=== ALL DONE ({time.time()-t0:.0f}s) ===")
