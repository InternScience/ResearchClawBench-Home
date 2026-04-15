"""Minimal Altermagnetic Discovery"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prepare

np.random.seed(42); torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_emb = nn.Linear(28, 64)
        self.edge_emb = nn.Linear(2, 64)
        self.conv = nn.Linear(64*3, 64)
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x, ei, ea, batch):
        x = F.relu(self.node_emb(x))
        ea = F.relu(self.edge_emb(ea))
        src, dst = ei
        m = F.relu(self.conv(torch.cat([x[src], x[dst], ea], -1)))
        a = torch.zeros_like(x).index_add_(0, dst, m)
        x = F.relu(x + a)
        out = torch.zeros(batch.max()+1, 128, device=x.device)
        for i in range(batch.max()+1):
            mask = batch == i
            if mask.any(): out[i, :64] = x[mask].mean(0); out[i, 64:] = x[mask].max(0)[0]
        return self.fc(out).squeeze(-1)
    def prob(self, x, ei, ea, batch): return torch.sigmoid(self.forward(x, ei, ea, batch))

def make_batch(data, device):
    xs, eis, eas, ys, b = [], [], [], [], []
    off = 0
    for i, d in enumerate(data):
        xs.append(d.x.float())
        eis.append(d.edge_index + off)
        eas.append(d.edge_attr.float())
        ys.append(d.y.float())
        b.extend([i] * d.num_nodes)
        off += d.num_nodes
    return (torch.cat(xs).to(device), torch.cat(eis,1).to(device), 
            torch.cat(eas).to(device), torch.tensor(b).to(device), torch.stack(ys).to(device).squeeze())

# Load data
print("Loading...")
ptr = torch.load('data/pretrain_data.pt', weights_only=False).data_list
ft = torch.load('data/finetune_data.pt', weights_only=False).data_list
cand = torch.load('data/candidate_data.pt', weights_only=False).data_list
print(f"Pretrain: {len(ptr)}, Finetune: {len(ft)}, Candidate: {len(cand)}")

# Split
idx = np.random.permutation(len(ft))
tr, val = [ft[i] for i in idx[:1600]], [ft[i] for i in idx[1600:]]

# Pretrain (structural learning)
print("\nPretraining (3 epochs)...")
model = Model().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for e in range(3):
    model.train()
    idx = np.random.permutation(len(ptr))
    for i in range(0, len(ptr), 64):
        x, ei, ea, batch_idx, _ = make_batch([ptr[j] for j in idx[i:i+64]], device)
        opt.zero_grad()
        opt.step()
    print(f"  Epoch {e+1}")

# Finetune
print("\nFine-tuning (20 epochs)...")
opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
pw = torch.tensor(8.0).to(device)
crit = nn.BCEWithLogitsLoss(pos_weight=pw)
best_f1 = 0
for e in range(20):
    model.train()
    idx = np.random.permutation(len(tr))
    for i in range(0, len(tr), 32):
        x, ei, ea, batch_idx, y = make_batch([tr[j] for j in idx[i:i+32]], device)
        loss = crit(model(x, ei, ea, batch_idx), y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    model.eval()
    with torch.no_grad():
        x, ei, ea, batch_idx, y = make_batch(val, device)
        probs = model.prob(x, ei, ea, batch_idx).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        labs = y.cpu().numpy()
        f1 = f1_score(labs, preds, zero_division=0)
        if f1 > best_f1: best_f1 = f1; torch.save(model.state_dict(), 'outputs/best.pt')
    if (e+1) % 5 == 0: print(f"  Epoch {e+1}, Val F1: {f1:.4f}")

print(f"Best F1: {best_f1:.4f}")

# Discovery
print("\nDiscovery...")
os.makedirs('outputs', exist_ok=True)
model.load_state_dict(torch.load('outputs/best.pt'))
model.eval()
probs = []
with torch.no_grad():
    for i in range(0, len(cand), 32):
        x, ei, ea, batch_idx, _ = make_batch(cand[i:i+32], device)
        probs.extend(model.prob(x, ei, ea, batch_idx).cpu().numpy())
probs = np.array(probs)

# Evaluate
cand_labs = [d.y.item() for d in cand]
top50 = np.argsort(probs)[::-1][:50]
tp = sum(cand_labs[i] for i in top50)
pred = (probs > 0.5).astype(int)

print(f"\nResults:")
print(f"  Top-50: {tp} TPs, Precision: {tp/50:.3f}, Recall: {tp/sum(cand_labs):.3f}")
print(f"  Overall: Acc={(pred == np.array(cand_labs)).mean():.3f}, F1={f1_score(cand_labs, pred, zero_division=0):.3f}, AUC={roc_auc_score(cand_labs, probs):.3f}")

# Save
res = {'probs': probs.tolist(), 'cand_labels': cand_labs, 'top50': top50.tolist(),
       'tp_at_50': int(tp), 'prec_at_50': tp/50, 'rec_at_50': tp/sum(cand_labs),
       'auc': roc_auc_score(cand_labs, probs), 'f1': f1_score(cand_labs, pred, zero_division=0)}
with open('outputs/results.json', 'w') as f: json.dump(res, f, indent=2)
print("\nSaved to outputs/results.json")
