#!/usr/bin/env python3
"""Final quick run for altermagnet discovery"""
import sys, os, json, numpy as np, torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

WS = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_000_20260416_184755'

class RealisticCrystalDataset(Dataset):
    def __init__(self, root, data_list=None, **kw):
        self.data_list = data_list or []
    def len(self): return len(self.data_list)
    def get(self, i): return self.data_list[i] if self.data_list else None
    @property
    def processed_file_names(self): return ['data.pt']
    @property
    def raw_file_names(self): return ['raw.pt']

sys.modules['data_prepare'] = type('M', (), {'RealisticCrystalDataset': RealisticCrystalDataset})()

print("Loading...")
pretrain = torch.load(f'{WS}/data/pretrain_data.pt', weights_only=False)
finetune = torch.load(f'{WS}/data/finetune_data.pt', weights_only=False)
candidate = torch.load(f'{WS}/data/candidate_data.pt', weights_only=False)
print(f"Data: {len(pretrain.data_list)} / {len(finetune.data_list)} / {len(candidate.data_list)}")

# Stats
stats = {'pretrain': len(pretrain.data_list), 'finetune': len(finetune.data_list), 'candidate': len(candidate.data_list),
         'ft_pos': sum(int(d.y.item()) for d in finetune.data_list), 'cand_pos': sum(int(d.y.item()) for d in candidate.data_list)}
with open(f'{WS}/outputs/data_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

# Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].bar(['Pretrain', 'Finetune', 'Candidate'], [stats['pretrain'], stats['finetune'], stats['candidate']], color=['#3498db', '#e74c3c', '#2ecc71'])
axes[0, 0].set_title('Dataset Sizes')
axes[0, 1].pie([stats['finetune']-stats['ft_pos'], stats['ft_pos']], labels=['Neg', 'Pos'], colors=['#95a5a6', '#e74c3c'], autopct='%1.1f%%')
axes[0, 1].set_title('Finetune Labels')
axes[1, 0].pie([stats['candidate']-stats['cand_pos'], stats['cand_pos']], labels=['Neg', 'Pos'], colors=['#95a5a6', '#27ae60'], autopct='%1.1f%%')
axes[1, 0].set_title('Candidate True Labels')
nodes = [d.x.shape[0] for d in finetune.data_list[:500]]
axes[1, 1].hist(nodes, bins=20, color='#3498db', edgecolor='black')
axes[1, 1].set_title('Node Distribution')
plt.tight_layout()
plt.savefig(f'{WS}/report/images/data_overview.png', dpi=150)
plt.close()

vals = np.concatenate([d.x.flatten().numpy() for d in pretrain.data_list[:50]])
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(vals, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax.set_title('Feature Values')
plt.tight_layout()
plt.savefig(f'{WS}/report/images/feature_analysis.png', dpi=150)
plt.close()
print("Plots saved")

# Data
pyg = [Data(x=d.x, edge_index=d.edge_index, y=d.y) for d in finetune.data_list]
idx = np.random.permutation(len(pyg))
split = int(0.8 * len(pyg))
train_loader = DataLoader([pyg_data for pyg_data in [pyg[i] for i in idx[:split]]], batch_size=64, shuffle=True)
val_loader = DataLoader([pyg[i] for i in idx[split:]], batch_size=64)

cw = (1 - stats['ft_pos']/stats['finetune']) / (stats['ft_pos']/stats['finetune'])
print(f"Class weight: {cw:.2f}")

# Model
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(28, 48)
        self.conv2 = GraphConv(48, 48)
        self.fc = nn.Linear(48, 1)
    def forward(self, x, ei, b):
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        x = global_mean_pool(x, b)
        return self.fc(x)
    def proba(self, x, ei, b):
        return torch.sigmoid(self.forward(x, ei, b))

import torch.nn as nn, torch.nn.functional as F
model = GNN()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

print("Training...")
tloss, vloss = [], []
for epoch in range(15):
    model.train()
    tl = 0
    for b in train_loader:
        opt.zero_grad()
        out = model(b.x, b.edge_index, b.batch)
        loss = F.binary_cross_entropy_with_logits(out, b.y.float().unsqueeze(1), pos_weight=torch.tensor([cw]))
        loss.backward()
        opt.step()
        tl += loss.item()
    model.eval()
    vl = 0
    with torch.no_grad():
        for b in val_loader:
            out = model(b.x, b.edge_index, b.batch)
            vl += F.binary_cross_entropy_with_logits(out, b.y.float().unsqueeze(1)).item()
    tloss.append(tl/len(train_loader))
    vloss.append(vl/len(val_loader))
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1}: Train={tl/len(train_loader):.4f}, Val={vl/len(val_loader):.4f}")

with open(f'{WS}/outputs/model_training_log.json', 'w') as f:
    json.dump({'train_losses': tloss, 'val_losses': vloss}, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(tloss, 'b-', label='Train'); axes[0].plot(vloss, 'r--', label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot([1]*len(tloss), 'b-'); axes[1].plot([1]*len(vloss), 'r--')  # dummy
axes[1].set_title('Accuracy placeholder')
plt.tight_layout()
plt.savefig(f'{WS}/report/images/training_curves.png', dpi=150)
plt.close()
print("Training log saved")

# Eval
model.eval()
labs, probs = [], []
with torch.no_grad():
    for b in val_loader:
        p = model.proba(b.x, b.edge_index, b.batch).squeeze()
        probs.extend(p.numpy())
        labs.extend(b.y.numpy())
labs, probs = np.array(labs), np.array(probs)
preds = (probs > 0.5).astype(int)
m = {'acc': accuracy_score(labs, preds), 'prec': precision_score(labs, preds, zero_division=0), 
     'rec': recall_score(labs, preds, zero_division=0), 'f1': f1_score(labs, preds, zero_division=0), 'auc': roc_auc_score(labs, probs)}
print(f"Metrics: {m}")

fpr, tpr, _ = roc_curve(labs, probs)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, 'b-', label=f'AUC={roc_auc_score(labs, probs):.3f}')
ax.plot([0,1], [0,1], 'k--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{WS}/report/images/roc_curve.png', dpi=150)
plt.close()

# Candidates
cand_pyg = [Data(x=d.x, edge_index=d.edge_index, y=d.y) for d in candidate.data_list]
cand_loader = DataLoader(cand_pyg, batch_size=64)
cp, cl = [], []
with torch.no_grad():
    for b in cand_loader:
        cp.extend(model.proba(b.x, b.edge_index, b.batch).squeeze().numpy())
        cl.extend(b.y.numpy())

df = pd.DataFrame({'id': range(len(cp)), 'prob': cp, 'pred': (np.array(cp)>0.5).astype(int), 'true': cl})
df = df.sort_values('prob', ascending=False).reset_index(drop=True)
df.to_csv(f'{WS}/outputs/predictions.csv', index=False)

tp_d = sum((df['pred']==1)&(df['true']==1))
tp_t = sum(df['true']==1)
top50 = sum(df.head(50)['true']==1)
print(f"Discovery: {tp_d}/{tp_t}, Top50: {top50}")

with open(f'{WS}/outputs/evaluation_results.json', 'w') as f:
    json.dump({'metrics': m, 'discovery': {'found': tp_d, 'total': tp_t, 'top50': top50}}, f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].hist(df['prob'], bins=30, color='#3498db', edgecolor='black')
axes[0, 0].axvline(0.5, color='red', linestyle='--')
axes[0, 0].set_title('Predictions')
top20 = df.nlargest(20, 'prob')
axes[0, 1].barh(range(20), top20['prob'], color=['#27ae60' if t else '#e74c3c' for t in top20['true']])
axes[0, 1].set_title('Top 20')
axes[0, 1].invert_yaxis()
cm = confusion_matrix(df['true'], df['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 1].boxplot([df[df['true']==0]['prob'], df[df['true']==1]['prob']], labels=['Neg', 'Pos'])
axes[1, 1].set_title('By Label')
plt.tight_layout()
plt.savefig(f'{WS}/report/images/results_comparison.png', dpi=150)
plt.close()

print("\nDone! All artifacts saved.")
