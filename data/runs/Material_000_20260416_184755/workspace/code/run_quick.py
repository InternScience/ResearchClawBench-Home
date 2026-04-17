#!/usr/bin/env python3
"""Quick altermagnet discovery script"""
import sys, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

WS = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_000_20260416_184755'
os.makedirs(f'{WS}/outputs', exist_ok=True)
os.makedirs(f'{WS}/report/images', exist_ok=True)

class RealisticCrystalDataset(Dataset):
    def __init__(self, root, data_list=None, **kw):
        self.data_list = data_list or []
    def len(self): return len(self.data_list)
    def get(self, i): return self.data_list[i] if self.data_list else None
    @property
    def processed_file_names(self): return ['data.pt']
    @property
    def raw_file_names(self): return ['raw.pt']

class MockDataPrepare:
    RealisticCrystalDataset = RealisticCrystalDataset
sys.modules['data_prepare'] = MockDataPrepare()

print("Loading data...")
pretrain = torch.load(f'{WS}/data/pretrain_data.pt', weights_only=False)
finetune = torch.load(f'{WS}/data/finetune_data.pt', weights_only=False)
candidate = torch.load(f'{WS}/data/candidate_data.pt', weights_only=False)
print(f"Pretrain: {len(pretrain.data_list)}, Finetune: {len(finetune.data_list)}, Candidate: {len(candidate.data_list)}")

# Save stats
stats = {
    'pretrain': {'samples': len(pretrain.data_list)},
    'finetune': {'samples': len(finetune.data_list), 'pos': sum(int(d.y.item()) for d in finetune.data_list)},
    'candidate': {'samples': len(candidate.data_list), 'true_pos': sum(int(d.y.item()) for d in candidate.data_list)}
}
with open(f'{WS}/outputs/data_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

# Plot data overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sizes = [len(pretrain.data_list), len(finetune.data_list), len(candidate.data_list)]
names = ['Pre-train', 'Fine-tune', 'Candidate']
colors = ['#3498db', '#e74c3c', '#2ecc71']
axes[0, 0].bar(names, sizes, color=colors)
axes[0, 0].set_title('Dataset Sizes')

ft_labels = [int(d.y.item()) for d in finetune.data_list]
axes[0, 1].pie([len(ft_labels)-sum(ft_labels), sum(ft_labels)], labels=['Neg', 'Pos'], colors=['#95a5a6', '#e74c3c'], autopct='%1.1f%%')
axes[0, 1].set_title(f'Fine-tune Labels')

cand_labels = [int(d.y.item()) for d in candidate.data_list]
axes[1, 0].pie([len(cand_labels)-sum(cand_labels), sum(cand_labels)], labels=['Neg', 'Pos'], colors=['#95a5a6', '#27ae60'], autopct='%1.1f%%')
axes[1, 0].set_title(f'Candidate True Labels')

node_stats = [{'dataset': n, 'nodes': d.x.shape[0]} for n, dl in [('Pre', pretrain.data_list[:200]), ('Fine', finetune.data_list), ('Cand', candidate.data_list)] for d in dl]
sns.boxplot(data=pd.DataFrame(node_stats), x='dataset', y='nodes', ax=axes[1, 1], palette=colors)
axes[1, 1].set_title('Node Distribution')
plt.tight_layout()
plt.savefig(f'{WS}/report/images/data_overview.png', dpi=150)
plt.close()
print("Saved data_overview.png")

# Feature plot
vals = np.concatenate([d.x.flatten().numpy() for d in pretrain.data_list[:50]])
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(vals, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax.set_title('Feature Value Distribution')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{WS}/report/images/feature_analysis.png', dpi=150)
plt.close()
print("Saved feature_analysis.png")

# Prepare data
print("Preparing data...")
pyg_data = [Data(x=d.x, edge_index=d.edge_index, y=d.y) for d in finetune.data_list]
idx = np.random.permutation(len(pyg_data))
split = int(0.8 * len(pyg_data))
train_loader = DataLoader([pyg_data[i] for i in idx[:split]], batch_size=32, shuffle=True)
val_loader = DataLoader([pyg_data[i] for i in idx[split:]], batch_size=32)

pos_ratio = sum(int(d.y.item()) for d in finetune.data_list) / len(finetune.data_list)
class_weight = (1 - pos_ratio) / pos_ratio
print(f"Class weight: {class_weight:.2f}")

# Model
class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(28, 64)
        self.conv2 = GraphConv(64, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    def proba(self, x, ei, b):
        return torch.sigmoid(self.forward(x, ei, b))

model = GNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
print("Training...")
device = torch.device('cpu')
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(20):
    model.train()
    t_loss, t_corr, t_tot = 0, 0, 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        labels = batch.y.float()
        pw = torch.tensor([class_weight], device=device)
        loss = F.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1), pos_weight=pw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        t_loss += loss.item()
        pred = (torch.sigmoid(logits) > 0.5).squeeze()
        t_corr += (pred == labels).sum().item()
        t_tot += labels.size(0)
    scheduler.step()
    
    model.eval()
    v_loss, v_corr, v_tot = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1))
            v_loss += loss.item()
            pred = (torch.sigmoid(logits) > 0.5).squeeze()
            v_corr += (pred == labels).sum().item()
            v_tot += labels.size(0)
    
    train_losses.append(t_loss/len(train_loader))
    val_losses.append(v_loss/len(val_loader))
    train_accs.append(t_corr/t_tot)
    val_accs.append(v_corr/v_tot)
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1}: Train Loss={t_loss/len(train_loader):.4f}, Val Loss={v_loss/len(val_loader):.4f}")

# Save training log
with open(f'{WS}/outputs/model_training_log.json', 'w') as f:
    json.dump({'train_losses': train_losses, 'val_losses': val_losses, 'train_accs': train_accs, 'val_accs': val_accs}, f, indent=2)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, 21)
axes[0].plot(epochs, train_losses, 'b-', label='Train')
axes[0].plot(epochs, val_losses, 'r--', label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(epochs, train_accs, 'b-', label='Train Acc')
axes[1].plot(epochs, val_accs, 'r--', label='Val Acc')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{WS}/report/images/training_curves.png', dpi=150)
plt.close()
print("Saved training_curves.png")

# Evaluate
print("Evaluating...")
model.eval()
all_labels, all_probs = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        probs = model.proba(batch.x, batch.edge_index, batch.batch)
        all_probs.extend(probs.squeeze().cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

all_labels, all_probs = np.array(all_labels), np.array(all_probs)
all_preds = (all_probs > 0.5).astype(int)
metrics = {
    'accuracy': accuracy_score(all_labels, all_preds),
    'precision': precision_score(all_labels, all_preds, zero_division=0),
    'recall': recall_score(all_labels, all_preds, zero_division=0),
    'f1': f1_score(all_labels, all_preds, zero_division=0),
    'roc_auc': roc_auc_score(all_labels, all_probs)
}
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{WS}/report/images/roc_curve.png', dpi=150)
plt.close()
print("Saved roc_curve.png")

# Predict candidates
print("Predicting candidates...")
cand_pyg = [Data(x=d.x, edge_index=d.edge_index, y=d.y) for d in candidate.data_list]
cand_loader = DataLoader(cand_pyg, batch_size=32)

probs, labels = [], []
with torch.no_grad():
    for b in cand_loader:
        b = b.to(device)
        probs.extend(model.proba(b.x, b.edge_index, b.batch).squeeze().cpu().numpy())
        labels.extend(b.y.cpu().numpy())

df = pd.DataFrame({'id': range(len(probs)), 'prob': probs, 'pred': (np.array(probs) > 0.5).astype(int), 'true': labels})
df = df.sort_values('prob', ascending=False).reset_index(drop=True)
df.to_csv(f'{WS}/outputs/predictions.csv', index=False)

tp_disc = sum((df['pred']==1) & (df['true']==1))
tp_total = sum(df['true']==1)
rate = tp_disc / tp_total if tp_total else 0
top50_tp = sum(df.head(50)['true']==1)
print(f"\nDiscovery: Found {tp_disc}/{tp_total} true positives (rate={rate:.4f})")
print(f"Top 50 contains {top50_tp} true positives")

# Results plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].hist(df['prob'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[0, 0].set_xlabel('Probability'); axes[0, 0].set_title('Prediction Distribution')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

top20 = df.nlargest(20, 'prob')
colors = ['#27ae60' if t else '#e74c3c' for t in top20['true']]
axes[0, 1].barh(range(20), top20['prob'], color=colors)
axes[0, 1].set_yticks(range(20)); axes[0, 1].set_yticklabels([f"#{i}" for i in range(20)])
axes[0, 1].set_xlabel('Probability'); axes[0, 1].set_title('Top 20 Candidates')
axes[0, 1].invert_yaxis()

cm = confusion_matrix(df['true'], df['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_xlabel('Pred'); axes[1, 0].set_ylabel('True'); axes[1, 0].set_title('Confusion Matrix')

axes[1, 1].boxplot([df[df['true']==0]['prob'], df[df['true']==1]['prob']], labels=['True Neg', 'True Pos'])
axes[1, 1].set_ylabel('Probability'); axes[1, 1].set_title('By True Label')
axes[1, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{WS}/report/images/results_comparison.png', dpi=150)
plt.close()
print("Saved results_comparison.png")

# Save evaluation results
eval_res = {
    'val_metrics': {k: float(v) for k, v in metrics.items()},
    'discovery': {'found': int(tp_disc), 'total': int(tp_total), 'rate': float(rate), 'top50': int(top50_tp)}
}
with open(f'{WS}/outputs/evaluation_results.json', 'w') as f:
    json.dump(eval_res, f, indent=2)

print("\n" + "="*60)
print("Complete!")
print("="*60)
