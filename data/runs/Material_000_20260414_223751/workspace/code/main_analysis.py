import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import json
import os
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

pretrain_data = torch.load('data/pretrain_data.pt', map_location='cpu', weights_only=False)
finetune_data = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
candidate_data = torch.load('data/candidate_data.pt', map_location='cpu', weights_only=False)

NUM_NODE_FEAT = pretrain_data[0].x.shape[1]
print(f"Pretrain: {len(pretrain_data)}, Finetune: {len(finetune_data)}, Candidate: {len(candidate_data)}")
print(f"Node features: {NUM_NODE_FEAT}")

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.conv4(x, edge_index)
        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        g = g_mean + g_max
        return g, x

class Classifier(nn.Module):
    def __init__(self, encoder, emb_dim, hidden):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )

    def forward(self, data):
        g, _ = self.encoder(data)
        return self.head(g).squeeze(-1)

# ── Pretraining ──
print("\n=== Pretraining ===")
HIDDEN = 128
EMB = 64
encoder = GCNEncoder(NUM_NODE_FEAT, HIDDEN, EMB).to(DEVICE)
proj = nn.Sequential(nn.Linear(EMB, EMB), nn.ReLU(), nn.Linear(EMB, EMB)).to(DEVICE)
opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
loader = DataLoader(pretrain_data, batch_size=128, shuffle=True, drop_last=True)

for epoch in range(1, 61):
    encoder.train(); proj.train()
    total = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        g1, _ = encoder(batch)
        z1 = proj(g1)
        # Augment
        b2 = batch.clone()
        mask = (torch.rand(b2.x.size(0), device=DEVICE) > 0.15).float().unsqueeze(-1)
        b2.x = b2.x * mask
        g2, _ = encoder(b2)
        z2 = proj(g2)
        z1n = F.normalize(z1, dim=1)
        z2n = F.normalize(z2, dim=1)
        sim = torch.mm(z1n, z2n.t()) / 0.3
        lbl = torch.arange(sim.size(0), device=DEVICE)
        loss = 0.5 * (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.t(), lbl))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(proj.parameters()), 1.0)
        opt.step()
        total += loss.item()
    sched.step()
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {total/len(loader):.4f}")

# ── Fine-tuning ──
print("\n=== Fine-tuning ===")
labels_arr = np.array([finetune_data[i].y.item() for i in range(len(finetune_data))])
pos_idx = np.where(labels_arr == 1)[0]
neg_idx = np.where(labels_arr == 0)[0]
rng = np.random.RandomState(SEED)
rng.shuffle(pos_idx); rng.shuffle(neg_idx)

n_pt = int(0.8 * len(pos_idx))
n_nt = int(0.8 * len(neg_idx))
train_idx = np.concatenate([pos_idx[:n_pt], neg_idx[:n_nt]])
val_idx = np.concatenate([pos_idx[n_pt:], neg_idx[n_nt:]])
rng.shuffle(train_idx); rng.shuffle(val_idx)

train_set = [finetune_data[i] for i in train_idx]
val_set = [finetune_data[i] for i in val_idx]
n_pos = sum(1 for d in train_set if d.y.item() == 1)
n_neg = len(train_set) - n_pos
print(f"Train: {len(train_set)} (pos={n_pos}), Val: {len(val_set)} (pos={sum(1 for d in val_set if d.y.item()==1)})")

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

clf = Classifier(copy.deepcopy(encoder), EMB, HIDDEN).to(DEVICE)
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
print(f"Pos weight: {pos_weight.item():.2f}")

opt_f = torch.optim.Adam(clf.parameters(), lr=5e-4, weight_decay=1e-5)
sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=100)

best_auc = 0
best_state = None

for epoch in range(1, 101):
    clf.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        opt_f.zero_grad()
        logits = clf(batch)
        targets = batch.y.float().view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        opt_f.step()
        total_loss += loss.item() * batch.num_graphs
    sched_f.step()

    clf.eval()
    vp, vl = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            p = torch.sigmoid(clf(batch)).cpu().numpy()
            vp.extend(p)
            vl.extend(batch.y.cpu().numpy().flatten())
    vp, vl = np.array(vp), np.array(vl)
    try:
        auc = roc_auc_score(vl, vp)
    except:
        auc = 0.5
    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_set):.4f} | Val AUC: {auc:.4f}")

clf.load_state_dict(best_state)
print(f"\nBest Val AUC: {best_auc:.4f}")

# ── Final eval ──
clf.eval()
val_preds, val_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(DEVICE)
        val_preds.extend(torch.sigmoid(clf(batch)).cpu().numpy())
        val_labels.extend(batch.y.cpu().numpy().flatten())
val_preds = np.array(val_preds)
val_labels = np.array(val_labels)

fpr, tpr, thr_roc = roc_curve(val_labels, val_preds)
prec, rec, thr_pr = precision_recall_curve(val_labels, val_preds)
f1s = 2 * prec * rec / (prec + rec + 1e-8)
best_f1_idx = np.argmax(f1s)
opt_thr = thr_pr[best_f1_idx] if best_f1_idx < len(thr_pr) else 0.5
print(f"Optimal threshold: {opt_thr:.4f}")

val_bin = (val_preds > opt_thr).astype(int)
val_metrics = {
    'accuracy': accuracy_score(val_labels, val_bin),
    'precision': precision_score(val_labels, val_bin, zero_division=0),
    'recall': recall_score(val_labels, val_bin, zero_division=0),
    'f1': f1_score(val_labels, val_bin, zero_division=0),
    'auc_roc': roc_auc_score(val_labels, val_preds),
    'auc_pr': average_precision_score(val_labels, val_preds),
    'optimal_threshold': float(opt_thr),
    'confusion_matrix': confusion_matrix(val_labels, val_bin).tolist()
}
print(f"\nValidation Metrics:\n{json.dumps(val_metrics, indent=2)}")

# ── Candidates ──
print("\n=== Candidate Prediction ===")
cand_loader = DataLoader(candidate_data, batch_size=64, shuffle=False)
cand_preds, cand_labels = [], []
with torch.no_grad():
    for batch in cand_loader:
        batch = batch.to(DEVICE)
        cand_preds.extend(torch.sigmoid(clf(batch)).cpu().numpy())
        cand_labels.extend(batch.y.cpu().numpy().flatten())
cand_preds = np.array(cand_preds)
cand_labels = np.array(cand_labels)
cand_bin = (cand_preds > opt_thr).astype(int)

cand_metrics = {
    'accuracy': accuracy_score(cand_labels, cand_bin),
    'precision': precision_score(cand_labels, cand_bin, zero_division=0),
    'recall': recall_score(cand_labels, cand_bin, zero_division=0),
    'f1': f1_score(cand_labels, cand_bin, zero_division=0),
    'auc_roc': roc_auc_score(cand_labels, cand_preds),
    'auc_pr': average_precision_score(cand_labels, cand_preds),
    'confusion_matrix': confusion_matrix(cand_labels, cand_bin).tolist(),
    'total_candidates': len(cand_labels),
    'predicted_positive': int(cand_bin.sum()),
    'true_positive': int(cand_labels.sum()),
    'true_negative': int((cand_labels == 0).sum()),
    'discovered_true_positives': int(((cand_bin == 1) & (cand_labels == 1)).sum()),
    'missed_positives': int(((cand_bin == 0) & (cand_labels == 1)).sum()),
    'false_positives': int(((cand_bin == 1) & (cand_labels == 0)).sum()),
}
print(f"Candidate Metrics:\n{json.dumps(cand_metrics, indent=2)}")

# ── Save ──
os.makedirs('outputs', exist_ok=True)
results = {
    'val_metrics': val_metrics,
    'candidate_metrics': cand_metrics,
    'candidate_probabilities': cand_preds.tolist(),
    'candidate_labels': cand_labels.tolist(),
    'candidate_predictions': cand_bin.tolist(),
    'val_probabilities': val_preds.tolist(),
    'val_labels': val_labels.tolist(),
    'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
    'pr_curve': {'precision': prec.tolist(), 'recall': rec.tolist()},
}
with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)

discovered = []
for i in range(len(cand_preds)):
    if cand_bin[i] == 1:
        discovered.append({'index': i, 'probability': float(cand_preds[i]), 'true_label': int(cand_labels[i]), 'correct': bool(cand_labels[i] == 1)})
discovered.sort(key=lambda x: x['probability'], reverse=True)
with open('outputs/discovered_candidates.json', 'w') as f:
    json.dump(discovered, f, indent=2)

print(f"\nTotal discovered: {len(discovered)}")
print(f"Correct discoveries: {sum(1 for d in discovered if d['correct'])}")

# ── Figures ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('report/images', exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})

# Fig 1: Data overview
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
nc = [pretrain_data[i].num_nodes for i in range(len(pretrain_data))]
axes[0].hist(nc, bins=30, color='#4C72B0', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Number of Nodes'); axes[0].set_ylabel('Count')
axes[0].set_title('Pre-training Set (5,000 structures)')
axes[0].axvline(np.mean(nc), color='red', linestyle='--', label=f'Mean={np.mean(nc):.1f}')
axes[0].legend()
ftl = [finetune_data[i].y.item() for i in range(len(finetune_data))]
axes[1].bar(['Non-AM (0)', 'Altermagnet (1)'], [ftl.count(0), ftl.count(1)], color=['#55A868', '#C44E52'], edgecolor='white', width=0.6)
axes[1].set_ylabel('Count'); axes[1].set_title('Fine-tuning Set (2,000 samples)')
for i, v in enumerate([ftl.count(0), ftl.count(1)]):
    axes[1].text(i, v + 30, str(v), ha='center', fontweight='bold')
cl = [candidate_data[i].y.item() for i in range(len(candidate_data))]
axes[2].bar(['Non-AM (0)', 'Altermagnet (1)'], [cl.count(0), cl.count(1)], color=['#55A868', '#C44E52'], edgecolor='white', width=0.6)
axes[2].set_ylabel('Count'); axes[2].set_title('Candidate Set (1,000 samples)')
for i, v in enumerate([cl.count(0), cl.count(1)]):
    axes[2].text(i, v + 15, str(v), ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig('report/images/data_overview.png'); plt.close()
print("Saved data_overview.png")

# Fig 2: ROC and PR
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'GNN (AUC={val_metrics["auc_roc"]:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve (Validation)'); axes[0].legend(loc='lower right'); axes[0].grid(True, alpha=0.3)
bl = val_labels.sum() / len(val_labels)
axes[1].plot(rec, prec, 'r-', linewidth=2, label=f'GNN (AP={val_metrics["auc_pr"]:.3f})')
axes[1].axhline(y=bl, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({bl:.3f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('PR Curve (Validation)'); axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/roc_pr_curves.png'); plt.close()
print("Saved roc_pr_curves.png")

# Fig 3: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title in [(axes[0], val_metrics['confusion_matrix'], 'Validation'), (axes[1], cand_metrics['confusion_matrix'], 'Candidate')]:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Non-AM', 'AM'], yticklabels=['Non-AM', 'AM'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'Confusion Matrix ({title})')
plt.tight_layout(); plt.savefig('report/images/confusion_matrices.png'); plt.close()
print("Saved confusion_matrices.png")

# Fig 4: Probability distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, labels, title in [(axes[0], val_preds, val_labels, 'Validation'), (axes[1], cand_preds, cand_labels, 'Candidate')]:
    ax.hist(preds[labels == 0], bins=40, alpha=0.7, color='#55A868', label='Non-AM', density=True, edgecolor='white')
    ax.hist(preds[labels == 1], bins=20, alpha=0.7, color='#C44E52', label='Altermagnet', density=True, edgecolor='white')
    ax.axvline(opt_thr, color='black', linestyle='--', label=f'Threshold={opt_thr:.3f}')
    ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Density'); ax.set_title(f'{title} Probability Distribution'); ax.legend()
plt.tight_layout(); plt.savefig('report/images/probability_distributions.png'); plt.close()
print("Saved probability_distributions.png")

# Fig 5: Discovery results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
si = np.argsort(-cand_preds)
ks = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
pak, rak = [], []
tp = cand_labels.sum()
for k in ks:
    tpk = cand_labels[si[:k]].sum()
    pak.append(tpk / k)
    rak.append(tpk / tp)
axes[0].plot(ks, pak, 'o-', color='#4C72B0', linewidth=2, markersize=6)
axes[0].axhline(y=tp/len(cand_labels), color='red', linestyle='--', alpha=0.7, label=f'Random ({tp/len(cand_labels):.3f})')
axes[0].set_xlabel('Top-k Candidates'); axes[0].set_ylabel('Precision@k'); axes[0].set_title('Discovery Precision@k')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(ks, rak, 's-', color='#C44E52', linewidth=2, markersize=6)
axes[1].set_xlabel('Top-k Candidates'); axes[1].set_ylabel('Recall@k'); axes[1].set_title('Discovery Recall@k')
axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/discovery_results.png'); plt.close()
print("Saved discovery_results.png")

# Fig 6: Metrics comparison
fig, ax = plt.subplots(figsize=(10, 5))
mn = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'AUC-PR']
vv = [val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['f1'], val_metrics['auc_roc'], val_metrics['auc_pr']]
cv = [cand_metrics['accuracy'], cand_metrics['precision'], cand_metrics['recall'], cand_metrics['f1'], cand_metrics['auc_roc'], cand_metrics['auc_pr']]
x = np.arange(len(mn)); w = 0.35
b1 = ax.bar(x - w/2, vv, w, label='Validation', color='#4C72B0', edgecolor='white')
b2 = ax.bar(x + w/2, cv, w, label='Candidate', color='#C44E52', edgecolor='white')
ax.set_ylabel('Score'); ax.set_title('Model Performance Comparison'); ax.set_xticks(x); ax.set_xticklabels(mn)
ax.legend(); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis='y')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
plt.tight_layout(); plt.savefig('report/images/metrics_comparison.png'); plt.close()
print("Saved metrics_comparison.png")

print("\n=== All done ===")
