#!/usr/bin/env python3
import sys
import types
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from collections import Counter
import os

# Patch for loading
data_prepare_mod = types.ModuleType('data_prepare')
sys.modules['data_prepare'] = data_prepare_mod
class RealisticCrystalDataset:
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        data = self.data_list[idx]
        data.idx = torch.tensor([idx])
        return data
data_prepare_mod.RealisticCrystalDataset = RealisticCrystalDataset

class GINEncoder(torch.nn.Module):
    def __init__(self, input_dim=28, hidden_dim=64, num_layers=3, out_dim=128):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.mlp1 = torch.nn.Linear(input_dim, hidden_dim)
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.mlp_out = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.mlp1(x))
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index))
            x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.mlp_out(x)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1)
        )

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index, data.batch)
        out = self.head(h).squeeze(-1)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load datasets
pretrain_ds = torch.load('data/pretrain_data.pt', weights_only=False)
finetune_ds = torch.load('data/finetune_data.pt', weights_only=False)
candidate_ds = torch.load('data/candidate_data.pt', weights_only=False)

# Pretrain
print('Pretraining...')
encoder = GINEncoder().to(device)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.001, weight_decay=1e-5)
loader = DataLoader(pretrain_ds, batch_size=64, shuffle=True)
def contrastive_loss(z1, z2, temp=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temp
    labels = torch.arange(len(z1)).to(device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels) / 2

pre_losses = []
for epoch in range(50):
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        edge1, _ = dropout_edge(data.edge_index, p=0.1)
        h1 = encoder(data.x, edge1, data.batch)
        edge2, _ = dropout_edge(data.edge_index, p=0.1)
        h2 = encoder(data.x, edge2, data.batch)
        loss = contrastive_loss(h1, h2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    pre_losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f'Pre epoch {epoch}, loss {avg_loss:.4f}')

torch.save({'encoder': encoder.state_dict(), 'losses': pre_losses}, 'outputs/pretrained_encoder.pt')

# Finetune
print('Finetune...')
# Split finetune 80/20
indices = torch.randperm(len(finetune_ds))
train_idx, val_idx = indices[:1600], indices[1600:]
train_ds = torch.utils.data.Subset(finetune_ds, train_idx)
val_ds = torch.utils.data.Subset(finetune_ds, val_idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

encoder.load_state_dict(torch.load('outputs/pretrained_encoder.pt')['encoder'])
encoder.requires_grad_(False)  # Freeze for now, or fine tune
model = Classifier(encoder).to(device)
optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001)
bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([19.0]).to(device))  # imbalance

fin_losses = []
fin_val_aucs = []
for epoch in range(50):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.float()
        loss = bce(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    fin_losses.append(avg_loss)

    # Val
    model.eval()
    val_probs = []
    val_ys = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            prob = torch.sigmoid(out)
            val_probs.append(prob.cpu())
            val_ys.append(data.y.cpu().float())
    val_probs = torch.cat(val_probs)
    val_ys = torch.cat(val_ys)
    auc = roc_auc_score(val_ys, val_probs)
    fin_val_aucs.append(auc)
    if epoch % 10 == 0:
        print(f'Fin epoch {epoch}, loss {avg_loss:.4f}, val_auc {auc:.4f}')

torch.save(model.state_dict(), 'outputs/finetuned_model.pt')

# Predict candidate
print('Predicting candidates...')
model.eval()
cand_loader = DataLoader(candidate_ds, batch_size=64, shuffle=False)
probs = []
ys_true = []
with torch.no_grad():
    for data in cand_loader:
        data = data.to(device)
        out = model(data)
        prob = torch.sigmoid(out)
        probs.append(prob.cpu())
        ys_true.append(data.y.cpu().float())
probs = torch.cat(probs).numpy()
ys_true = torch.cat(ys_true).numpy()

# Metrics
auc_pr = average_precision_score(ys_true, probs)
auc_roc = roc_auc_score(ys_true, probs)
top50_tp = sum(ys_true[np.argsort(probs)[-50:]])
top100_tp = sum(ys_true[np.argsort(probs)[-100:]])

metrics = {
    'pretrain_final_loss': pre_losses[-1],
    'finetune_final_loss': fin_losses[-1],
    'finetune_val_auc': fin_val_aucs[-1],
    'candidate_auc_roc': auc_roc,
    'candidate_auc_pr': auc_pr,
    'top50_recall': top50_tp / 50,
    'top50_tp': top50_tp,
    'top100_tp': top100_tp,
    'total_true_pos': int(sum(ys_true))
}
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Top candidates
top_idx = np.argsort(probs)[-50:]
top_cand = [{'idx': int(i), 'prob': float(p), 'true_label': int(ys_true[i])} for i,p in zip(top_idx, probs[top_idx])]
with open('outputs/top_50_candidates.json', 'w') as f:
    json.dump(top_cand, f, indent=2)

print('Metrics:', metrics)
print('Top 50 TP:', top50_tp)

# Figures
fig, axs = plt.subplots(2, 2, figsize=(12,10))
axs[0,0].plot(pre_losses)
axs[0,0].set_title('Pretrain Loss')
axs[0,1].plot(fin_losses, label='train')
axs[0,1].plot(fin_val_aucs, label='val AUC')
axs[0,1].set_title('Finetune')
axs[0,1].legend()

fpr, tpr, _ = roc_curve(ys_true, probs)
axs[1,0].plot(fpr, tpr)
axs[1,0].plot([0,1],[0,1], 'k--')
axs[1,0].set_title('Candidate ROC')
axs[1,1].hist(probs[ys_true==0], bins=30, alpha=0.5, label='neg')
axs[1,1].hist(probs[ys_true==1], bins=30, alpha=0.5, label='pos')
axs[1,1].set_title('Pred Probs Candidate')
axs[1,1].legend()
plt.savefig('report/images/main_results.png', dpi=300, bbox_inches='tight')
plt.close()

print('Figures saved')
print('Done')
