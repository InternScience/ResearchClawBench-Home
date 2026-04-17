#!/usr/bin/env python3
"""Final training script for DIDS-MFL."""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

print('Loading data...')
data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', weights_only=False)
store = data.stores[0]

# Use subset for faster training
n_samples = 8000
X = store.msg[:n_samples].float()
y_bin = store.label[:n_samples].long()
y_multi = store.attack[:n_samples].long()

print(f'Loaded {len(X)} samples')
print(f'Binary dist: {torch.bincount(y_bin).tolist()}')
print(f'Multi dist: {torch.bincount(y_multi).tolist()}')

# Split
n = len(X)
indices = torch.randperm(n)
train_idx = indices[:int(0.8*n)]
test_idx = indices[int(0.8*n):]

X_train, y_bin_train, y_multi_train = X[train_idx], y_bin[train_idx], y_multi[train_idx]
X_test, y_bin_test, y_multi_test = X[test_idx], y_bin[test_idx], y_multi[test_idx]

print(f'Train: {len(X_train)}, Test: {len(X_test)}')

# Model
class Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

crit = nn.CrossEntropyLoss()

# Binary training
print('\n=== Binary Classification ===')
model_bin = Net(2)
opt = torch.optim.Adam(model_bin.parameters(), lr=0.001)

train_losses = []
for i in range(10):
    opt.zero_grad()
    loss = crit(model_bin(X_train), y_bin_train)
    loss.backward()
    opt.step()
    train_losses.append(loss.item())
    if (i+1) % 5 == 0:
        print(f'Iter {i+1}: loss={loss.item():.4f}')

model_bin.eval()
with torch.no_grad():
    logits = model_bin(X_test)
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)

acc_bin = accuracy_score(y_bin_test.numpy(), preds.numpy())
f1_bin = f1_score(y_bin_test.numpy(), preds.numpy())
fpr, tpr, _ = roc_curve(y_bin_test.numpy(), probs.numpy()[:, 1])
roc_auc = auc(fpr, tpr)
print(f'Binary: Acc={acc_bin:.4f}, F1={f1_bin:.4f}, ROC-AUC={roc_auc:.4f}')

with open('outputs/binary_metrics.json', 'w') as f:
    json.dump({'accuracy': acc_bin, 'f1': f1_bin, 'roc_auc': roc_auc}, f, indent=2)

# Binary plots
cm_bin = confusion_matrix(y_bin_test.numpy(), preds.numpy())
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
plt.title('Binary Confusion Matrix')
plt.tight_layout()
plt.savefig('report/images/binary_confusion_matrix.png', dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('report/images/binary_roc_curve.png', dpi=150)
plt.close()

# Multiclass training
print('\n=== Multiclass Classification ===')
model_mc = Net(10)
opt_mc = torch.optim.Adam(model_mc.parameters(), lr=0.001)

mc_losses = []
for i in range(10):
    opt_mc.zero_grad()
    loss = crit(model_mc(X_train), y_multi_train)
    loss.backward()
    opt_mc.step()
    mc_losses.append(loss.item())
    if (i+1) % 5 == 0:
        print(f'Iter {i+1}: loss={loss.item():.4f}')

model_mc.eval()
with torch.no_grad():
    logits_mc = model_mc(X_test)
    preds_mc = torch.argmax(logits_mc, dim=-1)

acc_mc = accuracy_score(y_multi_test.numpy(), preds_mc.numpy())
f1_mc = f1_score(y_multi_test.numpy(), preds_mc.numpy(), average='weighted')
print(f'Multiclass: Acc={acc_mc:.4f}, F1={f1_mc:.4f}')

attack_names = ['Normal', 'DoS', 'Probe', 'U2R', 'R2L', 'DDoS', 'Bot', 'Web', 'Exploit', 'Shellcode']
class_f1 = f1_score(y_multi_test.numpy(), preds_mc.numpy(), average=None, zero_division=0)
per_class = {attack_names[i]: float(f1) for i, f1 in enumerate(class_f1)}
print('Per-class F1:', per_class)

with open('outputs/multiclass_metrics.json', 'w') as f:
    json.dump({'accuracy': acc_mc, 'f1_weighted': f1_mc, 'per_class_f1': per_class}, f, indent=2)

# Multiclass plots
cm_mc = confusion_matrix(y_multi_test.numpy(), preds_mc.numpy())
plt.figure(figsize=(10, 8))
sns.heatmap(cm_mc, annot=False, cmap='Blues', xticklabels=attack_names, yticklabels=attack_names)
plt.title('Multiclass Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('report/images/multiclass_confusion_matrix.png', dpi=150)
plt.close()

# Data distribution plots
unique_b, counts_b = np.unique(y_bin.numpy(), return_counts=True)
plt.figure(figsize=(6, 4))
plt.bar(['Benign', 'Attack'], counts_b, color='steelblue')
plt.title('Binary Class Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('report/images/binary_class_distribution.png', dpi=150)
plt.close()

unique_m, counts_m = np.unique(y_multi.numpy(), return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(range(len(unique_m)), counts_m, color='steelblue')
plt.xticks(range(len(unique_m)), attack_names, rotation=45, ha='right')
plt.title('Multiclass Attack Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('report/images/multiclass_class_distribution.png', dpi=150)
plt.close()

# Training curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Binary Train', marker='o')
plt.plot(mc_losses, label='Multi Train', marker='s')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/training_curves.png', dpi=150)
plt.close()

# Save summaries
with open('outputs/data_overview.json', 'w') as f:
    json.dump({
        'num_samples': n_samples,
        'num_features': 40,
        'binary_classes': 2,
        'multiclass_classes': 10
    }, f, indent=2)

with open('outputs/summary.json', 'w') as f:
    json.dump({
        'binary': {'accuracy': acc_bin, 'f1': f1_bin, 'roc_auc': roc_auc},
        'multiclass': {'accuracy': acc_mc, 'f1_weighted': f1_mc, 'per_class_f1': per_class}
    }, f, indent=2)

print('\n=== Training Complete ===')
print('Results saved to outputs/')
print('Figures saved to report/images/')
