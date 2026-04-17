#!/usr/bin/env python3
"""Quick training script for DIDS-MFL with subsampled data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load and subsample data
print('Loading data...')
data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', weights_only=False)
store = data.stores[0]

# Subsample for quick training
n_samples = 30000
X = store.msg[:n_samples].float()
y_binary = store.label[:n_samples].long()
y_multiclass = store.attack[:n_samples].long()

print(f'Data shape: X={X.shape}')
print(f'Binary distribution: {torch.bincount(y_binary)}')
print(f'Multiclass distribution: {torch.bincount(y_multiclass)}')

# Split
n = len(X)
indices = torch.randperm(n)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train, y_bin_train = X[train_idx], y_binary[train_idx]
X_val, y_bin_val = X[val_idx], y_binary[val_idx]
X_test, y_bin_test = X[test_idx], y_binary[test_idx]

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# Simple NIDS model
class SimpleNIDS(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()

# Binary classification
print('\n=== Binary Classification ===')
model = SimpleNIDS().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

train_dataset = TensorDataset(X_train, y_bin_train)
val_dataset = TensorDataset(X_val, y_bin_val)
test_dataset = TensorDataset(X_test, y_bin_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

train_losses, val_losses = [], []
best_val_loss = float('inf')
best_state = None

for epoch in range(15):
    model.train()
    total_loss = 0
    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            logits = model(X_batch)
            val_loss += criterion(logits, y_batch).item()
    
    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    scheduler.step(avg_val)
    
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_state = model.state_dict().copy()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}')

if best_state:
    model.load_state_dict(best_state)

# Evaluate binary
model.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for batch in test_loader:
        X_batch, y_batch = batch
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.numpy())
        all_probs.extend(probs.numpy())
        all_labels.extend(y_batch.numpy())

acc_bin = accuracy_score(all_labels, all_preds)
f1_bin = f1_score(all_labels, all_preds)
fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs)[:, 1])
roc_auc = auc(fpr, tpr)

print(f'Binary Test - Acc: {acc_bin:.4f}, F1: {f1_bin:.4f}, ROC-AUC: {roc_auc:.4f}')

# Save binary results
with open('outputs/binary_metrics.json', 'w') as f:
    json.dump({'accuracy': acc_bin, 'f1': f1_bin, 'roc_auc': roc_auc}, f, indent=2)

# Plot binary confusion matrix
cm_bin = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
plt.title('Binary Classification Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('report/images/binary_confusion_matrix.png', dpi=150)
plt.close()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Binary Classification ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('report/images/binary_roc_curve.png', dpi=150)
plt.close()

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Binary Classification Training Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/training_curves.png', dpi=150)
plt.close()

# Multiclass classification
print('\n=== Multiclass Classification ===')
X_train_m, y_multi_train = X[train_idx], y_multiclass[train_idx]
X_test_m, y_multi_test = X[test_idx], y_multiclass[test_idx]

model_mc = SimpleNIDS(num_classes=10).to(device)
optimizer_mc = torch.optim.Adam(model_mc.parameters(), lr=0.001)

train_dataset_mc = TensorDataset(X_train_m, y_multi_train)
train_loader_mc = DataLoader(train_dataset_mc, batch_size=256, shuffle=True)

for epoch in range(15):
    model_mc.train()
    total_loss = 0
    for batch in train_loader_mc:
        X_batch, y_batch = batch
        optimizer_mc.zero_grad()
        logits = model_mc(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer_mc.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader_mc):.4f}')

# Evaluate multiclass
model_mc.eval()
all_preds_mc, all_labels_mc = [], []
with torch.no_grad():
    for batch in test_loader:
        X_batch, y_batch = batch
        # Remap to multiclass labels
        y_multi_batch = y_multiclass[test_idx[test_idx >= val_end][:(len(X_batch))]]
        logits = model_mc(X_batch)
        preds = torch.argmax(logits, dim=-1)
        all_preds_mc.extend(preds.numpy())
        all_labels_mc.extend(y_multi_batch.numpy())

acc_mc = accuracy_score(all_labels_mc, all_preds_mc)
f1_mc = f1_score(all_labels_mc, all_preds_mc, average='weighted')

print(f'Multiclass Test - Acc: {acc_mc:.4f}, F1: {f1_mc:.4f}')

# Per-class F1
attack_names = ['Normal', 'DoS', 'Probe', 'U2R', 'R2L', 'DDoS', 'Bot', 'Web', 'Exploit', 'Shellcode']
class_f1 = f1_score(all_labels_mc, all_preds_mc, average=None, zero_division=0)
per_class = {attack_names[i]: float(f1) for i, f1 in enumerate(class_f1)}
print('Per-class F1:', per_class)

with open('outputs/multiclass_metrics.json', 'w') as f:
    json.dump({'accuracy': acc_mc, 'f1_weighted': f1_mc, 'per_class_f1': per_class}, f, indent=2)

# Plot multiclass confusion matrix
cm_mc = confusion_matrix(all_labels_mc, all_preds_mc)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_mc, annot=False, cmap='Blues', xticklabels=attack_names, yticklabels=attack_names)
plt.title('Multiclass Classification Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('report/images/multiclass_confusion_matrix.png', dpi=150)
plt.close()

# Data overview plots
unique_bin, counts_bin = np.unique(np.array(all_labels), return_counts=True)
plt.figure(figsize=(8, 6))
plt.bar(unique_bin.astype(float), counts_bin.astype(float), color='steelblue')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Binary Class Distribution')
plt.xticks([0, 1], ['Benign', 'Attack'])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('report/images/binary_class_distribution.png', dpi=150)
plt.close()

unique_multi, counts_multi = np.unique(np.array(all_labels_mc), return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(unique_multi.astype(float), counts_multi.astype(float), color='steelblue')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Multiclass Attack Type Distribution')
plt.xticks(unique_multi)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('report/images/multiclass_class_distribution.png', dpi=150)
plt.close()

# Save data overview
data_overview = {
    'num_samples': n_samples,
    'num_features': 40,
    'binary_classes': 2,
    'multiclass_classes': 10,
}
with open('outputs/data_overview.json', 'w') as f:
    json.dump(data_overview, f, indent=2)

# Save summary
summary = {
    'binary_metrics': {'accuracy': acc_bin, 'f1': f1_bin, 'roc_auc': roc_auc},
    'multiclass_metrics': {'accuracy': acc_mc, 'f1_weighted': f1_mc, 'per_class_f1': per_class},
}
with open('outputs/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('\n=== Training Complete ===')
print('Results saved to outputs/')
print('Figures saved to report/images/')
