import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', weights_only=False)
features = data.msg.numpy()
labels = data.label.numpy()
timestamps = data.t.numpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create temporal splits
n_samples = len(features)
sorted_idx = np.argsort(timestamps)
train_end = int(0.7 * n_samples)
val_end = int(0.8 * n_samples)

train_idx = sorted_idx[:train_end]
val_idx = sorted_idx[train_end:val_end]
test_idx = sorted_idx[val_end:]

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(features[train_idx])
X_val = scaler.transform(features[val_idx])
X_test = scaler.transform(features[test_idx])

# Use binary labels
y_train = labels[train_idx]
y_val = labels[val_idx]
y_test = labels[test_idx]

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Simple MLP baseline
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# Simplified DIDS-MFL
class SimpleDIDS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(128, out_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        attn = self.attention(h)
        h = h * attn.sum(dim=-1, keepdim=True)
        h = self.fusion(h)
        return self.classifier(h)

# Train function
def train_model(model, train_loader, val_loader, name, epochs=15):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_f1 = 0
    history = {'train_loss': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                out = model(X)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        f1 = f1_score(all_labels, all_preds, average='macro')
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_f1'].append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            
        if (epoch + 1) % 5 == 0:
            print(f"{name} Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val F1={f1:.4f}")
    
    return model, history, best_f1

# Train models
n_features = X_train.shape[1]
models_to_train = {
    'MLP': MLP(n_features, 2),
    'DIDS-MFL': SimpleDIDS(n_features, 2)
}

trained_models = {}
histories = {}

for name, model in models_to_train.items():
    print(f"Training {name}...")
    trained, hist, best = train_model(model, train_loader, val_loader, name, epochs=15)
    trained_models[name] = trained
    histories[name] = hist
    print(f"Best Val F1: {best:.4f}")

# Test evaluation
print("\n=== Test Results ===")
test_results = {}
for name, model in trained_models.items():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            out = model(X)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    test_results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    print(f"{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

# Save results
os.makedirs('outputs', exist_ok=True)
with open('outputs/test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, hist in histories.items():
    axes[0].plot(hist['train_loss'], label=name)
    axes[1].plot(hist['val_f1'], label=name)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[1].set_title('Validation F1')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
plt.savefig('report/images/training_curves.png', dpi=300)
plt.close()
print("Saved: report/images/training_curves.png")

# Performance comparison
fig, ax = plt.subplots(figsize=(8, 5))
models = list(test_results.keys())
x = np.arange(len(models))
width = 0.2

metrics = ['accuracy', 'precision', 'recall', 'f1']
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
for i, metric in enumerate(metrics):
    values = [test_results[m][metric] for m in models]
    ax.bar(x + i*width, values, width, label=metric, color=colors[i])

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Performance Comparison (Test Set)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('report/images/performance_comparison.png', dpi=300)
plt.close()
print("Saved: report/images/performance_comparison.png")
print("Done!")
