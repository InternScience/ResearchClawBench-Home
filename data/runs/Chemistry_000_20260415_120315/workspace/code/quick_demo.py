"""
Quick demonstration of KA-GNN vs MLP on synthetic molecular-like data.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class FourierKAN(nn.Module):
    """Simplified KAN using Fourier basis."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_freqs=4):
        super().__init__()
        self.num_freqs = num_freqs
        self.coeffs = nn.Parameter(torch.randn(hidden_dim, in_dim, num_freqs, 2) * 0.1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x_exp = x.unsqueeze(1).unsqueeze(-1)
        k = torch.arange(1, self.num_freqs + 1, device=x.device).float().view(1, 1, 1, -1)
        args = k * x_exp
        cos_b = torch.cos(args)
        sin_b = torch.sin(args)
        out = (cos_b * self.coeffs[..., 0].unsqueeze(0)).sum(dim=(-2, -1))
        out = out + (sin_b * self.coeffs[..., 1].unsqueeze(0)).sum(dim=(-2, -1))
        out = torch.relu(out + self.bias1)
        return self.fc(out).squeeze(-1)

class SimpleMLP(nn.Module):
    """MLP baseline."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def generate_synthetic_data(n_samples=1000, n_features=20):
    """Generate synthetic molecular-like data."""
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Non-linear relationship (harder for simple models)
    y = (np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * 3) + X[:, 2] ** 2 - X[:, 3] * X[:, 4] > 0).astype(float)
    return X, y

def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=20, lr=0.01):
    """Train and evaluate a model."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_auc': [], 'val_auc': []}
    best_val = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            train_preds, train_labels = [], []
            for X, y in train_loader:
                out = model(X)
                train_preds.extend(torch.sigmoid(out).numpy())
                train_labels.extend(y.numpy())
            train_auc = roc_auc_score(train_labels, train_preds)
            
            val_preds, val_labels = [], []
            for X, y in val_loader:
                out = model(X)
                val_preds.extend(torch.sigmoid(out).numpy())
                val_labels.extend(y.numpy())
            val_auc = roc_auc_score(val_labels, val_preds)
        
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        if val_auc > best_val:
            best_val = val_auc
    
    # Test
    model.eval()
    with torch.no_grad():
        test_preds, test_labels = [], []
        for X, y in test_loader:
            out = model(X)
            test_preds.extend(torch.sigmoid(out).numpy())
            test_labels.extend(y.numpy())
        test_auc = roc_auc_score(test_labels, test_preds)
        test_acc = accuracy_score(test_labels, (np.array(test_preds) > 0.5).astype(int))
    
    return test_auc, test_acc, history, best_val

def run_quick_experiment():
    """Run quick comparison experiment."""
    print("="*60)
    print("Quick Demo: KA-GNN (Fourier) vs MLP")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=2000, n_features=20)
    X_train, X_val, X_test = X[:1400], X[1400:1700], X[1700:]
    y_train, y_val, y_test = y[:1400], y[1400:1700], y[1700:]
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    print(f"\nData: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    print(f"Features: {X.shape[1]}")
    print(f"Positive class: {y.mean()*100:.1f}%\n")
    
    # Train KA-GNN
    print("Training KA-GNN (Fourier-based, num_freqs=4)...")
    model_kan = FourierKAN(20, 32, 1, num_freqs=4)
    auc_kan, acc_kan, hist_kan, best_val_kan = train_and_evaluate(
        model_kan, train_loader, val_loader, test_loader, epochs=30, lr=0.01
    )
    print(f"  Best Val AUC: {best_val_kan:.4f}, Test AUC: {auc_kan:.4f}, Test Acc: {acc_kan:.4f}\n")
    
    # Train MLP
    print("Training MLP baseline...")
    model_mlp = SimpleMLP(20, 32, 1)
    auc_mlp, acc_mlp, hist_mlp, best_val_mlp = train_and_evaluate(
        model_mlp, train_loader, val_loader, test_loader, epochs=30, lr=0.01
    )
    print(f"  Best Val AUC: {best_val_mlp:.4f}, Test AUC: {auc_mlp:.4f}, Test Acc: {acc_mlp:.4f}\n")
    
    # Summary
    print("="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Val AUC':<12} {'Test AUC':<12} {'Test Acc':<12}")
    print("-"*60)
    print(f"{'KA-GNN (Fourier)':<20} {best_val_kan:<12.4f} {auc_kan:<12.4f} {acc_kan:<12.4f}")
    print(f"{'MLP Baseline':<20} {best_val_mlp:<12.4f} {auc_mlp:<12.4f} {acc_mlp:<12.4f}")
    print("="*60)
    print(f"\nImprovement: {(auc_kan - auc_mlp)*100:.2f} percentage points")
    
    # Plot
    os.makedirs('report/images', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(hist_kan['train_auc'], label='KA-GNN', color='blue')
    axes[0].plot(hist_mlp['train_auc'], label='MLP', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train AUC')
    axes[0].set_title('Training Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(hist_kan['val_auc'], label='KA-GNN', color='blue')
    axes[1].plot(hist_mlp['val_auc'], label='MLP', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val AUC')
    axes[1].set_title('Validation Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/quick_demo_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to report/images/quick_demo_results.png")
    
    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    results = {
        'KA-GNN': {'val_auc': float(best_val_kan), 'test_auc': float(auc_kan), 'test_acc': float(acc_kan)},
        'MLP': {'val_auc': float(best_val_mlp), 'test_auc': float(auc_mlp), 'test_acc': float(acc_mlp)}
    }
    with open('outputs/results/quick_demo.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    results = run_quick_experiment()
