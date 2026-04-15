"""
Demonstration training script with lightweight implementation for KA-GNN.
Uses simplified data processing for faster execution.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Simple atom features without RDKit dependency for demo
def smiles_to_simple_features(smiles):
    """Extract simple molecular features from SMILES."""
    # Simple feature extraction based on character counts
    features = []
    
    # Atom type counts
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I']
    for atom in atom_types:
        features.append(smiles.count(atom))
    
    # Bond types
    features.append(smiles.count('='))  # Double bonds
    features.append(smiles.count('#'))  # Triple bonds
    features.append(smiles.count('('))  # Branches
    features.append(smiles.count('1') + smiles.count('2'))  # Rings
    
    # Special characters
    features.append(smiles.count('@'))  # Chirality
    features.append(smiles.count('+'))  # Positive charge
    features.append(smiles.count('-'))  # Negative charge
    
    # Length features
    features.append(len(smiles))
    features.append(len(set(smiles)))
    
    return np.array(features, dtype=np.float32)

class SimpleMoleculeDataset(Dataset):
    """Simple dataset using SMILES-derived features."""
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels
        self.features = [smiles_to_simple_features(s) for s in smiles_list]
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'label': torch.FloatTensor([self.labels[idx]]),
            'smiles': self.smiles_list[idx]
        }

def collate_fn(batch):
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {'features': features, 'labels': labels}

class SimpleFourierKAN(nn.Module):
    """Simplified KAN using Fourier basis."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_frequencies=4):
        super().__init__()
        self.num_frequencies = num_frequencies
        
        # Fourier coefficients
        self.coeffs = nn.Parameter(torch.randn(hidden_dim, in_dim, num_frequencies, 2) * 0.1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        # x: (batch, in_dim)
        x_expanded = x.unsqueeze(1).unsqueeze(-1)  # (batch, 1, in_dim, 1)
        
        k = torch.arange(1, self.num_frequencies + 1, device=x.device).float().view(1, 1, 1, -1)
        args = k * x_expanded  # (batch, 1, in_dim, num_freqs)
        
        cos_basis = torch.cos(args)
        sin_basis = torch.sin(args)
        
        cos_coeffs = self.coeffs[..., 0]
        sin_coeffs = self.coeffs[..., 1]
        
        out = (cos_basis * cos_coeffs.unsqueeze(0)).sum(dim=(-2, -1))
        out = out + (sin_basis * sin_coeffs.unsqueeze(0)).sum(dim=(-2, -1))
        out = out + self.bias1
        out = torch.relu(out)
        
        return self.fc(out).squeeze(-1)

class SimpleMLP(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device).squeeze()
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.sigmoid(outputs)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    
    return avg_loss, auc, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device).squeeze()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    
    return avg_loss, auc, acc, all_preds, all_labels

def run_demo_experiment(dataset_name='bace', epochs=30):
    """Run a quick demo experiment."""
    print(f"\n{'='*60}")
    print(f"Demo Experiment: KA-GNN vs MLP on {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    device = 'cpu'
    
    # Load data
    filepath = f'data/{dataset_name}.csv'
    df = pd.read_csv(filepath)
    
    if dataset_name == 'bace':
        smiles_col, label_col = 'smiles', 'label'
    elif dataset_name == 'bbbp':
        smiles_col, label_col = 'smiles', 'label'
    elif dataset_name == 'clintox':
        smiles_col = 'smiles'
        label_col = 'FDA_APPROVED'
    else:
        smiles_col, label_col = 'smiles', 'label'
    
    smiles_list = df[smiles_col].tolist()
    labels = df[label_col].astype(float).values
    
    # Remove invalid entries
    valid_mask = ~np.isnan(labels)
    smiles_list = [s for i, s in enumerate(smiles_list) if valid_mask[i]]
    labels = labels[valid_mask]
    
    print(f"Dataset: {len(smiles_list)} molecules")
    print(f"Positive class: {labels.mean()*100:.1f}%\n")
    
    # Split data
    train_idx, temp_idx = train_test_split(range(len(smiles_list)), test_size=0.3, random_state=42, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    train_smiles = [smiles_list[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_smiles = [smiles_list[i] for i in val_idx]
    val_labels = labels[val_idx]
    test_smiles = [smiles_list[i] for i in test_idx]
    test_labels = labels[test_idx]
    
    # Create datasets
    train_ds = SimpleMoleculeDataset(train_smiles, train_labels)
    val_ds = SimpleMoleculeDataset(val_smiles, val_labels)
    test_ds = SimpleMoleculeDataset(test_smiles, test_labels)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # Feature dimension
    feat_dim = len(train_ds[0]['features'])
    
    results = {}
    
    # Train KA-GNN
    print("Training KA-GNN (Fourier-based)...")
    model_kan = SimpleFourierKAN(feat_dim, 64, 1, num_frequencies=4).to(device)
    optimizer = optim.Adam(model_kan.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    history_kan = {'train_auc': [], 'val_auc': []}
    
    for epoch in range(epochs):
        train_loss, train_auc, _ = train_epoch(model_kan, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _, _ = evaluate(model_kan, val_loader, criterion, device)
        history_kan['train_auc'].append(train_auc)
        history_kan['val_auc'].append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model_kan.state_dict(), 'outputs/kan_best.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train AUC={train_auc:.3f}, Val AUC={val_auc:.3f}")
    
    model_kan.load_state_dict(torch.load('outputs/kan_best.pt'))
    _, test_auc_kan, test_acc_kan, _, _ = evaluate(model_kan, test_loader, criterion, device)
    results['KA-GNN'] = {'test_auc': test_auc_kan, 'test_acc': test_acc_kan, 'history': history_kan}
    print(f"  KA-GNN Test AUC: {test_auc_kan:.4f}\n")
    
    # Train MLP
    print("Training MLP baseline...")
    model_mlp = SimpleMLP(feat_dim, 64, 1).to(device)
    optimizer = optim.Adam(model_mlp.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_val_auc = 0
    history_mlp = {'train_auc': [], 'val_auc': []}
    
    for epoch in range(epochs):
        train_loss, train_auc, _ = train_epoch(model_mlp, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _, _ = evaluate(model_mlp, val_loader, criterion, device)
        history_mlp['train_auc'].append(train_auc)
        history_mlp['val_auc'].append(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model_mlp.state_dict(), 'outputs/mlp_best.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train AUC={train_auc:.3f}, Val AUC={val_auc:.3f}")
    
    model_mlp.load_state_dict(torch.load('outputs/mlp_best.pt'))
    _, test_auc_mlp, test_acc_mlp, _, _ = evaluate(model_mlp, test_loader, criterion, device)
    results['MLP'] = {'test_auc': test_auc_mlp, 'test_acc': test_acc_mlp, 'history': history_mlp}
    print(f"  MLP Test AUC: {test_auc_mlp:.4f}\n")
    
    # Summary
    print("="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"KA-GNN (Fourier): Test AUC = {test_auc_kan:.4f}, Test Acc = {test_acc_kan:.4f}")
    print(f"MLP Baseline:     Test AUC = {test_auc_mlp:.4f}, Test Acc = {test_acc_mlp:.4f}")
    print(f"Improvement:      {(test_auc_kan - test_auc_mlp)*100:.2f} percentage points")
    print("="*60)
    
    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    with open(f'outputs/results/demo_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    results = run_demo_experiment('bace', epochs=30)
