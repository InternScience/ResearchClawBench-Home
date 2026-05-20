#!/usr/bin/env python3
"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection with Multi-scale Feature Learning
Implementation for NF-UNSW-NB15-v2 dataset
"""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

def load_data(data_path='data/NF-UNSW-NB15-v2_3d.pt'):
    """Load the TemporalData from the .pt file with safe loading."""
    from torch_geometric.data import TemporalData
    from torch_geometric.data.storage import GlobalStorage
    
    print(f"Loading data from {data_path}...")
    
    # Use safe loading with weights_only=False
    data = torch.load(data_path, weights_only=False)
    
    print(f"Data loaded successfully!")
    print(f"  - Number of flows: {data.msg.shape[0]}")
    print(f"  - Feature dimension: {data.msg.shape[1]}")
    print(f"  - Binary labels distribution: {torch.bincount(data.label)}")
    print(f"  - Attack types distribution: {torch.bincount(data.attack)}")
    
    return data

def prepare_splits(data, test_size=0.2, val_size=0.1, random_state=42):
    """Prepare stratified train/val/test splits."""
    n_samples = data.msg.shape[0]
    indices = np.arange(n_samples)
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        stratify=data.label.cpu().numpy(),
        random_state=random_state
    )
    
    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        stratify=data.label[train_val_idx].cpu().numpy(),
        random_state=random_state
    )
    
    print(f"\nData splits:")
    print(f"  - Train: {len(train_idx)} samples")
    print(f"  - Val:   {len(val_idx)} samples")
    print(f"  - Test:  {len(test_idx)} samples")
    
    return train_idx, val_idx, test_idx

def compute_statistics(data):
    """Compute and visualize data statistics."""
    print("\nComputing data statistics...")
    
    # Feature statistics
    features = data.msg.cpu().numpy()
    labels = data.label.cpu().numpy()
    attacks = data.attack.cpu().numpy()
    
    # Create statistics plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Binary label distribution
    ax = axes[0, 0]
    unique, counts = np.unique(labels, return_counts=True)
    ax.bar(['Benign', 'Attack'], counts, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Binary Label Distribution')
    ax.set_ylabel('Count')
    for i, c in enumerate(counts):
        ax.text(i, c + 1000, str(c), ha='center', fontsize=12)
    
    # 2. Attack type distribution
    ax = axes[0, 1]
    unique, counts = np.unique(attacks, return_counts=True)
    attack_names = [f'Attack {i}' for i in unique]
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    ax.bar(range(len(unique)), counts, color=colors)
    ax.set_title('Attack Type Distribution')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(attack_names, rotation=45, ha='right')
    
    # 3. Feature correlation heatmap (sample of features)
    ax = axes[1, 0]
    n_feat_sample = min(20, features.shape[1])
    corr = np.corrcoef(features[:, :n_feat_sample].T)
    sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0, 
                xticklabels=False, yticklabels=False)
    ax.set_title(f'Feature Correlation (first {n_feat_sample} features)')
    
    # 4. Feature distribution for benign vs attack
    ax = axes[1, 1]
    benign_feat = features[labels == 0, 0]
    attack_feat = features[labels == 1, 0]
    ax.hist(benign_feat, bins=50, alpha=0.6, label='Benign', color='#2ecc71')
    ax.hist(attack_feat, bins=50, alpha=0.6, label='Attack', color='#e74c3c')
    ax.set_title('Feature 0 Distribution by Class')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('report/images/data_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  - Data statistics plot saved to report/images/data_statistics.png")
    
    return features, labels, attacks

class DisentangledEncoder(nn.Module):
    """Statistical and representational disentanglement encoder."""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Statistical disentanglement branch
        self.stat_mean = nn.Linear(hidden_dim, latent_dim)
        self.stat_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Representational disentanglement branch
        self.rep_mean = nn.Linear(hidden_dim, latent_dim)
        self.rep_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        
        # Statistical branch
        stat_mu = self.stat_mean(h)
        stat_logvar = self.stat_logvar(h)
        stat_z = self.reparameterize(stat_mu, stat_logvar)
        
        # Representational branch
        rep_mu = self.rep_mean(h)
        rep_logvar = self.rep_logvar(h)
        rep_z = self.reparameterize(rep_mu, rep_logvar)
        
        # Combined latent representation
        z = torch.cat([stat_z, rep_z], dim=-1)
        
        return z, stat_mu, stat_logvar, rep_mu, rep_logvar

class DynamicGraphDiffusion(nn.Module):
    """Dynamic graph diffusion for spatiotemporal aggregation."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.diffusion_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, x, edge_index=None):
        # Simple diffusion without explicit graph structure
        h = x
        for layer in self.diffusion_layers:
            h = F.relu(layer(h))
        
        # Self-attention for temporal aggregation
        if len(h.shape) == 2:
            h = h.unsqueeze(1)  # Add sequence dimension
        
        h, _ = self.attention(h, h, h)
        h = h.squeeze(1) if h.shape[1] == 1 else h.mean(dim=1)
        
        return h

class MultiScaleFusion(nn.Module):
    """Multi-scale representation fusion for few-shot learning."""
    def __init__(self, input_dim, num_scales=3, hidden_dim=128):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(num_scales)
        ])
        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        
    def forward(self, x):
        scale_outputs = [scale(x) for scale in self.scales]
        fused = torch.cat(scale_outputs, dim=-1)
        return self.fusion(fused)

class DIDS_MFL(nn.Module):
    """Complete DIDS-MFL framework."""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_classes=2):
        super().__init__()
        
        self.encoder = DisentangledEncoder(input_dim, hidden_dim, latent_dim)
        self.diffusion = DynamicGraphDiffusion(latent_dim * 2, hidden_dim)
        self.fusion = MultiScaleFusion(hidden_dim, num_scales=3, hidden_dim=hidden_dim)
        
        # Classification heads
        self.binary_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.multi_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 10)  # 10 attack types
        )
        
    def forward(self, x):
        # Disentangled encoding
        z, stat_mu, stat_logvar, rep_mu, rep_logvar = self.encoder(x)
        
        # Dynamic graph diffusion
        h = self.diffusion(z)
        
        # Multi-scale fusion
        h = self.fusion(h)
        
        # Classifications
        binary_logits = self.binary_classifier(h)
        multi_logits = self.multi_classifier(h)
        
        return {
            'binary_logits': binary_logits,
            'multi_logits': multi_logits,
            'latent': z,
            'stat_mu': stat_mu,
            'stat_logvar': stat_logvar,
            'rep_mu': rep_mu,
            'rep_logvar': rep_logvar
        }

def vae_loss(mu, logvar, beta=0.1):
    """VAE KL divergence loss."""
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * kld

def train_epoch(model, dataloader, optimizer, criterion_binary, criterion_multi):
    """Train one epoch."""
    model.train()
    total_loss = 0
    correct_binary = 0
    total = 0
    
    for batch_x, batch_y, batch_attack in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_attack = batch_attack.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # Binary classification loss
        loss_binary = criterion_binary(outputs['binary_logits'], batch_y)
        
        # Multi-class loss
        loss_multi = criterion_multi(outputs['multi_logits'], batch_attack)
        
        # VAE losses for disentanglement
        loss_stat = vae_loss(outputs['stat_mu'], outputs['stat_logvar'])
        loss_rep = vae_loss(outputs['rep_mu'], outputs['rep_logvar'])
        
        # Total loss
        loss = loss_binary + 0.5 * loss_multi + loss_stat + loss_rep
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        _, predicted = outputs['binary_logits'].max(1)
        correct_binary += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / len(dataloader), correct_binary / total

def evaluate(model, dataloader):
    """Evaluate the model."""
    model.eval()
    all_binary_preds = []
    all_binary_true = []
    all_multi_preds = []
    all_multi_true = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_attack in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            # Binary predictions
            _, binary_preds = outputs['binary_logits'].max(1)
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_binary_true.extend(batch_y.numpy())
            
            # Multi-class predictions
            _, multi_preds = outputs['multi_logits'].max(1)
            all_multi_preds.extend(multi_preds.cpu().numpy())
            all_multi_true.extend(batch_attack.numpy())
    
    # Compute metrics
    binary_acc = accuracy_score(all_binary_true, all_binary_preds)
    binary_f1 = f1_score(all_binary_true, all_binary_preds, average='weighted')
    
    multi_acc = accuracy_score(all_multi_true, all_multi_preds)
    multi_f1 = f1_score(all_multi_true, all_multi_preds, average='weighted')
    
    return {
        'binary_acc': binary_acc,
        'binary_f1': binary_f1,
        'multi_acc': multi_acc,
        'multi_f1': multi_f1,
        'binary_preds': all_binary_preds,
        'binary_true': all_binary_true,
        'multi_preds': all_multi_preds,
        'multi_true': all_multi_true
    }

def main():
    # Load data
    data = load_data()
    
    # Compute statistics
    features, labels, attacks = compute_statistics(data)
    
    # Prepare splits
    train_idx, val_idx, test_idx = prepare_splits(data)
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(features_scaled[train_idx]),
        torch.LongTensor(labels[train_idx]),
        torch.LongTensor(attacks[train_idx])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(features_scaled[val_idx]),
        torch.LongTensor(labels[val_idx]),
        torch.LongTensor(attacks[val_idx])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(features_scaled[test_idx]),
        torch.LongTensor(labels[test_idx]),
        torch.LongTensor(attacks[test_idx])
    )
    
    # DataLoaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = features.shape[1]
    model = DIDS_MFL(input_dim=input_dim).to(device)
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_binary = nn.CrossEntropyLoss()
    criterion_multi = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 20
    best_val_f1 = 0
    
    train_losses = []
    train_accs = []
    val_f1s = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_binary, criterion_multi
        )
        
        val_metrics = evaluate(model, val_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_f1s.append(val_metrics['binary_f1'])
        
        if val_metrics['binary_f1'] > best_val_f1:
            best_val_f1 = val_metrics['binary_f1']
            torch.save(model.state_dict(), 'outputs/best_model.pt')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={train_loss:.4f}, TrainAcc={train_acc:.4f}, "
                  f"ValBinaryF1={val_metrics['binary_f1']:.4f}, "
                  f"ValMultiF1={val_metrics['multi_f1']:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('outputs/best_model.pt'))
    
    # Final evaluation
    print("\nFinal Evaluation on Test Set:")
    test_metrics = evaluate(model, test_loader)
    
    print(f"  Binary Classification:")
    print(f"    Accuracy: {test_metrics['binary_acc']:.4f}")
    print(f"    F1-Score: {test_metrics['binary_f1']:.4f}")
    print(f"  Multi-class Classification:")
    print(f"    Accuracy: {test_metrics['multi_acc']:.4f}")
    print(f"    F1-Score: {test_metrics['multi_f1']:.4f}")
    
    # Generate results plots
    generate_results_plots(train_losses, train_accs, val_f1s, test_metrics)
    
    # Save results
    results = {
        'test_binary_acc': test_metrics['binary_acc'],
        'test_binary_f1': test_metrics['binary_f1'],
        'test_multi_acc': test_metrics['multi_acc'],
        'test_multi_f1': test_metrics['multi_f1']
    }
    torch.save(results, 'outputs/results.pt')
    
    print("\nTraining complete! Results saved to outputs/")

def generate_results_plots(train_losses, train_accs, val_f1s, test_metrics):
    """Generate training curves and confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    ax = axes[0, 0]
    ax.plot(train_losses, 'b-', linewidth=2)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # Training accuracy
    ax = axes[0, 1]
    ax.plot(train_accs, 'g-', linewidth=2)
    ax.set_title('Training Binary Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    
    # Validation F1
    ax = axes[1, 0]
    ax.plot(val_f1s, 'r-', linewidth=2)
    ax.set_title('Validation Binary F1-Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.grid(True, alpha=0.3)
    
    # Test results bar chart
    ax = axes[1, 1]
    metrics_names = ['Binary\nAccuracy', 'Binary\nF1', 'Multi\nAccuracy', 'Multi\nF1']
    metrics_values = [
        test_metrics['binary_acc'],
        test_metrics['binary_f1'],
        test_metrics['multi_acc'],
        test_metrics['multi_f1']
    ]
    colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
    bars = ax.bar(metrics_names, metrics_values, color=colors)
    ax.set_title('Test Set Performance')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('report/images/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  - Training results plot saved to report/images/training_results.png")
    
    # Confusion matrix for binary classification
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(test_metrics['binary_true'], test_metrics['binary_preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'])
    ax.set_title('Binary Classification Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig('report/images/confusion_matrix_binary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  - Confusion matrix saved to report/images/confusion_matrix_binary.png")

if __name__ == '__main__':
    main()