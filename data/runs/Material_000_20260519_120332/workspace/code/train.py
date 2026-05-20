"""
Altermagnetic Material Discovery via Graph Neural Networks
- Self-supervised pre-training with contrastive learning
- Fine-tuning for binary classification
- Candidate screening and evaluation
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets():
    pretrain = torch.load('data/pretrain_data.pt', map_location='cpu', weights_only=False)
    finetune = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
    candidate = torch.load('data/candidate_data.pt', map_location='cpu', weights_only=False)
    return pretrain.data_list, finetune.data_list, candidate.data_list

# ---------------------------------------------------------------------------
# Augmentations for contrastive learning
# ---------------------------------------------------------------------------

def augment_node_mask(data, mask_ratio=0.15):
    """Randomly mask node features."""
    x = data.x.clone()
    num_nodes = x.size(0)
    num_mask = max(1, int(mask_ratio * num_nodes))
    mask_idx = torch.randperm(num_nodes)[:num_mask]
    x[mask_idx] = 0.0
    new_data = Data(x=x, edge_index=data.edge_index.clone(),
                    edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None)
    if hasattr(data, 'y') and data.y is not None:
        new_data.y = data.y
    return new_data

def augment_edge_drop(data, drop_ratio=0.15):
    """Randomly drop edges."""
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone() if data.edge_attr is not None else None
    num_edges = edge_index.size(1)
    num_keep = max(1, int((1 - drop_ratio) * num_edges))
    keep_idx = torch.randperm(num_edges)[:num_keep]
    edge_index = edge_index[:, keep_idx]
    if edge_attr is not None:
        edge_attr = edge_attr[keep_idx]
    new_data = Data(x=data.x.clone(), edge_index=edge_index, edge_attr=edge_attr)
    if hasattr(data, 'y') and data.y is not None:
        new_data.y = data.y
    return new_data

def augment_graph(data):
    """Compose augmentations."""
    if random.random() < 0.5:
        data = augment_node_mask(data, mask_ratio=0.15)
    if random.random() < 0.5:
        data = augment_edge_drop(data, drop_ratio=0.15)
    return data

# ---------------------------------------------------------------------------
# Model: GNN Encoder + Classifier
# ---------------------------------------------------------------------------

class GNN_Encoder(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=128, out_dim=128, num_layers=4, heads=4):
        super().__init__()
        self.num_layers = num_layers
        
        # Edge MLP to project edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initial node embedding
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim
            self.convs.append(GATv2Conv(in_ch, out_ch // heads, heads=heads, edge_dim=hidden_dim, concat=True))
            self.bns.append(nn.BatchNorm1d(out_ch))
        
        # Projector for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, data, return_graph_emb=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Edge embedding
        if edge_attr is not None:
            edge_emb = self.edge_mlp(edge_attr)
        else:
            edge_emb = None
        
        # Node embedding
        x = self.node_emb(x)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = bn(x)
            x = F.relu(x)
        
        # Graph-level pooling
        graph_emb = global_mean_pool(x, batch)
        
        if return_graph_emb:
            return graph_emb
        
        z = self.projector(graph_emb)
        z = F.normalize(z, p=2, dim=1)
        return z

class Classifier(nn.Module):
    def __init__(self, encoder, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        emb = self.encoder(data, return_graph_emb=True)
        out = self.mlp(emb)
        return out.squeeze(-1)

# ---------------------------------------------------------------------------
# NT-Xent loss (Normalized Temperature-scaled Cross Entropy)
# ---------------------------------------------------------------------------

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """Contrastive loss for paired augmented views."""
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]
    
    mask = torch.eye(sim.size(0), device=sim.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    
    pos_sim = torch.cat([
        torch.diag(sim, z_i.size(0)),
        torch.diag(sim, -z_i.size(0))
    ], dim=0)
    
    # Numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    pos_sim = pos_sim - sim_max[:pos_sim.size(0)].detach().squeeze()
    
    numerator = torch.exp(pos_sim)
    denominator = torch.exp(sim).sum(dim=1)
    
    loss = -torch.log(numerator / denominator).mean()
    return loss

# ---------------------------------------------------------------------------
# Focal Loss for class imbalance
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch_pretrain(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        
        # Create two augmented views
        view1 = Batch.from_data_list([augment_graph(d) for d in batch.to_data_list()]).to(device)
        view2 = Batch.from_data_list([augment_graph(d) for d in batch.to_data_list()]).to(device)
        
        z1 = model(view1)
        z2 = model(view2)
        
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = batch.y.float().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_labels.extend(labels)
        all_preds.extend(preds)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        'pr_auc': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
    }
    return metrics, all_probs, all_labels, all_preds

def train_epoch_classifier(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        labels = batch.y.float()
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    
    # Load data
    print("Loading datasets...")
    pretrain_list, finetune_list, candidate_list = load_datasets()
    print(f"Pretrain: {len(pretrain_list)}, Finetune: {len(finetune_list)}, Candidate: {len(candidate_list)}")
    
    # For finetune, we need to convert labels to float for BCE
    for d in finetune_list:
        d.y = d.y.float()
    for d in candidate_list:
        d.y = d.y.float()
    
    # Split finetune into train/val/test
    n = len(finetune_list)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:int(0.7*n)]
    val_idx = indices[int(0.7*n):int(0.85*n)]
    test_idx = indices[int(0.85*n):]
    
    train_data = [finetune_list[i] for i in train_idx]
    val_data = [finetune_list[i] for i in val_idx]
    test_data = [finetune_list[i] for i in test_idx]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Train positives: {sum([d.y.item() for d in train_data])}")
    print(f"Val positives: {sum([d.y.item() for d in val_data])}")
    print(f"Test positives: {sum([d.y.item() for d in test_data])}")
    
    # DataLoaders
    pretrain_loader = PyGDataLoader(pretrain_list, batch_size=64, shuffle=True, num_workers=0)
    train_loader = PyGDataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
    candidate_loader = PyGDataLoader(candidate_list, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    encoder = GNN_Encoder(node_dim=28, edge_dim=2, hidden_dim=128, out_dim=128, num_layers=4, heads=4).to(device)
    
    # ========================
    # Phase 1: Pre-training
    # ========================
    print("\n=== Phase 1: Self-supervised Pre-training ===")
    pretrain_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=50)
    
    pretrain_losses = []
    for epoch in range(1, 51):
        loss = train_epoch_pretrain(encoder, pretrain_loader, pretrain_optimizer, device)
        pretrain_losses.append(loss)
        pretrain_scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss:.4f}")
    
    # Save pretrain loss curve
    plt.figure(figsize=(8,5))
    plt.plot(pretrain_losses)
    plt.xlabel('Epoch')
    plt.ylabel('NT-Xent Loss')
    plt.title('Self-supervised Pre-training Loss')
    plt.grid(True)
    plt.savefig('report/images/pretrain_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save encoder state
    torch.save(encoder.state_dict(), 'outputs/encoder_pretrained.pt')
    
    # ========================
    # Phase 2: Fine-tuning
    # ========================
    print("\n=== Phase 2: Fine-tuning ===")
    classifier = Classifier(encoder, hidden_dim=128).to(device)
    
    # Compute class weights
    pos_weight = (len(train_data) - sum([d.y.item() for d in train_data])) / max(1, sum([d.y.item() for d in train_data]))
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    print(f"Positive weight: {pos_weight:.2f}")
    
    # Use BCEWithLogitsLoss with pos_weight + Focal loss combination
    # We'll use BCEWithLogitsLoss with pos_weight for main training
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Use a smaller LR for fine-tuning encoder, larger for classifier head
    optimizer = torch.optim.Adam([
        {'params': classifier.encoder.parameters(), 'lr': 1e-4},
        {'params': classifier.mlp.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_auc = 0
    best_state = None
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(1, 101):
        train_loss = train_epoch_classifier(classifier, train_loader, optimizer, device, criterion)
        
        if epoch % 5 == 0 or epoch == 1:
            train_metrics, _, _, _ = evaluate_classifier(classifier, train_loader, device)
            val_metrics, val_probs, val_labels, val_preds = evaluate_classifier(classifier, val_loader, device)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                  f"Val Recall: {val_metrics['recall']:.4f}")
            
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                best_state = classifier.state_dict()
                print(f"  --> New best val AUC: {best_val_auc:.4f}")
        
        # Early stopping based on best_val_auc plateau
        if epoch > 20 and epoch % 5 == 0:
            recent_aucs = [m['roc_auc'] for m in val_metrics_history[-4:]]
            if len(recent_aucs) >= 4 and max(recent_aucs) <= best_val_auc and best_val_auc - recent_aucs[-1] > 0.05:
                print("Early stopping triggered.")
                break
    
    # Load best model
    if best_state is not None:
        classifier.load_state_dict(best_state)
    
    # Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    test_metrics, test_probs, test_labels, test_preds = evaluate_classifier(classifier, test_loader, device)
    print(f"Test AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save metrics
    with open('outputs/metrics.json', 'w') as f:
        json.dump({
            'test': test_metrics,
            'val_best_auc': best_val_auc,
            'train_history': train_metrics_history,
            'val_history': val_metrics_history
        }, f, indent=2)
    
    # ========================
    # Phase 3: Candidate screening
    # ========================
    print("\n=== Phase 3: Candidate Screening ===")
    classifier.eval()
    all_candidate_probs = []
    all_candidate_labels = []
    with torch.no_grad():
        for batch in candidate_loader:
            batch = batch.to(device)
            logits = classifier(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch.y.float().cpu().numpy()
            all_candidate_probs.extend(probs)
            all_candidate_labels.extend(labels)
    
    all_candidate_probs = np.array(all_candidate_probs)
    all_candidate_labels = np.array(all_candidate_labels)
    
    # Rank candidates
    top_k = 50
    top_indices = np.argsort(all_candidate_probs)[::-1][:top_k]
    top_probs = all_candidate_probs[top_indices]
    top_labels = all_candidate_labels[top_indices]
    
    discovered = int(top_labels.sum())
    print(f"Top-{top_k} candidates: {discovered} true positives discovered")
    print(f"Precision@50: {discovered/top_k:.4f}")
    print(f"Recall@50 (of hidden positives): {discovered/max(1, all_candidate_labels.sum()):.4f}")
    
    # Save candidate predictions
    candidate_results = {
        'probs': all_candidate_probs.tolist(),
        'labels': all_candidate_labels.tolist(),
        'top50_indices': top_indices.tolist(),
        'top50_probs': top_probs.tolist(),
        'top50_labels': top_labels.tolist(),
        'precision_at_50': discovered/top_k,
        'recall_at_50': discovered/max(1, all_candidate_labels.sum()),
        'num_true_positives_in_candidate': int(all_candidate_labels.sum())
    }
    with open('outputs/candidate_results.json', 'w') as f:
        json.dump(candidate_results, f, indent=2)
    
    # ========================
    # Generate Figures
    # ========================
    print("\n=== Generating Figures ===")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f'Test ROC (AUC = {test_metrics["roc_auc"]:.3f})')
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Altermagnet Classification')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('report/images/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. PR Curve
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_probs)
    plt.figure(figsize=(7,6))
    plt.plot(recall_curve, precision_curve, lw=2, label=f'Test PR (AP = {test_metrics["pr_auc"]:.3f})')
    plt.axhline(y=test_labels.mean(), color='r', linestyle='--', label=f'Baseline ({test_labels.mean():.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('report/images/pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-AM', 'AM'], yticklabels=['Non-AM', 'AM'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig('report/images/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Candidate Score Distribution
    plt.figure(figsize=(8,5))
    bins = np.linspace(0, 1, 31)
    plt.hist(all_candidate_probs[all_candidate_labels==0], bins=bins, alpha=0.6, label='Non-AM', color='steelblue')
    plt.hist(all_candidate_probs[all_candidate_labels==1], bins=bins, alpha=0.6, label='AM (True)', color='crimson')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Candidate Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('report/images/candidate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Top-50 Candidates Bar Chart
    plt.figure(figsize=(12,6))
    colors = ['crimson' if l == 1 else 'steelblue' for l in top_labels]
    plt.bar(range(top_k), top_probs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.xlabel('Candidate Rank')
    plt.ylabel('Predicted Probability')
    plt.title(f'Top-{top_k} Predicted Altermagnets (Red = True Positive)')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('report/images/top_candidates.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Precision@K curve
    k_values = list(range(10, 201, 10))
    precisions_at_k = []
    recalls_at_k = []
    sorted_idx = np.argsort(all_candidate_probs)[::-1]
    for k in k_values:
        prec = all_candidate_labels[sorted_idx[:k]].mean()
        rec = all_candidate_labels[sorted_idx[:k]].sum() / max(1, all_candidate_labels.sum())
        precisions_at_k.append(prec)
        recalls_at_k.append(rec)
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(k_values, precisions_at_k, 'b-o', label='Precision@K')
    ax1.set_xlabel('K (Number of Top Candidates)')
    ax1.set_ylabel('Precision@K', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(k_values, recalls_at_k, 'r-s', label='Recall@K')
    ax2.set_ylabel('Recall@K', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Precision and Recall at K (Candidate Set)')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.9))
    plt.grid(True, alpha=0.3)
    plt.savefig('report/images/precision_recall_at_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. Training curves
    epochs_tracked = [i*5 for i in range(1, len(val_metrics_history)+1)]
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_tracked, [m['roc_auc'] for m in train_metrics_history], 'o-', label='Train AUC')
    plt.plot(epochs_tracked, [m['roc_auc'] for m in val_metrics_history], 's-', label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC During Training')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(epochs_tracked, [m['f1'] for m in train_metrics_history], 'o-', label='Train F1')
    plt.plot(epochs_tracked, [m['f1'] for m in val_metrics_history], 's-', label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score During Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('report/images/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Data distribution overview
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    datasets = [pretrain_list, finetune_list, candidate_list]
    names = ['Pre-train\n(5000)', 'Fine-tune\n(2000)', 'Candidate\n(1000)']
    colors_pie = ['lightgreen', 'lightcoral']
    for ax, ds, name in zip(axes, datasets, names):
        if hasattr(ds[0], 'y') and ds[0].y is not None:
            counts = [sum([d.y.item()==0 for d in ds]), sum([d.y.item()==1 for d in ds])]
            ax.pie(counts, labels=['Non-AM', 'AM'], autopct='%1.1f%%', colors=colors_pie, startangle=90)
        else:
            ax.pie([len(ds)], labels=['Unlabeled'], colors=['lightblue'], startangle=90)
        ax.set_title(name)
    plt.suptitle('Dataset Label Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('report/images/data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save model
    torch.save(classifier.state_dict(), 'outputs/classifier_best.pt')
    
    print("\n=== All done! ===")
    print("Outputs saved to outputs/")
    print("Figures saved to report/images/")

if __name__ == '__main__':
    main()
