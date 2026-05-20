"""
GNN-based Altermagnetic Materials Discovery Pipeline v2
- Improved pre-training with contrastive learning
- Better handling of class imbalance
- Enhanced GAT-based architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, BatchNorm, GraphNorm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import json
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Model Definitions
# ============================================================

class GATEncoder(nn.Module):
    """Graph Attention Network encoder with residual connections."""
    def __init__(self, in_dim=28, hidden_dim=64, out_dim=64, num_layers=3, heads=4, dropout=0.15):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        for i in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
            self.norms.append(GraphNorm(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual.shape == x.shape:
                x = x + residual
        x = self.out_proj(x)
        if batch is not None:
            graph_emb = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        else:
            graph_emb = torch.cat([x.mean(dim=0, keepdim=True), x.max(dim=0, keepdim=True).values], dim=-1)
        return graph_emb, x


class GCNEncoder(nn.Module):
    """GCN encoder for comparison."""
    def __init__(self, in_dim=28, hidden_dim=64, out_dim=64, num_layers=3, dropout=0.15):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(GraphNorm(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual.shape == x.shape:
                x = x + residual
        x = self.out_proj(x)
        if batch is not None:
            graph_emb = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        else:
            graph_emb = torch.cat([x.mean(dim=0, keepdim=True), x.max(dim=0, keepdim=True).values], dim=-1)
        return graph_emb, x


# ============================================================
# Pre-training: Contrastive Learning
# ============================================================

def augment_graph(data, noise_std=0.05):
    """Simple graph augmentation: add noise to node features and drop edges."""
    x = data.x.clone()
    x = x + torch.randn_like(x) * noise_std
    edge_index = data.edge_index.clone()
    n_edges = edge_index.shape[1]
    keep_mask = torch.rand(n_edges) > 0.2
    edge_index = edge_index[:, keep_mask]
    return x, edge_index


class ContrastivePretrainModel(nn.Module):
    """Graph contrastive pre-training with augmented views."""
    def __init__(self, encoder, hidden_dim=64, projection_dim=32):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, data):
        x1, ei1 = augment_graph(data)
        x2, ei2 = augment_graph(data)
        batch = data.batch if hasattr(data, 'batch') else None
        
        _, node_emb1 = self.encoder(x1, ei1, batch)
        _, node_emb2 = self.encoder(x2, ei2, batch)
        
        if batch is not None:
            g1 = torch.cat([global_mean_pool(node_emb1, batch), global_max_pool(node_emb1, batch)], dim=-1)
            g2 = torch.cat([global_mean_pool(node_emb2, batch), global_max_pool(node_emb2, batch)], dim=-1)
        else:
            g1 = torch.cat([node_emb1.mean(0, keepdim=True), node_emb1.max(0).values.unsqueeze(0)], dim=-1)
            g2 = torch.cat([node_emb2.mean(0, keepdim=True), node_emb2.max(0).values.unsqueeze(0)], dim=-1)
        
        z1 = F.normalize(self.projector(g1), dim=-1)
        z2 = F.normalize(self.projector(g2), dim=-1)
        
        sim = torch.mm(z1, z2.t()) / self.temperature.clamp(min=0.01)
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


# ============================================================
# Classification Head
# ============================================================

class Classifier(nn.Module):
    """Classification head for altermagnet prediction."""
    def __init__(self, encoder, hidden_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        graph_emb, node_emb = self.encoder(x, edge_index, batch)
        logits = self.head(graph_emb)
        return logits, graph_emb


def collate_fn(batch):
    return Batch.from_data_list(batch)


# ============================================================
# Training Functions
# ============================================================

def contrastive_pretrain_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


def finetune_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        if class_weights is not None:
            loss = F.cross_entropy(logits, batch.y.view(-1), weight=class_weights)
        else:
            loss = F.cross_entropy(logits, batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch.y.view(-1)).sum().item()
        total += batch.num_graphs
        total_loss += loss.item() * batch.num_graphs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_graph_embs = []
    for batch in loader:
        batch = batch.to(device)
        logits, graph_emb = model(batch)
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())
        all_graph_embs.append(graph_emb.cpu().numpy())
    return np.array(all_probs), np.array(all_labels), np.concatenate(all_graph_embs, axis=0)


# ============================================================
# Oversampled DataLoader
# ============================================================
class OversampledDataset(torch.utils.data.Dataset):
    """Oversample minority class for balanced training."""
    def __init__(self, data_list, labels):
        self.data = data_list
        self.labels = labels
        pos_idx = [i for i, l in enumerate(labels) if l == 1]
        neg_idx = [i for i, l in enumerate(labels) if l == 0]
        n_neg = len(neg_idx)
        n_pos = len(pos_idx)
        oversample = n_neg // max(n_pos, 1)
        remainder = n_neg % max(n_pos, 1)
        self.indices = neg_idx.copy()
        for _ in range(oversample):
            self.indices.extend(pos_idx)
        if remainder > 0:
            self.indices.extend(pos_idx[:remainder])
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.data[self.indices[idx]]


# ============================================================
# Main Pipeline
# ============================================================

def main():
    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                  confusion_matrix, classification_report,
                                  average_precision_score)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    pretrain_data = torch.load('data/pretrain_data.pt', map_location='cpu', weights_only=False)
    finetune_data = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
    candidate_data = torch.load('data/candidate_data.pt', map_location='cpu', weights_only=False)
    
    ft_labels = [finetune_data[i].y.item() for i in range(len(finetune_data))]
    cd_labels = [candidate_data[i].y.item() for i in range(len(candidate_data))]
    
    np.random.seed(42)
    indices = list(range(len(finetune_data)))
    np.random.shuffle(indices)
    
    pos_idx = [i for i in indices if ft_labels[i] == 1]
    neg_idx = [i for i in indices if ft_labels[i] == 0]
    
    n_val_pos = min(10, len(pos_idx))
    n_val_neg = 190
    val_idx = pos_idx[:n_val_pos] + neg_idx[:n_val_neg]
    train_idx = [i for i in indices if i not in set(val_idx)]
    
    train_dataset = [finetune_data[i] for i in train_idx]
    train_labels = [ft_labels[i] for i in train_idx]
    val_dataset = [finetune_data[i] for i in val_idx]
    
    print(f"Train: {len(train_dataset)} (pos={sum(train_labels)}, neg={len(train_labels)-sum(train_labels)})")
    print(f"Val: {len(val_dataset)} (pos={n_val_pos}, neg={n_val_neg})")
    
    BATCH_SIZE = 64
    oversampled = OversampledDataset(train_dataset, train_labels)
    train_loader = DataLoader(oversampled, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    candidate_loader = DataLoader([candidate_data[i] for i in range(len(candidate_data))],
                                   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    HIDDEN_DIM = 64
    
    # ============================================
    # Phase 1: Contrastive Pre-training
    # ============================================
    print("\n" + "="*60)
    print("PHASE 1: Contrastive Self-Supervised Pre-training")
    print("="*60)
    
    all_pretrain = [pretrain_data[i] for i in range(len(pretrain_data))]
    for i in range(len(finetune_data)):
        all_pretrain.append(finetune_data[i])
    pretrain_loader = DataLoader(all_pretrain, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    encoder = GATEncoder(in_dim=28, hidden_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, num_layers=3, heads=4, dropout=0.15)
    pretrain_model = ContrastivePretrainModel(encoder, hidden_dim=HIDDEN_DIM, projection_dim=32).to(device)
    
    optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    pretrain_losses = []
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(20):
        loss = contrastive_pretrain_epoch(pretrain_model, pretrain_loader, optimizer, device)
        scheduler.step()
        pretrain_losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.cpu().clone() for k, v in pretrain_model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20, Loss: {loss:.4f} (best: {best_loss:.4f})")
    
    pretrain_model.load_state_dict(best_state)
    print(f"  Pre-training complete. Best loss: {best_loss:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 21), pretrain_losses, 'b-o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Contrastive Loss')
    ax.set_title('Self-Supervised Contrastive Pre-training Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig4_pretrain_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_pretrain_loss.png")
    
    # ============================================
    # Phase 2: Supervised Fine-tuning
    # ============================================
    print("\n" + "="*60)
    print("PHASE 2: Supervised Fine-tuning")
    print("="*60)
    
    classifier = Classifier(encoder, hidden_dim=HIDDEN_DIM, num_classes=2, dropout=0.3).to(device)
    
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    train_losses, val_losses, train_accs, val_accs, val_aucs = [], [], [], [], []
    best_val_auc = 0
    best_cls_state = None
    patience = 25
    no_improve = 0
    
    for epoch in range(80):
        cls_loss, cls_acc = finetune_epoch(classifier, train_loader, optimizer, device, class_weights=class_weights)
        train_losses.append(cls_loss)
        train_accs.append(cls_acc)
        
        val_probs, val_labels, _ = evaluate(classifier, val_loader, device)
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        val_auc = auc(fpr, tpr)
        val_aucs.append(val_auc)
        val_pred = (val_probs > 0.5).astype(int)
        val_acc = (val_pred == val_labels).mean()
        val_accs.append(val_acc)
        val_loss = -np.mean(val_labels * np.log(val_probs + 1e-8) + (1 - val_labels) * np.log(1 - val_probs + 1e-8))
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_cls_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/80, Train Loss: {cls_loss:.4f}, Val AUC: {val_auc:.4f} (best: {best_val_auc:.4f}), Val Acc: {val_acc:.4f}")
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    classifier.load_state_dict(best_cls_state)
    print(f"  Fine-tuning complete. Best val AUC: {best_val_auc:.4f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train')
    axes[0].plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Fine-tuning Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, len(train_accs)+1), train_accs, 'b-', label='Train')
    axes[1].plot(range(1, len(val_accs)+1), val_accs, 'r-', label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Fine-tuning Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(range(1, len(val_aucs)+1), val_aucs, 'g-', label='Val ROC AUC')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUC'); axes[2].set_title('Validation ROC AUC')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig5_finetune_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig5_finetune_curves.png")
    
    # ============================================
    # Phase 3: Validation Evaluation
    # ============================================
    print("\n" + "="*60)
    print("PHASE 3: Validation Evaluation")
    print("="*60)
    
    val_probs, val_labels, val_embs = evaluate(classifier, val_loader, device)
    val_pred = (val_probs > 0.5).astype(int)
    print(classification_report(val_labels, val_pred, target_names=['Non-Altermagnet', 'Altermagnet'], zero_division=0))
    
    fpr, tpr, roc_thresholds = roc_curve(val_labels, val_probs)
    roc_auc_val = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(val_labels, val_probs)
    pr_auc_val = average_precision_score(val_labels, val_probs)
    cm = confusion_matrix(val_labels, val_pred)
    
    print(f"  ROC AUC: {roc_auc_val:.4f}")
    print(f"  PR AUC: {pr_auc_val:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC AUC = {roc_auc_val:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve (Validation)'); axes[0].legend(fontsize=12); axes[0].grid(True, alpha=0.3)
    
    baseline = sum(val_labels) / len(val_labels)
    axes[1].plot(recall, precision, 'r-', linewidth=2, label=f'PR AUC = {pr_auc_val:.4f}')
    axes[1].axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline = {baseline:.3f}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve (Validation)'); axes[1].legend(fontsize=12); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig6_roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig6_roc_pr_curves.png")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-AM', 'Altermagnet'], yticklabels=['Non-AM', 'Altermagnet'],
                ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Validation)', fontsize=14)
    plt.tight_layout()
    plt.savefig('report/images/fig7_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig7_confusion_matrix.png")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(val_probs[val_labels == 0], bins=20, alpha=0.6, label='Non-AM', color='steelblue', edgecolor='black')
    if sum(val_labels == 1) > 0:
        ax.hist(val_probs[val_labels == 1], bins=20, alpha=0.6, label='Altermagnet', color='coral', edgecolor='black')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Boundary')
    ax.set_xlabel('Predicted Probability (Altermagnet)'); ax.set_ylabel('Count')
    ax.set_title('Prediction Probability Distribution (Validation)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig8_prob_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig8_prob_distribution.png")
    
    # ============================================
    # Phase 4: Candidate Prediction
    # ============================================
    print("\n" + "="*60)
    print("PHASE 4: Candidate Material Prediction")
    print("="*60)
    
    cand_probs, cand_true_labels, cand_embs = evaluate(classifier, candidate_loader, device)
    ranked_indices = np.argsort(-cand_probs)
    n_true = int(sum(cand_true_labels))
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
    
    for t in [0.3, 0.4, 0.5, best_threshold]:
        cp = (cand_probs >= t).astype(int)
        n_pred = int(cp.sum())
        n_tp = int((cp * cand_true_labels).sum())
        print(f"  Threshold={t:.3f}: predicted={n_pred}, TP={n_tp}, Precision={n_tp/max(n_pred,1):.3f}, Recall={n_tp/max(n_true,1):.3f}")
    
    cand_pred = (cand_probs >= best_threshold).astype(int)
    n_discovered = int(cand_pred.sum())
    n_correct = int((cand_pred * cand_true_labels).sum())
    
    print(f"\n  Best threshold: {best_threshold:.4f}")
    print(f"  Predicted altermagnets: {n_discovered}")
    print(f"  True positives found: {n_correct}/{n_true}")
    
    top50 = ranked_indices[:50]
    top50_tp = int(sum(cand_true_labels[top50]))
    print(f"\n  Top-50: TP={top50_tp}/{n_true}, Precision={top50_tp/50:.4f}, Recall={top50_tp/n_true:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].hist(cand_probs, bins=50, alpha=0.7, color='teal', edgecolor='black')
    axes[0,0].axvline(x=best_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold={best_threshold:.3f}')
    axes[0,0].set_xlabel('Predicted Probability'); axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Candidate Prediction Distribution'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    
    k_values = list(range(1, 201))
    p_at_k, r_at_k = [], []
    for k in k_values:
        top_k = ranked_indices[:k]
        n_correct_k = sum(cand_true_labels[top_k])
        p_at_k.append(n_correct_k / k)
        r_at_k.append(n_correct_k / n_true)
    
    axes[0,1].plot(k_values, p_at_k, 'b-', linewidth=1.5, label='Precision@K')
    axes[0,1].plot(k_values, r_at_k, 'r-', linewidth=1.5, label='Recall@K')
    axes[0,1].axhline(y=n_true/len(cand_true_labels), color='gray', linestyle='--', label='Baseline')
    axes[0,1].set_xlabel('K'); axes[0,1].set_ylabel('Score')
    axes[0,1].set_title('Precision and Recall @ K'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
    
    prob_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
    bin_labels = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '>0.9']
    bin_counts, bin_tp = [], []
    for i in range(len(prob_bins)-1):
        mask = (cand_probs >= prob_bins[i]) & (cand_probs < prob_bins[i+1])
        bin_counts.append(int(mask.sum()))
        bin_tp.append(int((mask & (cand_true_labels == 1)).sum()))
    
    x = np.arange(len(bin_labels)); w = 0.35
    axes[1,0].bar(x - w/2, bin_counts, w, label='Total', color='steelblue', edgecolor='black')
    axes[1,0].bar(x + w/2, bin_tp, w, label='True Positive', color='coral', edgecolor='black')
    axes[1,0].set_xlabel('Probability Range'); axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Predictions by Probability Range')
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(bin_labels, rotation=30)
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3, axis='y')
    
    cum_tp = np.cumsum([cand_true_labels[idx] for idx in ranked_indices])
    cum_prec = cum_tp / np.arange(1, len(cum_tp) + 1)
    axes[1,1].plot(range(1, min(201, len(cum_prec)+1)), cum_prec[:200], 'b-', linewidth=1.5)
    axes[1,1].axhline(y=n_true/len(cand_true_labels), color='red', linestyle='--', label='Random baseline')
    axes[1,1].set_xlabel('K'); axes[1,1].set_ylabel('Precision@K')
    axes[1,1].set_title('Cumulative Precision (Top-200)'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig9_candidate_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig9_candidate_analysis.png")
    
    # ============================================
    # Phase 5: Feature Importance
    # ============================================
    print("\n" + "="*60)
    print("PHASE 5: Feature Importance Analysis")
    print("="*60)
    
    classifier.eval()
    feature_importance = torch.zeros(28)
    n_count = 0
    
    for batch in candidate_loader:
        batch = batch.to(device)
        batch.x = batch.x.detach().requires_grad_(True)
        logits, _ = classifier(batch)
        logits[:, 1].sum().backward(retain_graph=True)
        if batch.x.grad is not None:
            importance = batch.x.grad.abs().mean(dim=0).cpu()
            feature_importance += importance
            n_count += 1
    
    feature_importance /= max(n_count, 1)
    feature_importance_norm = feature_importance / (feature_importance.max() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(28), feature_importance_norm.numpy(), color='steelblue', edgecolor='black')
    top_features = feature_importance_norm.topk(5).indices.numpy()
    for idx in top_features:
        bars[idx].set_color('coral')
    ax.set_xlabel('Node Feature Index'); ax.set_ylabel('Importance (Normalized)')
    ax.set_title('Feature Importance (Gradient-based Attribution)')
    ax.set_xticks(range(28)); ax.set_xticklabels(range(28))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('report/images/fig10_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig10_feature_importance.png")
    print(f"  Top 5 features: {top_features.tolist()}")
    
    # ============================================
    # Phase 6: Embedding Space Visualization
    # ============================================
    print("\n" + "="*60)
    print("PHASE 6: Embedding Space Visualization")
    print("="*60)
    
    from sklearn.manifold import TSNE
    
    if len(cand_embs) > 10:
        perplexity = min(30, len(cand_embs) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        emb_2d = tsne.fit_transform(cand_embs)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = ['#4a90d9', '#e74c3c']
        for label in [0, 1]:
            mask = cand_true_labels == label
            name = 'Non-AM' if label == 0 else 'Altermagnet'
            axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=colors[label], label=name,
                           alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
        axes[0].set_xlabel('t-SNE Dimension 1'); axes[0].set_ylabel('t-SNE Dimension 2')
        axes[0].set_title('Candidate Embeddings (True Labels)'); axes[0].legend()
        
        sc = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=cand_probs, cmap='RdYlBu_r',
                            alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
        plt.colorbar(sc, ax=axes[1], label='P(Altermagnet)')
        axes[1].set_xlabel('t-SNE Dimension 1'); axes[1].set_ylabel('t-SNE Dimension 2')
        axes[1].set_title('Candidate Embeddings (Predicted Probability)')
        
        plt.tight_layout()
        plt.savefig('report/images/fig11_embedding_tsne.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved fig11_embedding_tsne.png")
    
    # ============================================
    # Phase 7: Model Comparison
    # ============================================
    print("\n" + "="*60)
    print("PHASE 7: Model Comparison (GCN vs GAT)")
    print("="*60)
    
    gcn_encoder = GCNEncoder(in_dim=28, hidden_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, num_layers=3, dropout=0.15)
    gcn_pretrain = ContrastivePretrainModel(gcn_encoder, hidden_dim=HIDDEN_DIM, projection_dim=32).to(device)
    opt_gcn = torch.optim.AdamW(gcn_pretrain.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(10):
        contrastive_pretrain_epoch(gcn_pretrain, pretrain_loader, opt_gcn, device)
    
    gcn_cls = Classifier(gcn_encoder, hidden_dim=HIDDEN_DIM, num_classes=2, dropout=0.3).to(device)
    opt_gcn_cls = torch.optim.AdamW(gcn_cls.parameters(), lr=1e-3, weight_decay=1e-3)
    sched_gcn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gcn_cls, T_max=50)
    
    gcn_best_auc = 0; gcn_best_state = None; gcn_val_aucs = []
    
    for epoch in range(50):
        finetune_epoch(gcn_cls, train_loader, opt_gcn_cls, device, class_weights=class_weights)
        sched_gcn.step()
        vp, vl, _ = evaluate(gcn_cls, val_loader, device)
        fpr_g, tpr_g, _ = roc_curve(vl, vp)
        gcn_auc = auc(fpr_g, tpr_g)
        gcn_val_aucs.append(gcn_auc)
        if gcn_auc > gcn_best_auc:
            gcn_best_auc = gcn_auc
            gcn_best_state = {k: v.cpu().clone() for k, v in gcn_cls.state_dict().items()}
    
    gcn_cls.load_state_dict(gcn_best_state)
    
    gat_val_probs, gat_val_labels, _ = evaluate(classifier, val_loader, device)
    gcn_val_probs, gcn_val_labels, _ = evaluate(gcn_cls, val_loader, device)
    
    fpr_gat, tpr_gat, _ = roc_curve(gat_val_labels, gat_val_probs)
    gat_roc = auc(fpr_gat, tpr_gat)
    fpr_gcn, tpr_gcn, _ = roc_curve(gcn_val_labels, gcn_val_probs)
    gcn_roc = auc(fpr_gcn, tpr_gcn)
    
    print(f"  GAT ROC AUC: {gat_roc:.4f}")
    print(f"  GCN ROC AUC: {gcn_roc:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr_gat, tpr_gat, 'b-', linewidth=2, label=f'GAT (AUC={gat_roc:.4f})')
    axes[0].plot(fpr_gcn, tpr_gcn, 'r--', linewidth=2, label=f'GCN (AUC={gcn_roc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve: GAT vs GCN'); axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, len(val_aucs)+1), val_aucs, 'b-', linewidth=1.5, label='GAT')
    axes[1].plot(range(1, len(gcn_val_aucs)+1), gcn_val_aucs, 'r--', linewidth=1.5, label='GCN')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation ROC AUC')
    axes[1].set_title('Training Progress: GAT vs GCN'); axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig12_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig12_model_comparison.png")
    
    # ============================================
    # Save Results
    # ============================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    results = {
        'pretrain': {'method': 'Contrastive Learning', 'best_loss': float(best_loss), 'epochs': 20},
        'finetune': {'gat_best_val_auc': float(best_val_auc), 'gcn_best_val_auc': float(gcn_best_auc),
                     'train_size': len(train_dataset), 'val_size': len(val_dataset), 'oversampled_size': len(oversampled)},
        'validation': {'gat_roc_auc': float(gat_roc), 'gat_pr_auc': float(pr_auc_val),
                       'gcn_roc_auc': float(gcn_roc), 'confusion_matrix': cm.tolist(), 'best_threshold': float(best_threshold)},
        'candidate_predictions': {'total_candidates': len(cand_probs), 'threshold_used': float(best_threshold),
                                  'predicted_altermagnets': n_discovered, 'true_positives_found': n_correct,
                                  'true_positives_total': n_true, 'discovery_precision': float(n_correct / max(n_discovered, 1)),
                                  'discovery_recall': float(n_correct / n_true), 'top50_precision': float(top50_tp / 50),
                                  'top50_recall': float(top50_tp / n_true)},
        'ranking': {'ranked_indices': ranked_indices.tolist(), 'ranked_probs': cand_probs[ranked_indices].tolist(),
                    'ranked_true_labels': cand_true_labels[ranked_indices].tolist()},
        'feature_importance': feature_importance_norm.numpy().tolist(),
        'top_features': top_features.tolist()
    }
    
    with open('outputs/full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved full_results.json")
    
    predictions = []
    for i, idx in enumerate(ranked_indices):
        predictions.append({
            'rank': i + 1, 'candidate_index': int(idx), 'probability': float(cand_probs[idx]),
            'true_label': int(cand_true_labels[idx]),
            'predicted_label': 1 if cand_probs[idx] >= best_threshold else 0
        })
    with open('outputs/candidate_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    print("  Saved candidate_predictions.json")
    
    print("\n  === TOP 50 CANDIDATE ALTERNMAGNETS ===")
    for i in range(min(50, len(ranked_indices))):
        idx = ranked_indices[i]
        prob = cand_probs[idx]
        true = cand_true_labels[idx]
        marker = "✓" if true == 1 else "✗"
        if i < 10 or true == 1:
            print(f"  Rank {i+1:3d}: Candidate {idx:4d}, P(AM)={prob:.4f}, True={true} {marker}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    return results

if __name__ == '__main__':
    results = main()
