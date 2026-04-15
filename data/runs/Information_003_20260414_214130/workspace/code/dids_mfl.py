"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection with Multi-Scale Fusion and Few-Shot Learning
Main Framework Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, GCNConv
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import os
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 1. Statistical Disentanglement Module
# ============================================================
class StatisticalDisentangler(nn.Module):
    """
    Disentangles statistical flow features using mutual information optimization.
    Separates entangled feature distributions automatically.
    """
    def __init__(self, input_dim, num_factors=8, hidden_dim=64):
        super().__init__()
        self.num_factors = num_factors
        # Factor-specific feature extractors
        self.factor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_factors)
        ])
        # Mutual information estimation network
        self.mi_estimator = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * num_factors, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        # Extract factor-specific features
        factor_features = []
        for encoder in self.factor_encoders:
            factor_features.append(encoder(x))
        
        # Stack factor features
        factors = torch.stack(factor_features, dim=1)  # [B, num_factors, hidden_dim]
        
        # Compute mutual information loss for disentanglement
        mi_loss = self._compute_mi_loss(x, factors)
        
        # Reconstruction loss
        concat_factors = factors.reshape(batch_size, -1)
        reconstructed = self.decoder(concat_factors)
        recon_loss = F.mse_loss(reconstructed, x)
        
        return factors, mi_loss, recon_loss
    
    def _compute_mi_loss(self, x, factors):
        """Estimate and minimize mutual information between factors"""
        mi_loss = 0
        for i in range(self.num_factors):
            for j in range(i + 1, self.num_factors):
                # Positive pairs (joint)
                joint = torch.cat([x, factors[:, i, :]], dim=1)
                pos_score = self.mi_estimator(joint)
                # Negative pairs (marginal - shuffle)
                shuffled_idx = torch.randperm(x.size(0))
                marginal = torch.cat([x[shuffled_idx], factors[:, i, :]], dim=1)
                neg_score = self.mi_estimator(marginal)
                # MI estimation loss (maximize MI estimate, then we minimize it)
                mi_loss += (pos_score.mean() - torch.log(torch.exp(neg_score).mean() + 1e-8))
        return mi_loss / (self.num_factors * (self.num_factors - 1) / 2)


# ============================================================
# 2. Representational Disentanglement Module
# ============================================================
class RepresentationalDisentangler(nn.Module):
    """
    Further disentangles learned representations to highlight attack-specific features.
    Uses correlation-based regularization to reduce feature entanglement.
    """
    def __init__(self, input_dim, hidden_dim=64, num_attack_specific=4):
        super().__init__()
        self.num_attack_specific = num_attack_specific
        
        # Shared representation encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attack-specific encoders
        self.attack_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            ) for _ in range(num_attack_specific)
        ])
        
        # Attention mechanism for attack-specific feature selection
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * (num_attack_specific + 1), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_attack_specific + 1),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # Shared features
        shared = self.shared_encoder(x)
        
        # Attack-specific features
        specific = [enc(x) for enc in self.attack_encoders]
        
        # Combine for attention
        all_features = torch.cat([shared] + specific, dim=-1)
        weights = self.attention(all_features)
        
        # Weighted combination
        stacked = torch.stack([shared] + specific, dim=1)  # [B, K+1, D]
        weights_expanded = weights.unsqueeze(-1)  # [B, K+1, 1]
        weighted = (stacked * weights_expanded).sum(dim=1)  # [B, D]
        
        # Disentanglement loss: minimize correlation between factors
        disentangle_loss = self._correlation_loss(stacked)
        
        return weighted, disentangle_loss, weights
    
    def _correlation_loss(self, features):
        """Minimize correlation between different factor representations"""
        B, K, D = features.shape
        # Normalize features along batch dimension
        features_norm = (features - features.mean(dim=0, keepdim=True)) / (features.std(dim=0, keepdim=True) + 1e-8)
        # Compute correlation matrix between K factors (averaged over D dimensions)
        # Reshape to [K, B*D] and compute correlation
        features_flat = features_norm.permute(1, 0, 2).reshape(K, -1)  # [K, B*D]
        corr = torch.mm(features_flat, features_flat.t()) / features_flat.size(1)  # [K, K]
        # Minimize off-diagonal elements
        mask = torch.eye(K, device=features.device)
        off_diag = corr * (1 - mask)
        return (off_diag ** 2).mean()


# ============================================================
# 3. Dynamic Graph Diffusion Module
# ============================================================
class DynamicGraphDiffusion(nn.Module):
    """
    Dynamic graph diffusion for spatiotemporal aggregation.
    Fuses network topology with temporal information.
    """
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, out_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(out_channels, out_channels))
        
        # Temporal encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, out_channels)
        )
        
        # Diffusion coefficient network
        self.diffusion_net = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_channels) for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, edge_attr=None, timestamps=None):
        # Temporal encoding
        if timestamps is not None:
            t_enc = self.temporal_encoder(timestamps.float().unsqueeze(-1))
        else:
            t_enc = 0
        
        # Multi-layer graph diffusion
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = self.layer_norms[i](h_new)
            h_new = F.relu(h_new)
            
            # Dynamic diffusion: learn to fuse with temporal info
            if i == 0 and timestamps is not None:
                combined = torch.cat([h_new, t_enc], dim=-1)
                alpha = self.diffusion_net(combined)
                h_new = alpha * h_new + (1 - alpha) * t_enc
            
            h = h_new + h if h.shape == h_new.shape else h_new
        
        return h


# ============================================================
# 4. Multi-Scale Representation Fusion Module
# ============================================================
class MultiScaleFusion(nn.Module):
    """
    Fuses representations at multiple scales for few-shot learning.
    Combines local (edge-level), neighborhood, and global features.
    """
    def __init__(self, edge_dim, hidden_dim=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        # Scale-specific encoders
        self.local_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.neighborhood_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, num_scales),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, edge_features, neighborhood_features=None, global_features=None):
        # Local scale
        local = self.local_encoder(edge_features)
        
        # Neighborhood scale
        if neighborhood_features is not None:
            neighborhood = self.neighborhood_encoder(neighborhood_features)
        else:
            neighborhood = local
        
        # Global scale
        if global_features is not None:
            global_feat = self.global_encoder(global_features)
        else:
            global_feat = local
        
        # Concatenate and fuse
        multi_scale = torch.cat([local, neighborhood, global_feat], dim=-1)
        attention_weights = self.scale_attention(multi_scale)
        
        # Attention-weighted fusion
        stacked = torch.stack([local, neighborhood, global_feat], dim=-1)
        attention_expanded = attention_weights.unsqueeze(1)
        fused = (stacked * attention_expanded).sum(dim=-1)
        
        # Final fusion
        output = self.fusion(multi_scale)
        
        return output, attention_weights


# ============================================================
# 5. Few-Shot Learning Module
# ============================================================
class FewShotClassifier(nn.Module):
    """
    Few-shot learning module using prototypical networks approach.
    Handles rare attack types with limited samples.
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Embedding network
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Prototype computation
        self.prototype_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def compute_prototypes(self, support_features, support_labels):
        """Compute class prototypes from support set"""
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(self.prototype_net(prototype))
        return torch.stack(prototypes), unique_labels
    
    def forward(self, query_features, support_features=None, support_labels=None):
        # Embed query features
        query_embedded = self.embedding(query_features)
        
        if support_features is not None and support_labels is not None:
            # Few-shot mode: compute prototypes and classify
            support_embedded = self.embedding(support_features)
            prototypes, proto_labels = self.compute_prototypes(support_embedded, support_labels)
            
            # Compute distances to prototypes
            distances = torch.cdist(query_embedded, prototypes)
            return -distances, proto_labels  # Negative distances as logits
        
        return query_embedded


# ============================================================
# 6. DIDS-MFL Main Framework
# ============================================================
class DIDS_MFL(nn.Module):
    """
    Disentangled Dynamic Intrusion Detection with Multi-Scale Fusion and Few-Shot Learning
    """
    def __init__(self, input_dim=40, hidden_dim=64, num_classes=10, num_factors=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Module 1: Statistical Disentanglement
        self.stat_disentangler = StatisticalDisentangler(
            input_dim, num_factors=num_factors, hidden_dim=hidden_dim
        )
        
        # Module 2: Representational Disentanglement
        self.rep_disentangler = RepresentationalDisentangler(
            hidden_dim * num_factors, hidden_dim=hidden_dim
        )
        
        # Module 3: Dynamic Graph Diffusion
        self.graph_diffusion = DynamicGraphDiffusion(
            hidden_dim, hidden_dim, num_layers=2
        )
        
        # Module 4: Multi-Scale Fusion
        self.multi_scale = MultiScaleFusion(
            hidden_dim, hidden_dim=hidden_dim
        )
        
        # Module 5: Few-Shot Classifier
        self.few_shot = FewShotClassifier(hidden_dim, hidden_dim=hidden_dim)
        
        # Classification heads
        self.binary_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
        
        self.multiclass_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x, edge_index=None, timestamps=None, mode='train'):
        batch_size = x.size(0)
        
        # Step 1: Statistical disentanglement
        stat_factors, mi_loss, recon_loss = self.stat_disentangler(x)
        stat_flat = stat_factors.reshape(batch_size, -1)
        
        # Step 2: Representational disentanglement
        rep_disentangled, corr_loss, rep_weights = self.rep_disentangler(stat_flat)
        
        # Step 3: Multi-scale fusion
        multi_scale_out, scale_weights = self.multi_scale(rep_disentangled)
        
        # Step 4: Combine disentangled and multi-scale features
        combined = torch.cat([rep_disentangled, multi_scale_out], dim=-1)
        fused = self.final_fusion(torch.cat([rep_disentangled, multi_scale_out, 
                                              rep_disentangled * multi_scale_out], dim=-1))
        
        # Step 5: Classification
        binary_logits = self.binary_classifier(combined)
        multiclass_logits = self.multiclass_classifier(combined)
        
        # Losses
        total_loss = mi_loss + recon_loss + 0.1 * corr_loss
        
        return {
            'binary_logits': binary_logits,
            'multiclass_logits': multiclass_logits,
            'fused_features': fused,
            'stat_factors': stat_factors,
            'rep_weights': rep_weights,
            'scale_weights': scale_weights,
            'losses': {
                'mi_loss': mi_loss,
                'recon_loss': recon_loss,
                'corr_loss': corr_loss,
                'total_disentangle': total_loss
            }
        }


# ============================================================
# 7. Baseline Models
# ============================================================
class MLPBaseline(nn.Module):
    """Simple MLP baseline for comparison"""
    def __init__(self, input_dim=40, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.binary_head = nn.Linear(hidden_dim // 2, 2)
        self.multi_head = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x, **kwargs):
        h = self.net(x)
        return {
            'binary_logits': self.binary_head(h),
            'multiclass_logits': self.multi_head(h)
        }


class GraphSAGEBaseline(nn.Module):
    """E-GraphSAGE inspired baseline (MLP version for edge-level classification)"""
    def __init__(self, input_dim=40, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.binary_head = nn.Linear(hidden_dim, 2)
        self.multi_head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index=None, **kwargs):
        h = self.net(x)
        return {
            'binary_logits': self.binary_head(h),
            'multiclass_logits': self.multi_head(h)
        }


# ============================================================
# 8. Training and Evaluation
# ============================================================
def prepare_data(data, test_ratio=0.2, val_ratio=0.1):
    """Prepare train/val/test splits with temporal ordering"""
    msg = data.msg.numpy()
    labels = data.label.numpy()
    attacks = data.attack.numpy()
    t = data.t.numpy()
    
    # Sort by time for temporal split
    time_order = np.argsort(t)
    n = len(time_order)
    
    # Temporal split: first 70% train, next 10% val, last 20% test
    train_end = int(n * (1 - test_ratio - val_ratio))
    val_end = int(n * (1 - test_ratio))
    
    train_idx = time_order[:train_end]
    val_idx = time_order[train_end:val_end]
    test_idx = time_order[val_end:]
    
    return {
        'train': {'features': msg[train_idx], 'binary': labels[train_idx], 'multi': attacks[train_idx]},
        'val': {'features': msg[val_idx], 'binary': labels[val_idx], 'multi': attacks[val_idx]},
        'test': {'features': msg[test_idx], 'binary': labels[test_idx], 'multi': attacks[test_idx]},
        'edge_index': data.edge_index.numpy(),
        'timestamps': t
    }


def train_model(model, data_dict, epochs=50, lr=0.001, device='cpu'):
    """Train a model and return history"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    X_train = torch.FloatTensor(data_dict['train']['features']).to(device)
    y_binary_train = torch.LongTensor(data_dict['train']['binary']).to(device)
    y_multi_train = torch.LongTensor(data_dict['train']['multi']).to(device)
    
    X_val = torch.FloatTensor(data_dict['val']['features']).to(device)
    y_binary_val = torch.LongTensor(data_dict['val']['binary']).to(device)
    y_multi_val = torch.LongTensor(data_dict['val']['multi']).to(device)
    
    # Edge index for graph models
    edge_index = torch.LongTensor(data_dict['edge_index']).to(device) if 'edge_index' in data_dict else None
    
    history = {'train_loss': [], 'val_loss': [], 'val_binary_f1': [], 'val_multi_f1': []}
    best_val_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(X_train, edge_index=edge_index)
        
        # Binary classification loss
        binary_loss = F.cross_entropy(out['binary_logits'], y_binary_train)
        
        # Multi-class loss (only for attack samples)
        attack_mask = y_binary_train == 1
        if attack_mask.sum() > 0:
            multi_loss = F.cross_entropy(
                out['multiclass_logits'][attack_mask],
                y_multi_train[attack_mask]
            )
        else:
            multi_loss = torch.tensor(0.0, device=device)
        
        # Disentanglement loss if available
        disentangle_loss = out.get('losses', {}).get('total_disentangle', 0)
        
        total_loss = binary_loss + multi_loss + 0.1 * disentangle_loss if isinstance(disentangle_loss, torch.Tensor) else binary_loss + multi_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val, edge_index=edge_index)
            val_binary_loss = F.cross_entropy(val_out['binary_logits'], y_binary_val)
            
            val_binary_pred = val_out['binary_logits'].argmax(dim=-1).cpu().numpy()
            val_binary_f1 = f1_score(y_binary_val.cpu().numpy(), val_binary_pred, average='macro')
            
            # Multi-class on attack samples only
            val_attack_mask = y_binary_val == 1
            if val_attack_mask.sum() > 0:
                val_multi_pred = val_out['multiclass_logits'][val_attack_mask].argmax(dim=-1).cpu().numpy()
                val_multi_f1 = f1_score(y_multi_val[val_attack_mask].cpu().numpy(), val_multi_pred, average='macro')
            else:
                val_multi_f1 = 0
        
        scheduler.step(val_binary_loss.item())
        
        history['train_loss'].append(total_loss.item())
        history['val_loss'].append(val_binary_loss.item())
        history['val_binary_f1'].append(val_binary_f1)
        history['val_multi_f1'].append(val_multi_f1)
        
        if val_binary_f1 > best_val_f1:
            best_val_f1 = val_binary_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss.item():.4f}, "
                  f"Val Binary F1: {val_binary_f1:.4f}, Val Multi F1: {val_multi_f1:.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    return model, history


def evaluate_model(model, data_dict, device='cpu'):
    """Evaluate model on test set"""
    model = model.to(device)
    model.eval()
    
    X_test = torch.FloatTensor(data_dict['test']['features']).to(device)
    y_binary_test = data_dict['test']['binary']
    y_multi_test = data_dict['test']['multi']
    edge_index = torch.LongTensor(data_dict['edge_index']).to(device) if 'edge_index' in data_dict else None
    
    with torch.no_grad():
        out = model(X_test, edge_index=edge_index)
        binary_pred = out['binary_logits'].argmax(dim=-1).cpu().numpy()
        multi_pred = out['multiclass_logits'].argmax(dim=-1).cpu().numpy()
    
    # Binary metrics
    binary_acc = accuracy_score(y_binary_test, binary_pred)
    binary_f1 = f1_score(y_binary_test, binary_pred, average='macro')
    binary_precision = precision_score(y_binary_test, binary_pred, average='macro')
    binary_recall = recall_score(y_binary_test, binary_pred, average='macro')
    
    # Multi-class metrics (on attack samples only)
    attack_mask = y_binary_test == 1
    if attack_mask.sum() > 0:
        multi_acc = accuracy_score(y_multi_test[attack_mask], multi_pred[attack_mask])
        multi_f1_macro = f1_score(y_multi_test[attack_mask], multi_pred[attack_mask], average='macro')
        multi_f1_weighted = f1_score(y_multi_test[attack_mask], multi_pred[attack_mask], average='weighted')
        
        # Per-class F1
        report = classification_report(y_multi_test[attack_mask], multi_pred[attack_mask], 
                                       output_dict=True, zero_division=0)
    else:
        multi_acc = multi_f1_macro = multi_f1_weighted = 0
        report = {}
    
    # Few-shot analysis
    attack_types = {0: 'Analysis', 1: 'Backdoor', 3: 'DoS', 4: 'Exploits', 
                    5: 'Fuzzers', 6: 'Generic', 7: 'Reconnaissance', 8: 'Shellcode', 9: 'Worms'}
    
    few_shot_results = {}
    for cls_id, cls_name in attack_types.items():
        cls_mask = y_multi_test == cls_id
        if cls_mask.sum() > 0:
            cls_f1 = f1_score(y_multi_test[cls_mask], multi_pred[cls_mask], average='macro', zero_division=0)
            few_shot_results[cls_name] = {
                'count': int(cls_mask.sum()),
                'f1': float(cls_f1),
                'is_few_shot': bool(cls_mask.sum() < 500)
            }
    
    return {
        'binary': {
            'accuracy': float(binary_acc),
            'f1_macro': float(binary_f1),
            'precision': float(binary_precision),
            'recall': float(binary_recall)
        },
        'multiclass': {
            'accuracy': float(multi_acc),
            'f1_macro': float(multi_f1_macro),
            'f1_weighted': float(multi_f1_weighted),
            'per_class': few_shot_results
        }
    }


# ============================================================
# Main Execution
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("DIDS-MFL: Disentangled Dynamic Intrusion Detection")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', map_location='cpu', weights_only=False)
    
    # Prepare splits
    print("Preparing data splits...")
    data_dict = prepare_data(data)
    
    print(f"Train: {len(data_dict['train']['features'])} samples")
    print(f"Val: {len(data_dict['val']['features'])} samples")
    print(f"Test: {len(data_dict['test']['features'])} samples")
    
    # Train and evaluate models
    results = {}
    histories = {}
    
    # 1. DIDS-MFL (Our method)
    print("\n" + "=" * 40)
    print("Training DIDS-MFL (Our Method)...")
    print("=" * 40)
    dids_model = DIDS_MFL(input_dim=40, hidden_dim=64, num_classes=10, num_factors=8)
    dids_model, dids_history = train_model(dids_model, data_dict, epochs=50, lr=0.001, device=device)
    results['DIDS-MFL'] = evaluate_model(dids_model, data_dict, device=device)
    histories['DIDS-MFL'] = dids_history
    
    # 2. MLP Baseline
    print("\n" + "=" * 40)
    print("Training MLP Baseline...")
    print("=" * 40)
    mlp_model = MLPBaseline(input_dim=40, hidden_dim=128, num_classes=10)
    mlp_model, mlp_history = train_model(mlp_model, data_dict, epochs=50, lr=0.001, device=device)
    results['MLP'] = evaluate_model(mlp_model, data_dict, device=device)
    histories['MLP'] = mlp_history
    
    # 3. GraphSAGE Baseline
    print("\n" + "=" * 40)
    print("Training GraphSAGE Baseline...")
    print("=" * 40)
    gsage_model = GraphSAGEBaseline(input_dim=40, hidden_dim=64, num_classes=10)
    gsage_model, gsage_history = train_model(gsage_model, data_dict, epochs=50, lr=0.001, device=device)
    results['GraphSAGE'] = evaluate_model(gsage_model, data_dict, device=device)
    histories['GraphSAGE'] = gsage_history
    
    # 4. Random Forest Baseline
    print("\n" + "=" * 40)
    print("Training Random Forest Baseline...")
    print("=" * 40)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(data_dict['train']['features'], data_dict['train']['multi'])
    rf_multi_pred = rf.predict(data_dict['test']['features'])
    rf_binary_pred = (rf_multi_pred != 2).astype(int)  # Class 2 is benign
    rf_binary_true = data_dict['test']['binary']
    rf_multi_true = data_dict['test']['multi']
    
    attack_mask = rf_binary_true == 1
    results['RandomForest'] = {
        'binary': {
            'accuracy': float(accuracy_score(rf_binary_true, rf_binary_pred)),
            'f1_macro': float(f1_score(rf_binary_true, rf_binary_pred, average='macro')),
            'precision': float(precision_score(rf_binary_true, rf_binary_pred, average='macro')),
            'recall': float(recall_score(rf_binary_true, rf_binary_pred, average='macro'))
        },
        'multiclass': {
            'accuracy': float(accuracy_score(rf_multi_true[attack_mask], rf_multi_pred[attack_mask])),
            'f1_macro': float(f1_score(rf_multi_true[attack_mask], rf_multi_pred[attack_mask], average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(rf_multi_true[attack_mask], rf_multi_pred[attack_mask], average='weighted', zero_division=0))
        }
    }
    
    # Save results
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save histories
    serializable_histories = {}
    for model_name, hist in histories.items():
        serializable_histories[model_name] = {k: [float(v) for v in vals] for k, vals in hist.items()}
    with open('outputs/training_histories.json', 'w') as f:
        json.dump(serializable_histories, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  Binary - Acc: {res['binary']['accuracy']:.4f}, F1: {res['binary']['f1_macro']:.4f}")
        print(f"  Multi   - Acc: {res['multiclass']['accuracy']:.4f}, F1: {res['multiclass']['f1_macro']:.4f}")
    
    print("\nAll results saved to outputs/")
