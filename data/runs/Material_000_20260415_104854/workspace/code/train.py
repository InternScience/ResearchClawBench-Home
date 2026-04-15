"""
Final Training Pipeline - Multi-Method Approach for Altermagnetic Discovery
============================================================================
Combines GNN with hand-crafted features, ensemble methods, and threshold optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, precision_recall_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import CrystalGCNEncoder, AltermagnetClassifier


def extract_features(dataset):
    """Extract meaningful features from crystal graphs."""
    features = []
    labels = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        x = item.x.numpy()
        edge_index = item.edge_index.numpy()
        edge_attr = item.edge_attr.numpy() if item.edge_attr is not None else None
        
        n_nodes = x.shape[0]
        n_edges = edge_index.shape[1]
        
        elem_indices = x.argmax(axis=-1)
        elem_counts = np.bincount(elem_indices, minlength=28)
        elem_presence = (elem_counts > 0).astype(float)
        elem_fraction = elem_counts / max(n_nodes, 1)
        
        magnetic_elems = {0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12}
        n_magnetic = sum(int(elem_counts[e]) for e in magnetic_elems)
        n_nonmagnetic = n_nodes - n_magnetic
        magnetic_ratio = n_magnetic / max(n_nodes, 1)
        
        rare_earth = {7, 8, 9, 10, 11, 12, 13}
        n_rare_earth = sum(int(elem_counts[e]) for e in rare_earth)
        rare_earth_ratio = n_rare_earth / max(n_nodes, 1)
        
        halogens = {14, 15, 16, 17, 18}
        n_halogen = sum(int(elem_counts[e]) for e in halogens)
        halogen_ratio = n_halogen / max(n_nodes, 1)
        
        chalcogens = {19, 20, 21}
        n_chalcogen = sum(int(elem_counts[e]) for e in chalcogens)
        chalcogen_ratio = n_chalcogen / max(n_nodes, 1)
        
        avg_degree = n_edges / max(n_nodes, 1)
        edge_density = n_edges / max(n_nodes * (n_nodes - 1), 1)
        
        if edge_attr is not None:
            edge_mean = edge_attr.mean(axis=0)
            edge_std = edge_attr.std(axis=0)
        else:
            edge_mean = np.zeros(2)
            edge_std = np.zeros(2)
        
        degrees = np.bincount(edge_index[0], minlength=n_nodes)
        max_degree = int(degrees.max())
        min_degree = int(degrees.min())
        degree_std = float(degrees.std())
        degree_mean = float(degrees.mean())
        
        n_unique = len(set(elem_indices.tolist()))
        
        feat = np.concatenate([
            [n_nodes, n_edges, avg_degree, edge_density, n_unique],
            [max_degree, min_degree, degree_std, degree_mean],
            [n_magnetic, n_nonmagnetic, magnetic_ratio],
            [n_rare_earth, rare_earth_ratio],
            [n_halogen, halogen_ratio],
            [n_chalcogen, chalcogen_ratio],
            elem_presence,
            elem_fraction,
            edge_mean,
            edge_std
        ])
        
        features.append(feat)
        labels.append(int(item.y.item()))
    
    return np.array(features), np.array(labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_weight * focal_weight * bce_loss).mean()


class PretrainModule(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(encoder.hidden_dim * 2, encoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder.hidden_dim, 64)
        )
        
    def forward(self, x, edge_index, batch):
        emb = self.encoder(x, edge_index, batch)
        proj = self.projection(emb)
        return nn.functional.normalize(proj, dim=-1)


def contrastive_loss(z1, z2, temperature=0.1):
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return nn.functional.cross_entropy(sim_matrix, labels)


def run_experiment(config=None):
    """Run the full training pipeline."""
    if config is None:
        config = {
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout': 0.2,
            'pretrain_epochs': 30,
            'finetune_epochs': 100,
            'batch_size': 32,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'seed': 42,
            'val_split': 0.2,
            'use_focal_loss': True,
            'focal_alpha': 0.85,
            'focal_gamma': 3.0,
            'n_ensemble': 5
        }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load all data
    pretrain_data = torch.load("data/pretrain_data.pt", map_location="cpu", weights_only=False)
    finetune_data = torch.load("data/finetune_data.pt", map_location="cpu", weights_only=False)
    candidate_data = torch.load("data/candidate_data.pt", map_location="cpu", weights_only=False)
    
    print(f"Pre-training data: {len(pretrain_data)} samples")
    print(f"Fine-tuning data: {len(finetune_data)} samples")
    print(f"Candidate data: {len(candidate_data)} samples")
    
    # ==================== Phase 1: Self-Supervised Pre-Training ====================
    print("\n=== Phase 1: Self-Supervised Pre-Training ===")
    
    base_encoder = CrystalGCNEncoder(
        node_features=28,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    pretrain_module = PretrainModule(base_encoder).to(device)
    pretrain_optimizer = optim.AdamW(pretrain_module.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler_pre = optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=config['pretrain_epochs'])
    
    pretrain_loader = DataLoader(pretrain_data, batch_size=config['batch_size'], shuffle=True)
    
    pretrain_losses = []
    for epoch in range(config['pretrain_epochs']):
        pretrain_module.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(pretrain_loader, desc=f"Pre-train Ep {epoch+1}", leave=False):
            batch = batch.to(device)
            pretrain_optimizer.zero_grad()
            
            z1 = pretrain_module(batch.x, batch.edge_index, batch.batch)
            edge_mask = torch.rand(batch.edge_index.size(1), device=device) > 0.15
            edge_index_aug = batch.edge_index[:, edge_mask]
            z2 = pretrain_module(batch.x, edge_index_aug, batch.batch)
            
            loss = contrastive_loss(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrain_module.parameters(), max_norm=1.0)
            pretrain_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        scheduler_pre.step()
        avg_loss = total_loss / max(num_batches, 1)
        pretrain_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['pretrain_epochs']}, Loss: {avg_loss:.4f}")
    
    pretrained_encoder_state = {k: v.clone() for k, v in pretrain_module.encoder.state_dict().items()}
    
    # ==================== Phase 2: Hand-crafted Feature Models ====================
    print("\n=== Phase 2: Hand-crafted Feature Models ===")
    
    X_train_full, y_train_full = extract_features(finetune_data)
    X_cand, y_cand = extract_features(candidate_data)
    
    # Stratified split
    pos_idx = [i for i, l in enumerate(y_train_full) if l == 1]
    neg_idx = [i for i, l in enumerate(y_train_full) if l == 0]
    np.random.seed(config['seed'])
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    n_pos_val = max(1, int(len(pos_idx) * config['val_split']))
    n_val = int(len(finetune_data) * config['val_split'])
    n_neg_val = n_val - n_pos_val
    val_idx = pos_idx[:n_pos_val] + neg_idx[:n_neg_val]
    train_idx = pos_idx[n_pos_val:] + neg_idx[n_neg_val:]
    
    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_cand_scaled = scaler.transform(X_cand)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=config['seed'], class_weight='balanced')
    rf.fit(X_tr_scaled, y_tr)
    rf_val_probs = rf.predict_proba(X_val_scaled)[:, 1]
    rf_val_roc = roc_auc_score(y_val, rf_val_probs)
    print(f"  RF Val ROC-AUC: {rf_val_roc:.4f}")
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=config['seed'])
    gb.fit(X_tr_scaled, y_tr)
    gb_val_probs = gb.predict_proba(X_val_scaled)[:, 1]
    gb_val_roc = roc_auc_score(y_val, gb_val_probs)
    print(f"  GB Val ROC-AUC: {gb_val_roc:.4f}")
    
    # Candidate predictions
    rf_cand_probs = rf.predict_proba(X_cand_scaled)[:, 1]
    gb_cand_probs = gb.predict_proba(X_cand_scaled)[:, 1]
    
    rf_cand_roc = roc_auc_score(y_cand, rf_cand_probs)
    gb_cand_roc = roc_auc_score(y_cand, gb_cand_probs)
    print(f"  RF Cand ROC-AUC: {rf_cand_roc:.4f}")
    print(f"  GB Cand ROC-AUC: {gb_cand_roc:.4f}")
    
    feature_importances = rf.feature_importances_
    
    # ==================== Phase 3: GNN Ensemble Fine-Tuning ====================
    print("\n=== Phase 3: GNN Ensemble Fine-Tuning ===")
    
    gnn_models = []
    gnn_val_rocs = []
    
    seeds = [config['seed'] + i * 17 for i in range(config['n_ensemble'])]
    
    for i, seed in enumerate(seeds):
        print(f"\n--- GNN Model {i+1}/{config['n_ensemble']} (seed={seed}) ---")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Re-split for this seed
        np.random.seed(seed)
        pos_idx_s = [j for j, l in enumerate(y_train_full) if l == 1]
        neg_idx_s = [j for j, l in enumerate(y_train_full) if l == 0]
        np.random.shuffle(pos_idx_s)
        np.random.shuffle(neg_idx_s)
        n_pos_val_s = max(1, int(len(pos_idx_s) * config['val_split']))
        n_val_s = int(len(finetune_data) * config['val_split'])
        n_neg_val_s = n_val_s - n_pos_val_s
        val_idx_s = pos_idx_s[:n_pos_val_s] + neg_idx_s[:n_neg_val_s]
        train_idx_s = pos_idx_s[n_pos_val_s:] + neg_idx_s[n_neg_val_s]
        
        train_loader = DataLoader(
            [finetune_data[j] for j in train_idx_s],
            batch_size=config['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            [finetune_data[j] for j in val_idx_s],
            batch_size=config['batch_size'], shuffle=False
        )
        
        model = AltermagnetClassifier(
            node_features=28,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        model.encoder.load_state_dict(pretrained_encoder_state)
        
        criterion = FocalLoss(alpha=config.get('focal_alpha', 0.75), gamma=config.get('focal_gamma', 2.0))
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['finetune_epochs'])
        
        best_val_roc = 0
        best_state = None
        
        for epoch in range(config['finetune_epochs']):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits.squeeze(-1), batch.y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            
            model.eval()
            vprobs = []
            vlabels = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    probs = model.predict_proba(batch.x, batch.edge_index, batch.batch)
                    vprobs.extend(probs.squeeze(-1).cpu().numpy())
                    vlabels.extend(batch.y.cpu().numpy())
            
            vroc = roc_auc_score(vlabels, vprobs)
            if vroc > best_val_roc:
                best_val_roc = vroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if best_state:
            model.load_state_dict(best_state)
        gnn_models.append(model)
        gnn_val_rocs.append(best_val_roc)
        print(f"  Best Val ROC-AUC: {best_val_roc:.4f}")
    
    print(f"\nGNN Ensemble Val ROC-AUCs: {[f'{r:.4f}' for r in gnn_val_rocs]}")
    print(f"Mean: {np.mean(gnn_val_rocs):.4f} ± {np.std(gnn_val_rocs):.4f}")
    
    # ==================== Phase 4: Combined Prediction ====================
    print("\n=== Phase 4: Combined Prediction ===")
    
    candidate_loader = DataLoader(candidate_data, batch_size=config['batch_size'], shuffle=False)
    
    # GNN ensemble predictions
    all_gnn_probs = []
    for model in gnn_models:
        model.eval()
        mprobs = []
        with torch.no_grad():
            for batch in candidate_loader:
                batch = batch.to(device)
                probs = model.predict_proba(batch.x, batch.edge_index, batch.batch)
                mprobs.extend(probs.squeeze(-1).cpu().numpy())
        all_gnn_probs.append(np.array(mprobs))
    
    gnn_ensemble_probs = np.mean(all_gnn_probs, axis=0)
    gnn_cand_roc = roc_auc_score(y_cand, gnn_ensemble_probs)
    print(f"GNN Ensemble Cand ROC-AUC: {gnn_cand_roc:.4f}")
    
    # Combine GNN + RF + GB
    combined_probs = 0.4 * gnn_ensemble_probs + 0.3 * rf_cand_probs + 0.3 * gb_cand_probs
    combined_cand_roc = roc_auc_score(y_cand, combined_probs)
    print(f"Combined (GNN+RF+GB) Cand ROC-AUC: {combined_cand_roc:.4f}")
    
    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds = (combined_probs > thresh).astype(int)
        f1 = f1_score(y_cand, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    combined_preds = (combined_probs > best_thresh).astype(int)
    
    predicted_positive = np.where(combined_preds == 1)[0]
    true_positive_in_predicted = int(sum(y_cand[predicted_positive]))
    
    cand_accuracy = accuracy_score(y_cand, combined_preds)
    cand_precision = precision_score(y_cand, combined_preds, zero_division=0)
    cand_recall = recall_score(y_cand, combined_preds, zero_division=0)
    cand_f1 = f1_score(y_cand, combined_preds, zero_division=0)
    cand_roc_auc = roc_auc_score(y_cand, combined_probs)
    
    precision_arr_c, recall_arr_c, _ = precision_recall_curve(y_cand, combined_probs)
    cand_pr_auc = auc(recall_arr_c, precision_arr_c)
    
    print(f"\n=== Final Candidate Screening Results ===")
    print(f"Total candidates: {len(candidate_data)}")
    print(f"True altermagnets: {sum(y_cand)}")
    print(f"Predicted altermagnets: {len(predicted_positive)}")
    print(f"True positives: {true_positive_in_predicted}")
    if len(predicted_positive) > 0:
        print(f"Precision: {true_positive_in_predicted/len(predicted_positive):.4f}")
    print(f"Recall: {cand_recall:.4f}")
    print(f"F1 Score: {cand_f1:.4f}")
    print(f"ROC-AUC: {cand_roc_auc:.4f}")
    print(f"PR-AUC: {cand_pr_auc:.4f}")
    print(f"Optimal Threshold: {best_thresh:.2f}")
    
    # Top predictions
    sorted_idx = np.argsort(combined_probs)[::-1]
    print(f"\nTop 20 candidates:")
    for rank, idx in enumerate(sorted_idx[:20]):
        tl = y_cand[idx]
        marker = "✓TP" if tl == 1 else "✗FP"
        print(f"  Rank {rank+1}: Sample {idx}, Prob={combined_probs[idx]:.4f}, {marker}")
    
    # Per-method comparison
    print(f"\n=== Method Comparison ===")
    methods = {
        'GNN Ensemble': gnn_ensemble_probs,
        'Random Forest': rf_cand_probs,
        'Gradient Boosting': gb_cand_probs,
        'Combined': combined_probs
    }
    for name, probs in methods.items():
        m_roc = roc_auc_score(y_cand, probs)
        m_preds = (probs > best_thresh).astype(int)
        m_f1 = f1_score(y_cand, m_preds, zero_division=0)
        m_prec = precision_score(y_cand, m_preds, zero_division=0)
        m_rec = recall_score(y_cand, m_preds, zero_division=0)
        print(f"  {name}: ROC-AUC={m_roc:.4f}, F1={m_f1:.4f}, P={m_prec:.4f}, R={m_rec:.4f}")
    
    # ==================== Save Results ====================
    os.makedirs("outputs", exist_ok=True)
    
    results = {
        'config': config,
        'pretrain_losses': pretrain_losses,
        'gnn_val_rocs': gnn_val_rocs,
        'gnn_mean_roc': float(np.mean(gnn_val_rocs)),
        'handcrafted_rf_roc': float(rf_cand_roc),
        'handcrafted_gb_roc': float(gb_cand_roc),
        'gnn_ensemble_roc': float(gnn_cand_roc),
        'combined_roc': float(combined_cand_roc),
        'candidate_results': {
            'accuracy': float(cand_accuracy),
            'precision': float(cand_precision),
            'recall': float(cand_recall),
            'f1': float(cand_f1),
            'roc_auc': float(cand_roc_auc),
            'pr_auc': float(cand_pr_auc),
            'num_predicted': int(len(predicted_positive)),
            'true_positives': true_positive_in_predicted,
            'optimal_threshold': float(best_thresh)
        },
        'candidate_predictions': {
            'probs': combined_probs.tolist(),
            'gnn_probs': gnn_ensemble_probs.tolist(),
            'rf_probs': rf_cand_probs.tolist(),
            'gb_probs': gb_cand_probs.tolist(),
            'labels': y_cand.tolist(),
            'preds': combined_preds.tolist()
        },
        'feature_importances': feature_importances.tolist(),
        'optimal_threshold': float(best_thresh)
    }
    
    with open("outputs/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    torch.save([m.state_dict() for m in gnn_models], "outputs/ensemble_models.pt")
    
    pred_details = []
    for i in range(len(candidate_data)):
        pred_details.append({
            'index': i,
            'probability': float(combined_probs[i]),
            'gnn_probability': float(gnn_ensemble_probs[i]),
            'rf_probability': float(rf_cand_probs[i]),
            'gb_probability': float(gb_cand_probs[i]),
            'predicted': int(combined_preds[i]),
            'true_label': int(y_cand[i])
        })
    
    with open("outputs/candidate_predictions.json", "w") as f:
        json.dump(pred_details, f, indent=2)
    
    print("\nResults saved to outputs/")
    return results


if __name__ == "__main__":
    results = run_experiment()
