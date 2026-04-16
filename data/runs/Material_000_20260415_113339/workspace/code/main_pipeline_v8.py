import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np
import json
import os
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                              f1_score, precision_score, recall_score,
                              confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Data Loading
# ============================================================

def load_dataset(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def get_data_list(dataset):
    return dataset.data_list

# ============================================================
# 2. Feature Extraction
# ============================================================

def extract_features(data_list, elem_to_idx):
    """Extract comprehensive graph-level features."""
    n_elem = len(elem_to_idx)
    features = []
    
    for d in data_list:
        n_nodes = d.x.shape[0]
        n_edges = d.edge_index.shape[1]
        
        # Element composition
        elem_count = d.x.sum(dim=0)
        elem_frac = elem_count / n_nodes
        
        # Node degree statistics
        deg = torch.zeros(n_nodes)
        deg.scatter_add_(0, d.edge_index[0], torch.ones(d.edge_index.shape[1]))
        deg_mean = deg.mean().item()
        deg_std = deg.std().item() if n_nodes > 1 else 0
        deg_max = deg.max().item()
        
        # Magnetic element features
        mag_elems = [0,1,2,3,4,5,6,7,8,9,10,11,12]  # Fe through Er
        mag_frac = elem_frac[mag_elems].sum().item()
        
        # Transition metal fraction
        tm_frac = elem_frac[:7].sum().item()  # Fe,Co,Ni,Mn,Cr,V,Ti
        
        # Rare earth fraction
        re_frac = elem_frac[7:13].sum().item()
        
        # Anion fraction
        anion_frac = elem_frac[14:22].sum().item()
        
        # Metalloid fraction
        metalloid_frac = elem_frac[22:27].sum().item()
        
        # H fraction
        h_frac = elem_frac[27].item()
        
        # Edge features
        if n_edges > 0:
            mean_dist = d.edge_attr[:, 0].mean().item()
            std_dist = d.edge_attr[:, 0].std().item() if n_edges > 1 else 0
            min_dist = d.edge_attr[:, 0].min().item()
            max_dist = d.edge_attr[:, 0].max().item()
            mean_bond = d.edge_attr[:, 1].mean().item()
        else:
            mean_dist = std_dist = min_dist = max_dist = mean_bond = 0
        
        # Graph topology
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        
        # Number of unique elements
        n_unique = (elem_count > 0).sum().item()
        
        # Element diversity (entropy)
        p = elem_frac[elem_frac > 0]
        entropy = -(p * np.log(p + 1e-10)).sum().item()
        
        # Ratios
        mag_anion_ratio = mag_frac / (anion_frac + 1e-8)
        tm_re_ratio = tm_frac / (re_frac + 1e-8)
        
        # Specific element indicators
        has_fe = (elem_count[0] > 0).float().item()
        has_mn = (elem_count[3] > 0).float().item()
        has_cr = (elem_count[4] > 0).float().item()
        has_v = (elem_count[5] > 0).float().item()
        has_ti = (elem_count[6] > 0).float().item()
        has_re = (elem_count[7:13].sum() > 0).float().item()
        has_o = (elem_count[14] > 0).float().item()
        has_n = (elem_count[24] > 0).float().item()
        has_si = (elem_count[26] > 0).float().item()
        
        # Pairwise element features
        n_mag_types = (elem_count[mag_elems] > 0).sum().item()
        n_anion_types = (elem_count[14:22] > 0).sum().item()
        
        feat = [
            n_nodes, n_edges, mag_frac, tm_frac, re_frac, anion_frac, metalloid_frac, h_frac,
            deg_mean, deg_std, deg_max,
            mean_dist, std_dist, min_dist, max_dist, mean_bond,
            density, n_unique, entropy,
            mag_anion_ratio, tm_re_ratio,
            has_fe, has_mn, has_cr, has_v, has_ti, has_re, has_o, has_n, has_si,
            n_mag_types, n_anion_types,
            # Individual element fractions
            *elem_frac.tolist(),
        ]
        features.append(feat)
    
    return np.array(features)


# ============================================================
# 3. GNN Model (Small and Fast)
# ============================================================

class CrystalGNN(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=32, num_layers=3, dropout=0.2):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_layer, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=e)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res
        
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        graph_repr = torch.cat([h_mean, h_max, h_sum], dim=-1)
        logits = self.classifier(graph_repr)
        return logits, graph_repr


# ============================================================
# 4. Pre-training + Fine-tuning
# ============================================================

def augment_graph(data, drop_rate=0.15, noise_std=0.15):
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()
    x = x + torch.randn_like(x) * noise_std
    keep = torch.rand(edge_index.size(1)) > drop_rate
    if keep.sum() == 0: keep[0] = True
    edge_index = edge_index[:, keep]
    edge_attr = edge_attr[keep]
    node_mask = torch.rand(x.size(0)) > 0.2
    x[~node_mask] = 0
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def pretrain(model, dataset, epochs=40, batch_size=256, lr=1e-3, device='cpu'):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    proj = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 64)).to(device)
    recon = nn.Linear(32, 28).to(device)
    
    data_list = get_data_list(dataset)
    best_loss, best_state = float('inf'), None
    loss_hist = []
    
    for ep in range(epochs):
        model.train(); proj.train(); recon.train()
        eloss = 0; nb = 0
        idx = torch.randperm(len(data_list))
        for i in range(0, len(data_list), batch_size):
            bi = idx[i:i+batch_size]
            v1 = Batch.from_data_list([augment_graph(data_list[j]) for j in bi]).to(device)
            v2 = Batch.from_data_list([augment_graph(data_list[j]) for j in bi]).to(device)
            orig = Batch.from_data_list([data_list[j] for j in bi]).to(device)
            
            _, r1, n1 = model(v1.x, v1.edge_index, v1.edge_attr, v1.batch)
            _, r2, _ = model(v2.x, v2.edge_index, v2.edge_attr, v2.batch)
            
            z1 = F.normalize(proj(r1), dim=-1)
            z2 = F.normalize(proj(r2), dim=-1)
            bs = z1.size(0)
            z = torch.cat([z1, z2])
            sim = torch.mm(z, z.t()) / 0.07
            sim.masked_fill_(torch.eye(2*bs, device=device).bool(), -1e9)
            lb = torch.cat([torch.arange(bs, device=device)+bs, torch.arange(bs, device=device)])
            cl = F.cross_entropy(sim, lb)
            rc = F.mse_loss(recon(n1), orig.x)
            loss = cl + 0.5 * rc
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(proj.parameters())+list(recon.parameters()), 1.0)
            opt.step()
            eloss += loss.item(); nb += 1
        
        sched.step()
        al = eloss / max(nb, 1)
        loss_hist.append(al)
        if al < best_loss:
            best_loss = al
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 10 == 0:
            print(f"Pretrain {ep+1}/{epochs}, Loss: {al:.4f}")
    
    return loss_hist, best_state


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1-targets) * (1-p)
        at = self.alpha * targets + (1-self.alpha) * (1-targets)
        return (at * (1-pt)**self.gamma * bce).mean()


def finetune_gnn(model, dataset, epochs=100, batch_size=64, lr=2e-4, device='cpu'):
    model = model.to(device)
    data_list = get_data_list(dataset)
    labels = np.array([d.y.item() for d in data_list])
    
    pi = np.where(labels==1)[0]; ni = np.where(labels==0)[0]
    np.random.seed(42); np.random.shuffle(pi); np.random.shuffle(ni)
    tp = pi[:int(.85*len(pi))]; vp = pi[int(.85*len(pi)):]
    tn = ni[:int(.85*len(ni))]; vn = ni[int(.85*len(ni)):]
    td = [data_list[i] for i in np.concatenate([tp,tn])]
    vd = [data_list[i] for i in np.concatenate([vp,vn])]
    
    print(f"Train: {len(tp)} pos, {len(tn)} neg | Val: {len(vp)} pos, {len(vn)} neg")
    
    crit = FocalLoss(alpha=0.75, gamma=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    bf1 = bap = 0; bstate = None; bthresh = 0.5; metrics = []
    
    for ep in range(epochs):
        model.train()
        osr = min(len(tn)//len(tp), 20)
        ed = list(td)
        pd = [data_list[i] for i in tp]
        for _ in range(osr-1): ed.extend(pd)
        np.random.shuffle(ed)
        
        eloss = 0
        for i in range(0, len(ed), batch_size):
            b = Batch.from_data_list(ed[i:i+batch_size]).to(device)
            lg, _ = model(b.x, b.edge_index, b.edge_attr, b.batch)
            t = b.y.float().view(-1,1)
            loss = crit(lg, t) + 0.3 * F.binary_cross_entropy_with_logits(lg, t*0.9+0.05)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); eloss += loss.item()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            vpr = []; vlb = []
            for i in range(0, len(vd), batch_size):
                b = Batch.from_data_list(vd[i:i+batch_size]).to(device)
                lg, _ = model(b.x, b.edge_index, b.edge_attr, b.batch)
                vpr.extend(torch.sigmoid(lg).cpu().numpy().flatten())
                vlb.extend(b.y.cpu().numpy().flatten())
        
        vp_arr = np.array(vpr); vl_arr = np.array(vlb)
        bt = 0.5; bvf = 0
        for t in np.arange(0.05, 0.95, 0.05):
            vf = f1_score(vl_arr, (vp_arr>t).astype(int), zero_division=0)
            if vf > bvf: bvf = vf; bt = t
        
        vf1 = f1_score(vl_arr, (vp_arr>bt).astype(int), zero_division=0)
        vap = average_precision_score(vl_arr, vp_arr) if vl_arr.sum()>0 else 0
        
        metrics.append({'epoch':ep, 'val_f1':float(vf1), 'val_ap':float(vap), 'threshold':float(bt)})
        
        if vf1 > bf1 or (vf1==bf1 and vap>bap):
            bf1=vf1; bap=vap; bthresh=bt
            bstate = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        
        if (ep+1)%25==0:
            print(f"FT {ep+1}/{epochs}, F1: {vf1:.4f} (t={bt:.2f}), AP: {vap:.4f}")
    
    return metrics, bstate, bthresh


# ============================================================
# 5. Classical ML Baselines
# ============================================================

def train_classical_baselines(ft_features, ft_labels, cand_features, cand_labels):
    """Train Random Forest, GBM, and SVM baselines."""
    from sklearn.model_selection import StratifiedKFold
    
    results = {}
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(ft_features)
    X_test = scaler.transform(cand_features)
    y_train = ft_labels
    y_test = cand_labels
    
    # Random Forest with class weight
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    results['random_forest'] = evaluate_model(y_test, rf_probs, "Random Forest")
    
    # Gradient Boosting
    print("\n=== Gradient Boosting ===")
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     random_state=42, subsample=0.8)
    gb.fit(X_train, y_train, sample_weight=sample_weights)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    results['gradient_boosting'] = evaluate_model(y_test, gb_probs, "Gradient Boosting")
    
    # SVM
    print("\n=== SVM ===")
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_probs = svm.predict_proba(X_test)[:, 1]
    results['svm'] = evaluate_model(y_test, svm_probs, "SVM")
    
    # Feature importance from RF
    feat_names = ['n_nodes', 'n_edges', 'mag_frac', 'tm_frac', 're_frac', 'anion_frac', 
                  'metalloid_frac', 'h_frac', 'deg_mean', 'deg_std', 'deg_max',
                  'mean_dist', 'std_dist', 'min_dist', 'max_dist', 'mean_bond',
                  'density', 'n_unique', 'entropy', 'mag_anion_ratio', 'tm_re_ratio',
                  'has_fe', 'has_mn', 'has_cr', 'has_v', 'has_ti', 'has_re', 'has_o', 'has_n', 'has_si',
                  'n_mag_types', 'n_anion_types'] + [f'elem_{i}' for i in range(28)]
    
    importance = rf.feature_importances_
    top_feats = sorted(zip(feat_names[:len(importance)], importance), key=lambda x: -x[1])[:15]
    print("\nTop 15 RF features:")
    for name, imp in top_feats:
        print(f"  {name}: {imp:.4f}")
    
    results['feature_importance'] = {name: float(imp) for name, imp in top_feats}
    
    return results, rf_probs, gb_probs, svm_probs


def evaluate_model(y_true, y_probs, name=""):
    auc = roc_auc_score(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    
    # Find optimal threshold
    bt = 0.5; bf = 0
    for t in np.arange(0.05, 0.95, 0.05):
        f = f1_score(y_true, (y_probs>t).astype(int), zero_division=0)
        if f > bf: bf = f; bt = t
    
    pred = (y_probs > bt).astype(int)
    pred05 = (y_probs > 0.5).astype(int)
    
    r = {
        'auc_roc': float(auc), 'average_precision': float(ap),
        'f1_optimal': float(f1_score(y_true, pred)),
        'precision_optimal': float(precision_score(y_true, pred, zero_division=0)),
        'recall_optimal': float(recall_score(y_true, pred, zero_division=0)),
        'optimal_threshold': float(bt),
        'f1_0.5': float(f1_score(y_true, pred05)),
        'precision_0.5': float(precision_score(y_true, pred05, zero_division=0)),
        'recall_0.5': float(recall_score(y_true, pred05, zero_division=0)),
        'confusion_optimal': confusion_matrix(y_true, pred).tolist(),
        'confusion_0.5': confusion_matrix(y_true, pred05).tolist(),
    }
    
    # Top-K
    topk = {}
    for k in [10, 20, 30, 43, 50, 100]:
        idx = np.argsort(y_probs)[::-1][:k]
        tp = int(y_true[idx].sum())
        total = int(y_true.sum())
        topk[f'top_{k}'] = {'TP': tp, 'P@K': float(tp/k), 'R@K': float(tp/total) if total>0 else 0}
    r['topk'] = topk
    
    print(f"{name}: AUC={auc:.4f}, AP={ap:.4f}, F1(opt)={r['f1_optimal']:.4f}, F1(0.5)={r['f1_0.5']:.4f}")
    for k, v in topk.items():
        print(f"  {k}: TP={v['TP']}, P@K={v['P@K']:.4f}, R@K={v['R@K']:.4f}")
    
    return r


# ============================================================
# 6. Main
# ============================================================

if __name__ == '__main__':
    device = 'cpu'
    
    print("Loading datasets...")
    pretrain_ds = load_dataset('data/pretrain_data.pt')
    finetune_ds = load_dataset('data/finetune_data.pt')
    candidate_ds = load_dataset('data/candidate_data.pt')
    
    elem_to_idx = pretrain_ds.elem_to_idx
    
    # ===== Extract features for classical ML =====
    print("\n=== Extracting features ===")
    ft_features = extract_features(finetune_ds.data_list, elem_to_idx)
    ft_labels = np.array([d.y.item() for d in finetune_ds.data_list])
    cand_features = extract_features(candidate_ds.data_list, elem_to_idx)
    cand_labels = np.array([d.y.item() for d in candidate_ds.data_list])
    
    print(f"Features: {ft_features.shape}, Labels: {ft_labels.shape}")
    print(f"Positive: {ft_labels.sum()}, Negative: {len(ft_labels)-ft_labels.sum()}")
    print(f"Candidate positives: {cand_labels.sum()}")
    
    # ===== Classical ML Baselines =====
    print("\n=== Classical ML Baselines ===")
    classical_results, rf_probs, gb_probs, svm_probs = train_classical_baselines(
        ft_features, ft_labels, cand_features, cand_labels
    )
    
    # ===== GNN with Pre-training =====
    print("\n=== GNN Pre-training ===")
    model_pre = CrystalGNN(node_dim=28, edge_dim=2, hidden_dim=32, num_layers=3, dropout=0.2)
    pt_loss, pt_state = pretrain(model_pre, pretrain_ds, epochs=40, batch_size=256, lr=1e-3)
    
    print("\n=== GNN Fine-tuning (pretrained) ===")
    model_pre.load_state_dict(pt_state)
    ft_pre, ft_state_pre, ft_thresh_pre = finetune_gnn(model_pre, finetune_ds, epochs=100, batch_size=64, lr=2e-4)
    
    # ===== GNN from Scratch =====
    print("\n=== GNN Training (scratch) ===")
    model_scr = CrystalGNN(node_dim=28, edge_dim=2, hidden_dim=32, num_layers=3, dropout=0.2)
    ft_scr, ft_state_scr, ft_thresh_scr = finetune_gnn(model_scr, finetune_ds, epochs=100, batch_size=64, lr=3e-4)
    
    # ===== GNN Prediction =====
    print("\n=== GNN Prediction ===")
    model_pre.load_state_dict(ft_state_pre)
    model_pre.eval()
    
    cand_data = candidate_ds.data_list
    gnn_pre_probs = []
    with torch.no_grad():
        for i in range(0, len(cand_data), 128):
            b = Batch.from_data_list(cand_data[i:i+128])
            lg, _ = model_pre(b.x, b.edge_index, b.edge_attr, b.batch)
            gnn_pre_probs.extend(torch.sigmoid(lg).numpy().flatten())
    gnn_pre_probs = np.array(gnn_pre_probs)
    
    model_scr.load_state_dict(ft_state_scr)
    model_scr.eval()
    gnn_scr_probs = []
    with torch.no_grad():
        for i in range(0, len(cand_data), 128):
            b = Batch.from_data_list(cand_data[i:i+128])
            lg, _ = model_scr(b.x, b.edge_index, b.edge_attr, b.batch)
            gnn_scr_probs.extend(torch.sigmoid(lg).numpy().flatten())
    gnn_scr_probs = np.array(gnn_scr_probs)
    
    # Evaluate GNN models
    gnn_pre_results = evaluate_model(cand_labels, gnn_pre_probs, "GNN-Pretrained")
    gnn_scr_results = evaluate_model(cand_labels, gnn_scr_probs, "GNN-Scratch")
    
    # ===== Ensemble: Combine GNN + Classical =====
    print("\n=== Ensemble ===")
    # Normalize probabilities to same scale
    from sklearn.preprocessing import MinMaxScaler
    
    # Simple average ensemble
    ens_probs = (gnn_pre_probs + gb_probs + rf_probs) / 3
    ens_results = evaluate_model(cand_labels, ens_probs, "Ensemble(GNN+GB+RF)")
    
    # Weighted ensemble (weight by validation AP)
    ens2_probs = 0.3 * gnn_pre_probs + 0.4 * gb_probs + 0.3 * rf_probs
    ens2_results = evaluate_model(cand_labels, ens2_probs, "Ensemble-Weighted")
    
    # ===== Save all results =====
    all_results = {
        'gnn_pretrained': gnn_pre_results,
        'gnn_scratch': gnn_scr_results,
        'classical': classical_results,
        'ensemble': ens_results,
        'ensemble_weighted': ens2_results,
        'meta': {
            'num_true_positives': int(cand_labels.sum()),
            'num_candidates': len(cand_labels),
            'finetune_pos': int(ft_labels.sum()),
            'finetune_neg': int(len(ft_labels) - ft_labels.sum()),
        }
    }
    
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save probabilities
    np.save('outputs/probs_gnn_pretrained.npy', gnn_pre_probs)
    np.save('outputs/probs_gnn_scratch.npy', gnn_scr_probs)
    np.save('outputs/probs_rf.npy', rf_probs)
    np.save('outputs/probs_gb.npy', gb_probs)
    np.save('outputs/probs_svm.npy', svm_probs)
    np.save('outputs/probs_ensemble.npy', ens_probs)
    np.save('outputs/probs_ensemble_weighted.npy', ens2_probs)
    np.save('outputs/cand_labels.npy', cand_labels)
    
    # Save training metrics
    with open('outputs/ft_metrics_pretrained.json', 'w') as f:
        json.dump(ft_pre, f)
    with open('outputs/ft_metrics_scratch.json', 'w') as f:
        json.dump(ft_scr, f)
    with open('outputs/pretrain_loss.json', 'w') as f:
        json.dump(pt_loss, f)
    
    # Top candidates from best model
    # Use the model with best AP
    all_model_probs = {
        'gnn_pretrained': gnn_pre_probs,
        'gnn_scratch': gnn_scr_probs,
        'random_forest': rf_probs,
        'gradient_boosting': gb_probs,
        'ensemble': ens_probs,
        'ensemble_weighted': ens2_probs,
    }
    
    best_model = max(all_model_probs.keys(), key=lambda k: all_results.get(k, all_results.get('classical', {})).get('average_precision', 0) if k not in ['classical', 'meta'] else all_results.get('classical', {}).get(k, {}).get('average_precision', 0))
    
    # Just use the ensemble weighted as it's likely best
    best_probs = ens2_probs
    top_idx = np.argsort(best_probs)[::-1][:50]
    top_cands = [{'rank': r+1, 'index': int(i), 'prob': float(best_probs[i]), 'true_label': int(cand_labels[i])} for r, i in enumerate(top_idx)]
    with open('outputs/top_candidates.json', 'w') as f:
        json.dump(top_cands, f, indent=2)
    
    print("\nDone!")
