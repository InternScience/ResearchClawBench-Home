"""
Phase 3: Training and Evaluation of all models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score, 
                             accuracy_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import (DIDS_MFL, BaselineMLP, AblationSDOnly, 
                   AblationNoGraph, AblationNoRepDis)

# ===================== Load Data =====================
print("Loading data...")
data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', map_location='cpu', weights_only=False)
msg = data.msg.numpy()
label = data.label.numpy()
attack = data.attack.numpy()
src = data.src.numpy()
dst = data.dst.numpy()
t = data.t.numpy()

# Temporal train/test split (first 70% train, last 30% test)
split_idx = int(len(msg) * 0.7)
indices = np.argsort(t)
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_train, X_test = msg[train_idx], msg[test_idx]
y_train_bin, y_test_bin = label[train_idx], label[test_idx]
y_train_mul, y_test_mul = attack[train_idx], attack[test_idx]
src_train, src_test = src[train_idx], src[test_idx]
dst_train, dst_test = dst[train_idx], dst[test_idx]

print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
print(f"Train attack ratio: {y_train_bin.mean():.3f}")
print(f"Test attack ratio: {y_test_bin.mean():.3f}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_bin_t = torch.LongTensor(y_train_bin)
y_test_bin_t = torch.LongTensor(y_test_bin)
y_train_mul_t = torch.LongTensor(y_train_mul)
y_test_mul_t = torch.LongTensor(y_test_mul)

# ===================== Helper Functions =====================
def evaluate(y_true, y_pred, y_prob=None, task='binary'):
    """Compute comprehensive metrics."""
    results = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    if y_prob is not None:
        try:
            if task == 'binary':
                results['auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                results['auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        except:
            results['auc'] = 0.0
    
    # Per-class F1
    classes = np.unique(y_true)
    per_class = {}
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            per_class[int(c)] = {
                'f1': float(f1_score(y_true == c, y_pred == c, zero_division=0)),
                'support': int(mask.sum())
            }
    results['per_class'] = per_class
    
    return results


def train_pytorch_model(model, X_train, y_train_bin, y_train_mul, 
                        src_train, dst_train, epochs=50, batch_size=1024, lr=1e-3):
    """Train a PyTorch model."""
    dataset = TensorDataset(X_train, y_train_bin, y_train_mul)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    history = {'loss': [], 'binary_loss': [], 'multi_loss': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_bin_loss = 0
        epoch_mul_loss = 0
        n_batches = 0
        
        for xb, yb_bin, yb_mul in loader:
            optimizer.zero_grad()
            
            logits_bin, logits_mul, reps, aux = model(xb)
            loss, loss_dict = model.compute_loss(
                logits_bin, logits_mul, reps, yb_bin, yb_mul, aux,
                contrastive_weight=0.1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss_dict['total_loss']
            epoch_bin_loss += loss_dict['binary_loss']
            epoch_mul_loss += loss_dict['multi_loss']
            n_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)
        history['binary_loss'].append(epoch_bin_loss / n_batches)
        history['multi_loss'].append(epoch_mul_loss / n_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    
    return history


def predict_pytorch_model(model, X, batch_size=2048):
    """Get predictions from a PyTorch model."""
    model.eval()
    all_bin_logits = []
    all_mul_logits = []
    all_reps = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            logits_bin, logits_mul, reps, _ = model(xb)
            all_bin_logits.append(logits_bin)
            all_mul_logits.append(logits_mul)
            all_reps.append(reps)
    
    return (torch.cat(all_bin_logits), torch.cat(all_mul_logits), 
            torch.cat(all_reps))


# ===================== Phase 3a: Sklearn Baselines =====================
print("\n" + "="*60)
print("Phase 3a: Sklearn Baselines")
print("="*60)

results_all = {}

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Random Forest
print("\n[1] Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, 
                            n_jobs=-1, class_weight='balanced')
rf.fit(X_train_scaled, y_train_bin)
rf_time = time.time() - t0

y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)
results_rf_bin = evaluate(y_test_bin, y_pred_rf, y_prob_rf, 'binary')
results_rf_bin['time'] = rf_time
results_all['RandomForest_Binary'] = results_rf_bin
print(f"  Binary F1: {results_rf_bin['f1_macro']:.4f}, AUC: {results_rf_bin.get('auc', 0):.4f}")

# RF Multi-class
rf_multi = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42,
                                   n_jobs=-1, class_weight='balanced')
rf_multi.fit(X_train_scaled, y_train_mul)
y_pred_rf_m = rf_multi.predict(X_test_scaled)
y_prob_rf_m = rf_multi.predict_proba(X_test_scaled)
results_rf_mul = evaluate(y_test_mul, y_pred_rf_m, y_prob_rf_m, 'multi')
results_all['RandomForest_Multi'] = results_rf_mul
print(f"  Multi F1: {results_rf_mul['f1_macro']:.4f}, AUC: {results_rf_mul.get('auc', 0):.4f}")

# 2. Logistic Regression
print("\n[2] Logistic Regression...")
t0 = time.time()
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=1.0)
lr.fit(X_train_scaled, y_train_bin)
lr_time = time.time() - t0
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)
results_lr_bin = evaluate(y_test_bin, y_pred_lr, y_prob_lr, 'binary')
results_lr_bin['time'] = lr_time
results_all['LogisticRegression_Binary'] = results_lr_bin
print(f"  Binary F1: {results_lr_bin['f1_macro']:.4f}")

# LR Multi-class
lr_multi = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=1.0, solver='lbfgs')
lr_multi.fit(X_train_scaled, y_train_mul)
y_pred_lr_m = lr_multi.predict(X_test_scaled)
y_prob_lr_m = lr_multi.predict_proba(X_test_scaled)
results_lr_mul = evaluate(y_test_mul, y_pred_lr_m, y_prob_lr_m, 'multi')
results_all['LogisticRegression_Multi'] = results_lr_mul
print(f"  Multi F1: {results_lr_mul['f1_macro']:.4f}")

# 3. Gradient Boosting
print("\n[3] Gradient Boosting...")
t0 = time.time()
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train_scaled, y_train_bin)
gb_time = time.time() - t0
y_pred_gb = gb.predict(X_test_scaled)
y_prob_gb = gb.predict_proba(X_test_scaled)
results_gb_bin = evaluate(y_test_bin, y_pred_gb, y_prob_gb, 'binary')
results_gb_bin['time'] = gb_time
results_all['GradientBoosting_Binary'] = results_gb_bin
print(f"  Binary F1: {results_gb_bin['f1_macro']:.4f}")

# GB Multi-class
gb_multi = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_multi.fit(X_train_scaled, y_train_mul)
y_pred_gb_m = gb_multi.predict(X_test_scaled)
y_prob_gb_m = gb_multi.predict_proba(X_test_scaled)
results_gb_mul = evaluate(y_test_mul, y_pred_gb_m, y_prob_gb_m, 'multi')
results_all['GradientBoosting_Multi'] = results_gb_mul
print(f"  Multi F1: {results_gb_mul['f1_macro']:.4f}")

# Save feature importances
importances_rf = rf.feature_importances_
importances_gb = gb.feature_importances_
np.save('outputs/feature_importances_rf.npy', importances_rf)
np.save('outputs/feature_importances_gb.npy', importances_gb)

# ===================== Phase 3b: MLP Baseline =====================
print("\n" + "="*60)
print("Phase 3b: MLP Baseline")
print("="*60)

mlp_model = BaselineMLP(input_dim=40, hidden_dim=128, num_classes=10)
t0 = time.time()
mlp_history = train_pytorch_model(mlp_model, X_train_t, y_train_bin_t, y_train_mul_t,
                                   np.arange(len(X_train)), np.arange(len(X_train)),
                                   epochs=50, batch_size=2048, lr=1e-3)
mlp_time = time.time() - t0

mlp_bin_logits, mlp_mul_logits, mlp_reps = predict_pytorch_model(mlp_model, X_test_t)
y_prob_mlp = F.softmax(mlp_bin_logits, dim=-1).numpy()
y_pred_mlp = mlp_bin_logits.argmax(dim=-1).numpy()
y_prob_mlp_m = F.softmax(mlp_mul_logits, dim=-1).numpy()
y_pred_mlp_m = mlp_mul_logits.argmax(dim=-1).numpy()

results_mlp_bin = evaluate(y_test_bin, y_pred_mlp, y_prob_mlp, 'binary')
results_mlp_bin['time'] = mlp_time
results_all['MLP_Binary'] = results_mlp_bin
print(f"  Binary F1: {results_mlp_bin['f1_macro']:.4f}, AUC: {results_mlp_bin.get('auc', 0):.4f}")

results_mlp_mul = evaluate(y_test_mul, y_pred_mlp_m, y_prob_mlp_m, 'multi')
results_all['MLP_Multi'] = results_mlp_mul
print(f"  Multi F1: {results_mlp_mul['f1_macro']:.4f}")

# Save MLP embeddings
np.save('outputs/mlp_embeddings.npy', mlp_reps.numpy())

# ===================== Phase 3c: DIDS-MFL Framework =====================
print("\n" + "="*60)
print("Phase 3c: DIDS-MFL Framework")
print("="*60)

dids_model = DIDS_MFL(input_dim=40, num_groups=5, memory_size=64, 
                       hidden_dim=64, num_heads=3, head_dim=32,
                       num_hops=2, num_classes=10)

t0 = time.time()
dids_history = train_pytorch_model(dids_model, X_train_t, y_train_bin_t, y_train_mul_t,
                                    src_train, dst_train, epochs=30, batch_size=2048, lr=1e-3)
dids_time = time.time() - t0

dids_bin_logits, dids_mul_logits, dids_reps, dids_aux = dids_model(X_test_t)
y_prob_dids = F.softmax(dids_bin_logits, dim=-1).detach().numpy()
y_pred_dids = dids_bin_logits.argmax(dim=-1).detach().numpy()
y_prob_dids_m = F.softmax(dids_mul_logits, dim=-1).detach().numpy()
y_pred_dids_m = dids_mul_logits.argmax(dim=-1).detach().numpy()

results_dids_bin = evaluate(y_test_bin, y_pred_dids, y_prob_dids, 'binary')
results_dids_bin['time'] = dids_time
results_all['DIDSMFL_Binary'] = results_dids_bin
print(f"  Binary F1: {results_dids_bin['f1_macro']:.4f}, AUC: {results_dids_bin.get('auc', 0):.4f}")

results_dids_mul = evaluate(y_test_mul, y_pred_dids_m, y_prob_dids_m, 'multi')
results_dids_mul['time'] = dids_time
results_all['DIDSMFL_Multi'] = results_dids_mul
print(f"  Multi F1: {results_dids_mul['f1_macro']:.4f}")

# Save DIDS-MFL embeddings and intermediate outputs
np.save('outputs/dids_embeddings.npy', dids_reps.detach().numpy())
np.save('outputs/dids_group_assignments.npy', dids_aux['group_assignments'].detach().numpy())

# Save training history
np.save('outputs/dids_training_history.npy', 
        {k: np.array(v) for k, v in dids_history.items()})
np.save('outputs/mlp_training_history.npy', 
        {k: np.array(v) for k, v in mlp_history.items()})

# ===================== Phase 3d: Ablation Studies =====================
print("\n" + "="*60)
print("Phase 3d: Ablation Studies")
print("="*60)

ablation_results = {}

# Ablation 1: SD Only
print("\n[Ablation 1] SD Only...")
sd_model = AblationSDOnly(input_dim=40, num_classes=10)
train_pytorch_model(sd_model, X_train_t, y_train_bin_t, y_train_mul_t,
                    np.arange(len(X_train)), np.arange(len(X_train)),
                    epochs=30, batch_size=2048, lr=1e-3)
sb, sm, _, _ = sd_model(X_test_t)
sb, sm = sb.detach(), sm.detach()
y_pred_sd_bin = sb.argmax(dim=-1).numpy()
y_pred_sd_mul = sm.argmax(dim=-1).numpy()
y_prob_sd_bin = F.softmax(sb, dim=-1).numpy()
y_prob_sd_mul = F.softmax(sm, dim=-1).numpy()
ablation_results['SD_Only'] = {
    'binary': evaluate(y_test_bin, y_pred_sd_bin, y_prob_sd_bin, 'binary'),
    'multi': evaluate(y_test_mul, y_pred_sd_mul, y_prob_sd_mul, 'multi'),
}
print(f"  Binary F1: {ablation_results['SD_Only']['binary']['f1_macro']:.4f}")

# Ablation 2: No Graph Diffusion
print("\n[Ablation 2] No Graph Diffusion...")
ng_model = AblationNoGraph(input_dim=40, num_classes=10)
train_pytorch_model(ng_model, X_train_t, y_train_bin_t, y_train_mul_t,
                    np.arange(len(X_train)), np.arange(len(X_train)),
                    epochs=30, batch_size=2048, lr=1e-3)
sb, sm, _, _ = ng_model(X_test_t)
sb, sm = sb.detach(), sm.detach()
y_pred_ng_bin = sb.argmax(dim=-1).numpy()
y_pred_ng_mul = sm.argmax(dim=-1).numpy()
y_prob_ng_bin = F.softmax(sb, dim=-1).numpy()
y_prob_ng_mul = F.softmax(sm, dim=-1).numpy()
ablation_results['No_Graph'] = {
    'binary': evaluate(y_test_bin, y_pred_ng_bin, y_prob_ng_bin, 'binary'),
    'multi': evaluate(y_test_mul, y_pred_ng_mul, y_prob_ng_mul, 'multi'),
}
print(f"  Binary F1: {ablation_results['No_Graph']['binary']['f1_macro']:.4f}")

# Ablation 3: No Representational Disentanglement
print("\n[Ablation 3] No Representational Disentanglement...")
rd_model = AblationNoRepDis(input_dim=40, num_classes=10)
train_pytorch_model(rd_model, X_train_t, y_train_bin_t, y_train_mul_t,
                    np.arange(len(X_train)), np.arange(len(X_train)),
                    epochs=30, batch_size=2048, lr=1e-3)
sb, sm, _, _ = rd_model(X_test_t)
sb, sm = sb.detach(), sm.detach()
y_pred_rd_bin = sb.argmax(dim=-1).numpy()
y_pred_rd_mul = sm.argmax(dim=-1).numpy()
y_prob_rd_bin = F.softmax(sb, dim=-1).numpy()
y_prob_rd_mul = F.softmax(sm, dim=-1).numpy()
ablation_results['No_RepDis'] = {
    'binary': evaluate(y_test_bin, y_pred_rd_bin, y_prob_rd_bin, 'binary'),
    'multi': evaluate(y_test_mul, y_pred_rd_mul, y_prob_rd_mul, 'multi'),
}
print(f"  Binary F1: {ablation_results['No_RepDis']['binary']['f1_macro']:.4f}")

# Full DIDS-MFL in ablation results
print("  [Full DIDS-MFL already computed above]")
ablation_results['Full_DIDSMFL'] = {
    'binary': results_dids_bin,
    'multi': results_dids_mul,
}

# ===================== Phase 3e: Few-Shot and Unknown Attack Detection =====================
print("\n" + "="*60)
print("Phase 3e: Few-Shot & Unknown Attack Detection")
print("="*60)

# Few-shot evaluation: test on classes with <1000 training samples
few_shot_classes = []
for c in range(10):
    train_count = (y_train_mul == c).sum()
    if train_count < 1000 and c != 2:  # Exclude benign
        few_shot_classes.append(c)

print(f"Few-shot classes (<1000 train samples): {few_shot_classes}")

# For each few-shot class, evaluate performance
few_shot_results = {}
for c in few_shot_classes:
    test_mask = y_test_mul == c
    if test_mask.sum() == 0:
        continue
    
    # Binary: this class vs all others
    y_test_binary_c = (y_test_mul == c).astype(int)
    y_pred_binary_c = (y_pred_dids_m == c).astype(int)
    
    few_shot_results[int(c)] = {
        'train_count': int((y_train_mul == c).sum()),
        'test_count': int(test_mask.sum()),
        'f1': float(f1_score(y_test_binary_c, y_pred_binary_c, zero_division=0)),
        'precision': float(precision_score(y_test_binary_c, y_pred_binary_c, zero_division=0)),
        'recall': float(recall_score(y_test_binary_c, y_pred_binary_c, zero_division=0)),
    }
    print(f"  Attack {c}: F1={few_shot_results[int(c)]['f1']:.4f}, "
          f"Train={few_shot_results[int(c)]['train_count']}, "
          f"Test={few_shot_results[int(c)]['test_count']}")

# Unknown attack detection: leave-one-class-out evaluation
print("\nUnknown Attack Detection (Leave-One-Class-Out):")
unknown_results = {}

for leave_out in range(10):
    if leave_out == 2:  # Skip benign
        continue
    
    # Train without this class
    train_mask = y_train_mul != leave_out
    test_mask = y_test_mul == leave_out
    
    if test_mask.sum() == 0:
        continue
    
    X_tr = X_train_t[train_mask]
    y_tr_bin = y_train_bin_t[train_mask]
    y_tr_mul = y_train_mul_t[train_mask]
    
    # Retrain simplified model for this experiment
    simple_model = BaselineMLP(input_dim=40, hidden_dim=128, num_classes=10)
    dataset_lo = TensorDataset(X_tr, y_tr_bin, y_tr_mul)
    loader_lo = DataLoader(dataset_lo, batch_size=2048, shuffle=True)
    
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    simple_model.train()
    for epoch in range(15):
        for xb, yb_b, yb_m in loader_lo:
            optimizer.zero_grad()
            lb, lm, _, _ = simple_model(xb)
            loss = F.cross_entropy(lb, yb_b) + 0.5 * F.cross_entropy(lm, yb_m)
            loss.backward()
            optimizer.step()
    
    simple_model.eval()
    with torch.no_grad():
        lb_test, lm_test, _, _ = simple_model(X_test_t[test_mask])
        probs = F.softmax(lm_test, dim=-1)
        
        # For unknown attack: if max probability across known classes is low, flag as unknown
        max_known_prob = probs.max(dim=-1).values
        threshold = 0.5
        detected_as_unknown = (max_known_prob < threshold).sum().item()
        
        # Also check if it gets classified correctly or as "close" to correct
        pred = lm_test.argmax(dim=-1).numpy()
        
    unknown_results[int(leave_out)] = {
        'test_count': int(test_mask.sum()),
        'detected_as_unknown': detected_as_unknown,
        'unknown_detection_rate': float(detected_as_unknown / test_mask.sum()),
        'correctly_classified': int((pred == leave_out).sum()),
        'classification_rate': float((pred == leave_out).sum() / test_mask.sum()),
    }
    print(f"  Leave-out {leave_out}: detection_rate={unknown_results[int(leave_out)]['unknown_detection_rate']:.4f}, "
          f"cls_rate={unknown_results[int(leave_out)]['classification_rate']:.4f}")

# Save all results
results_all['ablation'] = ablation_results
results_all['few_shot'] = few_shot_results
results_all['unknown_attack'] = unknown_results

with open('outputs/full_results.json', 'w') as f:
    json.dump(results_all, f, indent=2, default=str)
print("\nSaved: outputs/full_results.json")

# Save comparison table data
comparison = {}
for key in ['RandomForest_Binary', 'RandomForest_Multi', 
            'LogisticRegression_Binary', 'LogisticRegression_Multi',
            'GradientBoosting_Binary', 'GradientBoosting_Multi',
            'MLP_Binary', 'MLP_Multi',
            'DIDSMFL_Binary', 'DIDSMFL_Multi']:
    if key in results_all:
        r = results_all[key]
        comparison[key] = {
            'accuracy': r['accuracy'],
            'f1_macro': r['f1_macro'],
            'f1_weighted': r['f1_weighted'],
            'precision_macro': r['precision_macro'],
            'recall_macro': r['recall_macro'],
            'auc': r.get('auc', None),
        }

with open('outputs/comparison_table.json', 'w') as f:
    json.dump(comparison, f, indent=2)
print("Saved: outputs/comparison_table.json")

print("\n" + "="*60)
print("ALL TRAINING AND EVALUATION COMPLETE!")
print("="*60)
