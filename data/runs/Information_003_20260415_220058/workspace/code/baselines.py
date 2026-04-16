"""
DIDS-MFL: Baseline Models (Optimized - faster training)
Implements baseline classifiers for comparison with DIDS-MFL.
"""
import torch
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(WORKSPACE, 'data', 'NF-UNSW-NB15-v2_3d.pt')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')

torch.serialization.add_safe_globals([__import__('torch_geometric.data.temporal').data.temporal.TemporalData])
data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)

msg = data.msg.numpy()
attack = data.attack.numpy()
label = data.label.numpy()
t = data.t.numpy()

ATTACK_NAMES = {
    0: 'Backdoor', 1: 'Analysis', 2: 'Benign',
    3: 'DoS', 4: 'Exploits', 5: 'Fuzzers',
    6: 'Generic', 7: 'Reconnaissance', 8: 'Shellcode', 9: 'Worms'
}

# Split data by time (temporal split)
max_t = t.max()
train_mask = t < max_t * 0.7
val_mask = (t >= max_t * 0.7) & (t < max_t * 0.85)
test_mask = t >= max_t * 0.85

X_train = msg[train_mask]
X_val = msg[val_mask]
X_test = msg[test_mask]

y_bin_train = label[train_mask]
y_bin_val = label[val_mask]
y_bin_test = label[test_mask]

y_multi_train = attack[train_mask]
y_multi_val = attack[val_mask]
y_multi_test = attack[test_mask]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ===================== Binary Classification =====================
binary_results = {}

models = {
    'LinearSVM': LinearSVC(C=1.0, max_iter=5000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=20),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42, early_stopping=True)
}

for name, model in models.items():
    print(f"\nTraining {name} for binary classification...")
    model.fit(X_train_scaled, y_bin_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_bin_test, y_pred)
    prec = precision_score(y_bin_test, y_pred, average='weighted')
    rec = recall_score(y_bin_test, y_pred, average='weighted')
    f1w = f1_score(y_bin_test, y_pred, average='weighted')
    
    binary_results[name] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_weighted': float(f1w)
    }
    print(f"  Accuracy: {acc:.4f}, F1: {f1w:.4f}")

# ===================== Multi-class Classification =====================
multi_results = {}
per_type_results = {}

for name, model in models.items():
    print(f"\nTraining {name} for multi-class classification...")
    model.fit(X_train_scaled, y_multi_train)
    y_pred = model.predict(X_test_scaled)
    
    # Overall metrics
    acc = accuracy_score(y_multi_test, y_pred)
    f1_macro = f1_score(y_multi_test, y_pred, average='macro')
    f1_weighted = f1_score(y_multi_test, y_pred, average='weighted')
    
    multi_results[name] = {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted)
    }
    print(f"  Accuracy: {acc:.4f}, F1_macro: {f1_macro:.4f}, F1_weighted: {f1_weighted:.4f}")
    
    # Per-type F1 scores
    per_type = {}
    for atype in sorted(np.unique(y_multi_test).tolist()):
        mask = y_multi_test == atype
        if mask.sum() > 0:
            type_f1 = f1_score(y_multi_test[mask], y_pred[mask], average='weighted')
            type_acc = accuracy_score(y_multi_test[mask], y_pred[mask])
            aname = ATTACK_NAMES[atype]
            per_type[aname] = {'f1': float(type_f1), 'accuracy': float(type_acc), 
                              'count': int(mask.sum()),
                              'is_few_shot': atype in [0, 1, 4, 5, 8, 9]}
            print(f"    {aname}: F1={type_f1:.4f}, Acc={type_acc:.4f}, n={mask.sum()}")
    
    per_type_results[name] = per_type

# Save results
with open(os.path.join(OUTPUT_DIR, 'baseline_binary_results.json'), 'w') as f:
    json.dump(binary_results, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'baseline_multi_results.json'), 'w') as f:
    json.dump(multi_results, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'baseline_per_type_results.json'), 'w') as f:
    json.dump(per_type_results, f, indent=2)

print("\nBaseline evaluation complete.")