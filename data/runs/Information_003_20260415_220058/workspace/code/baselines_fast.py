"""
DIDS-MFL: Fast Baseline Models using LightGBM + simple models
"""
import torch
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler
import json, os, warnings
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

# Temporal split
max_t = t.max()
train_mask = t < max_t * 0.7
val_mask = (t >= max_t * 0.7) & (t < max_t * 0.85)
test_mask = t >= max_t * 0.85

X_train, X_val, X_test = msg[train_mask], msg[val_mask], msg[test_mask]
y_bin_train, y_bin_val, y_bin_test = label[train_mask], label[val_mask], label[test_mask]
y_multi_train, y_multi_val, y_multi_test = attack[train_mask], attack[val_mask], attack[test_mask]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ===== Binary Classification =====
binary_results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42, C=1.0, solver='lbfgs')
lr.fit(X_train_s, y_bin_train)
y_pred = lr.predict(X_test_s)
binary_results['LogisticRegression'] = {
    'accuracy': float(accuracy_score(y_bin_test, y_pred)),
    'f1_weighted': float(f1_score(y_bin_test, y_pred, average='weighted'))
}
print(f"LR Binary: Acc={binary_results['LogisticRegression']['accuracy']:.4f}, F1={binary_results['LogisticRegression']['f1_weighted']:.4f}")

# LightGBM binary
lgb_bin = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
lgb_bin.fit(X_train_s, y_bin_train)
y_pred = lgb_bin.predict(X_test_s)
binary_results['LightGBM'] = {
    'accuracy': float(accuracy_score(y_bin_test, y_pred)),
    'f1_weighted': float(f1_score(y_bin_test, y_pred, average='weighted'))
}
print(f"LGB Binary: Acc={binary_results['LightGBM']['accuracy']:.4f}, F1={binary_results['LightGBM']['f1_weighted']:.4f}")

# ===== Multi-class Classification =====
multi_results = {}
per_type_results = {}

# LR multi-class
lr_m = LogisticRegression(max_iter=500, random_state=42, C=1.0, solver='lbfgs')
lr_m.fit(X_train_s, y_multi_train)
y_pred = lr_m.predict(X_test_s)
multi_results['LogisticRegression'] = {
    'accuracy': float(accuracy_score(y_multi_test, y_pred)),
    'f1_macro': float(f1_score(y_multi_test, y_pred, average='macro')),
    'f1_weighted': float(f1_score(y_multi_test, y_pred, average='weighted'))
}
print(f"LR Multi: Acc={multi_results['LogisticRegression']['accuracy']:.4f}, F1_macro={multi_results['LogisticRegression']['f1_macro']:.4f}")

per_type_lr = {}
for atype in sorted(np.unique(y_multi_test).tolist()):
    mask = y_multi_test == atype
    if mask.sum() > 0:
        type_f1 = f1_score(y_multi_test[mask], y_pred[mask], average='weighted')
        per_type_lr[ATTACK_NAMES[atype]] = {'f1': float(type_f1), 'count': int(mask.sum()),
                                              'is_few_shot': atype in [0,1,4,5,8,9]}
        print(f"  {ATTACK_NAMES[atype]}: F1={type_f1:.4f}")
per_type_results['LogisticRegression'] = per_type_lr

# LightGBM multi-class
lgb_multi = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
lgb_multi.fit(X_train_s, y_multi_train)
y_pred = lgb_multi.predict(X_test_s)
multi_results['LightGBM'] = {
    'accuracy': float(accuracy_score(y_multi_test, y_pred)),
    'f1_macro': float(f1_score(y_multi_test, y_pred, average='macro')),
    'f1_weighted': float(f1_score(y_multi_test, y_pred, average='weighted'))
}
print(f"LGB Multi: Acc={multi_results['LightGBM']['accuracy']:.4f}, F1_macro={multi_results['LightGBM']['f1_macro']:.4f}")

per_type_lgb = {}
for atype in sorted(np.unique(y_multi_test).tolist()):
    mask = y_multi_test == atype
    if mask.sum() > 0:
        type_f1 = f1_score(y_multi_test[mask], y_pred[mask], average='weighted')
        per_type_lgb[ATTACK_NAMES[atype]] = {'f1': float(type_f1), 'count': int(mask.sum()),
                                               'is_few_shot': atype in [0,1,4,5,8,9]}
        print(f"  {ATTACK_NAMES[atype]}: F1={type_f1:.4f}")
per_type_results['LightGBM'] = per_type_lgb

# Save results
with open(os.path.join(OUTPUT_DIR, 'baseline_binary_results.json'), 'w') as f:
    json.dump(binary_results, f, indent=2)
with open(os.path.join(OUTPUT_DIR, 'baseline_multi_results.json'), 'w') as f:
    json.dump(multi_results, f, indent=2)
with open(os.path.join(OUTPUT_DIR, 'baseline_per_type_results.json'), 'w') as f:
    json.dump(per_type_results, f, indent=2)

print("\nBaseline evaluation complete.")