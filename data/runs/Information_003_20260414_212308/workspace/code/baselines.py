import torch
import torch.nn.functional as F
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import SAGEConv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from collections import Counter

data = torch.load('../data/NF-UNSW-NB15-v2_3d.pt', weights_only=False)

# Load splits
train_idx = torch.load('../outputs/train_idx.pt')
val_idx = torch.load('../outputs/val_idx.pt')
test_idx = torch.load('../outputs/test_idx.pt')

train_data = data[train_idx]
val_data = data[val_idx]
test_data = data[test_idx]

def get_edge_labels(data, task='binary'):
    if task == 'binary':
        return data.label.numpy()
    else:
        return data.attack.numpy()

# Feature only baselines
X_train = train_data.msg.numpy()
y_train_bin = get_edge_labels(train_data, 'binary')
y_train_multi = get_edge_labels(train_data, 'multi')

X_val = val_data.msg.numpy()
y_val_bin = get_edge_labels(val_data, 'binary')
y_val_multi = get_edge_labels(val_data, 'multi')

X_test = test_data.msg.numpy()
y_test_bin = get_edge_labels(test_data, 'binary')
y_test_multi = get_edge_labels(test_data, 'multi')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# SVM binary
svm_bin = SVC(kernel='rbf', probability=True)
svm_bin.fit(X_train_scaled, y_train_bin)
svm_bin_pred = svm_bin.predict(X_test_scaled)
svm_bin_acc = accuracy_score(y_test_bin, svm_bin_pred)
svm_bin_f1 = f1_score(y_test_bin, svm_bin_pred, average='macro')

# SVM multi
svm_multi = SVC(kernel='rbf', probability=True)
svm_multi.fit(X_train_scaled, y_train_multi)
svm_multi_pred = svm_multi.predict(X_test_scaled)
svm_multi_acc = accuracy_score(y_test_multi, svm_multi_pred)
svm_multi_f1 = f1_score(y_test_multi, svm_multi_pred, average='macro')

# RF binary
rf_bin = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bin.fit(X_train, y_train_bin)
rf_bin_pred = rf_bin.predict(X_test)
rf_bin_acc = accuracy_score(y_test_bin, rf_bin_pred)
rf_bin_f1 = f1_score(y_test_bin, rf_bin_pred, average='macro')

# RF multi
rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multi.fit(X_train, y_train_multi)
rf_multi_pred = rf_multi.predict(X_test)
rf_multi_acc = accuracy_score(y_test_multi, rf_multi_pred)
rf_multi_f1 = f1_score(y_test_multi, rf_multi_pred, average='macro')

results = {
    'svm_bin': {'acc': float(svm_bin_acc), 'f1_macro': float(svm_bin_f1)},
    'svm_multi': {'acc': float(svm_multi_acc), 'f1_macro': float(svm_multi_f1)},
    'rf_bin': {'acc': float(rf_bin_acc), 'f1_macro': float(rf_bin_f1)},
    'rf_multi': {'acc': float(rf_multi_acc), 'f1_macro': float(rf_multi_f1)},
}

print(json.dumps(results))
torch.save(results, '../outputs/baselines_simple.json')

# Per class F1 for multi RF
print(classification_report(y_test_multi, rf_multi_pred))
