import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import TemporalDataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import os

data_path = '../data/NF-UNSW-NB15-v2_3d.pt'
splits_dir = '../outputs/'

data = torch.load(data_path, weights_only=False)

train_idx = torch.load(os.path.join(splits_dir, 'train_idx.pt'))
val_idx = torch.load(os.path.join(splits_dir, 'val_idx.pt'))
test_idx = torch.load(os.path.join(splits_dir, 'test_idx.pt'))

# Subsample for speed
sub_train = train_idx[::10]  # ~12k
sub_val = val_idx[::5]  # ~3k
sub_test = test_idx[::5]  # ~3k

train_data = data[sub_train]
val_data = data[sub_val]
test_data = data[sub_test]

print('Subsample sizes:', len(train_data), len(val_data), len(test_data))

# Simple ML on feats
X_train = train_data.msg.numpy()
y_train_bin = train_data.label.numpy()
y_train_multi = train_data.attack.numpy()

X_test = test_data.msg.numpy()
y_test_bin = test_data.label.numpy()
y_test_multi = test_data.attack.numpy()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf_bin = RandomForestClassifier(n_estimators=50, random_state=42, max_samples=0.5)
rf_bin.fit(X_train, y_train_bin)
rf_bin_pred = rf_bin.predict(X_test)
rf_bin_acc = accuracy_score(y_test_bin, rf_bin_pred)
rf_bin_f1 = f1_score(y_test_bin, rf_bin_pred, average='macro')

rf_multi = RandomForestClassifier(n_estimators=50, random_state=42, max_samples=0.5)
rf_multi.fit(X_train, y_train_multi)
rf_multi_pred = rf_multi.predict(X_test)
rf_multi_acc = accuracy_score(y_test_multi, rf_multi_pred)
rf_multi_f1 = f1_score(y_test_multi, rf_multi_pred, average='macro')

results = {
    'rf_bin_sub': {'acc': float(rf_bin_acc), 'f1_macro': float(rf_bin_f1)},
    'rf_multi_sub': {'acc': float(rf_multi_acc), 'f1_macro': float(rf_multi_f1)},
}

print(json.dumps(results))

torch.save(results, '../outputs/baselines_simple_sub.json')

print('Per class F1 multi:', f1_score(y_test_multi, rf_multi_pred, average=None))
