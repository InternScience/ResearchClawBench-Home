"""
DIDS-MFL: Unknown Attack Simulation (Optimized)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import json, os, warnings
warnings.filterwarnings('ignore')

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(WORKSPACE, 'data', 'NF-UNSW-NB15-v2_3d.pt')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')

torch.serialization.add_safe_globals([__import__('torch_geometric.data.temporal').data.temporal.TemporalData])
raw_data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)

ATTACK_NAMES = {0:'Backdoor',1:'Analysis',2:'Benign',3:'DoS',4:'Exploits',5:'Fuzzers',6:'Generic',7:'Reconnaissance',8:'Shellcode',9:'Worms'}

msg_np = raw_data.msg.numpy()
attack_np = raw_data.attack.numpy()
label_np = raw_data.label.numpy()
t_np = raw_data.t.numpy()
src_np = raw_data.src.numpy()

max_t = t_np.max()
train_mask = t_np < max_t * 0.7
test_mask = t_np >= max_t * 0.85

device = torch.device('cpu')

# Unknown attack scenarios
scenarios = {
    'scenario1': {'unknown_types': [0, 9], 'unknown_names': ['Backdoor', 'Worms']},
    'scenario2': {'unknown_types': [1, 8], 'unknown_names': ['Analysis', 'Shellcode']},
}

unknown_results = {}

# Pre-compute feature weights from full training data
mi_scores_full = mutual_info_classif(msg_np[train_mask], attack_np[train_mask], random_state=42)
mi_norm = mi_scores_full / (mi_scores_full.max() + 1e-8)
corr_matrix = np.corrcoef(msg_np[train_mask].T)
avg_corr = np.abs(corr_matrix).mean(axis=1) - 1.0/40
feature_weights = mi_norm / (avg_corr + 0.1)
feature_weights = feature_weights / feature_weights.max()
X_disentangled_full = msg_np * feature_weights[np.newaxis, :]

for scenario_name, scenario in scenarios.items():
    print(f"\nScenario: {scenario_name} - Unknown: {scenario['unknown_names']}")
    unknown_types = scenario['unknown_types']
    
    train_attack = attack_np[train_mask]
    known_mask_train = ~np.isin(train_attack, unknown_types)
    
    X_train_known = X_disentangled_full[train_mask][known_mask_train]
    y_bin_train_known = label_np[train_mask][known_mask_train]
    y_multi_train_known = train_attack[known_mask_train]
    
    remaining_types = sorted(set(np.unique(y_multi_train_known)) - set(unknown_types))
    type_to_new = {old: new for new, old in enumerate(remaining_types)}
    y_multi_mapped = np.array([type_to_new[t] for t in y_multi_train_known])
    num_known = len(remaining_types)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_known)
    X_test_s = scaler.transform(X_disentangled_full[test_mask])
    
    X_train_t = torch.FloatTensor(X_train_s)
    X_test_t = torch.FloatTensor(X_test_s)
    y_bin_train_t = torch.LongTensor(y_bin_train_known)
    y_bin_test_t = torch.LongTensor(label_np[test_mask])
    y_multi_train_t = torch.LongTensor(y_multi_mapped)
    y_multi_test_t = torch.LongTensor(attack_np[test_mask])
    
    # Simple DIDS-MFL model
    class DIDS_MFL_Simple(nn.Module):
        def __init__(self, feat_dim=40, hidden=128, n_classes=num_known, n_binary=2):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.memory = nn.Parameter(torch.randn(n_classes, 32, hidden))
            nn.init.xavier_uniform_(self.memory)
            self.multi_scale = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(hidden, 64), nn.ReLU(),
            )
            self.fusion = nn.Linear(32+64, hidden)
            self.bin_head = nn.Linear(hidden, n_binary)
            self.multi_head = nn.Linear(hidden, n_classes)
        
        def forward(self, x, labels=None):
            h = self.encoder(x)
            # Memory read
            mem_flat = self.memory.reshape(-1, self.memory.shape[-1])
            sim = torch.matmul(h, mem_flat.T)
            attn = F.softmax(sim, dim=1)
            mem_read = torch.matmul(attn, mem_flat)
            h_dis = h + 0.3 * mem_read
            # Multi-scale
            s1 = F.relu(self.multi_scale[0](h_dis))  # 32
            s2 = F.relu(self.multi_scale[2](h_dis))  # 64
            h_fused = self.fusion(torch.cat([s1, s2], dim=1))
            return self.bin_head(h_fused), self.multi_head(h_fused), h_dis
    
    model = DIDS_MFL_Simple()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    n_train = X_train_t.shape[0]
    batch_size = 1024
    
    for epoch in range(15):
        model.train()
        indices = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            optimizer.zero_grad()
            bin_out, multi_out, h_dis = model(X_train_t[batch_idx], y_multi_train_t[batch_idx])
            loss = F.cross_entropy(bin_out, y_bin_train_t[batch_idx]) + \
                   F.cross_entropy(multi_out, y_multi_train_t[batch_idx])
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        bin_out, multi_out, _ = model(X_test_t)
        bin_pred = bin_out.argmax(dim=1).numpy()
    
    bin_test_np = y_bin_test_t.numpy()
    multi_test_np = y_multi_test_t.numpy()
    
    overall_bin_f1 = f1_score(bin_test_np, bin_pred, average='weighted')
    
    unknown_mask = np.isin(multi_test_np, unknown_types)
    if unknown_mask.sum() > 0:
        unknown_det_rate = (bin_pred[unknown_mask] == 1).mean()
        unknown_bin_f1 = f1_score(bin_test_np[unknown_mask], bin_pred[unknown_mask], average='weighted')
    else:
        unknown_det_rate = 0
        unknown_bin_f1 = 0
    
    per_unknown = {}
    for utype in unknown_types:
        mask = multi_test_np == utype
        if mask.sum() > 0:
            det_rate = (bin_pred[mask] == 1).mean()
            per_unknown[ATTACK_NAMES[utype]] = {
                'count': int(mask.sum()),
                'detection_rate': float(det_rate),
                'binary_f1': float(f1_score(bin_test_np[mask], bin_pred[mask], average='weighted'))
            }
            print(f"  Unknown {ATTACK_NAMES[utype]}: det_rate={det_rate:.4f}, n={mask.sum()}")
    
    unknown_results[scenario_name] = {
        'unknown_types': scenario['unknown_names'],
        'overall_binary_f1': float(overall_bin_f1),
        'unknown_detection_rate': float(unknown_det_rate),
        'unknown_binary_f1': float(unknown_bin_f1),
        'per_unknown_type': per_unknown
    }
    print(f"  Overall Binary F1: {overall_bin_f1:.4f}")
    print(f"  Unknown Detection Rate: {unknown_det_rate:.4f}")

with open(os.path.join(OUTPUT_DIR, 'unknown_attack_results.json'), 'w') as f:
    json.dump(unknown_results, f, indent=2)

print("\nUnknown attack simulation complete.")