"""
DIDS-MFL: Unknown Attack Simulation
Simulates unknown attack scenarios by removing certain attack types from training.
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

ATTACK_NAMES = {
    0: 'Backdoor', 1: 'Analysis', 2: 'Benign',
    3: 'DoS', 4: 'Exploits', 5: 'Fuzzers',
    6: 'Generic', 7: 'Reconnaissance', 8: 'Shellcode', 9: 'Worms'
}

msg_np = raw_data.msg.numpy()
attack_np = raw_data.attack.numpy()
label_np = raw_data.label.numpy()
t_np = raw_data.t.numpy()
src_np = raw_data.src.numpy()

max_t = t_np.max()
train_mask = t_np < max_t * 0.7
val_mask = (t_np >= max_t * 0.7) & (t_np < max_t * 0.85)
test_mask = t_np >= max_t * 0.85

device = torch.device('cpu')

# Define unknown attack scenarios
# Scenario 1: Remove Backdoor (type 0) and Worms (type 9) from training
# Scenario 2: Remove Analysis (type 1) and Shellcode (type 8) from training

scenarios = {
    'scenario1': {'unknown_types': [0, 9], 'unknown_names': ['Backdoor', 'Worms']},
    'scenario2': {'unknown_types': [1, 8], 'unknown_names': ['Analysis', 'Shellcode']},
}

unknown_results = {}

for scenario_name, scenario in scenarios.items():
    print(f"\n{'='*60}")
    print(f"Unknown Attack Scenario: {scenario_name}")
    print(f"Unknown types removed from training: {scenario['unknown_names']}")
    
    unknown_types = scenario['unknown_types']
    
    # Create modified training data (remove unknown types)
    train_attack = attack_np[train_mask]
    known_mask = ~np.isin(train_attack, unknown_types)
    
    X_train_known = msg_np[train_mask][known_mask]
    y_multi_train_known = train_attack[known_mask]
    y_bin_train_known = label_np[train_mask][known_mask]
    
    # Map known attack types to new labels (removing unknown types)
    remaining_types = sorted(set(np.unique(y_multi_train_known)) - set(unknown_types))
    type_to_new = {old: new for new, old in enumerate(remaining_types)}
    y_multi_train_mapped = np.array([type_to_new[t] for t in y_multi_train_known])
    num_known_classes = len(remaining_types)
    
    # Statistical disentanglement on known data
    mi_scores = mutual_info_classif(X_train_known, y_multi_train_known, random_state=42)
    mi_norm = mi_scores / (mi_scores.max() + 1e-8)
    corr_matrix = np.corrcoef(X_train_known.T)
    avg_corr = np.abs(corr_matrix).mean(axis=1) - 1.0/40
    feature_weights = mi_norm / (avg_corr + 0.1)
    feature_weights = feature_weights / feature_weights.max()
    
    X_disentangled = msg_np * feature_weights[np.newaxis, :]
    
    scaler = StandardScaler()
    X_train_d = scaler.fit_transform(X_disentangled[train_mask][known_mask])
    X_val_d = scaler.transform(X_disentangled[val_mask])
    X_test_d = scaler.transform(X_disentangled[test_mask])
    
    X_train_t = torch.FloatTensor(X_train_d).to(device)
    X_val_t = torch.FloatTensor(X_val_d).to(device)
    X_test_t = torch.FloatTensor(X_test_d).to(device)
    
    y_bin_train_t = torch.LongTensor(y_bin_train_known).to(device)
    y_bin_val_t = torch.LongTensor(label_np[val_mask]).to(device)
    y_bin_test_t = torch.LongTensor(label_np[test_mask]).to(device)
    
    y_multi_train_t = torch.LongTensor(y_multi_train_mapped).to(device)
    y_multi_val_t = torch.LongTensor(attack_np[val_mask]).to(device)
    y_multi_test_t = torch.LongTensor(attack_np[test_mask]).to(device)
    
    t_train_t = torch.LongTensor(t_np[train_mask][known_mask]).to(device)
    t_val_t = torch.LongTensor(t_np[val_mask]).to(device)
    t_test_t = torch.LongTensor(t_np[test_mask]).to(device)
    
    n_train_t = torch.LongTensor(src_np[train_mask][known_mask]).to(device)
    n_val_t = torch.LongTensor(src_np[val_mask]).to(device)
    n_test_t = torch.LongTensor(src_np[test_mask]).to(device)
    
    dt_train_t = torch.FloatTensor(raw_data.dt.numpy()[train_mask][known_mask]).to(device)
    dt_val_t = torch.FloatTensor(raw_data.dt.numpy()[val_mask]).to(device)
    dt_test_t = torch.FloatTensor(raw_data.dt.numpy()[test_mask]).to(device)
    
    # DIDS-MFL model for this scenario
    class MemoryModule(nn.Module):
        def __init__(self, num_classes, mem_size=32, feat_dim=128):
            super().__init__()
            self.memory = nn.Parameter(torch.randn(num_classes, mem_size, feat_dim))
            nn.init.xavier_uniform_(self.memory)
        def forward(self, x, labels=None):
            mem_flat = self.memory.reshape(-1, self.memory.shape[-1])
            sim = torch.matmul(x, mem_flat.T)
            attention = F.softmax(sim, dim=1)
            read_out = torch.matmul(attention, mem_flat)
            return read_out, attention
    
    class DynamicGraphDiffusion(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc_diffusion = nn.Linear(in_dim, out_dim)
            self.fc_temporal = nn.Linear(1, out_dim)
            self.fc_combine = nn.Linear(out_dim * 2, out_dim)
        def forward(self, x, node_ids, timestamps, durations):
            t_norm = (timestamps.float() - timestamps.float().min()) / (timestamps.float().max() - timestamps.float().min() + 1e-8)
            t_enc = self.fc_temporal(t_norm.unsqueeze(1))
            spatial_enc = self.fc_diffusion(x)
            combined = self.fc_combine(torch.cat([spatial_enc, t_enc], dim=1))
            return combined
    
    class MultiScaleFusion(nn.Module):
        def __init__(self, in_dim, scales=[32, 64, 96]):
            super().__init__()
            self.scale_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(in_dim, s), nn.ReLU(), nn.Linear(s, s))
                for s in scales
            ])
            self.fusion = nn.Linear(sum(scales), in_dim)
        def forward(self, x):
            scale_outputs = [layer(x) for layer in self.scale_layers]
            fused = self.fusion(torch.cat(scale_outputs, dim=1))
            return fused
    
    class DIDS_MFL_Scenario(nn.Module):
        def __init__(self, feat_dim=40, hidden_dim=128, num_classes=num_known_classes, num_binary=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.memory = MemoryModule(num_classes, mem_size=32, feat_dim=hidden_dim)
            self.graph_diffusion = DynamicGraphDiffusion(hidden_dim, hidden_dim)
            self.multi_scale = MultiScaleFusion(hidden_dim, scales=[32, 64, 96])
            self.binary_head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_binary)
            )
            self.multi_head = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
            )
        
        def forward(self, x, node_ids=None, timestamps=None, durations=None, labels=None):
            h = self.encoder(x)
            mem_read, mem_attn = self.memory(h, labels)
            h_dis = h + 0.3 * mem_read
            if timestamps is not None:
                h_diff = self.graph_diffusion(h_dis, node_ids, timestamps, durations)
                h_combined = h_dis + 0.2 * h_diff
            else:
                h_combined = h_dis
            h_fused = self.multi_scale(h_combined)
            bin_out = self.binary_head(h_fused)
            multi_out = self.multi_head(h_fused)
            return bin_out, multi_out, h_dis
        
        def disentangle_loss(self, h_dis, labels):
            unique_labels = labels.unique()
            total_corr = 0.0
            for c in unique_labels:
                mask = labels == c
                if mask.sum() > 10:
                    h_c = h_dis[mask]
                    corr = torch.corrcoef(h_c.T)
                    total_corr += torch.abs(corr).mean()
            return total_corr / len(unique_labels)
    
    model = DIDS_MFL_Scenario().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_f1 = 0
    best_state = None
    n_train = X_train_t.shape[0]
    batch_size = 512
    
    for epoch in range(20):
        model.train()
        indices = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            x_b = X_train_t[batch_idx]
            yb_b = y_bin_train_t[batch_idx]
            ym_b = y_multi_train_t[batch_idx]
            t_b = t_train_t[batch_idx]
            n_b = n_train_t[batch_idx]
            d_b = dt_train_t[batch_idx]
            
            optimizer.zero_grad()
            bin_out, multi_out, h_dis = model(x_b, n_b, t_b, d_b, ym_b)
            
            bin_loss = F.cross_entropy(bin_out, yb_b)
            multi_loss = F.cross_entropy(multi_out, ym_b)
            dis_loss = model.disentangle_loss(h_dis, ym_b)
            loss = bin_loss + multi_loss + 0.1 * dis_loss
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation - check binary performance (unknown attacks should still be detected as attacks)
        model.eval()
        with torch.no_grad():
            bin_out, _, _ = model(X_val_t, n_val_t, t_val_t, dt_val_t)
            bin_pred = bin_out.argmax(dim=1).cpu().numpy()
            bin_val_np = y_bin_val_t.cpu().numpy()
            bin_f1 = f1_score(bin_val_np, bin_pred, average='weighted')
        
        if bin_f1 > best_val_f1:
            best_val_f1 = bin_f1
            best_state = model.state_dict().copy()
    
    model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        bin_out, multi_out, h_dis = model(X_test_t, n_test_t, t_test_t, dt_test_t)
        
        bin_pred = bin_out.argmax(dim=1).cpu().numpy()
        multi_pred_mapped = multi_out.argmax(dim=1).cpu().numpy()
    
    # Map multi predictions back to original attack types
    new_to_type = {new: old for old, new in type_to_new.items()}
    multi_pred_original = np.array([new_to_type.get(p, -1) for p in multi_pred_mapped])
    
    bin_test_np = y_bin_test_t.cpu().numpy()
    multi_test_np = y_multi_test_t.cpu().numpy()
    
    # Binary metrics overall
    overall_bin_acc = accuracy_score(bin_test_np, bin_pred)
    overall_bin_f1 = f1_score(bin_test_np, bin_pred, average='weighted')
    
    # Unknown attack binary detection
    unknown_mask = np.isin(multi_test_np, unknown_types)
    known_mask_test = ~unknown_mask
    
    if unknown_mask.sum() > 0:
        unknown_bin_f1 = f1_score(bin_test_np[unknown_mask], bin_pred[unknown_mask], average='weighted')
        unknown_detection_rate = (bin_pred[unknown_mask] == 1).mean()  # how many detected as attack
    else:
        unknown_bin_f1 = 0
        unknown_detection_rate = 0
    
    # Known attack metrics
    known_bin_f1 = f1_score(bin_test_np[known_mask_test], bin_pred[known_mask_test], average='weighted')
    
    print(f"  Overall Binary: Acc={overall_bin_acc:.4f}, F1={overall_bin_f1:.4f}")
    print(f"  Unknown Attack Detection Rate: {unknown_detection_rate:.4f}")
    print(f"  Unknown Attack Binary F1: {unknown_bin_f1:.4f}")
    print(f"  Known Data Binary F1: {known_bin_f1:.4f}")
    
    # Per-type analysis for unknown attacks
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
            print(f"  Unknown {ATTACK_NAMES[utype]}: detection_rate={det_rate:.4f}, n={mask.sum()}")
    
    unknown_results[scenario_name] = {
        'unknown_types': scenario['unknown_names'],
        'overall_binary': {'accuracy': float(overall_bin_acc), 'f1_weighted': float(overall_bin_f1)},
        'unknown_detection_rate': float(unknown_detection_rate),
        'unknown_binary_f1': float(unknown_bin_f1),
        'known_binary_f1': float(known_bin_f1),
        'per_unknown_type': per_unknown
    }

# Save results
with open(os.path.join(OUTPUT_DIR, 'unknown_attack_results.json'), 'w') as f:
    json.dump(unknown_results, f, indent=2)

print("\nUnknown attack simulation complete.")