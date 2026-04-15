"""GNN-based Altermagnetic Discovery with proper message passing"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prepare

np.random.seed(42); torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class GNN(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden=64):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden)
        self.edge_emb = nn.Linear(edge_dim, hidden)
        # Two message passing layers
        self.conv1 = nn.Linear(hidden * 3, hidden)
        self.conv2 = nn.Linear(hidden * 3, hidden)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x, edge_index, edge_attr, batch_idx):
        # Initial embedding
        x = F.relu(self.node_emb(x))
        edge_attr = F.relu(self.edge_emb(edge_attr))
        
        # Message passing 1
        src, dst = edge_index
        msg = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        msg = F.relu(self.conv1(msg))
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, dst, msg)
        x = x + aggr
        
        # Message passing 2
        msg = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        msg = F.relu(self.conv2(msg))
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, dst, msg)
        x = x + aggr
        
        # Global pooling
        num_graphs = batch_idx.max().item() + 1
        out = torch.zeros(num_graphs, self.classifier[0].in_features, device=x.device)
        for i in range(num_graphs):
            mask = batch_idx == i
            if mask.any():
                x_i = x[mask]
                out[i, :x_i.size(1)] = x_i.mean(0)
                out[i, x_i.size(1):] = x_i.max(0)[0]
        return self.classifier(out).squeeze(-1)
    
    def prob(self, x, ei, ea, batch):
        return torch.sigmoid(self.forward(x, ei, ea, batch))

def make_batch(data_list, device):
    xs, eis, eas, ys, batch = [], [], [], [], []
    node_offset = 0
    for i, data in enumerate(data_list):
        xs.append(data.x.float())
        eis.append(data.edge_index + node_offset)
        eas.append(data.edge_attr.float())
        if hasattr(data, 'y') and data.y is not None:
            ys.append(data.y.float())
        batch.extend([i] * data.num_nodes)
        node_offset += data.num_nodes
    
    x = torch.cat(xs).to(device)
    edge_index = torch.cat(eis, dim=1).to(device)
    edge_attr = torch.cat(eas).to(device)
    batch_idx = torch.tensor(batch, dtype=torch.long).to(device)
    y = torch.stack(ys).squeeze().to(device) if ys else None
    return x, edge_index, edge_attr, batch_idx, y

# Load data
print("\nLoading data...")
pretrain_data = torch.load('data/pretrain_data.pt', weights_only=False).data_list
finetune_data = torch.load('data/finetune_data.pt', weights_only=False).data_list
candidate_data = torch.load('data/candidate_data.pt', weights_only=False).data_list

print(f"Pretrain: {len(pretrain_data)}, Finetune: {len(finetune_data)}, Candidate: {len(candidate_data)}")

# Count labels
finetune_labels = [d.y.item() for d in finetune_data]
candidate_labels = [d.y.item() for d in candidate_data]
print(f"Finetune positives: {sum(finetune_labels)}/{len(finetune_labels)}")
print(f"Candidate positives: {sum(candidate_labels)}/{len(candidate_labels)}")

# Split finetune
indices = np.random.permutation(len(finetune_data))
split = int(0.8 * len(finetune_data))
train_data = [finetune_data[i] for i in indices[:split]]
val_data = [finetune_data[i] for i in indices[split:]]
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Initialize model
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

# Calculate class weight
pos_ratio = sum([d.y.item() for d in train_data]) / len(train_data)
pos_weight = torch.tensor([(1 - pos_ratio) / pos_ratio * 5]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"Using pos_weight: {pos_weight.item():.2f}")

# Training
print("\nTraining...")
best_f1 = 0
best_state = None

for epoch in range(15):
    model.train()
    train_loss = 0
    indices = np.random.permutation(len(train_data))
    
    for i in range(0, len(train_data), 32):
        batch_data = [train_data[j] for j in indices[i:i+32]]
        x, edge_index, edge_attr, batch_idx, y = make_batch(batch_data, device)
        
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_attr, batch_idx)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        x, edge_index, edge_attr, batch_idx, y = make_batch(val_data, device)
        val_probs = model.prob(x, edge_index, edge_attr, batch_idx).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
        val_labels = y.cpu().numpy()
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss={train_loss/(len(train_data)//32):.4f}, Val F1={val_f1:.4f}, Val AUC={val_auc:.4f}")

print(f"\nBest Val F1: {best_f1:.4f}")

# Load best model
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

# Discovery
print("\nDiscovery on candidates...")
model.eval()
candidate_probs = []
with torch.no_grad():
    for i in range(0, len(candidate_data), 32):
        x, edge_index, edge_attr, batch_idx, _ = make_batch(candidate_data[i:i+32], device)
        probs = model.prob(x, edge_index, edge_attr, batch_idx)
        candidate_probs.extend(probs.cpu().numpy())

candidate_probs = np.array(candidate_probs)
candidate_preds = (candidate_probs > 0.5).astype(int)

# Top-50 discovery
top_k = 50
top_indices = np.argsort(candidate_probs)[::-1][:top_k]
top_labels = [candidate_labels[i] for i in top_indices]
tp_at_k = sum(top_labels)

print(f"\n{'='*50}")
print("DISCOVERY RESULTS")
print(f"{'='*50}")
print(f"Top-{top_k} discoveries:")
for i, idx in enumerate(top_indices[:10]):
    print(f"  {i+1}. Candidate {idx}: prob={candidate_probs[idx]:.4f}, true_label={candidate_labels[idx]}")

print(f"\nTop-{top_k} Statistics:")
print(f"  True Positives: {tp_at_k}/{top_k} ({tp_at_k/top_k*100:.1f}%)")
print(f"  Precision@{top_k}: {tp_at_k/top_k:.4f}")
print(f"  Recall@{top_k}: {tp_at_k/sum(candidate_labels):.4f}")

print(f"\nOverall Performance:")
print(f"  Accuracy: {(candidate_preds == np.array(candidate_labels)).mean():.4f}")
print(f"  Precision: {precision_score(candidate_labels, candidate_preds, zero_division=0):.4f}")
print(f"  Recall: {recall_score(candidate_labels, candidate_preds, zero_division=0):.4f}")
print(f"  F1: {f1_score(candidate_labels, candidate_preds, zero_division=0):.4f}")
print(f"  AUC: {roc_auc_score(candidate_labels, candidate_probs):.4f}")
print(f"  AUPRC: {average_precision_score(candidate_labels, candidate_probs):.4f}")

# Save results
os.makedirs('outputs', exist_ok=True)
results = {
    'candidate_probs': candidate_probs.tolist(),
    'candidate_labels': [int(x) for x in candidate_labels],
    'top_k_indices': [int(x) for x in top_indices],
    'top_k_labels': [int(x) for x in top_labels],
    'true_positives_at_k': int(tp_at_k),
    'precision_at_k': float(tp_at_k/top_k),
    'recall_at_k': float(tp_at_k/sum(candidate_labels)),
    'overall_accuracy': float((candidate_preds == np.array(candidate_labels)).mean()),
    'overall_precision': float(precision_score(candidate_labels, candidate_preds, zero_division=0)),
    'overall_recall': float(recall_score(candidate_labels, candidate_preds, zero_division=0)),
    'overall_f1': float(f1_score(candidate_labels, candidate_preds, zero_division=0)),
    'overall_auc': float(roc_auc_score(candidate_labels, candidate_probs)),
    'overall_auprc': float(average_precision_score(candidate_labels, candidate_probs)),
    'val_f1': float(best_f1)
}

with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to outputs/results.json")
