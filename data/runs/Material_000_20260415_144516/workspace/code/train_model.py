"""
Optimized Altermagnetic Discovery Training
Minimal implementation for fast execution
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prepare

np.random.seed(42); torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.node_emb = nn.Linear(node_dim, hidden)
        self.edge_emb = nn.Linear(edge_dim, hidden)
        self.conv1 = nn.Linear(hidden * 3, hidden)
        self.conv2 = nn.Linear(hidden * 3, hidden)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.node_emb(x))
        ea = F.relu(self.edge_emb(edge_attr))
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2)]:
            src, dst = edge_index
            msg = F.relu(conv(torch.cat([x[src], x[dst], ea], -1)))
            aggr = torch.zeros_like(x)
            aggr.index_add_(0, dst, msg)
            x = bn(x + aggr)
        out = torch.zeros(batch.max() + 1, self.hidden * 2, device=x.device)
        for i in range(batch.max() + 1):
            mask = batch == i
            if mask.any(): out[i, :self.hidden] = x[mask].mean(0); out[i, self.hidden:] = x[mask].max(0)[0]
        return out

class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        self.fc = nn.Sequential(nn.Linear(encoder.hidden * 2, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, x, ei, ea, batch):
        return self.fc(self.enc(x, ei, ea, batch)).squeeze()
    def prob(self, x, ei, ea, batch):
        return torch.sigmoid(self.forward(x, ei, ea, batch))

def make_batch(data, device):
    xs, eis, eas, ys, batch = [], [], [], [], []
    off = 0
    for i, d in enumerate(data):
        xs.append(d.x.float())
        eis.append(d.edge_index + off)
        eas.append(d.edge_attr.float())
        if hasattr(d, 'y'): ys.append(d.y.float())
        batch.extend([i] * d.num_nodes)
        off += d.num_nodes
    return (torch.cat(xs).to(device), torch.cat(eis, 1).to(device), 
            torch.cat(eas).to(device), torch.tensor(batch).to(device),
            torch.stack(ys).to(device) if ys else None)

def pretrain(enc, data, epochs=10, bs=64):
    model = nn.Sequential(enc, nn.Linear(enc.hidden * 2, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    targets = torch.tensor([d.num_nodes for d in data], dtype=torch.float).to(device)
    losses = []
    for e in range(epochs):
        total = 0
        idx = np.random.permutation(len(data))
        for i in range(0, len(data), bs):
            b = idx[i:i+bs]
            x, ei, ea, batch, _ = make_batch([data[j] for j in b], device)
            loss = F.mse_loss(model[1](model[0](x, ei, ea, batch)), targets[b])
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        losses.append(total / (len(data)//bs))
        if (e+1) % 5 == 0: print(f"  Pretrain ep {e+1}, loss: {losses[-1]:.4f}")
    return losses

def finetune(model, train, val, epochs=30, bs=32, pw=8.0):
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]).to(device))
    best_f1, losses, metrics = 0, [], []
    for e in range(epochs):
        model.train(); total = 0
        idx = np.random.permutation(len(train))
        for i in range(0, len(train), bs):
            x, ei, ea, batch, y = make_batch([train[j] for j in idx[i:i+bs]], device)
            loss = crit(model(x, ei, ea, batch), y)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        losses.append(total / (len(train)//bs))
        
        model.eval(); probs, labs = [], []
        with torch.no_grad():
            for i in range(0, len(val), bs):
                x, ei, ea, batch, y = make_batch(val[i:i+bs], device)
                probs.extend(model.prob(x, ei, ea, batch).cpu().numpy())
                labs.extend(y.cpu().numpy())
        probs, labs, preds = np.array(probs), np.array(labs), (np.array(probs) > 0.5).astype(int)
        m = {'f1': f1_score(labs, preds, zero_division=0), 'auc': roc_auc_score(labs, probs)}
        metrics.append(m)
        if m['f1'] > best_f1: best_f1 = m['f1']; torch.save(model.state_dict(), 'outputs/best.pt')
        if (e+1) % 10 == 0: print(f"  Finetune ep {e+1}, loss: {losses[-1]:.4f}, F1: {m['f1']:.4f}, AUC: {m['auc']:.4f}")
    return losses, metrics, best_f1

def main():
    print("="*60 + "\nAltermagnetic Discovery\n" + "="*60)
    os.makedirs('outputs', exist_ok=True)
    
    print("\n1. Loading data...")
    ptr = torch.load('data/pretrain_data.pt', weights_only=False).data_list
    ft = torch.load('data/finetune_data.pt', weights_only=False).data_list
    cand = torch.load('data/candidate_data.pt', weights_only=False).data_list
    
    ft_labs = [d.y.item() for d in ft]
    cand_labs = [d.y.item() for d in cand]
    print(f"   Pretrain: {len(ptr)}, Finetune: {len(ft)} ({sum(ft_labs)} pos), Candidate: {len(cand)} ({sum(cand_labs)} pos)")
    
    # Split
    idx = np.random.permutation(len(ft))
    tr, val = [ft[i] for i in idx[:int(0.8*len(ft))]], [ft[i] for i in idx[int(0.8*len(ft)):]]
    
    print("\n2. Pretraining...")
    enc = Encoder().to(device)
    pre_losses = pretrain(enc, ptr, epochs=10, bs=64)
    
    print("\n3. Fine-tuning...")
    model = Classifier(enc).to(device)
    tr_labs = [d.y.item() for d in tr]
    pw = (1 - sum(tr_labs)/len(tr_labs)) / (sum(tr_labs)/len(tr_labs)) * 5
    tr_losses, val_mets, best_f1 = finetune(model, tr, val, epochs=30, bs=32, pw=pw)
    print(f"   Best F1: {best_f1:.4f}")
    
    model.load_state_dict(torch.load('outputs/best.pt'))
    
    print("\n4. Discovery...")
    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, len(cand), 32):
            x, ei, ea, batch, _ = make_batch(cand[i:i+32], device)
            probs.extend(model.prob(x, ei, ea, batch).cpu().numpy())
    probs = np.array(probs)
    
    top50 = np.argsort(probs)[::-1][:50]
    tp = sum(cand_labs[i] for i in top50)
    print(f"   Top-50: {tp} TPs, Precision: {tp/50:.4f}, Recall: {tp/sum(cand_labs):.4f}")
    
    pred = (probs > 0.5).astype(int)
    print(f"   Overall: Acc={(pred == np.array(cand_labs)).mean():.4f}, F1={f1_score(cand_labs, pred, zero_division=0):.4f}, AUC={roc_auc_score(cand_labs, probs):.4f}")
    
    res = {
        'pre_losses': pre_losses, 'train_losses': tr_losses, 'val_metrics': val_mets,
        'probs': probs.tolist(), 'cand_labels': cand_labs, 'top50': top50.tolist(),
        'tp_at_50': int(tp), 'prec_at_50': tp/50, 'rec_at_50': tp/sum(cand_labs),
        'auc': roc_auc_score(cand_labs, probs), 'f1': f1_score(cand_labs, pred, zero_division=0)
    }
    with open('outputs/results.json', 'w') as f: json.dump(res, f, indent=2)
    print("\n   Saved to outputs/results.json")
    return res

if __name__ == '__main__':
    main()
