
import os, sys, json, math, random, types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def install_data_prepare_stub():
    class RealisticCrystalDataset: pass
    m = types.ModuleType('data_prepare')
    m.RealisticCrystalDataset = RealisticCrystalDataset
    sys.modules['data_prepare'] = m


def load_dataset(path):
    install_data_prepare_stub()
    ds = torch.load(path, map_location='cpu', weights_only=False)
    return ds.data_list, ds


def graph_to_features(g):
    x = g.x.float()
    edge_attr = g.edge_attr.float() if getattr(g, 'edge_attr', None) is not None else torch.zeros((g.edge_index.shape[1],2))
    elems = x.argmax(dim=1)
    counts = torch.bincount(elems, minlength=x.shape[1]).float()
    num_nodes = float(x.shape[0])
    num_edges = float(g.edge_index.shape[1] // 2 if g.edge_index.shape[1] % 2 == 0 else g.edge_index.shape[1])
    density = num_edges / max(num_nodes * (num_nodes - 1) / 2, 1)
    feat = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'edge_dist_mean': float(edge_attr[:,0].mean()) if edge_attr.numel() else 0.0,
        'edge_dist_std': float(edge_attr[:,0].std()) if edge_attr.shape[0] > 1 else 0.0,
        'edge_type_mean': float(edge_attr[:,1].mean()) if edge_attr.numel() else 0.0,
        'edge_type_std': float(edge_attr[:,1].std()) if edge_attr.shape[0] > 1 else 0.0,
        'num_unique_elems': int((counts > 0).sum()),
        'max_elem_frac': float(counts.max() / counts.sum()),
        'entropy': float((-(counts[counts>0]/counts.sum()) * torch.log(counts[counts>0]/counts.sum())).sum()),
    }
    for i, c in enumerate(counts.tolist()):
        feat[f'elem_{i}'] = c
    return feat

class Encoder(nn.Module):
    def __init__(self, in_dim=28, hidden=64, layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(last, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))
            last = hidden
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g_mean = global_mean_pool(x, batch)
        g_add = global_add_pool(x, batch)
        g = g_mean + 0.1 * g_add / (torch.bincount(batch).float().unsqueeze(1) + 1e-6)
        z = self.proj(g)
        return z

class ContrastiveModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, d1, d2):
        z1 = self.encoder(d1)
        z2 = self.encoder(d2)
        return F.normalize(z1, dim=1), F.normalize(z2, dim=1)

class Classifier(nn.Module):
    def __init__(self, encoder, hidden=64):
        super().__init__()
        self.encoder = encoder
        dim = encoder.proj[-1].out_features
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
    def forward(self, data):
        z = self.encoder(data)
        return self.head(z).view(-1), z

def augment(batch):
    data = batch.clone()
    x = data.x.clone()
    mask = torch.rand(x.size(0)) < 0.1
    if mask.any():
        x[mask] = 0
    noise = torch.randn_like(x) * 0.01
    data.x = (x + noise).clamp(min=0)
    if getattr(data, 'edge_attr', None) is not None:
        ea = data.edge_attr.clone()
        ea[:,0] = (ea[:,0] + 0.02*torch.randn_like(ea[:,0])).clamp(min=0)
        data.edge_attr = ea
    return data

def nt_xent(z1, z2, temp=0.2):
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temp
    n = z1.size(0)
    mask = torch.eye(2*n, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.arange(n, 2*n, device=z.device)
    targets = torch.cat([targets, torch.arange(n, device=z.device)])
    return F.cross_entropy(sim, targets)

def evaluate_probs(y_true, probs, thr=0.5):
    y_pred = (probs >= thr).astype(int)
    out = {
        'roc_auc': float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float('nan'),
        'ap': float(average_precision_score(y_true, probs)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'threshold': thr,
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    out.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    return out

def infer(model, loader, device):
    model.eval()
    ys, probs, zs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, z = model(batch)
            p = torch.sigmoid(logits)
            probs.extend(p.cpu().numpy().tolist())
            ys.extend(batch.y.view(-1).cpu().numpy().tolist())
            zs.append(z.cpu().numpy())
    return np.array(ys), np.array(probs), (np.concatenate(zs, axis=0) if zs else np.zeros((0,1)))

def main():
    os.makedirs('outputs', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrain_list, pretrain_ds = load_dataset('data/pretrain_data.pt')
    finetune_list, finetune_ds = load_dataset('data/finetune_data.pt')
    candidate_list, candidate_ds = load_dataset('data/candidate_data.pt')

    overview = []
    for name, dslist in [('pretrain', pretrain_list), ('finetune', finetune_list), ('candidate', candidate_list)]:
        rows = [graph_to_features(g) | {'y': int(g.y.item())} for g in dslist]
        df = pd.DataFrame(rows)
        df.describe().to_csv(f'outputs/{name}_describe.csv')
        overview.append({
            'dataset': name,
            'n': len(df),
            'positive_rate': float(df['y'].mean()),
            'num_nodes_mean': float(df['num_nodes'].mean()),
            'num_edges_mean': float(df['num_edges'].mean()),
            'density_mean': float(df['density'].mean()),
            'edge_dist_mean': float(df['edge_dist_mean'].mean()),
            'num_unique_elems_mean': float(df['num_unique_elems'].mean()),
        })
    pd.DataFrame(overview).to_csv('outputs/data_overview.csv', index=False)

    enc = Encoder().to(device)
    contrast = ContrastiveModel(enc).to(device)
    opt = torch.optim.Adam(contrast.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(pretrain_list, batch_size=128, shuffle=True)
    pretrain_log = []
    for epoch in range(1, 31):
        contrast.train()
        losses = []
        for batch in loader:
            b1 = augment(batch).to(device)
            b2 = augment(batch).to(device)
            z1, z2 = contrast(b1, b2)
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        pretrain_log.append({'epoch': epoch, 'loss': float(np.mean(losses))})
    pd.DataFrame(pretrain_log).to_csv('outputs/pretrain_log.csv', index=False)
    torch.save(enc.state_dict(), 'outputs/pretrained_encoder.pt')

    y = np.array([int(g.y.item()) for g in finetune_list])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics, oof_probs = [], np.zeros(len(finetune_list))
    oof_embs = None
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        encoder = Encoder().to(device)
        encoder.load_state_dict(torch.load('outputs/pretrained_encoder.pt', map_location=device))
        model = Classifier(encoder).to(device)
        train_loader = DataLoader([finetune_list[i] for i in tr], batch_size=64, shuffle=True)
        val_loader = DataLoader([finetune_list[i] for i in va], batch_size=128)
        pos = y[tr].sum(); neg = len(tr)-pos
        pos_weight = torch.tensor([neg/max(pos,1)], device=device, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)
        best_ap, best_state = -1, None
        patience = 0
        for epoch in range(1, 61):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits, batch.y.view(-1).float(), pos_weight=pos_weight)
                opt.zero_grad(); loss.backward(); opt.step()
            vy, vp, _ = infer(model, val_loader, device)
            ap = average_precision_score(vy, vp)
            if ap > best_ap:
                best_ap = ap
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break
        model.load_state_dict(best_state)
        vy, vp, vz = infer(model, val_loader, device)
        oof_probs[va] = vp
        if oof_embs is None:
            oof_embs = np.zeros((len(finetune_list), vz.shape[1]))
        oof_embs[va] = vz
        metrics = evaluate_probs(vy, vp, 0.5)
        metrics['fold'] = fold
        fold_metrics.append(metrics)
        torch.save(best_state, f'outputs/fold_{fold}_model.pt')
    pd.DataFrame(fold_metrics).to_csv('outputs/cv_metrics.csv', index=False)
    overall = evaluate_probs(y, oof_probs, 0.5)
    pd.DataFrame([overall]).to_csv('outputs/cv_overall_metrics.csv', index=False)
    pd.DataFrame({'index': np.arange(len(y)), 'y_true': y, 'oof_prob': oof_probs}).to_csv('outputs/oof_predictions.csv', index=False)
    np.save('outputs/oof_embeddings.npy', oof_embs)

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load('outputs/pretrained_encoder.pt', map_location=device))
    final_model = Classifier(encoder).to(device)
    full_loader = DataLoader(finetune_list, batch_size=64, shuffle=True)
    pos = y.sum(); neg = len(y)-pos
    pos_weight = torch.tensor([neg/max(pos,1)], device=device, dtype=torch.float32)
    opt = torch.optim.Adam(final_model.parameters(), lr=8e-4, weight_decay=1e-4)
    for epoch in range(1, 41):
        final_model.train()
        for batch in full_loader:
            batch = batch.to(device)
            logits, _ = final_model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y.view(-1).float(), pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(final_model.state_dict(), 'outputs/final_model.pt')

    cand_loader = DataLoader(candidate_list, batch_size=128)
    cy, cp, cz = infer(final_model, cand_loader, device)
    cand_df = pd.DataFrame({'candidate_index': np.arange(len(cy)), 'true_label': cy, 'pred_prob': cp})
    cand_df = cand_df.sort_values('pred_prob', ascending=False).reset_index(drop=True)
    cand_df['rank'] = np.arange(1, len(cand_df)+1)
    cand_df.to_csv('outputs/candidate_predictions.csv', index=False)
    np.save('outputs/candidate_embeddings.npy', cz)

    top50 = cand_df.head(50)
    top100 = cand_df.head(100)
    discovery = {
        'top10_hits': int(cand_df.head(10)['true_label'].sum()),
        'top20_hits': int(cand_df.head(20)['true_label'].sum()),
        'top50_hits': int(top50['true_label'].sum()),
        'top100_hits': int(top100['true_label'].sum()),
        'candidate_roc_auc': float(roc_auc_score(cy, cp)),
        'candidate_ap': float(average_precision_score(cy, cp)),
        'precision_at_10': float(cand_df.head(10)['true_label'].mean()),
        'precision_at_20': float(cand_df.head(20)['true_label'].mean()),
        'precision_at_50': float(top50['true_label'].mean()),
        'recall_at_50': float(top50['true_label'].sum() / max(cy.sum(),1)),
    }
    pd.DataFrame([discovery]).to_csv('outputs/discovery_metrics.csv', index=False)

    elem_names = list(finetune_ds.elem_to_idx.keys())
    rows = []
    for idx, row in cand_df.head(50).iterrows():
        g = candidate_list[int(row['candidate_index'])]
        elems = g.x.argmax(dim=1).numpy()
        uniq, cnt = np.unique(elems, return_counts=True)
        formula = ';'.join([f'{elem_names[u]}{c}' for u,c in sorted(zip(uniq,cnt), key=lambda x:x[0])])
        rows.append({
            'rank': int(row['rank']),
            'candidate_index': int(row['candidate_index']),
            'pred_prob': float(row['pred_prob']),
            'true_label': int(row['true_label']),
            'num_nodes': int(g.x.shape[0]),
            'num_edges': int(g.edge_index.shape[1]//2 if g.edge_index.shape[1]%2==0 else g.edge_index.shape[1]),
            'formula_proxy': formula,
            'mean_edge_dist': float(g.edge_attr[:,0].mean()),
            'mean_edge_type': float(g.edge_attr[:,1].mean()),
        })
    pd.DataFrame(rows).to_csv('outputs/top50_candidates_detailed.csv', index=False)

    summary = {
        'overview': overview,
        'cv_mean': pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict(),
        'cv_overall': overall,
        'discovery': discovery,
    }
    with open('outputs/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
