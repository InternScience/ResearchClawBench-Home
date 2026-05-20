import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fourier KAN Module
class FourierKAN(nn.Module):
    def __init__(self, in_features, out_features, n_freqs=8):
        super(FourierKAN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_freqs = n_freqs
        
        self.freqs = nn.Parameter(torch.randn(n_freqs, in_features))
        self.coeffs = nn.Parameter(torch.randn(n_freqs * 2, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        # x: [batch, in_features]
        cos_terms = torch.cos(2 * np.pi * self.freqs @ x.T).T  # [batch, n_freqs]
        sin_terms = torch.sin(2 * np.pi * self.freqs @ x.T).T  # [batch, n_freqs]
        basis = torch.cat([cos_terms, sin_terms], dim=1)  # [batch, 2*n_freqs]
        out = basis @ self.coeffs + self.bias
        return out

# Atom and Bond Feature Extraction
def atom_features(atom):
    features = []
    features.append(atom.GetAtomicNum())
    features.append(atom.GetFormalCharge())
    features.append(atom.GetDegree())
    features.append(atom.GetHybridization())
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetNumImplicitHs())
    return np.array(features, dtype=np.float32)

def bond_features(bond):
    features = []
    bt = bond.GetBondType()
    features.append(bt == Chem.rdchem.BondType.SINGLE)
    features.append(bt == Chem.rdchem.BondType.DOUBLE)
    features.append(bt == Chem.rdchem.BondType.TRIPLE)
    features.append(bt == Chem.rdchem.BondType.AROMATIC)
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    return np.array(features, dtype=np.float32)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atoms
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    x = torch.tensor(np.array(atom_feats), dtype=torch.float)
    
    # Edges
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        bf = bond_features(bond)
        edge_attr.append(bf)
        edge_attr.append(bf)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# KA-GNN Layer
class KAGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(KAGNNLayer, self).__init__(aggr='mean')
        self.message_kan = FourierKAN(in_channels + edge_dim, out_channels)
        self.update_kan = FourierKAN(in_channels + out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        msg_input = torch.cat([x_j, edge_attr], dim=1)
        return self.message_kan(msg_input)
    
    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_kan(update_input)

# KA-GNN Model
class KAGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, num_classes=1):
        super(KAGNN, self).__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            KAGNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.pool = global_mean_pool
        self.classifier = FourierKAN(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
        
        x = self.pool(x, batch)
        out = self.classifier(x)
        return out

# Data Loading and Preparation
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Identify label columns (exclude metadata and smiles)
    meta_cols = {'smiles', 'ID', 'num', 'name', 'mol_id', 'CID', 'index'}
    label_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in [np.int64, np.float64, 'int64', 'float64']]
    if not label_cols:
        # fallback to common names
        if 'label' in df.columns:
            label_cols = ['label']
        elif 'activity' in df.columns:
            label_cols = ['activity']
        else:
            label_cols = [c for c in df.columns if c != 'smiles'][:1]
    graphs = []
    for _, row in df.iterrows():
        graph = smiles_to_graph(row['smiles'])
        if graph is not None:
            y = torch.tensor([row[c] for c in label_cols], dtype=torch.float)
            graph.y = y
            graphs.append(graph)
    return graphs, label_cols

# Training and Evaluation
def train_model(model, train_loader, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds.extend(torch.sigmoid(out).cpu().numpy())
            targets.extend(data.y.cpu().numpy())
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, (preds > 0.5).astype(int))
    return auc, acc

# Main Pipeline
def main():
    datasets = ['bace', 'bbbp', 'clintox', 'hiv', 'muv']
    results = {}
    
    for ds_name in datasets:
        print(f'\n=== Processing {ds_name} ===')
        csv_path = f'data/{ds_name}.csv'
        graphs, label_cols = load_dataset(csv_path)
        num_classes = len(label_cols)
        
        # Split
        n = len(graphs)
        train_size = int(0.8 * n)
        train_graphs = graphs[:train_size]
        test_graphs = graphs[train_size:]
        
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=32)
        
        # Model
        node_dim = graphs[0].x.shape[1]
        edge_dim = graphs[0].edge_attr.shape[1]
        model = KAGNN(node_dim, edge_dim, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        # Train
        train_model(model, train_loader, optimizer, criterion, epochs=30)
        
        # Evaluate
        auc, acc = evaluate_model(model, test_loader)
        results[ds_name] = {'AUC': auc, 'Accuracy': acc, 'num_tasks': num_classes}
        print(f'{ds_name} - AUC: {auc:.4f}, Accuracy: {acc:.4f}')
    
    # Results table
    df_results = pd.DataFrame(results).T
    print('\n=== Final Results ===')
    print(df_results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results.reset_index(), x='index', y='AUC')
    plt.title('KA-GNN Performance (AUC) on Molecular Datasets')
    plt.savefig('report/images/ka_gnn_auc.png')
    plt.close()
    
    # Save report
    with open('report/report.md', 'w') as f:
        f.write('# KA-GNN for Molecular Property Prediction\n\n')
        f.write('## Methodology\n')
        f.write('Fourier-based Kolmogorov-Arnold Networks replacing MLPs in GNN message passing.\n\n')
        f.write('## Results\n')
        f.write(df_results.to_markdown())
        f.write('\n\n![KA-GNN AUC](images/ka_gnn_auc.png)\n')
    
    print('Report saved to report/report.md')

if __name__ == '__main__':
    os.makedirs('report/images', exist_ok=True)
    main()
