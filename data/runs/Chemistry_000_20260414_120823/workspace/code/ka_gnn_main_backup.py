"""
KA-GNN: Kolmogorov-Arnold Graph Neural Networks for Molecular Property Prediction
==================================================================================
Implements:
- Molecular graph featurization from SMILES (RDKit)
- KAN (Kolmogorov-Arnold Network) layers with learnable univariate activation functions
- KA-GNN: GNN with KAN-based message passing
- Baselines: GCN, GAT
- Evaluation on BACE, BBBP, ClinTox, HIV, MUV
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Molecular Graph Featurization
# ============================================================

# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [0, 1],
    'is_in_ring': [0, 1],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'is_conjugated': [0, 1],
    'is_in_ring': [0, 1],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}

def one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    encoding[idx] = 1
    return encoding

def atom_features(atom):
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(one_hot(atom.GetTotalDegree(), ATOM_FEATURES['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.extend(one_hot(int(atom.GetIsAromatic()), ATOM_FEATURES['is_aromatic']))
    features.extend(one_hot(int(atom.IsInRing()), ATOM_FEATURES['is_in_ring']))
    # Additional scalar features
    features.append(atom.GetMass() / 200.0)  # normalized mass
    features.append(atom.GetExplicitValence() / 8.0)
    features.append(atom.GetImplicitValence() / 8.0)
    return features

def bond_features(bond):
    features = []
    features.extend(one_hot(bond.GetBondType(), BOND_FEATURES['bond_type']))
    features.extend(one_hot(int(bond.GetIsConjugated()), BOND_FEATURES['is_conjugated']))
    features.extend(one_hot(int(bond.IsInRing()), BOND_FEATURES['is_in_ring']))
    features.extend(one_hot(bond.GetStereo(), BOND_FEATURES['stereo']))
    return features

def smiles_to_graph(smiles):
    """Convert SMILES string to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
    x = torch.tensor(node_feats, dtype=torch.float)

    # Edge index and edge features
    edge_index = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Add both directions
        edge_index.extend([[i, j], [j, i]])
        edge_feats.extend([bf, bf])

    if len(edge_index) == 0:
        # No bonds - create self-loops
        edge_index = [[i, i] for i in range(mol.GetNumAtoms())]
        edge_feats = [[0] * len(bond_features_dummy()) for _ in range(mol.GetNumAtoms())]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def bond_features_dummy():
    return [0] * (len(BOND_FEATURES['bond_type']) + 1 +
                   len(BOND_FEATURES['is_conjugated']) + 1 +
                   len(BOND_FEATURES['is_in_ring']) + 1 +
                   len(BOND_FEATURES['stereo']) + 1)


# ============================================================
# 2. KAN Layer Implementation (Fourier-based Kolmogorov-Arnold)
# ============================================================

class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network linear layer.
    Uses learnable univariate functions parameterized by B-spline basis
    combined with Fourier features for enhanced expressiveness.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 base_activation='silu', grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Grid for B-spline
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer('grid', grid)

        # Spline coefficients: for each input-output pair
        num_splines = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_splines) * 0.1
        )

        # Base weight (linear transformation)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # Fourier coefficients for enhanced expressiveness
        self.fourier_weight = nn.Parameter(
            torch.randn(out_features, in_features, 4) * 0.1  # 4 Fourier components
        )

        # Scale and bias
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Base activation
        if base_activation == 'silu':
            self.base_activation = F.silu
        elif base_activation == 'relu':
            self.base_activation = F.relu
        elif base_activation == 'gelu':
            self.base_activation = F.gelu
        else:
            self.base_activation = F.silu

    def b_spline_basis(self, x):
        """Compute B-spline basis functions."""
        # x: (batch, in_features)
        # Expand grid to match
        grid = self.grid  # (grid_size + spline_order + 1,)
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)
        grid_expanded = grid.view(1, 1, -1)  # (1, 1, grid_size + spline_order + 1)

        # Compute basis using Cox-de Boor recursion
        # Start with degree 0 (indicator functions)
        bases = []
        for i in range(len(grid) - 1):
            left = grid[i]
            right = grid[i + 1]
            if i == len(grid) - 2:
                basis = ((x_expanded >= left) & (x_expanded <= right)).float()
            else:
                basis = ((x_expanded >= left) & (x_expanded < right)).float()
            bases.append(basis)

        B = torch.cat(bases, dim=-1)  # (batch, in_features, num_bases)

        # Apply higher-order B-spline recursion
        for k in range(1, self.spline_order + 1):
            B_new = []
            for i in range(len(grid) - k - 1):
                left = grid[i]
                mid = grid[i + k]
                right = grid[i + k + 1]

                denom1 = mid - left
                denom2 = right - grid[i + 1]

                if denom1 > 1e-6:
                    term1 = (x_expanded - left) / denom1 * B[:, :, i].unsqueeze(-1)
                else:
                    term1 = torch.zeros_like(B[:, :, i].unsqueeze(-1))

                if denom2 > 1e-6:
                    term2 = (right - x_expanded) / denom2 * B[:, :, i + 1].unsqueeze(-1)
                else:
                    term2 = torch.zeros_like(B[:, :, i + 1].unsqueeze(-1))

                B_new.append(term1 + term2)
            B = torch.cat(B_new, dim=-1)

        return B  # (batch, in_features, num_splines)

    def fourier_basis(self, x):
        """Compute Fourier basis features."""
        # x: (batch, in_features)
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)
        freqs = torch.arange(1, 5, device=x.device).float()  # 4 frequencies
        freqs = freqs.view(1, 1, -1)  # (1, 1, 4)

        sin_features = torch.sin(2 * np.pi * freqs * x_expanded)  # (batch, in_features, 4)
        cos_features = torch.cos(2 * np.pi * freqs * x_expanded)

        return torch.cat([sin_features, cos_features], dim=-1)  # (batch, in_features, 8)

    def forward(self, x):
        """
        x: (batch, in_features) or (..., in_features)
        """
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        # Base linear transformation
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # B-spline transformation
        B = self.b_spline_basis(x)  # (batch, in_features, num_splines)
        spline_output = torch.einsum('bik,oik->bo', B, self.spline_weight[:, :, :B.shape[-1]])

        # Fourier transformation
        # Use a simplified Fourier component
        fourier_output = torch.zeros(x.shape[0], self.out_features, device=x.device)
        for k in range(min(4, self.fourier_weight.shape[-1])):
            freq = k + 1
            sin_x = torch.sin(freq * np.pi * x)  # (batch, in_features)
            fourier_output += F.linear(sin_x, self.fourier_weight[:, :, k])

        # Combine with gating
        output = base_output + spline_output + 0.1 * fourier_output + self.bias

        new_shape = list(original_shape[:-1]) + [self.out_features]
        return output.view(new_shape)


class KANLayer(nn.Module):
    """Full KAN layer: linear + KAN activation."""
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.kan = KANLinear(in_features, out_features, grid_size=grid_size)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.norm(self.kan(x))


# ============================================================
# 3. GNN Architectures
# ============================================================

class KAGNNConv(nn.Module):
    """
    KAN-based Graph Convolution Layer.
    Replaces MLP in message passing with KAN modules.
    """
    def __init__(self, in_channels, out_channels, edge_dim=None, grid_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # KAN for transforming node features
        self.node_kan = KANLinear(in_channels, out_channels, grid_size=grid_size)

        # KAN for edge features if provided
        if edge_dim is not None:
            self.edge_kan = KANLinear(edge_dim, out_channels, grid_size=grid_size)
        else:
            self.edge_kan = None

        # KAN for combining messages
        self.combine_kan = KANLinear(out_channels, out_channels, grid_size=grid_size)

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index

        # Transform node features
        x_transformed = self.node_kan(x)

        # Message: aggregate neighbor features
        messages = x_transformed[col]  # target node features

        # Add edge features if available
        if edge_attr is not None and self.edge_kan is not None:
            edge_transformed = self.edge_kan(edge_attr)
            messages = messages + edge_transformed

        # Aggregate messages (sum over neighbors)
        from torch_geometric.utils import degree
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0

        # Weighted aggregation
        messages = messages * deg_inv[row].unsqueeze(-1)
        aggregated = torch.zeros_like(x_transformed)
        aggregated.index_add_(0, row, messages)
        aggregated = aggregated * deg_inv.unsqueeze(-1)

        # Combine with self-connection
        out = self.combine_kan(aggregated + x_transformed)
        return self.norm(out)


class KAGNN(nn.Module):
    """
    Kolmogorov-Arnold Graph Neural Network.
    Uses KAN-based message passing layers.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3,
                 num_classes=1, dropout=0.2, grid_size=5, pool='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = KANLinear(node_dim, hidden_dim, grid_size=grid_size)

        # KAN-GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(KAGNNConv(hidden_dim, hidden_dim, edge_dim, grid_size))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Output MLP (also KAN-based)
        self.output_kan1 = KANLinear(hidden_dim, hidden_dim, grid_size=grid_size)
        self.output_kan2 = KANLinear(hidden_dim, num_classes, grid_size=grid_size)

        self.pool = pool

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Input projection
        x = self.input_proj(x)

        # Message passing
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = x + x_res  # residual connection
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pool == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_add_pool(x, batch)

        # Output
        x = F.silu(self.output_kan1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_kan2(x)
        return x


class GCNBaseline(nn.Module):
    """Standard GCN baseline."""
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3,
                 num_classes=1, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_proj(x))

        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = x + x_res
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.output_mlp(x)


class GATBaseline(nn.Module):
    """Standard GAT baseline."""
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3,
                 num_classes=1, dropout=0.2, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_proj(x))

        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = x + x_res
            x = self.norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.output_mlp(x)


# ============================================================
# 4. Data Loading and Preparation
# ============================================================

def load_dataset(csv_path, dataset_name):
    """Load dataset and convert SMILES to graphs."""
    df = pd.read_csv(csv_path)
    print(f"\nLoading {dataset_name}: {len(df)} molecules")

    # Determine label columns based on dataset
    if dataset_name == 'bace':
        smiles_col = 'smiles'
        label_cols = ['label']
    elif dataset_name == 'bbbp':
        smiles_col = 'smiles'
        label_cols = ['label']
    elif dataset_name == 'clintox':
        smiles_col = 'smiles'
        label_cols = ['FDA_APPROVED', 'CT_TOX']
    elif dataset_name == 'hiv':
        smiles_col = 'smiles'
        label_cols = ['label']
    elif dataset_name == 'muv':
        smiles_col = 'smiles'
        label_cols = [c for c in df.columns if c.startswith('MUV')]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    graphs = []
    valid_indices = []

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        graph = smiles_to_graph(smiles)
        if graph is not None:
            # Get labels
            labels = []
            for lc in label_cols:
                val = row[lc]
                if pd.isna(val):
                    labels.append(float('nan'))
                else:
                    labels.append(float(val))
            graph.y = torch.tensor([labels], dtype=torch.float)
            graphs.append(graph)
            valid_indices.append(idx)

    print(f"  Valid graphs: {len(graphs)} / {len(df)}")
    return graphs, label_cols


# ============================================================
# 5. Training and Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        target = batch.y.view_as(out)

        # Handle missing labels
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            continue
        loss = criterion(out[mask], target[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
    return total_loss


@torch.no_grad()
def evaluate(model, loader, device, num_tasks=1):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        target = batch.y.view_as(out)

        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            continue

        all_preds.append(out[mask].cpu())
        all_targets.append(target[mask].cpu())

    if len(all_preds) == 0:
        return 0.0

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute ROC-AUC
    try:
        if num_tasks == 1:
            if len(np.unique(all_targets)) < 2:
                return 0.5
            auc = roc_auc_score(all_targets, all_preds)
        else:
            # Multi-task: average AUC across tasks
            aucs = []
            for t in range(num_tasks):
                task_mask = ~np.isnan(all_targets[:, t]) if all_targets.ndim > 1 else ~np.isnan(all_targets)
                if task_mask.sum() == 0:
                    continue
                yt = all_targets[task_mask, t] if all_targets.ndim > 1 else all_targets[task_mask]
                yp = all_preds[task_mask, t] if all_preds.ndim > 1 else all_preds[task_mask]
                if len(np.unique(yt)) >= 2:
                    aucs.append(roc_auc_score(yt, yp))
            auc = np.mean(aucs) if aucs else 0.5
    except:
        auc = 0.5

    return auc


def run_experiment(dataset_name, csv_path, model_class, model_kwargs,
                   num_epochs=100, lr=1e-3, batch_size=32, n_folds=5, seed=42):
    """Run cross-validation experiment for a model on a dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}, Model: {model_class.__name__}")
    print(f"Device: {device}")

    # Load data
    graphs, label_cols = load_dataset(csv_path, dataset_name)
    num_tasks = len(label_cols)

    # Get feature dimensions
    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1] if graphs[0].edge_attr is not None else None

    # For binary classification tasks, get labels for stratification
    if num_tasks == 1:
        labels = [g.y[0, 0].item() for g in graphs]
    else:
        labels = [g.y[0, 0].item() for g in graphs]  # use first task for stratification

    # Filter out NaN labels for stratification
    valid_mask = ~np.isnan(labels)
    valid_indices = np.where(valid_mask)[0]
    valid_labels = np.array(labels)[valid_mask]

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(valid_indices, valid_labels)):
        print(f"\n  Fold {fold + 1}/{n_folds}")

        # Map back to original indices
        train_graphs = [graphs[valid_indices[i]] for i in train_idx]
        test_graphs = [graphs[valid_indices[i]] for i in test_idx]

        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = model_class(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_classes=num_tasks,
            **model_kwargs
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0
        patience = 15
        patience_counter = 0

        for epoch in range(num_epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                auc = evaluate(model, test_loader, device, num_tasks)
                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        # Final evaluation
        final_auc = evaluate(model, test_loader, device, num_tasks)
        fold_aucs.append(final_auc)
        print(f"    Fold {fold + 1} AUC: {final_auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"\n  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    return {
        'dataset': dataset_name,
        'model': model_class.__name__,
        'fold_aucs': fold_aucs,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'num_tasks': num_tasks,
        'num_samples': len(graphs),
        'num_valid': len(valid_indices),
    }


# ============================================================
# 6. Main Experiment Runner
# ============================================================

def main():
    # Datasets to evaluate
    datasets = {
        'bace': 'data/bace.csv',
        'bbbp': 'data/bbbp.csv',
        'clintox': 'data/clintox.csv',
        'hiv': 'data/hiv.csv',
        'muv': 'data/muv.csv',
    }

    # Models to compare
    models = {
        'KA-GNN': (KAGNN, {'hidden_dim': 64, 'num_layers': 3, 'grid_size': 5}),
        'GCN': (GCNBaseline, {'hidden_dim': 64, 'num_layers': 3}),
        'GAT': (GATBaseline, {'hidden_dim': 64, 'num_layers': 3}),
    }

    all_results = []

    for dataset_name, csv_path in datasets.items():
        for model_name, (model_class, model_kwargs) in models.items():
            try:
                result = run_experiment(
                    dataset_name=dataset_name,
                    csv_path=csv_path,
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    num_epochs=80,
                    lr=5e-4,
                    batch_size=32,
                    n_folds=5,
                    seed=42
                )
                result['model_name'] = model_name
                all_results.append(result)
            except Exception as e:
                print(f"Error: {dataset_name} / {model_name}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n\nAll results saved to outputs/experiment_results.json")
    return all_results


if __name__ == '__main__':
    results = main()
