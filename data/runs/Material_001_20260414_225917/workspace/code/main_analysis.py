"""
AI-Accelerated Materials Discovery: Complete Analysis Pipeline
Implements three core workflows:
1. Property Prediction (CGCNN-inspired graph neural network)
2. Structure Generation (Variational Autoencoder)
3. Autonomous Optimization (Bayesian Optimization)
"""

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import norm
from scipy.optimize import minimize

# ============================================================
# SECTION 0: DATA PARSING
# ============================================================

def parse_dataset(filepath):
    """Parse the three data sections from the dataset file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    sections = content.strip().split('# 文件')
    data = {}
    
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split('\n')
        # Extract section name
        header = lines[0].strip().rstrip('数据').strip()
        arrays = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                arr = [float(x.strip()) for x in line[1:-1].split(',')]
                arrays.append(np.array(arr))
        data[header] = arrays
    
    return data

# Parse
data = parse_dataset('data/M-AI-Synth__Materials_AI_Dataset_.txt')

# Property prediction data
node_features = data['1: property_prediction.py'][0]   # 100 node feature values
edge_features = data['1: property_prediction.py'][1]   # 100 edge feature values
edge_index = data['1: property_prediction.py'][2]       # 20 edge indices
targets = data['1: property_prediction.py'][3]           # 96 target values

# Structure generation data
lattice_a = data['2: structure_generation.py'][0]       # 100 lattice a values
lattice_b = data['2: structure_generation.py'][1]       # 100 lattice b values

# Optimization data
temp_range = data['3: autonomous_optimization.py'][0]   # [200, 500]
time_range = data['3: autonomous_optimization.py'][1]   # [10, 30]
opt_temp = data['3: autonomous_optimization.py'][2]     # [350]
opt_time = data['3: autonomous_optimization.py'][3]     # [20]
opt_learning_rate = data['3: autonomous_optimization.py'][4]  # [0.1]
opt_exploration = data['3: autonomous_optimization.py'][5]    # [10.0]

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"Node features: {len(node_features)} values, range [{node_features.min():.2f}, {node_features.max():.2f}]")
print(f"Edge features: {len(edge_features)} values, range [{edge_features.min():.2f}, {edge_features.max():.2f}]")
print(f"Edge index: {len(edge_index)} values (edges)")
print(f"Target properties: {len(targets)} values, range [{targets.min():.4f}, {targets.max():.4f}]")
print(f"Lattice a: {len(lattice_a)} values, range [{lattice_a.min():.4f}, {lattice_a.max():.4f}]")
print(f"Lattice b: {len(lattice_b)} values, range [{lattice_b.min():.4f}, {lattice_b.max():.4f}]")
print(f"Temperature range: {temp_range}")
print(f"Time range: {time_range}")
print(f"Optimal temperature: {opt_temp}")
print(f"Optimal time: {opt_time}")

# ============================================================
# FIGURE 1: DATA OVERVIEW
# ============================================================

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Materials AI Dataset Overview', fontsize=16, fontweight='bold')

# Node features
axes[0, 0].hist(node_features, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 0].set_title('Node Feature Distribution')
axes[0, 0].set_xlabel('Feature Value')
axes[0, 0].set_ylabel('Count')

# Edge features
axes[0, 1].hist(edge_features, bins=20, color='coral', edgecolor='black', alpha=0.8)
axes[0, 1].set_title('Edge Feature Distribution')
axes[0, 1].set_xlabel('Feature Value')
axes[0, 1].set_ylabel('Count')

# Target properties
axes[0, 2].hist(targets, bins=20, color='mediumpurple', edgecolor='black', alpha=0.8)
axes[0, 2].set_title('Target Property Distribution')
axes[0, 2].set_xlabel('Property Value')
axes[0, 2].set_ylabel('Count')

# Lattice parameters
axes[1, 0].hist(lattice_a, bins=15, color='seagreen', edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Lattice Parameter a Distribution')
axes[1, 0].set_xlabel('Lattice a (Å)')
axes[1, 0].set_ylabel('Count')

axes[1, 1].hist(lattice_b, bins=15, color='darkorange', edgecolor='black', alpha=0.8)
axes[1, 1].set_title('Lattice Parameter b Distribution')
axes[1, 1].set_xlabel('Lattice b (Å)')
axes[1, 1].set_ylabel('Count')

# Crystal graph structure
edge_pairs = [(int(edge_index[i]), int(edge_index[i+1])) for i in range(0, len(edge_index), 2)]
unique_nodes = set()
for a, b in edge_pairs:
    unique_nodes.add(a)
    unique_nodes.add(b)
axes[1, 2].bar(['Nodes', 'Edges'], [len(unique_nodes), len(edge_pairs)], color=['steelblue', 'coral'])
axes[1, 2].set_title('Crystal Graph Statistics')
axes[1, 2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[SAVED] report/images/fig1_data_overview.png")

# ============================================================
# WORKFLOW 1: PROPERTY PREDICTION (CGCNN-Inspired)
# ============================================================

print("\n" + "=" * 60)
print("WORKFLOW 1: PROPERTY PREDICTION (CGCNN-Inspired)")
print("=" * 60)

# Build crystal graph dataset
# Each "crystal" is defined by a set of node features, edge features, and edge connectivity
# We create synthetic crystal graphs from the data

def build_crystal_graphs(node_feats, edge_feats, edge_idx, targets, num_crystals=20):
    """Build crystal graph dataset from raw arrays."""
    graphs = []
    n_edges = len(edge_idx) // 2
    n_nodes = len(node_feats)
    
    # Create multiple crystal graphs by sampling different subgraphs
    np.random.seed(42)
    for i in range(num_crystals):
        # Each crystal gets a subset of nodes and edges
        nodes_per_crystal = 5
        start_node = (i * nodes_per_crystal) % (n_nodes - nodes_per_crystal)
        node_slice = node_feats[start_node:start_node + nodes_per_crystal]
        
        # Edge features for this crystal
        edge_start = (i * 2) % (n_edges - 1)
        edge_slice = edge_feats[edge_start * 10:(edge_start + 1) * 10]
        
        # Target for this crystal
        target_idx = i % len(targets)
        
        graphs.append({
            'node_features': node_slice,
            'edge_features': edge_slice,
            'target': targets[target_idx]
        })
    
    return graphs

crystal_graphs = build_crystal_graphs(node_features, edge_features, edge_index, targets, num_crystals=96)

# Create feature vectors for each crystal (concatenate node + edge stats)
def extract_crystal_features(graph):
    """Extract fixed-length feature vector from crystal graph."""
    node_f = graph['node_features']
    edge_f = graph['edge_features']
    features = np.concatenate([
        node_f,
        [np.mean(node_f), np.std(node_f), np.min(node_f), np.max(node_f)],
        edge_f[:5],  # first 5 edge features
        [np.mean(edge_f), np.std(edge_f)]
    ])
    return features

X_crystal = np.array([extract_crystal_features(g) for g in crystal_graphs])
y_crystal = np.array([g['target'] for g in crystal_graphs])

print(f"Crystal feature matrix shape: {X_crystal.shape}")
print(f"Target vector shape: {y_crystal.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_crystal, y_crystal, test_size=0.2, random_state=42)

# Normalize
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

# Build CGCNN-inspired neural network
class CGCNNPredictor(nn.Module):
    """Crystal Graph Convolutional Neural Network for property prediction."""
    def __init__(self, input_dim, hidden_dim=64, n_conv_layers=3):
        super().__init__()
        # Convolutional layers (graph message passing simulation)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_conv_layers - 1):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_conv_layers)])
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Graph convolution layers with residual connections
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x_new = F.relu(bn(conv(x)))
            if x.shape == x_new.shape:
                x = x + x_new  # residual
            else:
                x = x_new
            x = self.dropout(x)
        
        # Pooling (mean over features) and prediction
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# Train
model = CGCNNPredictor(input_dim=X_train_norm.shape[1], hidden_dim=64, n_conv_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

X_train_t = torch.FloatTensor(X_train_norm)
y_train_t = torch.FloatTensor(y_train_norm)
X_test_t = torch.FloatTensor(X_test_norm)
y_test_t = torch.FloatTensor(y_test_norm)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

n_epochs = 500
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t)
        val_loss = F.mse_loss(val_pred, y_test_t).item()
        val_losses.append(val_loss)
    
    scheduler.step(val_loss)

# Final predictions
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_t).numpy() * y_std + y_mean
    y_pred_test = model(X_test_t).numpy() * y_std + y_mean

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nTraining MAE: {mae_train:.4f}, R²: {r2_train:.4f}")
print(f"Test MAE: {mae_test:.4f}, R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}")

# Save metrics
metrics = {
    'property_prediction': {
        'train_mae': float(mae_train),
        'test_mae': float(mae_test),
        'train_r2': float(r2_train),
        'test_r2': float(r2_test),
        'test_rmse': float(rmse_test),
        'n_train': len(y_train),
        'n_test': len(y_test)
    }
}

# FIGURE 2: Property Prediction Results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Workflow 1: Property Prediction (CGCNN-Inspired)', fontsize=14, fontweight='bold')

# Training curves
axes[0].plot(train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
axes[0].plot(val_losses, label='Validation Loss', color='coral', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training Convergence')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Parity plot
axes[1].scatter(y_test, y_pred_test, alpha=0.7, color='steelblue', edgecolors='navy', s=60)
lims = [min(y_test.min(), y_pred_test.min()) - 0.2, max(y_test.max(), y_pred_test.max()) + 0.2]
axes[1].plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
axes[1].set_xlabel('DFT Calculated Property')
axes[1].set_ylabel('CGCNN Predicted Property')
axes[1].set_title(f'Parity Plot (MAE={mae_test:.4f}, R²={r2_test:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Error distribution
errors = y_test - y_pred_test
axes[2].hist(errors, bins=12, color='mediumpurple', edgecolor='black', alpha=0.8)
axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('Prediction Error')
axes[2].set_ylabel('Count')
axes[2].set_title(f'Error Distribution (RMSE={rmse_test:.4f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig2_property_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] report/images/fig2_property_prediction.png")

# ============================================================
# WORKFLOW 2: STRUCTURE GENERATION (VAE)
# ============================================================

print("\n" + "=" * 60)
print("WORKFLOW 2: STRUCTURE GENERATION (VAE)")
print("=" * 60)

# Prepare lattice parameter data
lattice_data = np.column_stack([lattice_a, lattice_b])
print(f"Lattice data shape: {lattice_data.shape}")
print(f"Lattice a: mean={lattice_a.mean():.4f}, std={lattice_a.std():.4f}")
print(f"Lattice b: mean={lattice_b.mean():.4f}, std={lattice_b.std():.4f}")

# Normalize
lat_mean = lattice_data.mean(axis=0)
lat_std = lattice_data.std(axis=0) + 1e-8
lat_norm = (lattice_data - lat_mean) / lat_std

lat_tensor = torch.FloatTensor(lat_norm)

class LatticeVAE(nn.Module):
    """Variational Autoencoder for lattice parameter generation."""
    def __init__(self, input_dim=2, hidden_dim=32, latent_dim=2):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.5 * kl_loss

# Train VAE
vae = LatticeVAE(input_dim=2, hidden_dim=32, latent_dim=2)
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.005)

vae_losses = []
n_vae_epochs = 800

for epoch in range(n_vae_epochs):
    vae.train()
    recon, mu, logvar = vae(lat_tensor)
    loss = vae_loss(recon, lat_tensor, mu, logvar)
    
    vae_optimizer.zero_grad()
    loss.backward()
    vae_optimizer.step()
    
    vae_losses.append(loss.item())

# Generate new structures
vae.eval()
with torch.no_grad():
    # Sample from latent space
    n_generated = 100
    z_sample = torch.randn(n_generated, 2)
    generated_norm = vae.decode(z_sample).numpy()
    generated_lattice = generated_norm * lat_std + lat_mean
    
    # Also get reconstructions
    recon_norm, mu_all, _ = vae(lat_tensor)
    recon_lattice = recon_norm.numpy() * lat_std + lat_mean

recon_mae_a = mean_absolute_error(lattice_a, recon_lattice[:, 0])
recon_mae_b = mean_absolute_error(lattice_b, recon_lattice[:, 1])

print(f"\nReconstruction MAE - Lattice a: {recon_mae_a:.4f} Å")
print(f"Reconstruction MAE - Lattice b: {recon_mae_b:.4f} Å")
print(f"Generated structures: {n_generated}")
print(f"Generated a: mean={generated_lattice[:, 0].mean():.4f}, std={generated_lattice[:, 0].std():.4f}")
print(f"Generated b: mean={generated_lattice[:, 1].mean():.4f}, std={generated_lattice[:, 1].std():.4f}")

metrics['structure_generation'] = {
    'recon_mae_a': float(recon_mae_a),
    'recon_mae_b': float(recon_mae_b),
    'n_generated': n_generated,
    'generated_a_mean': float(generated_lattice[:, 0].mean()),
    'generated_a_std': float(generated_lattice[:, 0].std()),
    'generated_b_mean': float(generated_lattice[:, 1].mean()),
    'generated_b_std': float(generated_lattice[:, 1].std())
}

# FIGURE 3: Structure Generation Results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Workflow 2: Structure Generation (Variational Autoencoder)', fontsize=14, fontweight='bold')

# VAE training loss
axes[0].plot(vae_losses, color='seagreen', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('ELBO Loss')
axes[0].set_title('VAE Training Convergence')
axes[0].grid(True, alpha=0.3)

# Original vs Generated lattice parameters
axes[1].scatter(lattice_a, lattice_b, alpha=0.6, color='steelblue', label='Original', s=50, edgecolors='navy')
axes[1].scatter(generated_lattice[:, 0], generated_lattice[:, 1], alpha=0.5, color='coral', label='Generated', s=50, edgecolors='darkred', marker='^')
axes[1].set_xlabel('Lattice Parameter a (Å)')
axes[1].set_ylabel('Lattice Parameter b (Å)')
axes[1].set_title('Original vs Generated Lattice Parameters')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Distribution comparison
x_range = np.linspace(min(lattice_a.min(), generated_lattice[:, 0].min()) - 0.1,
                       max(lattice_a.max(), generated_lattice[:, 0].max()) + 0.1, 100)
from scipy.stats import gaussian_kde
kde_orig_a = gaussian_kde(lattice_a)
kde_gen_a = gaussian_kde(generated_lattice[:, 0])
axes[2].plot(x_range, kde_orig_a(x_range), color='steelblue', linewidth=2, label='Original a')
axes[2].plot(x_range, kde_gen_a(x_range), color='coral', linewidth=2, label='Generated a')
axes[2].fill_between(x_range, kde_orig_a(x_range), alpha=0.2, color='steelblue')
axes[2].fill_between(x_range, kde_gen_a(x_range), alpha=0.2, color='coral')
axes[2].set_xlabel('Lattice Parameter a (Å)')
axes[2].set_ylabel('Density')
axes[2].set_title('Distribution Comparison (Lattice a)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig3_structure_generation.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] report/images/fig3_structure_generation.png")

# ============================================================
# WORKFLOW 3: AUTONOMOUS OPTIMIZATION (Bayesian Optimization)
# ============================================================

print("\n" + "=" * 60)
print("WORKFLOW 3: AUTONOMOUS OPTIMIZATION (Bayesian Optimization)")
print("=" * 60)

# Define the objective function (simulated material property)
# Based on the optimization data: optimal at T=350, t=20
def objective(T, t):
    """Simulated material yield/property as function of temperature and time."""
    T_opt, t_opt = opt_temp[0], opt_time[0]
    # Gaussian-like response surface
    val = np.exp(-((T - T_opt)**2 / (2 * 50**2) + (t - t_opt)**2 / (2 * 5**2)))
    # Add some noise and secondary peaks
    val += 0.3 * np.exp(-((T - 280)**2 / (2 * 30**2) + (t - 15)**2 / (2 * 3**2)))
    val += 0.15 * np.sin(T / 50) * np.cos(t / 5)
    return val

# Gaussian Process for Bayesian Optimization
class SimpleGP:
    """Simple Gaussian Process with RBF kernel for Bayesian Optimization."""
    def __init__(self, length_scale=1.0, noise=0.01):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def rbf_kernel(self, X1, X2):
        """RBF kernel."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * sqdist / self.length_scale**2)
    
    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        K = self.rbf_kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
    
    def predict(self, X_new):
        K_s = self.rbf_kernel(X_new, self.X_train)
        K_ss = self.rbf_kernel(X_new, X_new) + 1e-6 * np.eye(len(X_new))
        
        mu = K_s @ self.K_inv @ self.y_train
        cov = K_ss - K_s @ self.K_inv @ K_s.T
        std = np.sqrt(np.maximum(np.diag(cov), 0))
        return mu, std

def expected_improvement(mu, std, best_y, xi=0.01):
    """Expected Improvement acquisition function."""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = mu - best_y - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-8] = 0.0
    return ei

# Run Bayesian Optimization
np.random.seed(42)
T_bounds = (temp_range[0], temp_range[1])
t_bounds = (time_range[0], time_range[1])

# Initial random samples
n_init = 5
T_init = np.random.uniform(T_bounds[0], T_bounds[1], n_init)
t_init = np.random.uniform(t_bounds[0], t_bounds[1], n_init)
y_init = np.array([objective(T, t) for T, t in zip(T_init, t_init)])

T_history = list(T_init)
t_history = list(t_init)
y_history = list(y_init)

n_iterations = 25
gp = SimpleGP(length_scale=30.0, noise=0.01)

for i in range(n_iterations):
    X_train_bo = np.column_stack([T_history, t_history])
    y_train_bo = np.array(y_history)
    gp.fit(X_train_bo, y_train_bo)
    
    # Grid search for acquisition
    T_grid = np.linspace(T_bounds[0], T_bounds[1], 50)
    t_grid = np.linspace(t_bounds[0], t_bounds[1], 50)
    T_mesh, t_mesh = np.meshgrid(T_grid, t_grid)
    X_grid = np.column_stack([T_mesh.ravel(), t_mesh.ravel()])
    
    mu_pred, std_pred = gp.predict(X_grid)
    best_y = max(y_history)
    ei = expected_improvement(mu_pred, std_pred, best_y)
    
    # Select next point
    next_idx = np.argmax(ei)
    T_next = X_grid[next_idx, 0]
    t_next = X_grid[next_idx, 1]
    y_next = objective(T_next, t_next)
    
    T_history.append(T_next)
    t_history.append(t_next)
    y_history.append(y_next)

# Results
best_idx = np.argmax(y_history)
best_T = T_history[best_idx]
best_t = t_history[best_idx]
best_y_val = y_history[best_idx]

print(f"\nOptimal found: T={best_T:.1f}°C, t={best_t:.1f}h, yield={best_y_val:.4f}")
print(f"True optimum: T={opt_temp[0]:.1f}°C, t={opt_time[0]:.1f}h")
print(f"Temperature error: {abs(best_T - opt_temp[0]):.1f}°C")
print(f"Time error: {abs(best_t - opt_time[0]):.1f}h")

metrics['autonomous_optimization'] = {
    'best_T': float(best_T),
    'best_t': float(best_t),
    'best_yield': float(best_y_val),
    'true_opt_T': float(opt_temp[0]),
    'true_opt_t': float(opt_time[0]),
    'T_error': float(abs(best_T - opt_temp[0])),
    't_error': float(abs(best_t - opt_time[0])),
    'n_iterations': n_iterations,
    'n_init': n_init
}

# FIGURE 4: Bayesian Optimization Results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Workflow 3: Autonomous Optimization (Bayesian Optimization)', fontsize=14, fontweight='bold')

# Objective surface with sampled points
T_grid = np.linspace(T_bounds[0], T_bounds[1], 100)
t_grid = np.linspace(t_bounds[0], t_bounds[1], 100)
T_mesh, t_mesh = np.meshgrid(T_grid, t_grid)
Z = np.array([[objective(T, t) for T in T_grid] for t in t_grid])

contour = axes[0].contourf(T_mesh, t_mesh, Z, levels=20, cmap='viridis', alpha=0.8)
axes[0].scatter(T_history[:n_init], t_history[:n_init], c='white', s=50, edgecolors='black', label='Initial', zorder=5)
axes[0].scatter(T_history[n_init:], t_history[n_init:], c='red', s=30, marker='x', label='BO samples', zorder=5)
axes[0].scatter(best_T, best_t, c='yellow', s=200, marker='*', edgecolors='black', label='Best found', zorder=6)
axes[0].scatter(opt_temp[0], opt_time[0], c='lime', s=200, marker='D', edgecolors='black', label='True optimum', zorder=6)
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Time (h)')
axes[0].set_title('Objective Surface & Sampling History')
axes[0].legend(fontsize=8, loc='upper left')
plt.colorbar(contour, ax=axes[0], label='Yield')

# Convergence plot
best_so_far = [max(y_history[:i+1]) for i in range(len(y_history))]
axes[1].plot(range(len(best_so_far)), best_so_far, 'b-o', markersize=4, linewidth=1.5)
axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='True optimum')
axes[1].axvline(x=n_init, color='gray', linestyle=':', label='BO starts')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Best Yield Found')
axes[1].set_title('Optimization Convergence')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Acquisition function (final iteration)
T_grid_ei = np.linspace(T_bounds[0], T_bounds[1], 50)
t_grid_ei = np.linspace(t_bounds[0], t_bounds[1], 50)
T_mesh_ei, t_mesh_ei = np.meshgrid(T_grid_ei, t_grid_ei)
X_grid_ei = np.column_stack([T_mesh_ei.ravel(), t_mesh_ei.ravel()])
mu_pred, std_pred = gp.predict(X_grid_ei)
ei_final = expected_improvement(mu_pred, std_pred, max(y_history))
ei_mesh = ei_final.reshape(T_mesh_ei.shape)
contour2 = axes[2].contourf(T_mesh_ei, t_mesh_ei, ei_mesh, levels=20, cmap='hot', alpha=0.8)
axes[2].scatter(T_history, t_history, c='cyan', s=30, edgecolors='black', zorder=5)
axes[2].set_xlabel('Temperature (°C)')
axes[2].set_ylabel('Time (h)')
axes[2].set_title('Final Acquisition Function (EI)')
plt.colorbar(contour2, ax=axes[2], label='Expected Improvement')

plt.tight_layout()
plt.savefig('report/images/fig4_bayesian_optimization.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] report/images/fig4_bayesian_optimization.png")

# ============================================================
# FIGURE 5: COMBINED SUMMARY
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('AI-Accelerated Materials Discovery: Summary of Three Workflows', fontsize=14, fontweight='bold')

# Workflow 1 summary
metrics_names = ['MAE\n(eV/atom)', 'R²', 'RMSE\n(eV/atom)']
metrics_vals = [mae_test, r2_test, rmse_test]
colors = ['steelblue', 'seagreen', 'coral']
bars = axes[0].bar(metrics_names, metrics_vals, color=colors, edgecolor='black', alpha=0.8)
axes[0].set_title('Property Prediction')
axes[0].set_ylabel('Value')
for bar, val in zip(bars, metrics_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontweight='bold')

# Workflow 2 summary
labels = ['Lattice a\n(Original)', 'Lattice a\n(Generated)', 'Lattice b\n(Original)', 'Lattice b\n(Generated)']
vals = [lattice_a.mean(), generated_lattice[:, 0].mean(), lattice_b.mean(), generated_lattice[:, 1].mean()]
errs = [lattice_a.std(), generated_lattice[:, 0].std(), lattice_b.std(), generated_lattice[:, 1].std()]
colors2 = ['steelblue', 'coral', 'steelblue', 'coral']
axes[1].bar(labels, vals, yerr=errs, color=colors2, edgecolor='black', alpha=0.8, capsize=5)
axes[1].set_title('Structure Generation')
axes[1].set_ylabel('Lattice Parameter (Å)')

# Workflow 3 summary
iters = list(range(len(y_history)))
axes[2].plot(iters[:n_init], y_history[:n_init], 'go-', markersize=6, label='Random init')
axes[2].plot(iters[n_init:], y_history[n_init:], 'r^-', markersize=6, label='BO iterations')
axes[2].axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Optimum')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Objective Value')
axes[2].set_title('Optimization Progress')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig5_combined_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] report/images/fig5_combined_summary.png")

# Save metrics
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\n[SAVED] outputs/metrics.json")

print("\n" + "=" * 60)
print("ALL ANALYSES COMPLETE")
print("=" * 60)
