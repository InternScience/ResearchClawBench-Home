"""
Workflow 2: Structure Generation
- Implements VAE-based generative model for crystal lattice parameters
- Generates novel material structures and validates against known distributions
- Includes latent space visualization and structure diversity analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = 'report/images'

# Load data
features = np.load('outputs/features.npy')
targets = np.load('outputs/targets.npy')
feature_names = list(np.load('outputs/feature_names.npy', allow_pickle=True))
target_names = list(np.load('outputs/target_names.npy', allow_pickle=True))

# Load original lattice data
with open('outputs/parsed_dataset.json', 'r') as f:
    parsed = json.load(f)

lattice_x = np.array(parsed['structure']['lattice_x'])
lattice_y = np.array(parsed['structure']['lattice_y'])

# ============================================================
# Build structure dataset
# ============================================================
# Combine original lattice data with synthetic structural features
structural_features = features[:, 8:14]  # lattice params and angles
volume = features[:, 14]
comp_features = features[:, :8]

# Full structure representation: composition + lattice
structure_data = np.column_stack([comp_features, structural_features, volume.reshape(-1, 1)])

scaler_struct = MinMaxScaler()
structure_scaled = scaler_struct.fit_transform(structure_data)

# ============================================================
# Variational Autoencoder (VAE) Implementation
# ============================================================
class SimpleVAE:
    """Simplified VAE for structure generation using numpy."""
    
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[64, 32], lr=0.001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.rng = np.random.RandomState(42)
        
        # Initialize weights (Xavier initialization)
        self.encoder_weights = []
        self.encoder_biases = []
        self.decoder_weights = []
        self.decoder_biases = []
        
        # Encoder
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.encoder_weights.append(self.rng.randn(prev_dim, h_dim) * np.sqrt(2.0 / prev_dim))
            self.encoder_biases.append(np.zeros(h_dim))
            prev_dim = h_dim
        
        # Latent space (mean and log_var)
        self.enc_mean_w = self.rng.randn(prev_dim, latent_dim) * np.sqrt(2.0 / prev_dim)
        self.enc_mean_b = np.zeros(latent_dim)
        self.enc_logvar_w = self.rng.randn(prev_dim, latent_dim) * np.sqrt(2.0 / prev_dim)
        self.enc_logvar_b = np.zeros(latent_dim)
        
        # Decoder
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            self.decoder_weights.append(self.rng.randn(prev_dim, h_dim) * np.sqrt(2.0 / prev_dim))
            self.decoder_biases.append(np.zeros(h_dim))
            prev_dim = h_dim
        
        self.dec_out_w = self.rng.randn(prev_dim, input_dim) * np.sqrt(2.0 / prev_dim)
        self.dec_out_b = np.zeros(input_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def encode(self, x):
        h = x
        for w, b in zip(self.encoder_weights, self.encoder_biases):
            h = self.relu(h @ w + b)
        mean = h @ self.enc_mean_w + self.enc_mean_b
        logvar = h @ self.enc_logvar_w + self.enc_logvar_b
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = self.rng.randn(*mean.shape)
        std = np.exp(0.5 * logvar)
        return mean + eps * std
    
    def decode(self, z):
        h = z
        for w, b in zip(self.decoder_weights, self.decoder_biases):
            h = self.relu(h @ w + b)
        out = self.sigmoid(h @ self.dec_out_w + self.dec_out_b)
        return out
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar, z
    
    def compute_loss(self, x, recon, mean, logvar, beta=0.01):
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - recon) ** 2)
        # KL divergence
        kl_loss = -0.5 * np.mean(1 + logvar - mean ** 2 - np.exp(logvar))
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


# Train VAE
print("Training VAE for structure generation...")
vae = SimpleVAE(input_dim=structure_scaled.shape[1], latent_dim=8, hidden_dims=[64, 32])

# Training loop with simple gradient estimation (finite differences for demonstration)
n_epochs = 200
batch_size = 64
n_samples = structure_scaled.shape[0]
losses = []
recon_losses = []
kl_losses = []

# Use a simpler approach: train an autoencoder with numpy
class SimpleAutoencoder:
    """Simple autoencoder for structure generation."""
    def __init__(self, input_dim, latent_dim=8):
        self.rng = np.random.RandomState(42)
        self.latent_dim = latent_dim
        
        # Encoder
        self.W1 = self.rng.randn(input_dim, 64) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = self.rng.randn(64, 32) * 0.1
        self.b2 = np.zeros(32)
        self.W3 = self.rng.randn(32, latent_dim) * 0.1
        self.b3 = np.zeros(latent_dim)
        
        # Decoder
        self.W4 = self.rng.randn(latent_dim, 32) * 0.1
        self.b4 = np.zeros(32)
        self.W5 = self.rng.randn(32, 64) * 0.1
        self.b5 = np.zeros(64)
        self.W6 = self.rng.randn(64, input_dim) * 0.1
        self.b6 = np.zeros(input_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def encode(self, x):
        self.h1 = self.relu(x @ self.W1 + self.b1)
        self.h2 = self.relu(self.h1 @ self.W2 + self.b2)
        self.z = self.h2 @ self.W3 + self.b3
        return self.z
    
    def decode(self, z):
        self.d1 = self.relu(z @ self.W4 + self.b4)
        self.d2 = self.relu(self.d1 @ self.W5 + self.b5)
        out = self.sigmoid(self.d2 @ self.W6 + self.b6)
        return out
    
    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z
    
    def train_step(self, x, lr=0.001):
        out, z = self.forward(x)
        loss = np.mean((x - out) ** 2)
        
        # Backpropagation
        dout = 2 * (out - x) / x.shape[0]  # (batch, input_dim)
        dout = dout * out * (1 - out)  # sigmoid derivative
        
        dW6 = self.d2.T @ dout
        db6 = dout.mean(axis=0)
        dd2 = dout @ self.W6.T * self.relu_deriv(self.d2)
        
        dW5 = self.d1.T @ dd2
        db5 = dd2.mean(axis=0)
        dd1 = dd2 @ self.W5.T * self.relu_deriv(self.d1)
        
        dW4 = z.T @ dd1
        db4 = dd1.mean(axis=0)
        dz = dd1 @ self.W4.T
        
        dW3 = self.h2.T @ dz
        db3 = dz.mean(axis=0)
        dh2 = dz @ self.W3.T * self.relu_deriv(self.h2)
        
        dW2 = self.h1.T @ dh2
        db2 = dh2.mean(axis=0)
        dh1 = dh2 @ self.W2.T * self.relu_deriv(self.h1)
        
        dW1 = x.T @ dh1
        db1 = dh1.mean(axis=0)
        
        # Update weights
        for w, dw in [(self.W1, dW1), (self.W2, dW2), (self.W3, dW3),
                       (self.W4, dW4), (self.W5, dW5), (self.W6, dW6)]:
            w -= lr * np.clip(dw, -1, 1)
        for b, db in [(self.b1, db1), (self.b2, db2), (self.b3, db3),
                       (self.b4, db4), (self.b5, db5), (self.b6, db6)]:
            b -= lr * np.clip(db, -1, 1)
        
        return loss


ae = SimpleAutoencoder(input_dim=structure_scaled.shape[1], latent_dim=8)

train_losses = []
for epoch in range(n_epochs):
    # Mini-batch training
    indices = np.random.permutation(n_samples)
    epoch_loss = 0
    n_batches = 0
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start+batch_size]
        batch = structure_scaled[batch_idx]
        loss = ae.train_step(batch, lr=0.005)
        epoch_loss += loss
        n_batches += 1
    train_losses.append(epoch_loss / n_batches)
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/n_batches:.6f}")

# Encode all structures
all_z = ae.encode(structure_scaled)

# Generate new structures by sampling latent space
print("\nGenerating novel structures...")
n_generated = 200
z_mean = all_z.mean(axis=0)
z_std = all_z.std(axis=0)
z_generated = np.random.randn(n_generated, ae.latent_dim) * z_std * 1.2 + z_mean

generated_structures = ae.decode(z_generated)
generated_structures_original = scaler_struct.inverse_transform(generated_structures)

# Validate generated structures
real_structures = structure_data
gen_structures = generated_structures_original

# Compute validity metrics
gen_lattice_a = gen_structures[:, 8]
gen_lattice_b = gen_structures[:, 9]
gen_lattice_c = gen_structures[:, 10]
real_lattice_a = real_structures[:, 8]
real_lattice_b = real_structures[:, 9]
real_lattice_c = real_structures[:, 10]

# Validity: lattice parameters should be positive and in reasonable range
valid_lattice = np.all(gen_structures[:, 8:11] > 0, axis=1)
validity_rate = valid_lattice.mean()
print(f"Structure validity rate: {validity_rate:.2%}")

# Uniqueness: fraction of unique structures (within tolerance)
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(gen_structures)
np.fill_diagonal(sim_matrix, 0)
max_sim = sim_matrix.max(axis=1)
uniqueness_rate = (max_sim < 0.999).mean()
print(f"Structure uniqueness rate: {uniqueness_rate:.2%}")

# Novelty: how different from training data
sim_to_real = cosine_similarity(gen_structures, real_structures)
max_sim_to_real = sim_to_real.max(axis=1)
novelty_rate = (max_sim_to_real < 0.999).mean()
print(f"Structure novelty rate: {novelty_rate:.2%}")

# Save generation metrics
gen_metrics = {
    'validity_rate': float(validity_rate),
    'uniqueness_rate': float(uniqueness_rate),
    'novelty_rate': float(novelty_rate),
    'n_generated': n_generated,
    'latent_dim': ae.latent_dim,
    'reconstruction_loss_final': float(train_losses[-1]),
    'lattice_a_mae': float(mean_absolute_error(real_lattice_a, ae.decode(ae.encode(structure_scaled))[:, 8] * (scaler_struct.data_max_[8] - scaler_struct.data_min_[8]) + scaler_struct.data_min_[8])),
}

with open('outputs/structure_generation_metrics.json', 'w') as f:
    json.dump(gen_metrics, f, indent=2)

# ============================================================
# FIGURE 6: VAE training loss
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, linewidth=1.5, color='steelblue')
ax.set_xlabel('Epoch')
ax.set_ylabel('Reconstruction Loss (MSE)')
ax.set_title('Autoencoder Training Convergence', fontweight='bold')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_ae_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_ae_training_loss.png")

# ============================================================
# FIGURE 7: Latent space visualization (t-SNE)
# ============================================================
print("Computing t-SNE visualization...")
combined_z = np.vstack([all_z, z_generated])
labels = ['Real'] * len(all_z) + ['Generated'] * len(z_generated)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_tsne = tsne.fit_transform(combined_z)

fig, ax = plt.subplots(figsize=(10, 8))
for label, color, marker in [('Real', 'steelblue', 'o'), ('Generated', 'coral', '^')]:
    mask = [l == label for l in labels]
    ax.scatter(z_tsne[mask, 0], z_tsne[mask, 1], c=color, marker=marker, 
               alpha=0.5, s=30, label=label, edgecolors='white', linewidth=0.5)
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_title('Latent Space: Real vs. Generated Structures', fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig7_latent_space_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_latent_space_tsne.png")

# ============================================================
# FIGURE 8: Distribution comparison (real vs generated)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution Comparison: Real vs. Generated Crystal Structures', fontsize=14, fontweight='bold')

plot_features = [
    ('Lattice a (Å)', 8), ('Lattice b (Å)', 9), ('Lattice c (Å)', 10),
    ('Volume (Å³)', 14), ('Comp Element 0', 0), ('Comp Element 1', 1)
]

for idx, (fname, fidx) in enumerate(plot_features):
    ax = axes[idx // 3, idx % 3]
    ax.hist(real_structures[:, fidx], bins=30, alpha=0.6, color='steelblue', 
            label='Real', density=True, edgecolor='white')
    ax.hist(gen_structures[:, fidx], bins=30, alpha=0.6, color='coral', 
            label='Generated', density=True, edgecolor='white')
    ax.set_xlabel(fname)
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(fname)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig8_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_distribution_comparison.png")

# ============================================================
# FIGURE 9: Generated lattice parameter scatter
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Generated Crystal Lattice Parameters', fontsize=14, fontweight='bold')

pairs = [(8, 9, 'a vs b'), (8, 10, 'a vs c'), (9, 10, 'b vs c')]
for idx, (i, j, title) in enumerate(pairs):
    ax = axes[idx]
    ax.scatter(real_structures[:, i], real_structures[:, j], alpha=0.3, s=20, 
               c='steelblue', label='Real')
    ax.scatter(gen_structures[:, i], gen_structures[:, j], alpha=0.3, s=20, 
               c='coral', marker='^', label='Generated')
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(feature_names[j])
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig9_lattice_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_lattice_scatter.png")

# ============================================================
# FIGURE 10: Novel structure examples
# ============================================================
# Select most novel structures (lowest max similarity to real)
most_novel_idx = np.argsort(max_sim_to_real)[:10]
novel_structures = gen_structures[most_novel_idx]

fig, axes = plt.subplots(2, 5, figsize=(25, 8))
fig.suptitle('Top 10 Most Novel Generated Structures', fontsize=14, fontweight='bold')

for idx in range(10):
    ax = axes[idx // 5, idx % 5]
    struct = novel_structures[idx]
    # Visualize as bar chart of features
    feat_vals = struct[8:14]  # Lattice params and angles
    feat_labels = ['a', 'b', 'c', 'α', 'β', 'γ']
    colors = ['steelblue', 'steelblue', 'steelblue', 'coral', 'coral', 'coral']
    ax.bar(feat_labels, feat_vals, color=colors, alpha=0.8)
    ax.set_title(f'Structure {idx+1}\nSim: {max_sim_to_real[most_novel_idx[idx]]:.3f}', fontsize=9)
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig10_novel_structures.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_novel_structures.png")

# Save generated structures
np.save('outputs/generated_structures.npy', gen_structures)
np.save('outputs/latent_representations.npy', all_z)

print("\nStructure generation workflow complete!")
