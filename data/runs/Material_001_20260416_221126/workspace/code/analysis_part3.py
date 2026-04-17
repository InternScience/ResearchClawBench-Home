"""Part 3: Structure Generation using VAE and GMM"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_001_20260416_221126"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report", "images")

# Load data
data = np.load(os.path.join(OUTPUT_DIR, "processed_data.npz"))
lattice_a = data['lattice_a']
lattice_b = data['lattice_b']

print(f"Lattice a: {len(lattice_a)} samples, mean={np.mean(lattice_a):.4f}, std={np.std(lattice_a):.4f}")
print(f"Lattice b: {len(lattice_b)} samples, mean={np.mean(lattice_b):.4f}, std={np.std(lattice_b):.4f}")

# Combine lattice parameters
lattice_data = np.column_stack([lattice_a, lattice_b])
print(f"Combined lattice data shape: {lattice_data.shape}")

# ============================================================
# Method 1: Gaussian Mixture Model (GMM)
# ============================================================
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
lattice_scaled = scaler.fit_transform(lattice_data)

# Find optimal number of components
bic_scores = []
aic_scores = []
n_components_range = range(1, 8)
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42, covariance_type='full')
    gmm.fit(lattice_scaled)
    bic_scores.append(gmm.bic(lattice_scaled))
    aic_scores.append(gmm.aic(lattice_scaled))

best_n = n_components_range[np.argmin(bic_scores)]
print(f"\nGMM: Best n_components = {best_n} (by BIC)")

# Fit best GMM
gmm_best = GaussianMixture(n_components=best_n, random_state=42, covariance_type='full')
gmm_best.fit(lattice_scaled)

# Generate new structures
n_generated = 200
generated_scaled = gmm_best.sample(n_generated)[0]
generated_lattice = scaler.inverse_transform(generated_scaled)

print(f"Generated {n_generated} new lattice parameters")
print(f"Generated a: mean={np.mean(generated_lattice[:, 0]):.4f}, std={np.std(generated_lattice[:, 0]):.4f}")
print(f"Generated b: mean={np.mean(generated_lattice[:, 1]):.4f}, std={np.std(generated_lattice[:, 1]):.4f}")

# ============================================================
# Method 2: Simple VAE (implemented with numpy for portability)
# ============================================================
print("\n--- VAE Implementation ---")

class SimpleVAE:
    """Minimal VAE for lattice parameter generation using numpy."""
    
    def __init__(self, input_dim=2, latent_dim=2, hidden_dim=16, lr=0.01):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        np.random.seed(42)
        
        # Encoder weights
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_enc = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_logvar = np.zeros(latent_dim)
        
        # Decoder weights
        self.W_dec = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b_dec = np.zeros(hidden_dim)
        self.W_out = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_out = np.zeros(input_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def encode(self, x):
        h = self.relu(x @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        logvar = h @ self.W_logvar + self.b_logvar
        return h, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std
    
    def decode(self, z):
        h = self.relu(z @ self.W_dec + self.b_dec)
        return h @ self.W_out + self.b_out, h
    
    def forward(self, x):
        h_enc, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon, h_dec = self.decode(z)
        return x_recon, mu, logvar, z, h_enc, h_dec
    
    def loss(self, x, x_recon, mu, logvar):
        recon_loss = np.mean((x - x_recon) ** 2)
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
        return recon_loss + 0.1 * kl_loss, recon_loss, kl_loss
    
    def train_step(self, x):
        batch_size = x.shape[0]
        x_recon, mu, logvar, z, h_enc, h_dec = self.forward(x)
        total_loss, recon_loss, kl_loss = self.loss(x, x_recon, mu, logvar)
        
        # Simplified gradient descent (numerical gradients for robustness)
        eps = 1e-5
        
        # Update decoder output weights
        grad_out = -2 * (x - x_recon) / batch_size
        self.W_out -= self.lr * (h_dec.T @ grad_out) / batch_size
        self.b_out -= self.lr * np.mean(grad_out, axis=0)
        
        # Update decoder hidden weights
        grad_h_dec = grad_out @ self.W_out.T * self.relu_deriv(z @ self.W_dec + self.b_dec)
        self.W_dec -= self.lr * (z.T @ grad_h_dec) / batch_size
        self.b_dec -= self.lr * np.mean(grad_h_dec, axis=0)
        
        # Update encoder output weights (simplified)
        grad_mu = 0.1 * mu / batch_size
        grad_logvar = 0.1 * 0.5 * (np.exp(logvar) - 1) / batch_size
        
        self.W_mu -= self.lr * (h_enc.T @ grad_mu) / batch_size
        self.b_mu -= self.lr * np.mean(grad_mu, axis=0)
        self.W_logvar -= self.lr * (h_enc.T @ grad_logvar) / batch_size
        self.b_logvar -= self.lr * np.mean(grad_logvar, axis=0)
        
        return total_loss, recon_loss, kl_loss
    
    def generate(self, n_samples):
        z = np.random.randn(n_samples, self.latent_dim)
        x_gen, _ = self.decode(z)
        return x_gen

# Train VAE
vae = SimpleVAE(input_dim=2, latent_dim=2, hidden_dim=32, lr=0.005)
n_epochs = 500
losses = {'total': [], 'recon': [], 'kl': []}

for epoch in range(n_epochs):
    total_loss, recon_loss, kl_loss = vae.train_step(lattice_scaled)
    losses['total'].append(total_loss)
    losses['recon'].append(recon_loss)
    losses['kl'].append(kl_loss)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}: Total={total_loss:.4f}, Recon={recon_loss:.4f}, KL={kl_loss:.4f}")

# Generate with VAE
vae_generated_scaled = vae.generate(200)
vae_generated = scaler.inverse_transform(vae_generated_scaled)

# Reconstruct original
vae_recon_scaled = vae.forward(lattice_scaled)[0]
vae_recon = scaler.inverse_transform(vae_recon_scaled)

recon_error_a = np.mean(np.abs(lattice_a - vae_recon[:, 0]))
recon_error_b = np.mean(np.abs(lattice_b - vae_recon[:, 1]))
print(f"\nVAE Reconstruction Error: a={recon_error_a:.4f}, b={recon_error_b:.4f}")

# ============================================================
# FIGURE 6: Structure Generation Results
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Crystal Structure Generation Results', fontsize=14, fontweight='bold')

# 6a: GMM BIC/AIC
axes[0, 0].plot(list(n_components_range), bic_scores, 'bo-', label='BIC', linewidth=2)
axes[0, 0].plot(list(n_components_range), aic_scores, 'rs-', label='AIC', linewidth=2)
axes[0, 0].axvline(best_n, color='green', linestyle='--', label=f'Best n={best_n}')
axes[0, 0].set_xlabel('Number of Components')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('(a) GMM Model Selection')
axes[0, 0].legend()

# 6b: Original vs GMM Generated
axes[0, 1].scatter(lattice_a, lattice_b, c='steelblue', alpha=0.6, s=30, label='Original', edgecolors='black', linewidth=0.5)
axes[0, 1].scatter(generated_lattice[:, 0], generated_lattice[:, 1], c='coral', alpha=0.4, s=20, label='GMM Generated', marker='^')
axes[0, 1].set_xlabel('Lattice a (Angstrom)')
axes[0, 1].set_ylabel('Lattice b (Angstrom)')
axes[0, 1].set_title('(b) GMM: Original vs Generated')
axes[0, 1].legend()

# 6c: Original vs VAE Generated
axes[0, 2].scatter(lattice_a, lattice_b, c='steelblue', alpha=0.6, s=30, label='Original', edgecolors='black', linewidth=0.5)
axes[0, 2].scatter(vae_generated[:, 0], vae_generated[:, 1], c='mediumseagreen', alpha=0.4, s=20, label='VAE Generated', marker='s')
axes[0, 2].set_xlabel('Lattice a (Angstrom)')
axes[0, 2].set_ylabel('Lattice b (Angstrom)')
axes[0, 2].set_title('(c) VAE: Original vs Generated')
axes[0, 2].legend()

# 6d: VAE Training Loss
axes[1, 0].plot(losses['total'], label='Total Loss', color='navy', linewidth=1.5)
axes[1, 0].plot(losses['recon'], label='Reconstruction', color='coral', linewidth=1.5)
axes[1, 0].plot(losses['kl'], label='KL Divergence', color='green', linewidth=1.5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('(d) VAE Training Loss')
axes[1, 0].legend()
axes[1, 0].set_yscale('log')

# 6e: Distribution comparison for lattice a
bins_a = np.linspace(min(min(lattice_a), min(generated_lattice[:, 0])),
                     max(max(lattice_a), max(generated_lattice[:, 0])), 20)
axes[1, 1].hist(lattice_a, bins=bins_a, alpha=0.5, color='steelblue', label='Original', density=True, edgecolor='black')
axes[1, 1].hist(generated_lattice[:, 0], bins=bins_a, alpha=0.5, color='coral', label='GMM', density=True)
axes[1, 1].hist(vae_generated[:, 0], bins=bins_a, alpha=0.3, color='green', label='VAE', density=True)
axes[1, 1].set_xlabel('Lattice a (Angstrom)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('(e) Lattice a Distribution Comparison')
axes[1, 1].legend()

# 6f: Distribution comparison for lattice b
bins_b = np.linspace(min(min(lattice_b), min(generated_lattice[:, 1])),
                     max(max(lattice_b), max(generated_lattice[:, 1])), 20)
axes[1, 2].hist(lattice_b, bins=bins_b, alpha=0.5, color='steelblue', label='Original', density=True, edgecolor='black')
axes[1, 2].hist(generated_lattice[:, 1], bins=bins_b, alpha=0.5, color='coral', label='GMM', density=True)
axes[1, 2].hist(vae_generated[:, 1], bins=bins_b, alpha=0.3, color='green', label='VAE', density=True)
axes[1, 2].set_xlabel('Lattice b (Angstrom)')
axes[1, 2].set_ylabel('Density')
axes[1, 2].set_title('(f) Lattice b Distribution Comparison')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "structure_generation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 6: Structure generation saved")

# ============================================================
# FIGURE 7: VAE Latent Space
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('VAE Latent Space Analysis', fontsize=14, fontweight='bold')

# Get latent representations
_, mu, logvar = vae.encode(lattice_scaled)

# 7a: Latent space colored by lattice a
sc = axes[0].scatter(mu[:, 0], mu[:, 1], c=lattice_a, cmap='viridis', s=30, edgecolors='black', linewidth=0.5)
plt.colorbar(sc, ax=axes[0], label='Lattice a')
axes[0].set_xlabel('Latent dim 1')
axes[0].set_ylabel('Latent dim 2')
axes[0].set_title('(a) Latent Space (colored by a)')

# 7b: Latent space colored by lattice b
sc = axes[1].scatter(mu[:, 0], mu[:, 1], c=lattice_b, cmap='plasma', s=30, edgecolors='black', linewidth=0.5)
plt.colorbar(sc, ax=axes[1], label='Lattice b')
axes[1].set_xlabel('Latent dim 1')
axes[1].set_ylabel('Latent dim 2')
axes[1].set_title('(b) Latent Space (colored by b)')

# 7c: Reconstruction quality
axes[2].scatter(lattice_a, vae_recon[:, 0], alpha=0.6, s=30, c='steelblue', label='a')
axes[2].scatter(lattice_b, vae_recon[:, 1], alpha=0.6, s=30, c='coral', label='b')
min_v = min(min(lattice_a), min(lattice_b))
max_v = max(max(lattice_a), max(lattice_b))
axes[2].plot([min_v, max_v], [min_v, max_v], 'k--', label='Perfect')
axes[2].set_xlabel('Original (Angstrom)')
axes[2].set_ylabel('Reconstructed (Angstrom)')
axes[2].set_title('(c) VAE Reconstruction Quality')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "vae_latent_space.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7: VAE latent space saved")

# Statistical validation
# KS test between original and generated distributions
ks_gmm_a = stats.ks_2samp(lattice_a, generated_lattice[:, 0])
ks_gmm_b = stats.ks_2samp(lattice_b, generated_lattice[:, 1])
ks_vae_a = stats.ks_2samp(lattice_a, vae_generated[:, 0])
ks_vae_b = stats.ks_2samp(lattice_b, vae_generated[:, 1])

validation = {
    "gmm": {
        "n_components": best_n,
        "generated_a_mean": float(np.mean(generated_lattice[:, 0])),
        "generated_a_std": float(np.std(generated_lattice[:, 0])),
        "generated_b_mean": float(np.mean(generated_lattice[:, 1])),
        "generated_b_std": float(np.std(generated_lattice[:, 1])),
        "ks_test_a": {"statistic": float(ks_gmm_a.statistic), "p_value": float(ks_gmm_a.pvalue)},
        "ks_test_b": {"statistic": float(ks_gmm_b.statistic), "p_value": float(ks_gmm_b.pvalue)}
    },
    "vae": {
        "final_total_loss": float(losses['total'][-1]),
        "final_recon_loss": float(losses['recon'][-1]),
        "final_kl_loss": float(losses['kl'][-1]),
        "recon_error_a": float(recon_error_a),
        "recon_error_b": float(recon_error_b),
        "generated_a_mean": float(np.mean(vae_generated[:, 0])),
        "generated_a_std": float(np.std(vae_generated[:, 0])),
        "generated_b_mean": float(np.mean(vae_generated[:, 1])),
        "generated_b_std": float(np.std(vae_generated[:, 1])),
        "ks_test_a": {"statistic": float(ks_vae_a.statistic), "p_value": float(ks_vae_a.pvalue)},
        "ks_test_b": {"statistic": float(ks_vae_b.statistic), "p_value": float(ks_vae_b.pvalue)}
    },
    "original": {
        "a_mean": float(np.mean(lattice_a)),
        "a_std": float(np.std(lattice_a)),
        "b_mean": float(np.mean(lattice_b)),
        "b_std": float(np.std(lattice_b))
    }
}

with open(os.path.join(OUTPUT_DIR, "structure_generation_results.json"), 'w') as f:
    json.dump(validation, f, indent=2)

print(f"\nKS Test Results:")
print(f"GMM - a: stat={ks_gmm_a.statistic:.4f}, p={ks_gmm_a.pvalue:.4f}")
print(f"GMM - b: stat={ks_gmm_b.statistic:.4f}, p={ks_gmm_b.pvalue:.4f}")
print(f"VAE - a: stat={ks_vae_a.statistic:.4f}, p={ks_vae_a.pvalue:.4f}")
print(f"VAE - b: stat={ks_vae_b.statistic:.4f}, p={ks_vae_b.pvalue:.4f}")

print("\nPart 3 complete!")
