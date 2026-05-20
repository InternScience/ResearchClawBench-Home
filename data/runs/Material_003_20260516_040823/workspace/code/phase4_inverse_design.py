#!/usr/bin/env python3
"""
Phase 4: Inverse Design - Generate Novel Vitrimer Candidates
Use the VAE latent space to identify promising regions and generate
candidates with desired glass transition temperatures.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import torch
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260516_040823')
DATA_DIR = WORKSPACE / 'data'
OUTPUTS_DIR = WORKSPACE / 'outputs'
IMAGES_DIR = WORKSPACE / 'report' / 'images'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)

# ============================================================
# Load artifacts
# ============================================================
print("Loading artifacts...")
latent_data = np.load(OUTPUTS_DIR / 'latent_representations.npz')
combined_latent = latent_data['combined_latent']
acid_latent = latent_data['acid_latent']
epoxide_latent = latent_data['epoxide_latent']
valid_indices = latent_data['valid_indices']
calibrated_tg = latent_data['calibrated_tg']

vitrimer_df = pd.read_csv(OUTPUTS_DIR / 'vitrimer_calibrated.csv')
vitrimer_valid = vitrimer_df.iloc[valid_indices].copy()

with open(OUTPUTS_DIR / 'property_predictor.pkl', 'rb') as f:
    rf = pickle.load(f)

print(f"Latent data: {combined_latent.shape}")
print(f"Calibrated Tg: {calibrated_tg.shape}")

# ============================================================
# Inverse Design Strategy
# ============================================================
# 1. Identify high-Tg vitrimers in latent space
# 2. Generate new latent vectors by interpolation and perturbation
# 3. Score using property predictor  
# 4. Rank and select candidates
# 5. Cross-reference to find closest real vitrimers

print("\n=== Inverse Design: Generating Novel Vitrimer Candidates ===")

# Define target Tg ranges
target_ranges = {
    'high': (420, 500),    # High-performance applications
    'medium': (370, 420),  # General purpose
    'low': (300, 350),     # Low-temperature applications
}

# ============================================================
# Method 1: Latent space interpolation between high-Tg pairs
# ============================================================
print("\nMethod 1: Latent space interpolation...")
high_tg_idx = np.where(calibrated_tg > 400)[0]
print(f"Number of high-Tg vitrimers (Tg > 400 K): {len(high_tg_idx)}")

# Select diverse high-Tg parent molecules using PCA clustering
pca_full = PCA(n_components=10)
latent_pca = pca_full.fit_transform(combined_latent[high_tg_idx])

# Pick 100 pairs of distant high-Tg molecules
np.random.seed(42)
n_interpolate = 200
interpolated_latent = []
for _ in range(n_interpolate):
    i, j = np.random.choice(len(high_tg_idx), 2, replace=False)
    alpha = np.random.beta(0.5, 0.5)  # Beta favors endpoints
    new_z = combined_latent[high_tg_idx[i]] * alpha + combined_latent[high_tg_idx[j]] * (1 - alpha)
    interpolated_latent.append(new_z)

interpolated_latent = np.array(interpolated_latent)
interp_tg_pred = rf.predict(interpolated_latent)
print(f"Generated {len(interpolated_latent)} interpolated candidates")
print(f"  Predicted Tg: mean={interp_tg_pred.mean():.1f}, std={interp_tg_pred.std():.1f}, range=[{interp_tg_pred.min():.1f}, {interp_tg_pred.max():.1f}]")

# ============================================================
# Method 2: Gaussian perturbation around top candidates
# ============================================================
print("\nMethod 2: Gaussian perturbation...")
# Find top 50 vitrimers by Tg
top_n = 50
top_idx = np.argsort(calibrated_tg)[-top_n:]
top_latent = combined_latent[top_idx]

# Compute local covariance for each parent
n_perturb = 10  # perturbations per parent
perturbed_latent = []
for z in top_latent:
    for _ in range(n_perturb):
        noise = np.random.randn(combined_latent.shape[1]) * 0.1 * np.std(combined_latent, axis=0)
        perturbed_latent.append(z + noise)

perturbed_latent = np.array(perturbed_latent)
perturb_tg_pred = rf.predict(perturbed_latent)
print(f"Generated {len(perturbed_latent)} perturbed candidates ({top_n} parents x {n_perturb})")
print(f"  Predicted Tg: mean={perturb_tg_pred.mean():.1f}, std={perturb_tg_pred.std():.1f}, range=[{perturb_tg_pred.min():.1f}, {perturb_tg_pred.max():.1f}]")

# ============================================================
# Method 3: Gradient-based optimization in latent space
# ============================================================
print("\nMethod 3: Gradient-guided latent space walk...")
# Simple approach: take steps along directions that increase Tg
# Using the RF feature importances as approximate gradients
importances = rf.feature_importances_

# Normalize importances to unit vector
grad_direction = importances / np.linalg.norm(importances)

# Start from top quartile of Tg
top_quartile = np.percentile(calibrated_tg, 75)
start_indices = np.where(calibrated_tg >= top_quartile)[0]
n_walks = 100
n_steps = 5
step_size = 0.5  # in latent space units

gradient_latent = []
for _ in range(n_walks):
    start_idx = np.random.choice(start_indices)
    z = combined_latent[start_idx].copy()
    for step in range(n_steps):
        # Take a step along gradient direction + random perturbation
        perturbation = np.random.randn(combined_latent.shape[1]) * 0.2
        z_new = z + step_size * grad_direction * (step + 1) + perturbation * 0.05
        tg_pred = rf.predict(z_new.reshape(1, -1))[0]
        # Accept if Tg improves or with some probability
        if tg_pred > rf.predict(z.reshape(1, -1))[0] or np.random.random() < 0.3:
            z = z_new
    gradient_latent.append(z)

gradient_latent = np.array(gradient_latent)
gradient_tg_pred = rf.predict(gradient_latent)
print(f"Generated {len(gradient_latent)} gradient-walked candidates")
print(f"  Predicted Tg: mean={gradient_tg_pred.mean():.1f}, std={gradient_tg_pred.std():.1f}, range=[{gradient_tg_pred.min():.1f}, {gradient_tg_pred.max():.1f}]")

# ============================================================
# Combine all generated candidates
# ============================================================
print("\nCombining all generated candidates...")
all_generated = np.vstack([interpolated_latent, perturbed_latent, gradient_latent])
all_tg_pred = np.concatenate([interp_tg_pred, perturb_tg_pred, gradient_tg_pred])

print(f"Total generated candidates: {len(all_generated)}")
print(f"Predicted Tg: mean={all_tg_pred.mean():.1f} ± {all_tg_pred.std():.1f}, range=[{all_tg_pred.min():.1f}, {all_tg_pred.max():.1f}]")

# ============================================================
# Map generated candidates to nearest real vitrimers
# ============================================================
print("\nMapping generated candidates to nearest real vitrimers...")
# For each generated latent vector, find the closest real vitrimer
nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn_model.fit(combined_latent)

distances, indices = nn_model.kneighbors(all_generated)

# Build candidate recommendations
candidate_results = []
for i in range(len(all_generated)):
    nearest_idx = indices[i][0]
    nearest_dist = distances[i][0]
    
    candidate_results.append({
        'candidate_id': i,
        'predicted_tg': float(all_tg_pred[i]),
        'nearest_vitrimer_idx': int(nearest_idx),
        'nearest_distance': float(nearest_dist),
        'nearest_acid_smiles': vitrimer_valid.iloc[nearest_idx]['acid'],
        'nearest_epoxide_smiles': vitrimer_valid.iloc[nearest_idx]['epoxide'],
        'nearest_actual_tg': float(calibrated_tg[nearest_idx]),
        'nearest_md_tg': float(vitrimer_valid.iloc[nearest_idx]['tg']),
    })

candidates_df = pd.DataFrame(candidate_results)
candidates_df = candidates_df.sort_values('predicted_tg', ascending=False)

# ============================================================
# Select top candidates for each target range
# ============================================================
print("\n=== Top Candidate Recommendations ===")

for range_name, (tg_min, tg_max) in target_ranges.items():
    mask = (candidates_df['predicted_tg'] >= tg_min) & (candidates_df['predicted_tg'] < tg_max)
    range_candidates = candidates_df[mask].head(10)
    print(f"\n{range_name.upper()} Tg [{tg_min}-{tg_max} K]:")
    for _, row in range_candidates.iterrows():
        print(f"  Candidate {int(row['candidate_id']):4d}: pred Tg={row['predicted_tg']:.1f} K, "
              f"nearest real Tg={row['nearest_actual_tg']:.1f} K, dist={row['nearest_distance']:.3f}")

# Save candidates
candidates_df.to_csv(OUTPUTS_DIR / 'generated_candidates.csv', index=False)
print(f"\nSaved {len(candidates_df)} candidates to generated_candidates.csv")

# Save top-50 candidates as recommendations
top50 = candidates_df.head(50)
top50.to_csv(OUTPUTS_DIR / 'top50_candidates.csv', index=False)

# ============================================================
# Figure 7: Inverse Design Results
# ============================================================
print("\nGenerating inverse design figures...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Predicted Tg distribution of generated vs real
axes[0, 0].hist(calibrated_tg, bins=40, alpha=0.5, color='steelblue', label=f'Real vitrimers (n={len(calibrated_tg)})', density=True)
axes[0, 0].hist(all_tg_pred, bins=40, alpha=0.5, color='coral', label=f'Generated (n={len(all_tg_pred)})', density=True)
axes[0, 0].set_xlabel('Predicted/Calibrated Tg (K)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('A: Tg Distribution: Real vs Generated Candidates')
axes[0, 0].legend()

# Panel B: Top candidates vs real vitrimers in PCA space
pca_2d = PCA(n_components=2)
all_latent_combined = np.vstack([combined_latent, all_generated])
all_pca = pca_2d.fit_transform(all_latent_combined)
real_pca = all_pca[:len(combined_latent)]
gen_pca = all_pca[len(combined_latent):]

# Color by predicted Tg
axes[0, 1].scatter(real_pca[:, 0], real_pca[:, 1], c=calibrated_tg, cmap='RdYlBu_r', s=3, alpha=0.5, label='Real')
# Highlight top-20 generated candidates
top20_mask = np.argsort(all_tg_pred)[-20:]
axes[0, 1].scatter(gen_pca[top20_mask, 0], gen_pca[top20_mask, 1], 
                   c='red', marker='*', s=100, edgecolors='black', linewidth=0.5,
                   label='Top 20 Generated', zorder=5)
axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
axes[0, 1].set_title('B: PCA: Real Vitrimers + Top Generated Candidates')
axes[0, 1].legend(markerscale=0.7)

# Panel C: Predicted Tg by generation method
methods = ['Interpolation', 'Perturbation', 'Gradient Walk']
method_data = [interp_tg_pred, perturb_tg_pred, gradient_tg_pred]
bp = axes[1, 0].boxplot(method_data, labels=methods, patch_artist=True, showmeans=True)
colors_box = ['#FFB74D', '#64B5F6', '#81C784']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].set_ylabel('Predicted Tg (K)')
axes[1, 0].set_title('C: Predicted Tg by Generation Method')
axes[1, 0].grid(axis='y', alpha=0.3)

# Panel D: Top candidates table with key info
top10 = candidates_df.head(10)
# Plot: distance to nearest vs predicted Tg
axes[1, 1].scatter(candidates_df['predicted_tg'], candidates_df['nearest_distance'], 
                   c=candidates_df['predicted_tg'], cmap='RdYlBu_r', s=10, alpha=0.5)
# Highlight top 10
axes[1, 1].scatter(top10['predicted_tg'], top10['nearest_distance'], 
                   c='red', marker='D', s=50, edgecolors='black', linewidth=0.5)
axes[1, 1].set_xlabel('Predicted Tg (K)')
axes[1, 1].set_ylabel('Distance to Nearest Real Vitrimer')
axes[1, 1].set_title('D: Candidate Diversity (Distance vs Tg)')

plt.tight_layout()
fig.savefig(IMAGES_DIR / 'figure7_inverse_design.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure7_inverse_design.png")

# ============================================================
# Figure 8: Candidate Validation
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Top candidates by target region
for ax, (range_name, (tg_min, tg_max)) in zip(axes, target_ranges.items()):
    mask = (candidates_df['predicted_tg'] >= tg_min) & (candidates_df['predicted_tg'] < tg_max)
    range_cands = candidates_df[mask]
    if len(range_cands) > 0:
        top5 = range_cands.head(5)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top5)))
        bars = ax.barh(range(len(top5)), top5['predicted_tg'].values, color=colors)
        ax.set_yticks(range(len(top5)))
        ax.set_yticklabels([f"C{int(c)}" for c in top5['candidate_id'].values])
        ax.axvline(tg_min, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(tg_max, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Predicted Tg (K)')
    ax.set_title(f'{range_name.capitalize()} Tg [{tg_min}-{tg_max} K]')

plt.tight_layout()
fig.savefig(IMAGES_DIR / 'figure8_candidate_targets.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure8_candidate_targets.png")

# ============================================================
# Export summary
# ============================================================
summary = {
    'total_generated': len(all_generated),
    'generated_tg_mean': float(all_tg_pred.mean()),
    'generated_tg_std': float(all_tg_pred.std()),
    'generated_tg_min': float(all_tg_pred.min()),
    'generated_tg_max': float(all_tg_pred.max()),
    'top10_predicted_tg': candidates_df.head(10)['predicted_tg'].tolist(),
    'top10_nearest_real_tg': candidates_df.head(10)['nearest_actual_tg'].tolist(),
    'methods': {
        'interpolation': {'n': len(interp_tg_pred), 'tg_mean': float(interp_tg_pred.mean()), 'tg_std': float(interp_tg_pred.std())},
        'perturbation': {'n': len(perturb_tg_pred), 'tg_mean': float(perturb_tg_pred.mean()), 'tg_std': float(perturb_tg_pred.std())},
        'gradient_walk': {'n': len(gradient_tg_pred), 'tg_mean': float(gradient_tg_pred.mean()), 'tg_std': float(gradient_tg_pred.std())},
    }
}

with open(OUTPUTS_DIR / 'inverse_design_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nPhase 4 complete!")
