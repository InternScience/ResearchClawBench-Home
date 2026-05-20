import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
import os

os.makedirs('report/images', exist_ok=True)

# Load data
cal = pd.read_csv('outputs/calibration_with_gp.csv')
md = pd.read_csv('outputs/vitrimer_calibrated.csv')
candidates = pd.read_csv('outputs/inverse_design_candidates.csv')
validation = pd.read_csv('outputs/validation_candidates.csv')

# Figure 5: Comprehensive validation
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Calibrated Tg distribution with candidate targets
ax = axes[0, 0]
ax.hist(md['tg_calibrated'], bins=60, color='steelblue', alpha=0.7, edgecolor='k', density=True, label='All vitrimers')
for target in [350, 400, 450, 500]:
    ax.axvline(target, color='red', linestyle='--', linewidth=2, label=f'Target {target}K')
ax.set_xlabel('Calibrated Tg (K)')
ax.set_ylabel('Density')
ax.set_title('Calibrated Tg Distribution & Design Targets')
ax.legend(fontsize=7, loc='upper left')

# 2. VAE predicted vs target
ax = axes[0, 1]
colors = {'350': 'blue', '400': 'green', '450': 'orange', '500': 'red'}
for target in [350, 400, 450, 500]:
    subset = candidates[candidates['target_tg'] == target]
    ax.scatter(subset['target_tg'], subset['predicted_tg'], 
               c=colors[str(target)], alpha=0.7, s=60, edgecolors='k', linewidths=0.5)
ax.plot([300, 550], [300, 550], 'k--', lw=1.5)
ax.set_xlabel('Target Tg (K)')
ax.set_ylabel('VAE Predicted Tg (K)')
ax.set_title('Inverse Design Accuracy')

# 3. Uncertainty analysis
ax = axes[0, 2]
ax.hist(md['tg_calibrated_std'], bins=50, color='coral', alpha=0.7, edgecolor='k')
ax.set_xlabel('GP Prediction Standard Deviation (K)')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Uncertainty Distribution')

# 4. MD vs Calibrated Tg for vitrimers
ax = axes[1, 0]
sample_idx = np.random.choice(len(md), 2000, replace=False)
ax.scatter(md['tg'].iloc[sample_idx], md['tg_calibrated'].iloc[sample_idx], 
           alpha=0.4, c='darkgreen', s=10, edgecolors='k', linewidths=0.1)
ax.plot([300, 600], [300, 600], 'r--', lw=1.5)
ax.set_xlabel('MD Simulated Tg (K)')
ax.set_ylabel('GP Calibrated Tg (K)')
ax.set_title('MD vs Calibrated Tg (Vitrimer Systems)')

# 5. Candidate reconstruction distances
ax = axes[1, 1]
x_pos = []
labels = []
acid_dists = []
epoxide_dists = []
for i, target in enumerate([350, 400, 450, 500]):
    subset = candidates[candidates['target_tg'] == target]
    x_pos.extend([i-0.2]*len(subset))
    x_pos.extend([i+0.2]*len(subset))
    acid_dists.extend(subset['acid_dist'].tolist())
    epoxide_dists.extend(subset['epoxide_dist'].tolist())
    labels.append(f'{target}K')

ax.boxplot([candidates[candidates['target_tg']==t]['acid_dist'].values for t in [350,400,450,500]], 
           positions=[0,1,2,3], widths=0.3, patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.7),
           medianprops=dict(color='black'))
ax.boxplot([candidates[candidates['target_tg']==t]['epoxide_dist'].values for t in [350,400,450,500]], 
           positions=[0.4,1.4,2.4,3.4], widths=0.3, patch_artist=True,
           boxprops=dict(facecolor='coral', alpha=0.7),
           medianprops=dict(color='black'))
ax.set_xticks([0.2, 1.2, 2.2, 3.2])
ax.set_xticklabels(labels)
ax.set_ylabel('Reconstruction Distance')
ax.set_title('Candidate Reconstruction Quality')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Acid'),
                   Patch(facecolor='coral', alpha=0.7, label='Epoxide')]
ax.legend(handles=legend_elements, fontsize=8)

# 6. Tg prediction error by target
ax = axes[1, 2]
errors = []
labels_err = []
for target in [350, 400, 450, 500]:
    subset = candidates[candidates['target_tg'] == target]
    err = (subset['predicted_tg'] - target).abs().values
    errors.append(err)
    labels_err.append(f'{target}K')
bp = ax.boxplot(errors, labels=labels_err, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('mediumpurple')
    patch.set_alpha(0.7)
ax.set_ylabel('|Predicted - Target| (K)')
ax.set_title('Design Error by Target Temperature')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('report/images/fig05_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# Generate summary table of top candidates
top_candidates = []
for target in [350, 400, 450, 500]:
    subset = candidates[candidates['target_tg'] == target].copy()
    subset['error'] = (subset['predicted_tg'] - target).abs()
    best = subset.nsmallest(1, 'error').iloc[0]
    top_candidates.append({
        'Target Tg (K)': target,
        'Predicted Tg (K)': round(best['predicted_tg'], 2),
        'Error (K)': round(best['error'], 4),
        'Acid SMILES': best['acid_smiles'],
        'Epoxide SMILES': best['epoxide_smiles'],
        'Acid Distance': round(best['acid_dist'], 2),
        'Epoxide Distance': round(best['epoxide_dist'], 2),
    })

top_df = pd.DataFrame(top_candidates)
top_df.to_csv('outputs/top_candidates_summary.csv', index=False)
print("\nTop candidates:")
print(top_df[['Target Tg (K)', 'Predicted Tg (K)', 'Error (K)', 'Acid Distance', 'Epoxide Distance']])
