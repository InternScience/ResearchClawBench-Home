#!/usr/bin/env python3
"""
Phase 1: Data Exploration and Overview
Load the calibration and vitrimer datasets, compute basic statistics,
and generate data overview figures.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Setup paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260516_040823')
DATA_DIR = WORKSPACE / 'data'
OUTPUTS_DIR = WORKSPACE / 'outputs'
IMAGES_DIR = WORKSPACE / 'report' / 'images'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)

# ============================================================
# Load Data
# ============================================================
print("Loading data...")
calib_df = pd.read_csv(DATA_DIR / 'tg_calibration.csv')
vitrimer_df = pd.read_csv(DATA_DIR / 'tg_vitrimer_MD.csv')

print(f"Calibration dataset: {len(calib_df)} entries")
print(f"Vitrimer dataset: {len(vitrimer_df)} entries")
print(f"Calibration columns: {list(calib_df.columns)}")
print(f"Vitrimer columns: {list(vitrimer_df.columns)}")

# ============================================================
# Basic Statistics
# ============================================================
stats = {}

# Calibration stats
calib_stats = {
    'n_samples': len(calib_df),
    'tg_exp': {
        'mean': float(calib_df['tg_exp'].mean()),
        'std': float(calib_df['tg_exp'].std()),
        'min': float(calib_df['tg_exp'].min()),
        'max': float(calib_df['tg_exp'].max()),
        'median': float(calib_df['tg_exp'].median()),
    },
    'tg_md': {
        'mean': float(calib_df['tg_md'].mean()),
        'std': float(calib_df['tg_md'].std()),
        'min': float(calib_df['tg_md'].min()),
        'max': float(calib_df['tg_md'].max()),
        'median': float(calib_df['tg_md'].median()),
    },
    'std': {
        'mean': float(calib_df['std'].mean()),
        'min': float(calib_df['std'].min()),
        'max': float(calib_df['std'].max()),
    }
}

# Vitrimer stats
vitrimer_stats = {
    'n_samples': len(vitrimer_df),
    'tg_md': {
        'mean': float(vitrimer_df['tg'].mean()),
        'std': float(vitrimer_df['tg'].std()),
        'min': float(vitrimer_df['tg'].min()),
        'max': float(vitrimer_df['tg'].max()),
        'median': float(vitrimer_df['tg'].median()),
    },
    'std': {
        'mean': float(vitrimer_df['std'].mean()),
        'min': float(vitrimer_df['std'].min()),
        'max': float(vitrimer_df['std'].max()),
    }
}

stats['calibration'] = calib_stats
stats['vitrimer'] = vitrimer_stats

with open(OUTPUTS_DIR / 'data_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\nCalibration Statistics:")
print(f"  tg_exp: mean={calib_stats['tg_exp']['mean']:.1f} ± {calib_stats['tg_exp']['std']:.1f}, range=[{calib_stats['tg_exp']['min']:.1f}, {calib_stats['tg_exp']['max']:.1f}]")
print(f"  tg_md:  mean={calib_stats['tg_md']['mean']:.1f} ± {calib_stats['tg_md']['std']:.1f}, range=[{calib_stats['tg_md']['min']:.1f}, {calib_stats['tg_md']['max']:.1f}]")

print("\nVitrimer Statistics:")
print(f"  tg_md:  mean={vitrimer_stats['tg_md']['mean']:.1f} ± {vitrimer_stats['tg_md']['std']:.1f}, range=[{vitrimer_stats['tg_md']['min']:.1f}, {vitrimer_stats['tg_md']['max']:.1f}]")

# ============================================================
# Figure 1: Calibration Data Overview
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Histogram of experimental Tg
ax = axes[0, 0]
ax.hist(calib_df['tg_exp'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(calib_df['tg_exp'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean = {calib_df['tg_exp'].mean():.1f} K")
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('Count')
ax.set_title('A: Experimental Tg Distribution (Calibration Set)')
ax.legend()

# Panel B: Histogram of MD Tg
ax = axes[0, 1]
ax.hist(calib_df['tg_md'], bins=40, color='coral', edgecolor='white', alpha=0.8)
ax.axvline(calib_df['tg_md'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean = {calib_df['tg_md'].mean():.1f} K")
ax.set_xlabel('MD Simulated Tg (K)')
ax.set_ylabel('Count')
ax.set_title('B: MD Tg Distribution (Calibration Set)')
ax.legend()

# Panel C: Tg_exp vs Tg_md with error bars
ax = axes[1, 0]
ax.errorbar(calib_df['tg_md'], calib_df['tg_exp'], 
            xerr=calib_df['std'], fmt='o', alpha=0.4, markersize=3,
            color='purple', ecolor='gray', capsize=0)
# Perfect calibration line
min_val = min(calib_df['tg_md'].min(), calib_df['tg_exp'].min())
max_val = max(calib_df['tg_md'].max(), calib_df['tg_exp'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='y = x')
ax.set_xlabel('MD Tg (K)')
ax.set_ylabel('Experimental Tg (K)')
ax.set_title('C: MD vs Experimental Tg')
ax.legend()

# Panel D: Residuals (tg_exp - tg_md)
ax = axes[1, 1]
residuals = calib_df['tg_exp'] - calib_df['tg_md']
ax.hist(residuals, bins=40, color='teal', edgecolor='white', alpha=0.8)
ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax.axvline(residuals.mean(), color='red', linestyle='-', linewidth=2, 
           label=f'Mean residual = {residuals.mean():.1f} K')
ax.set_xlabel('Residual (Exp - MD) (K)')
ax.set_ylabel('Count')
ax.set_title('D: Calibration Residuals')
ax.legend()

plt.tight_layout()
fig.savefig(IMAGES_DIR / 'figure1_calibration_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: figure1_calibration_overview.png")

# ============================================================
# Figure 2: Vitrimer Data Overview
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Histogram of vitrimer MD Tg
ax = axes[0]
ax.hist(vitrimer_df['tg'], bins=60, color='forestgreen', edgecolor='white', alpha=0.8)
ax.axvline(vitrimer_df['tg'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f"Mean = {vitrimer_df['tg'].mean():.1f} K")
ax.set_xlabel('MD Simulated Tg (K)')
ax.set_ylabel('Count')
ax.set_title('A: Vitrimer MD Tg Distribution')
ax.legend()

# Panel B: Tg uncertainty distribution
ax = axes[1]
ax.hist(vitrimer_df['std'], bins=50, color='darkorange', edgecolor='white', alpha=0.8)
ax.axvline(vitrimer_df['std'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean = {vitrimer_df['std'].mean():.1f} K")
ax.set_xlabel('MD Tg Std Dev (K)')
ax.set_ylabel('Count')
ax.set_title('B: Vitrimer Tg Uncertainty Distribution')
ax.legend()

plt.tight_layout()
fig.savefig(IMAGES_DIR / 'figure2_vitrimer_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure2_vitrimer_overview.png")

# ============================================================
# Figure 3: Comparison of Tg ranges
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

data_groups = [
    ('Calibration\nExperimental Tg', calib_df['tg_exp']),
    ('Calibration\nMD Tg', calib_df['tg_md']),
    ('Vitrimer\nMD Tg', vitrimer_df['tg']),
]

positions = [1, 2, 3]
colors = ['steelblue', 'coral', 'forestgreen']

bp = ax.boxplot([d[1] for d in data_groups], positions=positions, widths=0.5,
                patch_artist=True, showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels([d[0] for d in data_groups])
ax.set_ylabel('Temperature (K)')
ax.set_title('Comparison of Tg Distributions')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(IMAGES_DIR / 'figure3_tg_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure3_tg_comparison.png")

# ============================================================
# Save processed data summary
# ============================================================
summary = {
    'calibration_n': len(calib_df),
    'vitrimer_n': len(vitrimer_df),
    'calibration_tg_exp_range': [float(calib_df['tg_exp'].min()), float(calib_df['tg_exp'].max())],
    'calibration_tg_md_range': [float(calib_df['tg_md'].min()), float(calib_df['tg_md'].max())],
    'vitrimer_tg_md_range': [float(vitrimer_df['tg'].min()), float(vitrimer_df['tg'].max())],
    'calibration_correlation': float(calib_df['tg_exp'].corr(calib_df['tg_md'])),
}

with open(OUTPUTS_DIR / 'data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nCorrelation between MD and Experimental Tg (calibration): {summary['calibration_correlation']:.4f}")
print("\nPhase 1 complete!")
