"""
Phase 4: Generate all figures for the research report.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import r2_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
cal_df = pd.read_csv('data/tg_calibration.csv')
vit_df = pd.read_csv('data/tg_vitrimer_MD.csv')
gp_cal_results = pd.read_csv('outputs/gp_calibration_results.csv')
gp_vit_results = pd.read_csv('outputs/gp_vitrimer_predictions.csv')
gp_test = pd.read_csv('outputs/gp_test_predictions.csv')
gp_metrics = json.load(open('outputs/gp_metrics.json'))

with open('outputs/data_summary.json') as f:
    data_summary = json.load(f)

print("Generating figures...")

# Set style
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.family': 'sans-serif',
})

# ========== Figure 1: Data Overview ==========
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# 1a: Calibration Tg distribution
ax = fig.add_subplot(gs[0, 0])
ax.hist(cal_df['tg_exp'], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
ax.set_xlabel('Experimental Tg (K)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Calibration Dataset\nExperimental Tg Distribution', fontsize=13, fontweight='bold')
ax.axvline(cal_df['tg_exp'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {cal_df["tg_exp"].mean():.0f} K')
ax.legend(fontsize=10)

# 1b: MD vs Experimental Tg
ax = fig.add_subplot(gs[0, 1])
ax.scatter(cal_df['tg_md'], cal_df['tg_exp'], c='steelblue', s=15, alpha=0.6, edgecolors='none')
lims = [min(cal_df['tg_md'].min(), cal_df['tg_exp'].min()) - 20,
        max(cal_df['tg_md'].max(), cal_df['tg_exp'].max()) + 20]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('MD Simulated Tg (K)', fontsize=12)
ax.set_ylabel('Experimental Tg (K)', fontsize=12)
ax.set_title('MD vs Experimental Tg\nCalibration Dataset', fontsize=13, fontweight='bold')
bias = (cal_df['tg_md'] - cal_df['tg_exp']).mean()
ax.text(0.05, 0.95, f'Mean bias: {bias:.1f} K', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 1c: Vitrimer MD Tg distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(vit_df['tg'], bins=50, color='darkorange', edgecolor='white', alpha=0.85)
ax.set_xlabel('MD Simulated Tg (K)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Vitrimer Dataset\nMD Simulated Tg Distribution', fontsize=13, fontweight='bold')
ax.axvline(vit_df['tg'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {vit_df["tg"].mean():.0f} K')
ax.legend(fontsize=10)

# 1d: Calibration data by polymer class
ax = fig.add_subplot(gs[1, :2])
# Classify polymers by name prefix
def classify_polymer(name):
    name_lower = str(name).lower()
    if 'acryl' in name_lower or 'methacryl' in name_lower: return 'Acrylates/Methacrylates'
    if 'nylon' in name_lower or 'amide' in name_lower: return 'Polyamides (Nylons)'
    if 'styrene' in name_lower or 'vinyl' in name_lower: return 'Styrenics/Vinyls'
    if 'ester' in name_lower or 'terephthalate' in name_lower or 'succinate' in name_lower: return 'Polyesters'
    if 'ether' in name_lower or 'glycol' in name_lower: return 'Polyethers'
    if 'epoxy' in name_lower or 'bisphenol' in name_lower: return 'Epoxies'
    if 'butadiene' in name_lower or 'isoprene' in name_lower: return 'Dienes/Rubbers'
    if 'polyethylene' in name_lower or 'polypropylene' in name_lower or 'poly(' in name_lower: return 'Polyolefins'
    return 'Other'

cal_df['class'] = cal_df['name'].apply(classify_polymer)
class_means = cal_df.groupby('class')['tg_exp'].agg(['mean', 'std', 'count']).sort_values('mean')
classes = class_means.index
means = class_means['mean'].values
stds = class_means['std'].values
counts = class_means['count'].values

colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
bars = ax.barh(range(len(classes)), means, xerr=stds, color=colors, edgecolor='gray', capsize=3, height=0.7)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels([f'{c} (n={counts[i]})' for i, c in enumerate(classes)], fontsize=9)
ax.set_xlabel('Mean Experimental Tg (K)', fontsize=12)
ax.set_title('Glass Transition Temperature by Polymer Class', fontsize=13, fontweight='bold')
ax.axvline(cal_df['tg_exp'].mean(), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Overall mean')
ax.legend(fontsize=10)

# 1e: Summary statistics table
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
stats_text = (
    f"Calibration Dataset\n"
    f"{'='*30}\n"
    f"Total entries: {len(cal_df)}\n"
    f"Tg range: {cal_df['tg_exp'].min():.0f} - {cal_df['tg_exp'].max():.0f} K\n"
    f"Mean Tg: {cal_df['tg_exp'].mean():.1f} K\n"
    f"Std Tg: {cal_df['tg_exp'].std():.1f} K\n\n"
    f"Vitrimer Dataset\n"
    f"{'='*30}\n"
    f"Total entries: {len(vit_df)}\n"
    f"Tg range: {vit_df['tg'].min():.1f} - {vit_df['tg'].max():.1f} K\n"
    f"Mean Tg: {vit_df['tg'].mean():.1f} K\n"
    f"Std Tg: {vit_df['tg'].std():.1f} K\n\n"
    f"MD Bias (calibration)\n"
    f"{'='*30}\n"
    f"Mean MD bias: {(cal_df['tg_md'] - cal_df['tg_exp']).mean():.1f} K\n"
    f"MD overestimation: {((cal_df['tg_md'] > cal_df['tg_exp']).sum() / len(cal_df) * 100):.1f}%"
)
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.savefig('report/images/figure1_data_overview.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 1 saved: data overview")

# ========== Figure 2: GP Calibration Results ==========
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# 2a: Parity plot - predicted vs experimental (test set)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(gp_test['tg_exp'], gp_test['tg_pred'], c='steelblue', s=25, alpha=0.7, edgecolors='navy', linewidth=0.5)
lims = [min(gp_test['tg_exp'].min(), gp_test['tg_pred'].min()) - 20,
        max(gp_test['tg_exp'].max(), gp_test['tg_pred'].max()) + 20]
ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.5)
# Add error bars
for i in range(len(gp_test)):
    ax.errorbar(gp_test.iloc[i]['tg_exp'], gp_test.iloc[i]['tg_pred'], 
                yerr=gp_test.iloc[i]['pred_std'], fmt='none', alpha=0.3, color='gray')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Experimental Tg (K)', fontsize=12)
ax.set_ylabel('GP Predicted Tg (K)', fontsize=12)
ax.set_title('GP Calibration: Test Set\nPredicted vs Experimental', fontsize=13, fontweight='bold')
r2 = r2_score(gp_test['tg_exp'], gp_test['tg_pred'])
mae = np.mean(np.abs(gp_test['tg_exp'] - gp_test['tg_pred']))
ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f} K', transform=ax.transAxes,
        fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# 2b: Residual plot
ax = fig.add_subplot(gs[0, 1])
residuals = gp_test['tg_exp'] - gp_test['tg_pred']
ax.scatter(gp_test['tg_pred'], residuals, c='steelblue', s=25, alpha=0.7, edgecolors='navy', linewidth=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax.fill_between([gp_test['tg_pred'].min()-10, gp_test['tg_pred'].max()+10], 
                -gp_metrics['cv_rmse_mean'], gp_metrics['cv_rmse_mean'], 
                alpha=0.2, color='gray', label='±1 CV RMSE')
ax.set_xlabel('GP Predicted Tg (K)', fontsize=12)
ax.set_ylabel('Residual (Exp - Pred) (K)', fontsize=12)
ax.set_title('GP Calibration: Residual Analysis', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

# 2c: Calibrated vs MD Tg for calibration set
ax = fig.add_subplot(gs[0, 2])
ax.scatter(cal_df['tg_md'], gp_cal_results['tg_calibrated'], c='forestgreen', s=15, alpha=0.6, edgecolors='none')
ax.scatter(cal_df['tg_md'], cal_df['tg_exp'], c='red', s=10, alpha=0.4, edgecolors='none', label='Experimental')
lims = [min(cal_df['tg_md'].min(), gp_cal_results['tg_calibrated'].min()) - 20,
        max(cal_df['tg_md'].max(), gp_cal_results['tg_calibrated'].max()) + 20]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('MD Simulated Tg (K)', fontsize=12)
ax.set_ylabel('Calibrated Tg (K)', fontsize=12)
ax.set_title('GP-Calibrated Tg vs MD Tg', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

# 2d: Cross-validation results
ax = fig.add_subplot(gs[1, 0])
cv_results = pd.read_csv('outputs/gp_cv_results.csv')
folds = cv_results['fold'] + 1
width = 0.25
x = np.arange(5)
ax.bar(x - width, cv_results['mae'], width, label='MAE', color='steelblue', alpha=0.8)
ax.bar(x, cv_results['rmse'], width, label='RMSE', color='darkorange', alpha=0.8)
ax2 = ax.twinx()
ax2.bar(x + width, cv_results['r2'], width, label='R²', color='forestgreen', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i}' for i in folds])
ax.set_ylabel('Error (K)', fontsize=11)
ax2.set_ylabel('R² Score', fontsize=11)
ax.set_title('5-Fold Cross-Validation', fontsize=13, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

# 2e: Uncertainty analysis
ax = fig.add_subplot(gs[1, 1])
abs_errors = np.abs(gp_test['tg_exp'] - gp_test['tg_pred'])
sorted_idx = np.argsort(gp_test['pred_std'])
ax.scatter(gp_test.iloc[sorted_idx]['pred_std'].values, abs_errors[sorted_idx], 
           c='purple', s=30, alpha=0.7, edgecolors='none')
from scipy.stats import pearsonr
corr, pval = pearsonr(gp_test['pred_std'], abs_errors)
ax.plot(np.linspace(gp_test['pred_std'].min(), gp_test['pred_std'].max(), 100),
        np.poly1d(np.polyfit(gp_test['pred_std'], abs_errors, 1))(np.linspace(gp_test['pred_std'].min(), gp_test['pred_std'].max(), 100)),
        'r--', linewidth=2)
ax.set_xlabel('GP Predicted Uncertainty (K)', fontsize=12)
ax.set_ylabel('Absolute Error (K)', fontsize=12)
ax.set_title('Uncertainty Quantification', fontsize=13, fontweight='bold')
ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.4f}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2f: Model metrics summary
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
metrics_text = (
    f"GP Calibration Performance\n"
    f"{'='*30}\n\n"
    f"Test Set:\n"
    f"  MAE: {gp_metrics['test_mae']:.2f} K\n"
    f"  RMSE: {gp_metrics['test_rmse']:.2f} K\n"
    f"  R²: {gp_metrics['test_r2']:.4f}\n\n"
    f"5-Fold CV:\n"
    f"  MAE: {gp_metrics['cv_mae_mean']:.2f} ± {gp_metrics['cv_mae_std']:.2f} K\n"
    f"  RMSE: {gp_metrics['cv_rmse_mean']:.2f} ± {gp_metrics['cv_rmse_std']:.2f} K\n"
    f"  R²: {gp_metrics['cv_r2_mean']:.4f} ± {gp_metrics['cv_r2_std']:.4f}\n\n"
    f"Model:\n"
    f"  Kernel: {gp_metrics['gp_kernel'][:50]}...\n"
    f"  PCA components: {gp_metrics['pca_n_components']}\n"
    f"  PCA variance: {gp_metrics['pca_total_variance']:.4f}"
)
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))

plt.savefig('report/images/figure2_gp_calibration.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 2 saved: GP calibration")

# ========== Figure 3: Vitrimer Predictions ==========
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# 3a: Calibrated Tg distribution for vitrimers
ax = fig.add_subplot(gs[0, 0])
ax.hist(gp_vit_results['tg_calibrated'], bins=50, color='teal', edgecolor='white', alpha=0.85, density=True)
ax.hist(vit_df['tg'], bins=50, color='gray', edgecolor='white', alpha=0.5, density=True, label='MD Tg')
ax.set_xlabel('Calibrated Tg (K)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Vitrimer Calibrated Tg Distribution', fontsize=13, fontweight='bold')
ax.axvline(gp_vit_results['tg_calibrated'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean = {gp_vit_results["tg_calibrated"].mean():.0f} K')
ax.legend(fontsize=10)

# 3b: MD vs Calibrated Tg scatter
ax = fig.add_subplot(gs[0, 1])
# Sample for visibility
sample_idx = np.random.choice(len(gp_vit_results), min(2000, len(gp_vit_results)), replace=False)
ax.scatter(vit_df.iloc[sample_idx]['tg'], gp_vit_results.iloc[sample_idx]['tg_calibrated'],
           c='teal', s=8, alpha=0.4, edgecolors='none')
lims = [min(vit_df['tg'].min(), gp_vit_results['tg_calibrated'].min()) - 20,
        max(vit_df['tg'].max(), gp_vit_results['tg_calibrated'].max()) + 20]
ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('MD Simulated Tg (K)', fontsize=12)
ax.set_ylabel('GP Calibrated Tg (K)', fontsize=12)
ax.set_title('Vitrimer: MD vs Calibrated Tg', fontsize=13, fontweight='bold')
shift = gp_vit_results['tg_calibrated'].mean() - vit_df['tg'].mean()
ax.text(0.05, 0.95, f'Mean shift: {shift:.1f} K', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3c: Uncertainty distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(gp_vit_results['calibration_std'], bins=40, color='darkred', edgecolor='white', alpha=0.85)
ax.set_xlabel('Calibration Uncertainty (K)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Vitrimer Prediction Uncertainty', fontsize=13, fontweight='bold')
ax.axvline(gp_vit_results['calibration_std'].mean(), color='black', linestyle='--', linewidth=2,
           label=f'Mean = {gp_vit_results["calibration_std"].mean():.1f} K')
ax.legend(fontsize=10)

# 3d: Tg ranges for vitrimer selection
ax = fig.add_subplot(gs[1, :2])
tg_ranges = [(300, 350, 'Low Tg (300-350 K)'), (350, 400, 'Medium Tg (350-400 K)'), 
             (400, 450, 'High Tg (400-450 K)')]
range_counts = []
range_colors = ['#2ecc71', '#f39c12', '#e74c3c']
for t_min, t_max, label in tg_ranges:
    count = ((gp_vit_results['tg_calibrated'] >= t_min) & (gp_vit_results['tg_calibrated'] <= t_max)).sum()
    range_counts.append(count)

bars = ax.bar([l for _, _, l in tg_ranges], range_counts, color=range_colors, edgecolor='gray', alpha=0.85, width=0.6)
for bar, count in zip(bars, range_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'n = {count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Vitrimers', fontsize=12)
ax.set_title('Vitrimer Candidates by Target Tg Range', fontsize=13, fontweight='bold')

# Highlight the overall distribution
ax.axhline(y=len(gp_vit_results)/10, color='gray', linestyle=':', alpha=0.5)

# 3e: Top candidates table
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
# Select top candidates (lowest uncertainty in each range)
top_candidates = []
for t_min, t_max, label in tg_ranges:
    mask = (gp_vit_results['tg_calibrated'] >= t_min) & (gp_vit_results['tg_calibrated'] <= t_max)
    subset = gp_vit_results[mask].nsmallest(3, 'calibration_std')
    for _, row in subset.iterrows():
        acid_short = row['acid'][:30] + '...' if len(row['acid']) > 30 else row['acid']
        top_candidates.append({
            'Range': label,
            'Tg (K)': f"{row['tg_calibrated']:.1f}",
            'Unc.': f"±{row['calibration_std']:.1f}"
        })

table_text = "Top Low-Uncertainty Candidates\n" + "="*35 + "\n\n"
for tc in top_candidates:
    table_text += f"{tc['Range']}\n"
    table_text += f"  Tg = {tc['Tg (K)']} K\n"
    table_text += f"  Uncertainty: {tc['Unc.']}\n\n"
table_text += f"\nTotal vitrimers analyzed: {len(gp_vit_results)}\n"
table_text += f"Calibrated Tg range: {gp_vit_results['tg_calibrated'].min():.1f} - {gp_vit_results['tg_calibrated'].max():.1f} K"

ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.savefig('report/images/figure3_vitrimer_predictions.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 3 saved: vitrimer predictions")

# ========== Figure 4: VAE Latent Space & Inverse Design ==========
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# Try to load VAE latent space data
try:
    latent_meta = pd.read_csv('outputs/latent_space_data.csv')
    latent_cols = [c for c in latent_meta.columns if c.startswith('latent_dim_')]
    latent_vecs = latent_meta[latent_cols].values
    has_latent = True
except:
    has_latent = False

if has_latent:
    # 4a: t-SNE of latent space
    from sklearn.manifold import TSNE
    # Use a sample for t-SNE
    n_sample = min(1000, len(latent_vecs))
    sample_idx = np.random.choice(len(latent_vecs), n_sample, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vecs[sample_idx])
    
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c='steelblue', s=8, alpha=0.5)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('VAE Latent Space\n(t-SNE Projection)', fontsize=13, fontweight='bold')
else:
    ax = fig.add_subplot(gs[0, 0])
    ax.text(0.5, 0.5, 'VAE Latent Space\nVisualization', ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('VAE Latent Space', fontsize=13, fontweight='bold')
    ax.axis('off')

# 4b: Molecular descriptor distributions for generated candidates
ax = fig.add_subplot(gs[0, 1])
try:
    cand_df = pd.read_csv('outputs/generated_candidates.csv')
    if len(cand_df) > 0:
        # Compare calibration vs generated
        from rdkit.Chem import Descriptors as RDKitDesc
        
        # Sample from calibration
        cal_sample = cal_df.sample(min(200, len(cal_df)), random_state=42)
        cal_mw = []
        cal_logp = []
        for _, row in cal_sample.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                cal_mw.append(RDKitDesc.MolWt(mol))
                cal_logp.append(RDKitDesc.MolLogP(mol))
        
        ax.scatter(cal_mw, cal_logp, c='steelblue', s=20, alpha=0.6, label='Calibration', edgecolors='none')
        ax.scatter(cand_df['mol_wt'], cand_df['logp'], c='red', s=30, alpha=0.8, label='Generated', 
                   marker='x', linewidths=2)
        ax.set_xlabel('Molecular Weight (Da)', fontsize=12)
        ax.set_ylabel('LogP', fontsize=12)
        ax.set_title('Chemical Space:\nCalibration vs Generated', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No generated candidates', ha='center', va='center', transform=ax.transAxes)
except Exception as e:
    ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)

# 4c: Generated candidates by target range
ax = fig.add_subplot(gs[0, 2])
try:
    if len(cand_df) > 0:
        for label, color in [('low_Tg', '#2ecc71'), ('mid_Tg', '#f39c12'), ('high_Tg', '#e74c3c')]:
            subset = cand_df[cand_df['target_range'] == label]
            if len(subset) > 0:
                ax.hist(subset['predicted_tg'], bins=15, color=color, alpha=0.7, label=label.replace('_', ' '), edgecolor='white')
        ax.set_xlabel('Predicted Tg (K)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Generated Candidates\nby Target Tg Range', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No candidates generated', ha='center', va='center', transform=ax.transAxes)
except:
    ax.text(0.5, 0.5, 'No candidates', ha='center', va='center', transform=ax.transAxes)

# 4d: VAE reconstruction quality
ax = fig.add_subplot(gs[1, :2])
# Show example molecules from different regions of chemical space
example_smiles = [
    ("Poly(methyl methacrylate)", "*CC(*)(C)C(=O)OC"),
    ("Polystyrene", "*CC(*)c1ccccc1"),
    ("Poly(ethylene glycol)", "*CCO*"),
    ("Bisphenol-A epoxy", "*CC(O)COc1ccc(C(C)(C)c2ccc(O*)cc2)cc1"),
    ("Poly(caprolactone)", "*CCCCCC(=O)O*"),
]

for i, (name, smiles) in enumerate(example_smiles):
    ax_sub = plt.axes([0.08 + i*0.17, 0.15, 0.14, 0.7])
    try:
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles.replace('*', 'C'))  # Replace * for visualization
        if mol:
            img = Draw.MolToImage(mol, size=(200, 150))
            ax_sub.imshow(img)
            ax_sub.set_title(name, fontsize=8, pad=5)
        else:
            ax_sub.text(0.5, 0.5, 'Invalid', ha='center', va='center', transform=ax_sub.transAxes)
    except:
        ax_sub.text(0.5, 0.5, smiles[:15]+'...', ha='center', va='center', fontsize=6, transform=ax_sub.transAxes)
    ax_sub.axis('off')

ax.set_title('Representative Polymer Structures', fontsize=13, fontweight='bold')
ax.axis('off')

# 4e: Inverse design workflow schematic
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
workflow = (
    "Inverse Design Workflow\n"
    f"{'='*30}\n\n"
    "1. Train VAE on polymer SMILES\n"
    "   → Learn continuous latent space\n\n"
    "2. Map latent vectors → Tg\n"
    "   → MLP surrogate model\n\n"
    "3. Sample latent space\n"
    "   → Target specific Tg ranges\n\n"
    "4. Decode to SMILES\n"
    "   → Nearest-neighbor retrieval\n\n"
    "5. Validate candidates\n"
    "   → GP-calibrated Tg prediction\n"
    "   → Molecular descriptor analysis\n\n"
    f"{'='*30}\n"
    f"Candidates generated: {len(cand_df) if 'cand_df' in dir() else 0}\n"
    f"Valid molecules: {len(cand_df[cand_df['is_valid']==True]) if 'cand_df' in dir() else 0}"
)
ax.text(0.05, 0.95, workflow, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.8))

plt.savefig('report/images/figure4_vae_inverse_design.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 4 saved: VAE inverse design")

# ========== Figure 5: Validation & Comparison ==========
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# 5a: GP calibration improvement over raw MD
ax = fig.add_subplot(gs[0, 0])
md_errors = np.abs(cal_df['tg_md'] - cal_df['tg_exp'])
cal_errors = np.abs(gp_cal_results['tg_calibrated'] - cal_df['tg_exp'])

ax.hist(md_errors, bins=30, color='gray', alpha=0.6, label='MD Error', edgecolor='white')
ax.hist(cal_errors, bins=30, color='steelblue', alpha=0.7, label='GP-Calibrated Error', edgecolor='white')
ax.set_xlabel('Absolute Error (K)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Error Reduction:\nMD vs GP-Calibrated', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.axvline(md_errors.mean(), color='gray', linestyle='--', linewidth=2)
ax.axvline(cal_errors.mean(), color='steelblue', linestyle='--', linewidth=2)

# 5b: Cumulative error distribution
ax = fig.add_subplot(gs[0, 1])
sorted_md = np.sort(md_errors)
sorted_cal = np.sort(cal_errors)
percentile = np.arange(len(sorted_md)) / len(sorted_md) * 100
ax.plot(sorted_md, percentile, 'gray', linewidth=2, label='MD')
ax.plot(sorted_cal, percentile, 'steelblue', linewidth=2, label='GP-Calibrated')
ax.set_xlabel('Absolute Error (K)', fontsize=12)
ax.set_ylabel('Cumulative %', fontsize=12)
ax.set_title('Cumulative Error Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 5c: Tg prediction accuracy by polymer class
ax = fig.add_subplot(gs[0, 2])
cal_df_plot = cal_df.copy()
cal_df_plot['class'] = cal_df_plot['name'].apply(classify_polymer)
cal_df_plot['gp_error'] = np.abs(gp_cal_results['tg_calibrated'] - cal_df_plot['tg_exp'])
cal_df_plot['md_error'] = np.abs(cal_df_plot['tg_md'] - cal_df_plot['tg_exp'])

class_errors = cal_df_plot.groupby('class')[['md_error', 'gp_error']].mean().sort_values('gp_error')

x = np.arange(len(class_errors))
width = 0.35
bars1 = ax.bar(x - width/2, class_errors['md_error'], width, label='MD Error', color='gray', alpha=0.7)
bars2 = ax.bar(x + width/2, class_errors['gp_error'], width, label='GP Error', color='steelblue', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(class_errors.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean Absolute Error (K)', fontsize=11)
ax.set_title('Error by Polymer Class', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

# 5d: Candidate property comparison
ax = fig.add_subplot(gs[1, 0])
try:
    if len(cand_df) > 0 and 'mol_wt' in cand_df.columns:
        # Box plots comparing properties
        props_to_compare = ['mol_wt', 'logp', 'num_atoms']
        prop_labels = ['Molecular Weight', 'LogP', 'Num Atoms']
        
        cal_props = {}
        for _, row in cal_sample.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                if 'mol_wt' not in cal_props: cal_props['mol_wt'] = []
                cal_props['mol_wt'].append(RDKitDesc.MolWt(mol))
                if 'logp' not in cal_props: cal_props['logp'] = []
                cal_props['logp'].append(RDKitDesc.MolLogP(mol))
                if 'num_atoms' not in cal_props: cal_props['num_atoms'] = []
                cal_props['num_atoms'].append(mol.GetNumAtoms())
        
        data_to_plot = []
        labels_to_plot = []
        colors_box = []
        
        for prop, label in zip(props_to_compare, prop_labels):
            # Normalize for comparison
            cal_vals = np.array(cal_props[prop])
            gen_vals = cand_df[prop].values
            
            cal_norm = (cal_vals - cal_vals.mean()) / cal_vals.std()
            gen_norm = (gen_vals - cal_vals.mean()) / cal_vals.std()
            
            data_to_plot.extend([cal_norm, gen_norm])
            labels_to_plot.extend([f'{label}\n(Cal)', f'{label}\n(Gen)'])
            colors_box.extend(['steelblue', 'red'])
        
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Normalized Value', fontsize=11)
        ax.set_title('Property Comparison:\nCalibration vs Generated', fontsize=13, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No candidate data', ha='center', va='center', transform=ax.transAxes)
except Exception as e:
    ax.text(0.5, 0.5, f'Error: {str(e)[:40]}', ha='center', va='center', transform=ax.transAxes)

# 5e: Recommended candidates for experimental validation
ax = fig.add_subplot(gs[1, 1:])
ax.axis('off')

# Select best candidates from each Tg range
recommendations = []
try:
    if len(cand_df) > 0:
        for label in ['low_Tg', 'mid_Tg', 'high_Tg']:
            subset = cand_df[cand_df['target_range'] == label]
            if len(subset) > 0:
                best = subset.nsmallest(2, 'distance' if 'distance' in subset.columns else 'mol_wt')
                for _, row in best.iterrows():
                    recommendations.append({
                        'Target': label.replace('_', ' ').title(),
                        'Predicted Tg': f"{row['predicted_tg']:.1f} K",
                        'MW': f"{row['mol_wt']:.1f}",
                        'LogP': f"{row['logp']:.2f}",
                        'SMILES': row['smiles'][:40] + '...' if len(row['smiles']) > 40 else row['smiles']
                    })
except:
    pass

rec_text = "Recommended Candidates for Experimental Validation\n" + "="*55 + "\n\n"
if recommendations:
    for r in recommendations:
        rec_text += f"Target: {r['Target']}\n"
        rec_text += f"  Predicted Tg: {r['Predicted Tg']}\n"
        rec_text += f"  MW: {r['MW']} Da, LogP: {r['LogP']}\n"
        rec_text += f"  SMILES: {r['SMILES']}\n\n"
else:
    rec_text += "No candidates available for recommendation.\n"
    rec_text += "The GP-calibrated vitrimer dataset provides\n"
    rec_text += f"{len(gp_vit_results)} candidates with calibrated Tg values.\n\n"
    rec_text += "Top recommendations from vitrimer dataset:\n"
    for t_min, t_max, label in [(300, 350, 'Low'), (350, 400, 'Medium'), (400, 450, 'High')]:
        mask = (gp_vit_results['tg_calibrated'] >= t_min) & (gp_vit_results['tg_calibrated'] <= t_max)
        subset = gp_vit_results[mask].nsmallest(1, 'calibration_std')
        if len(subset) > 0:
            row = subset.iloc[0]
            rec_text += f"\n  {label} Tg Target ({t_min}-{t_max}K):\n"
            rec_text += f"    Calibrated Tg: {row['tg_calibrated']:.1f} ± {row['calibration_std']:.1f} K\n"
            acid_short = row['acid'][:40] + '...' if len(row['acid']) > 40 else row['acid']
            rec_text += f"    Acid: {acid_short}\n"

rec_text += f"\n{'='*55}\n"
rec_text += f"Framework Summary:\n"
rec_text += f"  GP calibration MAE: {gp_metrics['test_mae']:.1f} K\n"
rec_text += f"  Vitrimer candidates: {len(gp_vit_results)}\n"
rec_text += f"  Calibrated Tg range: {gp_vit_results['tg_calibrated'].min():.1f} - {gp_vit_results['tg_calibrated'].max():.1f} K"

ax.text(0.02, 0.98, rec_text, transform=ax.transAxes, fontsize=8.5,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

plt.savefig('report/images/figure5_validation.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 5 saved: validation")

print("\nAll figures generated successfully!")
