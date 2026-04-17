"""Part 5: Summary Figure and Final Validation"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_001_20260416_221126"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report", "images")

# Load all results
with open(os.path.join(OUTPUT_DIR, "property_prediction_results.json")) as f:
    pred_results = json.load(f)
with open(os.path.join(OUTPUT_DIR, "structure_generation_results.json")) as f:
    gen_results = json.load(f)
with open(os.path.join(OUTPUT_DIR, "optimization_results.json")) as f:
    opt_results = json.load(f)
with open(os.path.join(OUTPUT_DIR, "feature_importance.json")) as f:
    fi_results = json.load(f)

# ============================================================
# FIGURE 10: Comprehensive Summary
# ============================================================
fig = plt.figure(figsize=(20, 14))
fig.suptitle('M-AI-Synth: Comprehensive AI Materials Science Pipeline Summary', 
             fontsize=16, fontweight='bold', y=0.98)

# Create grid
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

# --- Row 1: Property Prediction ---
# 1a: Model comparison radar/bar
ax1 = fig.add_subplot(gs[0, 0:2])
models = list(pred_results.keys())
short_names = ['LR', 'Ridge', 'Lasso', 'KNN', 'SVR', 'RF', 'GBR', 'MLP']
r2_vals = [pred_results[m]['R2'] for m in models]
mae_vals = [pred_results[m]['MAE'] for m in models]
colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

x = np.arange(len(short_names))
width = 0.35
bars1 = ax1.bar(x - width/2, r2_vals, width, label='R2', color='steelblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, mae_vals, width, label='MAE (eV)', color='coral', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, rotation=45)
ax1.set_ylabel('Score')
ax1.set_title('Property Prediction: Model Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 1b: Best model scatter
ax2 = fig.add_subplot(gs[0, 2])
data = np.load(os.path.join(OUTPUT_DIR, "processed_data.npz"))
X = data['sample_features']
y = data['target']
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
svr = SVR(kernel='rbf', C=10, epsilon=0.1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(svr, X_s, y, cv=kf)
ax2.scatter(y, y_pred, alpha=0.6, s=25, c='steelblue', edgecolors='black', linewidth=0.5)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=1.5)
ax2.set_xlabel('Actual (eV)')
ax2.set_ylabel('Predicted (eV)')
ax2.set_title(f'Best Model (SVR, R2={pred_results["Support Vector Regression"]["R2"]:.3f})')

# 1c: Feature importance
ax3 = fig.add_subplot(gs[0, 3])
fi = fi_results['rf_importance']
names = list(fi.keys())
vals = list(fi.values())
sorted_idx = np.argsort(vals)[::-1]
ax3.barh([names[i] for i in sorted_idx], [vals[i] for i in sorted_idx], color='mediumseagreen', edgecolor='black')
ax3.set_xlabel('Importance')
ax3.set_title('Feature Importance (RF)')
ax3.invert_yaxis()

# --- Row 2: Structure Generation ---
ax4 = fig.add_subplot(gs[1, 0:2])
lattice_a = data['lattice_a']
lattice_b = data['lattice_b']
ax4.scatter(lattice_a, lattice_b, c='steelblue', alpha=0.6, s=30, label='Original', edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Lattice a (Angstrom)')
ax4.set_ylabel('Lattice b (Angstrom)')
ax4.set_title('Lattice Parameter Space')
ax4.legend()

# Generation statistics
ax5 = fig.add_subplot(gs[1, 2])
methods = ['Original', 'GMM', 'VAE']
a_means = [gen_results['original']['a_mean'], gen_results['gmm']['generated_a_mean'], gen_results['vae']['generated_a_mean']]
a_stds = [gen_results['original']['a_std'], gen_results['gmm']['generated_a_std'], gen_results['vae']['generated_a_std']]
x_pos = np.arange(len(methods))
ax5.bar(x_pos, a_means, yerr=a_stds, capsize=5, color=['steelblue', 'coral', 'mediumseagreen'], edgecolor='black')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(methods)
ax5.set_ylabel('Lattice a (Angstrom)')
ax5.set_title('Generation Quality: Lattice a')

# KS test results
ax6 = fig.add_subplot(gs[1, 3])
ks_data = {
    'GMM-a': gen_results['gmm']['ks_test_a']['p_value'],
    'GMM-b': gen_results['gmm']['ks_test_b']['p_value'],
    'VAE-a': gen_results['vae']['ks_test_a']['p_value'],
    'VAE-b': gen_results['vae']['ks_test_b']['p_value']
}
bars = ax6.bar(ks_data.keys(), ks_data.values(), color=['coral', 'coral', 'mediumseagreen', 'mediumseagreen'], edgecolor='black')
ax6.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
ax6.set_ylabel('p-value')
ax6.set_title('KS Test (Distribution Match)')
ax6.legend()
ax6.tick_params(axis='x', rotation=45)

# --- Row 3: Optimization ---
ax7 = fig.add_subplot(gs[2, 0:2])
bo_conv = opt_results['convergence_history']
rand_conv = opt_results['random_search_best']
ax7.plot(range(len(bo_conv)), bo_conv, 'b-o', markersize=3, label='Bayesian Opt.', linewidth=2)
ax7.plot(range(len(rand_conv)), rand_conv, 'r--s', markersize=3, label='Random Search', linewidth=1.5, alpha=0.7)
ax7.set_xlabel('Iteration')
ax7.set_ylabel('Best Quality')
ax7.set_title('Optimization Convergence')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Optimal parameters
ax8 = fig.add_subplot(gs[2, 2])
temps = opt_results['all_temperatures']
times = opt_results['all_times']
quals = opt_results['all_qualities']
sc = ax8.scatter(temps, times, c=quals, cmap='RdYlGn', s=40, edgecolors='black', linewidth=0.5)
ax8.scatter([opt_results['optimal_temperature_K']], [opt_results['optimal_time_min']], 
            c='red', s=200, marker='*', zorder=5)
plt.colorbar(sc, ax=ax8, label='Quality')
ax8.set_xlabel('Temperature (K)')
ax8.set_ylabel('Time (min)')
ax8.set_title('Parameter Space Exploration')

# Summary statistics
ax9 = fig.add_subplot(gs[2, 3])
ax9.axis('off')
summary_text = (
    f"Pipeline Summary\n"
    f"{'='*30}\n\n"
    f"Property Prediction:\n"
    f"  Best: SVR (R2={pred_results['Support Vector Regression']['R2']:.3f})\n"
    f"  MAE: {pred_results['Support Vector Regression']['MAE']:.3f} eV\n\n"
    f"Structure Generation:\n"
    f"  GMM: 7 components\n"
    f"  KS p-val(a): {gen_results['gmm']['ks_test_a']['p_value']:.3f}\n\n"
    f"Optimization:\n"
    f"  Best quality: {opt_results['best_quality']:.3f}\n"
    f"  T*={opt_results['optimal_temperature_K']:.0f}K\n"
    f"  t*={opt_results['optimal_time_min']:.0f}min\n"
    f"  BO vs Random: +{opt_results['bo_improvement_over_random']:.3f}"
)
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(os.path.join(IMAGE_DIR, "comprehensive_summary.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 10: Comprehensive summary saved")

# ============================================================
# FIGURE 11: Workflow Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('AI-Driven Materials Discovery Pipeline', fontsize=16, fontweight='bold', pad=20)

# Boxes
boxes = [
    (1, 2.5, 'Multimodal\nData Input', 'lightblue'),
    (4.5, 4, 'Property\nPrediction', 'lightcoral'),
    (4.5, 2.5, 'Structure\nGeneration', 'lightgreen'),
    (4.5, 1, 'Synthesis\nOptimization', 'lightyellow'),
    (8.5, 4, 'ML Models\n(SVR, RF, GBR)', 'wheat'),
    (8.5, 2.5, 'Generative Models\n(VAE, GMM)', 'wheat'),
    (8.5, 1, 'Bayesian\nOptimization', 'wheat'),
    (12.5, 2.5, 'Accelerated\nMaterials\nDiscovery', 'plum')
]

for x, y, text, color in boxes:
    rect = plt.Rectangle((x-1, y-0.4), 2.2, 0.8, facecolor=color, edgecolor='black', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x+0.1, y, text, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

# Arrows
arrow_props = dict(arrowstyle='->', color='black', linewidth=1.5)
from matplotlib.patches import FancyArrowPatch
arrows = [
    ((2.2, 2.9), (3.5, 4.0)),
    ((2.2, 2.5), (3.5, 2.5)),
    ((2.2, 2.1), (3.5, 1.0)),
    ((5.7, 4.0), (7.5, 4.0)),
    ((5.7, 2.5), (7.5, 2.5)),
    ((5.7, 1.0), (7.5, 1.0)),
    ((9.7, 4.0), (11.5, 2.9)),
    ((9.7, 2.5), (11.5, 2.5)),
    ((9.7, 1.0), (11.5, 2.1)),
]
for start, end in arrows:
    ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

plt.savefig(os.path.join(IMAGE_DIR, "workflow_diagram.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 11: Workflow diagram saved")

# Verify all images exist
images = os.listdir(IMAGE_DIR)
print(f"\nAll generated figures ({len(images)}):")
for img in sorted(images):
    size = os.path.getsize(os.path.join(IMAGE_DIR, img))
    print(f"  {img}: {size/1024:.1f} KB")

print("\nPart 5 complete!")
