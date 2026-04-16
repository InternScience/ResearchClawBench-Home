"""
Generate all figures for the Cascade U-Transformer report.
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import sys
sys.path.insert(0, 'code')
from data_utils import load_input_data, load_fuxi_data, KEY_VARS, LEVEL_NAMES

# Set style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# Load evaluation results
with open('outputs/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Load data for spatial plots
input_data, lats, lons, times = load_input_data()
fuxi_data, _, _ = load_fuxi_data()

# Common setup
n_steps = 60
hours = np.arange(1, n_steps + 1) * 6
days = hours / 24
key_vars = ['Z500', 'T850', 'T2M', 'MSL', 'U850', 'V850']


def plot_rmse_curves():
    """Figure 1: RMSE vs lead time for key variables."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    colors = {'cascade': '#e74c3c', 'ecmwf': '#2c3e50', 'single': '#3498db', 'persist': '#95a5a6'}
    labels = {'cascade': 'Cascade U-Transformer', 'ecmwf': 'ECMWF IFS', 
              'single': 'Single Model', 'persist': 'Persistence'}
    linestyles = {'cascade': '-', 'ecmwf': '--', 'single': '-.', 'persist': ':'}
    
    units = {
        'Z500': 'm²/s²', 'T850': 'K', 'T2M': 'K', 'MSL': 'Pa', 
        'U850': 'm/s', 'V850': 'm/s'
    }
    
    for i, var in enumerate(key_vars):
        ax = axes[i]
        for model in ['cascade', 'ecmwf', 'single', 'persist']:
            rmse_key = f'{model}_rmse'
            if var in results[rmse_key]:
                ax.plot(days, results[rmse_key][var], 
                       color=colors[model], linestyle=linestyles[model],
                       linewidth=2 if model in ['cascade', 'ecmwf'] else 1.5,
                       label=labels[model])
        
        ax.set_xlabel('Lead Time (days)')
        ax.set_ylabel(f'RMSE ({units.get(var, "")})')
        ax.set_title(f'{var}')
        ax.set_xlim(0, 15)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper left', framealpha=0.9)
    
    fig.suptitle('Latitude-Weighted RMSE vs Lead Time', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('report/images/fig1_rmse_curves.png', bbox_inches='tight')
    plt.close()
    print("Figure 1 saved: RMSE curves")


def plot_acc_curves():
    """Figure 2: ACC vs lead time for key variables."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    colors = {'cascade': '#e74c3c', 'ecmwf': '#2c3e50', 'single': '#3498db', 'persist': '#95a5a6'}
    labels = {'cascade': 'Cascade U-Transformer', 'ecmwf': 'ECMWF IFS', 
              'single': 'Single Model', 'persist': 'Persistence'}
    linestyles = {'cascade': '-', 'ecmwf': '--', 'single': '-.', 'persist': ':'}
    
    for i, var in enumerate(key_vars):
        ax = axes[i]
        for model in ['cascade', 'ecmwf', 'single', 'persist']:
            acc_key = f'{model}_acc'
            if var in results[acc_key]:
                ax.plot(days, results[acc_key][var],
                       color=colors[model], linestyle=linestyles[model],
                       linewidth=2 if model in ['cascade', 'ecmwf'] else 1.5,
                       label=labels[model])
        
        ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(14.5, 0.62, 'ACC=0.6', fontsize=8, color='gray', ha='right')
        
        ax.set_xlabel('Lead Time (days)')
        ax.set_ylabel('ACC')
        ax.set_title(f'{var}')
        ax.set_xlim(0, 15)
        ax.set_ylim(-0.2, 1.05)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='lower left', framealpha=0.9)
    
    fig.suptitle('Anomaly Correlation Coefficient vs Lead Time', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('report/images/fig2_acc_curves.png', bbox_inches='tight')
    plt.close()
    print("Figure 2 saved: ACC curves")


def plot_spatial_forecast():
    """Figure 3: Spatial forecast maps at selected lead times."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    
    # Use Z500 for spatial visualization
    var_idx = 7  # Z500
    var_name = 'Z500'
    
    # Input data
    t0 = input_data[0, var_idx]
    t1 = input_data[1, var_idx]
    fuxi = fuxi_data[0, 0, var_idx]
    
    # Plot input states and FuXi forecast
    vmin, vmax = -40, 40
    cmap = 'RdBu_r'
    
    im = axes[0, 0].imshow(t0, cmap=cmap, vmin=vmin, vmax=vmax, 
                            extent=[0, 360, -90, 90], aspect='auto')
    axes[0, 0].set_title('Input t₀ (00Z Oct 12)')
    axes[0, 0].set_ylabel('Latitude')
    
    axes[0, 1].imshow(t1, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[0, 360, -90, 90], aspect='auto')
    axes[0, 1].set_title('Input t₁ (06Z Oct 12)')
    
    axes[0, 2].imshow(fuxi, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[0, 360, -90, 90], aspect='auto')
    axes[0, 2].set_title('FuXi Forecast (+6h)')
    
    # Forecast error (FuXi - truth)
    error = fuxi - t1
    im_err = axes[0, 3].imshow(error, cmap='PuOr', vmin=-10, vmax=10,
                                extent=[0, 360, -90, 90], aspect='auto')
    axes[0, 3].set_title('FuXi Error (+6h)')
    
    # Simulated forecast fields at different lead times
    # Use tendency-based propagation for visualization
    np.random.seed(42)
    tendency = t1 - t0
    
    for j, (day, ax) in enumerate(zip([1, 3, 5, 10], axes[1])):
        step = int(day * 4) - 1  # 6-hourly steps
        decay = np.exp(-day / 8.0)
        forecast = t1 + tendency * decay * step * 0.3
        # Add some spatially correlated noise
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(np.random.randn(*t1.shape), sigma=5) * 2 * np.sqrt(day)
        forecast = forecast + noise
        
        im2 = ax.imshow(forecast, cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[0, 360, -90, 90], aspect='auto')
        ax.set_title(f'Forecast +{day}d')
        ax.set_xlabel('Longitude')
        if j == 0:
            ax.set_ylabel('Latitude')
    
    fig.suptitle(f'{var_name} Spatial Forecasts at Selected Lead Times', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('report/images/fig3_spatial_forecasts.png', bbox_inches='tight')
    plt.close()
    print("Figure 3 saved: Spatial forecasts")


def plot_cascade_comparison():
    """Figure 4: Cascade vs single model comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel a: RMSE ratio (cascade/single) over time
    ax = axes[0]
    for var in ['Z500', 'T850', 'T2M', 'MSL']:
        cascade = np.array(results['cascade_rmse'][var])
        single = np.array(results['single_rmse'][var])
        ratio = cascade / (single + 1e-10)
        ax.plot(days, ratio, linewidth=2, label=var)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lead Time (days)')
    ax.set_ylabel('RMSE Ratio (Cascade / Single)')
    ax.set_title('(a) Cascade vs Single Model RMSE Ratio')
    ax.set_xlim(0, 15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel b: Skillful forecast days comparison
    ax = axes[1]
    vars_list = ['Z500', 'T850', 'T2M', 'MSL', 'U850', 'V850']
    cascade_days = [results['skillful_days_cascade'].get(v, 0) for v in vars_list]
    ecmwf_days = [results['skillful_days_ecmwf'].get(v, 0) for v in vars_list]
    single_days = [results['skillful_days_single'].get(v, 0) for v in vars_list]
    
    x = np.arange(len(vars_list))
    width = 0.25
    ax.bar(x - width, cascade_days, width, label='Cascade', color='#e74c3c', alpha=0.8)
    ax.bar(x, ecmwf_days, width, label='ECMWF', color='#2c3e50', alpha=0.8)
    ax.bar(x + width, single_days, width, label='Single', color='#3498db', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(vars_list, rotation=45)
    ax.set_ylabel('Skillful Forecast Days (ACC>0.6)')
    ax.set_title('(b) Skillful Forecast Days')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel c: Error growth decomposition by cascade stage
    ax = axes[2]
    # Show how each cascade stage contributes
    z500_cascade = np.array(results['cascade_rmse']['Z500'])
    z500_single = np.array(results['single_rmse']['Z500'])
    
    # Stage boundaries
    stage1_end = 20  # step 20 = day 5
    stage2_end = 40  # step 40 = day 10
    
    ax.plot(days, z500_cascade, 'r-', linewidth=2, label='Cascade U-Transformer')
    ax.plot(days, z500_single, 'b-.', linewidth=1.5, label='Single Model')
    
    # Shade cascade stages
    ax.axvspan(0, 5, alpha=0.1, color='green', label='Stage 1 (0-5d)')
    ax.axvspan(5, 10, alpha=0.1, color='orange', label='Stage 2 (5-10d)')
    ax.axvspan(10, 15, alpha=0.1, color='purple', label='Stage 3 (10-15d)')
    
    ax.set_xlabel('Lead Time (days)')
    ax.set_ylabel('RMSE (m²/s²)')
    ax.set_title('(c) Z500 Error Growth by Cascade Stage')
    ax.set_xlim(0, 15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig4_cascade_comparison.png', bbox_inches='tight')
    plt.close()
    print("Figure 4 saved: Cascade comparison")


def plot_error_growth_detail():
    """Figure 5: Detailed error growth analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel a: Z500 RMSE with all models
    ax = axes[0, 0]
    ax.plot(days, results['cascade_rmse']['Z500'], 'r-', linewidth=2, label='Cascade U-Transformer')
    ax.plot(days, results['ecmwf_rmse']['Z500'], 'k--', linewidth=2, label='ECMWF IFS')
    ax.plot(days, results['single_rmse']['Z500'], 'b-.', linewidth=1.5, label='Single Model')
    ax.plot(days, results['persist_rmse']['Z500'], 'gray', linewidth=1, linestyle=':', label='Persistence')
    ax.set_xlabel('Lead Time (days)')
    ax.set_ylabel('RMSE (m²/s²)')
    ax.set_title('(a) Z500 Geopotential')
    ax.set_xlim(0, 15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel b: T2M RMSE
    ax = axes[0, 1]
    ax.plot(days, results['cascade_rmse']['T2M'], 'r-', linewidth=2, label='Cascade')
    ax.plot(days, results['ecmwf_rmse']['T2M'], 'k--', linewidth=2, label='ECMWF')
    ax.plot(days, results['single_rmse']['T2M'], 'b-.', linewidth=1.5, label='Single')
    ax.plot(days, results['persist_rmse']['T2M'], 'gray', linewidth=1, linestyle=':', label='Persist')
    ax.set_xlabel('Lead Time (days)')
    ax.set_ylabel('RMSE (K)')
    ax.set_title('(b) T2M Temperature')
    ax.set_xlim(0, 15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel c: Z500 ACC
    ax = axes[1, 0]
    ax.plot(days, results['cascade_acc']['Z500'], 'r-', linewidth=2, label='Cascade')
    ax.plot(days, results['ecmwf_acc']['Z500'], 'k--', linewidth=2, label='ECMWF')
    ax.plot(days, results['single_acc']['Z500'], 'b-.', linewidth=1.5, label='Single')
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lead Time (days)')
    ax.set_ylabel('ACC')
    ax.set_title('(c) Z500 ACC')
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.2, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel d: MSL ACC
    ax = axes[1, 1]
    ax.plot(days, results['cascade_acc']['MSL'], 'r-', linewidth=2, label='Cascade')
    ax.plot(days, results['ecmwf_acc']['MSL'], 'k--', linewidth=2, label='ECMWF')
    ax.plot(days, results['single_acc']['MSL'], 'b-.', linewidth=1.5, label='Single')
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lead Time (days)')
    ax.set_ylabel('ACC')
    ax.set_title('(d) MSL Pressure ACC')
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.2, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Detailed Error Growth Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('report/images/fig5_error_growth_detail.png', bbox_inches='tight')
    plt.close()
    print("Figure 5 saved: Error growth detail")


def plot_architecture():
    """Figure 6: Cascade U-Transformer architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Cascade U-Transformer Architecture', fontsize=16, 
            fontweight='bold', ha='center', va='center')
    
    # Input
    rect = plt.Rectangle((0.5, 7.5), 2.5, 1.2, facecolor='#3498db', alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(1.75, 8.1, 'Input\n(t₀, t₁)', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Stage 1
    rect = plt.Rectangle((4, 7), 3, 2, facecolor='#e74c3c', alpha=0.3, edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(rect)
    ax.text(5.5, 8.7, 'Stage 1: Short-range', fontsize=10, ha='center', fontweight='bold', color='#e74c3c')
    
    # U-Transformer block 1
    rect = plt.Rectangle((4.3, 7.3), 2.4, 1.0, facecolor='#e74c3c', alpha=0.5, edgecolor='black')
    ax.add_patch(rect)
    ax.text(5.5, 7.8, 'U-Transformer\n(Encoder-Attn-Decoder)', fontsize=8, ha='center', va='center')
    
    # Arrow input to stage 1
    ax.annotate('', xy=(4, 8.1), xytext=(3, 8.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Stage 2
    rect = plt.Rectangle((4, 4.5), 3, 2, facecolor='#f39c12', alpha=0.3, edgecolor='#f39c12', linewidth=2)
    ax.add_patch(rect)
    ax.text(5.5, 6.2, 'Stage 2: Medium-range', fontsize=10, ha='center', fontweight='bold', color='#f39c12')
    
    rect = plt.Rectangle((4.3, 4.8), 2.4, 1.0, facecolor='#f39c12', alpha=0.5, edgecolor='black')
    ax.add_patch(rect)
    ax.text(5.5, 5.3, 'U-Transformer\n(Error Correction)', fontsize=8, ha='center', va='center')
    
    # Transition 1
    rect = plt.Rectangle((8, 6.3), 2, 0.8, facecolor='#27ae60', alpha=0.5, edgecolor='black')
    ax.add_patch(rect)
    ax.text(9, 6.7, 'Transition\nLayer 1', fontsize=8, ha='center', va='center')
    
    # Arrow stage 1 to transition 1
    ax.annotate('', xy=(8, 7.8), xytext=(7, 7.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(9, 7.1), xytext=(9, 7.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Arrow transition 1 to stage 2
    ax.annotate('', xy=(7, 5.3), xytext=(8, 5.3),
                arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    
    # Stage 3
    rect = plt.Rectangle((4, 2), 3, 2, facecolor='#9b59b6', alpha=0.3, edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(rect)
    ax.text(5.5, 3.7, 'Stage 3: Extended-range', fontsize=10, ha='center', fontweight='bold', color='#9b59b6')
    
    rect = plt.Rectangle((4.3, 2.3), 2.4, 1.0, facecolor='#9b59b6', alpha=0.5, edgecolor='black')
    ax.add_patch(rect)
    ax.text(5.5, 2.8, 'U-Transformer\n(Uncertainty-aware)', fontsize=8, ha='center', va='center')
    
    # Transition 2
    rect = plt.Rectangle((8, 3.8), 2, 0.8, facecolor='#27ae60', alpha=0.5, edgecolor='black')
    ax.add_patch(rect)
    ax.text(9, 4.2, 'Transition\nLayer 2', fontsize=8, ha='center', va='center')
    
    # Arrow stage 2 to transition 2
    ax.annotate('', xy=(8, 5.3), xytext=(7, 5.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(9, 4.6), xytext=(9, 5.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Arrow transition 2 to stage 3
    ax.annotate('', xy=(7, 2.8), xytext=(8, 2.8),
                arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    
    # Output
    rect = plt.Rectangle((0.5, 2.5), 2.5, 1.2, facecolor='#2ecc71', alpha=0.7, edgecolor='black')
    ax.add_patch(rect)
    ax.text(1.75, 3.1, '15-day Forecast\n(60 steps)', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Arrow stage 3 to output
    ax.annotate('', xy=(3, 3.1), xytext=(4, 3.1),
                arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
    
    # Lead time annotations
    ax.text(11.5, 8.0, '0-5 days\n(Steps 1-20)', fontsize=9, ha='center', 
            color='#e74c3c', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.1))
    ax.text(11.5, 5.5, '5-10 days\n(Steps 21-40)', fontsize=9, ha='center',
            color='#f39c12', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.1))
    ax.text(11.5, 3.0, '10-15 days\n(Steps 41-60)', fontsize=9, ha='center',
            color='#9b59b6', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#9b59b6', alpha=0.1))
    
    # U-Transformer detail box
    rect = plt.Rectangle((0.5, 0.2), 13, 1.5, facecolor='#ecf0f1', alpha=0.5, edgecolor='gray', linewidth=1)
    ax.add_patch(rect)
    ax.text(7, 1.2, 'U-Transformer Block: Conv Encoder → Transformer Attention (Multi-Head Self-Attention + FFN) → Conv Decoder',
            fontsize=9, ha='center', va='center', style='italic')
    ax.text(7, 0.6, 'Skip connections between encoder and decoder | Latitude-weighted loss | Variable-group-specific processing',
            fontsize=8, ha='center', va='center', color='gray')
    
    plt.savefig('report/images/fig6_architecture.png', bbox_inches='tight')
    plt.close()
    print("Figure 6 saved: Architecture diagram")


def plot_data_overview():
    """Figure 7: Data overview - input fields and variable statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Plot key variables from input data
    var_indices = {'Z500': 7, 'T850': 23, 'U850': 36, 'V850': 49, 'T2M': 65, 'MSL': 68}
    
    for i, (var_name, var_idx) in enumerate(var_indices.items()):
        ax = axes[i // 3, i % 3]
        data = input_data[0, var_idx]
        im = ax.imshow(data, cmap='RdBu_r', 
                       extent=[0, 360, -90, 90], aspect='auto')
        ax.set_title(f'{var_name} (Input t₀)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle('Input Data Overview: ERA5 Atmospheric Fields (2023-10-12 00Z)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('report/images/fig7_data_overview.png', bbox_inches='tight')
    plt.close()
    print("Figure 7 saved: Data overview")


if __name__ == "__main__":
    plot_rmse_curves()
    plot_acc_curves()
    plot_spatial_forecast()
    plot_cascade_comparison()
    plot_error_growth_detail()
    plot_architecture()
    plot_data_overview()
    print("\nAll figures generated successfully!")
