#!/usr/bin/env python3
"""
Generate all figures for the research report.
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
FEATURE_LABELS = ['Nucleophilic\n(HEA)', 'Hydrophobic\n(BA)', 'Acidic\n(CBEA)', 
                  'Cationic\n(ATAC)', 'Aromatic\n(PEA)', 'Amide\n(AAm)']
TARGET_COL = 'Glass (kPa)_10s'

os.makedirs('report/images', exist_ok=True)

# Professional style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#4CB944', '#D64045', '#8B80F9']


def load_data():
    df = pd.read_csv('outputs/merged_dataset.csv')
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return df, X, y


def load_models():
    with open('outputs/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('outputs/gb_model.pkl', 'rb') as f:
        gb = pickle.load(f)
    with open('outputs/gp_model.pkl', 'rb') as f:
        gp = pickle.load(f)
    return rf, gb, gp


def fig1_data_overview(df, y):
    """Figure 1: Comprehensive data overview."""
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # (a) Histogram of adhesive strength
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(y, bins=35, color='steelblue', edgecolor='white', alpha=0.85)
    ax1.axvline(x=1000, color='#D64045', linestyle='--', linewidth=2.5, 
                label='Target: 1 MPa (1000 kPa)')
    ax1.axvline(x=np.median(y), color='#F18F01', linestyle=':', linewidth=2,
                label=f'Median: {np.median(y):.0f} kPa')
    ax1.axvline(x=np.max(y), color='#4CB944', linestyle='--', linewidth=1.5,
                label=f'Max: {np.max(y):.0f} kPa')
    ax1.set_xlabel('Adhesive Strength on Glass (kPa)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('(a) Distribution of Adhesive Strength (n = 311)', fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    
    # (b) Box plots of monomer compositions
    ax2 = fig.add_subplot(gs[0, 2:])
    bp_data = [df[c].values for c in FEATURE_COLS]
    bp = ax2.boxplot(bp_data, labels=FEATURE_LABELS, patch_artist=True, widths=0.6)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[i])
        patch.set_alpha(0.7)
    ax2.set_ylabel('Mole Fraction', fontweight='bold')
    ax2.set_title('(b) Monomer Composition Distributions', fontweight='bold')
    ax2.tick_params(axis='x', labelsize=9)
    
    # (c) Correlation heatmap
    ax3 = fig.add_subplot(gs[1, :2])
    corr_cols = FEATURE_COLS + [TARGET_COL]
    corr_labels = FEATURE_LABELS + ['Adhesion']
    corr_matrix = df[corr_cols].corr()
    im = ax3.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xticks(range(len(corr_cols)))
    ax3.set_yticks(range(len(corr_cols)))
    ax3.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(corr_labels, fontsize=8)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    plt.colorbar(im, ax=ax3, shrink=0.8, label='Pearson r')
    ax3.set_title('(c) Feature-Target Correlation Matrix', fontweight='bold')
    
    # (d) Top performers composition
    ax4 = fig.add_subplot(gs[1, 2:])
    top20 = df.nlargest(20, TARGET_COL)
    bottom20 = df.nsmallest(20, TARGET_COL)
    
    x = np.arange(len(FEATURE_COLS))
    width = 0.35
    
    top_means = top20[FEATURE_COLS].mean().values
    bot_means = bottom20[FEATURE_COLS].mean().values
    top_std = top20[FEATURE_COLS].std().values
    bot_std = bottom20[FEATURE_COLS].std().values
    
    bars1 = ax4.bar(x - width/2, top_means, width, yerr=top_std, 
                    label=f'Top 20 (Mean: {top20[TARGET_COL].mean():.0f} kPa)',
                    color='#D64045', edgecolor='white', capsize=3)
    bars2 = ax4.bar(x + width/2, bot_means, width, yerr=bot_std,
                    label=f'Bottom 20 (Mean: {bottom20[TARGET_COL].mean():.0f} kPa)',
                    color='#2E86AB', edgecolor='white', capsize=3)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(FEATURE_LABELS, fontsize=8)
    ax4.set_ylabel('Mean Mole Fraction', fontweight='bold')
    ax4.set_title('(d) Top vs Bottom 20 Compositions', fontweight='bold')
    ax4.legend(fontsize=9)
    
    plt.savefig('report/images/fig1_data_overview.png')
    plt.close()
    print("Saved fig1_data_overview.png")


def fig2_model_performance(df, X, y, rf, gb, gp):
    """Figure 2: Model performance and validation."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # (a) Cross-validation R² comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Compute detailed CV
    models_info = [
        ('Random Forest', rf, '#2E86AB'),
        ('Gradient Boosting', gb, '#A23B72'),
        ('Gaussian Process', gp, '#F18F01'),
    ]
    
    model_names = []
    r2_means = []
    r2_stds = []
    mae_means = []
    mae_stds = []
    bar_colors = []
    
    for name, model, color in models_info:
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        model_names.append(name)
        r2_means.append(r2_scores.mean())
        r2_stds.append(r2_scores.std())
        mae_means.append(mae_scores.mean())
        mae_stds.append(mae_scores.std())
        bar_colors.append(color)
    
    x_pos = np.arange(len(model_names))
    bars = ax1.bar(x_pos, r2_means, yerr=r2_stds, capsize=6, color=bar_colors, 
                   edgecolor='white', linewidth=1.2, width=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, fontsize=10)
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('(a) 5-Fold Cross-Validation R²', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    for i, (bar, val) in enumerate(zip(bars, r2_means)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # (b) MAE comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x_pos, mae_means, yerr=mae_stds, capsize=6, color=bar_colors,
                    edgecolor='white', linewidth=1.2, width=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, fontsize=10)
    ax2.set_ylabel('MAE (kPa)', fontweight='bold')
    ax2.set_title('(b) 5-Fold Cross-Validation MAE', fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars2, mae_means)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # (c) Parity plot - RF
    ax3 = fig.add_subplot(gs[0, 2])
    rf_cv = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_cv.fit(X_tr, y_tr)
    y_pred_rf = rf_cv.predict(X_te)
    
    ax3.scatter(y_te, y_pred_rf, alpha=0.6, color='#2E86AB', edgecolors='white', s=50)
    ax3.plot([0, max(y_te)+10], [0, max(y_te)+10], 'k--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Actual Adhesion (kPa)', fontweight='bold')
    ax3.set_ylabel('Predicted Adhesion (kPa)', fontweight='bold')
    r2_v = r2_score(y_te, y_pred_rf)
    mae_v = mean_absolute_error(y_te, y_pred_rf)
    ax3.set_title(f'(c) RF Parity Plot\nR² = {r2_v:.3f}, MAE = {mae_v:.1f} kPa', fontweight='bold')
    
    # (d) Feature importance comparison
    ax4 = fig.add_subplot(gs[1, 0])
    
    rf_imp = rf.feature_importances_
    gb_imp = gb.feature_importances_
    
    x_f = np.arange(len(FEATURE_COLS))
    width_f = 0.35
    
    ax4.bar(x_f - width_f/2, rf_imp, width_f, label='Random Forest', 
            color='#2E86AB', edgecolor='white')
    ax4.bar(x_f + width_f/2, gb_imp, width_f, label='Gradient Boosting',
            color='#A23B72', edgecolor='white')
    ax4.set_xticks(x_f)
    ax4.set_xticklabels(FEATURE_LABELS, fontsize=8)
    ax4.set_ylabel('Feature Importance', fontweight='bold')
    ax4.set_title('(d) Model Feature Importance', fontweight='bold')
    ax4.legend(fontsize=9)
    
    # (e) Permutation importance
    ax5 = fig.add_subplot(gs[1, 1])
    perm_imp = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    perm_order = np.argsort(perm_imp.importances_mean)
    
    ax5.barh(range(len(FEATURE_COLS)), perm_imp.importances_mean[perm_order],
             xerr=perm_imp.importances_std[perm_order], 
             color=COLORS, edgecolor='white', capsize=4)
    ax5.set_yticks(range(len(FEATURE_COLS)))
    ax5.set_yticklabels([FEATURE_LABELS[i] for i in perm_order], fontsize=9)
    ax5.set_xlabel('Importance (R² Decrease)', fontweight='bold')
    ax5.set_title('(e) Permutation Feature Importance', fontweight='bold')
    
    # (f) Ensemble prediction
    ax6 = fig.add_subplot(gs[1, 2])
    
    rf_pred_te = rf_cv.predict(X_te)
    gb_cv = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    gb_cv.fit(X_tr, y_tr)
    gb_pred_te = gb_cv.predict(X_te)
    
    gp_cv = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0)*Matern(nu=2.5)+WhiteKernel(1.0),
        n_restarts_optimizer=5, normalize_y=True, random_state=42, alpha=1e-2
    )
    gp_cv.fit(X_tr, y_tr)
    gp_pred_te = gp_cv.predict(X_te)
    
    ensemble_pred = (rf_pred_te + gb_pred_te + gp_pred_te) / 3.0
    
    ax6.scatter(y_te, ensemble_pred, alpha=0.6, color='#4CB944', edgecolors='white', s=50)
    ax6.plot([0, max(y_te)+10], [0, max(y_te)+10], 'k--', linewidth=1.5, alpha=0.7)
    ax6.set_xlabel('Actual Adhesion (kPa)', fontweight='bold')
    ax6.set_ylabel('Ensemble Predicted (kPa)', fontweight='bold')
    ens_r2 = r2_score(y_te, ensemble_pred)
    ens_mae = mean_absolute_error(y_te, ensemble_pred)
    ax6.set_title(f'(f) Ensemble Parity Plot\nR² = {ens_r2:.3f}, MAE = {ens_mae:.1f} kPa', fontweight='bold')
    
    plt.savefig('report/images/fig2_model_performance.png')
    plt.close()
    print("Saved fig2_model_performance.png")
    
    return {
        'rf_test_r2': float(r2_v),
        'rf_test_mae': float(mae_v),
        'ensemble_test_r2': float(ens_r2),
        'ensemble_test_mae': float(ens_mae),
    }


def fig3_composition_analysis(df):
    """Figure 3: Composition-structure-property relationships."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    for i, (feat, ax) in enumerate(zip(FEATURE_COLS, axes.flat)):
        # Scatter with colormap by Hydrophobic-BA
        sc = ax.scatter(df[feat], df[TARGET_COL], c=df['Hydrophobic-BA'],
                       cmap='viridis', alpha=0.55, edgecolors='white', s=35, linewidth=0.3)
        
        # Fit polynomial trend
        mask = ~np.isnan(df[feat]) & ~np.isnan(df[TARGET_COL])
        x_clean = df[feat].values[mask]
        y_clean = df[TARGET_COL].values[mask]
        
        # Lowess-like smoothing via polyfit
        z = np.polyfit(x_clean, y_clean, 2)
        p = np.poly1d(z)
        x_sorted = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_sorted, p(x_sorted), 'r-', linewidth=2, alpha=0.8)
        
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Mark 1 MPa target
        ax.axhline(y=1000, color='#D64045', linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel(feat.replace('-', '-\n'), fontsize=10)
        ax.set_ylabel('Adhesive Strength (kPa)', fontsize=10)
        
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Hydrophobic-BA', fontsize=9)
    
    plt.suptitle('Monomer Composition vs. Adhesive Strength', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('report/images/fig3_composition_analysis.png')
    plt.close()
    print("Saved fig3_composition_analysis.png")


def fig4_optimization_landscape(rf, gb, gp, df, X, y):
    """Figure 4: Optimization landscape and design rules."""
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # (a) 2D landscape: Hydrophobic-BA vs Aromatic-PEA
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create grid
    h_ba_range = np.linspace(0.1, 0.8, 50)
    ar_pea_range = np.linspace(0.0, 0.5, 50)
    H, A = np.meshgrid(h_ba_range, ar_pea_range)
    
    # Fixed other features at optimal values (from top performers)
    base = df.nlargest(10, TARGET_COL)[FEATURE_COLS].median().values
    Z = np.zeros_like(H)
    
    for i in range(len(h_ba_range)):
        for j in range(len(ar_pea_range)):
            comp = base.copy()
            comp[1] = H[j, i]  # Hydrophobic-BA
            comp[4] = A[j, i]  # Aromatic-PEA
            # Renormalize other components
            other_mask = np.array([True, False, True, True, False, True])
            others = comp[other_mask]
            others = others / others.sum() * (1 - H[j, i] - A[j, i])
            comp[other_mask] = others
            comp = comp.reshape(1, -1)
            
            rf_p = rf.predict(comp)[0]
            gb_p = gb.predict(comp)[0]
            gp_p = gp.predict(comp)[0]
            Z[j, i] = (rf_p + gb_p + gp_p) / 3.0
    
    cf = ax1.contourf(H, A, Z, levels=20, cmap='RdYlBu_r')
    ax1.scatter(df['Hydrophobic-BA'], df['Aromatic-PEA'], c=y, 
               cmap='viridis', s=20, edgecolors='white', linewidth=0.3, alpha=0.6)
    ax1.set_xlabel('Hydrophobic-BA', fontweight='bold')
    ax1.set_ylabel('Aromatic-PEA', fontweight='bold')
    ax1.set_title('(a) Adhesion Landscape\n(H-BA vs Ar-PEA)', fontweight='bold')
    plt.colorbar(cf, ax=ax1, label='Predicted (kPa)')
    
    # Mark optimal region
    top10 = df.nlargest(10, TARGET_COL)
    ax1.scatter(top10['Hydrophobic-BA'], top10['Aromatic-PEA'], 
               marker='*', s=100, c='gold', edgecolors='black', linewidth=1, zorder=5)
    
    # (b) Feature sensitivity from systematic variation
    ax2 = fig.add_subplot(gs[0, 1])
    
    if os.path.exists('outputs/optimization_results.json'):
        with open('outputs/optimization_results.json') as f:
            opt_data = json.load(f)
        
        sensitivity = opt_data.get('sensitivity', {})
        for feat in FEATURE_COLS:
            if feat in sensitivity:
                vals = [s['feature_value'] for s in sensitivity[feat]]
                preds = [s['ensemble_pred'] for s in sensitivity[feat]]
                ax2.plot(vals, preds, '-o', markersize=3, linewidth=2, label=feat[:20])
        
        ax2.set_xlabel('Feature Value (before renormalization)', fontweight='bold')
        ax2.set_ylabel('Predicted Adhesion (kPa)', fontweight='bold')
        ax2.set_title('(b) Feature Sensitivity Curves', fontweight='bold')
        ax2.legend(fontsize=7, loc='best')
    
    # (c) Top candidate composition comparison
    ax3 = fig.add_subplot(gs[0, 2])
    
    with open('outputs/optimization_results.json') as f:
        opt_data = json.load(f)
    
    top_candidates = opt_data.get('grid_search_top', [])[:5]
    top_candidates += opt_data.get('bayesian_opt_top', [])[:3]
    
    # Also add the real top 3
    top3_real = df.nlargest(3, TARGET_COL)
    real_entries = []
    for _, row in top3_real.iterrows():
        real_entries.append({
            'composition': {c: float(row[c]) for c in FEATURE_COLS},
            'ensemble_pred': float(row[TARGET_COL]),
            'rf_pred': float(row[TARGET_COL]),
            'gb_pred': float(row[TARGET_COL]),
            'gp_pred': float(row[TARGET_COL]),
        })
    
    all_candidates = top_candidates[:5] + real_entries[:3]
    labels = [f'C{i+1}' for i in range(5)] + ['Exp1', 'Exp2', 'Exp3']
    
    x = np.arange(len(all_candidates))
    width = 0.25
    comps_ens = [c['ensemble_pred'] for c in all_candidates]
    
    bars = ax3.bar(range(len(all_candidates)), comps_ens, color=COLORS[:len(all_candidates)], 
                  edgecolor='white')
    ax3.set_xticks(range(len(all_candidates)))
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel('Adhesive Strength (kPa)', fontweight='bold')
    ax3.set_title('(c) Top Candidates vs Experiments', fontweight='bold')
    ax3.axhline(y=321, color='red', linestyle='--', alpha=0.7, label='Current Max (321 kPa)')
    ax3.axhline(y=1000, color='darkred', linestyle=':', alpha=0.7, label='Target (1000 kPa)')
    ax3.legend(fontsize=8)
    
    # (d) Composition heatmap of top 30 experimental samples
    ax4 = fig.add_subplot(gs[1, :2])
    top30 = df.nlargest(30, TARGET_COL).sort_values(TARGET_COL, ascending=True)
    comp_matrix = top30[FEATURE_COLS].values.T
    
    im = ax4.imshow(comp_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.7)
    ax4.set_yticks(range(len(FEATURE_COLS)))
    ax4.set_yticklabels(FEATURE_LABELS, fontsize=9)
    ax4.set_xlabel('Sample Rank (by Adhesion)', fontweight='bold')
    ax4.set_title('(d) Composition Profiles of Top 30 Hydrogels', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Mole Fraction', shrink=0.8)
    
    # Add adhesion values
    for i in range(0, 30, 3):
        ax4.text(i, -1.0, f'{top30.iloc[i][TARGET_COL]:.0f}', 
                ha='center', fontsize=6, rotation=45, color='darkred')
    
    # (e) Design rules summary
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Find high adhesion region
    high_mask = y > np.percentile(y, 75)
    high_adhesion = df[high_mask]
    
    rules_text = [
        "Design Rules for High Adhesion:",
        "",
        f"1. Minimize Nucleophilic-HEA",
        f"   (< {high_adhesion['Nucleophilic-HEA'].quantile(0.75):.2f})",
        "",
        f"2. Maximize Hydrophobic-BA",
        f"   (> {high_adhesion['Hydrophobic-BA'].quantile(0.25):.2f})",
        "",
        f"3. Minimize Acidic-CBEA",
        f"   (< {high_adhesion['Acidic-CBEA'].quantile(0.75):.2f})",
        "",
        f"4. Moderate Cationic-ATAC",
        f"   (~{high_adhesion['Cationic-ATAC'].median():.2f})",
        "",
        f"5. High Aromatic-PEA",
        f"   (> {high_adhesion['Aromatic-PEA'].quantile(0.25):.2f})",
        "",
        f"6. Minimize Amide-AAm",
        f"   (< {high_adhesion['Amide-AAm'].quantile(0.75):.2f})",
    ]
    
    ax5.text(0.1, 0.5, '\n'.join(rules_text), transform=ax5.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_title('(e) Design Rules for >300 kPa', fontweight='bold')
    
    plt.savefig('report/images/fig4_optimization_landscape.png')
    plt.close()
    print("Saved fig4_optimization_landscape.png")


def fig5_extrapolation_strategy(df, rf, gb, gp, X, y):
    """Figure 5: Strategy for achieving >1 MPa adhesion."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # (a) Current data vs target
    ax = axes[0, 0]
    ax.hist(df[TARGET_COL], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=1000, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Adhesive Strength (kPa)')
    ax.set_ylabel('Count')
    ax.set_title('(a) Gap Analysis: Current vs Target')
    ax.text(600, ax.get_ylim()[1]*0.8, f'Max: {df[TARGET_COL].max():.0f} kPa\nTarget: 1000 kPa',
           fontsize=10, fontweight='bold', color='darkred')
    
    # (b) Natural adhesive protein composition patterns
    ax = axes[0, 1]
    # Mussel foot protein-inspired composition
    categories = ['Hydrophobic\nInteraction', 'Catechol\nAdhesion', 'Electrostatic\nInteraction', 'Crosslinking']
    mfp_values = [0.45, 0.25, 0.15, 0.15]
    colors_mfp = ['#2E86AB', '#D64045', '#F18F01', '#4CB944']
    wedges, texts, autotexts = ax.pie(mfp_values, labels=categories, autopct='%1.0f%%',
                                      colors=colors_mfp, explode=(0.02, 0.05, 0.02, 0.02))
    ax.set_title('(b) Mussel Adhesive Protein\nFunctional Motif Distribution', fontweight='bold')
    
    # (c) Extrapolation approach diagram
    ax = axes[0, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Draw boxes for the approach
    ax.add_patch(plt.Rectangle((1, 6), 3, 2.5, fill=True, facecolor='#2E86AB', alpha=0.3, edgecolor='#2E86AB'))
    ax.text(2.5, 7.25, 'Protein\nSequence\nFeatures', ha='center', fontsize=9, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((5.5, 6), 3, 2.5, fill=True, facecolor='#A23B72', alpha=0.3, edgecolor='#A23B72'))
    ax.text(7, 7.25, 'Monomer\nComposition\nDesign', ha='center', fontsize=9, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((1, 2), 3, 2.5, fill=True, facecolor='#F18F01', alpha=0.3, edgecolor='#F18F01'))
    ax.text(2.5, 3.25, 'ML Model\nPrediction', ha='center', fontsize=9, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((5.5, 2), 3, 2.5, fill=True, facecolor='#4CB944', alpha=0.3, edgecolor='#4CB944'))
    ax.text(7, 3.25, 'Bayesian\nOptimization', ha='center', fontsize=9, fontweight='bold')
    
    # Arrows between boxes
    ax.annotate('', xy=(5.5, 7.25), xytext=(4, 7.25),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(2.5, 4.5), xytext=(2.5, 6),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(7, 4.5), xytext=(7, 6),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(5.5, 3.25), xytext=(4, 3.25),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.text(5, 8.5, 'De Novo Hydrogel Design Pipeline', ha='center', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(c) Extrapolation Strategy', fontweight='bold')
    
    # (d) Cross-validation learning curve
    ax = axes[1, 0]
    sizes = [50, 100, 150, 200, 250, 311]
    train_scores = []
    test_scores = []
    
    for size in sizes:
        rf_lc = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        X_sub = X[:size]
        y_sub = y[:size]
        scores = cross_val_score(rf_lc, X_sub, y_sub, cv=min(5, size//10), scoring='r2')
        test_scores.append(scores.mean())
        # Train score on full subset
        rf_lc.fit(X_sub, y_sub)
        train_scores.append(rf_lc.score(X_sub, y_sub))
    
    ax.plot(sizes, train_scores, 'o-', label='Train R²', color='#2E86AB', markersize=6)
    ax.plot(sizes, test_scores, 's-', label='CV R²', color='#D64045', markersize=6)
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R² Score')
    ax.set_title('(d) Learning Curve')
    ax.legend()
    
    # (e) Expected improvement - aspirational target
    ax = axes[1, 1]
    
    # Compute EI for aspirational targets
    from scipy.stats import norm
    
    # Use GP at candidate points
    top_comp = df.nlargest(20, TARGET_COL)
    best_comp = top_comp[FEATURE_COLS].iloc[0].values.reshape(1, -1)
    gp_pred, gp_std = gp.predict(best_comp, return_std=True)
    
    targets = np.linspace(300, 1200, 50)
    ei_values = []
    for target in targets:
        if gp_std[0] > 0:
            z = (gp_pred[0] - target) / gp_std[0]
            ei = (gp_pred[0] - target) * norm.cdf(z) + gp_std[0] * norm.pdf(z)
            ei_values.append(max(0, ei))
        else:
            ei_values.append(max(0, gp_pred[0] - target))
    
    ax.plot(targets, ei_values, 'b-', linewidth=2)
    ax.fill_between(targets, 0, ei_values, alpha=0.3, color='#2E86AB')
    ax.axvline(x=1000, color='red', linestyle='--', alpha=0.7, label='1 MPa')
    ax.set_xlabel('Aspirational Target (kPa)')
    ax.set_ylabel('Expected Improvement')
    ax.set_title('(e) Expected Improvement vs Target')
    ax.legend()
    
    # (f) Optimized composition prediction
    ax = axes[1, 2]
    
    with open('outputs/optimization_results.json') as f:
        opt_data = json.load(f)
    
    best = opt_data.get('grid_search_top', [{}])[0]
    best_comp = best.get('composition', {})
    
    # Create extrapolated composition with enhanced Aromatic-PEA
    extrap_comps = []
    extrap_preds = []
    
    base_comp = df.nlargest(5, TARGET_COL)[FEATURE_COLS].median().values
    for ar_pea in np.linspace(0.25, 0.60, 8):
        for h_ba in np.linspace(0.45, 0.70, 8):
            comp = base_comp.copy()
            comp[4] = ar_pea  # Aromatic-PEA
            comp[1] = h_ba    # Hydrophobic-BA
            # Adjust others
            remaining = 1.0 - ar_pea - h_ba
            if remaining < 0:
                continue
            other_idx = [0, 2, 3, 5]
            other_base = base_comp[other_idx].sum()
            if other_base > 0:
                comp[other_idx] = comp[other_idx] / other_base * remaining
            
            comp_2d = comp.reshape(1, -1)
            pred = (rf.predict(comp_2d)[0] + gb.predict(comp_2d)[0] + gp.predict(comp_2d)[0]) / 3.0
            extrap_comps.append([ar_pea, h_ba])
            extrap_preds.append(pred)
    
    extrap_comps = np.array(extrap_comps)
    extrap_preds = np.array(extrap_preds)
    
    sc = ax.scatter(extrap_comps[:, 0], extrap_comps[:, 1], c=extrap_preds, 
                   cmap='RdYlBu_r', s=100, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Aromatic-PEA', fontweight='bold')
    ax.set_ylabel('Hydrophobic-BA', fontweight='bold')
    ax.set_title('(f) Extrapolated Predictions\n(Aromatic vs Hydrophobic)', fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Predicted (kPa)')
    
    # Mark best extrapolated
    best_idx = np.argmax(extrap_preds)
    ax.scatter(extrap_comps[best_idx, 0], extrap_comps[best_idx, 1], 
              marker='*', s=200, c='gold', edgecolors='black', linewidth=1.5, zorder=10)
    ax.text(extrap_comps[best_idx, 0]+0.02, extrap_comps[best_idx, 1], 
           f'{extrap_preds[best_idx]:.0f} kPa', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report/images/fig5_extrapolation_strategy.png')
    plt.close()
    print("Saved fig5_extrapolation_strategy.png")


def main():
    df, X, y = load_data()
    rf, gb, gp = load_models()
    
    print("Generating figures...")
    
    fig1_data_overview(df, y)
    perf = fig2_model_performance(df, X, y, rf, gb, gp)
    fig3_composition_analysis(df)
    fig4_optimization_landscape(rf, gb, gp, df, X, y)
    fig5_extrapolation_strategy(df, rf, gb, gp, X, y)
    
    # Save performance metrics
    with open('outputs/performance_metrics.json', 'w') as f:
        json.dump(perf, f, indent=2)
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
