#!/usr/bin/env python3
"""
Machine Learning Modeling: Train models, analyze feature importance, 
and perform Bayesian optimization for adhesive strength.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing optional packages
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not available")

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print("Warning: scikit-optimize not available")

# Setup
FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET_COL = 'Glass (kPa)_10s'

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def load_data():
    """Load the preprocessed dataset."""
    df = pd.read_csv('outputs/merged_dataset.csv')
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return df, X, y


def train_models(X, y):
    """Train multiple regression models and evaluate using cross-validation."""
    results = {}
    
    # KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Random Forest
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
    rf_mae = -cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    # Fit on full data
    rf.fit(X, y)
    results['RandomForest'] = {
        'model': rf,
        'cv_r2_mean': float(rf_scores.mean()),
        'cv_r2_std': float(rf_scores.std()),
        'cv_mae_mean': float(rf_mae.mean()),
        'cv_mae_std': float(rf_mae.std()),
    }
    
    # 2. Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_split=5, min_samples_leaf=2, random_state=42
    )
    gb_scores = cross_val_score(gb, X, y, cv=kf, scoring='r2')
    gb_mae = -cross_val_score(gb, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    gb.fit(X, y)
    results['GradientBoosting'] = {
        'model': gb,
        'cv_r2_mean': float(gb_scores.mean()),
        'cv_r2_std': float(gb_scores.std()),
        'cv_mae_mean': float(gb_mae.mean()),
        'cv_mae_std': float(gb_mae.std()),
    }
    
    # 3. Gaussian Process
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, 
        normalize_y=True, random_state=42, alpha=1e-3,
    )
    gp_scores = cross_val_score(gp, X, y, cv=kf, scoring='r2')
    gp_mae = -cross_val_score(gp, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    gp.fit(X, y)
    results['GaussianProcess'] = {
        'model': gp,
        'cv_r2_mean': float(gp_scores.mean()),
        'cv_r2_std': float(gp_scores.std()),
        'cv_mae_mean': float(gp_mae.mean()),
        'cv_mae_std': float(gp_mae.std()),
    }
    
    # Ensemble predictions
    # Train on subset for ensemble
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr)
    gp.fit(X_tr, y_tr)
    
    rf_pred = rf.predict(X_te)
    gb_pred = gb.predict(X_te)
    gp_pred = gp.predict(X_te)
    ensemble_pred = (rf_pred + gb_pred + gp_pred) / 3.0
    
    ensemble_r2 = r2_score(y_te, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_te, ensemble_pred)
    
    results['Ensemble'] = {
        'model': None,  # Ensemble doesn't have a single model
        'test_r2': float(ensemble_r2),
        'test_mae': float(ensemble_mae),
        'rf_component': rf,
        'gb_component': gb,
        'gp_component': gp,
    }
    
    return results


def compute_feature_importance(results, X, y, df):
    """Compute feature importance using multiple methods."""
    importance = {}
    
    # RF feature importance
    rf = results['RandomForest']['model']
    rf_imp = rf.feature_importances_
    importance['RandomForest_impurity'] = {c: float(v) for c, v in zip(FEATURE_COLS, rf_imp)}
    
    # Permutation importance
    perm_imp = permutation_importance(rf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    importance['Permutation'] = {c: float(v) for c, v in zip(FEATURE_COLS, perm_imp.importances_mean)}
    
    # GB feature importance
    gb = results['GradientBoosting']['model']
    gb_imp = gb.feature_importances_
    importance['GradientBoosting_impurity'] = {c: float(v) for c, v in zip(FEATURE_COLS, gb_imp)}
    
    # SHAP values for RF
    if HAS_SHAP:
        try:
            # Sample for efficiency
            n_samples = min(200, len(X))
            idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[idx]
            
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_sample)
            
            # Mean absolute SHAP
            mean_shap = np.abs(shap_values).mean(axis=0)
            importance['SHAP_RF'] = {c: float(v) for c, v in zip(FEATURE_COLS, mean_shap)}
            
            # Save SHAP values for later visualization
            np.save('outputs/shap_values.npy', shap_values)
            np.save('outputs/shap_X_sample.npy', X_sample)
        except Exception as e:
            print(f"SHAP computation failed: {e}")
    
    return importance


def bayesian_optimization(results, X_train, y_train):
    """Use Bayesian optimization to find compositions with high predicted adhesion."""
    if not HAS_SKOPT:
        print("scikit-optimize not available; skipping Bayesian optimization.")
        return None
    
    # Get best model (RF for exploitation, GP for uncertainty)
    rf = results['RandomForest']['model']
    gp = results['GaussianProcess']['model']
    gb = results['GradientBoosting']['model']
    
    # Define search space: each feature in [0, 1]
    # But we need sum = 1, so we use a simplex constraint via Dirichlet-based sampling
    space = [
        Real(0.0, 0.7, name='Nucleophilic-HEA'),
        Real(0.0, 0.8, name='Hydrophobic-BA'),
        Real(0.0, 0.5, name='Acidic-CBEA'),
        Real(0.0, 0.5, name='Cationic-ATAC'),
        Real(0.0, 0.7, name='Aromatic-PEA'),
        Real(0.0, 0.5, name='Amide-AAm'),
    ]
    
    def normalize_to_simplex(x):
        """Normalize a composition to sum to 1.0."""
        total = np.sum(x)
        if total > 0:
            return x / total
        else:
            return np.ones(6) / 6
    
    def objective_rf(x):
        """Objective for RF: maximize predicted adhesion."""
        x = np.array(x).reshape(1, -1)
        x_norm = normalize_to_simplex(x)
        pred = rf.predict(x_norm)[0]
        return -pred  # Minimize negative = maximize
    
    def objective_gp(x):
        """Objective for GP: maximize predicted adhesion."""
        x = np.array(x).reshape(1, -1)
        x_norm = normalize_to_simplex(x)
        pred = gp.predict(x_norm)[0]
        return -pred
    
    def objective_gb(x):
        """Objective for GB: maximize predicted adhesion."""
        x = np.array(x).reshape(1, -1)
        x_norm = normalize_to_simplex(x)
        pred = gb.predict(x_norm)[0]
        return -pred
    
    def objective_ei(x):
        """Expected improvement objective using GP."""
        x = np.array(x).reshape(1, -1)
        x_norm = normalize_to_simplex(x)
        pred, std = gp.predict(x_norm, return_std=True)
        
        # Expected improvement with target = max(y_train)
        y_best = np.max(y_train)
        if std[0] > 0:
            z = (pred[0] - y_best) / std[0]
            from scipy.stats import norm
            ei = (pred[0] - y_best) * norm.cdf(z) + std[0] * norm.pdf(z)
        else:
            ei = max(0, pred[0] - y_best)
        
        return -ei
    
    # For EI, use a higher target (aspiration beyond current max)
    def objective_ei_aspirational(x, target=500.0):
        """Expected improvement with aspirational target."""
        x = np.array(x).reshape(1, -1)
        x_norm = normalize_to_simplex(x)
        pred, std = gp.predict(x_norm, return_std=True)
        
        y_target = target
        if std[0] > 0:
            z = (pred[0] - y_target) / std[0]
            from scipy.stats import norm
            ei = (pred[0] - y_target) * norm.cdf(z) + std[0] * norm.pdf(z)
        else:
            ei = max(0, pred[0] - y_target)
        
        return -ei
    
    opt_results = {}
    
    # Run optimization with different strategies
    strategies = [
        ('RF_max', objective_rf, 'RF exploitation'),
        ('GP_max', objective_gp, 'GP exploitation'),
        ('GB_max', objective_gb, 'GB exploitation'),
        ('GP_EI_max', objective_ei, 'GP EI (current max)'),
        ('GP_EI_500', objective_ei_aspirational, 'GP EI (target=500 kPa)'),
    ]
    
    for name, obj_func, desc in strategies:
        print(f"\nRunning Bayesian optimization: {desc}...")
        try:
            result = gp_minimize(
                obj_func, space, n_calls=100, n_random_starts=20,
                random_state=42, verbose=False,
            )
            
            # Get top 10 compositions
            top_indices = np.argsort(result.func_vals)[:10]  # Lowest (most negative) = best
            top_compositions = []
            for idx in top_indices:
                comp = normalize_to_simplex(result.x_iters[idx])
                # Predict with all three models
                rf_pred = rf.predict(comp.reshape(1, -1))[0]
                gb_pred = gb.predict(comp.reshape(1, -1))[0]
                gp_pred, gp_std = gp.predict(comp.reshape(1, -1), return_std=True)
                gp_pred = gp_pred[0]
                gp_std = gp_std[0]
                ensemble_pred = (rf_pred + gb_pred + gp_pred) / 3.0
                
                top_compositions.append({
                    'composition': {c: float(v) for c, v in zip(FEATURE_COLS, comp)},
                    'rf_pred': float(rf_pred),
                    'gb_pred': float(gb_pred),
                    'gp_pred': float(gp_pred),
                    'gp_std': float(gp_std),
                    'ensemble_pred': float(ensemble_pred),
                    'obj_value': float(-result.func_vals[idx]),  # Convert back to positive
                })
            
            opt_results[name] = {
                'description': desc,
                'best_obj': float(-result.fun),
                'top_compositions': top_compositions,
            }
        except Exception as e:
            print(f"  Optimization failed: {e}")
    
    return opt_results


def generate_figures(results, importance, opt_results, df, X, y):
    """Generate all figures for the report."""
    
    # Figure 1: Data overview - distribution of adhesive strengths
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1a: Histogram
    axes[0].hist(y, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(y=1000, color='red', linestyle='--', linewidth=2, label='Target: 1 MPa')
    axes[0].axvline(y=np.max(y), color='darkgreen', linestyle='--', linewidth=1.5, 
                    label=f'Current max: {np.max(y):.0f} kPa')
    axes[0].set_xlabel('Adhesive Strength (kPa)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Adhesive Strength')
    axes[0].legend(fontsize=9)
    
    # 1b: Feature distributions
    box_data = [df[c].values for c in FEATURE_COLS]
    labels = [c.replace('-', '-\n') for c in FEATURE_COLS]
    bp = axes[1].boxplot(box_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[1].set_ylabel('Mole Fraction')
    axes[1].set_title('Monomer Composition Distribution')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 1c: Correlation heatmap
    corr_cols = FEATURE_COLS + [TARGET_COL]
    corr_matrix = df[corr_cols].corr()
    im = axes[2].imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[2].set_xticks(range(len(corr_cols)))
    axes[2].set_yticks(range(len(corr_cols)))
    axes[2].set_xticklabels([c[:15] for c in corr_cols], rotation=45, ha='right', fontsize=9)
    axes[2].set_yticklabels([c[:15] for c in corr_cols], fontsize=9)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            axes[2].text(j, i, f'{corr_matrix.values[i, j]:.2f}', 
                        ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    axes[2].set_title('Feature-Target Correlations')
    
    plt.tight_layout()
    fig.savefig('report/images/fig1_data_overview.png')
    plt.close()
    print("Saved fig1_data_overview.png")
    
    # Figure 2: Model performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = ['RandomForest', 'GradientBoosting', 'GaussianProcess']
    r2_means = [results[m]['cv_r2_mean'] for m in model_names]
    r2_stds = [results[m]['cv_r2_std'] for m in model_names]
    mae_means = [results[m]['cv_mae_mean'] for m in model_names]
    mae_stds = [results[m]['cv_mae_std'] for m in model_names]
    
    x_pos = np.arange(len(model_names))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 2a: R² comparison
    bars = axes[0].bar(x_pos, r2_means, yerr=r2_stds, capsize=5, color=colors, edgecolor='white')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names, rotation=15, fontsize=10)
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('5-Fold CV R² Scores')
    axes[0].set_ylim(0, 1)
    
    # 2b: MAE comparison
    axes[1].bar(x_pos, mae_means, yerr=mae_stds, capsize=5, color=colors, edgecolor='white')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names, rotation=15, fontsize=10)
    axes[1].set_ylabel('MAE (kPa)')
    axes[1].set_title('5-Fold CV MAE')
    
    # 2c: Parity plot
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = results['RandomForest']['model']
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    
    axes[2].scatter(y_te, y_pred_rf, alpha=0.6, color='#2E86AB', edgecolors='white')
    axes[2].plot([0, max(y_te)], [0, max(y_te)], 'k--', linewidth=1)
    axes[2].set_xlabel('Actual Adhesive Strength (kPa)')
    axes[2].set_ylabel('Predicted Adhesive Strength (kPa)')
    r2_val = r2_score(y_te, y_pred_rf)
    axes[2].set_title(f'RF Parity Plot (R² = {r2_val:.3f})')
    
    plt.tight_layout()
    fig.savefig('report/images/fig2_model_performance.png')
    plt.close()
    print("Saved fig2_model_performance.png")
    
    # Figure 3: Feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 3a: RF impurity importance
    rf_imp = importance['RandomForest_impurity']
    imp_df = pd.DataFrame({
        'Feature': list(rf_imp.keys()),
        'Importance': list(rf_imp.values())
    }).sort_values('Importance', ascending=True)
    
    colors_imp = ['#2E86AB' if v > 0 else '#A23B72' for v in imp_df['Importance']]
    axes[0].barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp, edgecolor='white')
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title('Random Forest Feature Importance')
    
    # 3b: Permutation importance
    perm_imp = importance['Permutation']
    perm_df = pd.DataFrame({
        'Feature': list(perm_imp.keys()),
        'Importance': list(perm_imp.values())
    }).sort_values('Importance', ascending=True)
    
    axes[1].barh(perm_df['Feature'], perm_df['Importance'], color='#F18F01', edgecolor='white')
    axes[1].set_xlabel('Importance (R² drop)')
    axes[1].set_title('Permutation Feature Importance')
    
    plt.tight_layout()
    fig.savefig('report/images/fig3_feature_importance.png')
    plt.close()
    print("Saved fig3_feature_importance.png")
    
    # Figure 4: SHAP analysis (if available)
    if HAS_SHAP and os.path.exists('outputs/shap_values.npy'):
        shap_values = np.load('outputs/shap_values.npy')
        X_sample = np.load('outputs/shap_X_sample.npy')
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # 4a: SHAP summary
        # Simplified SHAP summary
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_order = np.argsort(mean_abs_shap)
        
        for i, feat_idx in enumerate(shap_order):
            axes[0].scatter(
                shap_values[:, feat_idx], 
                X_sample[:, feat_idx],
                alpha=0.5, s=20, label=FEATURE_COLS[feat_idx],
            )
        axes[0].set_xlabel('SHAP Value')
        axes[0].set_ylabel('Feature Value')
        axes[0].set_title('SHAP Value Distribution')
        axes[0].legend(fontsize=8, loc='upper right')
        
        # 4b: SHAP bar
        shap_bar_df = pd.DataFrame({
            'Feature': [FEATURE_COLS[i] for i in shap_order],
            'mean(|SHAP|)': mean_abs_shap[shap_order]
        })
        axes[1].barh(shap_bar_df['Feature'], shap_bar_df['mean(|SHAP|)'], 
                    color='#A23B72', edgecolor='white')
        axes[1].set_xlabel('mean(|SHAP value|)')
        axes[1].set_title('SHAP Feature Importance')
        
        plt.tight_layout()
        fig.savefig('report/images/fig4_shap_analysis.png')
        plt.close()
        print("Saved fig4_shap_analysis.png")
    
    # Figure 5: Composition-Adhesion landscape
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (feat, ax) in enumerate(zip(FEATURE_COLS, axes.flat)):
        # Scatter with color by another key feature
        if i < 5:
            other_feat = FEATURE_COLS[i + 1]
        else:
            other_feat = FEATURE_COLS[0]
        
        sc = ax.scatter(df[feat], df[TARGET_COL], c=df[other_feat], 
                       cmap='viridis', alpha=0.6, edgecolors='white', s=40)
        ax.set_xlabel(feat)
        ax.set_ylabel('Adhesive Strength (kPa)')
        ax.set_title(f'{feat} vs Adhesion')
        
        # Add trend line
        z = np.polyfit(df[feat], y, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(df[feat])
        ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=1.5, alpha=0.7)
        
        corr = df[feat].corr(df[TARGET_COL])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top')
        
        plt.colorbar(sc, ax=ax, label=other_feat[:15])
    
    plt.tight_layout()
    fig.savefig('report/images/fig5_composition_adhesion.png')
    plt.close()
    print("Saved fig5_composition_adhesion.png")
    
    # Figure 6: Optimization results
    if opt_results:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for idx, (name, opt) in enumerate(opt_results.items()):
            if idx >= 6:
                break
            ax = axes.flat[idx]
            
            top = opt['top_compositions']
            labels = [f"C{i+1}" for i in range(min(5, len(top)))]
            values = [t['ensemble_pred'] for t in top[:5]]
            rf_vals = [t['rf_pred'] for t in top[:5]]
            gp_vals = [t['gp_pred'] for t in top[:5]]
            
            x = np.arange(len(labels))
            width = 0.25
            
            ax.bar(x - width, values, width, label='Ensemble', color='#2E86AB', edgecolor='white')
            ax.bar(x, rf_vals, width, label='RF', color='#A23B72', edgecolor='white')
            ax.bar(x + width, gp_vals, width, label='GP', color='#F18F01', edgecolor='white')
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Predicted Adhesion (kPa)')
            ax.set_title(opt['description'])
            ax.axhline(y=321, color='red', linestyle='--', alpha=0.5, label='Current Max')
            ax.legend(fontsize=8)
        
        # Last subplot: Composition bar of best candidate
        if len(opt_results) > 0:
            best_strategy = max(opt_results.items(), 
                              key=lambda x: x[1]['top_compositions'][0]['ensemble_pred'])
            best_comp = best_strategy[1]['top_compositions'][0]['composition']
            
            ax = axes.flat[-1]
            comp_vals = [best_comp[c] for c in FEATURE_COLS]
            colors_comp = ['#2E86AB', '#A23B72', '#F18F01', '#4CB944', '#D64045', '#8B80F9']
            bars = ax.bar(FEATURE_COLS, comp_vals, color=colors_comp, edgecolor='white')
            ax.set_ylabel('Mole Fraction')
            ax.set_title(f'Best Composition (Pred: {best_strategy[1]["top_compositions"][0]["ensemble_pred"]:.0f} kPa)')
            ax.tick_params(axis='x', rotation=45)
            
            # Save best composition
            best_info = {
                'strategy': best_strategy[0],
                'composition': best_comp,
                'predicted_adhesion_kPa': best_strategy[1]['top_compositions'][0]['ensemble_pred'],
                'rf_pred': best_strategy[1]['top_compositions'][0]['rf_pred'],
                'gp_pred': best_strategy[1]['top_compositions'][0]['gp_pred'],
                'gp_std': best_strategy[1]['top_compositions'][0]['gp_std'],
            }
            with open('outputs/best_composition.json', 'w') as f:
                json.dump(best_info, f, indent=2)
        
        plt.tight_layout()
        fig.savefig('report/images/fig6_optimization.png')
        plt.close()
        print("Saved fig6_optimization.png")
    
    # Figure 7: Optimization trajectory (pairplot of top candidates vs training)
    if opt_results:
        # Collect all top compositions across strategies
        all_tops = []
        for name, opt in opt_results.items():
            for comp in opt['top_compositions'][:3]:
                all_tops.append({
                    **comp['composition'],
                    'Predicted': comp['ensemble_pred'],
                    'Strategy': name
                })
        
        if all_tops:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            for i, feat in enumerate(FEATURE_COLS):
                ax = axes.flat[i]
                # Training data
                ax.scatter(df[feat], y, alpha=0.3, c='gray', s=20, label='Training')
                # Top candidates
                for t in all_tops:
                    ax.scatter(t[feat], t['Predicted'], s=100, marker='*', 
                             edgecolors='black', linewidth=1, zorder=5,
                             label=f'Candidate' if i == 0 else '')
                ax.set_xlabel(feat)
                ax.set_ylabel('Adhesion (kPa)')
                ax.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='1 MPa')
                if i == 0:
                    ax.legend(fontsize=7)
            
            plt.tight_layout()
            fig.savefig('report/images/fig7_optimization_trajectory.png')
            plt.close()
            print("Saved fig7_optimization_trajectory.png")


def main():
    print("=" * 60)
    print("Hydrogel Adhesive Strength ML Analysis")
    print("=" * 60)
    
    # Load data
    df, X, y = load_data()
    print(f"\nLoaded {len(X)} samples with {X.shape[1]} features")
    
    # Train models
    print("\n--- Training Models ---")
    results = train_models(X, y)
    
    # Print results
    print("\nModel Performance (5-fold CV):")
    print("-" * 60)
    for name, res in results.items():
        if 'cv_r2_mean' in res:
            print(f"  {name:20s}: R² = {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}, "
                  f"MAE = {res['cv_mae_mean']:.2f} ± {res['cv_mae_std']:.2f} kPa")
        elif 'test_r2' in res:
            print(f"  {name:20s}: Test R² = {res['test_r2']:.4f}, Test MAE = {res['test_mae']:.2f} kPa")
    
    # Save model results
    model_summary = {}
    for name, res in results.items():
        model_summary[name] = {k: v for k, v in res.items() if k != 'model' 
                               and 'component' not in k}
    with open('outputs/model_results.json', 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    # Feature importance
    print("\n--- Feature Importance ---")
    importance = compute_feature_importance(results, X, y, df)
    
    for method, imp in importance.items():
        print(f"\n  {method}:")
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        for feat, val in sorted_imp:
            print(f"    {feat:20s}: {val:.4f}")
    
    with open('outputs/feature_importance.json', 'w') as f:
        json.dump(importance, f, indent=2)
    
    # Bayesian optimization
    print("\n--- Bayesian Optimization ---")
    opt_results = bayesian_optimization(results, X, y)
    
    if opt_results:
        with open('outputs/optimization_results.json', 'w') as f:
            # Convert to serializable
            serializable = {}
            for name, opt in opt_results.items():
                serializable[name] = {
                    'description': opt['description'],
                    'best_obj': opt['best_obj'],
                    'top_compositions': opt['top_compositions'][:5],
                }
            json.dump(serializable, f, indent=2)
        
        # Print top results
        print("\nTop candidates from each strategy:")
        for name, opt in opt_results.items():
            if opt['top_compositions']:
                top = opt['top_compositions'][0]
                print(f"\n  {name} ({opt['description']}):")
                print(f"    Predicted adhesion: {top['ensemble_pred']:.1f} kPa")
                print(f"    Composition: {top['composition']}")
    
    # Generate figures
    print("\n--- Generating Figures ---")
    generate_figures(results, importance, opt_results, df, X, y)
    
    print("\nDone! All outputs saved.")
    return results, importance, opt_results


if __name__ == '__main__':
    results, importance, opt_results = main()
