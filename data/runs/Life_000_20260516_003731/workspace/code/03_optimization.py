#!/usr/bin/env python3
"""
Bayesian Optimization for Hydrogel Composition.
Searches for monomer compositions that maximize predicted adhesive strength.
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from skopt import gp_minimize
from skopt.space import Real

FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET_COL = 'Glass (kPa)_10s'


def load_models():
    """Load trained models."""
    with open('outputs/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('outputs/gb_model.pkl', 'rb') as f:
        gb = pickle.load(f)
    with open('outputs/gp_model.pkl', 'rb') as f:
        gp = pickle.load(f)
    return rf, gb, gp


def normalize_to_simplex(x):
    """Normalize composition to sum to 1.0."""
    x = np.asarray(x).flatten()
    total = np.sum(x)
    if total > 0:
        return x / total
    return np.ones(6) / 6


def grid_search(rf, gb, gp, n_points=50000):
    """Search a large grid of compositions for optimal adhesion."""
    print(f"Running grid search with {n_points} random compositions...")
    
    # Sample from Dirichlet distribution for diverse compositions
    np.random.seed(42)
    alpha = np.ones(6)
    compositions = np.random.dirichlet(alpha, size=n_points)
    
    # Predict with all models
    rf_pred = rf.predict(compositions)
    gb_pred = gb.predict(compositions)
    gp_pred = gp.predict(compositions)
    ensemble_pred = (rf_pred + gb_pred + gp_pred) / 3.0
    
    # Find top candidates
    top_k = min(20, n_points)
    top_indices = np.argsort(ensemble_pred)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'composition': {c: float(v) for c, v in zip(FEATURE_COLS, compositions[idx])},
            'rf_pred': float(rf_pred[idx]),
            'gb_pred': float(gb_pred[idx]),
            'gp_pred': float(gp_pred[idx]),
            'ensemble_pred': float(ensemble_pred[idx]),
        })
    
    # Also predict at known high-adhesion points from training
    df = pd.read_csv('outputs/merged_dataset.csv')
    top_real = df.nlargest(10, TARGET_COL)
    
    print(f"\nTop 10 candidates from grid search:")
    for i, r in enumerate(results[:10]):
        comp = r['composition']
        print(f"  {i+1}. Pred={r['ensemble_pred']:.1f} kPa | "
              f"N-HEA={comp['Nucleophilic-HEA']:.3f} "
              f"H-BA={comp['Hydrophobic-BA']:.3f} "
              f"A-CBEA={comp['Acidic-CBEA']:.3f} "
              f"C-ATAC={comp['Cationic-ATAC']:.3f} "
              f"Ar-PEA={comp['Aromatic-PEA']:.3f} "
              f"Am-AAm={comp['Amide-AAm']:.3f}")
    
    print(f"\nTop 5 real experimental values:")
    for _, row in top_real.head(5).iterrows():
        print(f"  Actual={row[TARGET_COL]:.1f} kPa | "
              f"N-HEA={row['Nucleophilic-HEA']:.3f} "
              f"H-BA={row['Hydrophobic-BA']:.3f} "
              f"A-CBEA={row['Acidic-CBEA']:.3f} "
              f"C-ATAC={row['Cationic-ATAC']:.3f} "
              f"Ar-PEA={row['Aromatic-PEA']:.3f} "
              f"Am-AAm={row['Amide-AAm']:.3f}")
    
    return results


def skopt_optimization(rf, gb, gp):
    """Run scikit-optimize Bayesian optimization."""
    print("\nRunning Bayesian optimization with GP...")
    
    space = [
        Real(0.0, 0.7, name='Nucleophilic-HEA'),
        Real(0.0, 0.8, name='Hydrophobic-BA'),
        Real(0.0, 0.5, name='Acidic-CBEA'),
        Real(0.0, 0.5, name='Cationic-ATAC'),
        Real(0.0, 0.7, name='Aromatic-PEA'),
        Real(0.0, 0.5, name='Amide-AAm'),
    ]
    
    def objective(x):
        comp = normalize_to_simplex(np.array(x).reshape(1, -1)).reshape(1, -1)
        rf_pred = rf.predict(comp)[0]
        gb_pred = gb.predict(comp)[0]
        gp_pred = gp.predict(comp)[0]
        ensemble = (rf_pred + gb_pred + gp_pred) / 3.0
        return -ensemble  # minimize negative = maximize
    
    result = gp_minimize(
        objective, space, n_calls=50, n_random_starts=10,
        random_state=42, verbose=False,
    )
    
    # Extract top results
    top_indices = np.argsort(result.func_vals)[:10]
    opt_results = []
    for idx in top_indices:
        comp = normalize_to_simplex(np.array(result.x_iters[idx]).reshape(1, -1)).reshape(1, -1)
        rf_pred = rf.predict(comp)[0]
        gb_pred = gb.predict(comp)[0]
        gp_pred = gp.predict(comp)[0]
        ensemble = (rf_pred + gb_pred + gp_pred) / 3.0
        
        opt_results.append({
            'composition': {c: float(v) for c, v in zip(FEATURE_COLS, comp.flatten())},
            'rf_pred': float(rf_pred),
            'gb_pred': float(gb_pred),
            'gp_pred': float(gp_pred),
            'ensemble_pred': float(ensemble),
        })
    
    print(f"\nTop 5 from Bayesian Optimization:")
    for i, r in enumerate(opt_results[:5]):
        comp = r['composition']
        print(f"  {i+1}. Pred={r['ensemble_pred']:.1f} kPa | "
              f"N-HEA={comp['Nucleophilic-HEA']:.3f} "
              f"H-BA={comp['Hydrophobic-BA']:.3f} "
              f"A-CBEA={comp['Acidic-CBEA']:.3f} "
              f"C-ATAC={comp['Cationic-ATAC']:.3f} "
              f"Ar-PEA={comp['Aromatic-PEA']:.3f} "
              f"Am-AAm={comp['Amide-AAm']:.3f}")
    
    return opt_results


def systematic_variation(rf, gb, gp):
    """Systematically vary each feature to understand sensitivity."""
    print("\nSystematic feature variation analysis...")
    
    # Start from best grid search composition
    df = pd.read_csv('outputs/merged_dataset.csv')
    best_idx = df[TARGET_COL].idxmax()
    base_comp = df.loc[best_idx, FEATURE_COLS].values.copy()
    
    sensitivity = {}
    n_steps = 20
    
    for i, feat in enumerate(FEATURE_COLS):
        variations = []
        for step in range(n_steps + 1):
            comp = base_comp.copy()
            # Vary this feature while keeping others proportional
            new_val = step / n_steps
            old_val = comp[i]
            comp[i] = new_val
            # Renormalize
            comp = comp / comp.sum()
            
            rf_pred = rf.predict(comp.reshape(1, -1))[0]
            gb_pred = gb.predict(comp.reshape(1, -1))[0]
            gp_pred = gp.predict(comp.reshape(1, -1))[0]
            ensemble = (rf_pred + gb_pred + gp_pred) / 3.0
            
            variations.append({
                'feature_value': float(new_val),
                'actual_fraction': float(comp[i]),
                'ensemble_pred': float(ensemble),
                'rf_pred': float(rf_pred),
                'gb_pred': float(gb_pred),
                'gp_pred': float(gp_pred),
            })
        
        sensitivity[feat] = variations
    
    return sensitivity


def main():
    rf, gb, gp = load_models()
    print("Models loaded.")
    
    # Grid search
    grid_results = grid_search(rf, gb, gp, n_points=50000)
    
    # Bayesian optimization
    opt_results = skopt_optimization(rf, gb, gp)
    
    # Systematic variation
    sensitivity = systematic_variation(rf, gb, gp)
    
    # Combine and save
    all_results = {
        'grid_search_top': grid_results,
        'bayesian_opt_top': opt_results,
        'sensitivity': sensitivity,
    }
    
    with open('outputs/optimization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save best composition
    best = max(grid_results + opt_results, key=lambda x: x['ensemble_pred'])
    with open('outputs/best_composition.json', 'w') as f:
        json.dump(best, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"BEST COMPOSITION:")
    print(f"  Predicted Adhesion: {best['ensemble_pred']:.1f} kPa ({best['ensemble_pred']/1000:.3f} MPa)")
    comp = best['composition']
    for feat in FEATURE_COLS:
        print(f"  {feat}: {comp[feat]:.4f}")
    print(f"  RF pred: {best['rf_pred']:.1f}")
    print(f"  GB pred: {best['gb_pred']:.1f}")
    print(f"  GP pred: {best['gp_pred']:.1f}")
    
    return all_results


if __name__ == '__main__':
    results = main()
