#!/usr/bin/env python3
"""
Data Preparation: Load, merge, and preprocess all hydrogel datasets.
"""

import pandas as pd
import numpy as np
import json
import os

# Feature columns (monomer compositions)
FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET_COL = 'Glass (kPa)_10s'

def load_verified_dataset(path):
    """Load the verified 184 dataset."""
    df = pd.read_excel(path)
    df = df[FEATURE_COLS + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL])
    df['source'] = 'verified_184'
    return df

def load_original_datasets(paths):
    """Load the three original training datasets."""
    dfs = []
    for i, path in enumerate(paths):
        df = pd.read_excel(path)
        # Different files have different target column names
        if 'Glass (kPa)_10s' in df.columns:
            target = 'Glass (kPa)_10s'
        elif 'Glass (kPa)' in df.columns:
            target = 'Glass (kPa)'
        else:
            continue
        
        available_features = [c for c in FEATURE_COLS if c in df.columns]
        if len(available_features) < 6:
            continue
        
        sub = df[available_features + [target]].copy()
        sub = sub.dropna(subset=[target])
        sub = sub.rename(columns={target: TARGET_COL})
        
        # Add missing feature columns with zeros
        for c in FEATURE_COLS:
            if c not in sub.columns:
                sub[c] = 0.0
        sub = sub[FEATURE_COLS + [TARGET_COL]]
        
        sub['source'] = f'batch_{i+1}'
        dfs.append(sub)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_optimization_datasets(paths):
    """Load the final optimization datasets."""
    dfs = []
    for path in paths:
        df = pd.read_excel(path)
        if 'Glass (kPa)_max' in df.columns:
            target = 'Glass (kPa)_max'
        else:
            continue
        
        available = [c for c in FEATURE_COLS if c in df.columns]
        sub = df[available + [target]].copy()
        
        # Convert all columns to numeric, coercing errors (handles 'NO GELATION', text, etc.)
        for c in available:
            if sub[c].dtype == 'object':
                sub[c] = pd.to_numeric(sub[c], errors='coerce')
        if sub[target].dtype == 'object':
            sub[target] = pd.to_numeric(sub[target], errors='coerce')
        
        sub = sub.dropna(subset=available + [target])
        sub = sub.rename(columns={target: TARGET_COL})
        sub['source'] = 'optimization'
        dfs.append(sub)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    # Load verified dataset
    verified = load_verified_dataset('data/184_verified_Original Data_ML_20230926.xlsx')
    print(f"Verified dataset: {len(verified)} samples")
    
    # Load original batches
    batch_paths = [
        'data/Original Data_ML_20220829.xlsx',
        'data/Original Data_ML_20221031.xlsx',
        'data/Original Data_ML_20221129.xlsx',
    ]
    original = load_original_datasets(batch_paths)
    print(f"Original batches: {len(original)} samples")
    
    # Load optimization datasets
    opt_paths = [
        'data/ML_ei&pred (1&2&3rounds)_20240408.xlsx',
        'data/ML_ei&pred_20240213.xlsx',
    ]
    optimization = load_optimization_datasets(opt_paths)
    print(f"Optimization datasets: {len(optimization)} samples")
    
    # Merge all data - use verified as base, add unique from others
    all_data = verified.copy()
    
    # Add unique samples from original batches
    if len(original) > 0:
        # Check which compositions are new
        original_rounded = original[FEATURE_COLS].round(4)
        verified_rounded = verified[FEATURE_COLS].round(4)
        
        merged_idx = original_rounded.merge(
            verified_rounded, on=FEATURE_COLS, how='left', indicator=True
        )
        new_mask = merged_idx['_merge'] == 'left_only'
        new_original = original[new_mask.values].copy()
        print(f"New samples from original batches: {len(new_original)}")
        all_data = pd.concat([all_data, new_original], ignore_index=True)
    
    # Add unique samples from optimization
    if len(optimization) > 0:
        opt_rounded = optimization[FEATURE_COLS].round(4)
        all_rounded = all_data[FEATURE_COLS].round(4)
        
        merged_idx = opt_rounded.merge(
            all_rounded, on=FEATURE_COLS, how='left', indicator=True
        )
        new_mask = merged_idx['_merge'] == 'left_only'
        new_opt = optimization[new_mask.values].copy()
        print(f"New samples from optimization: {len(new_opt)}")
        all_data = pd.concat([all_data, new_opt], ignore_index=True)
    
    # Remove duplicates based on composition
    all_data = all_data.drop_duplicates(subset=FEATURE_COLS, keep='first')
    
    # Drop NaN targets
    all_data = all_data.dropna(subset=[TARGET_COL])
    
    print(f"\nFinal merged dataset: {len(all_data)} samples")
    print(f"Target range: {all_data[TARGET_COL].min():.2f} - {all_data[TARGET_COL].max():.2f} kPa")
    print(f"Mean: {all_data[TARGET_COL].mean():.2f} kPa")
    print(f"Std: {all_data[TARGET_COL].std():.2f} kPa")
    print(f"Samples > 200 kPa: {(all_data[TARGET_COL] > 200).sum()}")
    print(f"Samples > 250 kPa: {(all_data[TARGET_COL] > 250).sum()}")
    print(f"Samples > 300 kPa: {(all_data[TARGET_COL] > 300).sum()}")
    
    # Feature stats
    print("\nFeature ranges:")
    for c in FEATURE_COLS:
        print(f"  {c}: [{all_data[c].min():.3f}, {all_data[c].max():.3f}] mean={all_data[c].mean():.3f}")
    
    # Save merged dataset
    os.makedirs('outputs', exist_ok=True)
    all_data.to_csv('outputs/merged_dataset.csv', index=False)
    
    # Save summary stats
    stats = {
        'n_samples': int(len(all_data)),
        'target_min': float(all_data[TARGET_COL].min()),
        'target_max': float(all_data[TARGET_COL].max()),
        'target_mean': float(all_data[TARGET_COL].mean()),
        'target_std': float(all_data[TARGET_COL].std()),
        'feature_stats': {},
        'correlations': {}
    }
    
    for c in FEATURE_COLS:
        stats['feature_stats'][c] = {
            'min': float(all_data[c].min()),
            'max': float(all_data[c].max()),
            'mean': float(all_data[c].mean()),
            'std': float(all_data[c].std()),
        }
        stats['correlations'][c] = float(all_data[c].corr(all_data[TARGET_COL]))
    
    with open('outputs/data_summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return all_data


if __name__ == '__main__':
    df = main()
