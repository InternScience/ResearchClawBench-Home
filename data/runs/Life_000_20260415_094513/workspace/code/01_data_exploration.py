"""
Phase 1: Data Exploration & Merging
Load all datasets, understand structure, merge, and save consolidated data.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# 1. Load all datasets
# ============================================================
# Batch 1-3: Initial training data
batch1 = pd.read_excel('data/Original Data_ML_20220829.xlsx', sheet_name='Sheet2')
batch2 = pd.read_excel('data/Original Data_ML_20221031.xlsx', sheet_name='Data_to_HU')
batch3 = pd.read_excel('data/Original Data_ML_20221129.xlsx', sheet_name='Data_to_HU')

# Verified dataset
verified = pd.read_excel('data/184_verified_Original Data_ML_20230926.xlsx', sheet_name='Data_to_HU')

# Optimization datasets
opt1 = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='EI')
opt1_pred = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='PRED')
opt2 = pd.read_excel('data/ML_ei&pred_20240213.xlsx', sheet_name='EI')
opt2_pred = pd.read_excel('data/ML_ei&pred_20240213.xlsx', sheet_name='PRED')

# ============================================================
# 2. Summarize datasets
# ============================================================
summary = {}
summary['batch1_shape'] = list(batch1.shape)
summary['batch2_shape'] = list(batch2.shape)
summary['batch3_shape'] = list(batch3.shape)
summary['verified_shape'] = list(verified.shape)
summary['opt1_ei_shape'] = list(opt1.shape)
summary['opt1_pred_shape'] = list(opt1_pred.shape)
summary['opt2_ei_shape'] = list(opt2.shape)
summary['opt2_pred_shape'] = list(opt2_pred.shape)

# Monomer features
monomer_cols = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']

# ============================================================
# 3. Use verified dataset as primary training data
# ============================================================
# Target: Glass (kPa)_10s or Glass (kPa)_60s
print("Verified columns:", list(verified.columns))
print("\nVerified head:")
print(verified.head())

# Check target columns
target_col = 'Glass (kPa)_10s'
if target_col not in verified.columns:
    target_col = 'Glass (kPa)'
print(f"\nUsing target column: {target_col}")

# Clean data - remove rows with NaN target
df_train = verified[monomer_cols + [target_col, 'No.']].copy()
df_train = df_train.dropna(subset=[target_col])
print(f"\nTraining data shape after dropping NaN target: {df_train.shape}")
print(f"Target statistics: mean={df_train[target_col].mean():.2f}, std={df_train[target_col].std():.2f}")
print(f"Target min={df_train[target_col].min():.2f}, max={df_train[target_col].max():.2f}")

# ============================================================
# 4. Analyze optimization datasets
# ============================================================
# Forward fill ML column
opt1['ML'] = opt1['ML'].ffill()
opt1_pred['ML'] = opt1_pred['ML'].ffill()
opt2['ML'] = opt2['ML'].ffill()
opt2_pred['ML'] = opt2_pred['ML'].ffill()

print("\n=== Optimization Dataset 1 (3 rounds) ===")
print("EI sheet ML methods:", opt1['ML'].unique())
print("PRED sheet ML methods:", opt1_pred['ML'].unique())

print("\n=== Optimization Dataset 2 ===")
print("EI sheet ML methods:", opt2['ML'].unique())
print("PRED sheet ML methods:", opt2_pred['ML'].unique())

# ============================================================
# 5. Save consolidated data
# ============================================================
df_train.to_csv('outputs/training_data_184.csv', index=False)
opt1.to_csv('outputs/opt1_ei_3rounds.csv', index=False)
opt1_pred.to_csv('outputs/opt1_pred_3rounds.csv', index=False)

# Save summary
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nPhase 1 complete. Data saved to outputs/")
