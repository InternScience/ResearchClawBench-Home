"""
Simplified analysis script for hydrogel adhesive strength data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
df = pd.read_excel('data/184_verified_Original Data_ML_20230926.xlsx')
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Define features
MONOMERS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
            'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
OUTPUTS = ['Glass (kPa)_10s', 'Glass (kPa)_60s', 'Steel (kPa)_10s', 'Steel (kPa)_60s']

# Convert to numeric
for col in df.columns:
    if col not in ['No.', 'Tanδ', 'Log_Slope']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nData loaded successfully")
print(f"Valid samples for Glass_60s: {df['Glass (kPa)_60s'].notna().sum()}")
