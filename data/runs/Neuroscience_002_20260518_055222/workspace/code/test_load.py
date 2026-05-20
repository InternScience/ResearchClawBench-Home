"""
Lightweight analysis
"""
import pandas as pd
import numpy as np
import json

# Load data
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')
print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Labels: {dict(train_df['label'].value_counts())}")
