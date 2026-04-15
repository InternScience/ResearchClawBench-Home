import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import json

# Load data
df = pd.read_csv('data/Together_1_targets_inserted.csv')

X = df.drop(columns=['Unnamed: 0', 'Feature_1', 'Feature_2', 'Attack', 'Sniffing'])
y_attack = df['Attack']
y_sniff = df['Sniffing']

# Data overview plots (as above)
# ... (copy the plot code here for reproducibility)

print('Data exploration complete.')