import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
df = pd.read_excel('data/184_verified_Original Data_ML_20230926.xlsx')

# Features are the monomer compositions
features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
# Target is adhesive strength
target = 'Glass (kPa)_10s'

# Create a boxplot of monomer compositions
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[features])
plt.xticks(rotation=45, ha='right')
plt.title('Monomer Composition Distribution (Initial Training Data)')
plt.ylabel('Molar Fraction')
plt.tight_layout()
plt.savefig('report/images/monomer_distribution.png', dpi=300)
plt.close()

# Create a histogram of adhesive strength
plt.figure(figsize=(8, 5))
sns.histplot(df[target], bins=20, kde=True)
plt.title('Distribution of Adhesive Strength (Initial Training Data)')
plt.xlabel('Adhesive Strength (kPa)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('report/images/adhesive_strength_dist.png', dpi=300)
plt.close()
