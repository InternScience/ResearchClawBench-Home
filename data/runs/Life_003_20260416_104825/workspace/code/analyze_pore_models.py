import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
dna_r9 = pd.read_csv('data/dna_r9.4.1_400bps_6mer_uncalled4.csv')
dna_r10 = pd.read_csv('data/dna_r10.4.1_400bps_9mer_uncalled4.csv')
rna_r9 = pd.read_csv('data/rna_r9.4.1_70bps_5mer_uncalled4.csv')
rna_r10 = pd.read_csv('data/rna004_130bps_9mer_uncalled4.csv')

def plot_distributions(df1, label1, df2, label2, feature, title, filename):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df1[feature], label=label1, fill=True, alpha=0.5)
    sns.kdeplot(df2[feature], label=label2, fill=True, alpha=0.5)
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# DNA Models: r9.4.1 vs r10.4.1
plot_distributions(dna_r9, 'DNA r9.4.1 (6-mer)', dna_r10, 'DNA r10.4.1 (9-mer)', 'current_mean', 
                   'Distribution of Current Mean: DNA r9.4.1 vs r10.4.1', 'report/images/dna_current_mean_dist.png')
plot_distributions(dna_r9, 'DNA r9.4.1 (6-mer)', dna_r10, 'DNA r10.4.1 (9-mer)', 'current_std', 
                   'Distribution of Current Std: DNA r9.4.1 vs r10.4.1', 'report/images/dna_current_std_dist.png')

# RNA Models: r9.4.1 vs rna004
plot_distributions(rna_r9, 'RNA r9.4.1 (5-mer)', rna_r10, 'RNA004 (9-mer)', 'current_mean', 
                   'Distribution of Current Mean: RNA r9.4.1 vs RNA004', 'report/images/rna_current_mean_dist.png')
plot_distributions(rna_r9, 'RNA r9.4.1 (5-mer)', rna_r10, 'RNA004 (9-mer)', 'current_std', 
                   'Distribution of Current Std: RNA r9.4.1 vs RNA004', 'report/images/rna_current_std_dist.png')

# K-mer length analysis
# Let's check how the current_mean changes across different central bases
def central_base(kmer):
    return kmer[len(kmer)//2]

dna_r9['central_base'] = dna_r9['kmer'].apply(central_base)
dna_r10['central_base'] = dna_r10['kmer'].apply(central_base)
rna_r9['central_base'] = rna_r9['kmer'].apply(central_base)
rna_r10['central_base'] = rna_r10['kmer'].apply(central_base)

def plot_central_base_boxplot(df, title, filename):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='central_base', y='current_mean', order=['A', 'C', 'G', 'T'])
    plt.title(title)
    plt.xlabel('Central Base')
    plt.ylabel('Current Mean')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_central_base_boxplot(dna_r9, 'Current Mean by Central Base (DNA r9.4.1)', 'report/images/dna_r9_central_base.png')
plot_central_base_boxplot(dna_r10, 'Current Mean by Central Base (DNA r10.4.1)', 'report/images/dna_r10_central_base.png')
plot_central_base_boxplot(rna_r9, 'Current Mean by Central Base (RNA r9.4.1)', 'report/images/rna_r9_central_base.png')
plot_central_base_boxplot(rna_r10, 'Current Mean by Central Base (RNA004)', 'report/images/rna_r10_central_base.png')

print("Pore model plots generated.")
