"""
Phase 2: SKEMPI Data Analysis
Extract barnase-barstar mutations, correlate with interface predictions
"""
import pandas as pd
import numpy as np
import json
import os
import re

# Load SKEMPI data
df = pd.read_csv('data/skempi_v2.csv', sep=';')
print(f"Total entries in SKEMPI v2: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Look for barnase-barstar related entries
# Barnase is from Bacillus amyloliquefaciens, barstar is its inhibitor
# PDB codes for barnase-barstar: 1BRS, 1BRX, 1BGC, etc.

# Search for barnase/barstar in protein columns
barnase_mask = df['Protein 1'].str.contains('barnase|Barnase|BARNASE', na=False) | \
               df['Protein 2'].str.contains('barnase|Barnase|BARNASE', na=False)
barstar_mask = df['Protein 1'].str.contains('barstar|Barstar|BARSTAR', na=False) | \
               df['Protein 2'].str.contains('barstar|Barstar|BARSTAR', na=False)

barnase_barstar_df = df[barnase_mask & barstar_mask]
print(f"\nBarnase-barstar entries: {len(barnase_barstar_df)}")

# Also search by PDB code
pdb_1brs = df[df['#Pdb'].str.contains('1BRS|1brs', na=False, case=False)]
print(f"Entries with 1BRS PDB code: {len(pdb_1brs)}")

# Combine all barnase-barstar related entries
all_bb = pd.concat([barnase_barstar_df, pdb_1brs]).drop_duplicates()
print(f"Total unique barnase-barstar entries: {len(all_bb)}")

if len(all_bb) > 0:
    print(f"\nProtein pairs found:")
    print(all_bb[['Protein 1', 'Protein 2']].drop_duplicates())
    
    print(f"\nMutation types:")
    print(all_bb['Mutation(s)_cleaned'].head(20))

# If no direct barnase-barstar, look for barnase alone
if len(all_bb) == 0:
    print("\nNo direct barnase-barstar entries found.")
    print("Searching for barnase-only entries...")
    barnase_only = df[df['Protein 1'].str.contains('barnase|Barnase|BARNASE', na=False) | 
                      df['Protein 2'].str.contains('barnase|Barnase|BARNASE', na=False)]
    print(f"Barnase entries: {len(barnase_only)}")
    if len(barnase_only) > 0:
        print(barnase_only[['Protein 1', 'Protein 2', '#Pdb']].drop_duplicates().head(20))

# Save relevant data
os.makedirs('outputs', exist_ok=True)

# Save all barnase-barstar data
if len(all_bb) > 0:
    all_bb.to_csv('outputs/barnase_barstar_skempi.csv', index=False)
    print(f"\nSaved {len(all_bb)} barnase-barstar entries to outputs/")
else:
    # Save a broader set - all protein-protein interactions for comparison
    # Focus on finding any relevant mutation data
    print("\nSaving general SKEMPI statistics...")
    
    # Get unique protein pairs
    protein_pairs = df.groupby(['Protein 1', 'Protein 2']).size().reset_index(name='count')
    protein_pairs = protein_pairs.sort_values('count', ascending=False)
    print("\nTop protein pairs in SKEMPI:")
    print(protein_pairs.head(20))
    
    # Save summary stats
    stats = {
        'total_entries': len(df),
        'unique_proteins_1': df['Protein 1'].nunique(),
        'unique_proteins_2': df['Protein 2'].nunique(),
        'top_pairs': protein_pairs.head(30).to_dict('records')
    }
    
    with open('outputs/skempi_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
