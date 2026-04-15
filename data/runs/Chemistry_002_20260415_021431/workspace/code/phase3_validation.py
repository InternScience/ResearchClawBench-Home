"""
Phase 3: Comprehensive Analysis - Interface Prediction and SKEMPI Validation
Correlate structural interface predictions with experimental mutation effects
"""
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

# Load PDB analysis results
with open('outputs/interface_analysis.json', 'r') as f:
    interface_data = json.load(f)

with open('outputs/pdb_analysis.json', 'r') as f:
    pdb_stats = json.load(f)

# Load SKEMPI barnase-barstar data
skempi_df = pd.read_csv('outputs/barnase_barstar_skempi.csv', sep=',')
print(f"SKEMPI barnase-barstar entries: {len(skempi_df)}")

# Parse mutations and calculate dDeltaG
def parse_mutation(mut_str):
    """Parse mutation string like 'KA25A' into (chain, wt_res, res_seq, mut_res)"""
    if ',' in str(mut_str):
        # Multiple mutations
        return [parse_single_mutation(m) for m in str(mut_str).split(',')]
    return [parse_single_mutation(mut_str)]

def parse_single_mutation(mut_str):
    """Parse single mutation like 'KA25A' -> {'chain': 'A', 'wt_res': 'K', 'res_seq': 25, 'mut_res': 'A'}"""
    mut_str = str(mut_str).strip()
    # Format: WT_RES + CHAIN + RES_SEQ + MUT_RES
    # e.g., KA25A -> K(chain A, pos 25) -> A
    match = re.match(r'([A-Z])([A-Z])(\d+)([A-Z])', mut_str)
    if match:
        return {
            'wt_res': match.group(1),
            'chain': match.group(2),
            'res_seq': int(match.group(3)),
            'mut_res': match.group(4)
        }
    return None

import re

# Parse all mutations
parsed_mutations = []
for idx, row in skempi_df.iterrows():
    mut_str = row['Mutation(s)_cleaned']
    parsed = parse_mutation(mut_str)
    if parsed and parsed[0]:
        for m in parsed:
            if m:
                m['original_idx'] = idx
                m['mutation_str'] = mut_str
                parsed_mutations.append(m)

print(f"Parsed individual mutations: {len(parsed_mutations)}")

# Calculate dDeltaG from affinity values
# dDeltaG = RT * ln(Kd_mut / Kd_wt) = RT * ln(Affinity_mut / Affinity_wt)
# R = 1.987 cal/(mol·K), T ≈ 298 K
R = 1.987  # cal/(mol·K)
T = 298.15  # K

def calculate_ddg(affinity_mut, affinity_wt):
    """Calculate binding free energy change upon mutation"""
    try:
        kd_mut = float(affinity_mut)
        kd_wt = float(affinity_wt)
        if kd_mut > 0 and kd_wt > 0:
            ddg = R * T * np.log(kd_mut / kd_wt) / 1000  # Convert to kcal/mol
            return ddg
    except:
        pass
    return np.nan

# Calculate dDeltaG for each entry
skempi_df['ddG_kcal_mol'] = skempi_df.apply(
    lambda row: calculate_ddg(row['Affinity_mut_parsed'], row['Affinity_wt_parsed']), axis=1
)

print(f"\ndDeltaG statistics:")
print(skempi_df['ddG_kcal_mol'].describe())

# Map mutations to interface residues
interface_res_set = set()
for pair in interface_data['interface_pairs']:
    res1 = pair['res1']  # (chain, res_seq)
    res2 = pair['res2']
    interface_res_set.add((res1[0], res1[1]))
    interface_res_set.add((res2[0], res2[1]))

print(f"\nInterface residues: {len(interface_res_set)}")

# Classify mutations as interface vs non-interface
def is_interface_mutation(mutations_list):
    """Check if any mutated residue is at the interface"""
    for m in mutations_list:
        if m and (m['chain'], m['res_seq']) in interface_res_set:
            return True
    return False

# Get unique mutations per entry
unique_entries = skempi_df.drop_duplicates(subset=['Mutation(s)_cleaned'])
print(f"\nUnique mutations: {len(unique_entries)}")

# Classify
interface_mask = unique_entries['Mutation(s)_cleaned'].apply(
    lambda x: is_interface_mutation(parse_mutation(x))
)

interface_mutations = unique_entries[interface_mask]
non_interface_mutations = unique_entries[~interface_mask]

print(f"Interface mutations: {len(interface_mutations)}")
print(f"Non-interface mutations: {len(non_interface_mutations)}")

# Statistical comparison
if len(interface_mutations) > 0 and len(non_interface_mutations) > 0:
    print(f"\ndDeltaG - Interface mutations:")
    print(interface_mutations['ddG_kcal_mol'].describe())
    
    print(f"\ndDeltaG - Non-interface mutations:")
    print(non_interface_mutations['ddG_kcal_mol'].describe())
    
    # Mann-Whitney U test
    from scipy import stats
    int_ddg = interface_mutations['ddG_kcal_mol'].dropna().values
    non_int_ddg = non_interface_mutations['ddG_kcal_mol'].dropna().values
    
    if len(int_ddg) > 0 and len(non_int_ddg) > 0:
        stat, p_value = stats.mannwhitneyu(int_ddg, non_int_ddg, alternative='greater')
        print(f"\nMann-Whitney U test (interface > non-interface):")
        print(f"U statistic: {stat:.2f}, p-value: {p_value:.6f}")

# Save results
os.makedirs('outputs', exist_ok=True)

results = {
    'total_mutations': len(skempi_df),
    'unique_mutations': len(unique_entries),
    'interface_mutations': len(interface_mutations),
    'non_interface_mutations': len(non_interface_mutations),
    'interface_ddg_stats': interface_mutations['ddG_kcal_mol'].describe().to_dict(),
    'non_interface_ddg_stats': non_interface_mutations['ddG_kcal_mol'].describe().to_dict(),
}

with open('outputs/validation_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save detailed mutation data
mutation_details = []
for idx, row in unique_entries.iterrows():
    mut_str = row['Mutation(s)_cleaned']
    parsed = parse_mutation(mut_str)
    is_interface = is_interface_mutation(parsed)
    
    mutation_details.append({
        'mutation': mut_str,
        'ddG_kcal_mol': float(row['ddG_kcal_mol']) if not np.isnan(row['ddG_kcal_mol']) else None,
        'is_interface': is_interface,
        'affinity_wt': row['Affinity_wt_parsed'],
        'affinity_mut': row['Affinity_mut_parsed']
    })

with open('outputs/mutation_details.json', 'w') as f:
    json.dump(mutation_details, f, indent=2)

print("\nValidation analysis complete!")
print(f"Results saved to outputs/")
