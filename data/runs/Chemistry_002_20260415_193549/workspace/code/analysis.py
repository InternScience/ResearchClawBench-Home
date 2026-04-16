import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, Selection, NeighborSearch
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load SKEMPI data
skempi_df = pd.read_csv('../data/skempi_v2.csv', sep=';')

# Filter for 1BRS_A_D
brs_df = skempi_df[skempi_df['#Pdb'].str.contains('1BRS_A_D')]

# Clean mutations
def parse_mutations(mut_str):
    muts = mut_str.split(',')
    parsed = []
    for m in muts:
        wt = m[0]
        chain = m[1]
        res_num = int(m[2:-1])
        mut = m[-1]
        parsed.append({'wt': wt, 'chain': chain, 'res_num': res_num, 'mut': mut})
    return parsed

mutations = []
for idx, row in brs_df.iterrows():
    muts = parse_mutations(row['Mutation(s)_cleaned'])
    for m in muts:
        mutations.append({
            'Mutation_str': row['Mutation(s)_cleaned'],
            'chain': m['chain'],
            'res_num': m['res_num'],
            'wt': m['wt'],
            'mut': m['mut'],
            'Affinity_mut': row['Affinity_mut_parsed'],
            'Affinity_wt': row['Affinity_wt_parsed'],
            'Temperature': row['Temperature']
        })

mut_df = pd.DataFrame(mutations)

# Calculate ddG = RT ln(Kd_mut / Kd_wt)
# R = 1.987 cal/(mol*K) = 0.001987 kcal/(mol*K)
R = 0.001987
mut_df['Temperature'] = pd.to_numeric(mut_df['Temperature'].replace('298(assumed)', '298'), errors='coerce').fillna(298)
mut_df['ddG'] = R * mut_df['Temperature'] * np.log(mut_df['Affinity_mut'] / mut_df['Affinity_wt'])

# Load PDB
parser = PDBParser(QUIET=True)
structure = parser.get_structure('1BRS', '../data/1brs_AD.pdb')

# Get interface residues
model = structure[0]
chain_A_atoms = Selection.unfold_entities(model['A'], 'A')
chain_D_atoms = Selection.unfold_entities(model['D'], 'A')

ns_A = NeighborSearch(chain_A_atoms)
ns_D = NeighborSearch(chain_D_atoms)

interface_A = set()
for atom in chain_D_atoms:
    close_atoms = ns_A.search(atom.coord, 5.0) # 5 Angstrom cutoff
    for a in close_atoms:
        interface_A.add(a.get_parent().id[1])

interface_D = set()
for atom in chain_A_atoms:
    close_atoms = ns_D.search(atom.coord, 5.0)
    for a in close_atoms:
        interface_D.add(a.get_parent().id[1])

mut_df['is_interface'] = mut_df.apply(lambda row: 
    (row['chain'] == 'A' and row['res_num'] in interface_A) or 
    (row['chain'] == 'D' and row['res_num'] in interface_D), axis=1)

# Calculate distance to interface
def min_dist_to_interface(chain, res_num):
    try:
        res = model[chain][res_num]
        res_atoms = Selection.unfold_entities(res, 'A')
        
        if chain == 'A':
            target_atoms = [a for r in interface_D for a in model['D'][r]]
        else:
            target_atoms = [a for r in interface_A for a in model['A'][r]]
            
        min_dist = float('inf')
        for a1 in res_atoms:
            for a2 in target_atoms:
                dist = np.linalg.norm(a1.coord - a2.coord)
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    except KeyError:
        return np.nan

mut_df['dist_to_interface'] = mut_df.apply(lambda row: min_dist_to_interface(row['chain'], row['res_num']), axis=1)

# Save intermediate data
os.makedirs('../outputs', exist_ok=True)
mut_df.to_csv('../outputs/mut_analysis.csv', index=False)

# Visualization
os.makedirs('../report/images', exist_ok=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=mut_df, x='dist_to_interface', y='ddG', hue='is_interface', palette='Set1', s=100)
plt.axhline(0, color='gray', linestyle='--')
plt.title('Binding Affinity Change ($\Delta\Delta G$) vs. Distance to Interface')
plt.xlabel('Minimum Distance to Opposite Chain ($\AA$)')
plt.ylabel('$\Delta\Delta G$ (kcal/mol)')
plt.savefig('../report/images/ddg_vs_dist.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(data=mut_df, x='is_interface', y='ddG', palette='Set2')
sns.stripplot(data=mut_df, x='is_interface', y='ddG', color='black', alpha=0.5)
plt.title('$\Delta\Delta G$ for Interface vs. Non-Interface Mutations')
plt.xlabel('Is Interface Residue?')
plt.ylabel('$\Delta\Delta G$ (kcal/mol)')
plt.savefig('../report/images/ddg_boxplot.png')
plt.close()

# Interface mapping heatmap
interface_ddg = mut_df[mut_df['is_interface'] == True].groupby(['chain', 'res_num'])['ddG'].mean().reset_index()
pivot_A = interface_ddg[interface_ddg['chain'] == 'A'].pivot(index='chain', columns='res_num', values='ddG')
pivot_D = interface_ddg[interface_ddg['chain'] == 'D'].pivot(index='chain', columns='res_num', values='ddG')

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
if not pivot_A.empty:
    sns.heatmap(pivot_A, cmap='coolwarm', center=0, ax=axes[0], annot=True, fmt=".1f", cbar_kws={'label': '$\Delta\Delta G$'})
    axes[0].set_title('Chain A (Barnase) Interface Mutations')
if not pivot_D.empty:
    sns.heatmap(pivot_D, cmap='coolwarm', center=0, ax=axes[1], annot=True, fmt=".1f", cbar_kws={'label': '$\Delta\Delta G$'})
    axes[1].set_title('Chain D (Barstar) Interface Mutations')
plt.tight_layout()
plt.savefig('../report/images/interface_heatmap.png')
plt.close()

print("Analysis complete. Outputs saved.")
