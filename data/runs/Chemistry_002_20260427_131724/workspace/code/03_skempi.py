"""Parse SKEMPI 2.0 for 1BRS_A_D entries, compute experimental ddG, and aggregate per residue."""
import pandas as pd, numpy as np, json, re

df = pd.read_csv('data/skempi_v2.csv', sep=';', low_memory=False)
df['pdb_id'] = df['#Pdb'].str.split('_').str[0]
b = df[df['#Pdb']=='1BRS_A_D'].copy()
print('Rows for 1BRS_A_D:', len(b))

# Parse mutations: format like "KA25A" = wt K, chain A, residue 25, mut A.
def parse_muts(s):
    if not isinstance(s,str): return []
    out=[]
    for tok in s.split(','):
        tok=tok.strip()
        m=re.match(r'^([A-Z])([A-Z])(\d+)([A-Z])$', tok)
        if not m: 
            return []  # skip ambiguous
        wt,ch,resi,mt = m.groups()
        out.append((wt,ch,int(resi),mt))
    return out

b['parsed'] = b['Mutation(s)_PDB'].apply(parse_muts)
b = b[b['parsed'].apply(lambda x: len(x)>0)].copy()
b = b[b['Affinity_mut_parsed'].notnull() & b['Affinity_wt_parsed'].notnull()]
b['T'] = pd.to_numeric(b['Temperature'].astype(str).str.replace(r'[^0-9.]','',regex=True), errors='coerce').fillna(298.0)
R = 1.9872041e-3  # kcal/mol/K
# ddG = -RT ln(Kd_wt/Kd_mut) so positive ddG = destabilizing
b['ddG'] = R*b['T']*np.log(b['Affinity_mut_parsed'].astype(float)/b['Affinity_wt_parsed'].astype(float))
b['n_muts'] = b['parsed'].apply(len)
b.to_csv('outputs/skempi_1brs.csv', index=False)
print('After filtering:', len(b))
print('single-mutant count:', (b.n_muts==1).sum())
print('ddG stats:', b['ddG'].describe())

# build per-residue summary using single mutants
single = b[b.n_muts==1].copy()
single['chain'] = single['parsed'].apply(lambda x: x[0][1])
single['resi']  = single['parsed'].apply(lambda x: x[0][2])
single['wt']    = single['parsed'].apply(lambda x: x[0][0])
single['mut']   = single['parsed'].apply(lambda x: x[0][3])
print('single-mutant chains:', single['chain'].value_counts())
agg = single.groupby(['chain','resi','wt']).agg(
    n_mutations=('ddG','size'),
    mean_ddG=('ddG','mean'),
    max_ddG=('ddG','max'),
    sum_abs_ddG=('ddG', lambda x: np.abs(x).sum())
).reset_index()
agg.to_csv('outputs/skempi_1brs_perresidue.csv', index=False)
print(agg.head(15))
print('residues per chain:', agg.groupby('chain').size())
