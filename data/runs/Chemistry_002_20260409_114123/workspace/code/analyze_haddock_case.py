import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')


def load_skempi(path):
    header = path.read_text().splitlines()[0].lstrip('#').split(';')
    df = pd.read_csv(path, sep=';', comment='#', header=None, engine='python')
    df.columns = header
    for col in ['Affinity_mut_parsed', 'Affinity_wt_parsed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    temp = df['Temperature'].astype(str).str.extract(r'([-+0-9.]+)')[0]
    df['Temperature_num'] = pd.to_numeric(temp, errors='coerce').fillna(298.15)
    valid = (df['Affinity_mut_parsed'] > 0) & (df['Affinity_wt_parsed'] > 0)
    R = 1.987e-3
    df['ddG_kcal_mol'] = np.where(
        valid,
        R * df['Temperature_num'] * np.log(df['Affinity_mut_parsed'] / df['Affinity_wt_parsed']),
        np.nan,
    )
    df['n_mutations'] = df['Mutation(s)_cleaned'].astype(str).str.count(',') + 1
    df['location_simple'] = df['iMutation_Location(s)'].astype(str).str.split(',').str[0]
    return df


def parse_structure(path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', str(path))
    model = next(structure.get_models())
    chains = {chain.id: chain for chain in model}
    chain_ids = sorted(chains)
    res_by_chain = {}
    atoms_by_chain = {}
    ca_coords = {}
    for cid, chain in chains.items():
        residues = [r for r in chain if r.id[0] == ' ']
        res_by_chain[cid] = residues
        atoms_by_chain[cid] = sum(1 for r in residues for _ in r)
        coords = []
        ids = []
        for r in residues:
            if 'CA' in r:
                coords.append(r['CA'].coord)
                ids.append(r.id[1])
        ca_coords[cid] = (ids, np.array(coords) if coords else np.empty((0, 3)))

    interactions = []
    interface_by_chain = {cid: set() for cid in chain_ids}
    pairwise = None
    if len(chain_ids) >= 2:
        c1, c2 = chain_ids[:2]
        res1 = res_by_chain[c1]
        res2 = res_by_chain[c2]
        mat = np.full((len(res1), len(res2)), np.nan)
        for i, r1 in enumerate(res1):
            for j, r2 in enumerate(res2):
                dmin = min(a1 - a2 for a1 in r1 for a2 in r2)
                mat[i, j] = dmin
                if dmin < 5.0:
                    interface_by_chain[c1].add(r1.id[1])
                    interface_by_chain[c2].add(r2.id[1])
                    interactions.append({
                        'chain_1': c1, 'resnum_1': r1.id[1], 'resname_1': r1.resname,
                        'chain_2': c2, 'resnum_2': r2.id[1], 'resname_2': r2.resname,
                        'min_distance': float(dmin),
                    })
        pairwise = {
            'chains': [c1, c2],
            'resids_1': [r.id[1] for r in res1],
            'resids_2': [r.id[1] for r in res2],
            'matrix': mat,
        }

    centers = {}
    for cid, (_, coords) in ca_coords.items():
        if len(coords):
            centers[cid] = coords.mean(axis=0).tolist()

    return {
        'chain_ids': chain_ids,
        'n_residues': {cid: len(res_by_chain[cid]) for cid in chain_ids},
        'n_atoms': atoms_by_chain,
        'centers': centers,
        'interface_by_chain': {k: sorted(v) for k, v in interface_by_chain.items()},
        'interactions': interactions,
        'pairwise': pairwise,
    }


def make_figures(df, struct):
    # Figure 1: ddG distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['ddG_kcal_mol'].dropna(), bins=50, color='#4477AA')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('ΔΔG (kcal/mol)')
    plt.ylabel('Count')
    plt.title('Distribution of SKEMPI mutation effects')
    plt.tight_layout()
    plt.savefig(IMG / 'ddg_distribution.png', dpi=300)
    plt.close()

    # Figure 2: mutation location effects
    order = df.groupby('location_simple')['ddG_kcal_mol'].median().sort_values(ascending=False).index.tolist()
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df[df['location_simple'].isin(order[:6])], x='location_simple', y='ddG_kcal_mol', order=order[:6], color='#66CCAA')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Mutation location class')
    plt.ylabel('ΔΔG (kcal/mol)')
    plt.title('Binding effect by mutation location')
    plt.tight_layout()
    plt.savefig(IMG / 'location_ddg_boxplot.png', dpi=300)
    plt.close()

    # Figure 3: interface heatmap
    pair = struct['pairwise']
    mat = pair['matrix']
    dfmat = pd.DataFrame(mat, index=pair['resids_1'], columns=pair['resids_2'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(dfmat, cmap='viridis_r', vmin=2, vmax=15, cbar_kws={'label': 'Minimum heavy-atom distance (Å)'})
    plt.xlabel(f"Chain {pair['chains'][1]} residue")
    plt.ylabel(f"Chain {pair['chains'][0]} residue")
    plt.title('Barnase–barstar inter-chain residue distance map')
    plt.tight_layout()
    plt.savefig(IMG / 'interface_distance_map.png', dpi=300)
    plt.close()

    # Figure 4: contact profile comparison
    chain_a = pd.Series(0, index=pair['resids_1'], dtype=float)
    chain_d = pd.Series(0, index=pair['resids_2'], dtype=float)
    for inter in struct['interactions']:
        strength = max(0, 5.0 - inter['min_distance'])
        chain_a.loc[inter['resnum_1']] += strength
        chain_d.loc[inter['resnum_2']] += strength
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].bar(chain_a.index.astype(str), chain_a.values, color='#CC6677')
    axes[0].set_title('Chain A interface contact intensity')
    axes[0].set_ylabel('Σ(5Å - dmin)+')
    axes[0].tick_params(axis='x', labelrotation=90)
    axes[1].bar(chain_d.index.astype(str), chain_d.values, color='#332288')
    axes[1].set_title('Chain D interface contact intensity')
    axes[1].set_ylabel('Σ(5Å - dmin)+')
    axes[1].set_xlabel('Residue number')
    axes[1].tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.savefig(IMG / 'interface_contact_profile.png', dpi=300)
    plt.close()


def summarize(df, struct):
    interface_sizes = {k: len(v) for k, v in struct['interface_by_chain'].items()}
    top_contacts = sorted(struct['interactions'], key=lambda x: x['min_distance'])[:15]
    summary = {
        'skempi': {
            'rows': int(len(df)),
            'unique_pdb': int(df['Pdb'].nunique()),
            'ddg_summary': df['ddG_kcal_mol'].describe().round(4).to_dict(),
            'location_counts': df['location_simple'].value_counts().head(10).to_dict(),
            'multi_mutant_fraction': float((df['n_mutations'] > 1).mean()),
        },
        'structure': {
            'chains': struct['chain_ids'],
            'n_residues': struct['n_residues'],
            'n_atoms': struct['n_atoms'],
            'interface_sizes': interface_sizes,
            'n_interactions_below_5A': len(struct['interactions']),
            'top_contacts': top_contacts,
        }
    }
    return summary


def main():
    df = load_skempi(DATA / 'skempi_v2.csv')
    struct = parse_structure(DATA / '1brs_AD.pdb')
    make_figures(df, struct)
    summary = summarize(df, struct)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(struct['interactions']).sort_values('min_distance').to_csv(OUT / 'interface_contacts.csv', index=False)
    df[['Pdb','Mutation(s)_cleaned','Protein 1','Protein 2','iMutation_Location(s)','location_simple','n_mutations','ddG_kcal_mol']].to_csv(OUT / 'skempi_ddg_processed.csv', index=False)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
