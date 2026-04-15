import json
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

RT_KCAL = 0.0019872041 * 298.0
AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}


def ensure_dirs():
    for p in ['outputs', 'report/images']:
        os.makedirs(p, exist_ok=True)


def parse_pdb(path):
    atoms = []
    residues = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            chain = line[21].strip()
            resseq = int(line[22:26])
            resname = line[17:20].strip()
            atom = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = (line[76:78].strip() or atom[0]).upper()
            if chain not in {'A', 'D'}:
                continue
            key = (chain, resseq)
            residues.setdefault(key, {'chain': chain, 'resseq': resseq, 'resname': resname, 'atoms': []})
            residues[key]['atoms'].append({'atom': atom, 'x': x, 'y': y, 'z': z, 'elem': elem})
            atoms.append((chain, resseq, atom, x, y, z))
    return residues


def atom_dist(a, b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)


def residue_min_distance(res1, res2):
    md = float('inf')
    for a in res1['atoms']:
        for b in res2['atoms']:
            if a['elem'] == 'H' or b['elem'] == 'H':
                continue
            d = atom_dist(a, b)
            if d < md:
                md = d
    return md


def build_interface_metrics(residues):
    chainA = [v for k, v in residues.items() if k[0] == 'A']
    chainD = [v for k, v in residues.items() if k[0] == 'D']
    pair_rows = []
    per_res = {}
    for r in chainA + chainD:
        per_res[(r['chain'], r['resseq'])] = {
            'chain': r['chain'], 'resseq': r['resseq'], 'resname': r['resname'],
            'aa1': AA3_TO_1.get(r['resname'], 'X'), 'min_cross_chain_distance': float('inf'),
            'contact_partners_5A': 0, 'contact_partners_8A': 0
        }
    for ra in chainA:
        for rd in chainD:
            d = residue_min_distance(ra, rd)
            pair_rows.append({'chain_A_resseq': ra['resseq'], 'chain_A_resname': ra['resname'],
                              'chain_D_resseq': rd['resseq'], 'chain_D_resname': rd['resname'],
                              'min_distance': d})
            pa = per_res[('A', ra['resseq'])]
            partner_d = per_res[('D', rd['resseq'])]
            pa['min_cross_chain_distance'] = min(pa['min_cross_chain_distance'], d)
            partner_d['min_cross_chain_distance'] = min(partner_d['min_cross_chain_distance'], d)
            if d <= 5.0:
                pa['contact_partners_5A'] += 1
                partner_d['contact_partners_5A'] += 1
            if d <= 8.0:
                pa['contact_partners_8A'] += 1
                partner_d['contact_partners_8A'] += 1
    return pd.DataFrame(pair_rows), pd.DataFrame(per_res.values()).sort_values(['chain', 'resseq'])


def parse_mutation_token(token):
    m = re.fullmatch(r'([A-Z])([A-Z])(\d+)([A-Z])', token.strip())
    if not m:
        return None
    wt, chain, resseq, mut = m.groups()
    return {'wt': wt, 'chain': chain, 'resseq': int(resseq), 'mut': mut}


def add_ddg(df):
    df = df.copy()
    df['Affinity_mut_parsed'] = pd.to_numeric(df['Affinity_mut_parsed'], errors='coerce')
    df['Affinity_wt_parsed'] = pd.to_numeric(df['Affinity_wt_parsed'], errors='coerce')
    df['ddG_kcal_mol'] = RT_KCAL * np.log(df['Affinity_mut_parsed'] / df['Affinity_wt_parsed'])
    df['log10_Kd_fold_change'] = np.log10(df['Affinity_mut_parsed'] / df['Affinity_wt_parsed'])
    return df


def summarize_structure(pair_df, res_df):
    interface_res = res_df[res_df['min_cross_chain_distance'] <= 5.0]
    return {
        'n_residues_chain_A': int((res_df['chain'] == 'A').sum()),
        'n_residues_chain_D': int((res_df['chain'] == 'D').sum()),
        'n_interface_residues_5A': int(len(interface_res)),
        'n_chain_A_interface_residues_5A': int(((interface_res['chain'] == 'A')).sum()),
        'n_chain_D_interface_residues_5A': int(((interface_res['chain'] == 'D')).sum()),
        'closest_residue_pair_distance_Ang': float(pair_df['min_distance'].min()),
        'mean_min_cross_chain_distance_Ang': float(res_df['min_cross_chain_distance'].mean())
    }


def make_figures(pair_df, res_df, skempi_all, sub_single, sub_all):
    sns.set_theme(style='whitegrid', context='talk')

    # Figure 1: structure overview
    topA = res_df[res_df['chain']=='A'][['resseq','min_cross_chain_distance']].copy()
    topD = res_df[res_df['chain']=='D'][['resseq','min_cross_chain_distance']].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    sns.scatterplot(data=pair_df, x='chain_A_resseq', y='chain_D_resseq', hue='min_distance',
                    palette='viridis_r', s=60, ax=axes[0], edgecolor=None)
    axes[0].set_title('Barnase-Barstar residue-pair proximity map')
    axes[0].set_xlabel('Barnase residue index (chain A)')
    axes[0].set_ylabel('Barstar residue index (chain D)')
    axes[0].legend(title='Min atom\ndistance (Å)', bbox_to_anchor=(1.02,1), loc='upper left')
    axes[1].plot(topA['resseq'], topA['min_cross_chain_distance'], label='Chain A (barnase)', lw=2)
    axes[1].plot(topD['resseq'], topD['min_cross_chain_distance'], label='Chain D (barstar)', lw=2)
    axes[1].axhline(5.0, ls='--', c='red', label='5 Å contact cutoff')
    axes[1].set_title('Per-residue nearest cross-chain distance')
    axes[1].set_xlabel('Residue index')
    axes[1].set_ylabel('Nearest opposite-chain atom distance (Å)')
    axes[1].legend()
    fig.savefig('report/images/structure_overview.png', dpi=200)
    plt.close(fig)

    # Figure 2: SKEMPI overview
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    loc_order = sub_single.groupby('iMutation_Location(s)')['ddG_kcal_mol'].median().sort_values(ascending=False).index
    sns.boxplot(data=sub_single, x='iMutation_Location(s)', y='ddG_kcal_mol', order=loc_order, ax=axes[0])
    axes[0].set_title('Single-mutation ΔΔG by SKEMPI location class')
    axes[0].set_xlabel('Location class')
    axes[0].set_ylabel('ΔΔG (kcal/mol)')
    axes[0].tick_params(axis='x', rotation=35)
    sns.histplot(data=sub_all, x='ddG_kcal_mol', hue='mutation_order', multiple='stack', bins=18, ax=axes[1])
    axes[1].set_title('Barnase-Barstar mutation effect distribution')
    axes[1].set_xlabel('ΔΔG (kcal/mol)')
    axes[1].set_ylabel('Count')
    fig.savefig('report/images/skempi_overview.png', dpi=200)
    plt.close(fig)

    # Figure 3: validation comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    sns.scatterplot(data=sub_single, x='min_cross_chain_distance', y='ddG_kcal_mol', hue='chain', style='iMutation_Location(s)', s=110, ax=axes[0])
    axes[0].axvline(5.0, ls='--', c='red')
    axes[0].set_title('Experimental ΔΔG vs structural proximity')
    axes[0].set_xlabel('Residue nearest opposite-chain distance (Å)')
    axes[0].set_ylabel('ΔΔG (kcal/mol)')
    sns.regplot(data=sub_single, x='contact_partners_5A', y='ddG_kcal_mol', scatter=False, ax=axes[1], color='black')
    sns.stripplot(data=sub_single, x='contact_partners_5A', y='ddG_kcal_mol', hue='chain', dodge=False, alpha=0.7, ax=axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles[:2], labels[:2], title='Chain')
    axes[1].set_title('Experimental ΔΔG vs number of close residue partners')
    axes[1].set_xlabel('Opposite-chain residue partners within 5 Å')
    axes[1].set_ylabel('ΔΔG (kcal/mol)')
    fig.savefig('report/images/validation_comparison.png', dpi=200)
    plt.close(fig)


def main():
    ensure_dirs()
    residues = parse_pdb('data/1brs_AD.pdb')
    pair_df, res_df = build_interface_metrics(residues)
    pair_df.to_csv('outputs/interface_pair_distances.csv', index=False)
    res_df.to_csv('outputs/residue_interface_metrics.csv', index=False)

    skempi = pd.read_csv('data/skempi_v2.csv', sep=';')
    skempi = add_ddg(skempi)
    sub = skempi[skempi['#Pdb'].str.upper() == '1BRS_A_D'].copy()
    sub['mutation_order'] = np.where(sub['Mutation(s)_cleaned'].str.contains(','), 'multiple', 'single')

    parsed_rows = []
    for _, row in sub.iterrows():
        toks = [x.strip() for x in str(row['Mutation(s)_PDB']).split(',')]
        parsed = [parse_mutation_token(t) for t in toks]
        valid = [p for p in parsed if p is not None]
        rowd = row.to_dict()
        rowd['parsed_mutations'] = valid
        rowd['n_parsed_mutations'] = len(valid)
        parsed_rows.append(rowd)
    sub = pd.DataFrame(parsed_rows)

    single = sub[sub['n_parsed_mutations'] == 1].copy()
    single['chain'] = single['parsed_mutations'].apply(lambda x: x[0]['chain'])
    single['resseq'] = single['parsed_mutations'].apply(lambda x: x[0]['resseq'])
    single['wt_from_mutation'] = single['parsed_mutations'].apply(lambda x: x[0]['wt'])
    single['mut_to'] = single['parsed_mutations'].apply(lambda x: x[0]['mut'])
    single = single.merge(res_df, on=['chain', 'resseq'], how='left')
    single['wt_matches_structure'] = single['wt_from_mutation'] == single['aa1']

    sub.to_csv('outputs/skempi_1brs_subset.csv', index=False)
    single.to_csv('outputs/skempi_1brs_single_mutations_mapped.csv', index=False)

    structure_summary = summarize_structure(pair_df, res_df)
    location_summary = (single.groupby('iMutation_Location(s)')
                        .agg(n=('ddG_kcal_mol','size'), median_ddG=('ddG_kcal_mol','median'), mean_distance=('min_cross_chain_distance','mean'))
                        .reset_index())
    location_summary.to_csv('outputs/location_ddg_summary.csv', index=False)

    corr_distance = spearmanr(single['min_cross_chain_distance'], single['ddG_kcal_mol'], nan_policy='omit')
    corr_contacts = spearmanr(single['contact_partners_5A'], single['ddG_kcal_mol'], nan_policy='omit')
    strong_single = single.sort_values('ddG_kcal_mol', ascending=False).head(10)[[
        'Mutation(s)_cleaned','chain','resseq','iMutation_Location(s)','ddG_kcal_mol','min_cross_chain_distance','contact_partners_5A'
    ]]
    strong_single.to_csv('outputs/top_single_mutation_effects.csv', index=False)

    all_summary = {
        'structure_summary': structure_summary,
        'skempi_1brs_rows': int(len(sub)),
        'skempi_1brs_single_mutations': int(len(single)),
        'skempi_1brs_multiple_mutations': int((sub['n_parsed_mutations'] > 1).sum()),
        'single_mutation_wt_matches_structure_fraction': float(single['wt_matches_structure'].mean()),
        'single_mutation_ddG_summary': {
            'mean_kcal_mol': float(single['ddG_kcal_mol'].mean()),
            'median_kcal_mol': float(single['ddG_kcal_mol'].median()),
            'max_kcal_mol': float(single['ddG_kcal_mol'].max())
        },
        'spearman': {
            'distance_vs_ddG_rho': None if pd.isna(corr_distance.statistic) else float(corr_distance.statistic),
            'distance_vs_ddG_p': None if pd.isna(corr_distance.pvalue) else float(corr_distance.pvalue),
            'contacts5A_vs_ddG_rho': None if pd.isna(corr_contacts.statistic) else float(corr_contacts.statistic),
            'contacts5A_vs_ddG_p': None if pd.isna(corr_contacts.pvalue) else float(corr_contacts.pvalue)
        },
        'location_summary': location_summary.to_dict(orient='records')
    }
    with open('outputs/pdb_interface_summary.json', 'w') as fh:
        json.dump(structure_summary, fh, indent=2)
    with open('outputs/analysis_results.json', 'w') as fh:
        json.dump(all_summary, fh, indent=2)

    make_figures(pair_df, res_df, skempi, single, sub)

    claim_recovery = [
        {
            'claim': 'The provided 1BRS chains A and D form a large direct interface with many residues within 5 Å cross-chain distance.',
            'supporting_artifacts': ['outputs/pdb_interface_summary.json', 'outputs/residue_interface_metrics.csv', 'report/images/structure_overview.png']
        },
        {
            'claim': 'The SKEMPI subset contains extensive barnase-barstar mutation data, including both single and combinatorial mutations.',
            'supporting_artifacts': ['outputs/skempi_1brs_subset.csv', 'outputs/analysis_results.json', 'report/images/skempi_overview.png']
        },
        {
            'claim': 'Single mutations closer to the interface tend to show larger destabilization, but the relationship is moderate rather than deterministic.',
            'supporting_artifacts': ['outputs/skempi_1brs_single_mutations_mapped.csv', 'outputs/analysis_results.json', 'report/images/validation_comparison.png']
        }
    ]
    with open('outputs/claim_recovery_table.json', 'w') as fh:
        json.dump(claim_recovery, fh, indent=2)


if __name__ == '__main__':
    main()
