#!/usr/bin/env python3
"""
Foldseek-Multimer Style Structural Alignment: 7xg4 vs 6n40
Implements pairwise and complex-level structural alignment using TM-score.
"""

import numpy as np
import json
import os
from collections import defaultdict
from Bio.PDB import PDBParser
from tmtools.io import get_structure
from tmtools import tm_align
import warnings
warnings.filterwarnings('ignore')

def parse_pdb_chains(pdb_path):
    """Parse PDB file and extract chain information."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_path)
    model = structure[0]
    
    chains_info = {}
    for chain in model:
        chain_id = chain.get_id()
        residues = [r for r in chain if r.get_id()[0] == ' ']
        
        residue_names = [r.get_resname().strip() for r in residues]
        protein_residues = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS',
                          'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}
        na_residues = {'A','C','G','T','U','DA','DC','DG','DT','ADE','CYT','GUA','THY','URA'}
        
        n_protein = sum(1 for r in residue_names if r in protein_residues)
        n_na = sum(1 for r in residue_names if r in na_residues)
        
        if n_protein > n_na:
            mol_type = 'protein'
        elif n_na > 0:
            mol_type = 'nucleic_acid'
        else:
            mol_type = 'unknown'
        
        chains_info[chain_id] = {
            'n_residues': len(residues),
            'mol_type': mol_type,
        }
    
    return structure, model, chains_info


def extract_ca_data(chain):
    """Extract C-alpha coordinates and residue names from a chain."""
    coords = []
    names = []
    for r in chain:
        if r.get_id()[0] == ' ' and 'CA' in r:
            coords.append(r['CA'].get_coord())
            names.append(r.get_resname().strip()[0])
    return np.array(coords) if coords else np.empty((0,3)), ''.join(names)


def count_aligned(seqM):
    """Count aligned residue pairs from TM-align alignment string."""
    if seqM is None:
        return 0
    return sum(1 for c in seqM if c == ':')


def compute_tm_score_chain(chain1, chain2):
    """Compute TM-score between two chains using tmtools."""
    try:
        coords1, names1 = extract_ca_data(chain1)
        coords2, names2 = extract_ca_data(chain2)
        
        if len(coords1) < 5 or len(coords2) < 5:
            return None
        
        result = tm_align(coords1, coords2, names1, names2)
        n_aligned = count_aligned(result.seqM)
        
        return {
            'tm_chain1': result.tm_norm_chain1,
            'tm_chain2': result.tm_norm_chain2,
            'tm_avg': (result.tm_norm_chain1 + result.tm_norm_chain2) / 2,
            'rmsd': result.rmsd,
            'n_aligned': n_aligned,
            'n_query': len(coords1),
            'n_target': len(coords2),
            'rotation': result.u.tolist(),
            'translation': result.t.tolist(),
            'seqM': result.seqM,
            'seqxA': result.seqxA,
            'seqyA': result.seqyA,
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def greedy_chain_assignment(query_model, target_model, query_chains, target_chains):
    """
    Greedy chain assignment similar to US-align/EGS algorithm.
    """
    tm_matrix = {}
    for q_id in query_chains:
        q_chain = query_model[q_id]
        for t_id in target_chains:
            t_chain = target_model[t_id]
            info = compute_tm_score_chain(q_chain, t_chain)
            if info is not None:
                tm_matrix[(q_id, t_id)] = info
    
    # Greedy assignment
    assigned_query = set()
    assigned_target = set()
    assignments = []
    
    sorted_pairs = sorted(tm_matrix.items(), key=lambda x: x[1]['tm_avg'], reverse=True)
    
    for (q_id, t_id), info in sorted_pairs:
        if q_id not in assigned_query and t_id not in assigned_target:
            assignments.append({
                'query_chain': q_id,
                'target_chain': t_id,
                'tm_score_query': round(info['tm_chain1'], 4),
                'tm_score_target': round(info['tm_chain2'], 4),
                'tm_score_avg': round(info['tm_avg'], 4),
                'rmsd': round(info['rmsd'], 2),
                'n_aligned': info['n_aligned'],
                'n_query': info['n_query'],
                'n_target': info['n_target'],
                'rotation_matrix': info['rotation'],
                'translation_vector': info['translation'],
            })
            assigned_query.add(q_id)
            assigned_target.add(t_id)
    
    return assignments, tm_matrix


def compute_complex_tm_score(assignments, query_model, target_model,
                              query_chain_ids, target_chain_ids):
    """Compute complex-level TM-score."""
    total_query_len = 0
    for q_id in query_chain_ids:
        coords, _ = extract_ca_data(query_model[q_id])
        total_query_len += len(coords)
    
    total_target_len = 0
    for t_id in target_chain_ids:
        coords, _ = extract_ca_data(target_model[t_id])
        total_target_len += len(coords)
    
    if total_target_len == 0 or total_query_len == 0:
        return {}
    
    weighted_tm = 0.0
    total_aligned = 0
    for a in assignments:
        n_q = a['n_query']
        weighted_tm += a['tm_score_avg'] * n_q
        total_aligned += a['n_aligned']
    
    tm_q = weighted_tm / total_query_len
    tm_t = weighted_tm / total_target_len
    
    return {
        'tm_score_normalized_by_query': round(tm_q, 4),
        'tm_score_normalized_by_target': round(tm_t, 4),
        'tm_score_avg': round((tm_q + tm_t) / 2, 4),
        'coverage_query': round(total_aligned / total_query_len, 4),
        'coverage_target': round(total_aligned / total_target_len, 4),
        'total_aligned_residues': total_aligned,
        'query_total_residues': total_query_len,
        'target_total_residues': total_target_len
    }


def main():
    print("=" * 70)
    print("Foldseek-Multimer Style Structural Alignment")
    print("Query: 7xg4 (Pseudomonas aeruginosa Type IV-A CRISPR-Cas)")
    print("Target: 6n40 (Mycobacterium smegmatis MmpL3)")
    print("=" * 70)
    
    # Parse structures
    print("\n[1] Parsing PDB structures...")
    q_struct, q_model, q_info = parse_pdb_chains('data/7xg4.pdb')
    t_struct, t_model, t_info = parse_pdb_chains('data/6n40.pdb')
    
    print(f"\n  7xg4 chains: {len(q_info)}")
    for cid, info in sorted(q_info.items()):
        print(f"    Chain {cid}: {info['mol_type']:15s} | {info['n_residues']:4d} residues")
    
    print(f"\n  6n40 chains: {len(t_info)}")
    for cid, info in sorted(t_info.items()):
        print(f"    Chain {cid}: {info['mol_type']:15s} | {info['n_residues']:4d} residues")
    
    q_protein_chains = [cid for cid, info in q_info.items() if info['mol_type'] == 'protein']
    t_protein_chains = [cid for cid, info in t_info.items() if info['mol_type'] == 'protein']
    
    # Step 1: Pairwise chain alignments
    print("\n[2] Computing pairwise chain TM-scores...")
    pairwise_results = []
    for q_id in sorted(q_info.keys()):
        info = compute_tm_score_chain(q_model[q_id], t_model['A'])
        if info is not None:
            pairwise_results.append({
                'query_chain': q_id,
                'target_chain': 'A',
                'mol_type': q_info[q_id]['mol_type'],
                'tm_score_query_norm': round(info['tm_chain1'], 4),
                'tm_score_target_norm': round(info['tm_chain2'], 4),
                'tm_score_avg': round(info['tm_avg'], 4),
                'rmsd': round(info['rmsd'], 2),
                'n_query': info['n_query'],
                'n_target': info['n_target'],
                'n_aligned': info['n_aligned'],
            })
            print(f"    {q_id} ({q_info[q_id]['mol_type']:15s}): TM={info['tm_chain1']:.4f}/{info['tm_chain2']:.4f} "
                  f"RMSD={info['rmsd']:.2f}A Aligned={info['n_aligned']}")
        else:
            print(f"    {q_id} ({q_info[q_id]['mol_type']:15s}): Alignment failed")
    
    pairwise_results.sort(key=lambda x: x['tm_score_avg'], reverse=True)
    
    # Step 2: Greedy chain assignment
    print("\n[3] Performing greedy chain assignment (Foldseek-Multimer style)...")
    assignments, tm_matrix = greedy_chain_assignment(q_model, t_model,
                                                      list(q_info.keys()),
                                                      list(t_info.keys()))
    
    print(f"\n  Chain assignments ({len(assignments)} pairs):")
    for a in assignments:
        print(f"    7xg4:{a['query_chain']} <-> 6n40:{a['target_chain']} | "
              f"TM={a['tm_score_avg']:.4f} RMSD={a['rmsd']:.2f}A Aligned={a['n_aligned']}")
    
    # Step 3: Complex-level TM-score
    print("\n[4] Computing complex-level TM-score...")
    complex_scores = compute_complex_tm_score(assignments, q_model, t_model,
                                               list(q_info.keys()),
                                               list(t_info.keys()))
    
    for k, v in complex_scores.items():
        print(f"  {k}: {v}")
    
    # Step 4: Full alignment matrix
    print("\n[5] Computing full protein chain alignment matrix...")
    full_matrix = []
    for q_id in q_protein_chains:
        row = {}
        for t_id in t_protein_chains:
            info = compute_tm_score_chain(q_model[q_id], t_model[t_id])
            if info is not None:
                row[t_id] = {'tm_avg': round(info['tm_avg'], 4), 'rmsd': round(info['rmsd'], 2),
                            'n_aligned': info['n_aligned']}
            else:
                row[t_id] = {'tm_avg': 0.0, 'rmsd': None, 'n_aligned': 0}
        full_matrix.append({'query_chain': q_id, 'targets': row})
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    
    results = {
        'query': '7xg4',
        'target': '6n40',
        'query_info': q_info,
        'target_info': t_info,
        'pairwise_chain_alignments': pairwise_results,
        'chain_assignments': assignments,
        'complex_scores': complex_scores,
        'full_alignment_matrix': full_matrix
    }
    
    with open('outputs/alignment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n[6] Results saved to outputs/alignment_results.json")
    return results


if __name__ == '__main__':
    results = main()
