#!/usr/bin/env python3
"""
Structural Alignment of Protein Complexes 7xg4 vs 6n40
=======================================================
Implements pairwise and chain-level structural alignment using TM-score,
producing alignment results, superimposition vectors, and similarity metrics.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from Bio import PDB
from tmtools import tm_align

# ── Paths ──────────────────────────────────────────────────────────────
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, "data")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

PDB_QUERY = os.path.join(DATA_DIR, "7xg4.pdb")
PDB_TARGET = os.path.join(DATA_DIR, "6n40.pdb")

# ── Helper: parse PDB, extract per-chain Cα coords and sequences ──────
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    '5MC': 'C', 'OMC': 'C', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C', 'DA': 'A', 'DT': 'T',
    'DG': 'G', 'DC': 'C', 'DU': 'U',
}

def parse_pdb_chains(pdb_path):
    """Return dict {chain_id: {'coords': ndarray(N,3), 'seq': str, 'resnames': list}}"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    chains = {}
    for model in structure:
        for chain in model:
            coords = []
            seq_chars = []
            resnames = []
            for residue in chain:
                if not PDB.is_aa(residue, standard=False):
                    # Try nucleic acid residues
                    resname = residue.get_resname().strip()
                    if resname in THREE_TO_ONE:
                        if 'CA' in residue or "C1'" in residue:
                            atom = residue['CA'] if 'CA' in residue else residue["C1'"]
                            coords.append(atom.get_coord())
                            seq_chars.append(THREE_TO_ONE.get(resname, 'X'))
                            resnames.append(resname)
                    continue
                resname = residue.get_resname().strip()
                if 'CA' not in residue:
                    continue
                coords.append(residue['CA'].get_coord())
                aa = THREE_TO_ONE.get(resname, 'X')
                seq_chars.append(aa)
                resnames.append(resname)
            if coords:
                chains[chain.id] = {
                    'coords': np.array(coords, dtype=np.float64),
                    'seq': ''.join(seq_chars),
                    'resnames': resnames,
                }
        break  # only first model
    return chains


def run_tm_align(coords_q, seq_q, coords_t, seq_t):
    """Run TM-align on two sets of Cα coordinates."""
    result = tm_align(coords_q, coords_t, seq_q, seq_t)
    return {
        'tm_norm_query': float(result.tm_norm_chain1),   # TM-score normalized by query length
        'tm_norm_target': float(result.tm_norm_chain2),  # TM-score normalized by target length
        'rmsd': float(result.rmsd),
        'translation': result.t.tolist(),                # translation vector
        'rotation': result.u.tolist(),                   # rotation matrix
        'aligned_seq_query': result.seqxA,
        'aligned_seq_target': result.seqyA,
        'match_line': result.seqM,
    }


def compute_alignment_coverage(match_line):
    """Compute fraction of aligned residues from match line."""
    if not match_line:
        return 0.0
    aligned = sum(1 for c in match_line if c != ' ' and c != '-')
    total = len(match_line.replace(' ', ''))
    return aligned / max(total, 1)


# ── Main analysis ──────────────────────────────────────────────────────
def main():
    print("Parsing PDB files...")
    query_chains = parse_pdb_chains(PDB_QUERY)
    target_chains = parse_pdb_chains(PDB_TARGET)

    print(f"Query (7xg4): {len(query_chains)} chains: {sorted(query_chains.keys())}")
    print(f"Target (6n40): {len(target_chains)} chains: {sorted(target_chains.keys())}")

    # Chain summary
    chain_summary = {}
    for cid, cdata in query_chains.items():
        chain_summary[f"7xg4_{cid}"] = {
            'length': len(cdata['seq']),
            'sequence': cdata['seq'][:50] + ('...' if len(cdata['seq']) > 50 else ''),
        }
    for cid, cdata in target_chains.items():
        chain_summary[f"6n40_{cid}"] = {
            'length': len(cdata['seq']),
            'sequence': cdata['seq'][:50] + ('...' if len(cdata['seq']) > 50 else ''),
        }

    with open(os.path.join(OUTPUT_DIR, "chain_summary.json"), 'w') as f:
        json.dump(chain_summary, f, indent=2)

    # ── Pairwise chain-level alignment ──────────────────────────────────
    print("\nRunning pairwise chain-level alignments...")
    pairwise_results = {}
    pairwise_matrix = {}  # (q_chain, t_chain) -> tm_norm_target

    for qcid in sorted(query_chains.keys()):
        for tcid in sorted(target_chains.keys()):
            qdata = query_chains[qcid]
            tdata = target_chains[tcid]
            key = f"{qcid}_vs_{tcid}"
            try:
                res = run_tm_align(qdata['coords'], qdata['seq'],
                                   tdata['coords'], tdata['seq'])
                res['query_chain'] = qcid
                res['target_chain'] = tcid
                res['query_length'] = len(qdata['seq'])
                res['target_length'] = len(tdata['seq'])
                res['alignment_coverage'] = compute_alignment_coverage(res['match_line'])
                pairwise_results[key] = res
                pairwise_matrix[(qcid, tcid)] = res['tm_norm_target']
                print(f"  {key}: TM(q)={res['tm_norm_query']:.4f}, TM(t)={res['tm_norm_target']:.4f}, "
                      f"RMSD={res['rmsd']:.2f}Å, cov={res['alignment_coverage']:.3f}")
            except Exception as e:
                print(f"  {key}: FAILED - {e}")
                pairwise_results[key] = {'error': str(e)}

    with open(os.path.join(OUTPUT_DIR, "pairwise_chain_alignments.json"), 'w') as f:
        # Convert non-serializable items
        serializable = {}
        for k, v in pairwise_results.items():
            sv = {}
            for kk, vv in v.items():
                if isinstance(vv, np.floating):
                    sv[kk] = float(vv)
                elif isinstance(vv, np.integer):
                    sv[kk] = int(vv)
                elif isinstance(vv, np.ndarray):
                    sv[kk] = vv.tolist()
                else:
                    sv[kk] = vv
            serializable[k] = sv
        json.dump(serializable, f, indent=2)

    # ── Complex-level alignment ─────────────────────────────────────────
    print("\nRunning complex-level alignment...")

    # Combine all protein chains (exclude nucleic acid chains I, J, K for 7xg4)
    protein_chains_query = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']
    nucleic_chains_query = ['I', 'J', 'K']

    # All chains combined
    all_q_coords = []
    all_q_seq = []
    all_q_chain_ids = []
    for cid in sorted(query_chains.keys()):
        cdata = query_chains[cid]
        all_q_coords.append(cdata['coords'])
        all_q_seq.append(cdata['seq'])
        all_q_chain_ids.extend([cid] * len(cdata['seq']))

    all_t_coords = []
    all_t_seq = []
    all_t_chain_ids = []
    for cid in sorted(target_chains.keys()):
        cdata = target_chains[cid]
        all_t_coords.append(cdata['coords'])
        all_t_seq.append(cdata['seq'])
        all_t_chain_ids.extend([cid] * len(cdata['seq']))

    # Protein-only complex alignment
    prot_q_coords = []
    prot_q_seq = []
    for cid in protein_chains_query:
        if cid in query_chains:
            cdata = query_chains[cid]
            prot_q_coords.append(cdata['coords'])
            prot_q_seq.append(cdata['seq'])

    prot_t_coords = []
    prot_t_seq = []
    for cid in sorted(target_chains.keys()):
        cdata = target_chains[cid]
        prot_t_coords.append(cdata['coords'])
        prot_t_seq.append(cdata['seq'])

    complex_results = {}

    # All-chains complex alignment
    try:
        all_q_c = np.vstack(all_q_coords)
        all_t_c = np.vstack(all_t_coords)
        all_q_s = ''.join(all_q_seq)
        all_t_s = ''.join(all_t_seq)
        res_all = run_tm_align(all_q_c, all_q_s, all_t_c, all_t_s)
        res_all['query_total_length'] = len(all_q_s)
        res_all['target_total_length'] = len(all_t_s)
        res_all['alignment_coverage'] = compute_alignment_coverage(res_all['match_line'])
        complex_results['all_chains'] = res_all
        print(f"  All chains: TM(q)={res_all['tm_norm_query']:.4f}, TM(t)={res_all['tm_norm_target']:.4f}, "
              f"RMSD={res_all['rmsd']:.2f}Å")
    except Exception as e:
        print(f"  All chains: FAILED - {e}")
        complex_results['all_chains'] = {'error': str(e)}

    # Protein-only complex alignment
    try:
        prot_q_c = np.vstack(prot_q_coords)
        prot_t_c = np.vstack(prot_t_coords)
        prot_q_s = ''.join(prot_q_seq)
        prot_t_s = ''.join(prot_t_seq)
        res_prot = run_tm_align(prot_q_c, prot_q_s, prot_t_c, prot_t_s)
        res_prot['query_total_length'] = len(prot_q_s)
        res_prot['target_total_length'] = len(prot_t_s)
        res_prot['alignment_coverage'] = compute_alignment_coverage(res_prot['match_line'])
        complex_results['protein_chains_only'] = res_prot
        print(f"  Protein chains: TM(q)={res_prot['tm_norm_query']:.4f}, TM(t)={res_prot['tm_norm_target']:.4f}, "
              f"RMSD={res_prot['rmsd']:.2f}Å")
    except Exception as e:
        print(f"  Protein chains: FAILED - {e}")
        complex_results['protein_chains_only'] = {'error': str(e)}

    with open(os.path.join(OUTPUT_DIR, "complex_alignments.json"), 'w') as f:
        serializable = {}
        for k, v in complex_results.items():
            sv = {}
            for kk, vv in v.items():
                if isinstance(vv, np.floating):
                    sv[kk] = float(vv)
                elif isinstance(vv, np.integer):
                    sv[kk] = int(vv)
                elif isinstance(vv, np.ndarray):
                    sv[kk] = vv.tolist()
                else:
                    sv[kk] = vv
            serializable[k] = sv
        json.dump(serializable, f, indent=2)

    # ── Best chain correspondence ───────────────────────────────────────
    print("\nBest chain correspondence (highest TM-score per query chain):")
    best_matches = {}
    for qcid in sorted(query_chains.keys()):
        best_tm = -1
        best_tcid = None
        best_res = None
        for tcid in sorted(target_chains.keys()):
            key = (qcid, tcid)
            if key in pairwise_matrix:
                tm = pairwise_matrix[key]
                if tm > best_tm:
                    best_tm = tm
                    best_tcid = tcid
                    best_res = pairwise_results[f"{qcid}_vs_{tcid}"]
        if best_tcid:
            best_matches[qcid] = {
                'best_target_chain': best_tcid,
                'tm_norm_target': float(best_tm),
                'tm_norm_query': float(best_res['tm_norm_query']),
                'rmsd': float(best_res['rmsd']),
                'alignment_coverage': float(best_res.get('alignment_coverage', 0)),
                'rotation': best_res['rotation'],
                'translation': best_res['translation'],
            }
            print(f"  {qcid} -> {best_tcid}: TM(t)={best_tm:.4f}, RMSD={best_res['rmsd']:.2f}Å")

    with open(os.path.join(OUTPUT_DIR, "best_chain_correspondence.json"), 'w') as f:
        json.dump(best_matches, f, indent=2)

    # ── Superimposition vectors summary ─────────────────────────────────
    superimposition_summary = {}
    for qcid, match in best_matches.items():
        rot = np.array(match['rotation'])
        trans = np.array(match['translation'])
        # Compute Euler angles from rotation matrix (ZYX convention)
        sy = np.sqrt(rot[0, 0]**2 + rot[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            x_angle = np.arctan2(rot[2, 1], rot[2, 2])
            y_angle = np.arctan2(-rot[2, 0], sy)
            z_angle = np.arctan2(rot[1, 0], rot[0, 0])
        else:
            x_angle = np.arctan2(-rot[1, 2], rot[1, 1])
            y_angle = np.arctan2(-rot[2, 0], sy)
            z_angle = 0.0

        superimposition_summary[qcid] = {
            'rotation_matrix_deg': (np.degrees([x_angle, y_angle, z_angle])).tolist(),
            'translation_vector_angstrom': match['translation'],
            'tm_score_normalized_target': match['tm_norm_target'],
            'rmsd_angstrom': match['rmsd'],
        }

    with open(os.path.join(OUTPUT_DIR, "superimposition_vectors.json"), 'w') as f:
        json.dump(superimposition_summary, f, indent=2)

    print("\nAnalysis complete. Results saved to outputs/")
    return pairwise_results, complex_results, best_matches, query_chains, target_chains


if __name__ == '__main__':
    main()
