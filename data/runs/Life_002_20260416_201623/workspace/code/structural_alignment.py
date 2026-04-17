#!/usr/bin/env python3
"""
Protein Complex Structural Alignment Analysis
==============================================
Implements Foldseek-Multimer-inspired structural alignment between
protein complexes 7xg4 and 6n40, including:
- Chain correspondence identification
- Per-chain TM-align structural alignment
- Superimposition vectors (rotation matrix + translation)
- TM-score computation
- Structural alphabet (3Di-like) encoding
"""

import os
import sys
import json
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Superimposer, Selection
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
import tmtools
import warnings
warnings.filterwarnings('ignore')

# Paths
WORKSPACE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_002_20260416_201623'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def three_to_one(resname):
    """Convert 3-letter amino acid code to 1-letter."""
    resname_cap = resname.strip().capitalize()
    if resname_cap in protein_letters_3to1:
        return protein_letters_3to1[resname_cap]
    return 'X'


def parse_pdb(pdb_file):
    """Parse PDB file and extract chain information."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    return structure

def get_ca_coords_and_seq(chain):
    """Extract CA coordinates and sequence from a chain."""
    coords = []
    seq = []
    residues = []
    for residue in chain.get_residues():
        if is_aa(residue, standard=True):
            if 'CA' in residue:
                coords.append(residue['CA'].get_vector().get_array())
                try:
                    seq.append(three_to_one(residue.get_resname()))
                except:
                    seq.append('X')
                residues.append(residue)
    return np.array(coords), ''.join(seq), residues

def get_nucleic_ca_coords(chain):
    """Extract backbone coords from nucleic acid chain (C3' atoms)."""
    coords = []
    seq = []
    for residue in chain.get_residues():
        resname = residue.get_resname().strip()
        if resname in ['A', 'U', 'G', 'C', 'DA', 'DT', 'DG', 'DC', 'T']:
            for atom_name in ["C3'", "C3*", "P"]:
                if atom_name in residue:
                    coords.append(residue[atom_name].get_vector().get_array())
                    seq.append(resname[0] if len(resname) <= 2 else resname[-1])
                    break
    return np.array(coords) if coords else np.array([]).reshape(0,3), ''.join(seq)

def compute_tm_score_manual(coords1, coords2, alignment_map):
    """
    Compute TM-score manually following Zhang & Skolnick (2005).
    TM-score = (1/L_target) * sum_i 1/(1 + (d_i/d0)^2)
    """
    L_target = len(coords2)
    if L_target < 15:
        return 0.0
    d0 = 1.24 * (L_target - 15) ** (1.0/3.0) - 1.8
    if d0 < 0.5:
        d0 = 0.5
    
    tm_score = 0.0
    for i, j in alignment_map:
        d = np.linalg.norm(coords1[i] - coords2[j])
        tm_score += 1.0 / (1.0 + (d / d0) ** 2)
    
    tm_score /= L_target
    return tm_score

def compute_rmsd(coords1, coords2):
    """Compute RMSD between two sets of coordinates."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def kabsch_superimpose(P, Q):
    """
    Kabsch algorithm for optimal rotation matrix.
    Returns rotation matrix R and translation vector t
    such that P_aligned = R @ P + t minimizes RMSD with Q.
    """
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    
    R = Vt.T @ sign_matrix @ U.T
    t = centroid_Q - R @ centroid_P
    
    return R, t

def assign_secondary_structure(coords):
    """
    Assign secondary structure based on CA distances (simplified).
    Uses the approach from TM-align paper.
    """
    n = len(coords)
    ss = ['C'] * n  # Default: coil
    
    if n < 5:
        return ss
    
    for i in range(2, n - 2):
        # Check alpha helix pattern
        d_i_i2 = np.linalg.norm(coords[i] - coords[i-2])
        d_i_i3 = np.linalg.norm(coords[i] - coords[i+2]) if i+2 < n else 999
        d_i1_i3 = np.linalg.norm(coords[i-1] - coords[i+1]) if i+1 < n else 999
        
        # Alpha helix: CA(i)-CA(i+2) ~ 5.4-5.5 Å
        if 4.0 < d_i_i2 < 7.0 and 4.0 < d_i_i3 < 7.0:
            # Check if it's helix-like
            if d_i_i2 < 6.5 and d_i_i3 < 6.5:
                ss[i] = 'H'
        
        # Beta strand: CA(i)-CA(i+2) ~ 6.5-7.0 Å
        if 6.0 < d_i_i2 < 8.0 and 6.0 < d_i_i3 < 8.0:
            ss[i] = 'E'
    
    return ss

def compute_3di_alphabet(coords, seq):
    """
    Compute a simplified 3Di structural alphabet encoding.
    Based on Foldseek's concept of encoding tertiary interactions.
    Uses local geometry (distances, angles) to assign structural states.
    """
    n = len(coords)
    if n < 3:
        return 'X' * n
    
    alphabet = []
    for i in range(n):
        if i == 0 or i == n - 1:
            alphabet.append('X')
            continue
        
        # Local geometry features
        v1 = coords[i] - coords[i-1]
        v2 = coords[i+1] - coords[i]
        
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        
        if d1 > 0 and d2 > 0:
            cos_angle = np.dot(v1, v2) / (d1 * d2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
        else:
            angle = np.pi
        
        # Torsion angle if possible
        if i >= 2 and i < n - 1:
            v0 = coords[i-1] - coords[i-2]
            n1 = np.cross(v0, v1)
            n2 = np.cross(v1, v2)
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            if n1_norm > 0 and n2_norm > 0:
                cos_torsion = np.dot(n1, n2) / (n1_norm * n2_norm)
                cos_torsion = np.clip(cos_torsion, -1, 1)
                torsion = np.arccos(cos_torsion)
            else:
                torsion = np.pi
        else:
            torsion = np.pi
        
        # Map to 20-letter structural alphabet based on angle/torsion bins
        angle_deg = np.degrees(angle)
        torsion_deg = np.degrees(torsion)
        
        # 4 angle bins x 5 torsion bins = 20 states
        if angle_deg < 90:
            a_bin = 0
        elif angle_deg < 120:
            a_bin = 1
        elif angle_deg < 150:
            a_bin = 2
        else:
            a_bin = 3
        
        if torsion_deg < 36:
            t_bin = 0
        elif torsion_deg < 72:
            t_bin = 1
        elif torsion_deg < 108:
            t_bin = 2
        elif torsion_deg < 144:
            t_bin = 3
        else:
            t_bin = 4
        
        state_idx = a_bin * 5 + t_bin
        state_chars = 'ACDEFGHIKLMNPQRSTVWY'
        alphabet.append(state_chars[state_idx])
    
    return ''.join(alphabet)

def analyze_structure(pdb_file, name):
    """Comprehensive analysis of a PDB structure."""
    structure = parse_pdb(pdb_file)
    model = structure[0]
    
    result = {
        'name': name,
        'pdb_file': os.path.basename(pdb_file),
        'chains': {},
        'total_residues': 0,
        'total_atoms': 0,
        'protein_chains': [],
        'nucleic_chains': [],
    }
    
    for chain in model.get_chains():
        chain_id = chain.get_id()
        
        # Try protein first
        ca_coords, seq, residues = get_ca_coords_and_seq(chain)
        
        if len(ca_coords) > 0:
            ss = assign_secondary_structure(ca_coords)
            three_di = compute_3di_alphabet(ca_coords, seq)
            
            chain_info = {
                'chain_id': chain_id,
                'type': 'protein',
                'n_residues': len(seq),
                'sequence': seq,
                'ss_composition': {
                    'helix': ss.count('H') / len(ss) if ss else 0,
                    'strand': ss.count('E') / len(ss) if ss else 0,
                    'coil': ss.count('C') / len(ss) if ss else 0,
                },
                '3di_sequence': three_di,
                'ca_coords': ca_coords.tolist(),
                'center_of_mass': ca_coords.mean(axis=0).tolist(),
                'radius_of_gyration': np.sqrt(np.mean(np.sum((ca_coords - ca_coords.mean(axis=0))**2, axis=1))),
            }
            result['chains'][chain_id] = chain_info
            result['protein_chains'].append(chain_id)
            result['total_residues'] += len(seq)
        else:
            # Try nucleic acid
            na_coords, na_seq = get_nucleic_ca_coords(chain)
            if len(na_coords) > 0:
                chain_info = {
                    'chain_id': chain_id,
                    'type': 'nucleic_acid',
                    'n_residues': len(na_seq),
                    'sequence': na_seq,
                    'center_of_mass': na_coords.mean(axis=0).tolist(),
                    'radius_of_gyration': np.sqrt(np.mean(np.sum((na_coords - na_coords.mean(axis=0))**2, axis=1))),
                }
                result['chains'][chain_id] = chain_info
                result['nucleic_chains'].append(chain_id)
                result['total_residues'] += len(na_seq)
    
    result['total_atoms'] = sum(1 for _ in model.get_atoms())
    
    return result

def pairwise_chain_alignment(chain_info1, chain_info2):
    """Perform TM-align between two protein chains."""
    coords1 = np.array(chain_info1['ca_coords'])
    coords2 = np.array(chain_info2['ca_coords'])
    seq1 = chain_info1['sequence']
    seq2 = chain_info2['sequence']
    
    if len(coords1) < 5 or len(coords2) < 5:
        return None
    
    try:
        result = tmtools.tm_align(coords1, coords2, seq1, seq2)
        
        # Extract results
        tm_score1 = result.tm_norm_chain1  # normalized by chain1 length
        tm_score2 = result.tm_norm_chain2  # normalized by chain2 length
        
        # Get rotation and translation
        t = result.t
        u = result.u
        
        # Compute RMSD of aligned residues
        aligned_seq1 = result.alignment[0] if hasattr(result, 'alignment') else ''
        aligned_seq2 = result.alignment[1] if hasattr(result, 'alignment') else ''
        
        return {
            'tm_score_norm1': float(tm_score1),
            'tm_score_norm2': float(tm_score2),
            'tm_score_avg': float((tm_score1 + tm_score2) / 2),
            'rotation_matrix': u.tolist() if hasattr(u, 'tolist') else [[float(u[i][j]) for j in range(3)] for i in range(3)],
            'translation_vector': t.tolist() if hasattr(t, 'tolist') else [float(t[i]) for i in range(3)],
            'aligned_length': sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' and b != '-') if aligned_seq1 else 0,
            'alignment': [aligned_seq1, aligned_seq2],
            'chain1_length': len(seq1),
            'chain2_length': len(seq2),
        }
    except Exception as e:
        print(f"  TM-align failed: {e}")
        return None

def complex_alignment(struct1_info, struct2_info):
    """
    Perform complex-level structural alignment (Foldseek-Multimer style).
    
    Steps:
    1. Compute all pairwise chain TM-scores
    2. Find optimal chain correspondence (Hungarian algorithm)
    3. Compute overall complex TM-score
    4. Return superimposition parameters
    """
    from scipy.optimize import linear_sum_assignment
    
    protein_chains1 = struct1_info['protein_chains']
    protein_chains2 = struct2_info['protein_chains']
    
    print(f"\nComplex alignment: {struct1_info['name']} vs {struct2_info['name']}")
    print(f"  Protein chains in {struct1_info['name']}: {protein_chains1}")
    print(f"  Protein chains in {struct2_info['name']}: {protein_chains2}")
    
    # Compute all pairwise chain alignments
    pairwise_results = {}
    tm_score_matrix = np.zeros((len(protein_chains1), len(protein_chains2)))
    
    for i, c1 in enumerate(protein_chains1):
        for j, c2 in enumerate(protein_chains2):
            chain1_info = struct1_info['chains'][c1]
            chain2_info = struct2_info['chains'][c2]
            
            result = pairwise_chain_alignment(chain1_info, chain2_info)
            pairwise_results[(c1, c2)] = result
            
            if result:
                tm_score_matrix[i, j] = result['tm_score_avg']
                print(f"  Chain {c1} vs {c2}: TM-score = {result['tm_score_avg']:.4f}")
            else:
                tm_score_matrix[i, j] = 0.0
                print(f"  Chain {c1} vs {c2}: Failed")
    
    # Find optimal chain correspondence using Hungarian algorithm
    # Maximize TM-score (minimize negative)
    cost_matrix = -tm_score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    chain_correspondence = []
    for r, c in zip(row_ind, col_ind):
        chain_correspondence.append({
            'chain1': protein_chains1[r],
            'chain2': protein_chains2[c],
            'tm_score': float(tm_score_matrix[r, c]),
        })
    
    # Compute overall complex TM-score
    total_tm = sum(item['tm_score'] for item in chain_correspondence)
    n_matched = len(chain_correspondence)
    avg_tm = total_tm / n_matched if n_matched > 0 else 0
    
    # Get the best chain pair for superimposition
    best_pair = max(chain_correspondence, key=lambda x: x['tm_score'])
    best_result = pairwise_results[(best_pair['chain1'], best_pair['chain2'])]
    
    return {
        'pairwise_tm_scores': {f"{k[0]}-{k[1]}": v for k, v in pairwise_results.items() if v},
        'tm_score_matrix': tm_score_matrix.tolist(),
        'chain_correspondence': chain_correspondence,
        'overall_tm_score': avg_tm,
        'best_chain_pair': best_pair,
        'superimposition': {
            'rotation_matrix': best_result['rotation_matrix'],
            'translation_vector': best_result['translation_vector'],
        } if best_result else None,
        'n_protein_chains_1': len(protein_chains1),
        'n_protein_chains_2': len(protein_chains2),
    }

# ============ MAIN ANALYSIS ============

print("=" * 60)
print("Protein Complex Structural Alignment Analysis")
print("=" * 60)

# Step 1: Parse and analyze structures
print("\n--- Step 1: Parsing PDB structures ---")
struct1_info = analyze_structure(os.path.join(DATA_DIR, '7xg4.pdb'), '7XG4')
struct2_info = analyze_structure(os.path.join(DATA_DIR, '6n40.pdb'), '6N40')

print(f"\n7XG4 structure:")
print(f"  Total residues: {struct1_info['total_residues']}")
print(f"  Total atoms: {struct1_info['total_atoms']}")
print(f"  Protein chains: {struct1_info['protein_chains']}")
print(f"  Nucleic acid chains: {struct1_info['nucleic_chains']}")
for cid in struct1_info['protein_chains']:
    c = struct1_info['chains'][cid]
    print(f"    Chain {cid}: {c['n_residues']} residues, Rg={c['radius_of_gyration']:.1f}Å")

print(f"\n6N40 structure:")
print(f"  Total residues: {struct2_info['total_residues']}")
print(f"  Total atoms: {struct2_info['total_atoms']}")
print(f"  Protein chains: {struct2_info['protein_chains']}")
print(f"  Nucleic acid chains: {struct2_info['nucleic_chains']}")
for cid in struct2_info['protein_chains']:
    c = struct2_info['chains'][cid]
    print(f"    Chain {cid}: {c['n_residues']} residues, Rg={c['radius_of_gyration']:.1f}Å")

# Step 2: Pairwise chain alignment
print("\n--- Step 2: Pairwise Chain Alignment ---")
alignment_result = complex_alignment(struct1_info, struct2_info)

# Step 3: Save results
print("\n--- Step 3: Saving Results ---")

# Save structure summaries
summary = {
    'query': {
        'name': struct1_info['name'],
        'n_chains': len(struct1_info['chains']),
        'protein_chains': struct1_info['protein_chains'],
        'nucleic_chains': struct1_info['nucleic_chains'],
        'total_residues': struct1_info['total_residues'],
        'chain_details': {cid: {
            'type': struct1_info['chains'][cid]['type'],
            'n_residues': struct1_info['chains'][cid]['n_residues'],
            'radius_of_gyration': struct1_info['chains'][cid].get('radius_of_gyration', 0),
        } for cid in struct1_info['chains']}
    },
    'target': {
        'name': struct2_info['name'],
        'n_chains': len(struct2_info['chains']),
        'protein_chains': struct2_info['protein_chains'],
        'nucleic_chains': struct2_info['nucleic_chains'],
        'total_residues': struct2_info['total_residues'],
        'chain_details': {cid: {
            'type': struct2_info['chains'][cid]['type'],
            'n_residues': struct2_info['chains'][cid]['n_residues'],
            'radius_of_gyration': struct2_info['chains'][cid].get('radius_of_gyration', 0),
        } for cid in struct2_info['chains']}
    },
}

with open(os.path.join(OUTPUT_DIR, 'structure_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Save alignment results (without large coordinate arrays)
alignment_output = {
    'chain_correspondence': alignment_result['chain_correspondence'],
    'overall_tm_score': alignment_result['overall_tm_score'],
    'best_chain_pair': alignment_result['best_chain_pair'],
    'superimposition': alignment_result['superimposition'],
    'tm_score_matrix': alignment_result['tm_score_matrix'],
    'n_protein_chains_query': alignment_result['n_protein_chains_1'],
    'n_protein_chains_target': alignment_result['n_protein_chains_2'],
}

# Save per-pair details
pair_details = {}
for key, val in alignment_result['pairwise_tm_scores'].items():
    if val:
        pair_details[key] = {
            'tm_score_norm1': val['tm_score_norm1'],
            'tm_score_norm2': val['tm_score_norm2'],
            'tm_score_avg': val['tm_score_avg'],
            'aligned_length': val['aligned_length'],
            'chain1_length': val['chain1_length'],
            'chain2_length': val['chain2_length'],
            'rotation_matrix': val['rotation_matrix'],
            'translation_vector': val['translation_vector'],
        }
alignment_output['pairwise_details'] = pair_details

with open(os.path.join(OUTPUT_DIR, 'alignment_results.json'), 'w') as f:
    json.dump(alignment_output, f, indent=2)

# Save 3Di structural alphabet sequences
threedi_output = {}
for struct_info in [struct1_info, struct2_info]:
    threedi_output[struct_info['name']] = {}
    for cid in struct_info['protein_chains']:
        chain = struct_info['chains'][cid]
        if '3di_sequence' in chain:
            threedi_output[struct_info['name']][cid] = {
                'aa_sequence': chain['sequence'],
                '3di_sequence': chain['3di_sequence'],
                'ss_composition': chain['ss_composition'],
            }

with open(os.path.join(OUTPUT_DIR, '3di_sequences.json'), 'w') as f:
    json.dump(threedi_output, f, indent=2)

print("\nResults saved to outputs/")
print(f"  - structure_summary.json")
print(f"  - alignment_results.json")
print(f"  - 3di_sequences.json")

# Print summary
print("\n" + "=" * 60)
print("ALIGNMENT SUMMARY")
print("=" * 60)
print(f"\nQuery: {struct1_info['name']} ({len(struct1_info['protein_chains'])} protein chains)")
print(f"Target: {struct2_info['name']} ({len(struct2_info['protein_chains'])} protein chains)")
print(f"\nChain Correspondence:")
for cc in alignment_result['chain_correspondence']:
    print(f"  {cc['chain1']} -> {cc['chain2']}: TM-score = {cc['tm_score']:.4f}")
print(f"\nOverall Complex TM-score: {alignment_result['overall_tm_score']:.4f}")
print(f"Best Chain Pair: {alignment_result['best_chain_pair']['chain1']} -> {alignment_result['best_chain_pair']['chain2']}")
print(f"Best TM-score: {alignment_result['best_chain_pair']['tm_score']:.4f}")

if alignment_result['superimposition']:
    R = alignment_result['superimposition']['rotation_matrix']
    t = alignment_result['superimposition']['translation_vector']
    print(f"\nSuperimposition (best pair):")
    print(f"  Rotation matrix:")
    for row in R:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
    print(f"  Translation vector: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
