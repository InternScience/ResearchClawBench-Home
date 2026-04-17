#!/usr/bin/env python3
"""
Extended Analysis: Intra-complex chain comparisons and additional metrics.
"""

import os
import sys
import json
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
import tmtools
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_002_20260416_201623'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')

def three_to_one(resname):
    resname_cap = resname.strip().capitalize()
    return protein_letters_3to1.get(resname_cap, 'X')

def get_ca_coords_and_seq(chain):
    coords = []
    seq = []
    for residue in chain.get_residues():
        if is_aa(residue, standard=True) and 'CA' in residue:
            coords.append(residue['CA'].get_vector().get_array())
            seq.append(three_to_one(residue.get_resname()))
    return np.array(coords), ''.join(seq)

def compute_contact_map(coords, threshold=8.0):
    """Compute contact map from CA coordinates."""
    n = len(coords)
    contact_map = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < threshold:
                contact_map[i, j] = True
                contact_map[j, i] = True
    return contact_map

def compute_inter_chain_contacts(structure, threshold=8.0):
    """Compute inter-chain contacts."""
    model = structure[0]
    chains = list(model.get_chains())
    contacts = {}
    
    for i, chain1 in enumerate(chains):
        coords1, seq1 = get_ca_coords_and_seq(chain1)
        if len(coords1) == 0:
            continue
        for j, chain2 in enumerate(chains):
            if j <= i:
                continue
            coords2, seq2 = get_ca_coords_and_seq(chain2)
            if len(coords2) == 0:
                continue
            
            n_contacts = 0
            min_dist = float('inf')
            for c1 in coords1:
                for c2 in coords2:
                    d = np.linalg.norm(c1 - c2)
                    min_dist = min(min_dist, d)
                    if d < threshold:
                        n_contacts += 1
            
            key = f"{chain1.get_id()}-{chain2.get_id()}"
            contacts[key] = {
                'n_contacts': n_contacts,
                'min_distance': float(min_dist),
            }
    
    return contacts

# Parse structures
parser = PDBParser(QUIET=True)
struct1 = parser.get_structure('7xg4', os.path.join(DATA_DIR, '7xg4.pdb'))
struct2 = parser.get_structure('6n40', os.path.join(DATA_DIR, '6n40.pdb'))

# ============ Intra-complex chain comparisons for 7XG4 ============
print("=" * 60)
print("Intra-complex Chain Comparisons (7XG4)")
print("=" * 60)

model1 = struct1[0]
protein_chains = []
for chain in model1.get_chains():
    coords, seq = get_ca_coords_and_seq(chain)
    if len(coords) > 5:
        protein_chains.append((chain.get_id(), coords, seq))

n = len(protein_chains)
intra_tm_matrix = np.zeros((n, n))
intra_results = {}

for i in range(n):
    for j in range(n):
        if i == j:
            intra_tm_matrix[i, j] = 1.0
            continue
        cid1, coords1, seq1 = protein_chains[i]
        cid2, coords2, seq2 = protein_chains[j]
        
        try:
            result = tmtools.tm_align(coords1, coords2, seq1, seq2)
            tm_avg = (result.tm_norm_chain1 + result.tm_norm_chain2) / 2
            intra_tm_matrix[i, j] = tm_avg
            
            key = f"{cid1}-{cid2}"
            intra_results[key] = {
                'tm_score_norm1': float(result.tm_norm_chain1),
                'tm_score_norm2': float(result.tm_norm_chain2),
                'tm_score_avg': float(tm_avg),
                'chain1_length': len(seq1),
                'chain2_length': len(seq2),
            }
            
            if i < j:
                print(f"  Chain {cid1} vs {cid2}: TM-score = {tm_avg:.4f} (len: {len(seq1)}, {len(seq2)})")
        except Exception as e:
            print(f"  Chain {cid1} vs {cid2}: Failed ({e})")

# Save intra-complex results
chain_ids = [pc[0] for pc in protein_chains]
intra_output = {
    'chain_ids': chain_ids,
    'tm_score_matrix': intra_tm_matrix.tolist(),
    'pairwise_details': intra_results,
}

with open(os.path.join(OUTPUT_DIR, 'intra_complex_alignment.json'), 'w') as f:
    json.dump(intra_output, f, indent=2)

# ============ Inter-chain contacts ============
print("\n" + "=" * 60)
print("Inter-chain Contacts (7XG4)")
print("=" * 60)

contacts = compute_inter_chain_contacts(struct1)
for key, val in sorted(contacts.items()):
    if val['n_contacts'] > 0:
        print(f"  {key}: {val['n_contacts']} contacts, min_dist = {val['min_distance']:.1f}Å")

with open(os.path.join(OUTPUT_DIR, 'inter_chain_contacts.json'), 'w') as f:
    json.dump(contacts, f, indent=2)

# ============ Per-chain structural features ============
print("\n" + "=" * 60)
print("Per-chain Structural Features")
print("=" * 60)

chain_features = {}
for cid, coords, seq in protein_chains:
    # Contact density
    contact_map = compute_contact_map(coords)
    contact_density = contact_map.sum() / (len(coords) * (len(coords) - 1)) if len(coords) > 1 else 0
    
    # End-to-end distance
    e2e_dist = np.linalg.norm(coords[0] - coords[-1]) if len(coords) > 1 else 0
    
    # Radius of gyration
    com = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - com)**2, axis=1)))
    
    # Asphericity
    centered = coords - com
    gyration_tensor = centered.T @ centered / len(coords)
    eigenvalues = np.linalg.eigvalsh(gyration_tensor)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    asphericity = eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])
    
    chain_features[cid] = {
        'n_residues': len(seq),
        'radius_of_gyration': float(rg),
        'end_to_end_distance': float(e2e_dist),
        'contact_density': float(contact_density),
        'asphericity': float(asphericity),
        'eigenvalues': eigenvalues.tolist(),
    }
    
    print(f"  Chain {cid}: Rg={rg:.1f}Å, E2E={e2e_dist:.1f}Å, contacts={contact_density:.3f}")

# Also for 6N40
model2 = struct2[0]
for chain in model2.get_chains():
    coords, seq = get_ca_coords_and_seq(chain)
    if len(coords) > 5:
        cid = chain.get_id()
        contact_map = compute_contact_map(coords)
        contact_density = contact_map.sum() / (len(coords) * (len(coords) - 1))
        e2e_dist = np.linalg.norm(coords[0] - coords[-1])
        com = coords.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - com)**2, axis=1)))
        centered = coords - com
        gyration_tensor = centered.T @ centered / len(coords)
        eigenvalues = np.linalg.eigvalsh(gyration_tensor)
        eigenvalues = np.sort(eigenvalues)[::-1]
        asphericity = eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])
        
        chain_features[f"6N40_{cid}"] = {
            'n_residues': len(seq),
            'radius_of_gyration': float(rg),
            'end_to_end_distance': float(e2e_dist),
            'contact_density': float(contact_density),
            'asphericity': float(asphericity),
            'eigenvalues': eigenvalues.tolist(),
        }
        print(f"  6N40 Chain {cid}: Rg={rg:.1f}Å, E2E={e2e_dist:.1f}Å, contacts={contact_density:.3f}")

with open(os.path.join(OUTPUT_DIR, 'chain_features.json'), 'w') as f:
    json.dump(chain_features, f, indent=2)

print("\nExtended analysis complete!")
