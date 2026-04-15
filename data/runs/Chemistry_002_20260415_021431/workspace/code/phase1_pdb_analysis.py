"""
Phase 1: Data Understanding and Interface Analysis
Parse PDB file, extract structural features, identify interface residues
"""
import os
import numpy as np
import json
from collections import defaultdict

def parse_pdb(pdb_path):
    """Parse PDB file and extract atom coordinates"""
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                record = {
                    'record_type': line[0:6].strip(),
                    'serial': int(line[6:11]),
                    'name': line[12:16].strip(),
                    'alt_loc': line[16],
                    'res_name': line[17:20].strip(),
                    'chain': line[21],
                    'res_seq': int(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]) if line[54:60].strip() else 1.0,
                    'temp_factor': float(line[60:66]) if line[60:66].strip() else 0.0,
                }
                atoms.append(record)
    return atoms

def get_residue_centroids(atoms):
    """Calculate centroid of each residue"""
    residues = defaultdict(lambda: {'coords': [], 'res_name': '', 'chain': '', 'res_seq': 0})
    for atom in atoms:
        key = (atom['chain'], atom['res_seq'])
        residues[key]['coords'].append([atom['x'], atom['y'], atom['z']])
        residues[key]['res_name'] = atom['res_name']
        residues[key]['chain'] = atom['chain']
        residues[key]['res_seq'] = atom['res_seq']
    
    centroids = {}
    for key, data in residues.items():
        coords = np.array(data['coords'])
        centroids[key] = {
            'centroid': np.mean(coords, axis=0),
            'res_name': data['res_name'],
            'chain': data['chain'],
            'res_seq': data['res_seq'],
            'n_atoms': len(coords)
        }
    return centroids

def find_interface_residues(centroids, chain1='A', chain2='D', cutoff=5.0):
    """Find interface residues based on distance cutoff between chains"""
    chain1_res = {k: v for k, v in centroids.items() if v['chain'] == chain1}
    chain2_res = {k: v for k, v in centroids.items() if v['chain'] == chain2}
    
    interface_pairs = []
    interface_residues = set()
    
    for k1, res1 in chain1_res.items():
        for k2, res2 in chain2_res.items():
            dist = np.linalg.norm(res1['centroid'] - res2['centroid'])
            if dist < cutoff:
                interface_pairs.append({
                    'res1': k1,
                    'res2': k2,
                    'distance': float(dist),
                    'res1_name': res1['res_name'],
                    'res2_name': res2['res_name']
                })
                interface_residues.add(k1)
                interface_residues.add(k2)
    
    return interface_pairs, interface_residues

def calculate_buried_surface_area(atoms, chain1='A', chain2='D'):
    """Simple approximation of buried surface area using contact counting"""
    chain1_atoms = [a for a in atoms if a['chain'] == chain1]
    chain2_atoms = [a for a in atoms if a['chain'] == chain2]
    
    contacts = 0
    contact_cutoff = 4.0
    for a1 in chain1_atoms:
        for a2 in chain2_atoms:
            dist = np.sqrt((a1['x']-a2['x'])**2 + (a1['y']-a2['y'])**2 + (a1['z']-a2['z'])**2)
            if dist < contact_cutoff:
                contacts += 1
    
    return contacts

if __name__ == '__main__':
    pdb_path = 'data/1brs_AD.pdb'
    atoms = parse_pdb(pdb_path)
    print(f"Total atoms parsed: {len(atoms)}")
    
    chains = set(a['chain'] for a in atoms)
    print(f"Chains found: {chains}")
    
    centroids = get_residue_centroids(atoms)
    print(f"Total residues: {len(centroids)}")
    
    # Save basic stats
    chain_stats = defaultdict(lambda: {'n_atoms': 0, 'n_residues': set()})
    for atom in atoms:
        chain_stats[atom['chain']]['n_atoms'] += 1
        chain_stats[atom['chain']]['n_residues'].add(atom['res_seq'])
    
    stats = {k: {'n_atoms': v['n_atoms'], 'n_residues': len(v['n_residues'])} for k, v in chain_stats.items()}
    print(json.dumps(stats, indent=2))
    
    # Find interface
    interface_pairs, interface_residues = find_interface_residues(centroids, 'A', 'D', cutoff=6.0)
    print(f"\nInterface residue pairs (cutoff=6.0Å): {len(interface_pairs)}")
    print(f"Unique interface residues: {len(interface_residues)}")
    
    # Save interface data
    os.makedirs('outputs', exist_ok=True)
    
    interface_data = {
        'interface_pairs': interface_pairs,
        'interface_residues': [{'chain': k[0], 'res_seq': k[1]} for k in interface_residues],
        'n_pairs': len(interface_pairs),
        'n_unique_residues': len(interface_residues)
    }
    
    with open('outputs/interface_analysis.json', 'w') as f:
        json.dump(interface_data, f, indent=2)
    
    # Calculate contacts
    contacts = calculate_buried_surface_area(atoms, 'A', 'D')
    print(f"\nInter-chain atomic contacts (<4.0Å): {contacts}")
    
    # Save all outputs
    output_data = {
        'total_atoms': len(atoms),
        'chains': list(chains),
        'chain_stats': stats,
        'n_residues': len(centroids),
        'interface_pairs': len(interface_pairs),
        'interface_residues': len(interface_residues),
        'atomic_contacts': contacts
    }
    
    with open('outputs/pdb_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\nPDB analysis complete. Results saved to outputs/")
