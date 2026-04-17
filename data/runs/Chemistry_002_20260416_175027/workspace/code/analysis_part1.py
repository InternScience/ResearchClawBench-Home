#!/usr/bin/env python3
"""
HADDOCK3-style Analysis of Barnase-Barstar Complex (1BRS)
with SKEMPI v2 Validation

Part 1: Structural Analysis and Interface Identification
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist

# Paths
BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Chemistry_002_20260416_175027'
PDB_FILE = os.path.join(BASE, 'data/1brs_AD.pdb')
SKEMPI_FILE = os.path.join(BASE, 'data/skempi_v2.csv')
OUTPUT_DIR = os.path.join(BASE, 'outputs')
IMG_DIR = os.path.join(BASE, 'report/images')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ============================================================
# 1. Parse PDB
# ============================================================
def parse_pdb(filepath):
    """Parse PDB file into atom records."""
    atoms = []
    for line in open(filepath):
        if line.startswith('ATOM'):
            atom = {
                'serial': int(line[6:11].strip()),
                'name': line[12:16].strip(),
                'altloc': line[16].strip(),
                'resname': line[17:20].strip(),
                'chain': line[21],
                'resnum': int(line[22:26].strip()),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
                'occupancy': float(line[54:60]) if line[54:60].strip() else 1.0,
                'bfactor': float(line[60:66]) if line[60:66].strip() else 0.0,
                'element': line[76:78].strip() if len(line) > 76 else ''
            }
            atoms.append(atom)
    return atoms

atoms = parse_pdb(PDB_FILE)
print(f"Parsed {len(atoms)} atoms")

# Organize by chain and residue
chain_atoms = defaultdict(list)
residues = defaultdict(lambda: defaultdict(list))
for a in atoms:
    chain_atoms[a['chain']].append(a)
    residues[a['chain']][a['resnum']].append(a)

for c in sorted(chain_atoms):
    print(f"  Chain {c}: {len(chain_atoms[c])} atoms, {len(residues[c])} residues")

# ============================================================
# 2. Interface Identification
# ============================================================
INTERFACE_CUTOFF = 5.0  # Angstroms for interface contacts
CONTACT_CUTOFF = 4.0    # Angstroms for atomic contacts

def get_coords(atom_list):
    return np.array([[a['x'], a['y'], a['z']] for a in atom_list])

def get_ca_coords(atom_list):
    cas = [a for a in atom_list if a['name'] == 'CA']
    return cas, np.array([[a['x'], a['y'], a['z']] for a in cas])

# Get coordinates for each chain
coords_A = get_coords(chain_atoms['A'])
coords_D = get_coords(chain_atoms['D'])

# Compute distance matrix between all atoms
dist_matrix = cdist(coords_A, coords_D)
print(f"\nDistance matrix shape: {dist_matrix.shape}")
print(f"Min distance: {dist_matrix.min():.2f} A")

# Identify interface residues
interface_residues = {'A': set(), 'D': set()}
contacts = []

for i, atom_a in enumerate(chain_atoms['A']):
    for j, atom_d in enumerate(chain_atoms['D']):
        d = dist_matrix[i, j]
        if d < INTERFACE_CUTOFF:
            interface_residues['A'].add(atom_a['resnum'])
            interface_residues['D'].add(atom_d['resnum'])
            if d < CONTACT_CUTOFF:
                contacts.append({
                    'chain_a': 'A', 'resnum_a': atom_a['resnum'],
                    'resname_a': atom_a['resname'], 'atom_a': atom_a['name'],
                    'chain_d': 'D', 'resnum_d': atom_d['resnum'],
                    'resname_d': atom_d['resname'], 'atom_d': atom_d['name'],
                    'distance': d
                })

print(f"\nInterface residues (5A cutoff):")
print(f"  Chain A: {len(interface_residues['A'])} residues: {sorted(interface_residues['A'])}")
print(f"  Chain D: {len(interface_residues['D'])} residues: {sorted(interface_residues['D'])}")
print(f"  Total contacts (<4A): {len(contacts)}")

# ============================================================
# 3. Classify contacts
# ============================================================
# Identify hydrogen bonds (N-O, O-N distances < 3.5A)
hbonds = []
salt_bridges = []
hydrophobic_contacts = []

POLAR_ATOMS = {'N', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'OG', 'OG1', 'OH', 'SG'}
CHARGED_POS = {'NZ', 'NH1', 'NH2', 'NE'}  # Lys, Arg
CHARGED_NEG = {'OD1', 'OD2', 'OE1', 'OE2'}  # Asp, Glu
HYDROPHOBIC_RES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}

for c in contacts:
    a_elem = c['atom_a'][0] if c['atom_a'] else ''
    d_elem = c['atom_d'][0] if c['atom_d'] else ''
    
    # Hydrogen bonds
    if c['distance'] < 3.5:
        if (a_elem in ('N', 'O') and d_elem in ('N', 'O')):
            hbonds.append(c)
    
    # Salt bridges
    if c['distance'] < 4.0:
        if (c['atom_a'] in CHARGED_POS and c['atom_d'] in CHARGED_NEG) or \
           (c['atom_a'] in CHARGED_NEG and c['atom_d'] in CHARGED_POS):
            salt_bridges.append(c)
    
    # Hydrophobic contacts
    if c['distance'] < 4.5:
        if a_elem == 'C' and d_elem == 'C':
            hydrophobic_contacts.append(c)

print(f"\nContact classification:")
print(f"  Hydrogen bonds: {len(hbonds)}")
print(f"  Salt bridges: {len(salt_bridges)}")
print(f"  Hydrophobic contacts: {len(hydrophobic_contacts)}")

# ============================================================
# 4. Per-residue contact analysis
# ============================================================
residue_contacts = defaultdict(lambda: {'total': 0, 'hbond': 0, 'salt': 0, 'hydrophobic': 0})

for c in contacts:
    key_a = f"A_{c['resnum_a']}_{c['resname_a']}"
    key_d = f"D_{c['resnum_d']}_{c['resname_d']}"
    residue_contacts[key_a]['total'] += 1
    residue_contacts[key_d]['total'] += 1

for c in hbonds:
    key_a = f"A_{c['resnum_a']}_{c['resname_a']}"
    key_d = f"D_{c['resnum_d']}_{c['resname_d']}"
    residue_contacts[key_a]['hbond'] += 1
    residue_contacts[key_d]['hbond'] += 1

for c in salt_bridges:
    key_a = f"A_{c['resnum_a']}_{c['resname_a']}"
    key_d = f"D_{c['resnum_d']}_{c['resname_d']}"
    residue_contacts[key_a]['salt'] += 1
    residue_contacts[key_d]['salt'] += 1

for c in hydrophobic_contacts:
    key_a = f"A_{c['resnum_a']}_{c['resname_a']}"
    key_d = f"D_{c['resnum_d']}_{c['resname_d']}"
    residue_contacts[key_a]['hydrophobic'] += 1
    residue_contacts[key_d]['hydrophobic'] += 1

print("\nTop interface residues by total contacts:")
sorted_res = sorted(residue_contacts.items(), key=lambda x: x[1]['total'], reverse=True)
for res, counts in sorted_res[:20]:
    print(f"  {res}: total={counts['total']}, hbond={counts['hbond']}, salt={counts['salt']}, hydro={counts['hydrophobic']}")

# Save interface data
interface_data = {
    'interface_residues_A': sorted(list(interface_residues['A'])),
    'interface_residues_D': sorted(list(interface_residues['D'])),
    'n_contacts': len(contacts),
    'n_hbonds': len(hbonds),
    'n_salt_bridges': len(salt_bridges),
    'n_hydrophobic': len(hydrophobic_contacts),
    'top_residues': {k: v for k, v in sorted_res[:20]}
}
with open(os.path.join(OUTPUT_DIR, 'interface_analysis.json'), 'w') as f:
    json.dump(interface_data, f, indent=2)

print("\nInterface analysis saved to outputs/interface_analysis.json")
