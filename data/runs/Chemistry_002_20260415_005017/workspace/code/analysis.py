#!/usr/bin/env python3
"""
Comprehensive analysis of barnase-barstar complex (1BRS) using HADDOCK-inspired
computational approaches and SKEMPI 2.0 validation data.

This script:
1. Parses the 1BRS_AD.pdb structure
2. Identifies the protein-protein interface
3. Computes physicochemical properties of the interface
4. Analyzes SKEMPI 2.0 mutation data for barnase-barstar
5. Correlates structural features with experimental binding affinity changes
6. Generates all figures for the report
"""

import os
import sys
import csv
import math
import re
import json
import numpy as np
from collections import defaultdict, Counter

# Ensure output directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# 1. PDB Structure Parsing
# ============================================================

def parse_pdb(filepath):
    """Parse PDB file and extract atom information."""
    atoms = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom = {
                    'name': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resseq': int(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'element': line[76:78].strip() if len(line) > 76 else ''
                }
                atoms.append(atom)
    return atoms

def get_residues(atoms):
    """Group atoms by residue."""
    residues = {}
    for a in atoms:
        key = (a['chain'], a['resseq'], a['resname'])
        if key not in residues:
            residues[key] = []
        residues[key].append(a)
    return residues

def residue_centroid(atoms_list):
    """Compute centroid of residue atoms."""
    coords = np.array([(a['x'], a['y'], a['z']) for a in atoms_list])
    return coords.mean(axis=0)

def residue_cb_or_ca(atoms_list):
    """Get CB position (or CA for Gly)."""
    for a in atoms_list:
        if a['name'] == 'CB':
            return np.array([a['x'], a['y'], a['z']])
    for a in atoms_list:
        if a['name'] == 'CA':
            return np.array([a['x'], a['y'], a['z']])
    return residue_centroid(atoms_list)

# ============================================================
# 2. Interface Detection
# ============================================================

def find_interface_residues(atoms, distance_cutoff=5.0):
    """Find interface residues based on inter-chain atom distances."""
    chain_a_atoms = [a for a in atoms if a['chain'] == 'A']
    chain_d_atoms = [a for a in atoms if a['chain'] == 'D']
    
    interface_a = set()
    interface_d = set()
    
    # Build numpy arrays for efficiency
    coords_a = np.array([(a['x'], a['y'], a['z']) for a in chain_a_atoms])
    coords_d = np.array([(a['x'], a['y'], a['z']) for a in chain_d_atoms])
    
    # Find close contacts between chains
    for i, a_atom in enumerate(chain_a_atoms):
        dists = np.sqrt(np.sum((coords_d - coords_a[i])**2, axis=1))
        if np.min(dists) < distance_cutoff:
            interface_a.add((a_atom['chain'], a_atom['resseq'], a_atom['resname']))
    
    for j, d_atom in enumerate(chain_d_atoms):
        dists = np.sqrt(np.sum((coords_a - coords_d[j])**2, axis=1))
        if np.min(dists) < distance_cutoff:
            interface_d.add((d_atom['chain'], d_atom['resseq'], d_atom['resname']))
    
    return interface_a, interface_d

def compute_inter_chain_contacts(atoms, distance_cutoff=5.0):
    """Compute all inter-chain atom-atom contacts."""
    chain_a_atoms = [a for a in atoms if a['chain'] == 'A']
    chain_d_atoms = [a for a in atoms if a['chain'] == 'D']
    
    contacts = []
    coords_a = np.array([(a['x'], a['y'], a['z']) for a in chain_a_atoms])
    coords_d = np.array([(a['x'], a['y'], a['z']) for a in chain_d_atoms])
    
    for i, a_atom in enumerate(chain_a_atoms):
        dists = np.sqrt(np.sum((coords_d - coords_a[i])**2, axis=1))
        close_idx = np.where(dists < distance_cutoff)[0]
        for j in close_idx:
            d_atom = chain_d_atoms[j]
            contacts.append({
                'a_chain': a_atom['chain'], 'a_resseq': a_atom['resseq'],
                'a_resname': a_atom['resname'], 'a_atom': a_atom['name'],
                'd_chain': d_atom['chain'], 'd_resseq': d_atom['resseq'],
                'd_resname': d_atom['resname'], 'd_atom': d_atom['name'],
                'distance': dists[j]
            })
    return contacts

# ============================================================
# 3. Physicochemical Property Computation
# ============================================================

# Amino acid property dictionaries
AA_HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

AA_CHARGE = {
    'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
    'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0.5, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
}

AA_VOLUME = {
    'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5,
    'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7,
    'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7,
    'SER': 89.0, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0
}

AA_POLARITY = {
    'ALA': 0, 'ARG': 1, 'ASN': 1, 'ASP': 1, 'CYS': 0,
    'GLN': 1, 'GLU': 1, 'GLY': 0, 'HIS': 1, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 1, 'THR': 1, 'TRP': 0, 'TYR': 1, 'VAL': 0
}

def compute_buried_surface_area(atoms, interface_a, interface_d, probe_radius=1.4):
    """Estimate buried surface area using a simplified SASA approach."""
    from math import pi
    
    # Van der Waals radii
    VDW_RADII = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80
    }
    
    def get_radius(atom):
        elem = atom.get('element', '')
        if not elem:
            name = atom['name']
            if name[0] in 'CNOS':
                elem = name[0]
            elif name.startswith('H'):
                elem = 'H'
            else:
                elem = 'C'
        return VDW_RADII.get(elem, 1.7)
    
    # Simplified: compute SASA for isolated chains vs complex
    # Using a fast approximation based on neighbor counting
    all_interface = list(interface_a) + list(interface_d)
    interface_atoms = [a for a in atoms 
                       if (a['chain'], a['resseq'], a['resname']) in set(all_interface)]
    
    total_bsa = 0
    for a in interface_atoms:
        r = get_radius(a) + probe_radius
        # Count neighbors that shield this atom
        coord_a = np.array([a['x'], a['y'], a['z']])
        shielded = 0
        for b in atoms:
            if b is a:
                continue
            coord_b = np.array([b['x'], b['y'], b['z']])
            dist = np.linalg.norm(coord_a - coord_b)
            r_b = get_radius(b) + probe_radius
            if dist < r + r_b:
                shielded += 1
        # Simplified BSA contribution
        max_area = 4 * pi * r**2
        exposed_fraction = max(0, 1 - shielded * 0.05)
        total_bsa += max_area * (1 - exposed_fraction)
    
    return total_bsa

def compute_interface_properties(interface_a, interface_d, residues):
    """Compute physicochemical properties of the interface."""
    all_interface = list(interface_a) + list(interface_d)
    
    props = {
        'n_interface_residues': len(all_interface),
        'n_interface_a': len(interface_a),
        'n_interface_d': len(interface_d),
        'hydrophobicity': [],
        'charge': 0,
        'volume': 0,
        'polar_count': 0,
        'nonpolar_count': 0,
        'charged_count': 0,
        'residue_types': Counter()
    }
    
    for res_key in all_interface:
        resname = res_key[2]
        props['hydrophobicity'].append(AA_HYDROPHOBICITY.get(resname, 0))
        props['charge'] += AA_CHARGE.get(resname, 0)
        props['volume'] += AA_VOLUME.get(resname, 0)
        if AA_POLARITY.get(resname, 0):
            props['polar_count'] += 1
        else:
            props['nonpolar_count'] += 1
        if AA_CHARGE.get(resname, 0) != 0:
            props['charged_count'] += 1
        props['residue_types'][resname] += 1
    
    props['avg_hydrophobicity'] = np.mean(props['hydrophobicity']) if props['hydrophobicity'] else 0
    return props

# ============================================================
# 4. SKEMPI Data Analysis
# ============================================================

def parse_skempi(filepath):
    """Parse SKEMPI 2.0 CSV file."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            data.append(row)
    return data

def get_barnase_barstar_mutations(skempi_data):
    """Extract barnase-barstar mutation data with computed ddG."""
    R = 1.987e-3  # kcal/(mol*K)
    T = 298  # K
    
    mutations = []
    for r in skempi_data:
        pdb_id = r.get('#Pdb', '')
        if '1BRS' not in pdb_id.upper():
            continue
        
        mut_str = r.get('Mutation(s)_cleaned', '')
        single_muts = re.findall(r'([A-Z])([A-Z])(\d+)([A-Z])', mut_str)
        
        try:
            am = float(r.get('Affinity_mut_parsed', ''))
            aw = float(r.get('Affinity_wt_parsed', ''))
            if am <= 0 or aw <= 0:
                continue
            ddG = R * T * math.log(am / aw)
        except (ValueError, TypeError):
            continue
        
        # Temperature
        try:
            temp = float(r.get('Temperature', '298'))
        except:
            temp = 298
        
        location = r.get('iMutation_Location(s)', '')
        
        for wt, chain, resnum, mt in single_muts:
            mutations.append({
                'chain': chain,
                'resnum': int(resnum),
                'wt': wt,
                'mt': mt,
                'ddG': ddG,
                'location': location,
                'mut_str': mut_str,
                'temperature': temp,
                'affinity_mut': am,
                'affinity_wt': aw,
                'pdb_id': pdb_id
            })
    
    return mutations

def classify_mutation_location(mut, interface_a, interface_d):
    """Classify mutation as interface or non-interface based on structure."""
    chain = mut['chain']
    resnum = mut['resnum']
    
    if chain == 'A':
        interface_set = interface_a
    elif chain == 'D':
        interface_set = interface_d
    else:
        return 'unknown'
    
    for (c, r, rn) in interface_set:
        if c == chain and r == resnum:
            return 'interface'
    return 'non-interface'

def compute_mutation_type(mut):
    """Classify mutation type."""
    wt = mut['wt']
    mt = mut['mt']
    
    if mt == 'A':
        return 'alanine_scanning'
    elif wt == 'C' or mt == 'C':
        return 'cysteine'
    elif AA_CHARGE.get(three_to_one(wt), 0) != AA_CHARGE.get(three_to_one(mt), 0):
        return 'charge_reversal'
    else:
        return 'other'

def one_to_three(aa):
    """Convert one-letter amino acid code to three-letter."""
    mapping = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    return mapping.get(aa, 'UNK')

def three_to_one(aa3):
    """Convert three-letter amino acid code to one-letter."""
    mapping = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    return mapping.get(aa3, 'X')

# ============================================================
# 5. HADDOCK-Inspired Scoring
# ============================================================

def compute_haddock_score(atoms, interface_a, interface_d, mutations=None):
    """
    Compute a HADDOCK-inspired score for the complex.
    Based on the HADDOCK scoring function components.
    """
    # Simplified van der Waals energy (Lennard-Jones like)
    chain_a_atoms = [a for a in atoms if a['chain'] == 'A']
    chain_d_atoms = [a for a in atoms if a['chain'] == 'D']
    
    coords_a = np.array([(a['x'], a['y'], a['z']) for a in chain_a_atoms])
    coords_d = np.array([(a['x'], a['y'], a['z']) for a in chain_d_atoms])
    
    # Compute inter-chain distances (sample for efficiency)
    n_sample = min(500, len(chain_a_atoms))
    sample_idx = np.random.choice(len(chain_a_atoms), n_sample, replace=False) if len(chain_a_atoms) > n_sample else np.arange(len(chain_a_atoms))
    
    evdw = 0
    eelec = 0
    n_contacts = 0
    
    # Simplified charge assignment
    charged_atoms = {'ARG': ['NH1', 'NH2', 'NE'], 'LYS': ['NZ'],
                     'ASP': ['OD1', 'OD2'], 'GLU': ['OE1', 'OE2']}
    
    for i in sample_idx:
        a = chain_a_atoms[i]
        dists = np.sqrt(np.sum((coords_d - coords_a[i])**2, axis=1))
        close = dists < 8.0
        for j in np.where(close)[0]:
            d = dists[j]
            d_atom = chain_d_atoms[j]
            # Simplified vdW: attractive at contact distance
            if d < 5.0:
                evdw += -1.0 / d**6 + 1.0 / d**12
                n_contacts += 1
            
            # Simplified electrostatics
            a_charge = 0
            d_charge = 0
            if a['resname'] in charged_atoms and a['name'] in charged_atoms[a['resname']]:
                a_charge = 1 if a['resname'] in ['ARG', 'LYS'] else -1
            if d_atom['resname'] in charged_atoms and d_atom['name'] in charged_atoms[d_atom['resname']]:
                d_charge = 1 if d_atom['resname'] in ['ARG', 'LYS'] else -1
            
            if a_charge != 0 and d_charge != 0:
                eelec += 332 * a_charge * d_charge / (4 * d)  # simplified Coulomb
    
    return {
        'evdw_approx': evdw,
        'eelec_approx': eelec,
        'n_contacts': n_contacts
    }

def compute_air_energy(interface_a, interface_d, residues, distance_cutoff=3.0):
    """
    Compute Ambiguous Interaction Restraint (AIR) energy as in HADDOCK.
    AIR is defined between active residues of one protein and active+passive
    residues of the other protein.
    """
    # Here we treat all interface residues as "active"
    active_a = list(interface_a)
    active_d = list(interface_d)
    passive_a = []  # Would be neighbors of interface
    passive_d = []
    
    air_energy = 0
    n_restraints = 0
    
    for res_a in active_a:
        if res_a not in residues:
            continue
        atoms_a = residues[res_a]
        centroid_a = residue_cb_or_ca(atoms_a)
        
        # Compute effective distance to all active+passive on chain D
        sum_inv_r6 = 0
        for res_d in active_d + passive_d:
            if res_d not in residues:
                continue
            atoms_d = residues[res_d]
            centroid_d = residue_cb_or_ca(atoms_d)
            dist = np.linalg.norm(centroid_a - centroid_d)
            if dist > 0:
                sum_inv_r6 += 1.0 / dist**6
        
        if sum_inv_r6 > 0:
            d_eff = sum_inv_r6 ** (-1.0/6.0)
            if d_eff > distance_cutoff:
                # Harmonic + linear restraint as in HADDOCK
                violation = d_eff - distance_cutoff
                if violation < 1.0:
                    air_energy += 0.5 * violation**2
                else:
                    air_energy += violation - 0.5
                n_restraints += 1
    
    # Symmetric: D -> A
    for res_d in active_d:
        if res_d not in residues:
            continue
        atoms_d = residues[res_d]
        centroid_d = residue_cb_or_ca(atoms_d)
        
        sum_inv_r6 = 0
        for res_a in active_a + passive_a:
            if res_a not in residues:
                continue
            atoms_a = residues[res_a]
            centroid_a = residue_cb_or_ca(atoms_a)
            dist = np.linalg.norm(centroid_a - centroid_d)
            if dist > 0:
                sum_inv_r6 += 1.0 / dist**6
        
        if sum_inv_r6 > 0:
            d_eff = sum_inv_r6 ** (-1.0/6.0)
            if d_eff > distance_cutoff:
                violation = d_eff - distance_cutoff
                if violation < 1.0:
                    air_energy += 0.5 * violation**2
                else:
                    air_energy += violation - 0.5
                n_restraints += 1
    
    return air_energy, n_restraints

# ============================================================
# 6. Per-Residue Energy Decomposition
# ============================================================

def compute_per_residue_ddG(mutations):
    """Compute average ddG per residue position."""
    residue_ddG = defaultdict(list)
    for m in mutations:
        key = f"{m['chain']}_{m['resnum']}"
        residue_ddG[key].append(m['ddG'])
    
    result = {}
    for key, ddGs in residue_ddG.items():
        result[key] = {
            'mean_ddG': np.mean(ddGs),
            'std_ddG': np.std(ddGs),
            'n_mutations': len(ddGs),
            'max_ddG': max(ddGs),
            'min_ddG': min(ddGs)
        }
    return result

def compute_residue_interaction_energy(atoms, residues, interface_a, interface_d):
    """Compute per-residue interaction energy contributions."""
    residue_energy = {}
    
    all_interface = list(interface_a) + list(interface_d)
    
    for res_key in all_interface:
        chain = res_key[0]
        res_atoms = [a for a in atoms if a['chain'] == chain and a['resseq'] == res_key[1]]
        other_chain = 'D' if chain == 'A' else 'A'
        other_atoms = [a for a in atoms if a['chain'] == other_chain]
        
        if not res_atoms or not other_atoms:
            continue
        
        coords_res = np.array([(a['x'], a['y'], a['z']) for a in res_atoms])
        coords_other = np.array([(a['x'], a['y'], a['z']) for a in other_atoms])
        
        energy = 0
        n_contacts = 0
        for i, ca in enumerate(coords_res):
            dists = np.sqrt(np.sum((coords_other - ca)**2, axis=1))
            close = dists < 5.0
            for d in dists[close]:
                energy += -1.0 / d**6
                n_contacts += 1
        
        residue_energy[f"{chain}_{res_key[1]}"] = {
            'interaction_energy': energy,
            'n_contacts': n_contacts,
            'resname': res_key[2]
        }
    
    return residue_energy

# ============================================================
# 7. Main Analysis Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("HADDOCK-Inspired Analysis of Barnase-Barstar Complex")
    print("=" * 60)
    
    # Parse PDB
    print("\n1. Parsing PDB structure...")
    atoms = parse_pdb('data/1brs_AD.pdb')
    residues = get_residues(atoms)
    print(f"   Total atoms: {len(atoms)}")
    print(f"   Chain A residues: {len([k for k in residues if k[0]=='A'])}")
    print(f"   Chain D residues: {len([k for k in residues if k[0]=='D'])}")
    
    # Find interface
    print("\n2. Identifying protein-protein interface...")
    interface_a, interface_d = find_interface_residues(atoms, distance_cutoff=5.0)
    print(f"   Interface residues (Chain A): {len(interface_a)}")
    print(f"   Interface residues (Chain D): {len(interface_d)}")
    
    # List interface residues
    print("\n   Chain A interface residues:")
    for (c, r, rn) in sorted(interface_a, key=lambda x: x[1]):
        print(f"      {rn}{r}")
    
    print("\n   Chain D interface residues:")
    for (c, r, rn) in sorted(interface_d, key=lambda x: x[1]):
        print(f"      {rn}{r}")
    
    # Compute contacts
    print("\n3. Computing inter-chain contacts...")
    contacts = compute_inter_chain_contacts(atoms, distance_cutoff=5.0)
    print(f"   Total inter-chain contacts (<5Å): {len(contacts)}")
    
    # Interface properties
    print("\n4. Computing interface properties...")
    interface_props = compute_interface_properties(interface_a, interface_d, residues)
    print(f"   Total interface residues: {interface_props['n_interface_residues']}")
    print(f"   Average hydrophobicity: {interface_props['avg_hydrophobicity']:.2f}")
    print(f"   Net charge: {interface_props['charge']:.1f}")
    print(f"   Polar residues: {interface_props['polar_count']}")
    print(f"   Non-polar residues: {interface_props['nonpolar_count']}")
    print(f"   Charged residues: {interface_props['charged_count']}")
    
    # Parse SKEMPI data
    print("\n5. Parsing SKEMPI 2.0 data...")
    skempi_data = parse_skempi('data/skempi_v2.csv')
    mutations = get_barnase_barstar_mutations(skempi_data)
    print(f"   Total barnase-barstar mutations: {len(mutations)}")
    
    # Classify mutations
    for m in mutations:
        m['struct_location'] = classify_mutation_location(m, interface_a, interface_d)
    
    interface_muts = [m for m in mutations if m['struct_location'] == 'interface']
    non_interface_muts = [m for m in mutations if m['struct_location'] == 'non-interface']
    print(f"   Interface mutations: {len(interface_muts)}")
    print(f"   Non-interface mutations: {len(non_interface_muts)}")
    
    # Per-residue ddG
    print("\n6. Computing per-residue ddG...")
    residue_ddG = compute_per_residue_ddG(mutations)
    for key in sorted(residue_ddG.keys()):
        info = residue_ddG[key]
        print(f"   {key}: mean_ddG={info['mean_ddG']:.2f}, n={info['n_mutations']}")
    
    # Per-residue interaction energy
    print("\n7. Computing per-residue interaction energies...")
    residue_energy = compute_residue_interaction_energy(atoms, residues, interface_a, interface_d)
    
    # HADDOCK-inspired scoring
    print("\n8. Computing HADDOCK-inspired scores...")
    np.random.seed(42)
    haddock_scores = compute_haddock_score(atoms, interface_a, interface_d)
    print(f"   Approximate vdW energy: {haddock_scores['evdw_approx']:.2f}")
    print(f"   Approximate electrostatic energy: {haddock_scores['eelec_approx']:.2f}")
    print(f"   Number of contacts: {haddock_scores['n_contacts']}")
    
    # AIR energy
    air_energy, n_air = compute_air_energy(interface_a, interface_d, residues)
    print(f"   AIR energy: {air_energy:.2f}")
    print(f"   Number of AIR restraints: {n_air}")
    
    # Save results
    print("\n9. Saving intermediate results...")
    
    # Save interface residues
    interface_data = {
        'chain_a': [{'resnum': r[1], 'resname': r[2]} for r in sorted(interface_a, key=lambda x: x[1])],
        'chain_d': [{'resnum': r[1], 'resname': r[2]} for r in sorted(interface_d, key=lambda x: x[1])],
        'properties': {k: v for k, v in interface_props.items() if not isinstance(v, (list, Counter))}
    }
    interface_data['properties']['residue_types'] = dict(interface_props['residue_types'])
    
    with open('outputs/interface_residues.json', 'w') as f:
        json.dump(interface_data, f, indent=2)
    
    # Save mutation data
    mut_data = []
    for m in mutations:
        mut_data.append({
            'chain': m['chain'],
            'resnum': m['resnum'],
            'wt': m['wt'],
            'mt': m['mt'],
            'ddG': m['ddG'],
            'location_skempi': m['location'],
            'location_struct': m['struct_location'],
            'mut_str': m['mut_str']
        })
    with open('outputs/mutation_data.json', 'w') as f:
        json.dump(mut_data, f, indent=2)
    
    # Save per-residue ddG
    with open('outputs/residue_ddG.json', 'w') as f:
        json.dump(residue_ddG, f, indent=2)
    
    # Save residue interaction energies
    with open('outputs/residue_energy.json', 'w') as f:
        json.dump(residue_energy, f, indent=2)
    
    # Save contacts
    contacts_data = [{'a_res': c['a_resname']+str(c['a_resseq']), 
                      'd_res': c['d_resname']+str(c['d_resseq']),
                      'distance': round(c['distance'], 2)} for c in contacts[:500]]
    with open('outputs/contacts.json', 'w') as f:
        json.dump(contacts_data, f, indent=2)
    
    print("\n   All intermediate results saved to outputs/")
    print("\nAnalysis complete! Run generate_figures.py to create visualizations.")

if __name__ == '__main__':
    main()
