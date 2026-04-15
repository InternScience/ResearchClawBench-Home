"""Parse XYZ datasets and extract structures, energies, forces, charges."""
import numpy as np
import json
import re
import os

def parse_xyz(filepath):
    """Parse extended XYZ file with properties in comment line."""
    structures = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        
        comment = lines[i+1].strip()
        
        # Parse properties from comment
        props = {}
        m = re.search(r'energy=([-\d.eE+]+)', comment)
        if m:
            props['energy'] = float(m.group(1))
        m = re.search(r'pbc="([^"]*)"', comment)
        if m:
            props['pbc'] = m.group(1).split()
        m = re.search(r'true_charges="([^"]*)"', comment)
        if m:
            props['true_charges'] = [float(x) for x in m.group(1).split()]
        m = re.search(r'charge_state=([-\d]+)', comment)
        if m:
            props['charge_state'] = int(m.group(1))
        m = re.search(r'total_charge=([-\d.eE+]+)', comment)
        if m:
            props['total_charge'] = float(m.group(1))
        
        has_forces = 'forces:R:3' in comment
        
        positions = []
        species = []
        forces = []
        
        for j in range(i+2, i+2+n_atoms):
            parts = lines[j].split()
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if has_forces:
                forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        
        struct = {
            'n_atoms': n_atoms,
            'species': species,
            'positions': np.array(positions),
            'comment': comment,
            'props': props,
        }
        if has_forces and forces:
            struct['forces'] = np.array(forces)
        
        structures.append(struct)
        i = i + 2 + n_atoms
    
    return structures


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Parse random_charges
    rc = parse_xyz(os.path.join(base, 'data/random_charges.xyz'))
    print(f"random_charges: {len(rc)} structures, {rc[0]['n_atoms']} atoms each")
    print(f"  Energy range: {[s['props'].get('energy', None) for s in rc[:3]]}")
    print(f"  True charges available: {'true_charges' in rc[0]['props']}")
    
    # Parse charged_dimer
    cd = parse_xyz(os.path.join(base, 'data/charged_dimer.xyz'))
    print(f"\ncharged_dimer: {len(cd)} structures, {cd[0]['n_atoms']} atoms each")
    energies_cd = [s['props'].get('energy', None) for s in cd]
    print(f"  Energy range: [{min(energies_cd):.4f}, {max(energies_cd):.4f}]")
    
    # Parse ag3_chargestates
    ag3 = parse_xyz(os.path.join(base, 'data/ag3_chargestates.xyz'))
    print(f"\nag3_chargestates: {len(ag3)} structures, {ag3[0]['n_atoms']} atoms each")
    cs = [s['props'].get('charge_state', None) for s in ag3]
    print(f"  Charge states: {set(cs)}")
    energies_ag3 = [s['props'].get('energy', None) for s in ag3]
    print(f"  Energy range: [{min(energies_ag3):.4f}, {max(energies_ag3):.4f}]")
    
    # Save parsed data summary
    summary = {
        'random_charges': {'n_structures': len(rc), 'n_atoms': rc[0]['n_atoms'], 
                           'has_true_charges': True, 'has_forces': 'forces' in rc[0]},
        'charged_dimer': {'n_structures': len(cd), 'n_atoms': cd[0]['n_atoms'],
                          'energy_min': float(min(energies_cd)), 'energy_max': float(max(energies_cd)),
                          'has_forces': 'forces' in cd[0]},
        'ag3_chargestates': {'n_structures': len(ag3), 'n_atoms': ag3[0]['n_atoms'],
                             'charge_states': sorted(set(cs)),
                             'energy_min': float(min(energies_ag3)), 'energy_max': float(max(energies_ag3)),
                             'has_forces': 'forces' in ag3[0]}
    }
    
    os.makedirs(os.path.join(base, 'outputs'), exist_ok=True)
    with open(os.path.join(base, 'outputs/data_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nData summary saved to outputs/data_summary.json")
