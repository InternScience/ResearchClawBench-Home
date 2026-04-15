import numpy as np
import re

def parse_xyz(filepath):
    frames = []
    with open(filepath) as f:
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
        header = lines[i+1].strip()
        atoms = []
        for j in range(n_atoms):
            atoms.append(lines[i+2+j].strip().split())
        frame = {'n_atoms': n_atoms, 'header': header, 'atoms': atoms}
        props = {}
        m = re.search(r'energy=([-\d.e+]+)', header)
        if m: props['energy'] = float(m.group(1))
        m = re.search(r'charge_state=(\S+)', header)
        if m: props['charge_state'] = int(m.group(1))
        m = re.search(r'total_charge=(\S+)', header)
        if m: props['total_charge'] = int(m.group(1))
        m = re.search(r'true_charges="([^"]+)"', header)
        if m: props['true_charges'] = [float(x) for x in m.group(1).split()]
        m = re.search(r'pbc="([^"]+)"', header)
        if m: props['pbc'] = m.group(1).split()
        species, positions, forces = [], [], []
        for atom in atoms:
            species.append(atom[0])
            positions.append([float(x) for x in atom[1:4]])
            if len(atom) >= 7:
                forces.append([float(x) for x in atom[4:7]])
        frame['species'] = species
        frame['positions'] = np.array(positions)
        frame['forces'] = np.array(forces) if forces else None
        frame['props'] = props
        frames.append(frame)
        i += n_atoms + 2
    return frames

if __name__ == '__main__':
    rc = parse_xyz('data/random_charges.xyz')
    cd = parse_xyz('data/charged_dimer.xyz')
    ag = parse_xyz('data/ag3_chargestates.xyz')
    print(f"random_charges: {len(rc)} frames, {rc[0]['n_atoms']} atoms")
    tc = rc[0]['props']['true_charges']
    print(f"  +1: {tc.count(1.0)}, -1: {tc.count(-1.0)}")
    print(f"  Has energy: {'energy' in rc[0]['props']}")
    print(f"  Has forces: {rc[0]['forces'] is not None}")
    print(f"\ncharged_dimer: {len(cd)} frames, {cd[0]['n_atoms']} atoms")
    en = [f['props'].get('energy', 0) for f in cd]
    print(f"  Energy: {min(en):.4f} to {max(en):.4f}")
    print(f"  Has forces: {cd[0]['forces'] is not None}")
    print(f"\nag3_chargestates: {len(ag)} frames, {ag[0]['n_atoms']} atoms")
    cs = set(f['props'].get('charge_state') for f in ag)
    print(f"  Charge states: {cs}")
    ag_pos = [f for f in ag if f['props'].get('charge_state') == 1]
    ag_neg = [f for f in ag if f['props'].get('charge_state') == -1]
    print(f"  +1: {len(ag_pos)}, -1: {len(ag_neg)}")
    for label, frames in [('+1', ag_pos), ('-1', ag_neg)]:
        bonds = []
        for f in frames:
            p = f['positions']
            bonds.append((np.linalg.norm(p[0]-p[1])+np.linalg.norm(p[0]-p[2])+np.linalg.norm(p[1]-p[2]))/3)
        print(f"  {label} bond: {np.mean(bonds):.3f}+/-{np.std(bonds):.3f}")
