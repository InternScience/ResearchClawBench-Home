"""Parse all three XYZ datasets and save structured numpy arrays."""
import numpy as np
import re
import json
import os

def parse_xyz_full(filepath):
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        n_atoms = int(line)
        comment_line = lines[i+1].strip()
        props = {}
        spec_match = re.match(r'Properties=([^ ]+(?::[^ ]+)*)', comment_line)
        if spec_match:
            spec = spec_match.group(1)
            props['_spec'] = spec
            remaining = comment_line[spec_match.end():].strip()
        else:
            remaining = comment_line
        kv_pattern = re.findall(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)', remaining)
        for key, val in kv_pattern:
            val = val.strip('"')
            props[key] = val
        pbc_match = re.search(r'pbc="([^"]+)"', comment_line)
        if pbc_match:
            props['pbc'] = pbc_match.group(1)
        atoms = []
        header_offset = 2
        for j in range(n_atoms):
            atom_line = lines[i + header_offset + j].strip().split()
            atom = {
                'species': atom_line[0],
                'x': float(atom_line[1]),
                'y': float(atom_line[2]),
                'z': float(atom_line[3]),
            }
            if len(atom_line) >= 7:
                atom['fx'] = float(atom_line[4])
                atom['fy'] = float(atom_line[5])
                atom['fz'] = float(atom_line[6])
            atoms.append(atom)
        frame = {'n_atoms': n_atoms, 'comment': comment_line, 'properties': props, 'atoms': atoms}
        frames.append(frame)
        i += n_atoms + header_offset
    return frames

def frames_to_arrays(frames):
    """Convert frames to arrays: positions (n_frames, n_atoms, 3), species list, properties dict."""
    n_frames = len(frames)
    n_atoms = frames[0]['n_atoms']
    positions = np.zeros((n_frames, n_atoms, 3))
    species_list = []
    forces = np.zeros((n_frames, n_atoms, 3))
    has_forces = 'fx' in frames[0]['atoms'][0]
    
    for fi, frame in enumerate(frames):
        for ai, atom in enumerate(frame['atoms']):
            positions[fi, ai, 0] = atom['x']
            positions[fi, ai, 1] = atom['y']
            positions[fi, ai, 2] = atom['z']
            if has_forces:
                forces[fi, ai, 0] = atom.get('fx', 0)
                forces[fi, ai, 1] = atom.get('fy', 0)
                forces[fi, ai, 2] = atom.get('fz', 0)
        if fi == 0:
            species_list = [a['species'] for a in frame['atoms']]
    
    props_all = {}
    for key in frames[0]['properties']:
        vals = []
        for frame in frames:
            if key in frame['properties']:
                vals.append(frame['properties'][key])
        if vals:
            props_all[key] = vals
    
    result = {'positions': positions, 'species': species_list, 'n_frames': n_frames, 'n_atoms': n_atoms}
    if has_forces:
        result['forces'] = forces
    result['properties'] = props_all
    return result

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    
    # Dataset 1: random_charges
    frames_rc = parse_xyz_full('data/random_charges.xyz')
    data_rc = frames_to_arrays(frames_rc)
    if 'true_charges' in data_rc['properties']:
        tc_list = []
        for tc_str in data_rc['properties']['true_charges']:
            tc_list.append([float(x) for x in tc_str.split()])
        data_rc['true_charges'] = np.array(tc_list)
    np.savez('outputs/random_charges_parsed.npz', **{k: v for k, v in data_rc.items() if k != 'properties'})
    with open('outputs/random_charges_props.json', 'w') as f:
        json.dump(data_rc['properties'], f, indent=2)
    print(f"random_charges: {data_rc['n_frames']} frames, {data_rc['n_atoms']} atoms")
    
    # Dataset 2: charged_dimer
    frames_cd = parse_xyz_full('data/charged_dimer.xyz')
    data_cd = frames_to_arrays(frames_cd)
    energies = np.array([float(p) for p in data_cd['properties']['energy']])
    data_cd['energies'] = energies
    np.savez('outputs/charged_dimer_parsed.npz', **{k: v for k, v in data_cd.items() if k not in ['properties']})
    with open('outputs/charged_dimer_props.json', 'w') as f:
        json.dump(data_cd['properties'], f, indent=2)
    print(f"charged_dimer: {data_cd['n_frames']} frames, {data_cd['n_atoms']} atoms")
    
    # Dataset 3: ag3_chargestates
    frames_ag = parse_xyz_full('data/ag3_chargestates.xyz')
    data_ag = frames_to_arrays(frames_ag)
    energies = np.array([float(p) for p in data_ag['properties']['energy']])
    charge_states = np.array([int(p) for p in data_ag['properties']['charge_state']])
    total_charges = np.array([int(p) for p in data_ag['properties']['total_charge']])
    data_ag['energies'] = energies
    data_ag['charge_states'] = charge_states
    data_ag['total_charges'] = total_charges
    np.savez('outputs/ag3_chargestates_parsed.npz', **{k: v for k, v in data_ag.items() if k not in ['properties']})
    with open('outputs/ag3_chargestates_props.json', 'w') as f:
        json.dump(data_ag['properties'], f, indent=2)
    print(f"ag3_chargestates: {data_ag['n_frames']} frames, {data_ag['n_atoms']} atoms")
    
    print("Parsing complete.")
