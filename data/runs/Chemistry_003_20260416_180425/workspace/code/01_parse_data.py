#!/usr/bin/env python3
"""Parse and analyze all three datasets for the LES study."""
import numpy as np
import json
import os

WORKDIR = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Chemistry_003_20260416_180425"

def parse_xyz_extended(filepath):
    """Parse extended XYZ format files."""
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
        i += 1
        
        # Parse comment line (properties)
        comment = lines[i].strip()
        i += 1
        
        # Extract properties from comment
        props = {}
        # Parse key=value pairs
        import re
        # Find energy
        energy_match = re.search(r'energy=([-\d.eE+]+)', comment)
        energy = float(energy_match.group(1)) if energy_match else None
        
        # Find pbc
        pbc_match = re.search(r'pbc="([^"]+)"', comment)
        pbc = pbc_match.group(1) if pbc_match else None
        
        # Find charge_state
        cs_match = re.search(r'charge_state=([-\d]+)', comment)
        charge_state = int(cs_match.group(1)) if cs_match else None
        
        # Find total_charge
        tc_match = re.search(r'total_charge=([-\d]+)', comment)
        total_charge = int(tc_match.group(1)) if tc_match else None
        
        # Find true_charges
        tq_match = re.search(r'true_charges="([^"]+)"', comment)
        true_charges = None
        if tq_match:
            true_charges = np.array([float(x) for x in tq_match.group(1).split()])
        
        # Determine what columns are present
        has_forces = 'forces' in comment
        
        # Parse atom data
        species = []
        positions = []
        forces = []
        for j in range(n_atoms):
            parts = lines[i].split()
            i += 1
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if has_forces:
                forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        
        frame = {
            'n_atoms': n_atoms,
            'species': species,
            'positions': np.array(positions),
            'energy': energy,
            'pbc': pbc,
            'charge_state': charge_state,
            'total_charge': total_charge,
            'true_charges': true_charges,
        }
        if forces:
            frame['forces'] = np.array(forces)
        
        frames.append(frame)
    
    return frames

# Parse all datasets
print("=" * 60)
print("DATASET 1: random_charges.xyz")
print("=" * 60)
rc_frames = parse_xyz_extended(os.path.join(WORKDIR, "data/random_charges.xyz"))
print(f"Number of frames: {len(rc_frames)}")
print(f"Atoms per frame: {rc_frames[0]['n_atoms']}")
print(f"Species: {set(rc_frames[0]['species'])}")
print(f"Has energy: {rc_frames[0]['energy'] is not None}")
print(f"Has forces: {'forces' in rc_frames[0]}")
print(f"Has true_charges: {rc_frames[0]['true_charges'] is not None}")
print(f"PBC: {rc_frames[0]['pbc']}")
if rc_frames[0]['true_charges'] is not None:
    charges = rc_frames[0]['true_charges']
    print(f"Charges: {np.sum(charges == 1.0)} positive, {np.sum(charges == -1.0)} negative")
    print(f"Total charge: {np.sum(charges)}")

# Check positions range
all_pos = np.concatenate([f['positions'] for f in rc_frames])
print(f"Position range: [{all_pos.min():.2f}, {all_pos.max():.2f}]")

print("\n" + "=" * 60)
print("DATASET 2: charged_dimer.xyz")
print("=" * 60)
cd_frames = parse_xyz_extended(os.path.join(WORKDIR, "data/charged_dimer.xyz"))
print(f"Number of frames: {len(cd_frames)}")
print(f"Atoms per frame: {cd_frames[0]['n_atoms']}")
print(f"Species: {set(cd_frames[0]['species'])}")
print(f"Has energy: {cd_frames[0]['energy'] is not None}")
print(f"Has forces: {'forces' in cd_frames[0]}")
print(f"PBC: {cd_frames[0]['pbc']}")
energies = [f['energy'] for f in cd_frames]
print(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}]")

# Analyze dimer separations
print("\nDimer analysis:")
for idx in [0, len(cd_frames)//4, len(cd_frames)//2, 3*len(cd_frames)//4, len(cd_frames)-1]:
    f = cd_frames[idx]
    pos = f['positions']
    # First 4 atoms = dimer 1 (C + 3H), last 4 = dimer 2 (C + 3H)
    com1 = pos[:4].mean(axis=0)
    com2 = pos[4:].mean(axis=0)
    dist = np.linalg.norm(com2 - com1)
    print(f"  Frame {idx}: COM distance = {dist:.3f} Å, Energy = {f['energy']:.4f}")

print("\n" + "=" * 60)
print("DATASET 3: ag3_chargestates.xyz")
print("=" * 60)
ag_frames = parse_xyz_extended(os.path.join(WORKDIR, "data/ag3_chargestates.xyz"))
print(f"Number of frames: {len(ag_frames)}")
print(f"Atoms per frame: {ag_frames[0]['n_atoms']}")
print(f"Species: {set(ag_frames[0]['species'])}")
print(f"Has energy: {ag_frames[0]['energy'] is not None}")
print(f"Has forces: {'forces' in ag_frames[0]}")
print(f"PBC: {ag_frames[0]['pbc']}")

# Analyze charge states
cs_plus = [f for f in ag_frames if f['charge_state'] == 1]
cs_minus = [f for f in ag_frames if f['charge_state'] == -1]
print(f"Charge state +1: {len(cs_plus)} frames")
print(f"Charge state -1: {len(cs_minus)} frames")

if cs_plus:
    e_plus = [f['energy'] for f in cs_plus]
    print(f"  +1 energy range: [{min(e_plus):.4f}, {max(e_plus):.4f}]")
if cs_minus:
    e_minus = [f['energy'] for f in cs_minus]
    print(f"  -1 energy range: [{min(e_minus):.4f}, {max(e_minus):.4f}]")

# Bond lengths
print("\nBond length analysis:")
for cs_label, frames in [("charge +1", cs_plus), ("charge -1", cs_minus)]:
    if frames:
        bond_lengths = []
        for f in frames:
            pos = f['positions']
            for a in range(3):
                for b in range(a+1, 3):
                    bl = np.linalg.norm(pos[a] - pos[b])
                    bond_lengths.append(bl)
        bl_arr = np.array(bond_lengths)
        print(f"  {cs_label}: mean BL = {bl_arr.mean():.3f} ± {bl_arr.std():.3f} Å, range [{bl_arr.min():.3f}, {bl_arr.max():.3f}]")

# Save summary
summary = {
    "random_charges": {
        "n_frames": len(rc_frames),
        "n_atoms": rc_frames[0]['n_atoms'],
        "n_positive": int(np.sum(rc_frames[0]['true_charges'] == 1.0)),
        "n_negative": int(np.sum(rc_frames[0]['true_charges'] == -1.0)),
        "has_energy": rc_frames[0]['energy'] is not None,
        "has_forces": 'forces' in rc_frames[0],
        "pbc": rc_frames[0]['pbc'],
    },
    "charged_dimer": {
        "n_frames": len(cd_frames),
        "n_atoms": cd_frames[0]['n_atoms'],
        "species": list(set(cd_frames[0]['species'])),
        "energy_range": [min(energies), max(energies)],
        "has_forces": 'forces' in cd_frames[0],
    },
    "ag3_chargestates": {
        "n_frames": len(ag_frames),
        "n_atoms": ag_frames[0]['n_atoms'],
        "n_charge_plus1": len(cs_plus),
        "n_charge_minus1": len(cs_minus),
        "has_forces": 'forces' in ag_frames[0],
    }
}

with open(os.path.join(WORKDIR, "outputs/data_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSummary saved to outputs/data_summary.json")
