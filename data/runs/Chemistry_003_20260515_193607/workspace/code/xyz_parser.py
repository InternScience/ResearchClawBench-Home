"""
XYZ File Parser for Latent Ewald Summation Analysis
Parses .xyz files with various metadata (charges, energies, forces, etc.)
"""

import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Configuration:
    """A single atomic configuration."""
    elements: np.ndarray          # Element symbols
    positions: np.ndarray         # (N, 3) positions
    n_atoms: int
    forces: Optional[np.ndarray] = None     # (N, 3) forces
    energy: Optional[float] = None          # Total energy
    true_charges: Optional[np.ndarray] = None  # True point charges
    pbc: Optional[np.ndarray] = None        # Periodic boundary conditions
    charge_state: Optional[int] = None
    total_charge: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_xyz(filepath: str) -> List[Configuration]:
    """
    Parse an XYZ file, handling various metadata formats.
    
    Returns a list of Configuration objects.
    """
    configs = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Normalize line endings
    lines = [line.rstrip('\r\n') for line in lines]
    
    idx = 0
    while idx < len(lines):
        # Read atom count
        n_atoms = int(lines[idx].strip())
        idx += 1
        
        # Read comment line
        comment = lines[idx].strip()
        idx += 1
        
        # Parse comment line metadata
        metadata = parse_comment_line(comment)
        
        # Read atom data
        elements = []
        positions = []
        forces = []
        
        for i in range(n_atoms):
            parts = lines[idx].strip().split()
            idx += 1
            
            elements.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            # Check for forces (columns 4-6)
            if len(parts) >= 7:
                forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        
        # Create configuration
        config = Configuration(
            elements=np.array(elements),
            positions=np.array(positions),
            n_atoms=n_atoms,
            forces=np.array(forces) if forces else None,
            energy=metadata.get('energy'),
            true_charges=metadata.get('true_charges'),
            pbc=metadata.get('pbc'),
            charge_state=metadata.get('charge_state'),
            total_charge=metadata.get('total_charge'),
            metadata=metadata
        )
        configs.append(config)
    
    return configs


def parse_comment_line(comment: str) -> Dict[str, Any]:
    """Parse metadata from XYZ comment line."""
    metadata = {}
    
    # Parse energy
    energy_match = re.search(r'energy=([-\d.Ee+]+)', comment)
    if energy_match:
        metadata['energy'] = float(energy_match.group(1))
    
    # Parse true_charges
    charges_match = re.search(r'true_charges="([^"]+)"', comment)
    if charges_match:
        charges_str = charges_match.group(1)
        metadata['true_charges'] = np.array([float(x) for x in charges_str.split()])
    
    # Parse pbc
    pbc_match = re.search(r'pbc="([^"]+)"', comment)
    if pbc_match:
        pbc_str = pbc_match.group(1)
        metadata['pbc'] = np.array([x.strip() == 'T' for x in pbc_str.split()])
    
    # Parse charge_state
    cs_match = re.search(r'charge_state=(-?\d+)', comment)
    if cs_match:
        metadata['charge_state'] = int(cs_match.group(1))
    
    # Parse total_charge
    tc_match = re.search(r'total_charge=(-?\d+)', comment)
    if tc_match:
        metadata['total_charge'] = int(tc_match.group(1))
    
    # Parse Properties
    props_match = re.search(r'Properties=([^\s]+)', comment)
    if props_match:
        metadata['properties'] = props_match.group(1)
    
    return metadata


if __name__ == '__main__':
    # Test parsing
    for fname in ['random_charges.xyz', 'charged_dimer.xyz', 'ag3_chargestates.xyz']:
        configs = parse_xyz(f'data/{fname}')
        print(f'\n{fname}: {len(configs)} configurations')
        c = configs[0]
        print(f'  Atoms: {c.n_atoms}')
        print(f'  Elements: {np.unique(c.elements)}')
        print(f'  Energy: {c.energy}')
        print(f'  Forces: {c.forces is not None}')
        print(f'  True charges: {c.true_charges is not None}')
        if c.true_charges is not None:
            print(f'  Charge range: [{c.true_charges.min()}, {c.true_charges.max()}]')
            print(f'  Net charge: {c.true_charges.sum():.1f}')
        print(f'  PBC: {c.pbc}')
