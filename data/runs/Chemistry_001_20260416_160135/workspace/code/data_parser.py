"""
Data parsing utilities for protein (PDB) and ligand (SDF) structures.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import Descriptors, Lipinski, Crippen


def parse_pdb_file(pdb_path: str) -> Dict:
    """
    Parse a PDB file and extract structural information.
    
    Args:
        pdb_path: Path to the PDB file
        
    Returns:
        Dictionary containing parsed structure data
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    # Extract CA atoms only (as specified in task)
    ca_atoms = []
    all_atoms = []
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_info = {
                    "residue_name": residue.get_resname(),
                    "residue_id": residue.get_id()[1],
                    "chain": chain.get_id()
                }
                residues.append(res_info)
                
                for atom in residue:
                    atom_info = {
                        "atom_name": atom.get_name(),
                        "element": atom.element,
                        "coordinates": atom.get_coord().tolist(),
                        "residue_name": residue.get_resname(),
                        "residue_id": residue.get_id()[1],
                        "chain": chain.get_id()
                    }
                    all_atoms.append(atom_info)
                    
                    if atom.get_name() == "CA":
                        ca_atoms.append({
                            "residue_name": residue.get_resname(),
                            "residue_id": residue.get_id()[1],
                            "chain": chain.get_id(),
                            "coordinates": atom.get_coord().tolist()
                        })
    
    # Extract sequence from SEQRES records
    sequence = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("SEQRES"):
                parts = line.split()
                if len(parts) > 5:
                    sequence.extend(parts[5:])
    
    return {
        "source_file": pdb_path,
        "num_ca_atoms": len(ca_atoms),
        "num_total_atoms": len(all_atoms),
        "num_residues": len(residues),
        "sequence_length": len(sequence),
        "sequence": sequence[:50],  # First 50 residues
        "ca_coordinates": np.array([a["coordinates"] for a in ca_atoms]),
        "all_atoms": all_atoms,
        "residues": residues
    }


def parse_sdf_file(sdf_path: str) -> Dict:
    """
    Parse an SDF file and extract molecular information.
    
    Args:
        sdf_path: Path to the SDF file
        
    Returns:
        Dictionary containing parsed molecule data
    """
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    
    if mol is None:
        raise ValueError(f"Failed to parse SDF file: {sdf_path}")
    
    # Get conformer (3D coordinates)
    conformer = mol.GetConformer()
    
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conformer.GetAtomPosition(i)
        atoms.append({
            "atom_index": i,
            "element": atom.GetSymbol(),
            "atomic_num": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "coordinates": [pos.x, pos.y, pos.z],
            "is_aromatic": atom.GetIsAromatic(),
            "hybridization": str(atom.GetHybridization())
        })
    
    bonds = []
    for bond in mol.GetBonds():
        bonds.append({
            "begin_atom": bond.GetBeginAtomIdx(),
            "end_atom": bond.GetEndAtomIdx(),
            "bond_type": str(bond.GetBondType()),
            "is_aromatic": bond.GetIsAromatic()
        })
    
    # Compute molecular properties
    mol_weight = Descriptors.MolWt(mol)
    num_rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    num_h_donors = Lipinski.NumHDonors(mol)
    num_h_acceptors = Lipinski.NumHAcceptors(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    return {
        "source_file": sdf_path,
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "molecular_weight": mol_weight,
        "num_rotatable_bonds": num_rotatable_bonds,
        "num_h_donors": num_h_donors,
        "num_h_acceptors": num_h_acceptors,
        "logp": logp,
        "tpsa": tpsa,
        "atoms": atoms,
        "bonds": bonds,
        "coordinates": np.array([a["coordinates"] for a in atoms])
    }


def compute_center_of_mass(coordinates: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute center of mass for a set of coordinates."""
    if masses is None:
        return np.mean(coordinates, axis=0)
    return np.sum(coordinates * masses[:, np.newaxis], axis=0) / np.sum(masses)


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Compute Root Mean Square Deviation between two coordinate sets.
    
    Args:
        coords1: First set of coordinates (N, 3)
        coords2: Second set of coordinates (N, 3)
        
    Returns:
        RMSD value in Angstroms
    """
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape")
    
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def kabsch_alignment(coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align coords2 to coords1 using the Kabsch algorithm.
    
    Args:
        coords1: Reference coordinates (N, 3)
        coords2: Coordinates to align (N, 3)
        
    Returns:
        Tuple of (aligned coordinates, rotation matrix)
    """
    # Center both coordinate sets
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    
    centered1 = coords1 - centroid1
    centered2 = coords2 - centroid2
    
    # Compute covariance matrix
    cov = np.dot(centered2.T, centered1)
    
    # SVD decomposition with error handling
    try:
        U, S, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        # Return unrotated but translated coordinates if SVD fails
        return centered2 + centroid1, np.eye(3)
    
    # Compute rotation matrix
    d = np.sign(np.linalg.det(np.dot(U, Vt)))
    diag = np.diag([1, 1, d])
    rotation = np.dot(np.dot(U, diag), Vt)
    
    # Apply rotation and translation
    aligned = np.dot(centered2, rotation.T) + centroid1
    
    return aligned, rotation


if __name__ == "__main__":
    # Test parsing
    protein_data = parse_pdb_file("data/sample/2l3r/2l3r_protein.pdb")
    print(f"Protein: {protein_data['num_ca_atoms']} CA atoms, {protein_data['num_residues']} residues")
    
    ligand_data = parse_sdf_file("data/sample/2l3r/2l3r_ligand.sdf")
    print(f"Ligand: {ligand_data['num_atoms']} atoms, MW={ligand_data['molecular_weight']:.2f}")
