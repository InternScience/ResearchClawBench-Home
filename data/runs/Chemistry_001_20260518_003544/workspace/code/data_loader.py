"""
Data loader for protein-ligand complexes.
Supports PDB (CA atoms or full) and SDF.
"""
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem

def load_protein_ca(pdb_path):
    """Load CA atoms from PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    return np.array(ca_coords)

def load_ligand_sdf(sdf_path):
    """Load ligand from SDF using RDKit."""
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError("Failed to load SDF")
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    return mol, coords

if __name__ == "__main__":
    pdb_path = "data/sample/2l3r/2l3r_protein.pdb"
    sdf_path = "data/sample/2l3r/2l3r_ligand.sdf"
    ca = load_protein_ca(pdb_path)
    print(f"Protein CA shape: {ca.shape}")
    mol, lig_coords = load_ligand_sdf(sdf_path)
    print(f"Ligand coords shape: {lig_coords.shape}")
