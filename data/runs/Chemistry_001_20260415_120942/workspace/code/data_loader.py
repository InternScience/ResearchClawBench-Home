"""
Data loading and preprocessing module for biomolecular complex structure prediction.
Handles protein PDB files and small molecule SDF files.
"""

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch
from torch.utils.data import Dataset


class ProteinStructure:
    """Represents a protein structure from PDB file."""
    
    def __init__(self, pdb_path):
        self.pdb_path = pdb_path
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure('protein', pdb_path)
        self.atoms = []
        self.residues = []
        self.coords = []
        self.extract_structure()
        
    def extract_structure(self):
        """Extract atomic coordinates and residue information."""
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.resname
                    res_id = residue.id[1]
                    self.residues.append({
                        'name': res_name,
                        'id': res_id,
                        'chain': chain.id
                    })
                    # Extract CA atoms for backbone representation
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        self.coords.append(ca_atom.coord)
                        self.atoms.append({
                            'name': 'CA',
                            'resname': res_name,
                            'resid': res_id,
                            'coord': ca_atom.coord
                        })
        self.coords = np.array(self.coords)
        
    def get_ca_coords(self):
        """Return CA atom coordinates."""
        return self.coords
    
    def get_sequence(self):
        """Extract amino acid sequence."""
        ppb = PPBuilder()
        sequences = []
        for pp in ppb.build_peptides(self.structure):
            sequences.append(str(pp.get_sequence()))
        return ''.join(sequences)
    
    def get_residue_features(self):
        """Get one-hot encoded residue features."""
        aa_to_idx = {
            'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
            'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
            'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
        }
        features = []
        for res in self.residues:
            feat = np.zeros(20)
            if res['name'] in aa_to_idx:
                feat[aa_to_idx[res['name']]] = 1
            features.append(feat)
        return np.array(features)


class LigandStructure:
    """Represents a small molecule ligand from SDF file."""
    
    def __init__(self, sdf_path):
        self.sdf_path = sdf_path
        self.mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if self.mol is None:
            raise ValueError(f"Failed to load molecule from {sdf_path}")
        self.atoms = []
        self.coords = []
        self.features = []
        self.extract_structure()
        
    def extract_structure(self):
        """Extract atomic coordinates and chemical features."""
        conf = self.mol.GetConformer()
        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            coord = np.array([pos.x, pos.y, pos.z])
            self.coords.append(coord)
            
            # Atom features
            atomic_num = atom.GetAtomicNum()
            hybridization = int(atom.GetHybridization())
            aromatic = int(atom.GetIsAromatic())
            degree = atom.GetDegree()
            
            self.atoms.append({
                'atomic_num': atomic_num,
                'hybridization': hybridization,
                'aromatic': aromatic,
                'degree': degree,
                'coord': coord
            })
            
            # One-hot encoding for atom type (first 100 elements)
            feat = np.zeros(100)
            if atomic_num < 100:
                feat[atomic_num] = 1
            feat = np.concatenate([feat, [hybridization, aromatic, degree]])
            self.features.append(feat)
            
        self.coords = np.array(self.coords)
        self.features = np.array(self.features)
        
    def get_coords(self):
        """Return atom coordinates."""
        return self.coords
    
    def get_atom_features(self):
        """Return atom features."""
        return self.features
    
    def get_molecular_descriptors(self):
        """Compute molecular descriptors."""
        return {
            'molecular_weight': Descriptors.MolWt(self.mol),
            'logp': Descriptors.MolLogP(self.mol),
            'hbd': Descriptors.NumHDonors(self.mol),
            'hba': Descriptors.NumHAcceptors(self.mol),
            'tpsa': Descriptors.TPSA(self.mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(self.mol)
        }


class BiomolecularComplex:
    """Represents a biomolecular complex with protein and ligand."""
    
    def __init__(self, protein_path, ligand_path):
        self.protein = ProteinStructure(protein_path)
        self.ligand = LigandStructure(ligand_path)
        self.protein_coords = self.protein.get_ca_coords()
        self.ligand_coords = self.ligand.get_coords()
        
    def get_bounding_box(self):
        """Compute bounding box of the complex."""
        all_coords = np.vstack([self.protein_coords, self.ligand_coords])
        return {
            'min': np.min(all_coords, axis=0),
            'max': np.max(all_coords, axis=0),
            'center': np.mean(all_coords, axis=0)
        }
    
    def compute_distance_matrix(self):
        """Compute pairwise distance matrix between protein CA and ligand atoms."""
        distances = np.sqrt(
            np.sum((self.protein_coords[:, None, :] - self.ligand_coords[None, :, :]) ** 2, axis=2)
        )
        return distances
    
    def get_interface_residues(self, cutoff=5.0):
        """Identify interface residues within cutoff distance."""
        distances = self.compute_distance_matrix()
        min_distances = np.min(distances, axis=1)
        interface_mask = min_distances < cutoff
        return {
            'indices': np.where(interface_mask)[0],
            'distances': min_distances[interface_mask]
        }


def load_complex(protein_path, ligand_path):
    """Load a biomolecular complex from PDB and SDF files."""
    return BiomolecularComplex(protein_path, ligand_path)


if __name__ == "__main__":
    # Test loading
    protein_path = "data/sample/2l3r/2l3r_protein.pdb"
    ligand_path = "data/sample/2l3r/2l3r_ligand.sdf"
    
    complex_data = load_complex(protein_path, ligand_path)
    print(f"Protein CA atoms: {len(complex_data.protein_coords)}")
    print(f"Ligand atoms: {len(complex_data.ligand_coords)}")
    print(f"Protein sequence length: {len(complex_data.protein.get_sequence())}")
    print(f"Interface residues: {len(complex_data.get_interface_residues(cutoff=5.0)['indices'])}")
