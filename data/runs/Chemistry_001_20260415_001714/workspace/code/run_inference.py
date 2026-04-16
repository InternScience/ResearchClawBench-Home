"""
Run BioDiffusion3D inference on the 2l3r protein-ligand complex.
Parse input data, run the model, compute metrics, and save results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
import json

from biodiffusion3d_model import BioDiffusion3D, AMINO_ACIDS, ATOM_TYPES

def parse_protein_pdb(pdb_path):
    """Parse protein PDB file and extract sequence and CA coordinates."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    residues = []
    ca_coords = []
    all_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != ' ':
                    continue
                resname = residue.get_resname()
                # Map 3-letter to 1-letter amino acid code
                aa_map = {
                    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                }
                aa_one = aa_map.get(resname, 'X')
                residues.append(aa_one)
                
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord().tolist())
                
                for atom in residue:
                    all_coords.append(atom.get_coord().tolist())
    
    return residues, np.array(ca_coords), np.array(all_coords)


def parse_ligand_sdf(sdf_path):
    """Parse ligand SDF file and extract atom info and coordinates."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = suppl[0]
    
    elements = []
    degrees = []
    charges = []
    coords = []
    
    conf = mol.GetConformer()
    
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        elements.append(symbol)
        degrees.append(atom.GetDegree())
        charges.append(atom.GetFormalCharge())
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    
    # Build adjacency matrix
    n_atoms = mol.GetNumAtoms()
    adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1
        adj[j, i] = 1
    
    return elements, degrees, charges, np.array(coords, dtype=np.float32), adj


def encode_protein_sequence(sequence):
    """Encode protein amino acid sequence as indices."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    aa_to_idx['X'] = len(AMINO_ACIDS)  # unknown
    indices = [aa_to_idx.get(aa, len(AMINO_ACIDS)) for aa in sequence]
    return torch.tensor([indices], dtype=torch.long)


def encode_molecule(elements, degrees, charges):
    """Encode molecule atoms as indices."""
    atom_to_idx = {atom: i for i, atom in enumerate(ATOM_TYPES)}
    atom_to_idx['X'] = len(ATOM_TYPES)
    
    elem_indices = [atom_to_idx.get(e, len(ATOM_TYPES)) for e in elements]
    deg_indices = [min(d, 5) for d in degrees]
    charge_vals = [float(c) for c in charges]
    
    return (torch.tensor([elem_indices], dtype=torch.long),
            torch.tensor([deg_indices], dtype=torch.long),
            torch.tensor([charge_vals], dtype=torch.float32).unsqueeze(-1))


def compute_rmsd(coords1, coords2):
    """Compute RMSD between two sets of coordinates."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=-1)))


def compute_ca_rmsd(pred_coords, gt_coords):
    """Compute CA-RMSD after optimal superposition (Kabsch algorithm)."""
    # Center both structures
    pred_centered = pred_coords - pred_coords.mean(axis=0)
    gt_centered = gt_coords - gt_coords.mean(axis=0)
    
    # Kabsch algorithm
    H = gt_centered.T @ pred_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)
    R = Vt.T @ sign_matrix @ U.T
    
    pred_aligned = pred_centered @ R.T
    rmsd = compute_rmsd(pred_aligned, gt_centered)
    return rmsd, R, pred_centered, gt_centered


def compute_ligand_rmsd(pred_coords, gt_coords):
    """Compute ligand RMSD after optimal superposition."""
    return compute_ca_rmsd(pred_coords, gt_coords)


def run_inference():
    """Run full inference pipeline on 2l3r complex."""
    print("=" * 60)
    print("BioDiffusion3D: Inference on 2L3R Protein-Ligand Complex")
    print("=" * 60)
    
    # Parse input data
    data_dir = "data/sample/2l3r"
    protein_seq, protein_ca, protein_all = parse_protein_pdb(
        os.path.join(data_dir, "2l3r_protein.pdb"))
    mol_elements, mol_degrees, mol_charges, mol_coords, mol_adj = parse_ligand_sdf(
        os.path.join(data_dir, "2l3r_ligand.sdf"))
    
    print(f"\nProtein: {len(protein_seq)} residues, {len(protein_ca)} CA atoms, {len(protein_all)} total atoms")
    print(f"Ligand (FK506): {len(mol_elements)} atoms ({sum(1 for e in mol_elements if e != 'H')} heavy)")
    print(f"Protein sequence: {''.join(protein_seq[:20])}...")
    
    # Encode inputs
    protein_idx = encode_protein_sequence(protein_seq)
    mol_elem_idx, mol_deg_idx, mol_chg = encode_molecule(mol_elements, mol_degrees, mol_charges)
    mol_adj_t = torch.tensor([mol_adj], dtype=torch.float32)
    
    # Initialize model
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = BioDiffusion3D(
        d_model=64,
        n_heads=4,
        n_encoder_layers=3,
        n_diffusion_layers=3,
        d_ff=256,
        timesteps=100,
        dropout=0.0  # No dropout at inference
    )
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Run structure prediction
    print("\nRunning diffusion-based structure prediction...")
    with torch.no_grad():
        results = model.predict_structure(
            protein_seq=protein_idx,
            mol_elements=mol_elem_idx,
            mol_degrees=mol_deg_idx,
            mol_charges=mol_chg,
            mol_adj=mol_adj_t,
            n_diffusion_steps=50
        )
    
    pred_coords = results['coords'].numpy()[0]
    confidence = results['confidence'].numpy()[0].flatten()
    trajectory = [t.numpy()[0] for t in results['trajectory']]
    attn_weights = results['attn_weights']
    pair_features = results['pair_features'].numpy()[0]
    
    n_prot = len(protein_seq)
    n_mol = len(mol_elements)
    n_total = n_prot + n_mol
    
    print(f"Predicted coordinates shape: {pred_coords.shape}")
    print(f"Total tokens: {n_total} (protein: {n_prot}, ligand: {n_mol})")
    
    # Compute metrics
    # Protein CA-RMSD
    pred_protein_coords = pred_coords[:n_prot]
    gt_protein_ca = protein_ca
    
    # Match predicted protein coords to CA atoms
    min_len = min(len(pred_protein_coords), len(gt_protein_ca))
    ca_rmsd, R, pred_centered, gt_centered = compute_ca_rmsd(
        pred_protein_coords[:min_len], gt_protein_ca[:min_len])
    
    # Ligand RMSD (heavy atoms only)
    pred_ligand_coords = pred_coords[n_prot:n_prot + n_mol]
    # Get heavy atom indices
    heavy_idx = [i for i, e in enumerate(mol_elements) if e != 'H']
    gt_ligand_heavy = mol_coords[heavy_idx]
    pred_ligand_heavy = pred_ligand_coords[heavy_idx]
    
    lig_rmsd, _, _, _ = compute_ligand_rmsd(pred_ligand_heavy, gt_ligand_heavy)
    
    print(f"\n{'='*40}")
    print(f"RESULTS")
    print(f"{'='*40}")
    print(f"Protein CA-RMSD: {ca_rmsd:.3f} Å")
    print(f"Ligand RMSD (heavy atoms): {lig_rmsd:.3f} Å")
    print(f"Mean confidence (pLDDT): {confidence.mean():.3f}")
    print(f"Min confidence: {confidence.min():.3f}")
    print(f"Max confidence: {confidence.max():.3f}")
    
    # Save results
    results_dict = {
        'protein_ca_rmsd': float(ca_rmsd),
        'ligand_rmsd_heavy': float(lig_rmsd),
        'mean_confidence': float(confidence.mean()),
        'n_protein_residues': n_prot,
        'n_ligand_atoms': n_mol,
        'n_ligand_heavy_atoms': len(heavy_idx),
        'model_parameters': n_params,
        'n_diffusion_steps': 50,
        'protein_sequence': ''.join(protein_seq),
        'ligand_elements': mol_elements
    }
    
    with open('outputs/inference_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Save predicted coordinates
    np.savez('outputs/predicted_coords.npz',
             pred_coords=pred_coords,
             pred_protein=pred_protein_coords,
             pred_ligand=pred_ligand_coords,
             gt_protein_ca=protein_ca,
             gt_ligand=mol_coords,
             confidence=confidence,
             trajectory=np.array(trajectory))
    
    # Save cross-modal attention weights
    # Average attention across layers
    if isinstance(attn_weights, list) and len(attn_weights) > 0:
        avg_attn = torch.stack([aw.mean(dim=1) for aw in attn_weights]).mean(dim=0).numpy()[0]
        np.save('outputs/cross_modal_attention.npy', avg_attn)
    
    # Save pair features (distance map)
    np.save('outputs/pair_features.npy', pair_features)
    
    print("\nResults saved to outputs/")
    return results_dict


if __name__ == "__main__":
    results = run_inference()
