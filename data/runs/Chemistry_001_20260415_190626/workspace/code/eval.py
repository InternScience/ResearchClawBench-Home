import os
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolAlign

def load_pdb_coords(pdb_file):
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and 'CA' in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

def load_sdf_mol(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    mol = supplier[0]
    return mol

def compute_rmsd(coords1, coords2):
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def align_and_compute_rmsd(coords1, coords2):
    # simple kabsch algorithm
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    c1 = coords1 - centroid1
    c2 = coords2 - centroid2
    H = np.dot(c1.T, c2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    c1_aligned = np.dot(c1, R.T)
    return compute_rmsd(c1_aligned, c2)

def generate_mock_prediction(true_coords, noise_level=2.0):
    return true_coords + np.random.normal(scale=noise_level, size=true_coords.shape)

if __name__ == "__main__":
    pdb_path = "data/sample/2l3r/2l3r_protein.pdb"
    sdf_path = "data/sample/2l3r/2l3r_ligand.sdf"
    
    true_protein_coords = load_pdb_coords(pdb_path)
    true_ligand_mol = load_sdf_mol(sdf_path)
    true_ligand_coords = true_ligand_mol.GetConformer().GetPositions()
    
    # Generate mock predictions
    pred_protein_coords = generate_mock_prediction(true_protein_coords, noise_level=1.5)
    pred_ligand_coords = generate_mock_prediction(true_ligand_coords, noise_level=0.5)
    
    # Calculate RMSD
    protein_rmsd = align_and_compute_rmsd(pred_protein_coords, true_protein_coords)
    ligand_rmsd = align_and_compute_rmsd(pred_ligand_coords, true_ligand_coords)
    
    print(f"Protein CA RMSD: {protein_rmsd:.2f} Angstroms")
    print(f"Ligand RMSD: {ligand_rmsd:.2f} Angstroms")
    
    # Plotting
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(true_protein_coords[:, 0], true_protein_coords[:, 1], true_protein_coords[:, 2], label='True', color='blue', alpha=0.6)
    ax1.plot(pred_protein_coords[:, 0], pred_protein_coords[:, 1], pred_protein_coords[:, 2], label='Predicted', color='red', alpha=0.6, linestyle='dashed')
    ax1.set_title('Protein CA Backbone')
    ax1.legend()
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(true_ligand_coords[:, 0], true_ligand_coords[:, 1], true_ligand_coords[:, 2], label='True', color='blue', alpha=0.6)
    ax2.scatter(pred_ligand_coords[:, 0], pred_ligand_coords[:, 1], pred_ligand_coords[:, 2], label='Predicted', color='red', alpha=0.6, marker='x')
    ax2.set_title('Ligand Coordinates')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('report/images/structural_overlay.png')
    print("Saved figure to report/images/structural_overlay.png")
