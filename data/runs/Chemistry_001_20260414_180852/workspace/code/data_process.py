import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Paths
protein_path = 'data/sample/2l3r/2l3r_protein.pdb'
ligand_path = 'data/sample/2l3r/2l3r_ligand.sdf'

# Parse protein
parser = PDBParser(QUIET=True)
structure = parser.get_structure('protein', protein_path)
ca_coords = []
residues = []
for model in structure:
    for chain in model:
        for residue in chain:
            if 'CA' in residue:
                ca_coords.append(residue['CA'].get_coord())
                residues.append(residue.get_resname())
ca_coords = np.array(ca_coords)

# Stats
num_res = len(ca_coords)
summary = {
    'protein': {
        'num_residues': num_res,
        'ca_coords_shape': ca_coords.shape.tolist(),
        'sequence_preview': ''.join(residues[:20]) + '...'
    }
}

# Parse ligand
with open(ligand_path, 'r') as f:
    sdf_str = f.read()
mol = Chem.MolFromMolBlock(sdf_str)
if mol:
    conf = mol.GetConformer()
    ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    summary['ligand'] = {
        'num_atoms': mol.GetNumAtoms(),
        'coords_shape': ligand_coords.shape.tolist(),
        'smiles': Chem.MolToSmiles(mol)
    }
else:
    summary['ligand'] = {'error': 'Parse failed'}

# Save summary
os.makedirs('outputs', exist_ok=True)
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Data summary saved.')

# Plot protein CA chain
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(ca_coords[:,0], ca_coords[:,1], ca_coords[:,2], 'b-', linewidth=2)
ax.scatter(ca_coords[:,0], ca_coords[:,1], ca_coords[:,2], c=range(num_res), cmap='viridis', s=50)
ax.set_title('FKBP12 Protein CA Backbone')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
os.makedirs('report/images', exist_ok=True)
plt.savefig('report/images/protein_ca.png', dpi=300, bbox_inches='tight')
plt.close()

# Ligand plot
if mol:
    fig, ax = plt.subplots(figsize=(8, 6))
    rdMolDraw2D.MolToImage(mol, size=(600, 400)).savefig('report/images/ligand_2d.png', dpi=300, bbox_inches='tight')
    # 3D scatter
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(ligand_coords[:,0], ligand_coords[:,1], ligand_coords[:,2], c='r', s=100)
    ax3d.set_title('FK506 Ligand 3D Structure')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    plt.savefig('report/images/ligand_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

print('Figures saved.')
