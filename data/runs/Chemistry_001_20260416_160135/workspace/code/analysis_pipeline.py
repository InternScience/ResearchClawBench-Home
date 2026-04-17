"""
Main analysis pipeline for biomolecular complex structure prediction.

This script:
1. Parses input data (protein PDB and ligand SDF)
2. Runs the diffusion model to generate predictions
3. Computes evaluation metrics (RMSD)
4. Generates visualization figures
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import torch

from data_parser import parse_pdb_file, parse_sdf_file, compute_rmsd, kabsch_alignment
from diffusion_model import BiomolecularDiffusionModel, DiffusionConfig


def save_parsed_data(protein_data: Dict, ligand_data: Dict, output_dir: str):
    """Save parsed data to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save protein data (without large arrays)
    protein_out = {
        "source_file": protein_data["source_file"],
        "num_ca_atoms": protein_data["num_ca_atoms"],
        "num_total_atoms": protein_data["num_total_atoms"],
        "num_residues": protein_data["num_residues"],
        "sequence_length": protein_data["sequence_length"],
        "sequence": protein_data["sequence"],
        "residues": protein_data["residues"],
        "ca_coordinates_mean": protein_data["ca_coordinates"].mean(axis=0).tolist(),
        "ca_coordinates_std": protein_data["ca_coordinates"].std(axis=0).tolist()
    }
    with open(os.path.join(output_dir, "parsed_protein.json"), "w") as f:
        json.dump(protein_out, f, indent=2)
    
    # Save ligand data
    ligand_out = {
        "source_file": ligand_data["source_file"],
        "num_atoms": ligand_data["num_atoms"],
        "num_heavy_atoms": ligand_data["num_heavy_atoms"],
        "num_bonds": ligand_data["num_bonds"],
        "molecular_weight": ligand_data["molecular_weight"],
        "num_rotatable_bonds": ligand_data["num_rotatable_bonds"],
        "num_h_donors": ligand_data["num_h_donors"],
        "num_h_acceptors": ligand_data["num_h_acceptors"],
        "logp": ligand_data["logp"],
        "tpsa": ligand_data["tpsa"],
        "atoms": ligand_data["atoms"],
        "bonds": ligand_data["bonds"],
        "coordinates_mean": ligand_data["coordinates"].mean(axis=0).tolist(),
        "coordinates_std": ligand_data["coordinates"].std(axis=0).tolist()
    }
    with open(os.path.join(output_dir, "parsed_ligand.json"), "w") as f:
        json.dump(ligand_out, f, indent=2)
    
    print(f"Saved parsed data to {output_dir}")


def generate_data_overview(protein_data: Dict, ligand_data: Dict, 
                           output_path: str):
    """Generate data overview figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Protein CA atom distribution
    ax = axes[0, 0]
    ca_coords = protein_data["ca_coordinates"]
    ax.scatter(ca_coords[:, 0], ca_coords[:, 1], c=range(len(ca_coords)), 
               cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title(f"Protein CA Atoms (n={len(ca_coords)})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Ligand 3D structure
    ax = axes[0, 1]
    lig_coords = ligand_data["coordinates"]
    
    # Convert elements to numeric indices for coloring
    element_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'P': 5}
    element_colors = [element_map.get(a['element'], 6) for a in ligand_data['atoms']]
    
    ax.scatter(lig_coords[:, 0], lig_coords[:, 1], 
               c=element_colors,
               cmap='tab10', s=100, alpha=0.8)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title(f"Ligand Structure (n={ligand_data['num_atoms']} atoms)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Residue composition
    ax = axes[1, 0]
    residue_counts = {}
    for res in protein_data["residues"]:
        name = res["residue_name"]
        residue_counts[name] = residue_counts.get(name, 0) + 1
    
    sorted_res = sorted(residue_counts.items(), key=lambda x: -x[1])[:15]
    names = [r[0] for r in sorted_res]
    counts = [r[1] for r in sorted_res]
    
    bars = ax.bar(range(len(names)), counts, color='steelblue')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Count")
    ax.set_title("Top 15 Amino Acid Types")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Ligand atom type distribution
    ax = axes[1, 1]
    element_counts = {}
    for atom in ligand_data["atoms"]:
        elem = atom["element"]
        element_counts[elem] = element_counts.get(elem, 0) + 1
    
    elements = list(element_counts.keys())
    counts = list(element_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(elements)))
    
    wedges = ax.pie(counts, labels=elements, autopct='%1.1f%%', colors=colors)
    ax.set_title(f"Ligand Atom Composition (MW={ligand_data['molecular_weight']:.1f})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved data overview to {output_path}")


def generate_model_architecture_diagram(output_path: str):
    """Generate model architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Define components
    components = {
        "Input": (0.5, 0.9),
        "Protein\nEncoder": (0.2, 0.7),
        "Ligand\nEncoder": (0.8, 0.7),
        "Cross-\nAttention": (0.5, 0.5),
        "Denoising\nNetwork": (0.5, 0.3),
        "Output\nCoords": (0.5, 0.1)
    }
    
    # Draw boxes
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                     edgecolor='navy', linewidth=2)
    
    for name, (x, y) in components.items():
        ax.text(x, y, name, ha='center', va='center', fontsize=12, 
                fontweight='bold', bbox=box_props)
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', color='gray', linewidth=2)
    
    arrows = [
        ((0.5, 0.85), (0.2, 0.75)),  # Input -> Protein Encoder
        ((0.5, 0.85), (0.8, 0.75)),  # Input -> Ligand Encoder
        ((0.2, 0.65), (0.45, 0.55)),  # Protein Encoder -> Cross-Attention
        ((0.8, 0.65), (0.55, 0.55)),  # Ligand Encoder -> Cross-Attention
        ((0.5, 0.45), (0.5, 0.35)),   # Cross-Attention -> Denoising
        ((0.5, 0.25), (0.5, 0.15)),   # Denoising -> Output
    ]
    
    for (start, end) in arrows:
        ax.annotate('', xy=end, xytext=start, 
                    arrowprops=arrow_props)
    
    # Add timestep input
    ax.text(0.15, 0.3, "Timestep\nEmbedding", ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    ax.annotate('', xy=(0.45, 0.3), xytext=(0.2, 0.3), 
                arrowprops=dict(arrowstyle='->', color='orange', linewidth=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Biomolecular Diffusion Model Architecture", fontsize=16, pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved architecture diagram to {output_path}")


def run_prediction(model: BiomolecularDiffusionModel, 
                   protein_data: Dict, ligand_data: Dict,
                   num_samples: int = 5) -> Dict:
    """Run model predictions."""
    model.eval()
    
    with torch.no_grad():
        results = model.sample(protein_data, ligand_data, num_samples=num_samples)
    
    return results


def compute_prediction_metrics(true_coords: np.ndarray, 
                                pred_coords: np.ndarray) -> Dict:
    """Compute prediction quality metrics."""
    # Align predicted to true coordinates
    aligned_pred, rotation = kabsch_alignment(true_coords, pred_coords)
    
    # Compute RMSD
    rmsd = compute_rmsd(true_coords, aligned_pred)
    
    # Per-atom distances
    per_atom_dist = np.sqrt(np.sum((true_coords - aligned_pred) ** 2, axis=1))
    
    return {
        "rmsd": float(rmsd),
        "max_distance": float(per_atom_dist.max()),
        "min_distance": float(per_atom_dist.min()),
        "mean_distance": float(per_atom_dist.mean()),
        "std_distance": float(per_atom_dist.std()),
        "aligned_coordinates": aligned_pred.tolist()
    }


def generate_prediction_comparison(true_coords: np.ndarray,
                                   pred_coords_list: List[np.ndarray],
                                   ligand_data: Dict,
                                   output_path: str):
    """Generate prediction comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. True structure
    ax = axes[0, 0]
    element_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'P': 5}
    element_colors = [element_map.get(a['element'], 6) for a in ligand_data['atoms']]
    
    ax.scatter(true_coords[:, 0], true_coords[:, 1], 
               c=element_colors,
               cmap='tab10', s=100, alpha=0.8, label='True')
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title("Ground Truth Ligand Structure")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Best prediction overlay
    ax = axes[0, 1]
    
    # Find best prediction (lowest RMSD)
    best_idx = 0
    best_rmsd = float('inf')
    for i, pred in enumerate(pred_coords_list):
        aligned, _ = kabsch_alignment(true_coords, pred)
        rmsd = compute_rmsd(true_coords, aligned)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_idx = i
            best_aligned = aligned
    
    ax.scatter(true_coords[:, 0], true_coords[:, 1], c='green', 
               s=80, alpha=0.5, label='True')
    ax.scatter(best_aligned[:, 0], best_aligned[:, 1], c='red', 
               s=80, alpha=0.5, label=f'Predicted (RMSD={best_rmsd:.2f}Å)')
    
    # Draw lines between corresponding atoms
    for i in range(len(true_coords)):
        ax.plot([true_coords[i, 0], best_aligned[i, 0]],
                [true_coords[i, 1], best_aligned[i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title("Prediction vs Ground Truth Overlay")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. RMSD distribution across samples
    ax = axes[1, 0]
    rmsds = []
    for pred in pred_coords_list:
        aligned, _ = kabsch_alignment(true_coords, pred)
        rmsd = compute_rmsd(true_coords, aligned)
        rmsds.append(rmsd)
    
    ax.hist(rmsds, bins=10, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.axvline(np.mean(rmsds), color='red', linestyle='--', 
               label=f'Mean: {np.mean(rmsds):.2f}Å')
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"RMSD Distribution ({len(rmsds)} samples)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Per-atom error heatmap
    ax = axes[1, 1]
    errors = []
    for i in range(len(true_coords)):
        dists = []
        for pred in pred_coords_list:
            aligned, _ = kabsch_alignment(true_coords, pred)
            dist = np.sqrt(np.sum((true_coords[i] - aligned[i]) ** 2))
            dists.append(dist)
        errors.append(np.mean(dists))
    
    im = ax.imshow(np.array(errors).reshape(-1, 1), cmap='YlOrRd', aspect='auto')
    ax.set_xlabel("Error")
    ax.set_ylabel("Atom Index")
    ax.set_title("Per-Atom Prediction Error")
    plt.colorbar(im, ax=ax, label="Mean Distance (Å)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction comparison to {output_path}")
    
    return {"rmsd_mean": np.mean(rmsds), "rmsd_std": np.std(rmsds), "rmsds": rmsds}


def save_results(results: Dict, output_path: str):
    """Save prediction results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)
    print(f"Saved results to {output_path}")


def main():
    """Main analysis pipeline."""
    # Paths
    protein_path = "data/sample/2l3r/2l3r_protein.pdb"
    ligand_path = "data/sample/2l3r/2l3r_ligand.sdf"
    outputs_dir = "outputs"
    images_dir = "report/images"
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print("=" * 60)
    print("Biomolecular Complex Structure Prediction Pipeline")
    print("=" * 60)
    
    # Step 1: Parse input data
    print("\n[1/5] Parsing input data...")
    protein_data = parse_pdb_file(protein_path)
    ligand_data = parse_sdf_file(ligand_path)
    
    print(f"  Protein: {protein_data['num_ca_atoms']} CA atoms, "
          f"{protein_data['num_residues']} residues")
    print(f"  Ligand: {ligand_data['num_atoms']} atoms, "
          f"MW = {ligand_data['molecular_weight']:.2f}")
    
    # Save parsed data
    save_parsed_data(protein_data, ligand_data, outputs_dir)
    
    # Step 2: Generate data overview figure
    print("\n[2/5] Generating data overview...")
    generate_data_overview(
        protein_data, ligand_data,
        os.path.join(images_dir, "data_overview.png")
    )
    
    # Step 3: Generate architecture diagram
    print("\n[3/5] Creating model architecture diagram...")
    generate_model_architecture_diagram(
        os.path.join(images_dir, "model_architecture.png")
    )
    
    # Step 4: Run predictions
    print("\n[4/5] Running diffusion model predictions...")
    config = DiffusionConfig(
        num_timesteps=100,
        hidden_dim=128,
        num_layers=4,
        num_heads=4
    )
    
    model = BiomolecularDiffusionModel(config)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run sampling
    sample_results = run_prediction(model, protein_data, ligand_data, num_samples=10)
    
    # Extract predictions
    pred_coords_np = sample_results['samples'].numpy()
    
    # Get true coordinates
    true_coords = ligand_data['coordinates']
    
    # Compute metrics
    print("\n[5/5] Computing evaluation metrics...")
    all_metrics = []
    for i in range(pred_coords_np.shape[0]):
        metrics = compute_prediction_metrics(true_coords, pred_coords_np[i])
        all_metrics.append(metrics)
        print(f"  Sample {i+1}: RMSD = {metrics['rmsd']:.3f} Å")
    
    # Generate comparison figure
    pred_list = [pred_coords_np[i] for i in range(pred_coords_np.shape[0])]
    comparison_stats = generate_prediction_comparison(
        true_coords, pred_list, ligand_data,
        os.path.join(images_dir, "prediction_comparison.png")
    )
    
    # Save results
    results = {
        "model_config": {
            "num_timesteps": config.num_timesteps,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads
        },
        "num_samples": pred_coords_np.shape[0],
        "ligand_info": {
            "num_atoms": ligand_data["num_atoms"],
            "molecular_weight": ligand_data["molecular_weight"]
        },
        "metrics_summary": {
            "mean_rmsd": comparison_stats["rmsd_mean"],
            "std_rmsd": comparison_stats["rmsd_std"]
        },
        "per_sample_rmsds": comparison_stats["rmsds"]
    }
    save_results(results, os.path.join(outputs_dir, "prediction_results.json"))
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"  Mean RMSD: {comparison_stats['rmsd_mean']:.3f} ± {comparison_stats['rmsd_std']:.3f} Å")
    print(f"  Figures saved to: {images_dir}/")
    print(f"  Results saved to: {outputs_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
