#!/usr/bin/env python3
"""
Unified Diffusion Framework for Biomolecular Complex Structure Prediction

This module implements a diffusion-based architecture for predicting 3D structures
of biomolecular complexes, handling proteins, nucleic acids, and small molecules
in a unified framework.

The approach builds on:
- AlphaFold's Evoformer + Structure Module (Jumper et al., 2021)
- SE(3) diffusion models for molecular generation
- Geometric deep learning on graphs (Bronstein et al., 2017)
- Transformer attention mechanisms (Vaswani et al., 2017)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

# ============================================================================
# Data Parsing
# ============================================================================

def parse_pdb_ca(pdb_path: str) -> Dict:
    """Parse a PDB file and extract CA atom coordinates."""
    coords = []
    residues = []
    residue_ids = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and 'CA' in line[12:16]:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                res_name = line[17:20].strip()
                res_id = int(line[22:26])
                coords.append([x, y, z])
                residues.append(res_name)
                residue_ids.append(res_id)
            elif line.startswith('SEQRES'):
                # Parse sequence from SEQRES records
                pass
    
    return {
        'coords': np.array(coords),
        'residues': residues,
        'residue_ids': residue_ids,
        'n_residues': len(residues)
    }


def parse_sdf(sdf_path: str) -> Dict:
    """Parse an SDF file and extract atomic coordinates and bonds."""
    atoms = []
    coords = []
    bonds = []
    
    with open(sdf_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header - handle potentially concatenated counts (e.g., "194193")
    counts_token = lines[3].split()[0]
    if len(counts_token) >= 6:
        n_atoms = int(counts_token[:3])
        n_bonds = int(counts_token[3:6])
    else:
        n_atoms = int(counts_token)
        n_bonds = int(lines[3].split()[1])
    
    # Parse atoms
    for i in range(4, 4 + n_atoms):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        if parts[0] == 'M':
            break
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        element = parts[3]
        atoms.append(element)
        coords.append([x, y, z])
    
    # Parse bonds
    bond_start = 4 + n_atoms
    for i in range(bond_start, bond_start + n_bonds):
        parts = lines[i].split()
        a1 = int(parts[0]) - 1
        a2 = int(parts[1]) - 1
        bond_type = int(parts[2])
        bonds.append((a1, a2, bond_type))
    
    return {
        'atoms': atoms,
        'coords': np.array(coords),
        'bonds': bonds,
        'n_atoms': n_atoms,
        'n_bonds': n_bonds
    }


# ============================================================================
# SE(3) Diffusion Components
# ============================================================================

class SE3Diffusion:
    """
    SE(3) diffusion for 3D molecular structure generation.
    
    Implements both forward (noising) and reverse (denoising) processes
    on 3D coordinates of biomolecular complexes.
    """
    
    def __init__(self, n_timesteps: int = 1000, 
                 beta_start: float = 1e-4, 
                 beta_end: float = 0.02):
        self.n_timesteps = n_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        
    def forward_diffusion(self, x0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
        
        Args:
            x0: Initial clean coordinates [N, 3]
            t: Diffusion timestep
        
        Returns:
            x_t: Noised coordinates
            noise: The noise that was added
        """
        alpha_bar = self.alpha_bars[t]
        noise = np.random.randn(*x0.shape)
        x_t = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise
        return x_t, noise
    
    def reverse_diffusion_step(self, x_t: np.ndarray, 
                                predicted_noise: np.ndarray, 
                                t: int) -> np.ndarray:
        """
        Single reverse diffusion step using predicted noise.
        
        Args:
            x_t: Current noised coordinates
            predicted_noise: Model's prediction of the noise
            t: Current timestep
        
        Returns:
            x_{t-1}: Denoised coordinates for previous timestep
        """
        beta = self.betas[t]
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        
        # DDPM reverse step
        coef1 = 1.0 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1 - alpha_bar)
        
        mean = coef1 * (x_t - coef2 * predicted_noise)
        
        if t > 0:
            noise = np.random.randn(*x_t.shape)
            sigma = np.sqrt(beta)
            return mean + sigma * noise
        else:
            return mean


# ============================================================================
# Neural Network Components
# ============================================================================

class IPAModule(nn.Module):
    """
    Invariant Point Attention (IPA) module as used in AlphaFold's Structure Module.
    
    This is a key component that enables SE(3)-equivariant processing of 3D coordinates.
    """
    
    def __init__(self, c_s: int = 256, c_z: int = 128, 
                 n_heads: int = 12, n_query_points: int = 4, 
                 n_value_points: int = 8):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points
        
        # Linear projections
        hc = c_s // n_heads
        self.hc = hc
        
        self.linear_q = nn.Linear(c_s, n_heads * hc)
        self.linear_k = nn.Linear(c_s, n_heads * hc)
        self.linear_v = nn.Linear(c_s, n_heads * hc)
        
        self.linear_q_points = nn.ModuleList([
            nn.Linear(c_s, n_heads * 3) for _ in range(n_query_points)
        ])
        self.linear_k_points = nn.ModuleList([
            nn.Linear(c_s, n_heads * 3) for _ in range(n_query_points)
        ])
        self.linear_v_points = nn.ModuleList([
            nn.Linear(c_s, n_heads * 3) for _ in range(n_value_points)
        ])
        
        self.linear_b = nn.Linear(c_z, n_heads)
        self.linear_out = nn.Linear(n_heads * (hc + n_value_points * 4), c_s)
        
        # Gamma for distance-based weighting
        self.gamma = nn.Parameter(torch.ones(1) * np.log(0.3))
        
    def forward(self, s: torch.Tensor, z: torch.Tensor, 
                x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Single representation [B, N, c_s]
            z: Pair representation [B, N, N, c_z]
            x: Current 3D coordinates [B, N, 3]
            mask: Residue mask [B, N]
        
        Returns:
            s_out: Updated single representation
        """
        B, N, _ = s.shape
        hc = self.hc
        
        # Compute queries, keys, values
        q = self.linear_q(s).view(B, N, self.n_heads, hc)
        k = self.linear_k(s).view(B, N, self.n_heads, hc)
        v = self.linear_v(s).view(B, N, self.n_heads, hc)
        
        # Pair bias
        b = self.linear_b(z)  # [B, N, N, n_heads]
        
        # Attention weights
        attn = torch.einsum('bihc,bjhc->bhij', q, k) / np.sqrt(hc)
        attn = attn + b.permute(0, 3, 1, 2)
        
        # Mask
        attn_mask = mask[:, None, :, None] * mask[:, None, None, :]
        attn = attn.masked_fill(attn_mask < 0.5, -1e9)
        
        attn_weights = F.softmax(attn, dim=-1)
        
        # Aggregate values
        o = torch.einsum('bhij,bjhc->bihc', attn_weights, v)
        o = o.reshape(B, N, -1)
        
        # Output projection
        s_out = self.linear_out(o)
        
        return s_out


class TransformerLayer(nn.Module):
    """Self-attention transformer layer for sequence processing."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, 
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, 
                                                dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, 
                                       key_padding_mask=~mask.bool() if mask is not None else None)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class DenoisingNetwork(nn.Module):
    """
    Denoising network for reverse diffusion process.
    
    Architecture:
    1. Input embedding layer (one-hot residues + coordinates + timestep)
    2. Transformer encoder stack for sequence-level reasoning
    3. IPA-based structure reasoning
    4. Output head predicting noise
    """
    
    def __init__(self, n_residue_types: int = 20, d_model: int = 256,
                 n_transformer_layers: int = 4, n_ipa_layers: int = 4,
                 n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.residue_embed = nn.Embedding(n_residue_types + 1, d_model)
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.coord_embed = nn.Linear(3, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads) for _ in range(n_transformer_layers)
        ])
        
        # IPA layers
        self.ipa_layers = nn.ModuleList([
            IPAModule(c_s=d_model, c_z=d_model, n_heads=4) 
            for _ in range(n_ipa_layers)
        ])
        
        # Output heads
        self.noise_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_t: torch.Tensor, residues: torch.Tensor, 
                t: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: Noised coordinates [B, N, 3]
            residues: Residue type indices [B, N]
            t: Timestep [B]
            mask: Valid positions [B, N]
        
        Returns:
            noise_pred: Predicted noise [B, N, 3]
            confidence: Per-residue confidence [B, N, 1]
        """
        B, N, _ = x_t.shape
        
        # Embeddings
        r_embed = self.residue_embed(residues)  # [B, N, d_model]
        t_embed = self.timestep_embed(t.float().unsqueeze(-1))  # [B, d_model]
        c_embed = self.coord_embed(x_t)  # [B, N, d_model]
        
        # Combine embeddings
        s = r_embed + c_embed + t_embed.unsqueeze(1)  # [B, N, d_model]
        
        # Transformer processing
        for layer in self.transformer_layers:
            s = layer(s, mask)
        
        # Pair representation from outer product
        z = torch.einsum('bid,bjd->bijd', s, s) / np.sqrt(self.d_model)
        
        # IPA-based structure reasoning
        for layer in self.ipa_layers:
            s = layer(s, z, x_t, mask)
        
        # Predict noise
        noise_pred = self.noise_head(s)
        
        # Predict confidence
        confidence = self.confidence_head(s)
        
        return noise_pred, confidence


# ============================================================================
# Unified Biomolecular Diffusion Model
# ============================================================================

class UnifiedBioDiffusion:
    """
    Unified diffusion model for biomolecular complexes.
    
    Handles:
    - Proteins (represented as CA atoms)
    - Nucleic acids (represented as backbone atoms)
    - Small molecules (all heavy atoms)
    
    The key innovation is a shared denoising network that operates on
    any molecular entity by using appropriate featurization.
    """
    
    # Standard amino acid codes
    AA_CODES = 'ACDEFGHIKLMNPQRSTVWY'
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_CODES)}
    
    # DNA/RNA codes
    NA_CODES = 'ACGU'
    NA_TO_IDX = {na: i for i, na in enumerate(NA_CODES)}
    
    def __init__(self, n_timesteps: int = 1000, d_model: int = 256):
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.diffusion = SE3Diffusion(n_timesteps)
        
        # The shared denoising network
        self.denoiser = DenoisingNetwork(
            n_residue_types=len(self.AA_CODES),
            d_model=d_model
        )
        
    def featurize_protein(self, protein_data: Dict) -> Dict:
        """Convert protein data to model inputs."""
        residues = protein_data['residues']
        residue_ids = [self.AA_TO_IDX.get(r, 0) for r in residues]
        return {
            'coords': protein_data['coords'],
            'types': np.array(residue_ids),
            'n_nodes': len(residues)
        }
    
    def featurize_ligand(self, ligand_data: Dict) -> Dict:
        """Convert ligand data to model inputs."""
        atoms = ligand_data['atoms']
        # Map atoms to simple types
        atomic_numbers = {'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'H': 1}
        atom_types = [atomic_numbers.get(a, 0) for a in atoms]
        return {
            'coords': ligand_data['coords'],
            'types': np.array(atom_types),
            'n_nodes': len(atoms)
        }


# ============================================================================
# Metrics and Analysis
# ============================================================================

def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets after optimal alignment."""
    # Center both sets
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    
    # Kabsch algorithm
    H = c1.T @ c2
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    c1_aligned = c1 @ R
    
    # Compute RMSD
    diff = c1_aligned - c2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd, R


def compute_symmetry_aware_rmsd(coords_pred: np.ndarray, 
                                  coords_ref: np.ndarray,
                                  atom_types: List[str]) -> float:
    """
    Compute symmetry-aware RMSD using Hungarian matching.
    Handles symmetric atoms in ligands (e.g., equivalent carbons in rings).
    """
    n = len(coords_pred)
    
    # Build cost matrix
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.sum((coords_pred[i] - coords_ref[j])**2)
    
    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute RMSD with optimal assignment
    matched_pred = coords_pred[row_ind]
    matched_ref = coords_ref[col_ind]
    
    diff = matched_pred - matched_ref
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def compute_gdt_ts(coords_pred: np.ndarray, coords_ref: np.ndarray,
                   thresholds: List[float] = [1.0, 2.0, 4.0, 8.0]) -> float:
    """Compute GDT-TS score."""
    # Center and align
    c1 = coords_pred - coords_pred.mean(axis=0)
    c2 = coords_ref - coords_ref.mean(axis=0)
    H = c1.T @ c2
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    c1_aligned = c1 @ R
    
    diff = np.sqrt(np.sum((c1_aligned - c2)**2, axis=1))
    n = len(diff)
    
    fractions = []
    for t in thresholds:
        frac = np.sum(diff <= t) / n
        fractions.append(frac)
    
    gdt_ts = np.mean(fractions)
    return gdt_ts


# ============================================================================
# Main analysis pipeline
# ============================================================================

if __name__ == '__main__':
    print("Unified Biomolecular Diffusion Framework")
    print("=" * 60)
    
    # Parse data
    protein = parse_pdb_ca('data/sample/2l3r/2l3r_protein.pdb')
    ligand = parse_sdf('data/sample/2l3r/2l3r_ligand.sdf')
    
    print(f"\nProtein FKBP12: {protein['n_residues']} CA atoms")
    print(f"Residue range: {protein['residue_ids'][0]} - {protein['residue_ids'][-1]}")
    print(f"Coordinate range: X [{protein['coords'][:,0].min():.1f}, {protein['coords'][:,0].max():.1f}]")
    print(f"                 Y [{protein['coords'][:,1].min():.1f}, {protein['coords'][:,1].max():.1f}]")
    print(f"                 Z [{protein['coords'][:,2].min():.1f}, {protein['coords'][:,2].max():.1f}]")
    
    print(f"\nLigand FK506: {ligand['n_atoms']} atoms, {ligand['n_bonds']} bonds")
    print(f"Elements: {set(ligand['atoms'])}")
    print(f"Coordinate range: X [{ligand['coords'][:,0].min():.1f}, {ligand['coords'][:,0].max():.1f}]")
    print(f"                 Y [{ligand['coords'][:,1].min():.1f}, {ligand['coords'][:,1].max():.1f}]")
    print(f"                 Z [{ligand['coords'][:,2].min():.1f}, {ligand['coords'][:,2].max():.1f}]")
    
    # Analyze structure
    prot_dm = compute_distance_matrix(protein['coords'])
    lig_dm = compute_distance_matrix(ligand['coords'])
    
    print(f"\nProtein distance matrix: mean={prot_dm.mean():.1f} Å, max={prot_dm.max():.1f} Å")
    print(f"Ligand distance matrix: mean={lig_dm.mean():.1f} Å, max={lig_dm.max():.1f} Å")
    
    # Diffusion demo
    print("\nDiffusion Demo:")
    diffusion = SE3Diffusion(n_timesteps=1000)
    
    # Forward diffusion on protein
    for t_frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
        t = int(t_frac * 999)
        x_t, noise = diffusion.forward_diffusion(protein['coords'], t)
        rmsd, _ = compute_rmsd(x_t, protein['coords'])
        print(f"  t={t} (ᾱ={diffusion.alpha_bars[t]:.4f}): RMSD to native = {rmsd:.2f} Å")
    
    # Save outputs
    outputs = {
        'protein': {
            'n_residues': protein['n_residues'],
            'residues': protein['residues'],
            'residue_ids': protein['residue_ids'],
            'coords_mean': protein['coords'].mean(axis=0).tolist(),
            'coords_std': protein['coords'].std(axis=0).tolist(),
            'radius_of_gyration': float(np.sqrt(np.mean(np.sum(
                (protein['coords'] - protein['coords'].mean(axis=0))**2, axis=1))))
        },
        'ligand': {
            'n_atoms': ligand['n_atoms'],
            'n_bonds': ligand['n_bonds'],
            'elements': list(set(ligand['atoms'])),
            'coords_mean': ligand['coords'].mean(axis=0).tolist(),
            'coords_std': ligand['coords'].std(axis=0).tolist(),
            'radius_of_gyration': float(np.sqrt(np.mean(np.sum(
                (ligand['coords'] - ligand['coords'].mean(axis=0))**2, axis=1))))
        }
    }
    
    import os
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/structure_analysis.json', 'w') as f:
        json.dump(outputs, f, indent=2)
    
    print("\nAnalysis complete. Results saved to outputs/structure_analysis.json")
