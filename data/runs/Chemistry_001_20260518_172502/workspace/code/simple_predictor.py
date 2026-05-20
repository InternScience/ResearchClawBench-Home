"""
Simplified but effective predictor for biomolecular complex structures.
This version uses a more direct architecture that can learn from limited data.
"""

import torch
import torch.nn as nn
import numpy as np

class SimpleComplexPredictor(nn.Module):
    """
    Simplified predictor that directly outputs 3D coordinates.
    Uses attention-based pooling and MLP regression.
    """
    def __init__(self, protein_len, ligand_atoms, d_model=128):
        super().__init__()
        self.protein_len = protein_len
        self.ligand_atoms = ligand_atoms
        self.d_model = d_model
        
        # Simple embeddings
        self.aa_embed = nn.Embedding(21, d_model)
        self.atom_embed = nn.Linear(15, d_model)
        
        # Protein processing
        self.protein_conv = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Ligand processing
        self.ligand_conv = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Coordinate predictors
        self.protein_coord_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)
        )
        
        self.ligand_coord_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)
        )
        
        # Learned global offset
        self.global_offset = nn.Parameter(torch.zeros(3))
        
    def forward(self, protein_seq, ligand_features, ligand_edge_index=None):
        # Protein encoding
        p_emb = self.aa_embed(protein_seq)  # [B, L, D]
        p_emb = self.protein_conv(p_emb)
        
        # Ligand encoding
        l_emb = self.atom_embed(ligand_features)  # [N_lig, D]
        l_emb = self.ligand_conv(l_emb)
        
        # Cross attention: ligand attends to protein
        l_emb_attn, _ = self.cross_attn(l_emb.unsqueeze(0), p_emb, p_emb)
        l_emb = l_emb + l_emb_attn.squeeze(0)
        
        # Predict coordinates
        p_coords = self.protein_coord_head(p_emb).squeeze(0)  # [L, 3]
        l_coords = self.ligand_coord_head(l_emb)  # [N_lig, 3]
        
        # Add global offset to ligand (binding site localization)
        l_coords = l_coords + self.global_offset
        
        return torch.cat([p_coords, l_coords], dim=0)


def kabsch_alignment(P, Q):
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    if d < 0:
        S[-1] = -S[-1]
        Vt[-1, :] *= -1
    R = Vt.T @ U.T
    t = Q_mean - R @ P_mean
    return R, t
