"""
Unified Diffusion-Based Biomolecular Complex Structure Prediction Framework.

This module implements:
1. A multi-modal encoder for protein sequences and ligand molecular graphs
2. A coordinate-based diffusion model for joint protein-ligand structure prediction
3. Training and sampling procedures

Architecture overview:
- Protein encoder: Transformer-based sequence encoder with positional embeddings
- Ligand encoder: Graph neural network (GNN) over molecular graph
- Joint representation: Cross-attention between protein and ligand representations
- Denoising network: SE(3)-equivariant network operating on 3D coordinates
- Diffusion process: Gaussian noise added to coordinates, learned denoising (x_0 prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict


# ============================================================
# Positional Encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence inputs."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimestepEmbedding(nn.Module):
    """Embeds diffusion timestep into a continuous vector."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        emb = F.silu(self.linear1(emb))
        emb = self.linear2(emb)
        return emb


# ============================================================
# Protein Sequence Encoder
# ============================================================

class ProteinEncoder(nn.Module):
    """Transformer-based encoder for protein amino acid sequences."""
    
    def __init__(self, num_aa: int = 20, d_model: int = 128, n_heads: int = 8, 
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(num_aa, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, onehot_seq: torch.Tensor) -> torch.Tensor:
        x = self.embedding(onehot_seq)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


# ============================================================
# Ligand Graph Encoder (GNN)
# ============================================================

class MessagePassingLayer(nn.Module):
    """Single message passing layer for molecular graph."""
    
    def __init__(self, node_dim: int, edge_dim: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim)
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.norm = nn.LayerNorm(node_dim)
    
    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, n_atoms, _ = node_features.shape
        edge_feat = self.edge_encoder(adj.unsqueeze(-1))
        src = node_features.unsqueeze(2).expand(-1, -1, n_atoms, -1)
        dst = node_features.unsqueeze(1).expand(-1, n_atoms, -1, -1)
        messages = torch.cat([src, dst, edge_feat], dim=-1)
        messages = self.message_mlp(messages)
        agg = torch.sum(messages * adj.unsqueeze(-1), dim=2)
        updated = torch.cat([node_features, agg], dim=-1)
        updated = self.update_mlp(updated)
        return self.norm(node_features + updated)


class LigandEncoder(nn.Module):
    """Graph neural network encoder for small molecule structures."""
    
    def __init__(self, num_atom_types: int = 11, d_model: int = 128, 
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.atom_embedding = nn.Embedding(num_atom_types, d_model)
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        self.init_proj = nn.Linear(d_model + d_model // 2, d_model)
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(d_model) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, atom_types: torch.Tensor, coords: torch.Tensor, 
                adj: torch.Tensor) -> torch.Tensor:
        atom_emb = self.atom_embedding(atom_types)
        coord_emb = self.coord_encoder(coords)
        x = torch.cat([atom_emb, coord_emb], dim=-1)
        x = self.init_proj(x)
        x = self.dropout(x)
        for mp_layer in self.mp_layers:
            x = mp_layer(x, adj)
        return self.norm(x)


# ============================================================
# Cross-Attention Module
# ============================================================

class CrossAttentionModule(nn.Module):
    """Cross-attention between protein and ligand representations."""
    
    def __init__(self, d_model: int = 128, n_heads: int = 8):
        super().__init__()
        self.protein_to_ligand = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ligand_to_protein = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_l = nn.LayerNorm(d_model)
    
    def forward(self, protein_repr: torch.Tensor, ligand_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_attended, _ = self.protein_to_ligand(
            query=protein_repr, key=ligand_repr, value=ligand_repr
        )
        protein_repr = self.norm_p(protein_repr + p_attended)
        l_attended, _ = self.ligand_to_protein(
            query=ligand_repr, key=protein_repr, value=protein_repr
        )
        ligand_repr = self.norm_l(ligand_repr + l_attended)
        return protein_repr, ligand_repr


# ============================================================
# Coordinate Denoising Network
# ============================================================

class CoordinateUpdate(nn.Module):
    """SE(3)-inspired coordinate update module."""
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.coord_predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 3)
        )
    
    def forward(self, node_repr: torch.Tensor, coords: torch.Tensor, 
                t_embed: torch.Tensor) -> torch.Tensor:
        batch_size, n_nodes, _ = node_repr.shape
        t_expanded = t_embed.unsqueeze(1).expand(-1, n_nodes, -1)
        combined = node_repr + t_expanded
        displacement = self.coord_predictor(combined)
        return displacement


class DenoisingNetwork(nn.Module):
    """
    Main denoising network for the diffusion model.
    
    Takes noisy coordinates and timestep, predicts the CLEAN coordinates (x_0) directly.
    This is the "x_0 prediction" parameterization of the diffusion model.
    """
    
    def __init__(self, d_model: int = 128, n_protein_layers: int = 4, 
                 n_ligand_layers: int = 4, n_cross_attn_heads: int = 8,
                 n_denoising_blocks: int = 3):
        super().__init__()
        self.d_model = d_model
        self.timestep_embed = TimestepEmbedding(d_model)
        
        self.protein_encoder = ProteinEncoder(
            num_aa=20, d_model=d_model, n_heads=n_cross_attn_heads,
            n_layers=n_protein_layers
        )
        self.ligand_encoder = LigandEncoder(
            num_atom_types=11, d_model=d_model, n_layers=n_ligand_layers
        )
        self.cross_attention = CrossAttentionModule(d_model, n_cross_attn_heads)
        
        self.denoising_blocks = nn.ModuleList([
            nn.ModuleDict({
                'protein_coord_update': CoordinateUpdate(d_model),
                'ligand_coord_update': CoordinateUpdate(d_model),
                'norm_p': nn.LayerNorm(d_model),
                'norm_l': nn.LayerNorm(d_model),
            })
            for _ in range(n_denoising_blocks)
        ])
        
        self.final_protein_coord = CoordinateUpdate(d_model)
        self.final_ligand_coord = CoordinateUpdate(d_model)
    
    def forward(self, noisy_protein_coords: torch.Tensor, 
                noisy_ligand_coords: torch.Tensor,
                protein_onehot: torch.Tensor,
                ligand_atom_types: torch.Tensor,
                ligand_adj: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predicts CLEAN coordinates (x_0) from noisy input.
        """
        t_embed = self.timestep_embed(t)
        
        # Encode inputs
        protein_repr = self.protein_encoder(protein_onehot)
        ligand_repr = self.ligand_encoder(ligand_atom_types, noisy_ligand_coords, ligand_adj)
        
        # Cross-attention
        protein_repr, ligand_repr = self.cross_attention(protein_repr, ligand_repr)
        
        # Apply denoising blocks
        for block in self.denoising_blocks:
            p_disp = block['protein_coord_update'](protein_repr, noisy_protein_coords, t_embed)
            noisy_protein_coords = noisy_protein_coords + p_disp * 0.1
            l_disp = block['ligand_coord_update'](ligand_repr, noisy_ligand_coords, t_embed)
            noisy_ligand_coords = noisy_ligand_coords + l_disp * 0.1
            protein_repr, ligand_repr = self.cross_attention(protein_repr, ligand_repr)
            protein_repr = block['norm_p'](protein_repr)
            ligand_repr = block['norm_l'](ligand_repr)
        
        # Final coordinate prediction (this is the predicted x_0)
        pred_protein = self.final_protein_coord(protein_repr, noisy_protein_coords, t_embed)
        pred_ligand = self.final_ligand_coord(ligand_repr, noisy_ligand_coords, t_embed)
        
        return pred_protein, pred_ligand


# ============================================================
# Diffusion Process (x_0 parameterization)
# ============================================================

class DiffusionScheduler:
    """
    Variance-preserving diffusion scheduler (DDPM-style).
    Uses x_0 prediction parameterization: the model directly predicts clean coordinates.
    """
    
    def __init__(self, n_timesteps: int = 1000, beta_start: float = 1e-4, 
                 beta_end: float = 0.02, schedule_type: str = 'linear'):
        self.n_timesteps = n_timesteps
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif schedule_type == 'cosine':
            s = 0.008
            t = torch.linspace(0, n_timesteps, n_timesteps + 1) / n_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clamp(self.betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, 
                  noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean coordinates at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        batch_size = x_0.shape[0]
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(batch_size, *([1] * (x_0.dim() - 1))).to(x_0.device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(batch_size, *([1] * (x_0.dim() - 1))).to(x_0.device)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
    
    def p_sample(self, model: DenoisingNetwork, x_t_protein: torch.Tensor, 
                 x_t_ligand: torch.Tensor, protein_onehot: torch.Tensor,
                 ligand_atom_types: torch.Tensor, ligand_adj: torch.Tensor,
                 t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step reverse diffusion sampling using x_0 prediction.
        """
        t_normalized = t.float() / self.n_timesteps
        
        # Model predicts clean coordinates (x_0) directly
        pred_x0_protein, pred_x0_ligand = model(
            x_t_protein, x_t_ligand, protein_onehot,
            ligand_atom_types, ligand_adj, t_normalized
        )
        
        # Compute posterior mean
        batch_size = x_t_protein.shape[0]
        alpha_t = self.alphas[t].view(batch_size, 1, 1).to(x_t_protein.device)
        alpha_bar_t = self.alphas_cumprod[t].view(batch_size, 1, 1).to(x_t_protein.device)
        alpha_bar_prev = self.alphas_cumprod_prev[t].view(batch_size, 1, 1).to(x_t_protein.device)
        beta_t = self.betas[t].view(batch_size, 1, 1).to(x_t_protein.device)
        
        coef_x0 = (beta_t * torch.sqrt(alpha_bar_prev)) / (1.0 - alpha_bar_t)
        coef_xt = ((1.0 - alpha_bar_prev) * torch.sqrt(alpha_t)) / (1.0 - alpha_bar_t)
        
        mu_protein = coef_x0 * pred_x0_protein + coef_xt * x_t_protein
        mu_ligand = coef_x0 * pred_x0_ligand + coef_xt * x_t_ligand
        
        # Add noise (except at t=0)
        variance = beta_t
        noise_protein = torch.randn_like(mu_protein) * torch.sqrt(variance)
        noise_ligand = torch.randn_like(mu_ligand) * torch.sqrt(variance)
        
        mask = (t > 0).float().view(batch_size, 1, 1)
        x_prev_protein = mu_protein + mask * noise_protein
        x_prev_ligand = mu_ligand + mask * noise_ligand
        
        return x_prev_protein, x_prev_ligand
    
    @torch.no_grad()
    def sample(self, model: DenoisingNetwork, protein_onehot: torch.Tensor,
               ligand_atom_types: torch.Tensor, ligand_adj: torch.Tensor,
               n_protein_residues: int, n_ligand_atoms: int,
               n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full reverse diffusion sampling from pure noise."""
        model.eval()
        device = protein_onehot.device
        
        # Expand inputs to match n_samples
        if protein_onehot.shape[0] == 1 and n_samples > 1:
            protein_onehot = protein_onehot.expand(n_samples, -1, -1)
        if ligand_atom_types.shape[0] == 1 and n_samples > 1:
            ligand_atom_types = ligand_atom_types.expand(n_samples, -1)
        if ligand_adj.shape[0] == 1 and n_samples > 1:
            ligand_adj = ligand_adj.expand(n_samples, -1, -1)
        
        # Start from pure noise
        x_protein = torch.randn(n_samples, n_protein_residues, 3, device=device)
        x_ligand = torch.randn(n_samples, n_ligand_atoms, 3, device=device)
        
        # Reverse diffusion loop
        for t_step in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
            x_protein, x_ligand = self.p_sample(
                model, x_protein, x_ligand, protein_onehot,
                ligand_atom_types, ligand_adj, t
            )
        
        return x_protein, x_ligand


# ============================================================
# Loss Functions
# ============================================================

def coordinate_mse_loss(pred: torch.Tensor, target: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean squared error on coordinates."""
    if mask is not None:
        pred = pred * mask
        target = target * mask
    return F.mse_loss(pred, target)


# ============================================================
# Main Model Wrapper
# ============================================================

class BioComplexDiffusionModel(nn.Module):
    """Unified model wrapping the denoising network and diffusion scheduler."""
    
    def __init__(self, d_model: int = 128, n_timesteps: int = 1000,
                 schedule_type: str = 'cosine'):
        super().__init__()
        self.denoising_network = DenoisingNetwork(d_model=d_model)
        self.scheduler = DiffusionScheduler(
            n_timesteps=n_timesteps, schedule_type=schedule_type
        )
    
    def forward(self, protein_coords: torch.Tensor, ligand_coords: torch.Tensor,
                protein_onehot: torch.Tensor, ligand_atom_types: torch.Tensor,
                ligand_adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Training forward pass."""
        batch_size = protein_coords.shape[0]
        device = protein_coords.device
        
        t = torch.randint(0, self.scheduler.n_timesteps, (batch_size,), device=device)
        t_normalized = t.float() / self.scheduler.n_timesteps
        
        noisy_protein, _ = self.scheduler.add_noise(protein_coords, t)
        noisy_ligand, _ = self.scheduler.add_noise(ligand_coords, t)
        
        pred_protein, pred_ligand = self.denoising_network(
            noisy_protein, noisy_ligand, protein_onehot,
            ligand_atom_types, ligand_adj, t_normalized
        )
        
        protein_loss = coordinate_mse_loss(pred_protein, protein_coords)
        ligand_loss = coordinate_mse_loss(pred_ligand, ligand_coords)
        total_loss = protein_loss + ligand_loss
        
        return {
            'total_loss': total_loss,
            'protein_loss': protein_loss,
            'ligand_loss': ligand_loss,
            'pred_protein_coords': pred_protein,
            'pred_ligand_coords': pred_ligand,
            'noisy_protein_coords': noisy_protein,
            'noisy_ligand_coords': noisy_ligand,
            'timesteps': t,
        }
    
    @torch.no_grad()
    def sample(self, protein_onehot: torch.Tensor, ligand_atom_types: torch.Tensor,
               ligand_adj: torch.Tensor, n_protein_residues: int, 
               n_ligand_atoms: int, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples via reverse diffusion."""
        return self.scheduler.sample(
            self.denoising_network, protein_onehot, ligand_atom_types, ligand_adj,
            n_protein_residues, n_ligand_atoms, n_samples
        )


if __name__ == '__main__':
    batch_size = 2
    n_residues = 161
    n_atoms = 194
    d_model = 128
    
    model = BioComplexDiffusionModel(d_model=d_model, n_timesteps=100)
    
    protein_coords = torch.randn(batch_size, n_residues, 3)
    ligand_coords = torch.randn(batch_size, n_atoms, 3)
    protein_onehot = torch.zeros(batch_size, n_residues, 20)
    protein_onehot[:, :, 0] = 1.0
    ligand_atom_types = torch.zeros(batch_size, n_atoms, dtype=torch.long)
    ligand_adj = torch.zeros(batch_size, n_atoms, n_atoms)
    
    output = model(protein_coords, ligand_coords, protein_onehot, 
                   ligand_atom_types, ligand_adj)
    
    print(f"Total loss: {output['total_loss'].item():.6f}")
    print(f"Protein loss: {output['protein_loss'].item():.6f}")
    print(f"Ligand loss: {output['ligand_loss'].item():.6f}")
    print(f"Pred protein shape: {output['pred_protein_coords'].shape}")
    print(f"Pred ligand shape: {output['pred_ligand_coords'].shape}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
