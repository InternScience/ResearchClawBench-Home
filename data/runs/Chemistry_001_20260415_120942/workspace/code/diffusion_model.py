"""
Diffusion-based generative model for biomolecular complex structure prediction.
Implements a denoising diffusion probabilistic model (DDPM) for 3D coordinate generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EquivariantGraphConv(nn.Module):
    """
    E(n) Equivariant Graph Convolutional Layer.
    Maintains equivariance to rotations and translations.
    """
    
    def __init__(self, in_features, out_features, hidden_features=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Edge network (invariant to E(n))
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_features * 2 + 1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU()
        )
        
        # Coordinate weight network
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1)
        )
        
        # Node feature update
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + hidden_features, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features)
        )
        
    def forward(self, x, coords, edge_index=None):
        """
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            coords: Node coordinates (batch_size, num_nodes, 3)
            edge_index: Edge indices (optional, if None uses fully connected)
        Returns:
            x_out: Updated node features
            coords_out: Updated coordinates
        """
        batch_size, num_nodes, _ = x.size()
        
        # Compute relative positions and distances
        coords_i = coords.unsqueeze(2)  # (batch_size, num_nodes, 1, 3)
        coords_j = coords.unsqueeze(1)  # (batch_size, 1, num_nodes, 3)
        
        rel_pos = coords_j - coords_i  # (batch_size, num_nodes, num_nodes, 3)
        dists = torch.sqrt(torch.sum(rel_pos ** 2, dim=-1, keepdim=True) + 1e-8)
        
        # Edge features
        x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        edge_feats = torch.cat([x_i, x_j, dists], dim=-1)
        
        # Edge messages
        edge_messages = self.edge_mlp(edge_feats)
        
        # Coordinate weights (for equivariant update)
        coord_weights = self.coord_mlp(edge_messages)
        
        # Equivariant coordinate update
        coord_update = torch.sum(coord_weights * rel_pos, dim=2)
        coords_out = coords + coord_update
        
        # Aggregate edge messages
        node_messages = torch.sum(edge_messages, dim=2)
        
        # Update node features
        x_out = self.node_mlp(torch.cat([x, node_messages], dim=-1))
        
        return x_out, coords_out


class DiffusionTransformerBlock(nn.Module):
    """Transformer block conditioned on time."""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x, time_emb):
        """
        Args:
            x: (batch_size, num_nodes, hidden_dim)
            time_emb: (batch_size, hidden_dim)
        """
        # Time conditioning
        time_scale = self.time_mlp(time_emb).unsqueeze(1)
        
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out * time_scale
        
        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        
        return x


class DiffusionModel(nn.Module):
    """
    Diffusion model for biomolecular complex structure prediction.
    """
    
    def __init__(
        self,
        protein_nodes,
        ligand_nodes,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        timesteps=1000,
        dropout=0.1
    ):
        super().__init__()
        
        self.protein_nodes = protein_nodes
        self.ligand_nodes = ligand_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.timesteps = timesteps
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(hidden_dim)
        
        # Input projections
        self.protein_proj = nn.Linear(20, hidden_dim)
        self.ligand_proj = nn.Linear(103, hidden_dim)
        
        # Equivariant graph convolution layers
        self.equivariant_layers = nn.ModuleList([
            EquivariantGraphConv(hidden_dim, hidden_dim)
            for _ in range(num_layers // 2)
        ])
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projections for noise prediction
        self.protein_noise_proj = nn.Linear(hidden_dim, 3)
        self.ligand_noise_proj = nn.Linear(hidden_dim, 3)
        
        # Initialize noise schedule
        self.register_buffer('betas', self.linear_beta_schedule(timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
        
    def linear_beta_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        """Linear noise schedule."""
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule (improved)."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, protein_coords_noisy, ligand_coords_noisy, 
                protein_features, ligand_features, timesteps):
        """
        Predict noise at given timestep.
        
        Args:
            protein_coords_noisy: Noised protein coordinates
            ligand_coords_noisy: Noised ligand coordinates
            protein_features: Protein features
            ligand_features: Ligand features
            timesteps: Diffusion timesteps
        Returns:
            protein_noise_pred: Predicted protein noise
            ligand_noise_pred: Predicted ligand noise
        """
        batch_size = protein_coords_noisy.size(0)
        
        # Time embedding
        t_emb = self.time_embedding(timesteps)
        
        # Project features
        protein_h = self.protein_proj(protein_features)
        ligand_h = self.ligand_proj(ligand_features)
        
        # Concatenate for joint processing
        h = torch.cat([protein_h, ligand_h], dim=1)
        coords = torch.cat([protein_coords_noisy, ligand_coords_noisy], dim=1)
        
        # Equivariant graph convolutions
        for layer in self.equivariant_layers:
            h, coords = layer(h, coords)
        
        # Transformer layers
        for layer in self.transformer_layers:
            h = layer(h, t_emb)
        
        # Split back
        protein_h = h[:, :self.protein_nodes, :]
        ligand_h = h[:, self.protein_nodes:, :]
        
        # Predict noise
        protein_noise_pred = self.protein_noise_proj(protein_h)
        ligand_noise_pred = self.ligand_noise_proj(ligand_h)
        
        return protein_noise_pred, ligand_noise_pred
    
    def add_noise(self, coords, t, noise=None):
        """Add noise to coordinates at timestep t."""
        if noise is None:
            noise = torch.randn_like(coords)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_cumprod_t * coords + sqrt_one_minus_alpha_cumprod_t * noise
    
    def sample(self, protein_features, ligand_features, num_samples=1, device='cpu'):
        """
        Sample structures using the diffusion model.
        
        Args:
            protein_features: Protein features
            ligand_features: Ligand features
            num_samples: Number of samples to generate
            device: Device to run on
        Returns:
            protein_coords: Generated protein coordinates
            ligand_coords: Generated ligand coordinates
        """
        batch_size = protein_features.size(0)
        
        # Start from random noise
        protein_coords = torch.randn(batch_size, self.protein_nodes, 3, device=device)
        ligand_coords = torch.randn(batch_size, self.ligand_nodes, 3, device=device)
        
        # Iterative denoising
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            protein_noise_pred, ligand_noise_pred = self.forward(
                protein_coords, ligand_coords,
                protein_features, ligand_features,
                t_batch
            )
            
            # Compute denoising step
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            # Compute coefficient
            coef = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            
            # Update coordinates
            protein_coords = (protein_coords - coef * protein_noise_pred) / torch.sqrt(alpha_t)
            ligand_coords = (ligand_coords - coef * ligand_noise_pred) / torch.sqrt(alpha_t)
            
            # Add noise (except at final step)
            if t > 0:
                noise_scale = torch.sqrt(beta_t)
                protein_coords = protein_coords + noise_scale * torch.randn_like(protein_coords)
                ligand_coords = ligand_coords + noise_scale * torch.randn_like(ligand_coords)
        
        return protein_coords, ligand_coords


def compute_rmsd(coords_pred, coords_true, align=True):
    """
    Compute RMSD between predicted and true coordinates.
    
    Args:
        coords_pred: Predicted coordinates
        coords_true: True coordinates
        align: Whether to align before computing RMSD
    """
    if align:
        # Kabsch algorithm for optimal alignment
        coords_pred = coords_pred - coords_pred.mean(dim=1, keepdim=True)
        coords_true = coords_true - coords_true.mean(dim=1, keepdim=True)
        
        # Compute covariance matrix
        H = torch.matmul(coords_pred.transpose(-2, -1), coords_true)
        
        # SVD
        U, S, Vt = torch.linalg.svd(H)
        R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
        
        # Apply rotation
        coords_pred_aligned = torch.matmul(coords_pred, R.transpose(-2, -1))
    else:
        coords_pred_aligned = coords_pred
    
    # Compute RMSD
    rmsd = torch.sqrt(torch.mean((coords_pred_aligned - coords_true) ** 2))
    
    return rmsd


if __name__ == "__main__":
    # Test the diffusion model
    batch_size = 2
    protein_nodes = 107
    ligand_nodes = 50
    
    protein_features = torch.randn(batch_size, protein_nodes, 20)
    ligand_features = torch.randn(batch_size, ligand_nodes, 103)
    protein_coords = torch.randn(batch_size, protein_nodes, 3)
    ligand_coords = torch.randn(batch_size, ligand_nodes, 3)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    model = DiffusionModel(protein_nodes, ligand_nodes)
    
    # Forward pass
    protein_noise, ligand_noise = model(
        protein_coords, ligand_coords,
        protein_features, ligand_features,
        timesteps
    )
    
    print(f"Protein noise shape: {protein_noise.shape}")
    print(f"Ligand noise shape: {ligand_noise.shape}")
    
    # Test sampling
    with torch.no_grad():
        protein_sampled, ligand_sampled = model.sample(
            protein_features, ligand_features, num_samples=1
        )
    
    print(f"Sampled protein shape: {protein_sampled.shape}")
    print(f"Sampled ligand shape: {ligand_sampled.shape}")
