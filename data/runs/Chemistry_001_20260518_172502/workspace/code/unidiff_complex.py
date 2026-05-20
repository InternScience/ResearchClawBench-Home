"""
UniDiff-Complex: Unified Diffusion Framework for Biomolecular Complex Structure Prediction

A diffusion-based architecture that takes protein sequences, nucleic acid sequences,
and small molecule structures as input, and outputs accurate 3D structures of
biomolecular complexes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
import math

# ============================================================================
# Utility Functions
# ============================================================================

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings as in Transformer.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def compute_rmsd(pred: torch.Tensor, true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Compute RMSD between two coordinate sets."""
    if mask is not None:
        pred = pred[mask.bool()]
        true = true[mask.bool()]
    diff = pred - true
    return torch.sqrt((diff ** 2).sum() / len(pred)).item()


def kabsch_alignment(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kabsch algorithm for optimal superposition.
    Returns (R, t) such that R @ P + t best aligns to Q.
    """
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


# ============================================================================
# Input Encoders
# ============================================================================

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX['UNK'] = len(AMINO_ACIDS)

NUCLEOTIDES = ['A', 'C', 'G', 'T', 'U']
NT_TO_IDX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}
NT_TO_IDX['N'] = len(NUCLEOTIDES)

class ProteinSequenceEncoder(nn.Module):
    """Encodes protein amino acid sequences into embeddings."""
    def __init__(self, d_model: int = 256, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(len(AMINO_ACIDS) + 1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, seq_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_indices: [B, L] integer indices of amino acids
        Returns:
            [B, L, d_model] sequence embeddings
        """
        B, L = seq_indices.shape
        x = self.embedding(seq_indices)
        x = x + self.pos_encoding[:, :L, :]
        x = self.transformer(x)
        return x


class NucleicAcidEncoder(nn.Module):
    """Encodes nucleic acid sequences into embeddings."""
    def __init__(self, d_model: int = 256, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(len(NUCLEOTIDES) + 1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, seq_indices: torch.Tensor) -> torch.Tensor:
        B, L = seq_indices.shape
        x = self.embedding(seq_indices)
        x = x + self.pos_encoding[:, :L, :]
        x = self.transformer(x)
        return x


class SmallMoleculeEncoder(nn.Module):
    """
    Graph neural network encoder for small molecules.
    Uses a simple message-passing architecture.
    """
    def __init__(self, in_features: int = 15, d_model: int = 256, num_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        self.atom_embed = nn.Linear(in_features, d_model)
        
        self.mp_layers = nn.ModuleList([
            nn.ModuleDict({
                'edge_mlp': nn.Linear(d_model * 2 + 1, d_model),
                'node_mlp': nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.LayerNorm(d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model)
                )
            })
            for _ in range(num_layers)
        ])
        
    def forward(self, atom_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            atom_features: [N_atoms, in_features]
            edge_index: [2, N_edges] connectivity
            edge_attr: [N_edges, edge_features] optional edge features
        Returns:
            [N_atoms, d_model] atom embeddings
        """
        x = self.atom_embed(atom_features)
        
        for layer in self.mp_layers:
            src, dst = edge_index[0], edge_index[1]
            x_src = x[src]
            x_dst = x[dst]
            
            if edge_attr is None:
                edge_attr = torch.zeros(len(src), 1, device=x.device)
            
            edge_msg = torch.cat([x_src, x_dst, edge_attr], dim=-1)
            edge_msg = F.relu(layer['edge_mlp'](edge_msg))
            
            # Aggregate messages
            aggr = torch.zeros_like(x)
            aggr.index_add_(0, dst, edge_msg)
            
            x = layer['node_mlp'](torch.cat([x, aggr], dim=-1)) + x
            
        return x


# ============================================================================
# Cross-Modal Attention for Inter-Molecular Interactions
# ============================================================================

class CrossModalAttention(nn.Module):
    """Cross-attention between different molecular modalities."""
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attention(query, key, value, key_padding_mask=mask)
        return self.norm(query + attn_out)


class UnifiedEncoder(nn.Module):
    """
    Unified encoder that processes protein, nucleic acid, and small molecule inputs
    and produces cross-modal representations for complex structure prediction.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.protein_encoder = ProteinSequenceEncoder(d_model=d_model)
        self.na_encoder = NucleicAcidEncoder(d_model=d_model)
        self.mol_encoder = SmallMoleculeEncoder(d_model=d_model)
        
        # Cross-modal attention layers
        self.protein_to_mol = CrossModalAttention(d_model, num_heads)
        self.mol_to_protein = CrossModalAttention(d_model, num_heads)
        self.protein_to_na = CrossModalAttention(d_model, num_heads)
        self.na_to_protein = CrossModalAttention(d_model, num_heads)
        self.mol_to_na = CrossModalAttention(d_model, num_heads)
        self.na_to_mol = CrossModalAttention(d_model, num_heads)
        
        # Fusion MLPs
        self.protein_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.mol_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.na_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        
    def forward(self, protein_seq: torch.Tensor, na_seq: Optional[torch.Tensor] = None,
                mol_features: Optional[torch.Tensor] = None,
                mol_edge_index: Optional[torch.Tensor] = None,
                mol_edge_attr: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            protein_seq: [B, L_protein]
            na_seq: [B, L_na] or None
            mol_features: [N_atoms, in_features] or None
            mol_edge_index: [2, N_edges] or None
        Returns:
            Dictionary of encoded representations
        """
        protein_repr = self.protein_encoder(protein_seq)
        
        outputs = {'protein': protein_repr}
        
        if mol_features is not None:
            mol_repr = self.mol_encoder(mol_features, mol_edge_index, mol_edge_attr)
            outputs['molecule'] = mol_repr
            
            # Cross-modal interaction
            B, Lp, D = protein_repr.shape
            # Expand protein to match molecule batch
            protein_flat = protein_repr.reshape(-1, D)
            
            mol_to_prot = self.mol_to_protein(mol_repr.unsqueeze(0), protein_flat.unsqueeze(0), protein_flat.unsqueeze(0))
            prot_to_mol = self.protein_to_mol(protein_flat.unsqueeze(0), mol_repr.unsqueeze(0), mol_repr.unsqueeze(0))
            
            outputs['protein_interaction'] = prot_to_mol.reshape(B, Lp, D)
            outputs['molecule_interaction'] = mol_to_prot.squeeze(0)
            
        if na_seq is not None:
            na_repr = self.na_encoder(na_seq)
            outputs['nucleic_acid'] = na_repr
            
        return outputs


# ============================================================================
# SE(3)-Equivariant Components
# ============================================================================

class SE3EquivariantLayer(nn.Module):
    """
    Simplified SE(3)-equivariant graph convolution layer.
    Updates node features and coordinates while preserving SE(3) equivariance.
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Scalar feature MLP
        self.scalar_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Coordinate update weight
        self.coord_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, in_dim] scalar features
            coords: [N, 3] coordinates
            edge_index: [2, E] edges
            edge_attr: [E, edge_dim] edge features
        Returns:
            Updated features and coordinates
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Edge vectors (SE(3) equivariant)
        edge_vec = coords[dst] - coords[src]  # [E, 3]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True) + 1e-6  # [E, 1]
        edge_dir = edge_vec / edge_dist  # [E, 3]
        
        if edge_attr is None:
            edge_attr = edge_dist
        else:
            edge_attr = torch.cat([edge_attr, edge_dist], dim=-1)
        
        # Message computation
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.scalar_mlp(edge_input)  # [E, out_dim]
        
        # Coordinate weights
        coord_weights = self.coord_mlp(edge_input)  # [E, 1]
        
        # Aggregate messages
        aggr_features = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
        aggr_features.index_add_(0, dst, messages)
        
        # Coordinate update (equivariant: uses edge_dir)
        coord_update = torch.zeros(x.size(0), 3, device=coords.device, dtype=coords.dtype)
        coord_update.index_add_(0, dst, coord_weights * edge_dir)
        
        return aggr_features, coord_update


# ============================================================================
# Diffusion Model for 3D Coordinate Generation
# ============================================================================

class DiffusionModel(nn.Module):
    """
    Denoising diffusion probabilistic model for 3D biomolecular complex structures.
    """
    def __init__(self, d_model: int = 256, num_layers: int = 6, num_heads: int = 8,
                 timesteps: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.timesteps = timesteps
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Coordinate embedding
        self.coord_embed = nn.Linear(3, d_model)
        
        # SE(3)-equivariant layers
        self.se3_layers = nn.ModuleList([
            SE3EquivariantLayer(d_model, d_model) for _ in range(num_layers)
        ])
        
        # Feature update MLPs after each SE3 layer
        self.feature_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_layers)
        ])
        
        # Cross-attention with sequence representations
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_layers // 2)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers // 2)
        ])
        
        # Output: predicted noise
        self.noise_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)
        )
        
        # Beta schedule (cosine)
        self.register_buffer('betas', self._cosine_beta_schedule(timesteps))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, coords_noisy: torch.Tensor, t: torch.Tensor,
                seq_repr: torch.Tensor, edge_index: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict noise given noisy coordinates at timestep t.
        
        Args:
            coords_noisy: [N, 3] noisy coordinates
            t: [B] or scalar timestep
            seq_repr: [B, L, d_model] sequence representation
            edge_index: [2, E] edges between atoms/residues
            mask: [N] optional mask
        Returns:
            [N, 3] predicted noise
        """
        # Time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = get_timestep_embedding(t, self.d_model)
        t_emb = self.time_embed(t_emb)  # [B, d_model]
        
        # Coordinate features
        x = self.coord_embed(coords_noisy)  # [N, d_model]
        
        # Add time embedding to all nodes
        x = x + t_emb.mean(dim=0, keepdim=True)  # [N, d_model]
        
        coords = coords_noisy.clone()
        
        for i, (se3_layer, feat_mlp) in enumerate(zip(self.se3_layers, self.feature_mlps)):
            feat_update, coord_update = se3_layer(x, coords, edge_index)
            x = feat_mlp(torch.cat([x, feat_update], dim=-1)) + x
            coords = coords + coord_update
            
            # Cross-attention with sequence at selected layers
            if i < len(self.cross_attn_layers) and seq_repr is not None:
                # Reshape for attention: [1, N, D] and [1, L, D]
                x_attn, _ = self.cross_attn_layers[i](
                    x.unsqueeze(0), seq_repr, seq_repr
                )
                x = self.cross_norms[i](x + x_attn.squeeze(0))
        
        # Predict noise
        noise_pred = self.noise_predictor(x)
        return noise_pred
    
    @torch.no_grad()
    def sample(self, seq_repr: torch.Tensor, edge_index: torch.Tensor,
               num_atoms: int, num_steps: int = 50,
               mask: Optional[torch.Tensor] = None,
               device: str = 'cpu') -> torch.Tensor:
        """
        Sample 3D coordinates via DDPM sampling.
        
        Args:
            seq_repr: [B, L, d_model]
            edge_index: [2, E]
            num_atoms: number of atoms to generate
            num_steps: number of denoising steps
            mask: [N] optional mask
        Returns:
            [N, 3] sampled coordinates
        """
        # Start from random noise
        coords = torch.randn(num_atoms, 3, device=device) * 10.0
        
        # Use strided sampling
        times = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()
        
        for t in times:
            t_batch = torch.tensor([t], device=device)
            noise_pred = self.forward(coords, t_batch, seq_repr, edge_index, mask)
            
            alpha_t = 1.0 - self.betas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            if t > 0:
                noise = torch.randn_like(coords)
                beta_t = self.betas[t]
                coords = (coords - beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_t)
                coords = coords + torch.sqrt(beta_t) * noise
            else:
                coords = (coords - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_t)
        
        return coords


# ============================================================================
# UniDiff-Complex: Main Model
# ============================================================================

class UniDiffComplex(nn.Module):
    """
    Unified diffusion-based model for biomolecular complex structure prediction.
    """
    def __init__(self, d_model: int = 256, num_encoder_layers: int = 4,
                 num_diffusion_layers: int = 6, timesteps: int = 1000):
        super().__init__()
        self.encoder = UnifiedEncoder(d_model=d_model)
        self.diffusion = DiffusionModel(d_model=d_model, num_layers=num_diffusion_layers,
                                         timesteps=timesteps)
        
    def forward(self, coords_noisy: torch.Tensor, t: torch.Tensor,
                protein_seq: torch.Tensor, edge_index: torch.Tensor,
                na_seq: Optional[torch.Tensor] = None,
                mol_features: Optional[torch.Tensor] = None,
                mol_edge_index: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training: predict noise.
        """
        # Encode inputs
        encoded = self.encoder(protein_seq, na_seq, mol_features, mol_edge_index)
        protein_repr = encoded['protein']
        
        # Use combined representation
        if 'protein_interaction' in encoded:
            combined_repr = protein_repr + 0.5 * encoded['protein_interaction']
        else:
            combined_repr = protein_repr
            
        # Predict noise
        noise_pred = self.diffusion(coords_noisy, t, combined_repr, edge_index, mask)
        return noise_pred
    
    @torch.no_grad()
    def predict_structure(self, protein_seq: torch.Tensor, num_atoms: int,
                          edge_index: torch.Tensor,
                          na_seq: Optional[torch.Tensor] = None,
                          mol_features: Optional[torch.Tensor] = None,
                          mol_edge_index: Optional[torch.Tensor] = None,
                          num_steps: int = 50, device: str = 'cpu') -> torch.Tensor:
        """
        Predict 3D structure from sequence inputs.
        """
        self.eval()
        encoded = self.encoder(protein_seq, na_seq, mol_features, mol_edge_index)
        protein_repr = encoded['protein']
        
        if 'protein_interaction' in encoded:
            combined_repr = protein_repr + 0.5 * encoded['protein_interaction']
        else:
            combined_repr = protein_repr
            
        coords = self.diffusion.sample(combined_repr, edge_index, num_atoms,
                                        num_steps=num_steps, device=device)
        return coords


if __name__ == '__main__':
    print("UniDiff-Complex model definition loaded successfully.")
