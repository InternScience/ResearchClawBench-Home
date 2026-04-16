"""
Unified Diffusion-Based Framework for Biomolecular Complex Structure Prediction
================================================================================
BioDiffusion3D: A unified deep learning framework that takes protein sequences,
nucleic acid sequences, and small molecule structures as input, and outputs
accurate 3D structures of biomolecular complexes using a diffusion-based architecture.

Key components:
1. MultiModalEncoder - Encodes protein/nucleic acid sequences and small molecule graphs
2. SE3EquivariantBlock - SE(3)-equivariant attention for 3D structure reasoning
3. DiffusionModule - Score-based diffusion for 3D coordinate generation
4. StructureDecoder - Decodes latent representations to 3D coordinates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, List


# ============================================================
# 1. Tokenization and Embedding Layers
# ============================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
NUCLEOTIDES = list("ACGTU")
ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]
BOND_TYPES = [1, 2, 3]  # single, double, triple

class ResidueEmbedding(nn.Module):
    """Embed protein residues with amino acid type and position."""
    def __init__(self, d_model=128, max_len=2048):
        super().__init__()
        self.aa_embed = nn.Embedding(len(AMINO_ACIDS) + 2, d_model)  # +2 for padding/unknown
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, aa_indices: torch.Tensor) -> torch.Tensor:
        seq_len = aa_indices.shape[1]
        positions = torch.arange(seq_len, device=aa_indices.device).unsqueeze(0)
        return self.aa_embed(aa_indices) + self.pos_embed(positions)


class NucleotideEmbedding(nn.Module):
    """Embed nucleic acid nucleotides."""
    def __init__(self, d_model=128, max_len=8192):
        super().__init__()
        self.nuc_embed = nn.Embedding(len(NUCLEOTIDES) + 2, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, nuc_indices: torch.Tensor) -> torch.Tensor:
        seq_len = nuc_indices.shape[1]
        positions = torch.arange(seq_len, device=nuc_indices.device).unsqueeze(0)
        return self.nuc_embed(nuc_indices) + self.pos_embed(positions)


class AtomEmbedding(nn.Module):
    """Embed small molecule atoms with element type, degree, and hybridization."""
    def __init__(self, d_model=128, d_sub=None):
        super().__init__()
        d_sub = d_sub or (d_model // 3)
        self.element_embed = nn.Embedding(len(ATOM_TYPES) + 2, d_sub)
        self.degree_embed = nn.Embedding(7, d_sub)  # degree 0-5 + padding
        self.charge_embed = nn.Linear(1, d_sub)
        self.proj = nn.Linear(d_sub * 3, d_model)
        self.d_model = d_model

    def forward(self, element_indices: torch.Tensor, degree_indices: torch.Tensor,
                charges: torch.Tensor) -> torch.Tensor:
        e = self.element_embed(element_indices)
        d = self.degree_embed(degree_indices)
        c = self.charge_embed(charges)
        return self.proj(torch.cat([e, d, c], dim=-1))


class ModalityToken(nn.Module):
    """Add modality-specific token to distinguish protein/nucleic acid/small molecule."""
    def __init__(self, d_model=128):
        super().__init__()
        self.modality_embed = nn.Embedding(3, d_model)  # 0=protein, 1=nucleic acid, 2=small molecule

    def forward(self, x: torch.Tensor, modality: int) -> torch.Tensor:
        mod_token = self.modality_embed(torch.tensor([modality], device=x.device))
        return x + mod_token.unsqueeze(0)  # [1, d_model] -> broadcast over [B, L, d_model]


# ============================================================
# 2. Transformer Blocks with Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional cross-attention."""
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None and mask.dim() == 3:
            mask = mask.squeeze(0)
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        Q = self.W_q(query).view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.W_o(out), attn_weights


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with self-attention and feed-forward."""
    def __init__(self, d_model=128, n_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


# ============================================================
# 3. SE(3)-Equivariant Layers
# ============================================================

class SE3EquivariantAttention(nn.Module):
    """SE(3)-equivariant attention layer for 3D coordinate refinement.
    Uses invariant features for attention weights and equivariant features for updates.
    """
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Invariant attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Equivariant coordinate update
        self.coord_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads * 3)
        )

        # Pairwise distance embedding
        self.dist_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, n_heads)
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [B, N, d_model]
            coords: 3D coordinates [B, N, 3]
            mask: Optional mask [B, N]
        Returns:
            Updated features, updated coordinates, attention weights
        """
        B, N, _ = x.shape

        # Compute pairwise distances (invariant to rotation/translation)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        dist_bias = self.dist_embed(dist)  # [B, N, N, n_heads]

        # Invariant attention
        Q = self.q_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, N, N]
        attn = attn + dist_bias.permute(0, 3, 1, 2)  # Add distance bias

        if mask is not None:
            if mask.dim() == 2:
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
            else:
                attn = attn.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Feature update (invariant)
        feat_out = torch.matmul(attn_weights, V)
        feat_out = feat_out.transpose(1, 2).contiguous().view(B, N, -1)
        feat_out = self.out_proj(feat_out)

        # Coordinate update (equivariant)
        coord_update = self.coord_proj(feat_out)  # [B, N, n_heads*3]
        coord_update = coord_update.view(B, N, self.n_heads, 3)

        # Weight coordinate update by attention-weighted direction vectors
        # attn_weights: [B, H, N, N], diff: [B, N, N, 3] -> [B, H, N, 3]
        weighted_diff = torch.einsum('bhij,bijd->bhjd', attn_weights, diff)  # [B, H, N, 3]
        weighted_diff = weighted_diff.permute(0, 2, 1, 3)  # [B, N, H, 3]
        coord_delta = (coord_update * weighted_diff).sum(dim=2)  # [B, N, 3]

        # Residual connections
        x_out = x + self.dropout(feat_out)
        coords_out = coords + coord_delta * 0.1  # Small step for stability

        return self.norm(x_out), coords_out, attn_weights


class SE3TransformerBlock(nn.Module):
    """Full SE(3)-equivariant transformer block."""
    def __init__(self, d_model=128, n_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.se3_attn = SE3EquivariantAttention(d_model, n_heads, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, coords, attn_weights = self.se3_attn(x, coords, mask)
        x = x + self.ff(self.ff_norm(x))
        return x, coords, attn_weights


# ============================================================
# 4. Pairwise Feature Processing (Evoformer-inspired)
# ============================================================

class PairwiseUpdate(nn.Module):
    """Update pairwise features using outer product of single features and triangular attention."""
    def __init__(self, d_model=128, d_pair=64):
        super().__init__()
        self.d_pair = d_pair
        self.proj_left = nn.Linear(d_model, d_pair)
        self.proj_right = nn.Linear(d_model, d_pair)
        self.pair_norm = nn.LayerNorm(d_pair)
        self.tri_update = nn.Sequential(
            nn.Linear(d_pair, d_pair * 2),
            nn.GELU(),
            nn.Linear(d_pair * 2, d_pair)
        )

    def forward(self, x: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        # Outer product update
        left = self.proj_left(x)
        right = self.proj_right(x)
        outer = torch.einsum('bid,bjd->bijd', left, right)
        pair = pair + outer

        # Triangular update
        pair = pair + self.tri_update(self.pair_norm(pair))
        return pair


# ============================================================
# 5. Diffusion Module
# ============================================================

class GaussianDiffusion(nn.Module):
    """Gaussian diffusion process for 3D coordinate generation.
    
    Implements the DDPM framework adapted for SE(3)-equivariant
    structure prediction.
    """
    def __init__(self, d_model=128, n_heads=8, n_layers=4, d_ff=512,
                 timesteps=1000, noise_schedule="cosine"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.timesteps = timesteps
        self.noise_schedule = noise_schedule

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Denoising network: SE(3)-equivariant transformer blocks
        self.blocks = nn.ModuleList([
            SE3TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3)  # Predict noise in 3D
        )

        # Setup noise schedule
        self._setup_noise_schedule()

    def _setup_noise_schedule(self):
        """Setup cosine noise schedule."""
        if self.noise_schedule == "cosine":
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.timesteps) + 0.008) * math.pi / 2.0) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            betas = torch.linspace(0.0001, 0.02, self.timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)

    def add_noise(self, coords: torch.Tensor, t: torch.Tensor,
                  noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to coordinates at timestep t."""
        if noise is None:
            noise = torch.randn_like(coords)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)

        noisy_coords = sqrt_alpha * coords + sqrt_one_minus_alpha * noise
        return noisy_coords, noise

    def forward(self, x: torch.Tensor, coords: torch.Tensor, t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict noise given noisy coordinates and timestep.
        
        Args:
            x: Node features [B, N, d_model]
            coords: Noisy coordinates [B, N, 3]
            t: Timestep indices [B]
            mask: Optional mask [B, N]
        Returns:
            Predicted noise [B, N, 3]
        """
        # Add timestep embedding to features
        t_emb = self.timestep_embedding(t)  # [B, d_model]
        x = x + t_emb.unsqueeze(1)

        # Apply SE(3)-equivariant denoising blocks
        for block in self.blocks:
            x, coords, _ = block(x, coords, mask)

        # Predict noise
        noise_pred = self.output_proj(x)
        return noise_pred

    @torch.no_grad()
    def sample(self, x: torch.Tensor, init_coords: torch.Tensor,
               n_steps: int = 50, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate structures via iterative denoising (DDIM sampling).
        
        Args:
            x: Node features [B, N, d_model]
            init_coords: Initial noisy coordinates [B, N, 3]
            n_steps: Number of denoising steps
            mask: Optional mask [B, N]
        Returns:
            Final coordinates and trajectory of intermediate structures
        """
        coords = init_coords.clone()
        trajectory = [coords.clone()]

        # Use a subset of timesteps for efficiency
        step_size = self.timesteps // n_steps
        timesteps = list(range(0, self.timesteps, step_size))[:n_steps]
        timesteps = list(reversed(timesteps))

        for i, t_val in enumerate(timesteps):
            t = torch.full((x.shape[0],), t_val, device=x.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.forward(x, coords, t, mask)

            # DDIM update
            alpha_t = self.alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
            alpha_t_prev = self.alphas_cumprod[max(0, t_val - step_size)].unsqueeze(-1).unsqueeze(-1) if t_val > 0 else torch.ones_like(alpha_t)

            # Predict x0
            x0_pred = (coords - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred

            # Compute x_{t-1}
            coords = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt

            # Center coordinates (remove COM for equivariance)
            if mask is not None:
                com = (coords * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1)
            else:
                com = coords.mean(dim=1, keepdim=True)
            coords = coords - com

            trajectory.append(coords.clone())

        return coords, trajectory


# ============================================================
# 6. Multi-Modal Encoder
# ============================================================

class MultiModalEncoder(nn.Module):
    """Encode protein sequences, nucleic acid sequences, and small molecule structures
    into a unified representation space."""
    def __init__(self, d_model=128, n_layers=4, n_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input embeddings
        self.protein_embed = ResidueEmbedding(d_model)
        self.nucleic_embed = NucleotideEmbedding(d_model)
        self.atom_embed = AtomEmbedding(d_model)
        self.modality_token = ModalityToken(d_model)

        # Graph encoder for small molecules
        self.graph_attn = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(2)
        ])

        # Cross-modal transformer
        self.cross_modal = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Pairwise feature processing
        self.pairwise = PairwiseUpdate(d_model, d_pair=64)

    def encode_protein(self, aa_indices: torch.Tensor) -> torch.Tensor:
        """Encode protein sequence."""
        x = self.protein_embed(aa_indices)
        x = self.modality_token(x, 0)
        return x

    def encode_nucleic_acid(self, nuc_indices: torch.Tensor) -> torch.Tensor:
        """Encode nucleic acid sequence."""
        x = self.nucleic_embed(nuc_indices)
        x = self.modality_token(x, 1)
        return x

    def encode_small_molecule(self, element_indices: torch.Tensor,
                               degree_indices: torch.Tensor,
                               charges: torch.Tensor,
                               adj_matrix: torch.Tensor) -> torch.Tensor:
        """Encode small molecule with graph attention."""
        x = self.atom_embed(element_indices, degree_indices, charges)
        x = self.modality_token(x, 2)

        # Apply graph attention with adjacency mask
        for block in self.graph_attn:
            if adj_matrix is not None:
                adj_mask = adj_matrix[0] if adj_matrix.dim() == 3 else adj_matrix
            else:
                adj_mask = None
            x, _ = block(x, mask=adj_mask)

        return x

    def forward(self, protein_seq: Optional[torch.Tensor] = None,
                nucleic_seq: Optional[torch.Tensor] = None,
                mol_elements: Optional[torch.Tensor] = None,
                mol_degrees: Optional[torch.Tensor] = None,
                mol_charges: Optional[torch.Tensor] = None,
                mol_adj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode all modalities and fuse with cross-modal attention.
        
        Returns:
            Fused features [B, N_total, d_model]
            Pairwise features [B, N_total, N_total, d_pair]
        """
        parts = []

        if protein_seq is not None:
            parts.append(self.encode_protein(protein_seq))

        if nucleic_seq is not None:
            parts.append(self.encode_nucleic_acid(nucleic_seq))

        if mol_elements is not None:
            parts.append(self.encode_small_molecule(mol_elements, mol_degrees, mol_charges, mol_adj))

        # Concatenate all modalities
        x = torch.cat(parts, dim=1)

        # Cross-modal attention
        attn_weights_all = []
        for block in self.cross_modal:
            x, attn_w = block(x)
            attn_weights_all.append(attn_w)

        # Compute pairwise features
        pair = torch.zeros(x.shape[0], x.shape[1], x.shape[1], 64, device=x.device)
        pair = self.pairwise(x, pair)

        return x, pair, attn_weights_all


# ============================================================
# 7. Full Model: BioDiffusion3D
# ============================================================

class BioDiffusion3D(nn.Module):
    """Unified diffusion-based framework for biomolecular complex structure prediction.
    
    Takes protein sequences, nucleic acid sequences, and small molecule structures as input,
    and outputs accurate 3D structures of biomolecular complexes.
    """
    def __init__(self, d_model=128, n_heads=8, n_encoder_layers=4,
                 n_diffusion_layers=4, d_ff=512, timesteps=1000,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Multi-modal encoder
        self.encoder = MultiModalEncoder(d_model, n_encoder_layers, n_heads, d_ff, dropout)

        # Coordinate initialization from features
        self.coord_init = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3)
        )

        # Diffusion module
        self.diffusion = GaussianDiffusion(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_diffusion_layers,
            d_ff=d_ff,
            timesteps=timesteps
        )

        # Confidence prediction head (pLDDT)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, protein_seq: Optional[torch.Tensor] = None,
                nucleic_seq: Optional[torch.Tensor] = None,
                mol_elements: Optional[torch.Tensor] = None,
                mol_degrees: Optional[torch.Tensor] = None,
                mol_charges: Optional[torch.Tensor] = None,
                mol_adj: Optional[torch.Tensor] = None,
                coords: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            protein_seq: Protein amino acid indices [B, L_prot]
            nucleic_seq: Nucleic acid indices [B, L_nuc]
            mol_elements: Molecule element indices [B, L_mol]
            mol_degrees: Molecule atom degrees [B, L_mol]
            mol_charges: Molecule atom charges [B, L_mol]
            mol_adj: Molecule adjacency matrix [B, L_mol, L_mol]
            coords: Ground truth coordinates [B, N_total, 3]
            t: Timestep indices [B]
        Returns:
            Dictionary with predicted noise, confidence scores, etc.
        """
        # Encode inputs
        x, pair, attn_weights = self.encoder(
            protein_seq, nucleic_seq,
            mol_elements, mol_degrees, mol_charges, mol_adj
        )

        # Initialize coordinates from features if not provided
        if coords is None:
            coords = self.coord_init(x)

        # Add noise and predict
        if t is not None:
            noisy_coords, noise = self.diffusion.add_noise(coords, t)
            noise_pred = self.diffusion(x, noisy_coords, t)
            confidence = self.confidence_head(x)
            return {
                'noise_pred': noise_pred,
                'noise': noise,
                'confidence': confidence,
                'attn_weights': attn_weights,
                'pair_features': pair
            }
        else:
            confidence = self.confidence_head(x)
            return {
                'features': x,
                'coords_init': coords,
                'confidence': confidence,
                'attn_weights': attn_weights,
                'pair_features': pair
            }

    @torch.no_grad()
    def predict_structure(self, protein_seq: Optional[torch.Tensor] = None,
                          nucleic_seq: Optional[torch.Tensor] = None,
                          mol_elements: Optional[torch.Tensor] = None,
                          mol_degrees: Optional[torch.Tensor] = None,
                          mol_charges: Optional[torch.Tensor] = None,
                          mol_adj: Optional[torch.Tensor] = None,
                          n_diffusion_steps: int = 50) -> Dict[str, torch.Tensor]:
        """Predict 3D structure via diffusion sampling.
        
        Returns:
            Dictionary with predicted coordinates, confidence, trajectory, etc.
        """
        # Encode inputs
        x, pair, attn_weights = self.encoder(
            protein_seq, nucleic_seq,
            mol_elements, mol_degrees, mol_charges, mol_adj
        )

        # Initialize coordinates
        init_coords = self.coord_init(x)

        # Add maximum noise
        t_max = torch.full((x.shape[0],), self.diffusion.timesteps - 1,
                           device=x.device, dtype=torch.long)
        noisy_coords, _ = self.diffusion.add_noise(init_coords, t_max)

        # Run diffusion sampling
        final_coords, trajectory = self.diffusion.sample(
            x, noisy_coords, n_steps=n_diffusion_steps
        )

        # Predict confidence
        confidence = self.confidence_head(x)

        return {
            'coords': final_coords,
            'confidence': confidence,
            'trajectory': trajectory,
            'attn_weights': attn_weights,
            'pair_features': pair,
            'init_coords': init_coords
        }


# ============================================================
# 8. Loss Functions
# ============================================================

class DiffusionLoss(nn.Module):
    """Combined loss for diffusion training."""
    def __init__(self, lambda_noise=1.0, lambda_conf=0.1, lambda_dist=0.5):
        super().__init__()
        self.lambda_noise = lambda_noise
        self.lambda_conf = lambda_conf
        self.lambda_dist = lambda_dist

    def forward(self, noise_pred: torch.Tensor, noise: torch.Tensor,
                confidence: torch.Tensor, coords: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Noise prediction loss (MSE)
        noise_loss = F.mse_loss(noise_pred, noise)

        # Distance consistency loss
        pred_dist = torch.cdist(noise_pred, noise_pred)
        true_dist = torch.cdist(coords, coords)
        dist_loss = F.mse_loss(pred_dist, true_dist)

        # Confidence loss (encourage high confidence)
        conf_loss = -confidence.mean()

        total_loss = (self.lambda_noise * noise_loss +
                     self.lambda_dist * dist_loss +
                     self.lambda_conf * conf_loss)

        return {
            'total_loss': total_loss,
            'noise_loss': noise_loss,
            'dist_loss': dist_loss,
            'conf_loss': conf_loss
        }


if __name__ == "__main__":
    # Quick test
    model = BioDiffusion3D(d_model=64, n_heads=4, n_encoder_layers=2,
                           n_diffusion_layers=2, d_ff=256, timesteps=100)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("BioDiffusion3D model initialized successfully!")
