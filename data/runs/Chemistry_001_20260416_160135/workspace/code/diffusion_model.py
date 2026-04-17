"""
Diffusion-based architecture for biomolecular complex structure prediction.

This module implements a simplified diffusion model inspired by AlphaFold and
geometric deep learning principles for predicting 3D structures of protein-ligand complexes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion model."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(0)]


class ProteinEncoder(nn.Module):
    """
    Encoder for protein sequences using Transformer architecture.
    Inspired by AlphaFold's Evoformer and the Transformer architecture.
    """
    
    def __init__(self, config: DiffusionConfig, vocab_size: int = 22):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode protein sequence.
        
        Args:
            sequence: Tokenized protein sequence (batch, seq_len)
            
        Returns:
            Encoded representations (batch, seq_len, hidden_dim)
        """
        x = self.embedding(sequence) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.output_proj(x)


class LigandGraphEncoder(nn.Module):
    """
    Graph neural network encoder for ligand molecules.
    Uses geometric deep learning principles for molecular graph processing.
    """
    
    def __init__(self, config: DiffusionConfig, atom_vocab_size: int = 50, 
                 edge_vocab_size: int = 10):
        super().__init__()
        self.atom_embedding = nn.Embedding(atom_vocab_size, config.hidden_dim)
        self.edge_embedding = nn.Embedding(edge_vocab_size, config.hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            nn.MultiheadAttention(config.hidden_dim, config.num_heads, 
                                 dropout=config.dropout, batch_first=True)
            for _ in range(config.num_layers // 2)
        ])
        
        self.node_ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(self, atom_features: torch.Tensor, 
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Encode ligand molecular graph.
        
        Args:
            atom_features: Atom feature matrix (batch, num_atoms, hidden_dim)
            adjacency: Adjacency matrix (batch, num_atoms, num_atoms)
            
        Returns:
            Encoded node representations (batch, num_atoms, hidden_dim)
        """
        x = self.atom_embedding(atom_features)
        
        # Apply graph attention with adjacency masking
        for gat in self.gat_layers:
            # Use adjacency as attention bias - 2D mask for multihead attention
            if adjacency is not None:
                # Create mask: True where attention should be blocked
                # MultiheadAttention expects 2D mask (seq_len, seq_len) or None
                adj_2d = adjacency.squeeze(0) if adjacency.dim() > 2 else adjacency
                attn_mask = (1 - adj_2d).bool()
            else:
                attn_mask = None
            x_attn, _ = gat(x, x, x, attn_mask=attn_mask, need_weights=False)
            x = x + x_attn
            x = x + self.node_ffn(x)
        
        return self.output_proj(x)


class DiffusionScheduler:
    """
    Noise scheduler for the diffusion process.
    Implements the forward diffusion and reverse sampling schedules.
    """
    
    def __init__(self, config: DiffusionConfig):
        self.num_timesteps = config.num_timesteps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        
        # Precompute noise schedule
        betas = torch.linspace(
            config.beta_start ** 0.5,
            config.beta_end ** 0.5,
            config.num_timesteps
        ) ** 2
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def register_buffer(self, name: str, values: torch.Tensor):
        """Register buffer for state dict compatibility."""
        setattr(self, name, values)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean coordinates (forward diffusion).
        
        Args:
            x0: Clean coordinates (batch, N, 3)
            t: Timestep indices (batch,)
            
        Returns:
            Noisy coordinates and noise used
        """
        batch_size = x0.shape[0]
        noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1, 1)
        
        xt = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas * noise
        return xt, noise
    
    def get_score_input(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Prepare timestep embedding for the denoising network."""
        # Sinusoidal timestep embedding
        half_dim = 128
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class DenoisingNetwork(nn.Module):
    """
    Neural network for predicting noise in the diffusion process.
    Combines protein and ligand representations to predict structure updates.
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Cross-attention between protein and ligand
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads,
            dropout=config.dropout, batch_first=True
        )
        
        # Coordinate refinement MLP
        self.coord_mlp = nn.Sequential( 
            nn.Linear(config.hidden_dim + 3, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3)  
        )
        
        # Confidence prediction (like pLDDT in AlphaFold)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, protein_repr: torch.Tensor, 
                ligand_repr: torch.Tensor,
                noisy_coords: torch.Tensor,
                timestep_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict noise/coordinate updates from noisy input.
        
        Args:
            protein_repr: Protein representations (batch, seq_len, hidden_dim)
            ligand_repr: Ligand representations (batch, num_atoms, hidden_dim)
            noisy_coords: Noisy coordinates (batch, N, 3)
            timestep_emb: Timestep embeddings (batch, hidden_dim)
            
        Returns:
            Tuple of (predicted coordinates, confidence scores)
        """
        batch_size = protein_repr.shape[0]
        
        # Time conditioning
        time_feat = self.time_embed(timestep_emb)
        time_feat = time_feat.unsqueeze(1)
        
        # Cross-attention: ligand attends to protein
        ligand_attended, _ = self.cross_attention(
            ligand_repr, protein_repr, protein_repr,
            need_weights=False
        )
        
        # Combine features
        combined = ligand_attended + time_feat.expand(-1, ligand_repr.shape[1], -1)
        
        # Predict coordinate updates
        coord_input = torch.cat([ligand_attended, noisy_coords], dim=-1)
        coord_update = self.coord_mlp(coord_input)
        
        pred_coords = noisy_coords + coord_update
        
        # Confidence scores - fix dimension mismatch
        conf_input = combined.mean(dim=1)  # (batch, hidden_dim)
        confidence = self.confidence_head(conf_input)
        
        return pred_coords, confidence


class BiomolecularDiffusionModel(nn.Module):
    """
    Complete diffusion model for biomolecular complex structure prediction.
    
    This model combines:
    - Protein sequence encoding (Transformer-based)
    - Ligand graph encoding (GNN-based)
    - Diffusion-based coordinate generation
    - Confidence estimation
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.scheduler = DiffusionScheduler(config)
        
        self.protein_encoder = ProteinEncoder(config)
        self.ligand_encoder = LigandGraphEncoder(config)
        self.denoiser = DenoisingNetwork(config)
        
        # Amino acid vocabulary
        self.aa_vocab = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
            'UNK': 20, 'PAD': 21
        }
        
        # Element vocabulary for ligands
        self.element_vocab = {
            'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'P': 5,
            'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'B': 10,
            'UNK': 11
        }
    
    def tokenize_sequence(self, residues: List[str]) -> torch.Tensor:
        """Convert residue names to token indices."""
        tokens = [self.aa_vocab.get(res, 20) for res in residues]
        return torch.tensor(tokens, dtype=torch.long)
    
    def tokenize_atoms(self, atoms: List[Dict]) -> torch.Tensor:
        """Convert atom elements to token indices."""
        tokens = []
        for atom in atoms:
            element = atom.get('element', 'UNK')
            tokens.append(self.element_vocab.get(element, 11))
        return torch.tensor(tokens, dtype=torch.long)
    
    def build_adjacency(self, bonds: List[Dict], num_atoms: int) -> torch.Tensor:
        """Build adjacency matrix from bond list."""
        adj = torch.zeros(num_atoms, num_atoms)
        for bond in bonds:
            i, j = bond['begin_atom'], bond['end_atom']
            adj[i, j] = 1
            adj[j, i] = 1
        return adj
    
    def forward(self, protein_data: Dict, ligand_data: Dict, 
                t: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass through the diffusion model.
        
        Args:
            protein_data: Parsed protein data
            ligand_data: Parsed ligand data
            t: Optional timestep (for training)
            
        Returns:
            Dictionary containing predictions
        """
        # Encode protein
        protein_tokens = self.tokenize_sequence(
            [r['residue_name'] for r in protein_data['residues']]
        ).unsqueeze(0)
        protein_repr = self.protein_encoder(protein_tokens)
        
        # Encode ligand
        atom_tokens = self.tokenize_atoms(ligand_data['atoms']).unsqueeze(0)
        adj = self.build_adjacency(
            ligand_data['bonds'], 
            len(ligand_data['atoms'])
        ).unsqueeze(0)
        ligand_repr = self.ligand_encoder(atom_tokens, adj)
        
        # Get ground truth coordinates for training
        true_coords = torch.from_numpy(ligand_data['coordinates']).float().unsqueeze(0)
        
        # Sample timestep if not provided
        if t is None:
            t = torch.randint(0, self.config.num_timesteps, (1,))
        
        # Add noise
        noisy_coords, noise = self.scheduler.add_noise(true_coords, t)
        
        # Get timestep embedding
        timestep_emb = self.scheduler.get_score_input(noisy_coords, t)
        
        # Predict denoised coordinates
        pred_coords, confidence = self.denoiser(
            protein_repr, ligand_repr, noisy_coords, timestep_emb
        )
        
        return {
            'predicted_coords': pred_coords,
            'confidence': confidence,
            'noisy_coords': noisy_coords,
            'true_coords': true_coords,
            'timestep': t
        }
    
    @torch.no_grad()
    def sample(self, protein_data: Dict, ligand_data: Dict, 
               num_samples: int = 1) -> Dict:
        """
        Generate samples using reverse diffusion.
        
        Args:
            protein_data: Parsed protein data
            ligand_data: Parsed ligand data
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing generated samples
        """
        # Encode inputs
        protein_tokens = self.tokenize_sequence(
            [r['residue_name'] for r in protein_data['residues']]
        ).unsqueeze(0)
        
        # Truncate sequence if too long for positional encoding (max 50)
        max_seq_len = min(protein_tokens.shape[1], 50)
        protein_tokens = protein_tokens[:, :max_seq_len]
        
        protein_repr = self.protein_encoder(protein_tokens)
        
        atom_tokens = self.tokenize_atoms(ligand_data['atoms']).unsqueeze(0)
        adj = self.build_adjacency(
            ligand_data['bonds'], 
            len(ligand_data['atoms'])
        ).unsqueeze(0)
        ligand_repr = self.ligand_encoder(atom_tokens, adj)
        
        # Expand for multiple samples
        protein_repr = protein_repr.expand(num_samples, -1, -1)
        ligand_repr = ligand_repr.expand(num_samples, -1, -1)
        
        # Initialize from noise
        shape = (num_samples, len(ligand_data['atoms']), 3)
        xt = torch.randn(shape)
        
        # Reverse diffusion
        all_samples = []
        for t in reversed(range(self.config.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long)
            timestep_emb = self.scheduler.get_score_input(xt, t_tensor)
            
            pred_coords, confidence = self.denoiser(
                protein_repr, ligand_repr, xt, timestep_emb
            )
            
            # Compute noise prediction
            alpha_t = float(self.scheduler.alphas[t])
            alpha_cumprod_t = float(self.scheduler.alphas_cumprod[t])
            if t > 0:
                alpha_cumprod_prev = float(self.scheduler.alphas_cumprod[t - 1])
            else:
                alpha_cumprod_prev = 1.0
            
            # Posterior variance
            beta_t = float(self.scheduler.betas[t])
            sigma_t = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            
            # Mean of posterior
            sqrt_alpha_cumprod_t = np.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha = np.sqrt(1 - alpha_cumprod_t)
            pred_x0 = (xt - sqrt_one_minus_alpha * (xt - pred_coords)) / sqrt_alpha_cumprod_t
            mean = np.sqrt(alpha_cumprod_prev) * (xt - np.sqrt(1 - alpha_cumprod_t) * (xt - pred_x0)) / (1 - alpha_cumprod_t)
            
            # Sample
            if t > 0:
                noise = torch.randn_like(xt)
                xt = mean + np.sqrt(sigma_t) * noise
            else:
                xt = mean
            
            all_samples.append(xt.clone())
        
        return {
            'samples': xt,
            'trajectory': all_samples,
            'confidence': confidence
        }


if __name__ == "__main__":
    # Test model construction
    config = DiffusionConfig(
        num_timesteps=100,
        hidden_dim=128,
        num_layers=4,
        num_heads=4
    )
    
    model = BiomolecularDiffusionModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data
    dummy_protein = {
        'residues': [{'residue_name': 'ALA'}, {'residue_name': 'GLY'}] * 10
    }
    dummy_ligand = {
        'atoms': [{'element': 'C'}] * 10 + [{'element': 'O'}] * 5,
        'bonds': [{'begin_atom': i, 'end_atom': i+1} for i in range(14)],
        'coordinates': np.random.randn(15, 3)
    }
    
    # Test forward pass
    output = model(dummy_protein, dummy_ligand)
    print(f"Output shapes: coords={output['predicted_coords'].shape}, conf={output['confidence'].shape}")
