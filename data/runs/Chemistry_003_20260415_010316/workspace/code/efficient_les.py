"""
Efficient LES model implementation optimized for the benchmark datasets.
Uses vectorized operations and simpler architectures for faster training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class EfficientRadialBasis(nn.Module):
    """Efficient radial basis functions."""
    def __init__(self, n_rbf, cutoff, start=0.5):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        centers = torch.linspace(start, cutoff, n_rbf)
        self.register_buffer('centers', centers)
        self.width = (cutoff - start) / n_rbf
        
    def forward(self, dist):
        """dist: (...,) -> (..., n_rbf)"""
        return torch.exp(-self.width * (dist[..., None] - self.centers)**2)


class SmoothCutoff(nn.Module):
    """Smooth cosine cutoff."""
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
    
    def forward(self, dist):
        return torch.where(
            dist < self.cutoff,
            0.5 * (torch.cos(np.pi * dist / self.cutoff) + 1.0),
            torch.zeros_like(dist)
        )


class SimpleMessagePassing(nn.Module):
    """Simple but efficient message passing layer."""
    def __init__(self, n_hidden, n_rbf):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden)
        )
        self.update_net = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden)
        )
        self.layer_norm = nn.LayerNorm(n_hidden)
    
    def forward(self, h, rbf_vals, cutoff_vals, mask):
        """
        h: (N, F)
        rbf_vals: (N, N, n_rbf)
        cutoff_vals: (N, N)
        mask: (N, N) bool
        """
        # Compute filter from radial basis
        filters = self.filter_net(rbf_vals)  # (N, N, F)
        
        # Apply cutoff and mask
        weight = cutoff_vals[..., None] * mask[..., None].float()  # (N, N, 1)
        filters = filters * weight
        
        # Message: weighted sum of neighbor features
        messages = torch.einsum('ijF,jF->iF', filters, h)  # (N, F)
        
        # Update
        h_new = self.update_net(messages)
        h = self.layer_norm(h + h_new)
        
        return h


class EfficientLESModel(nn.Module):
    """Efficient LES model for benchmark datasets.
    
    Key features:
    - Vectorized pairwise computations
    - Simple but effective message passing
    - Latent charge prediction with total charge constraint
    - Ewald/direct Coulomb energy computation
    - Short-range energy from message passing
    """
    def __init__(self, n_elements, n_rbf=8, cutoff=8.0, n_hidden=32, 
                 n_mp_layers=2, alpha=0.2, k_max=3):
        super().__init__()
        self.n_elements = n_elements
        self.n_hidden = n_hidden
        self.cutoff = cutoff
        
        # Element embedding
        self.elem_embed = nn.Embedding(n_elements, n_hidden)
        
        # Radial basis and cutoff
        self.rbf = EfficientRadialBasis(n_rbf, cutoff)
        self.cutoff_fn = SmoothCutoff(cutoff)
        
        # Message passing layers for charge prediction
        self.charge_mp = nn.ModuleList([
            SimpleMessagePassing(n_hidden, n_rbf) for _ in range(n_mp_layers)
        ])
        self.charge_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        
        # Message passing layers for short-range energy
        self.sr_mp = nn.ModuleList([
            SimpleMessagePassing(n_hidden, n_rbf) for _ in range(n_mp_layers)
        ])
        self.sr_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        
        # Energy scaling
        self.energy_scale = nn.Parameter(torch.ones(1))
        self.energy_bias = nn.Parameter(torch.zeros(1))
        
        # Ewald parameters
        self.alpha = alpha
        self.k_max = k_max
        
    def compute_pairwise(self, positions, cell=None, pbc=None):
        """Compute pairwise distances and displacement vectors.
        
        Returns:
            dist: (N, N) distances
            diff: (N, N, 3) displacement vectors
            mask: (N, N) neighbor mask
        """
        diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
        
        # Apply PBC
        if cell is not None and pbc is not None:
            if pbc.any():
                inv_cell = torch.linalg.inv(cell)
                frac = torch.einsum('ij,kj->ki', inv_cell, diff.reshape(-1, 3))
                frac = frac - torch.round(frac)
                diff = torch.einsum('ij,kj->ki', cell, frac).reshape_as(diff)
        
        dist = torch.norm(diff, dim=-1)  # (N, N)
        mask = (dist > 1e-6) & (dist < self.cutoff)
        
        return dist, diff, mask
    
    def predict_charges(self, positions, elem_idx, cell=None, pbc=None, total_charge=0.0):
        """Predict latent charges from local environments."""
        dist, diff, mask = self.compute_pairwise(positions, cell, pbc)
        
        # Element embeddings
        h = self.elem_embed(elem_idx)  # (N, F)
        
        # Radial basis and cutoff
        rbf_vals = self.rbf(dist)  # (N, N, n_rbf)
        cutoff_vals = self.cutoff_fn(dist)  # (N, N)
        
        # Message passing
        for layer in self.charge_mp:
            h = layer(h, rbf_vals, cutoff_vals, mask)
        
        # Predict charges
        charges = self.charge_head(h).squeeze(-1)  # (N,)
        
        # Apply total charge constraint
        charges = charges - charges.mean() + total_charge / charges.shape[0]
        
        return charges
    
    def predict_short_range(self, positions, elem_idx, cell=None, pbc=None):
        """Predict short-range atomic energies."""
        dist, diff, mask = self.compute_pairwise(positions, cell, pbc)
        
        # Element embeddings (separate from charge network)
        h = self.elem_embed(elem_idx)
        
        # Radial basis and cutoff
        rbf_vals = self.rbf(dist)
        cutoff_vals = self.cutoff_fn(dist)
        
        # Message passing
        for layer in self.sr_mp:
            h = layer(h, rbf_vals, cutoff_vals, mask)
        
        # Predict atomic energies
        atomic_energies = self.sr_head(h).squeeze(-1)  # (N,)
        
        return atomic_energies
    
    def compute_coulomb_energy(self, positions, charges, cell=None, pbc=None):
        """Compute Coulomb energy (direct sum for non-periodic)."""
        diff = positions[None, :, :] - positions[:, None, :]
        dist = torch.norm(diff, dim=-1)
        
        # Avoid division by zero
        safe_dist = torch.where(dist > 1e-8, dist, torch.ones_like(dist))
        
        # Charge product
        qq = charges[:, None] * charges[None, :]
        
        # Coulomb energy (upper triangle)
        energy = torch.triu(qq / safe_dist, diagonal=1).sum()
        
        return energy
    
    def forward(self, positions, elem_idx, cell=None, pbc=None, total_charge=0.0):
        """Compute total energy and latent charges."""
        # Predict latent charges
        charges = self.predict_charges(positions, elem_idx, cell, pbc, total_charge)
        
        # Compute electrostatic energy
        E_elec = self.compute_coulomb_energy(positions, charges, cell, pbc)
        
        # Compute short-range energy
        atomic_energies = self.predict_short_range(positions, elem_idx, cell, pbc)
        E_sr = atomic_energies.sum()
        
        # Total energy with learnable scaling
        E_total = self.energy_scale * (E_sr + E_elec) + self.energy_bias
        
        return E_total, charges


class EfficientShortRangeModel(nn.Module):
    """Short-range only baseline model."""
    def __init__(self, n_elements, n_rbf=8, cutoff=8.0, n_hidden=32, n_mp_layers=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.cutoff = cutoff
        
        self.elem_embed = nn.Embedding(n_elements, n_hidden)
        self.rbf = EfficientRadialBasis(n_rbf, cutoff)
        self.cutoff_fn = SmoothCutoff(cutoff)
        
        self.mp_layers = nn.ModuleList([
            SimpleMessagePassing(n_hidden, n_rbf) for _ in range(n_mp_layers)
        ])
        self.energy_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        
        self.energy_scale = nn.Parameter(torch.ones(1))
        self.energy_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, positions, elem_idx, cell=None, pbc=None, total_charge=0.0, **kwargs):
        """Compute total energy (short-range only)."""
        diff = positions[None, :, :] - positions[:, None, :]
        dist = torch.norm(diff, dim=-1)
        mask = (dist > 1e-6) & (dist < self.cutoff)
        
        h = self.elem_embed(elem_idx)
        rbf_vals = self.rbf(dist)
        cutoff_vals = self.cutoff_fn(dist)
        
        for layer in self.mp_layers:
            h = layer(h, rbf_vals, cutoff_vals, mask)
        
        atomic_energies = self.energy_head(h).squeeze(-1)
        E_total = self.energy_scale * atomic_energies.sum() + self.energy_bias
        
        return E_total, None


class EfficientLESWithChargeEmbedding(nn.Module):
    """LES model with global charge state embedding for Ag3 experiment."""
    def __init__(self, n_elements, n_rbf=8, cutoff=8.0, n_hidden=32, n_mp_layers=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.cutoff = cutoff
        
        self.elem_embed = nn.Embedding(n_elements, n_hidden)
        self.charge_embed = nn.Linear(1, n_hidden)
        self.rbf = EfficientRadialBasis(n_rbf, cutoff)
        self.cutoff_fn = SmoothCutoff(cutoff)
        
        # Combined message passing
        self.mp_layers = nn.ModuleList([
            SimpleMessagePassing(n_hidden, n_rbf) for _ in range(n_mp_layers)
        ])
        
        # Charge head
        self.charge_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        
        # Energy head (with charge input)
        self.energy_head = nn.Sequential(
            nn.Linear(n_hidden + n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        
        self.energy_scale = nn.Parameter(torch.ones(1))
        self.energy_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, positions, elem_idx, cell=None, pbc=None, total_charge=0.0):
        """Compute total energy with charge embedding."""
        diff = positions[None, :, :] - positions[:, None, :]
        dist = torch.norm(diff, dim=-1)
        mask = (dist > 1e-6) & (dist < self.cutoff)
        
        h = self.elem_embed(elem_idx)
        
        # Add charge embedding
        charge_feat = self.charge_embed(
            torch.full((positions.shape[0], 1), total_charge, device=positions.device)
        )
        h = h + charge_feat
        
        rbf_vals = self.rbf(dist)
        cutoff_vals = self.cutoff_fn(dist)
        
        for layer in self.mp_layers:
            h = layer(h, rbf_vals, cutoff_vals, mask)
        
        # Predict charges
        charges = self.charge_head(h).squeeze(-1)
        charges = charges - charges.mean() + total_charge / charges.shape[0]
        
        # Predict energy with charge context
        energy_input = torch.cat([h, charge_feat], dim=-1)
        atomic_energies = self.energy_head(energy_input).squeeze(-1)
        
        # Coulomb energy
        safe_dist = torch.where(dist > 1e-8, dist, torch.ones_like(dist))
        qq = charges[:, None] * charges[None, :]
        E_elec = torch.triu(qq / safe_dist, diagonal=1).sum()
        
        E_total = self.energy_scale * (atomic_energies.sum() + E_elec) + self.energy_bias
        
        return E_total, charges
