"""
Latent Ewald Summation (LES) Model Implementation

This module implements the LES approach for incorporating long-range
electrostatic interactions into machine-learning interatomic potentials.

Key components:
1. Latent charge prediction network (from local atomic environments)
2. Ewald summation for long-range electrostatic energy
3. Short-range energy network
4. Combined model with differentiable force computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GaussianBasis(nn.Module):
    """Radial basis functions using Gaussian expansions."""
    def __init__(self, n_rbf, cutoff, start=0.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        # Spread Gaussian centers evenly from start to cutoff
        self.centers = nn.Parameter(
            torch.linspace(start, cutoff, n_rbf), requires_grad=False
        )
        self.width = (cutoff - start) / n_rbf
        
    def forward(self, distances):
        """Compute Gaussian basis functions.
        
        Args:
            distances: (..., ) tensor of distances
        
        Returns:
            (..., n_rbf) tensor of basis function values
        """
        return torch.exp(-self.width * (distances[..., None] - self.centers)**2)


class CosineCutoff(nn.Module):
    """Cosine cutoff function for smooth decay."""
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
        
    def forward(self, distances):
        """Apply cosine cutoff.
        
        Args:
            distances: (..., ) tensor of distances
        
        Returns:
            (..., ) tensor of cutoff values
        """
        return torch.where(
            distances < self.cutoff,
            0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1),
            torch.zeros_like(distances)
        )


class LatentChargeNetwork(nn.Module):
    """Predict latent charges from local atomic environments.
    
    Uses a simple message-passing-like architecture to compute
    local atomic descriptors, then predicts a scalar charge per atom.
    """
    def __init__(self, n_elements, n_rbf=16, cutoff=6.0, n_hidden=64, n_layers=3):
        super().__init__()
        self.n_elements = n_elements
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        
        # Element embedding
        self.element_embedding = nn.Embedding(n_elements, n_hidden)
        
        # Radial basis
        self.rbf = GaussianBasis(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        
        # Interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.interaction_layers.append(
                InteractionLayer(n_hidden, n_rbf)
            )
        
        # Charge output
        self.charge_output = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        
    def forward(self, positions, element_indices, cell=None, pbc=None, total_charge=0.0):
        """Predict latent charges.
        
        Args:
            positions: (N, 3) tensor of atomic positions
            element_indices: (N,) tensor of element indices
            cell: (3, 3) tensor of cell vectors (optional)
            pbc: (3,) tensor of PBC flags (optional)
            total_charge: float, total charge constraint
        
        Returns:
            (N,) tensor of latent charges
        """
        N = positions.shape[0]
        
        # Element embeddings
        h = self.element_embedding(element_indices)  # (N, n_hidden)
        
        # Compute pairwise distances
        diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
        
        # Apply PBC if needed
        if cell is not None and pbc is not None:
            # Shift vectors into the unit cell
            inv_cell = torch.linalg.inv(cell)
            frac = torch.einsum('ij,kj->ki', inv_cell, diff.reshape(-1, 3))
            frac = frac - torch.round(frac)
            diff = torch.einsum('ij,kj->ki', cell, frac).reshape(N, N, 3)
        
        dist = torch.norm(diff, dim=-1)  # (N, N)
        
        # Create neighbor mask (exclude self and beyond cutoff)
        mask = (dist > 0.01) & (dist < self.cutoff)
        
        # Radial basis
        rbf_vals = self.rbf(dist)  # (N, N, n_rbf)
        cutoff_vals = self.cutoff_fn(dist)  # (N, N)
        
        # Interaction layers
        for layer in self.interaction_layers:
            h = layer(h, rbf_vals, cutoff_vals, mask)
        
        # Predict charges
        charges = self.charge_output(h).squeeze(-1)  # (N,)
        
        # Apply total charge constraint
        charges = charges - charges.mean() + total_charge / N
        
        return charges


class InteractionLayer(nn.Module):
    """Simple interaction layer for message passing."""
    def __init__(self, n_hidden, n_rbf):
        super().__init__()
        self.n_hidden = n_hidden
        
        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(n_hidden + n_rbf, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(2 * n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden)
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(n_hidden)
        
    def forward(self, h, rbf_vals, cutoff_vals, mask):
        """Process one interaction layer.
        
        Args:
            h: (N, n_hidden) node features
            rbf_vals: (N, N, n_rbf) radial basis values
            cutoff_vals: (N, N) cutoff values
            mask: (N, N) neighbor mask
        
        Returns:
            (N, n_hidden) updated node features
        """
        N = h.shape[0]
        
        # Compute messages
        # Expand h for pairwise computation
        h_i = h[:, None, :].expand(-1, N, -1)  # (N, N, n_hidden)
        
        # Message input: concatenate h_j with radial basis
        msg_input = torch.cat([h_i, rbf_vals], dim=-1)  # (N, N, n_hidden + n_rbf)
        messages = self.message_net(msg_input)  # (N, N, n_hidden)
        
        # Apply cutoff and mask
        messages = messages * cutoff_vals[..., None] * mask[..., None]
        
        # Aggregate messages
        agg = messages.sum(dim=1)  # (N, n_hidden)
        
        # Update
        update_input = torch.cat([h, agg], dim=-1)  # (N, 2*n_hidden)
        h_new = self.update_net(update_input)
        
        # Residual connection with layer norm
        h = self.layer_norm(h + h_new)
        
        return h


class ShortRangeNetwork(nn.Module):
    """Short-range energy network.
    
    Predicts atomic energy contributions from local environments.
    """
    def __init__(self, n_elements, n_rbf=16, cutoff=6.0, n_hidden=64, n_layers=3):
        super().__init__()
        self.n_elements = n_elements
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        
        # Element embedding
        self.element_embedding = nn.Embedding(n_elements, n_hidden)
        
        # Radial basis
        self.rbf = GaussianBasis(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        
        # Interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.interaction_layers.append(
                InteractionLayer(n_hidden, n_rbf)
            )
        
        # Energy output
        self.energy_output = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        
    def forward(self, positions, element_indices, cell=None, pbc=None):
        """Predict short-range atomic energies.
        
        Args:
            positions: (N, 3) tensor of atomic positions
            element_indices: (N,) tensor of element indices
            cell: (3, 3) tensor of cell vectors (optional)
            pbc: (3,) tensor of PBC flags (optional)
        
        Returns:
            (N,) tensor of atomic energy contributions
        """
        N = positions.shape[0]
        
        # Element embeddings
        h = self.element_embedding(element_indices)
        
        # Compute pairwise distances
        diff = positions[None, :, :] - positions[:, None, :]
        
        if cell is not None and pbc is not None:
            inv_cell = torch.linalg.inv(cell)
            frac = torch.einsum('ij,kj->ki', inv_cell, diff.reshape(-1, 3))
            frac = frac - torch.round(frac)
            diff = torch.einsum('ij,kj->ki', cell, frac).reshape(N, N, 3)
        
        dist = torch.norm(diff, dim=-1)
        
        mask = (dist > 0.01) & (dist < self.cutoff)
        
        rbf_vals = self.rbf(dist)
        cutoff_vals = self.cutoff_fn(dist)
        
        for layer in self.interaction_layers:
            h = layer(h, rbf_vals, cutoff_vals, mask)
        
        # Predict atomic energies
        atomic_energies = self.energy_output(h).squeeze(-1)
        
        return atomic_energies


class EwaldSummation(nn.Module):
    """Ewald summation for electrostatic energy with learnable charges.
    
    Computes the electrostatic energy using the standard Ewald method:
    E_elec = E_real + E_recip + E_self + E_charged_system
    
    For non-periodic systems, uses direct Coulomb sum.
    """
    def __init__(self, alpha=0.2, k_max=5):
        super().__init__()
        self.alpha = alpha  # Ewald screening parameter
        self.k_max = k_max  # Maximum k-vector index
        
    def forward(self, positions, charges, cell=None, pbc=None):
        """Compute electrostatic energy via Ewald summation.
        
        Args:
            positions: (N, 3) tensor of atomic positions
            charges: (N,) tensor of atomic charges
            cell: (3, 3) tensor of cell vectors
            pbc: (3,) tensor of PBC flags
        
        Returns:
            Scalar electrostatic energy
        """
        is_periodic = (cell is not None and pbc is not None and pbc.any())
        
        if is_periodic:
            return self._ewald_periodic(positions, charges, cell, pbc)
        else:
            return self._coulomb_direct(positions, charges)
    
    def _coulomb_direct(self, positions, charges):
        """Direct Coulomb sum for non-periodic systems.
        
        E = sum_{i<j} q_i * q_j / r_ij
        """
        N = positions.shape[0]
        
        # Pairwise displacement vectors
        diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (N, N)
        
        # Avoid division by zero
        dist = torch.where(dist > 1e-8, dist, torch.ones_like(dist))
        
        # Charge product matrix
        qq = charges[:, None] * charges[None, :]  # (N, N)
        
        # Coulomb energy (upper triangle only)
        energy = torch.triu(qq / dist, diagonal=1).sum()
        
        return energy
    
    def _ewald_periodic(self, positions, charges, cell, pbc):
        """Ewald summation for periodic systems.
        
        E = E_real + E_recip + E_self
        """
        alpha = self.alpha
        N = positions.shape[0]
        
        # Volume
        volume = torch.abs(torch.det(cell))
        
        # Reciprocal lattice vectors
        inv_cell = torch.linalg.inv(cell)
        # recip_vectors are rows of 2*pi*inv_cell
        recip = 2 * math.pi * inv_cell  # (3, 3)
        
        # Real space sum
        E_real = self._real_space_sum(positions, charges, cell, alpha)
        
        # Reciprocal space sum
        E_recip = self._recip_space_sum(positions, charges, recip, volume, alpha)
        
        # Self energy correction
        E_self = -alpha / math.sqrt(math.pi) * (charges ** 2).sum()
        
        return E_real + E_recip + E_self
    
    def _real_space_sum(self, positions, charges, cell, alpha):
        """Real space part of Ewald summation."""
        N = positions.shape[0]
        
        # For simplicity, compute direct sum with erfc
        diff = positions[None, :, :] - positions[:, None, :]
        dist = torch.norm(diff, dim=-1)
        dist = torch.where(dist > 1e-8, dist, torch.ones_like(dist))
        
        qq = charges[:, None] * charges[None, :]
        
        # erfc(alpha * r) / r
        erfc_vals = torch.erfc(alpha * dist) / dist
        
        energy = torch.triu(qq * erfc_vals, diagonal=1).sum()
        
        return energy
    
    def _recip_space_sum(self, positions, charges, recip, volume, alpha):
        """Reciprocal space part of Ewald summation."""
        N = positions.shape[0]
        k_max = self.k_max
        
        energy = torch.tensor(0.0, device=positions.device)
        
        # Generate k-vectors
        for i in range(-k_max, k_max + 1):
            for j in range(-k_max, k_max + 1):
                for k in range(-k_max, k_max + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    
                    kvec = torch.tensor([i, j, k], dtype=recip.dtype, device=recip.device)
                    k_cart = recip @ kvec  # (3,)
                    k_sq = (k_cart ** 2).sum()
                    
                    # Structure factor
                    kr = positions @ k_cart  # (N,)
                    S_real = (charges * torch.cos(kr)).sum()
                    S_imag = (charges * torch.sin(kr)).sum()
                    S_sq = S_real ** 2 + S_imag ** 2
                    
                    # Reciprocal space contribution
                    contrib = (2 * math.pi / volume) * S_sq * torch.exp(-k_sq / (4 * alpha ** 2)) / k_sq
                    energy = energy + contrib
        
        return energy


class LESModel(nn.Module):
    """Latent Ewald Summation Model.
    
    Combines:
    1. Latent charge prediction from local environments
    2. Long-range electrostatic energy via Ewald summation
    3. Short-range energy from a separate network
    
    Total energy: E = E_short_range + E_electrostatic(latent_charges)
    """
    def __init__(self, n_elements, n_rbf=16, cutoff=6.0, n_hidden=64, 
                 n_layers=3, alpha=0.2, k_max=5, use_ewald=True):
        super().__init__()
        self.use_ewald = use_ewald
        
        # Latent charge network
        self.charge_net = LatentChargeNetwork(
            n_elements, n_rbf, cutoff, n_hidden, n_layers
        )
        
        # Short-range energy network
        self.short_range_net = ShortRangeNetwork(
            n_elements, n_rbf, cutoff, n_hidden, n_layers
        )
        
        # Ewald summation
        self.ewald = EwaldSummation(alpha=alpha, k_max=k_max)
        
    def forward(self, positions, element_indices, cell=None, pbc=None, 
                total_charge=0.0):
        """Compute total energy and latent charges.
        
        Args:
            positions: (N, 3) tensor of atomic positions
            element_indices: (N,) tensor of element indices
            cell: (3, 3) tensor of cell vectors
            pbc: (3,) tensor of PBC flags
            total_charge: float, total charge constraint
        
        Returns:
            energy: scalar total energy
            charges: (N,) tensor of latent charges
        """
        # Predict latent charges
        charges = self.charge_net(positions, element_indices, cell, pbc, total_charge)
        
        # Compute electrostatic energy
        if self.use_ewald:
            E_elec = self.ewald(positions, charges, cell, pbc)
        else:
            E_elec = self.ewald._coulomb_direct(positions, charges)
        
        # Compute short-range energy
        atomic_energies = self.short_range_net(positions, element_indices, cell, pbc)
        E_sr = atomic_energies.sum()
        
        # Total energy
        E_total = E_sr + E_elec
        
        return E_total, charges


class ShortRangeOnlyModel(nn.Module):
    """Short-range only model (baseline without long-range electrostatics).
    
    This is a baseline model that only uses local atomic environments
    to predict energies, without any long-range electrostatic treatment.
    """
    def __init__(self, n_elements, n_rbf=16, cutoff=6.0, n_hidden=64, n_layers=3):
        super().__init__()
        self.short_range_net = ShortRangeNetwork(
            n_elements, n_rbf, cutoff, n_hidden, n_layers
        )
        
    def forward(self, positions, element_indices, cell=None, pbc=None, **kwargs):
        """Compute total energy (short-range only)."""
        atomic_energies = self.short_range_net(positions, element_indices, cell, pbc)
        E_total = atomic_energies.sum()
        return E_total, None


class LESModelWithChargeEmbedding(nn.Module):
    """LES model with global charge state embedding.
    
    Extends the LES model by providing the total charge as an
    additional input to the charge and short-range networks.
    """
    def __init__(self, n_elements, n_rbf=16, cutoff=6.0, n_hidden=64, 
                 n_layers=3, alpha=0.2, k_max=5):
        super().__init__()
        
        # Charge embedding
        self.charge_embedding = nn.Linear(1, n_hidden)
        
        # Latent charge network (with charge input)
        self.charge_net = LatentChargeNetwork(
            n_elements, n_rbf, cutoff, n_hidden, n_layers
        )
        
        # Short-range energy network (with charge input)
        self.short_range_net = ShortRangeNetwork(
            n_elements, n_rbf, cutoff, n_hidden, n_layers
        )
        
        # Charge-aware energy mixing
        self.charge_mix = nn.Sequential(
            nn.Linear(n_hidden + 1, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        
        # Ewald summation
        self.ewald = EwaldSummation(alpha=alpha, k_max=k_max)
        
    def forward(self, positions, element_indices, cell=None, pbc=None, 
                total_charge=0.0):
        """Compute total energy with charge embedding."""
        # Predict latent charges
        charges = self.charge_net(positions, element_indices, cell, pbc, total_charge)
        
        # Compute electrostatic energy
        E_elec = self.ewald(positions, charges, cell, pbc)
        
        # Compute short-range energy with charge awareness
        atomic_energies = self.short_range_net(positions, element_indices, cell, pbc)
        
        # Mix with charge information
        charge_feat = torch.full((positions.shape[0], 1), total_charge, 
                                  device=positions.device)
        mix_input = torch.cat([
            self.short_range_net.interaction_layers[-1].layer_norm(
                self.short_range_net.element_embedding(element_indices)
            ), 
            charge_feat
        ], dim=-1)
        atomic_energy_corrections = self.charge_mix(mix_input).squeeze(-1)
        
        E_sr = (atomic_energies + atomic_energy_corrections).sum()
        E_total = E_sr + E_elec
        
        return E_total, charges


def compute_forces(positions, energy_fn, create_graph=True):
    """Compute forces as negative gradient of energy w.r.t. positions.
    
    Args:
        positions: (N, 3) tensor of atomic positions
        energy_fn: function that computes energy from positions
        create_graph: whether to create computation graph for higher-order gradients
    
    Returns:
        (N, 3) tensor of forces
    """
    positions.requires_grad_(True)
    energy = energy_fn(positions)
    forces = -torch.autograd.grad(energy, positions, create_graph=create_graph)[0]
    return forces
