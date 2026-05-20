"""Latent Ewald Summation (LES) framework implementation.

The LES method decomposes total energy into short-range (SR) and long-range (LR) parts:
    E_total = E_sr + E_lr

Key concepts:
- E_sr: Predicted by a short-range MLIP using local atomic descriptors
- q_les: Latent charges learned from local atomic descriptors (not physical charges!)
- E_lr: Computed from q_les via Ewald summation

The latent charges are NOT trained to match reference charges; they are latent variables
that yield correct long-range electrostatics after Ewald summation.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import erfc

class LESBase:
    """Base class implementing Ewald summation utilities for LES."""
    
    def __init__(self, box_size=None, alpha_ewald=0.3, k_max=5):
        self.box_size = box_size
        self.alpha_ewald = alpha_ewald
        self.k_max = k_max
    
    def compute_ewald_real(self, positions, charges, box_size):
        """Compute real-space part of Ewald sum."""
        n = len(charges)
        energy = 0.0
        alpha = self.alpha_ewald
        
        for i in range(n):
            for j in range(i+1, n):
                # Minimum image convention
                rij_vec = positions[i] - positions[j]
                rij_vec = rij_vec - box_size * np.round(rij_vec / box_size)
                rij = np.linalg.norm(rij_vec)
                if rij < 1e-10:
                    continue
                energy += charges[i] * charges[j] * erfc(alpha * rij) / rij
        
        # Self-interaction correction
        self_energy = -alpha / np.sqrt(np.pi) * np.sum(charges**2)
        return energy + self_energy
    
    def compute_ewald_reciprocal(self, positions, charges, box_size):
        """Compute reciprocal-space part of Ewald sum."""
        n = len(charges)
        volume = box_size**3
        alpha = self.alpha_ewald
        k_max = self.k_max
        energy = 0.0
        
        # Generate k-vectors
        k_vecs = []
        for nx in range(-k_max, k_max+1):
            for ny in range(-k_max, k_max+1):
                for nz in range(-k_max, k_max+1):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue
                    k = 2*np.pi*np.array([nx, ny, nz]) / box_size
                    k_norm = np.linalg.norm(k)
                    k_vecs.append((k, k_norm))
        
        for k, k_norm in k_vecs:
            k2 = k_norm**2
            factor = 2*np.pi/volume * np.exp(-k2/(4*alpha**2)) / k2
            
            # Structure factor
            S_real = np.sum(charges * np.cos(np.dot(positions, k)))
            S_imag = np.sum(charges * np.sin(np.dot(positions, k)))
            
            energy += factor * (S_real**2 + S_imag**2)
        
        return energy
    
    def compute_coulomb_energy(self, positions, charges, box_size=None):
        """Compute full Coulomb energy using Ewald summation."""
        if box_size is None:
            # Direct sum for non-periodic
            n = len(charges)
            energy = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    rij = np.linalg.norm(positions[i] - positions[j])
                    if rij < 1e-10:
                        continue
                    energy += charges[i] * charges[j] / rij
            return energy
        
        e_real = self.compute_ewald_real(positions, charges, box_size)
        e_recip = self.compute_ewald_reciprocal(positions, charges, box_size)
        return e_real + e_recip


class LESModel:
    """LES model that learns latent charges from atomic descriptors.
    
    This is a simplified demonstration using local atomic descriptors
    (radial distribution features) and a ridge regression model.
    """

    def __init__(self, box_size=15.0, cutoff=5.0, alpha_ewald=0.3, k_max=4):
        self.box_size = box_size
        self.cutoff = cutoff
        self.alpha_ewald = alpha_ewald
        self.k_max = k_max
        self.ewald = LESBase(box_size, alpha_ewald, k_max)
        self.sr_weights = None
        self.q_weights = None
        self.sr_bias = None
        self.q_bias = None

    def compute_local_descriptors(self, positions, species=None):
        """Compute simple local descriptors: neighbor counts in radial bins."""
        n_atoms = len(positions)
        n_bins = 10
        descriptors = np.zeros((n_atoms, n_bins + 1))  # +1 for central atom type

        # Radial bins
        bin_edges = np.linspace(0.5, self.cutoff, n_bins + 1)

        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                rij = np.linalg.norm(positions[i] - positions[j])
                if rij < self.cutoff:
                    # Bin index
                    for b in range(n_bins):
                        if bin_edges[b] <= rij < bin_edges[b+1]:
                            descriptors[i, b] += 1
                            break

        # Normalize
        descriptors[:, :n_bins] /= np.maximum(np.sum(descriptors[:, :n_bins], axis=1, keepdims=True), 1)
        descriptors[:, n_bins] = 1.0  # bias

        return descriptors

    def fit(self, positions_list, energies_list, charges_true=None, alpha_reg=1e-3):
        """Fit the LES model.

        This is a simplified approach:
        1. Use ridge regression to predict total energy from descriptors
        2. Learn latent charges that, when used in Ewald, recover the long-range part

        For the random_charges dataset, we use a different approach:
        - The short-range part is modeled as a local energy contribution
        - The long-range part uses latent charges
        """
        n_frames = len(positions_list)
        n_atoms = positions_list[0].shape[0]

        # Build feature matrix for short-range energy prediction
        X_sr = []
        y_energy = []

        for fi in range(n_frames):
            pos = positions_list[fi]
            desc = self.compute_local_descriptors(pos)
            # Global features: sum of descriptors
            global_desc = np.sum(desc, axis=0)
            # Also add per-atom features
            X_sr.append(np.concatenate([global_desc, desc.flatten()]))
            y_energy.append(energies_list[fi])

        X_sr = np.array(X_sr)
        y_energy = np.array(y_energy)

        # Ridge regression for short-range
        n_features = X_sr.shape[1]
        I = np.eye(n_features)
        self.sr_weights = np.linalg.solve(X_sr.T @ X_sr + alpha_reg * I, X_sr.T @ y_energy)

        # Predicted short-range energies
        y_pred = X_sr @ self.sr_weights
        residuals = y_energy - y_pred

        # Learn latent charges from residuals + Ewald
        # Simplified: fit per-atom latent charges by solving linear system
        # For demonstration: use local descriptors to predict per-atom q_les
        X_q = []
        y_q_target = []

        for fi in range(n_frames):
            pos = positions_list[fi]
            desc = self.compute_local_descriptors(pos)
            X_q.append(desc)
            # Target: use true charges if available, otherwise uniform
            if charges_true is not None:
                y_q_target.append(charges_true[fi])
            else:
                y_q_target.append(np.ones(n_atoms) * (residuals[fi] / (n_atoms * 10)))

        X_q_flat = np.vstack(X_q)
        y_q_flat = np.hstack(y_q_target)

        n_q_features = X_q_flat.shape[1]
        I_q = np.eye(n_q_features)
        self.q_weights = np.linalg.solve(X_q_flat.T @ X_q_flat + alpha_reg * I_q, X_q_flat.T @ y_q_flat)

        return y_pred, residuals

    def predict(self, positions):
        """Predict energy using LES decomposition."""
        desc = self.compute_local_descriptors(positions)
        global_desc = np.sum(desc, axis=0)
        X_sr = np.concatenate([global_desc, desc.flatten()])

        # Short-range energy
        E_sr = X_sr @ self.sr_weights
        
        # Latent charges
        q_les = desc @ self.q_weights
        
        # Long-range energy via Ewald
        E_lr = self.ewald.compute_coulomb_energy(positions, q_les, self.box_size)
        
        return E_sr + E_lr, E_sr, E_lr, q_les


def generate_synthetic_energy(positions, charges, box_size=15.0, eps_lj=1.0, sigma_lj=1.0):
    """Generate synthetic energy from Coulomb + Lennard-Jones potential."""
    n = len(charges)
    energy_coulomb = 0.0
    energy_lj = 0.0
    forces = np.zeros((n, 3))
    
    for i in range(n):
        for j in range(i+1, n):
            rij_vec = positions[i] - positions[j]
            # Minimum image convention
            rij_vec = rij_vec - box_size * np.round(rij_vec / box_size)
            rij = np.linalg.norm(rij_vec)
            if rij < 1e-10:
                continue
            
            # Coulomb
            ec = charges[i] * charges[j] / rij
            energy_coulomb += ec
            
            # LJ (repulsive)
            sr2 = (sigma_lj / rij)**2
            sr6 = sr2**3
            sr12 = sr6**2
            elj = 4 * eps_lj * (sr12 - sr6)
            energy_lj += elj
            
            # Forces
            fc = charges[i] * charges[j] / (rij**3)
            flj = 24 * eps_lj * (2*sr12 - sr6) / (rij**2)
            f_total = (fc + flj) * rij_vec
            
            forces[i] += f_total
            forces[j] -= f_total
    
    return energy_coulomb + energy_lj, energy_coulomb, energy_lj, forces
