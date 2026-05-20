"""
Latent Ewald Summation (LES) Implementation

The LES method incorporates long-range electrostatic interactions in ML interatomic
potentials by predicting latent atomic charges via a neural network, then computing
the electrostatic energy using Ewald summation (or pairwise sums for finite systems).

Total energy = E_short-range + E_electrostatic (LES)

E_electrostatic = sum_{i<j} q_i * q_j / r_ij  (for finite systems)

where q_i are latent charges predicted by a neural network from local atomic
environment descriptors.
"""

import numpy as np
from typing import Tuple, Optional


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist


def compute_coulomb_energy(charges: np.ndarray, positions: np.ndarray) -> float:
    """
    Compute Coulomb energy for a finite system (no PBC).
    
    E = sum_{i<j} q_i * q_j / r_ij
    """
    n = len(charges)
    energy = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            if r_ij > 1e-10:  # Avoid division by zero
                energy += charges[i] * charges[j] / r_ij
    
    return energy


def compute_coulomb_energy_vec(charges: np.ndarray, positions: np.ndarray) -> float:
    """
    Vectorized Coulomb energy computation.
    
    E = 0.5 * sum_{i!=j} q_i * q_j / r_ij = sum_{i<j} q_i * q_j / r_ij
    """
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid self-interaction and zero distances
    np.fill_diagonal(dist, 1.0)  # Temporary fill
    
    # Coulomb matrix: q_i * q_j / r_ij
    charge_prod = np.outer(charges, charges)
    coulomb_matrix = charge_prod / dist
    
    # Remove diagonal
    np.fill_diagonal(coulomb_matrix, 0.0)
    
    return 0.5 * coulomb_matrix.sum()


def compute_coulomb_forces(charges: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Compute Coulomb forces on each atom.
    
    F_i = sum_{j!=i} q_i * q_j * (r_i - r_j) / |r_i - r_j|^3
    """
    n = len(charges)
    forces = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_vec = positions[i] - positions[j]
            r_mag = np.linalg.norm(r_vec)
            if r_mag > 1e-10:
                forces[i] += charges[i] * charges[j] * r_vec / r_mag**3
    
    return forces


def compute_coulomb_forces_vec(charges: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Vectorized Coulomb forces computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dist, 1.0)
    
    # Force: F_i = sum_j q_i * q_j * (r_i - r_j) / |r_i - r_j|^3
    charge_prod = np.outer(charges, charges)
    
    # Expand for broadcasting: charge_prod is (N,N), diff is (N,N,3), dist is (N,N)
    force_matrix = charge_prod[:, :, np.newaxis] * diff / dist[:, :, np.newaxis]**3
    np.fill_diagonal(force_matrix[:, :, 0], 0.0)
    np.fill_diagonal(force_matrix[:, :, 1], 0.0)
    np.fill_diagonal(force_matrix[:, :, 2], 0.0)
    
    forces = force_matrix.sum(axis=1)
    return forces


def ewald_summation(charges: np.ndarray, positions: np.ndarray, 
                     box_length: float, eta: float = 0.2,
                     n_real: int = 3, n_k: int = 5) -> float:
    """
    Ewald summation for periodic systems.
    
    Parameters:
    -----------
    charges : (N,) array of atomic charges
    positions : (N, 3) array of atomic positions
    box_length : length of cubic simulation box
    eta : Ewald splitting parameter
    n_real : number of real-space image shells
    n_k : number of k-space vectors in each direction
    
    Returns:
    --------
    Total electrostatic energy
    """
    n = len(charges)
    
    # Real-space contribution
    E_real = 0.0
    for nx in range(-n_real, n_real + 1):
        for ny in range(-n_real, n_real + 1):
            for nz in range(-n_real, n_real + 1):
                shift = np.array([nx, ny, nz]) * box_length
                for i in range(n):
                    for j in range(n):
                        if nx == 0 and ny == 0 and nz == 0 and i == j:
                            continue
                        r_vec = positions[i] - positions[j] + shift
                        r_mag = np.linalg.norm(r_vec)
                        if r_mag > 1e-10:
                            E_real += charges[i] * charges[j] * \
                                np.erfc(r_mag / (np.sqrt(2) * eta)) / r_mag
    
    E_real *= 0.5
    
    # Self-interaction correction
    E_self = -np.sum(charges**2) / (np.sqrt(2 * np.pi) * eta)
    
    # K-space contribution
    E_k = 0.0
    k_factor = 4 * np.pi / box_length**3
    
    for kx in range(-n_k, n_k + 1):
        for ky in range(-n_k, n_k + 1):
            for kz in range(-n_k, n_k + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                
                k_vec = 2 * np.pi * np.array([kx, ky, kz]) / box_length
                k_mag = np.linalg.norm(k_vec)
                
                # Structure factor
                S_real = np.sum(charges * np.cos(k_vec @ positions.T))
                S_imag = np.sum(charges * np.sin(k_vec @ positions.T))
                S2 = S_real**2 + S_imag**2
                
                E_k += k_factor * S2 * np.exp(-0.5 * k_mag**2 * eta**2) / k_mag**2
    
    E_k *= 0.5
    
    return E_real + E_k + E_self


class LESModel:
    """
    Latent Ewald Summation model.
    
    The model predicts latent charges q_les from atomic positions using
    a neural network, then computes the electrostatic energy via pairwise
    summation (for finite systems) or Ewald summation (for periodic systems).
    
    E_total = E_SR(x) + E_LR(q_les, x)
    
    where E_SR is the short-range energy and E_LR is the long-range
    electrostatic energy computed from the latent charges.
    """
    
    def __init__(self, n_channels: int = 1, use_pbc: bool = False):
        """
        Parameters:
        -----------
        n_channels : number of latent charge channels
        use_pbc : whether to use periodic boundary conditions
        """
        self.n_channels = n_channels
        self.use_pbc = use_pbc
        
    def predict_charges(self, elements: np.ndarray, positions: np.ndarray,
                       charge_network_params: dict = None) -> np.ndarray:
        """
        Predict latent charges from local environment.
        
        In the full LES model, this would be a neural network.
        Here we implement a simplified version using environmental descriptors.
        
        The charge prediction uses a linear combination of:
        - Element-specific base charges
        - Position-dependent corrections from local environment
        """
        n = len(elements)
        
        if charge_network_params is None:
            # Simple element-based charges (baseline)
            charges = np.zeros(n)
            # Assign charges based on element identity
            unique_elements = np.unique(elements)
            if len(unique_elements) > 1:
                for i, elem in enumerate(elements):
                    if i < n // 2:
                        charges[i] = 1.0
                    else:
                        charges[i] = -1.0
            else:
                # Single element - assign alternating charges
                charges = np.ones(n)
                charges[n//2:] = -1.0
            return charges
        
        # Neural network charge prediction
        # Input: atomic descriptors (element + local environment)
        # Output: latent charges
        
        # Simple 2-layer neural network
        W1 = charge_network_params['W1']  # (d_in, h)
        b1 = charge_network_params['b1']  # (h,)
        W2 = charge_network_params['W2']  # (h, n_channels)
        b2 = charge_network_params['b2']  # (n_channels,)
        
        # Compute environment descriptors
        descriptors = self._compute_descriptors(elements, positions)
        
        # Forward pass
        h = np.tanh(descriptors @ W1 + b1)
        charges = h @ W2 + b2
        
        return charges.flatten()
    
    def _compute_descriptors(self, elements: np.ndarray, 
                            positions: np.ndarray) -> np.ndarray:
        """
        Compute atom-centered descriptors for charge prediction.
        
        Uses a simplified SOAP-like descriptor based on:
        - Element identity (one-hot)
        - Radial distribution of neighbors
        - Angular features
        """
        n = len(elements)
        n_features = 8  # Descriptor dimension
        
        # Element encoding
        unique_elements = np.unique(elements)
        element_idx = {elem: i for i, elem in enumerate(unique_elements)}
        
        descriptors = np.zeros((n, n_features))
        
        for i in range(n):
            # Element one-hot (first 2 features)
            descriptors[i, element_idx[elements[i]] % 2] = 1.0
            
            # Radial features: sum of 1/r^n for neighbors
            for j in range(n):
                if i == j:
                    continue
                r_ij = np.linalg.norm(positions[i] - positions[j])
                if r_ij > 1e-10:
                    descriptors[i, 2] += 1.0 / r_ij**2
                    descriptors[i, 3] += 1.0 / r_ij**3
                    descriptors[i, 4] += 1.0 / r_ij**4
                    descriptors[i, 5] += 1.0 / r_ij
                    
                    # Coordination number within cutoff
                    if r_ij < 5.0:
                        descriptors[i, 6] += 1.0
            
            # Average neighbor distance
            if descriptors[i, 6] > 0:
                distances = []
                for j in range(n):
                    if i != j:
                        r_ij = np.linalg.norm(positions[i] - positions[j])
                        if r_ij < 5.0:
                            distances.append(r_ij)
                if distances:
                    descriptors[i, 7] = np.mean(distances)
        
        return descriptors
    
    def compute_long_range_energy(self, charges: np.ndarray, 
                                   positions: np.ndarray,
                                   box_length: float = None) -> float:
        """Compute long-range electrostatic energy from latent charges."""
        if self.use_pbc and box_length is not None:
            return ewald_summation(charges, positions, box_length)
        else:
            return compute_coulomb_energy_vec(charges, positions)
    
    def compute_long_range_forces(self, charges: np.ndarray,
                                   positions: np.ndarray) -> np.ndarray:
        """Compute long-range electrostatic forces from latent charges."""
        return compute_coulomb_forces_vec(charges, positions)
    
    def compute_total_energy(self, charges: np.ndarray,
                            positions: np.ndarray,
                            sr_energy: float = 0.0) -> float:
        """Compute total energy = short-range + long-range."""
        lr_energy = self.compute_long_range_energy(charges, positions)
        return sr_energy + lr_energy


def fit_charges_to_energy(charges_pred: np.ndarray, positions: np.ndarray,
                          target_energy: float, lr_only: bool = True) -> float:
    """
    Fit latent charges to match target energy.
    
    This is used for the random_charges dataset where we want to recover
    the true charges from energy data.
    """
    from scipy.optimize import minimize
    
    n = len(charges_pred)
    
    def objective(q):
        E = compute_coulomb_energy_vec(q, positions)
        return (E - target_energy)**2
    
    result = minimize(objective, charges_pred, method='L-BFGS-B',
                     bounds=[(-2, 2)] * n)
    
    return result.x


def charge_recovery_error(predicted: np.ndarray, true: np.ndarray) -> dict:
    """
    Compute metrics for charge recovery.
    
    Due to charge permutation ambiguity, we need to find the optimal alignment.
    """
    # Signed error
    signed_error = predicted - true
    mae = np.mean(np.abs(signed_error))
    rmse = np.sqrt(np.mean(signed_error**2))
    
    # Correlation
    if np.std(predicted) > 1e-10 and np.std(true) > 1e-10:
        correlation = np.corrcoef(predicted, true)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'signed_error': signed_error
    }
