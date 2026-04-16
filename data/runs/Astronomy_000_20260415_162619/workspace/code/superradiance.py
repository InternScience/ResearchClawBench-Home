import numpy as np
import math

def compute_gamma(M_sol, a_star, mu_eV, l=1, m=1, n=0):
    """
    Compute superradiance rate Gamma for a given state (l, m, n).
    M_sol: Black hole mass in solar masses
    a_star: Dimensionless spin (0 <= a_star < 1)
    mu_eV: Boson mass in eV
    Returns Gamma in s^-1.
    """
    # Constants
    G = 6.67430e-11 # m^3 kg^-1 s^-2
    c = 2.99792458e8 # m/s
    M_sun = 1.98847e30 # kg
    hbar_J = 1.054571817e-34 # J s
    eV_to_J = 1.602176634e-19
    hbar_eV = hbar_J / eV_to_J # eV s

    # Mass of BH in kg
    M_kg = M_sol * M_sun
    
    # Gravitational radius in meters
    r_g = G * M_kg / c**2
    
    # Mass of boson in kg
    mu_kg = mu_eV * eV_to_J / c**2
    
    # Dimensionless coupling alpha = G M mu / (hbar c)
    alpha = G * M_kg * mu_kg / (hbar_J * c)
    
    # Event horizons r_plus_hat = r_+ / r_g
    r_plus_hat = 1 + np.sqrt(1 - a_star**2)
    
    # Horizon angular velocity w_plus_hat = Omega_H * r_g / c
    w_plus_hat = a_star / (2 * r_plus_hat)
    
    # Boson mass in natural units mu_a_hat = mu * r_g / hbar c = alpha
    mu_a_hat = alpha
    
    # Check superradiance condition (omega < m * Omega_H)
    # omega_hat = mu_a_hat (in the non-relativistic approximation)
    if mu_a_hat >= m * w_plus_hat:
        return 0.0 # No superradiance
        
    # Rate approximation from Arvanitaki et al. Eq (18)
    term1 = (2**(4*l+2) * math.factorial(2*l+n+1)) / ((l+n+1)**(2*l+4) * math.factorial(n))
    term2 = (math.factorial(l) / (math.factorial(2*l) * math.factorial(2*l+1)))**2
    
    prod = 1.0
    for j in range(1, l+1):
        prod *= (j**2 * (1 - a_star**2) + 4 * r_plus_hat**2 * (m * w_plus_hat - mu_a_hat)**2)
        
    C_lmn = term1 * term2 * prod
    
    Gamma_hat = 2 * mu_a_hat * (alpha**(4*l+4)) * r_plus_hat * (m * w_plus_hat - mu_a_hat) * C_lmn
    
    # Gamma_hat is dimensionless (Gamma * r_g / c).
    # Gamma in s^-1 is Gamma_hat * c / r_g
    Gamma_s = Gamma_hat * c / r_g
    
    return Gamma_s

