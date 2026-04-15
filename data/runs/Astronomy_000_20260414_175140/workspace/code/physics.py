import numpy as np

# Constants
MU_CONST = 1.175e10  # mu_eV * M_Msun * MU_CONST = mu M dimensionless geom

GMsun_sec = 4.925490947e-6  # G M_sun / c^3 [sec]

def mu_M(mu_eV, M_Msun):
    return mu_eV * M_Msun * MU_CONST

def r_plus(a_star):
    return 1 + np.sqrt(1 - a_star**2)

def omega_I_M_scalar_small(alpha, a_star, l=1):
    \"\"\" Detweiler approx for small alpha, scalar l=m=1 dominant \"\"\"
    rpl = r_plus(a_star)
    factor = a_star / (2 * rpl)
    power = alpha ** (4*l + 4)
    return factor * power / 48.0

def omega_I_M_scalar(alpha, a_star):
    \"\"\" Approx combining small and peak \"\"\"
    im_small = omega_I_M_scalar_small(alpha, a_star)
    # Peak approx: max ~ 2.3e-7 for a=0.99 l=1 at alpha~0.42
    peak_val = 2.3e-7 * a_star / 0.99
    peak_alpha = 0.42
    if alpha > 0.8:
        # WKB rough exp(-alpha)
        return 1e-8 * np.exp(-1.5*(alpha - 0.42))
    elif alpha > 0.2:
        return np.maximum(im_small, peak_val * np.exp( - (alpha - peak_alpha)**2 / 0.1 ))
    else:
        return im_small

def t_growth_sec(M_Msun, mu_eV, a_star):
    alpha = mu_M(mu_eV, M_Msun)
    imom = omega_I_M_scalar(alpha, a_star)
    if imom <= 0:
        return np.inf
    t_geom = 1.0 / imom  # since omega in 1/M units
    return t_geom * GMsun_sec * M_Msun

def is_superradiant(a_star, alpha):
    # omega_R ~ mu (1 - alpha^2/2) < m Omega_H ~1 * a / (2 r+)
    omega_r_mu = 1 - 0.5 * alpha**2
    om_h = a_star / (2 * r_plus(a_star))
    return omega_r_mu < om_h

def beta_max(g, alpha):
    # Placeholder: beta_max decreases with |g|, assume gaussian or 1/(1+g^2)
    # From lit ~ 0.1 for g=0, drops when g phi^2 ~1
    # phi_rms ~ sqrt(N) mu r ~ alpha Mpl roughly
    # g_crit ~ 1 / (alpha Mpl)^2 ~ very small, but for ULB g defined as lambda (Mpl phi / f)^2 or dimless large allowed.
    # For now placeholder
    return 0.05 * np.exp( - np.abs(g)**2 / 10.0 )  # broad

# Test
print('Test 10 Msun, mu=1e-12 eV, a=0.99')
M = 10
mu = 1e-12
a = 0.99
alpha = mu_M(mu, M)
print('alpha:', alpha)
print('M Im omega:', omega_I_M_scalar(alpha, a))
print('t_growth [sec]:', t_growth_sec(M, mu, a))
print('t_growth [yr]:', t_growth_sec(M, mu, a) / (3.15576e7))
