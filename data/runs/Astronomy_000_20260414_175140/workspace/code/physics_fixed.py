import numpy as np

# Physical constants
MU_TO_M_CONST = 1.175e10  # mu [eV] * M [Msun] * const = mu M geom
GMSUN_SEC = 4.925490947e-6  # G M_sun / c^3 [s]
YR_SEC = 3.15576e7

def r_plus(a):
    return 1 + np.sqrt(1 - a**2)

def omega_I_M_scalar(alpha, a, l=1):
    # Small alpha Detweiler for l=m=1 scalar
    rp = r_plus(a)
    fact = a / (2 * rp)
    powr = alpha ** (4*l + 4)
    im_small = fact * powr / 48.0
    # Peak adjustment
    peak = 2.3e-7 * (a / 0.99)
    if alpha < 0.3:
        return im_small
    else:
        return np.max([im_small, peak * np.exp( -((alpha - 0.42)/0.15)**2 )])

def t_growth_yr(M_Msun, mu_eV, a):
    alpha = MU_TO_M_CONST * mu_eV * M_Msun
    imom = omega_I_M_scalar(alpha, a)
    if imom <= 0:
        return np.inf
    t_geom_M = 1 / imom
    t_sec = t_geom_M * GMSUN_SEC * M_Msun
    return t_sec / YR_SEC

def beta_max(g, alpha):
    # Dimless g = log10(lambda), lambda_crit ~ 10^3 / alpha^3 for Delta chi ~0.05
    lambda_c = 10**3 / alpha**3
    lambda_g = 10**g
    b = 0.05 * (lambda_c / lambda_g)
    return np.clip(b, 0, 0.05)

# Test
print('Test:')
print(t_growth_yr(10, 1e-12, 0.99))
print(beta_max(0, 0.4))
