"""Background cosmology utilities (flat FLRW, with optional w0wa or EDE-like
fluid). Distances in Mpc, H in km/s/Mpc, c = 299792.458 km/s.

EDE is treated phenomenologically as an extra component with f_EDE at a_c that
behaves like a cosmological-constant-like fluid for a<a_c and dilutes as a^-6
for a>a_c (n=3 axion-like, see Poulin et al. 2018). This captures the impact
on the sound horizon while staying lightweight.
"""
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid

C_KMS = 299792.458


def E_LCDM(z, Om):
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def E_w0wa(z, Om, w0, wa):
    a = 1.0 / (1 + z)
    # de density evolution: rho_de(a)/rho_de0 = a^{-3(1+w0+wa)} exp(-3 wa (1-a))
    rho_de = a ** (-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om) * rho_de)


def E_EDE(z, Om, f_EDE, log10_ac, n=3):
    """Phenomenological axion-like EDE: extra fluid with peak fraction f_EDE
    at scale factor a_c, equation of state transitioning from -1 (a<a_c) to
    (n-1)/(n+1) ~ 0.5 (a>a_c, n=3). Implementation follows the simple peaked
    fraction approximation in Poulin et al. 2019.
    """
    a = 1.0 / (1 + z)
    a_c = 10 ** log10_ac
    # Background fraction in EDE component
    # f(a) = 2*f_EDE / ((a/a_c)^(3(w_n+1)) + 1) with w_n -> 1 effective
    # Use the Karwal/Poulin parametrisation
    Om_a = Om * (1 + z) ** 3
    Or_a = 0.0  # neglect radiation for late distances (z<3)
    OL_a = (1 - Om)  # background DE absorbed into LCDM-like remainder
    rho_lcdm = Om_a + OL_a + Or_a
    # Peaked EDE density: rho_ede(a) = rho_ede0 * 2 / ((a/a_c)^(3(1+w_n))+1)
    # For axion n=3 -> w_n = 1/2 -> 3(1+w_n)=4.5
    expn = 3 * (1 + 0.5)  # = 4.5
    peak = 2.0 / ((a / a_c) ** expn + 1.0)
    # Normalize: at a=a_c the peak is 1, and f_EDE(a_c) = f_EDE
    rho_ede = f_EDE * peak / (1.0 - f_EDE) * rho_lcdm * 0  # see below
    # Simpler: define total rho/rho_crit0 = rho_lcdm / (1 - f_EDE_local)
    f_local = f_EDE * peak  # fraction at a
    # avoid division blow up
    f_local = np.minimum(f_local, 0.95)
    return np.sqrt(rho_lcdm / np.maximum(1.0 - f_local, 1e-3))


def comoving_distance(z, Efun, *args):
    """D_C(z) in Mpc."""
    z_arr = np.atleast_1d(z).astype(float)
    out = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        val, _ = quad(lambda zp: 1.0 / Efun(zp, *args), 0.0, zi, limit=200)
        out[i] = val
    H0 = args_H0_helper.get('H0', 67.36)  # placeholder, set externally
    return C_KMS / H0 * out


# Avoid global state: provide explicit functions taking H0
def D_C(z, H0, Efun, *args):
    z_arr = np.atleast_1d(z).astype(float)
    out = np.zeros_like(z_arr)
    for i, zi in enumerate(z_arr):
        val, _ = quad(lambda zp: 1.0 / Efun(zp, *args), 0.0, zi, limit=200)
        out[i] = val
    return C_KMS / H0 * out


def D_M(z, H0, Efun, *args):
    return D_C(z, H0, Efun, *args)  # flat universe


def D_H(z, H0, Efun, *args):
    z_arr = np.atleast_1d(z).astype(float)
    return C_KMS / (H0 * Efun(z_arr, *args))


def D_V(z, H0, Efun, *args):
    dm = D_M(z, H0, Efun, *args)
    dh = D_H(z, H0, Efun, *args)
    z_arr = np.atleast_1d(z).astype(float)
    return (z_arr * dm ** 2 * dh) ** (1.0 / 3.0)


def F_AP(z, H0, Efun, *args):
    return D_M(z, H0, Efun, *args) / D_H(z, H0, Efun, *args)


def mu_dist(z, H0, Efun, *args):
    """Distance modulus, flat universe."""
    z_arr = np.atleast_1d(z).astype(float)
    dl = (1 + z_arr) * D_M(z_arr, H0, Efun, *args)  # Mpc
    return 5 * np.log10(dl) + 25.0


# placeholder dict (not actually used)
args_H0_helper = {'H0': 67.36}
