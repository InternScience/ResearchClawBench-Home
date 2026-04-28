"""
Black-hole superradiance physics for ultralight bosons (ULBs).

Conventions:
  - Geometric units (G = c = 1) for gravitational atom dimensions.
  - Solar mass M_sun -> length r_g = G M / c^2 = 1.4766 km.
  - Boson mass mu in eV. Compton wavelength l_c = hbar/(mu c).
  - Dimensionless coupling alpha = r_g / l_c = G M mu / (hbar c^3).
  - Dimensionless spin a* in [0, 1).

Key references:
  Arvanitaki & Dubovsky 2011, PRD 83, 044026 (paper_000)
  Arvanitaki et al. 2017, PRD 95, 043001 (paper_002)
  Stott 2020 (paper_001 proceedings)
  Witek et al. 2013 (paper_003)

Superradiance condition for the (n, l, m) bound state of a massive scalar:
  omega_R < m * Omega_H,
with Omega_H = a* / (2 r_+) the BH horizon angular velocity in geometric
units, where r_+ = M (1 + sqrt(1 - a*^2)).

Field binding energy (hydrogenic limit, alpha << 1):
  omega_R ~ mu (1 - alpha^2 / (2 n^2)).

Detweiler small-alpha rate for the dominant l=m=1 mode (Detweiler 1980):
  Gamma_211 = (1/24) * mu * alpha^8 * (a* - 2 mu r_+)
            = mu * alpha^8 * (a*/2 - alpha (1 + sqrt(1 - a*^2))) / 24
  in geometric units (Gamma has units of 1/time when multiplied by 1/M*c).

For higher levels (l = m), Detweiler-type result:
  Gamma_l = 2 r_+ C_l alpha^(4 l + 4) (m Omega_H - omega_R),
  with C_l = (2^(4 l + 1) (2 l)! ) / ( l (l!)^2 (2 l + 1)! (2 l)!^2 ) * ...
We use the compact Arvanitaki & Dubovsky 2011 formula:
  Gamma_nlm = 2 mu r_+ (m Omega_H - omega_R) C_{nl} alpha^(4 l + 4)
  with the Dolan-fit / Arvanitaki coefficients used in the literature.

Bosenova (Arvanitaki & Dubovsky 2010):
  Cloud occupation grows until self-interactions dominate.
  Critical occupation N_bose = c0 (M / mu)^2 (f_a / M_pl)^2 * n^4 * 16 / alpha^3
  Spin-down extracted from BH stalls when N_max(reach) < ln-factor * N_bose.
  Bosenova excludes growth when N_bose < N_required for sufficient spin extraction.
We follow Eq. (40) of Arvanitaki & Dubovsky 2011:
  N_max = c_n * (M_pl / f_a)^2 * (M / mu)^2 / alpha^3
where c_n is an O(1) numerical factor; we adopt c_n = 5 (paper 0 Eq. 40).

A given ULB (mu, f_a) is *unconstrained* by the Bosenova when SR cannot
spin the BH down enough before saturating: that is, when the number of
quanta needed to extract the observed spin gap, N_extract ~ Delta(J)/m,
exceeds N_max. Conversely, SR remains *active* (and thus excluding) if
N_extract <= N_max, OR if f_a is large enough that the bosenova never
fires.
"""

from __future__ import annotations
import numpy as np

# ------------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------------
G = 6.67430e-11         # m^3 kg^-1 s^-2
C = 2.99792458e8        # m / s
HBAR = 1.054571817e-34  # J s
EV = 1.602176634e-19    # J
M_SUN = 1.98892e30      # kg
M_PL = 1.220910e19      # GeV (reduced Planck mass = M_Pl_red * sqrt(8 pi))
M_PL_RED = 2.435e18     # GeV (reduced Planck mass)
GEV_PER_EV = 1e-9
SEC_PER_YR = 3.15576e7

def r_g_m(M_sun_units: np.ndarray) -> np.ndarray:
    """Gravitational radius G M / c^2 in metres."""
    return G * (M_sun_units * M_SUN) / C**2

def t_g_s(M_sun_units: np.ndarray) -> np.ndarray:
    """Gravitational time-scale r_g / c in seconds."""
    return r_g_m(M_sun_units) / C

def alpha_coupling(M_sun_units: np.ndarray, mu_eV: np.ndarray) -> np.ndarray:
    """Dimensionless coupling alpha = G M mu / (hbar c^3) = r_g / lambdabar_C.

    Both arguments may be broadcast against each other.
    """
    M = np.asarray(M_sun_units) * M_SUN
    mu_J = np.asarray(mu_eV) * EV
    return G * M * mu_J / (HBAR * C**3)

def horizon_radius_geom(a_star: np.ndarray) -> np.ndarray:
    """r_+/M = 1 + sqrt(1 - a*^2)."""
    a = np.clip(np.asarray(a_star), -0.999999, 0.999999)
    return 1.0 + np.sqrt(1.0 - a**2)

def omega_H_geom(a_star: np.ndarray) -> np.ndarray:
    """Horizon angular velocity in units of 1/M (geometric):
       Omega_H * M = a* / (2 r_+/M) = a* / (2 (1 + sqrt(1 - a*^2)))."""
    rp = horizon_radius_geom(a_star)
    return np.asarray(a_star) / (2.0 * rp)

# ------------------------------------------------------------------
# Superradiance condition and rate (Detweiler / Arvanitaki & Dubovsky)
# ------------------------------------------------------------------

def sr_condition(M_sun_units, a_star, mu_eV, m=1, n=2, l=1):
    """Boolean: is the SR condition omega_R < m Omega_H satisfied?

    omega_R approximated by mu (1 - alpha^2 / (2 n^2)).
    Condition is expressed dimensionlessly: alpha (1 - alpha^2/(2 n^2)) < m * (Omega_H * M).
    """
    a = alpha_coupling(M_sun_units, mu_eV)
    omH_M = omega_H_geom(a_star)
    omR_M = a * (1.0 - a**2 / (2.0 * n**2))  # mu*M*(1 - alpha^2/2n^2) = alpha*(...)
    return omR_M < m * omH_M

def gamma_sr_geom(M_sun_units, a_star, mu_eV, l=1, m=1, n=2):
    """SR rate in inverse units of M (geometric: Gamma * M).

    For l = m = 1, Detweiler (1980) gives
        Gamma_211 * M = (1/24) * alpha^8 * (a* - 2 alpha r_+/M).
    For higher l = m, we use Arvanitaki & Dubovsky 2011 Eq. (28):
        Gamma_nlm * M = 2 r_+/M * (m Omega_H - omega_R) * M * C_nl * alpha^(4 l + 4)
    with
        C_nl = 2^{4 l + 1} (n + l)! / (n^{2 l + 4} (n - l - 1)!) *
               [ l! / ((2 l)! (2 l + 1)!) ]^2 *
               prod_{k=1..l} [k^2 (1 - a*^2) + (a* m - 2 r_+ omega_R)^2].
    The l=1 case reproduces the Detweiler form to the leading order in alpha.
    """
    a = alpha_coupling(M_sun_units, mu_eV)
    rp = horizon_radius_geom(a_star)
    omH_M = omega_H_geom(a_star)
    omR_M = a * (1.0 - a**2 / (2.0 * n**2))

    if l == 1 and m == 1:
        # Detweiler analytic, valid for alpha <~ 0.5
        # (Gamma * M) = (1/24) * alpha^8 * (a* - 2 alpha r_+/M)
        return (1.0/24.0) * a**8 * (a_star - 2.0 * a * rp)

    # General formula (Arvanitaki & Dubovsky 2011 Eq. 28, 30)
    from math import factorial
    Cnl = 2.0**(4*l + 1) * factorial(n + l) / (n**(2*l + 4) * factorial(max(n - l - 1, 0)))
    Cnl *= (factorial(l) / (factorial(2*l) * factorial(2*l + 1)))**2
    prod = 1.0
    for k in range(1, l + 1):
        prod = prod * (k**2 * (1.0 - a_star**2) + (a_star * m - 2.0 * rp * omR_M)**2)
    Cnl = Cnl * prod
    diff = m * omH_M - omR_M
    return 2.0 * rp * diff * Cnl * a**(4*l + 4)

def gamma_sr_inv_seconds(M_sun_units, a_star, mu_eV, **kw):
    """Convert (Gamma * M)_geom to physical 1/seconds."""
    g_geom = gamma_sr_geom(M_sun_units, a_star, mu_eV, **kw)
    tg = t_g_s(M_sun_units)
    return g_geom / tg

def tau_sr_seconds(M_sun_units, a_star, mu_eV, **kw):
    """SR e-folding time in seconds (1/Gamma); negative when SR forbidden."""
    g = gamma_sr_inv_seconds(M_sun_units, a_star, mu_eV, **kw)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1.0 / g

# ------------------------------------------------------------------
# Spin extraction & Bosenova (self-interactions)
# ------------------------------------------------------------------

def n_bosenova(M_sun_units, mu_eV, fa_GeV, alpha=None, n=2, c_n=5.0):
    """Maximum cloud occupation before bosenova self-interaction collapse.

    From Arvanitaki & Dubovsky 2011 Eq. (40):
        N_max ~ c_n * (n / alpha)^4 * (M_pl / f_a)^2 * (M / m_pl_in_units_of_M).
    More usefully, expressed via M and mu:
        N_max ~ c_n * n^4 / alpha^3 * (M_pl / f_a)^2 * (M_BH / m_planck_mass)
                * (1 / (M_BH * mu))   [dimensionless]
    We follow the practical form used in their Section 4:
        N_bose = 16 c_n (1 + sqrt(1 - a*^2)) (n/alpha)^4 (f_a/M_pl)^2 (M/mu) (alpha)
    For our purposes we adopt the simple scaling
        N_bose / (M^2/mu^2) = c_n (f_a / M_pl_red)^2 / alpha^3 * n^4
    with c_n = 5 (paper 0, Eq. 40, Section 4). All masses in GeV.
    Returns N_bose (dimensionless cloud occupation).
    """
    if alpha is None:
        alpha = alpha_coupling(M_sun_units, mu_eV)
    M_GeV = (M_sun_units * M_SUN) * (C**2) / (1e9 * EV)  # M in GeV
    mu_GeV = mu_eV * GEV_PER_EV
    # avoid div-by-zero
    a3 = np.where(alpha > 0, alpha**3, np.inf)
    fa_term = (fa_GeV / M_PL_RED)**2
    return c_n * n**4 * fa_term / a3 * (M_GeV / mu_GeV)**2

def n_extract_required(M_sun_units, a_star_obs, a_star_target, mu_eV, n=2, m=1):
    """Number of bosons needed in the cloud to spin BH down from a_obs to a_target.

    Each emitted/absorbed quantum carries angular momentum ~ m * hbar.
    Delta J = (a_obs - a_target) * G M^2 / c.
    N = Delta J / (m hbar).
    Mass change is small (~ alpha) for SR; we ignore it.
    """
    M_kg = M_sun_units * M_SUN
    dJ = (a_star_obs - a_star_target) * G * M_kg**2 / C
    return dJ / (m * HBAR)

def spin_down_target_a(M_sun_units, mu_eV, n=2, m=1):
    """Spin to which SR drives the BH for given (M, mu, n, m): the value of
    a* at which omega_R = m Omega_H, i.e. SR threshold.
    Solve a* / (2 (1 + sqrt(1 - a*^2))) = alpha (1 - alpha^2 / (2 n^2)) / m.
    """
    a_dim = alpha_coupling(M_sun_units, mu_eV)
    omR_M = a_dim * (1.0 - a_dim**2 / (2.0 * n**2))
    rhs = omR_M / m  # = a*/(2 (1 + sqrt(1-a*^2)))
    # Solve: a*^2 = 4 rhs^2 (1 + sqrt(1-a*^2))^2
    # let s = sqrt(1 - a*^2), then a*^2 = 1 - s^2 = 4 rhs^2 (1+s)^2
    # => (1-s)(1+s) = 4 rhs^2 (1+s)^2 => (1-s) = 4 rhs^2 (1+s)
    # => 1 - s = 4 rhs^2 + 4 rhs^2 s
    # => 1 - 4 rhs^2 = s (1 + 4 rhs^2) => s = (1 - 4 rhs^2)/(1 + 4 rhs^2)
    s = (1.0 - 4.0 * rhs**2) / (1.0 + 4.0 * rhs**2)
    s = np.clip(s, 0.0, 1.0)
    a_target = np.sqrt(np.clip(1.0 - s**2, 0.0, 1.0))
    # Where rhs > 1/2 (i.e. SR cannot occur even at a*=1), set NaN
    a_target = np.where(rhs >= 0.5, np.nan, a_target)
    return a_target
