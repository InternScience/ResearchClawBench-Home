#!/usr/bin/env python3
"""
MMGA Parameter Identification for Li-ion Battery ECAT Model.
Complete framework: ECAT model + LHS + ANN surrogate + GA optimization + validation.
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import os, json, time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# ECAT MODEL (Single Particle Model + Lumped Thermal)
# ═══════════════════════════════════════════════════════════════
PARAM_NAMES = [
    'Rp_neg',     # 0: Negative particle radius (m)
    'Rp_pos',     # 1: Positive particle radius (m)
    'k0_neg',     # 2: Negative reaction rate (m/s)
    'k0_pos',     # 3: Positive reaction rate (m/s)
    'Ds_neg',     # 4: Negative solid diffusivity (m^2/s)
    'Ds_pos',     # 5: Positive solid diffusivity (m^2/s)
    'sigma_neg',  # 6: Negative electronic conductivity (S/m)
    'sigma_pos',  # 7: Positive electronic conductivity (S/m)
    'eps_s_neg',  # 8: Negative active material fraction
    'eps_s_pos',  # 9: Positive active material fraction
    'L_neg',      # 10: Negative electrode thickness (m)
    'L_pos',      # 11: Positive electrode thickness (m)
    'h_conv',     # 12: Convective HTC (W/m^2/K)
    'Cp_cell',    # 13: Cell heat capacity (J/K)
    'R_ohmic',    # 14: Total ohmic resistance (Ohm)
    'alpha_neg',  # 15: Negative charge transfer coeff
    'alpha_pos',  # 16: Positive charge transfer coeff
]
N_PARAMS = len(PARAM_NAMES)

# Parameter bounds: [lower, upper]
BOUNDS_LOWER = np.array([
    2e-7, 2e-7,           # Rp_neg, Rp_pos (m)
    1e-7, 1e-7,           # k0_neg, k0_pos (m/s)
    1e-14, 1e-14,         # Ds_neg, Ds_pos (m^2/s)
    10.0, 1.0,            # sigma_neg, sigma_pos (S/m)
    0.35, 0.35,           # eps_s_neg, eps_s_pos
    40e-6, 40e-6,         # L_neg, L_pos (m)
    5.0, 0.5,             # h_conv (W/m^2/K), Cp_cell (J/K)
    0.005,                # R_ohmic (Ohm)
    0.4, 0.4,             # alpha_neg, alpha_pos
])
BOUNDS_UPPER = np.array([
    3e-6, 3e-6,
    1e-5, 1e-5,
    5e-13, 5e-13,
    100.0, 10.0,
    0.65, 0.65,
    80e-6, 80e-6,
    50.0, 5.0,
    0.15,
    0.6, 0.6,
])

F_C = 96485.3329
R_C = 8.314462

def ocv_neg(theta):
    """Graphite anode OCP (vs Li/Li+)."""
    th = np.clip(theta, 0.005, 0.995)
    return np.clip(
        0.124 + 1.5*np.exp(-80*(th-0.13))
        + 0.035*np.tanh((th-0.10)/0.01)
        - 0.012*np.tanh((th-0.30)/0.01)
        + 0.008*np.tanh((th-0.50)/0.05)
        - 0.005*np.tanh((th-0.70)/0.05),
        0.005, 0.40)

def ocv_pos(theta):
    """NMC cathode OCP (vs Li/Li+)."""
    th = np.clip(theta, 0.005, 0.995)
    return np.clip(
        3.4 + 0.5*(1-th) + 0.15*np.log(th/(1-th+0.01)),
        3.0, 4.3)

def simulate_ecat(pv, I_app=1.1, dt=1.0, T_amb=298.15, n_nodes=5):
    """Simulate CC discharge with ECAT model."""
    p = dict(zip(PARAM_NAMES, pv))
    A_cell = 0.015  # m^2 (cell cross-section)
    cs_max_n = 33133.0  # mol/m^3
    cs_max_p = 37800.0  # mol/m^3

    # Estimate max discharge time
    Q_neg_cap = A_cell * p['L_neg'] * p['eps_s_neg'] * cs_max_n * F_C
    Q_pos_cap = A_cell * p['L_pos'] * p['eps_s_pos'] * cs_max_p * F_C
    Q_avail = min(Q_neg_cap * 0.73, Q_pos_cap * 0.65)
    n_steps = min(max(int(Q_avail / I_app / dt), 100), 3000)

    dr_n = p['Rp_neg'] / n_nodes
    dr_p = p['Rp_pos'] / n_nodes

    # Initial solid concentrations
    cs_n = np.ones(n_nodes) * 0.85 * cs_max_n
    cs_p = np.ones(n_nodes) * 0.25 * cs_max_p
    T_cell = T_amb
    Q_discharged = 0.0

    t_out = np.zeros(n_steps)
    v_out = np.zeros(n_steps)
    T_out = np.zeros(n_steps)
    c_out = np.zeros(n_steps)

    for s in range(n_steps):
        th_n = cs_n[-1] / cs_max_n
        th_p = cs_p[-1] / cs_max_p

        Un = ocv_neg(th_n)
        Up = ocv_pos(th_p)

        # Volumetric reaction area
        a_n = 3.0 * p['eps_s_neg'] / p['Rp_neg']
        a_p = 3.0 * p['eps_s_pos'] / p['Rp_pos']

        # Volumetric current density
        j_n = I_app / (A_cell * p['L_neg'] * a_n + 1e-30)
        j_p = I_app / (A_cell * p['L_pos'] * a_p + 1e-30)

        # Exchange current density (Butler-Volmer)
        an_coeff = p['alpha_neg']
        ap_coeff = p['alpha_pos']
        # k0 is in m/s, multiply by F*C to get A/m^2 exchange
        i0n = F_C * p['k0_neg'] * ((cs_max_n - cs_n[-1])**an_coeff * cs_n[-1]**(1-an_coeff))
        i0p = F_C * p['k0_pos'] * ((cs_max_p - cs_p[-1])**ap_coeff * cs_p[-1]**(1-ap_coeff))
        i0n = max(i0n, 1e-10)
        i0p = max(i0p, 1e-10)

        # Overpotentials (inverse sinh for stability)
        arg_n = np.clip(j_n / (2.0 * i0n), -50, 50)
        arg_p = np.clip(j_p / (2.0 * i0p), -50, 50)
        eta_n = (R_C * T_cell / (an_coeff * F_C)) * np.arcsinh(arg_n)
        eta_p = (R_C * T_cell / (ap_coeff * F_C)) * np.arcsinh(arg_p)

        # Terminal voltage
        V = Up + eta_p - Un + eta_n - I_app * p['R_ohmic']
        V = np.clip(V, 2.5, 4.3)

        # Solid-state diffusion
        Dn = p['Ds_neg'] * (1.0 + 0.2 * (1.0 - th_n))
        Dp = p['Ds_pos'] * (1.0 + 0.2 * (1.0 - th_p))

        cn, cp = cs_n.copy(), cs_p.copy()
        for r in range(1, n_nodes - 1):
            cn[r] += Dn * dt / (dr_n**2 + 1e-30) * (cs_n[r+1] - 2*cs_n[r] + cs_n[r-1])
            cp[r] += Dp * dt / (dr_p**2 + 1e-30) * (cs_p[r+1] - 2*cs_p[r] + cs_p[r-1])
        # Center: symmetry
        cn[0] = cn[1]
        cp[0] = cp[1]
        # Surface: Butler-Volmer flux
        flux_n = j_n / F_C
        flux_p = j_p / F_C
        cn[-1] = cs_n[-1] - flux_n * dt / (dr_n + 1e-30) * 0.5
        cp[-1] = cs_p[-1] + flux_p * dt / (dr_p + 1e-30) * 0.5
        cs_n = np.clip(cn, 1.0, cs_max_n - 1.0)
        cs_p = np.clip(cp, 1.0, cs_max_p - 1.0)

        # Thermal model
        Q_gen = max(I_app * (Up - Un + I_app * p['R_ohmic']), 0.0)
        dT = dt * (Q_gen - p['h_conv'] * A_cell * (T_cell - T_amb)) / max(p['Cp_cell'], 0.1)
        T_cell += dT

        t_out[s] = s * dt
        v_out[s] = V
        T_out[s] = T_cell
        c_out[s] = Q_discharged / 3600.0

        Q_discharged += I_app * dt

        if V < 2.7:
            return {'time': t_out[:s+1], 'voltage': v_out[:s+1],
                    'temperature': T_out[:s+1], 'capacity': c_out[:s+1]}

    return {'time': t_out, 'voltage': v_out, 'temperature': T_out, 'capacity': c_out}


# Quick validation
print("Testing ECAT model...")
pv_mid = (BOUNDS_LOWER + BOUNDS_UPPER) / 2
r = simulate_ecat(pv_mid, dt=1.0)
print(f"  Nominal: {len(r['voltage'])} pts, V=[{r['voltage'].min():.3f},{r['voltage'].max():.3f}], "
      f"Cap={r['capacity'][-1]:.3f} Ah")

# LHS test
np.random.seed(42)
lhs_u = np.zeros((200, N_PARAMS))
for j in range(N_PARAMS):
    perm = np.random.permutation(200)
    for i in range(200): lhs_u[i,j] = (perm[i]+np.random.uniform())/200
lhs_samples = BOUNDS_LOWER + lhs_u * (BOUNDS_UPPER - BOUNDS_LOWER)

success = 0
for i in range(200):
    try:
        r = simulate_ecat(lhs_samples[i], dt=1.0)
        if len(r['voltage']) >= 30: success += 1
    except: pass
print(f"LHS test: {success}/200 valid samples")
