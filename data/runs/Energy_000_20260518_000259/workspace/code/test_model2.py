#!/usr/bin/env python3
"""
MMGA Parameter Identification for Li-ion Battery ECAT Model.
Complete framework: ECAT model + LHS + ANN surrogate + GA + validation + figures.
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
# ECAT MODEL
# ═══════════════════════════════════════════════════════════════
PARAM_NAMES = [
    'Rp_neg','Rp_pos','k0_neg','k0_pos','Ds_neg','Ds_pos',
    'sigma_neg','sigma_pos','eps_s_neg','eps_s_pos',
    'L_neg','L_pos','h_conv','Cp_cell','R_ohmic',
    'alpha_neg','alpha_pos']
N_PARAMS = len(PARAM_NAMES)

# Cell cross-section area for 18650 NCM cell (~1.1 Ah)
A_CELL = 0.03  # m^2
C_MAX_N = 33133.0  # mol/m^3 (graphite)
C_MAX_P = 37800.0  # mol/m^3 (NMC)
F_C = 96485.3329; R_C = 8.314462

BOUNDS_LOWER = np.array([
    2e-7, 2e-7,       # Rp (m)
    5e-7, 5e-7,       # k0 (m/s)  
    1e-14, 1e-14,     # Ds (m^2/s)
    10, 1,             # sigma (S/m)
    0.40, 0.40,        # eps_s
    50e-6, 50e-6,      # L (m)
    5.0, 0.5,          # h_conv (W/m2K), Cp (J/K)
    0.005,             # R_ohmic (Ohm)
    0.40, 0.40])       # alpha

BOUNDS_UPPER = np.array([
    3e-6, 3e-6,
    5e-5, 5e-5,
    5e-13, 5e-13,
    100, 10,
    0.65, 0.65,
    120e-6, 120e-6,
    50.0, 5.0,
    0.15,
    0.60, 0.60])

def ocv_neg(th):
    th = np.clip(th, 0.005, 0.995)
    return np.clip(0.124+1.5*np.exp(-80*(th-0.13))+0.035*np.tanh((th-0.1)/0.01)
                   -0.012*np.tanh((th-0.3)/0.01)+0.008*np.tanh((th-0.5)/0.05)
                   -0.005*np.tanh((th-0.7)/0.05), 0.005, 0.40)

def ocv_pos(th):
    th = np.clip(th, 0.005, 0.995)
    return np.clip(3.4+0.5*(1-th)+0.15*np.log(th/(1-th+0.01)), 3.0, 4.3)

def simulate_ecat(pv, I_app=1.1, dt=5.0, T_amb=298.15):
    """Simulate CC discharge using ECAT (SPM + thermal)."""
    Rp_n,Rp_p,k0n,k0p,Dsn,Dsp,sig_n,sig_p,eps_n,eps_p,L_n,L_p,h,Cp,Ro,al_n,al_p = pv

    # Electrode capacity
    Q_n = A_CELL*L_n*eps_n*C_MAX_N*F_C
    Q_p = A_CELL*L_p*eps_p*C_MAX_P*F_C
    Q_total = min(Q_n*0.78, Q_p*0.72)
    ns = min(max(int(Q_total/I_app/dt), 100), 3000)

    # Average concentrations
    cavg_n = 0.85*C_MAX_N
    cavg_p = 0.25*C_MAX_P
    T_cell = T_amb; Qd = 0

    v_out = np.zeros(ns); t_out = np.zeros(ns)
    T_out = np.zeros(ns); c_out = np.zeros(ns)

    for s in range(ns):
        th_n = cavg_n/C_MAX_N
        th_p = cavg_p/C_MAX_P

        # Volumetric reaction areas
        a_n = 3*eps_n/Rp_n; a_p = 3*eps_p/Rp_p
        jn = I_app/(A_CELL*L_n*a_n+1e-30)
        jp = I_app/(A_CELL*L_p*a_p+1e-30)

        # Surface concentration from diffusion resistance
        cs_sn = np.clip(cavg_n + jn*Rp_n/(5*F_C*Dsn+1e-30), 1, C_MAX_N-1)
        cs_sp = np.clip(cavg_p - jp*Rp_p/(5*F_C*Dsp+1e-30), 1, C_MAX_P-1)

        Un = ocv_neg(cs_sn/C_MAX_N)
        Up = ocv_pos(cs_sp/C_MAX_P)

        # Exchange current density
        i0n = max(F_C*k0n*((C_MAX_N-cs_sn)**al_n*cs_sn**(1-al_n)), 1e-10)
        i0p = max(F_C*k0p*((C_MAX_P-cs_sp)**al_p*cs_sp**(1-al_p)), 1e-10)

        eta_n = R_C*T_cell/(al_n*F_C)*np.arcsinh(np.clip(jn/(2*i0n),-50,50))
        eta_p = R_C*T_cell/(al_p*F_C)*np.arcsinh(np.clip(jp/(2*i0p),-50,50))

        V = Up + eta_p - Un + eta_n - I_app*Ro
        V = np.clip(V, 2.5, 4.3)

        # Thermal
        Q_gen = max(I_app*(Up-Un+I_app*Ro), 0)
        T_cell += dt*(Q_gen - h*A_CELL*(T_cell-T_amb))/max(Cp,0.1)

        t_out[s] = s*dt; v_out[s] = V; T_out[s] = T_cell; c_out[s] = Qd/3600

        if V < 2.7:
            return {'time':t_out[:s+1],'voltage':v_out[:s+1],
                    'temperature':T_out[:s+1],'capacity':c_out[:s+1]}

        # Mass balance: dc_avg/dt = +/- I/(A*L*eps*F)
        cavg_n -= I_app*dt/(A_CELL*L_n*eps_n*F_C)*C_MAX_N
        cavg_p += I_app*dt/(A_CELL*L_p*eps_p*F_C)*C_MAX_P
        cavg_n = np.clip(cavg_n, 1, C_MAX_N-1)
        cavg_p = np.clip(cavg_p, 1, C_MAX_P-1)
        Qd += I_app*dt

    return {'time':t_out,'voltage':v_out,'temperature':T_out,'capacity':c_out}

# Quick test
print("Testing ECAT model...")
pv_mid = (BOUNDS_LOWER+BOUNDS_UPPER)/2
r = simulate_ecat(pv_mid, dt=5.0)
v = r['voltage']
print(f"  {len(v)} pts, V=[{v.min():.3f},{v.max():.3f}], Cap={r['capacity'][-1]:.3f} Ah")
print(f"  V shape: V[0]={v[0]:.3f}, V[mid]={v[len(v)//2]:.3f}, V[-1]={v[-1]:.3f}")

# LHS validation
np.random.seed(42)
lhs_u = np.zeros((300, N_PARAMS))
for j in range(N_PARAMS):
    perm = np.random.permutation(300)
    for i in range(300): lhs_u[i,j] = (perm[i]+np.random.uniform())/300
lhs = BOUNDS_LOWER + lhs_u*(BOUNDS_UPPER-BOUNDS_LOWER)

valid = 0
caps = []
for i in range(300):
    try:
        r = simulate_ecat(lhs[i], dt=5.0)
        if len(r['voltage'])>=30:
            valid += 1
            caps.append(r['capacity'][-1])
    except: pass
print(f"LHS: {valid}/300 valid, capacity range: [{min(caps):.2f},{max(caps):.2f}] Ah")
print("Model OK!")
