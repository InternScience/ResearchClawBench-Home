"""
Phase 2: ECAT Single-Particle Model Simulation & LHS Sampling

Implements a simplified electrochemical single-particle model (SPM)
with thermal coupling that generates discharge voltage curves from
internal battery parameters.

During DISCHARGE (I_app > 0):
- Li moves FROM anode TO cathode
- Cathode gains Li (sto_p increases) -> U_p decreases
- Anode loses Li (sto_n decreases) -> U_n increases
- Terminal voltage V = U_p - U_n decreases
"""
import numpy as np
import os
import json

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_000_20260415_130453"
OUTPUTS = os.path.join(WORKSPACE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

F = 96485.3329
R_gas = 8.314462618

PARAM_BOUNDS = {
    'Rs_p':       (1e-6, 10e-6),
    'Rs_n':       (1e-6, 15e-6),
    'k_p':        (1e-11, 1e-9),
    'k_n':        (1e-11, 5e-10),
    'Ds_p':       (1e-15, 1e-12),
    'Ds_n':       (1e-15, 5e-12),
    'h_coeff':    (5, 50),
    'eps_s_p':    (0.3, 0.7),
    'eps_s_n':    (0.3, 0.7),
    'cs_max_p':   (2e4, 6e4),
    'cs_max_n':   (1.5e4, 3.5e4),
}

NOMINAL_PARAMS = {
    'Rs_p': 2.0e-6, 'Rs_n': 5.0e-6,
    'k_p': 3.0e-11, 'k_n': 2.0e-11,
    'Ds_p': 1.0e-14, 'Ds_n': 3.0e-14,
    'h_coeff': 15.0,
    'eps_s_p': 0.52, 'eps_s_n': 0.55,
    'cs_max_p': 51000.0, 'cs_max_n': 28000.0,
}

L_p, L_n, L_s = 70e-6, 80e-6, 25e-6
A_cell = 0.015
T_amb = 297.15


def ocv_positive(sto):
    """OCV of NMC positive electrode vs Li/Li+.
    sto = cs/cs_max. Higher sto = more Li in cathode = lower potential.
    Range: sto ~0.3 (charged) to ~0.8 (discharged).
    """
    sto = np.clip(sto, 0.01, 0.99)
    # NMC-like: ~4.2V at sto=0.3, ~3.5V at sto=0.8
    return 4.4 - 1.2 * sto + 0.3 * sto**2


def ocv_negative(sto):
    """OCV of graphite negative electrode vs Li/Li+.
    sto = cs/cs_max. Higher sto = more Li in anode = lower potential.
    Range: sto ~0.85 (charged) to ~0.2 (discharged).
    """
    sto = np.clip(sto, 0.01, 0.99)
    # Graphite-like: ~0.08V at sto=0.85, ~0.15V at sto=0.2
    return 0.05 + 0.12 * np.exp(-5 * sto) + 0.03 * sto


def simulate_discharge(params, I_app, t_eval, T_init=297.15):
    """
    Simulate constant-current discharge using explicit Euler SPM with thermal coupling.
    
    I_app > 0 means discharge current.
    During discharge:
      - Cathode: Li insertion (reduction), cs_p increases
      - Anode: Li extraction (oxidation), cs_n decreases
    """
    Rs_p = params['Rs_p']
    Rs_n = params['Rs_n']
    k_p = params['k_p']
    k_n = params['k_n']
    Ds_p = params['Ds_p']
    Ds_n = params['Ds_n']
    h_coeff = params['h_coeff']
    eps_s_p = params['eps_s_p']
    eps_s_n = params['eps_s_n']
    cs_max_p = params['cs_max_p']
    cs_max_n = params['cs_max_n']

    a_p = 3.0 * eps_s_p / Rs_p
    a_n = 3.0 * eps_s_n / Rs_n

    # Initial state: fully charged
    # Cathode: delithiated (low sto)
    # Anode: lithiated (high sto)
    sto_p_init = 0.30
    sto_n_init = 0.85
    cs_surf_p = sto_p_init * cs_max_p
    cs_surf_n = sto_n_init * cs_max_n

    # Current densities (positive = anodic/oxidation direction)
    # During discharge: cathode undergoes reduction (j_p < 0), anode oxidation (j_n > 0)
    j_p = -I_app / (a_p * L_p * A_cell)  # negative = reduction at cathode
    j_n = I_app / (a_n * L_n * A_cell)   # positive = oxidation at anode
    cs_e = 1000.0

    dt_arr = np.diff(t_eval)
    dt_arr = np.append(dt_arr, dt_arr[-1])
    n_steps = len(t_eval)

    voltage = np.zeros(n_steps)
    temp_arr = np.zeros(n_steps)
    T_cell = T_init

    for i in range(n_steps):
        cs_surf_p = max(cs_surf_p, 0.001 * cs_max_p)
        cs_surf_n = max(cs_surf_n, 0.001 * cs_max_n)
        cs_surf_p = min(cs_surf_p, 0.999 * cs_max_p)
        cs_surf_n = min(cs_surf_n, 0.999 * cs_max_n)

        sto_p = cs_surf_p / cs_max_p
        sto_n = cs_surf_n / cs_max_n

        U_p = ocv_positive(sto_p)
        U_n = ocv_negative(sto_n)

        # Exchange current density
        i0_p = F * k_p * np.sqrt(cs_e) * np.sqrt(max(cs_max_p - cs_surf_p, 1e-10)) * np.sqrt(max(cs_surf_p, 1e-10))
        i0_n = F * k_n * np.sqrt(cs_e) * np.sqrt(max(cs_max_n - cs_surf_n, 1e-10)) * np.sqrt(max(cs_surf_n, 1e-10))

        # Overpotential from inverse Butler-Volmer
        arg_p = np.clip(j_p / (2.0 * i0_p + 1e-30), -50, 50)
        arg_n = np.clip(j_n / (2.0 * i0_n + 1e-30), -50, 50)
        eta_p = (2.0 * R_gas * T_cell / F) * np.arcsinh(arg_p)
        eta_n = (2.0 * R_gas * T_cell / F) * np.arcsinh(arg_n)

        V_term = U_p - U_n + eta_p - eta_n
        voltage[i] = V_term
        temp_arr[i] = T_cell

        if V_term < 1.5 or V_term > 5.0:
            return None, None, False

        # Update surface concentrations
        # dcs/dt = -j/(F * Rs/3)
        # Cathode: j_p < 0 -> dcs_p > 0 (gains Li during discharge) ✓
        # Anode: j_n > 0 -> dcs_n < 0 (loses Li during discharge) ✓
        dcs_p = -j_p / (F * Rs_p / 3.0) * dt_arr[i]
        dcs_n = -j_n / (F * Rs_n / 3.0) * dt_arr[i]

        # Diffusion relaxation (small correction toward average)
        diff_p = 3.0 * Ds_p / (Rs_p**2) * (0.5 * cs_max_p - cs_surf_p) * dt_arr[i] * 0.01
        diff_n = 3.0 * Ds_n / (Rs_n**2) * (0.5 * cs_max_n - cs_surf_n) * dt_arr[i] * 0.01

        cs_surf_p += dcs_p + diff_p
        cs_surf_n += dcs_n + diff_n

        # Thermal update
        V_ocv = U_p - U_n
        Q_gen = abs(I_app * (V_ocv - V_term))
        rho_cp = 2.0e6
        V_cell = A_cell * (L_p + L_n + L_s)
        A_heat = 2.0 * A_cell
        dT = (Q_gen - h_coeff * A_heat * (T_cell - T_amb)) / (rho_cp * V_cell) * dt_arr[i]
        T_cell += dT

    if np.any(np.isnan(voltage)) or np.any(np.isinf(voltage)):
        return None, None, False
    v_range = voltage.max() - voltage.min()
    if v_range < 0.1:
        return None, None, False

    return voltage, temp_arr, True


def generate_lhs_samples(n_samples, seed=42):
    from scipy.stats import qmc
    param_names = list(PARAM_BOUNDS.keys())
    n_params = len(param_names)
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    lhs_sample = sampler.random(n=n_samples)
    samples = np.zeros_like(lhs_sample)
    for j, name in enumerate(param_names):
        lo, hi = PARAM_BOUNDS[name]
        if hi / lo > 100:
            samples[:, j] = 10**(np.log10(lo) + lhs_sample[:, j] * (np.log10(hi) - np.log10(lo)))
        else:
            samples[:, j] = lo + lhs_sample[:, j] * (hi - lo)
    return samples, param_names


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: ECAT Model Simulation & LHS Sampling")
    print("=" * 60)

    print("\nTesting nominal parameters...")
    t_sim = np.linspace(0, 3600, 200)
    I_test = 2.0

    v_nom, T_nom, success = simulate_discharge(NOMINAL_PARAMS, I_test, t_sim)
    if success:
        print(f"  Nominal simulation successful!")
        print(f"    Voltage range: [{v_nom.min():.4f}, {v_nom.max():.4f}] V")
        print(f"    Voltage drop: {v_nom.max()-v_nom.min():.4f} V")
        print(f"    Temperature: {T_nom[0]:.2f} -> {T_nom[-1]:.2f} K")
        print(f"    First 5 voltages: {v_nom[:5]}")
        print(f"    Last 5 voltages: {v_nom[-5:]}")
    else:
        print("  Nominal simulation FAILED")

    print("\nGenerating LHS samples...")
    n_samples = 500
    samples, param_names = generate_lhs_samples(n_samples)
    print(f"  Generated {n_samples} samples with {len(param_names)} parameters")

    np.savez(os.path.join(OUTPUTS, "lhs_samples.npz"),
             samples=samples, param_names=np.array(param_names))

    print(f"\nRunning {n_samples} simulations...")
    voltages = np.zeros((n_samples, len(t_sim)))
    temperatures = np.zeros((n_samples, len(t_sim)))
    success_count = 0

    for i in range(n_samples):
        params = {name: samples[i, j] for j, name in enumerate(param_names)}
        v, T, ok = simulate_discharge(params, I_test, t_sim)
        if ok and v is not None:
            voltages[i] = v
            temperatures[i] = T
            success_count += 1
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_samples} ({success_count} successful)")

    print(f"\n  Total successful: {success_count}/{n_samples}")

    np.savez(os.path.join(OUTPUTS, "spm_simulation_results.npz"),
             voltages=voltages, temperatures=temperatures,
             time=t_sim, current=I_test,
             success_rate=success_count / n_samples)

    meta = {
        'n_samples': int(n_samples), 'n_successful': int(success_count),
        'time_points': len(t_sim),
        'time_range': [float(t_sim[0]), float(t_sim[-1])],
        'current_A': float(I_test),
        'param_names': param_names,
        'param_bounds': {k: [float(v[0]), float(v[1])] for k, v in PARAM_BOUNDS.items()},
        'nominal_params': {k: float(v) for k, v in NOMINAL_PARAMS.items()},
    }
    with open(os.path.join(OUTPUTS, "simulation_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\nAll simulation results saved to outputs/")
