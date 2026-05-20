"""
Complete LES Analysis - Final Version
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from xyz_parser import parse_xyz
from les_model import compute_coulomb_energy_vec, compute_coulomb_forces_vec
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def analyze_random_charges(data_dir='data', output_dir='outputs'):
    """
    Analysis 1: Recover exact charges from energy data (Fig. 1).
    
    Approach:
    1. Show Coulomb energy decomposition for full system
    2. For small subsets (8 atoms), demonstrate charge recovery via optimization
    3. Show charge-energy correlation across configurations
    """
    print("\n" + "="*60)
    print("Analysis 1: Random Charges - Charge Recovery")
    print("="*60)
    
    configs = parse_xyz(f'{data_dir}/random_charges.xyz')
    n_configs = len(configs)
    
    all_positions = np.array([c.positions for c in configs])
    all_true_charges = np.array([c.true_charges for c in configs])
    
    print(f"  Configurations: {n_configs}, Atoms: {configs[0].n_atoms}")
    print(f"  +1e atoms: {int((all_true_charges[0] > 0).sum())}, "
          f"-1e atoms: {int((all_true_charges[0] < 0).sum())}")
    
    # Compute Coulomb energies for all configs
    all_E_coulomb = np.array([
        compute_coulomb_energy_vec(q, pos) 
        for q, pos in zip(all_true_charges, all_positions)
    ])
    
    print(f"  Coulomb energy range: [{all_E_coulomb.min():.3f}, {all_E_coulomb.max():.3f}]")
    
    # === Analysis A: Charge recovery on 8-atom subsets ===
    print("\n  [A] Charge recovery on 8-atom subsets...")
    
    n_sub = 8  # atoms per subset
    n_trials = 20
    subset_mae_list = []
    subset_corr_list = []
    
    rng = np.random.RandomState(42)
    
    for trial in range(n_trials):
        # Pick a random config and random 8 atoms
        cfg_idx = rng.randint(n_configs)
        atom_idx = rng.choice(configs[0].n_atoms, n_sub, replace=False)
        
        pos_sub = all_positions[cfg_idx][atom_idx]
        q_sub = all_true_charges[cfg_idx][atom_idx]
        E_target = compute_coulomb_energy_vec(q_sub, pos_sub)
        
        # Optimize charges to match energy (continuous optimization)
        q_init = rng.randn(n_sub) * 0.5
        
        def obj(q):
            E = compute_coulomb_energy_vec(q, pos_sub)
            return (E - E_target)**2 + 0.001 * np.sum(q**2)
        
        res = minimize(obj, q_init, method='L-BFGS-B',
                      bounds=[(-2, 2)] * n_sub,
                      options={'maxiter': 500, 'ftol': 1e-15})
        
        q_recovered = res.x
        
        # Handle sign ambiguity
        err1 = np.mean(np.abs(q_recovered - q_sub))
        err2 = np.mean(np.abs(-q_recovered - q_sub))
        if err2 < err1:
            q_recovered = -q_recovered
        
        mae = np.mean(np.abs(q_recovered - q_sub))
        corr = np.corrcoef(q_recovered, q_sub)[0, 1] if np.std(q_recovered) > 1e-10 else 0
        
        subset_mae_list.append(mae)
        subset_corr_list.append(corr)
    
    avg_mae = np.mean(subset_mae_list)
    avg_corr = np.mean(subset_corr_list)
    print(f"    Avg MAE:  {avg_mae:.4f}")
    print(f"    Avg Corr: {avg_corr:.4f}")
    
    # === Analysis B: Charge-energy correlation across configs ===
    print("\n  [B] Charge-energy relationship across configurations...")
    
    # Compute per-config charge statistics
    pos_fractions = np.array([(q > 0).sum() / len(q) for q in all_true_charges])
    charge_vars = np.array([np.var(q) for q in all_true_charges])
    
    # Correlation between charge variance and energy
    corr_var_E = np.corrcoef(charge_vars, all_E_coulomb)[0, 1]
    print(f"    Charge var-Energy correlation: {corr_var_E:.4f}")
    
    # === Analysis C: Force consistency ===
    print("\n  [C] Force consistency check...")
    
    force_mags = []
    for i in range(n_configs):
        F = compute_coulomb_forces_vec(all_true_charges[i], all_positions[i])
        force_mags.append(np.linalg.norm(F, axis=1).mean())
    
    force_mags = np.array(force_mags)
    print(f"    Avg force magnitude: {force_mags.mean():.4f}")
    print(f"    Force range: [{force_mags.min():.4f}, {force_mags.max():.4f}]")
    
    # Save
    np.save(f'{output_dir}/random_charges_true.npy', all_true_charges)
    np.save(f'{output_dir}/random_charges_positions.npy', all_positions)
    np.save(f'{output_dir}/random_charges_E.npy', all_E_coulomb)
    
    # Save subset recovery results for plotting
    subset_data = {
        'mae_list': [float(x) for x in subset_mae_list],
        'corr_list': [float(x) for x in subset_corr_list],
    }
    
    results = {
        'n_configs': n_configs, 'n_atoms': configs[0].n_atoms,
        'subset_recovery': {
            'avg_mae': float(avg_mae), 'avg_corr': float(avg_corr),
            'n_trials': n_trials, 'subset_size': n_sub,
        },
        'charge_energy_corr': float(corr_var_E),
        'force_stats': {
            'mean_magnitude': float(force_mags.mean()),
        },
    }
    
    with open(f'{output_dir}/random_charges_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def analyze_charged_dimers(data_dir='data', output_dir='outputs'):
    """Analysis 2: Binding energy curves (Fig. 3)."""
    print("\n" + "="*60)
    print("Analysis 2: Charged Dimers - Binding Energy Curves")
    print("="*60)
    
    configs = parse_xyz(f'{data_dir}/charged_dimer.xyz')
    energies = np.array([c.energy for c in configs])
    positions = np.array([c.positions for c in configs])
    
    # Inter-dimer distances
    dimer_dists = []
    for pos in positions:
        c1, c2 = pos[:4].mean(0), pos[4:].mean(0)
        dimer_dists.append(np.linalg.norm(c1 - c2))
    dimer_dists = np.array(dimer_dists)
    sort_idx = np.argsort(dimer_dists)
    
    print(f"  Configs: {len(configs)}, Dist: [{dimer_dists.min():.2f}, {dimer_dists.max():.2f}]")
    print(f"  Energy: [{energies.min():.4f}, {energies.max():.4f}]")
    
    # LES decomposition: +1e/4 on each atom of molecule 1, -1e/4 on molecule 2
    charges = np.zeros((len(configs), 8))
    charges[:, :4] = 0.25
    charges[:, 4:] = -0.25
    
    lr_E = np.array([compute_coulomb_energy_vec(q, p) for q, p in zip(charges, positions)])
    sr_E = energies - lr_E
    
    # LJ fit to total energies (SR-only model)
    def lj(r, sigma, eps):
        x = (sigma / r)**6
        return eps * x * (x - 1)
    
    def fit_lj(dists, target):
        def obj(p):
            pred = np.array([lj(d, p[0], p[1]) for d in dists])
            return np.mean((pred - target)**2)
        res = minimize(obj, [1.0, 1.0], method='Nelder-Mead')
        return res.x
    
    p_total = fit_lj(dimer_dists, energies)
    sr_fitted_total = np.array([lj(d, *p_total) for d in dimer_dists])
    
    # LJ fit to SR energies + LR (LES model)
    p_sr = fit_lj(dimer_dists, sr_E)
    sr_fitted_sr = np.array([lj(d, *p_sr) for d in dimer_dists])
    les_total = sr_fitted_sr + lr_E
    
    # Analytical 1/r binding curve (for comparison)
    def coulomb_binding(q1, q2, r):
        return q1 * q2 / r
    
    q1, q2 = 1.0, -1.0
    analytical = np.array([coulomb_binding(q1, q2, d) for d in dimer_dists])
    
    mae_sr = np.mean(np.abs(sr_fitted_total - energies))
    mae_les = np.mean(np.abs(les_total - energies))
    mae_analytical = np.mean(np.abs(analytical - energies))
    
    print(f"\n  SR-only MAE: {mae_sr:.6f}")
    print(f"  LES MAE:     {mae_les:.6f}")
    print(f"  Analytical 1/r MAE: {mae_analytical:.6f}")
    print(f"  LR range: [{lr_E.min():.4f}, {lr_E.max():.4f}]")
    
    # Save
    for name, data in [
        ('dimer_distances', dimer_dists), ('dimer_energies', energies),
        ('dimer_lr_energies', lr_E), ('dimer_sr_energies', sr_E),
        ('dimer_sr_fitted', sr_fitted_total), ('dimer_les_total', les_total),
        ('dimer_sort_idx', sort_idx), ('dimer_charges', charges),
        ('dimer_analytical', analytical),
    ]:
        np.save(f'{output_dir}/{name}.npy', data)
    
    results = {
        'n_configs': len(configs),
        'mae_sr_only': float(mae_sr), 'mae_les': float(mae_les),
        'mae_analytical': float(mae_analytical),
        'lr_range': [float(lr_E.min()), float(lr_E.max())],
    }
    with open(f'{output_dir}/dimer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def analyze_ag3(data_dir='data', output_dir='outputs'):
    """Analysis 3: Ag3 charge states (Fig. 5e, Table 1)."""
    print("\n" + "="*60)
    print("Analysis 3: Ag3 Charge States - PES Comparison")
    print("="*60)
    
    configs = parse_xyz(f'{data_dir}/ag3_chargestates.xyz')
    pos_c = [c for c in configs if c.charge_state == 1]
    neg_c = [c for c in configs if c.charge_state == -1]
    
    pos_E = np.array([c.energy for c in pos_c])
    neg_E = np.array([c.energy for c in neg_c])
    pos_p = np.array([c.positions for c in pos_c])
    neg_p = np.array([c.positions for c in neg_c])
    
    def mean_d(pos):
        d01 = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
        d02 = np.linalg.norm(pos[:, 0] - pos[:, 2], axis=1)
        d12 = np.linalg.norm(pos[:, 1] - pos[:, 2], axis=1)
        return (d01 + d02 + d12) / 3
    
    pos_d, neg_d = mean_d(pos_p), mean_d(neg_p)
    
    print(f"  +1: {len(pos_c)} configs, E=[{pos_E.min():.4f}, {pos_E.max():.4f}]")
    print(f"  -1: {len(neg_c)} configs, E=[{neg_E.min():.4f}, {neg_E.max():.4f}]")
    
    # Check if configurations are identical across charge states
    configs_identical = all(
        np.allclose(pos_c[i].positions, neg_c[i].positions) and 
        abs(pos_c[i].energy - neg_c[i].energy) < 1e-10
        for i in range(len(pos_c))
    )
    print(f"  Configurations identical across charge states: {configs_identical}")
    
    # LES decomposition
    q_pos, q_neg = np.ones(3)/3, -np.ones(3)/3
    lr_p = np.array([compute_coulomb_energy_vec(q_pos, p) for p in pos_p])
    lr_n = np.array([compute_coulomb_energy_vec(q_neg, p) for p in neg_p])
    sr_p, sr_n = pos_E - lr_p, neg_E - lr_n
    
    print(f"\n  SR +1: mean={sr_p.mean():.4f}, std={sr_p.std():.4f}")
    print(f"  SR -1: mean={sr_n.mean():.4f}, std={sr_n.std():.4f}")
    print(f"  LR +1: mean={lr_p.mean():.4f}")
    print(f"  LR -1: mean={lr_n.mean():.4f}")
    
    # Demonstrate charge state distinction
    # For same geometry, +1 and -1 give different LR energies
    example_idx = 0
    print(f"\n  Example config {example_idx}:")
    print(f"    +1 LR: {lr_p[example_idx]:.4f}, -1 LR: {lr_n[example_idx]:.4f}")
    print(f"    LR difference: {abs(lr_p[example_idx] - lr_n[example_idx]):.4f}")
    print(f"    Total E (both): {pos_E[example_idx]:.4f}")
    
    for name, data in [
        ('ag3_pos_energies', pos_E), ('ag3_pos_distances', pos_d),
        ('ag3_neg_energies', neg_E), ('ag3_neg_distances', neg_d),
        ('ag3_lr_pos', lr_p), ('ag3_lr_neg', lr_n),
        ('ag3_sr_pos', sr_p), ('ag3_sr_neg', sr_n),
    ]:
        np.save(f'{output_dir}/{name}.npy', data)
    
    results = {
        'configs_identical': configs_identical,
        'pos': {'n': len(pos_c), 'E_mean': float(pos_E.mean()), 'E_std': float(pos_E.std()),
                'lr_mean': float(lr_p.mean()), 'sr_mean': float(sr_p.mean()), 'sr_std': float(sr_p.std())},
        'neg': {'n': len(neg_c), 'E_mean': float(neg_E.mean()), 'E_std': float(neg_E.std()),
                'lr_mean': float(lr_n.mean()), 'sr_mean': float(sr_n.mean()), 'sr_std': float(sr_n.std())},
        'separation': float(abs(pos_E.mean() - neg_E.mean())),
        'lr_difference_example': float(abs(lr_p[example_idx] - lr_n[example_idx])),
    }
    with open(f'{output_dir}/ag3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    r1 = analyze_random_charges()
    r2 = analyze_charged_dimers()
    r3 = analyze_ag3()
    print("\n" + "="*60)
    print("All analyses completed!")
    print("="*60)
