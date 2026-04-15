"""
Main analysis script for LES benchmark.
"""
import numpy as np
import json
import os
import sys
sys.path.insert(0, 'code')
from parse_data import parse_xyz
from scipy.optimize import curve_fit
from scipy import stats

def analyze_dataset1():
    """Dataset 1: Random charges - charge recovery benchmark."""
    print("\n" + "="*60)
    print("Dataset 1: Random Charges - Charge Recovery")
    print("="*60)
    
    frames = parse_xyz('data/random_charges.xyz')
    print(f"Loaded {len(frames)} frames with {frames[0]['n_atoms']} atoms each")
    
    frame = frames[0]
    positions = frame['positions']
    true_charges = np.array(frame['props']['true_charges'])
    box_length = 15.0
    
    print(f"True charges: +1 count = {np.sum(true_charges > 0)}, -1 count = {np.sum(true_charges < 0)}")
    print(f"Total charge: {np.sum(true_charges):.1f}")
    
    # Compute target energy and forces using Coulomb + LJ
    N = len(true_charges)
    target_energy = 0.0
    target_forces = np.zeros((N, 3))
    
    for i in range(N):
        for j in range(i+1, N):
            dr = positions[i] - positions[j]
            dr = dr - box_length * np.round(dr / box_length)
            r = np.linalg.norm(dr)
            if r > 1e-10:
                # Coulomb
                e_coul = true_charges[i] * true_charges[j] / r
                target_energy += e_coul
                f_coul = true_charges[i] * true_charges[j] / r**3 * dr
                # Repulsive LJ
                sr6 = (1.0/r)**6
                e_lj = 4 * sr6 * sr6
                target_energy += e_lj
                f_lj = 24 * sr6 * sr6 / r**2 * dr
                target_forces[i] += f_coul + f_lj
                target_forces[j] -= f_coul + f_lj
    
    print(f"Target energy: {target_energy:.4f}")
    
    # Charge recovery via optimization
    # Use subset for speed
    N_sub = 32
    pos_sub = positions[:N_sub]
    tc_sub = true_charges[:N_sub]
    
    # Recompute target for subset
    target_en_sub = 0.0
    target_f_sub = np.zeros((N_sub, 3))
    for i in range(N_sub):
        for j in range(i+1, N_sub):
            dr = pos_sub[i] - pos_sub[j]
            dr = dr - box_length * np.round(dr / box_length)
            r = np.linalg.norm(dr)
            if r > 1e-10:
                e_coul = tc_sub[i] * tc_sub[j] / r
                target_en_sub += e_coul
                f_coul = tc_sub[i] * tc_sub[j] / r**3 * dr
                sr6 = (1.0/r)**6
                e_lj = 4 * sr6 * sr6
                target_en_sub += e_lj
                f_lj = 24 * sr6 * sr6 / r**2 * dr
                target_f_sub[i] += f_coul + f_lj
                target_f_sub[j] -= f_coul + f_lj
    
    # Optimize latent charges
    import torch
    import torch.nn as nn
    
    latent_charges = nn.Parameter(torch.randn(N_sub, dtype=torch.float64) * 0.1)
    pos_t = torch.tensor(pos_sub, dtype=torch.float64)
    target_en_t = torch.tensor(target_en_sub, dtype=torch.float64)
    target_f_t = torch.tensor(target_f_sub, dtype=torch.float64)
    
    optimizer = torch.optim.Adam([latent_charges], lr=0.02)
    
    losses_hist, charge_mse_hist = [], []
    
    for epoch in range(400):
        optimizer.zero_grad()
        
        dr = pos_t[:, None, :] - pos_t[None, :, :]
        dr = dr - box_length * torch.round(dr / box_length)
        r = torch.sqrt((dr**2).sum(-1))
        r = r + torch.eye(N_sub, dtype=torch.float64) * 1e20
        qi = latent_charges[:, None]
        qj = latent_charges[None, :]
        
        pred_energy = 0.5 * (qi * qj / r).sum()
        
        f_mag = qi * qj / r**3
        f_mag = f_mag - torch.diag_embed(torch.diag(f_mag))
        pred_forces = (f_mag[:, :, None] * dr).sum(dim=1)
        
        en_loss = (pred_energy - target_en_t)**2
        f_loss = torch.mean((pred_forces - target_f_t)**2)
        charge_sum_loss = latent_charges.sum()**2
        
        loss = en_loss + 0.1 * f_loss + 0.01 * charge_sum_loss
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            mse = torch.mean((latent_charges - torch.tensor(tc_sub, dtype=torch.float64))**2).item()
        
        losses_hist.append(loss.item())
        charge_mse_hist.append(mse)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.2f}, charge_MSE={mse:.6f}")
    
    recovered = latent_charges.detach().numpy()
    
    # Compute correlation
    std_r = np.std(recovered)
    std_t = np.std(tc_sub)
    if std_r > 1e-10 and std_t > 1e-10:
        corr = np.corrcoef(recovered, tc_sub)[0,1]
        corr_flip = np.corrcoef(-recovered, tc_sub)[0,1]
        best_corr = max(abs(corr), abs(corr_flip)) if not (np.isnan(corr) or np.isnan(corr_flip)) else 0.0
    else:
        best_corr = 0.0
    
    charge_mae = np.mean(np.abs(recovered - tc_sub))
    charge_rmse = np.sqrt(np.mean((recovered - tc_sub)**2))
    
    print(f"\nFinal charge MSE: {charge_mse_hist[-1]:.6f}")
    print(f"Charge MAE: {charge_mae:.6f}")
    print(f"Correlation: {best_corr:.4f}")
    
    return {
        'n_atoms': N_sub,
        'target_energy': float(target_en_sub),
        'charge_mae': float(charge_mae),
        'charge_rmse': float(charge_rmse),
        'charge_correlation': float(best_corr),
        'final_loss': float(losses_hist[-1]),
        'training_history': {
            'loss': [float(x) for x in losses_hist],
            'charge_mae': [float(x) for x in charge_mse_hist]
        },
        'true_charges': tc_sub.tolist(),
        'recovered_charges': recovered.tolist()
    }

def analyze_dataset2():
    """Dataset 2: Charged dimer - binding curve analysis."""
    print("\n" + "="*60)
    print("Dataset 2: Charged Dimer - Binding Curves")
    print("="*60)
    
    frames = parse_xyz('data/charged_dimer.xyz')
    print(f"Loaded {len(frames)} frames with {frames[0]['n_atoms']} atoms each")
    
    energies = []
    distances = []
    
    for frame in frames:
        energies.append(frame['props']['energy'])
        pos = frame['positions']
        com1 = np.mean(pos[:4], axis=0)
        com2 = np.mean(pos[4:], axis=0)
        dist = np.linalg.norm(com2 - com1)
        distances.append(dist)
    
    energies = np.array(energies)
    distances = np.array(distances)
    
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    energies = energies[sort_idx]
    
    print(f"Distance range: {distances.min():.2f} - {distances.max():.2f} Angstrom")
    print(f"Energy range: {energies.min():.4f} - {energies.max():.4f}")
    
    # Fit models
    def coulomb_plus_repulsion(r, A, B, C):
        return A/r + B/r**12 + C
    
    def exp_coulomb(r, A, B, C, D):
        return A * np.exp(-B * r) + C/r + D
    
    try:
        popt1, _ = curve_fit(coulomb_plus_repulsion, distances, energies, p0=[1.0, 1.0, 0.0], maxfev=5000)
        pred1 = coulomb_plus_repulsion(distances, *popt1)
        mae1 = np.mean(np.abs(pred1 - energies))
        rmse1 = np.sqrt(np.mean((pred1 - energies)**2))
        print(f"Coulomb+Repulsion: MAE={mae1:.6f}, RMSE={rmse1:.6f}")
    except Exception as e:
        mae1, rmse1, pred1 = None, None, np.zeros_like(energies)
        print(f"Coulomb+Repulsion fit failed: {e}")
    
    try:
        popt2, _ = curve_fit(exp_coulomb, distances, energies, p0=[1.0, 1.0, 0.5, 0.0], maxfev=5000)
        pred2 = exp_coulomb(distances, *popt2)
        mae2 = np.mean(np.abs(pred2 - energies))
        rmse2 = np.sqrt(np.mean((pred2 - energies)**2))
        print(f"Exp+Coulomb: MAE={mae2:.6f}, RMSE={rmse2:.6f}")
    except Exception as e:
        mae2, rmse2, pred2 = None, None, np.zeros_like(energies)
        print(f"Exp+Coulomb fit failed: {e}")
    
    return {
        'n_frames': len(frames),
        'distance_range': [float(distances.min()), float(distances.max())],
        'energy_range': [float(energies.min()), float(energies.max())],
        'distances': distances.tolist(),
        'energies': energies.tolist(),
        'pred_coulomb_repulsion': pred1.tolist(),
        'pred_exp_coulomb': pred2.tolist(),
        'mae_coulomb_repulsion': float(mae1) if mae1 else None,
        'rmse_coulomb_repulsion': float(rmse1) if rmse1 else None,
        'mae_exp_coulomb': float(mae2) if mae2 else None,
        'rmse_exp_coulomb': float(rmse2) if rmse2 else None,
    }

def analyze_dataset3():
    """Dataset 3: Ag3 charge states - PES analysis."""
    print("\n" + "="*60)
    print("Dataset 3: Ag3 Charge States - PES Analysis")
    print("="*60)
    
    frames = parse_xyz('data/ag3_chargestates.xyz')
    print(f"Loaded {len(frames)} frames with {frames[0]['n_atoms']} atoms each")
    
    charge_states = {}
    for frame in frames:
        cs = frame['props'].get('charge_state')
        if cs not in charge_states:
            charge_states[cs] = []
        charge_states[cs].append(frame)
    
    print(f"Charge states: {sorted(charge_states.keys())}")
    
    results_by_state = {}
    for cs, fl in charge_states.items():
        energies = np.array([f['props']['energy'] for f in fl])
        
        bond_lengths = []
        for f in fl:
            pos = f['positions']
            d12 = np.linalg.norm(pos[0] - pos[1])
            d13 = np.linalg.norm(pos[0] - pos[2])
            d23 = np.linalg.norm(pos[1] - pos[2])
            bond_lengths.append((d12 + d13 + d23) / 3)
        bond_lengths = np.array(bond_lengths)
        
        sort_idx = np.argsort(bond_lengths)
        bond_lengths = bond_lengths[sort_idx]
        energies = energies[sort_idx]
        
        print(f"\nCharge state {cs}: {len(fl)} frames")
        print(f"  Bond length: {bond_lengths.min():.3f} - {bond_lengths.max():.3f}")
        print(f"  Energy: {energies.min():.4f} - {energies.max():.4f}")
        print(f"  Mean: {energies.mean():.4f} +/- {energies.std():.4f}")
        
        results_by_state[str(cs)] = {
            'n_frames': len(fl),
            'bond_lengths': bond_lengths.tolist(),
            'energies': energies.tolist(),
            'mean_energy': float(energies.mean()),
            'std_energy': float(energies.std()),
        }
    
    if 1 in charge_states and -1 in charge_states:
        en_pos = np.array([f['props']['energy'] for f in charge_states[1]])
        en_neg = np.array([f['props']['energy'] for f in charge_states[-1]])
        energy_diff = en_pos.mean() - en_neg.mean()
        t_stat, p_value = stats.ttest_ind(en_pos, en_neg)
        print(f"\nEnergy difference (+1 vs -1): {energy_diff:.4f}")
        print(f"t={t_stat:.4f}, p={p_value:.6f}")
        
        results_by_state['comparison'] = {
            'mean_diff': float(energy_diff),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
        }
    
    return results_by_state

def main():
    os.makedirs('outputs', exist_ok=True)
    
    r1 = analyze_dataset1()
    r2 = analyze_dataset2()
    r3 = analyze_dataset3()
    
    with open('outputs/dataset1_charge_recovery.json', 'w') as f:
        json.dump(r1, f, indent=2)
    with open('outputs/dataset2_binding_curves.json', 'w') as f:
        json.dump(r2, f, indent=2)
    with open('outputs/dataset3_charge_states.json', 'w') as f:
        json.dump(r3, f, indent=2)
    
    print("\nAll analyses complete. Results saved.")

if __name__ == '__main__':
    main()
