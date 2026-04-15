"""
Latent Ewald Summation (LES) - Vectorized Implementation
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.special import erfc
from scipy.optimize import curve_fit
import json, os, sys
sys.path.insert(0, 'code')
from parse_data import parse_xyz

def compute_coulomb_lj_vec(positions, charges, box_length, sigma=1.0, epsilon=1.0):
    """Vectorized Coulomb + LJ energy and forces."""
    N = len(charges)
    # Pairwise displacement
    dr = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    dr = dr - box_length * np.round(dr / box_length)
    r = np.sqrt((dr**2).sum(axis=-1))  # (N, N)
    np.fill_diagonal(r, np.inf)
    
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    
    # Coulomb
    qi, qj = np.meshgrid(charges, charges, indexing='ij')
    e_coul = np.where(mask, qi * qj / r_safe, 0.0)
    
    # LJ repulsive
    sr6 = (sigma / r_safe)**6
    e_lj = np.where(mask, 4 * epsilon * sr6 * sr6, 0.0)
    
    energy = 0.5 * (e_coul + e_lj).sum()
    
    # Forces
    f_mag_coul = np.where(mask, qi * qj / r_safe**3, 0.0)
    f_mag_lj = np.where(mask, 24 * epsilon * sr6 * sr6 / r_safe**2, 0.0)
    f_mag = f_mag_coul + f_mag_lj
    forces = (f_mag[:, :, None] * dr).sum(axis=1)
    
    return energy, forces

def les_charge_recovery():
    """Recover latent charges from energy+force data."""
    frames = parse_xyz('data/random_charges.xyz')
    frame = frames[0]
    positions = frame['positions']
    true_charges = np.array(frame['props']['true_charges'])
    box_length = 15.0
    N = len(true_charges)
    
    target_energy, target_forces = compute_coulomb_lj_vec(positions, true_charges, box_length)
    print(f"Target energy: {target_energy:.4f}")
    
    # Use subset for speed (first 32 atoms)
    N_sub = 32
    pos_sub = positions[:N_sub]
    tc_sub = true_charges[:N_sub]
    target_en, target_f = compute_coulomb_lj_vec(pos_sub, tc_sub, box_length)
    
    latent_charges = nn.Parameter(torch.randn(N_sub, dtype=torch.float64) * 0.1)
    pos_t = torch.tensor(pos_sub, dtype=torch.float64)
    target_en_t = torch.tensor(target_en, dtype=torch.float64)
    target_f_t = torch.tensor(target_f, dtype=torch.float64)
    
    optimizer = torch.optim.Adam([latent_charges], lr=0.02)
    
    losses_hist, charge_mse_hist = [], []
    
    for epoch in range(400):
        optimizer.zero_grad()
        
        # Vectorized energy computation
        dr = pos_t[:, None, :] - pos_t[None, :, :]
        dr = dr - box_length * torch.round(dr / box_length)
        r = torch.sqrt((dr**2).sum(-1))
        r = r + torch.eye(N_sub, dtype=torch.float64) * 1e20  # mask diagonal
        qi = latent_charges[:, None]
        qj = latent_charges[None, :]
        
        pred_energy = 0.5 * (qi * qj / r).sum()
        
        # Force prediction
        f_mag = qi * qj / r**3
        f_mag = f_mag - torch.diag_embed(torch.diag(f_mag))  # zero diagonal
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
    corr = np.corrcoef(recovered, tc_sub)[0,1]
    corr_flip = np.corrcoef(-recovered, tc_sub)[0,1]
    if abs(corr_flip) > abs(corr):
        recovered = -recovered
        corr = corr_flip
    
    print(f"\nFinal charge MSE: {charge_mse_hist[-1]:.6f}")
    print(f"Correlation: {corr:.4f}")
    
    return {
        'true_charges': tc_sub, 'recovered_charges': recovered,
        'losses': losses_hist, 'charge_errors': charge_mse_hist,
        'correlation': corr
    }

def charged_dimer_analysis():
    """Binding energy curves for charged dimers."""
    frames = parse_xyz('data/charged_dimer.xyz')
    distances, energies = [], []
    for f in frames:
        pos = f['positions']
        d = np.linalg.norm(pos[4] - pos[0])
        distances.append(d)
        energies.append(f['props']['energy'])
    distances, energies = np.array(distances), np.array(energies)
    idx = np.argsort(distances)
    distances, energies = distances[idx], energies[idx]
    
    def coulomb_exp(r, A, B, C, D):
        return A/r + B*np.exp(-C*r) + D
    
    def lj_model(r, A, B, C):
        return A/r**12 - B/r**6 + C
    
    fits = {}
    for name, func, p0 in [('Coulomb+Exp', coulomb_exp, [1.0, 5.0, 1.0, 0.0]),
                            ('LJ-like', lj_model, [1.0, 1.0, 0.0])]:
        try:
            popt, _ = curve_fit(func, distances, energies, p0=p0, maxfev=10000)
            pred = func(distances, *popt)
            mae = np.mean(np.abs(pred - energies))
            rmse = np.sqrt(np.mean((pred - energies)**2))
            fits[name] = {'mae': mae, 'rmse': rmse, 'pred': pred}
            print(f"{name}: MAE={mae:.6f}, RMSE={rmse:.6f}")
        except Exception as e:
            print(f"{name}: {e}")
    
    return {'distances': distances, 'energies': energies, 'fits': fits}

def ag3_analysis():
    """Ag3 charge state PES analysis."""
    frames = parse_xyz('data/ag3_chargestates.xyz')
    
    def get_data(frame_list):
        bonds, en = [], []
        for f in frame_list:
            p = f['positions']
            bonds.append((np.linalg.norm(p[0]-p[1])+np.linalg.norm(p[0]-p[2])+np.linalg.norm(p[1]-p[2]))/3)
            en.append(f['props']['energy'])
        return np.array(bonds), np.array(en)
    
    pos_f = [f for f in frames if f['props'].get('charge_state') == 1]
    neg_f = [f for f in frames if f['props'].get('charge_state') == -1]
    pos_bonds, pos_en = get_data(pos_f)
    neg_bonds, neg_en = get_data(neg_f)
    
    # Sort
    pos_idx, neg_idx = np.argsort(pos_bonds), np.argsort(neg_bonds)
    pos_bonds, pos_en = pos_bonds[pos_idx], pos_en[pos_idx]
    neg_bonds, neg_en = neg_bonds[neg_idx], neg_en[neg_idx]
    
    def morse(r, De, a, re, E0):
        return De * (1 - np.exp(-a*(r - re)))**2 + E0
    
    fits = {}
    for label, bonds, en in [('+1', pos_bonds, pos_en), ('-1', neg_bonds, neg_en)]:
        try:
            popt, _ = curve_fit(morse, bonds, en, p0=[5.0, 1.0, 2.7, 0.0], maxfev=10000)
            pred = morse(bonds, *popt)
            mae = np.mean(np.abs(pred - en))
            fits[label] = {'De': popt[0], 'a': popt[1], 're': popt[2], 'mae': mae, 'pred': pred}
            print(f"Ag3 {label}: De={popt[0]:.4f}, a={popt[1]:.4f}, re={popt[2]:.4f}, MAE={mae:.6f}")
        except Exception as e:
            print(f"Ag3 {label}: {e}")
    
    common = np.linspace(max(pos_bonds.min(), neg_bonds.min()),
                          min(pos_bonds.max(), neg_bonds.max()), 20)
    pos_interp = np.interp(common, pos_bonds, pos_en)
    neg_interp = np.interp(common, neg_bonds, neg_en)
    energy_diff = pos_interp - neg_interp
    print(f"Mean energy diff (+1 vs -1): {np.mean(energy_diff):.4f}")
    
    return {'pos_bonds': pos_bonds, 'pos_en': pos_en, 'neg_bonds': neg_bonds, 'neg_en': neg_en,
            'fits': fits, 'common_bonds': common, 'energy_diff': energy_diff}

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    
    print("="*60)
    print("Dataset 1: Random Charges - Latent Charge Recovery")
    print("="*60)
    rc = les_charge_recovery()
    
    print("\n" + "="*60)
    print("Dataset 2: Charged Dimer - Binding Energy Curves")
    print("="*60)
    cd = charged_dimer_analysis()
    
    print("\n" + "="*60)
    print("Dataset 3: Ag3 Charge States - PES Analysis")
    print("="*60)
    ag = ag3_analysis()
    
    np.savez('outputs/rc_results.npz',
             true_charges=rc['true_charges'], recovered_charges=rc['recovered_charges'],
             losses=rc['losses'], charge_errors=rc['charge_errors'])
    np.savez('outputs/cd_results.npz',
             distances=cd['distances'], energies=cd['energies'])
    for name, fit in cd['fits'].items():
        np.savez(f'outputs/cd_fit_{name.replace(" ","_").replace("/","_")}.npz', pred=fit['pred'])
    np.savez('outputs/ag_results.npz',
             pos_bonds=ag['pos_bonds'], pos_en=ag['pos_en'],
             neg_bonds=ag['neg_bonds'], neg_en=ag['neg_en'],
             common_bonds=ag['common_bonds'], energy_diff=ag['energy_diff'])
    
    summary = {
        'rc': {'correlation': float(rc['correlation']), 'final_mse': float(rc['charge_errors'][-1])},
        'cd': {n: {'mae': float(f['mae']), 'rmse': float(f['rmse'])} for n, f in cd['fits'].items()},
        'ag': {l: {'De': float(f['De']), 're': float(f['re']), 'mae': float(f['mae'])} for l, f in ag['fits'].items()}
    }
    with open('outputs/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nAll results saved.")
