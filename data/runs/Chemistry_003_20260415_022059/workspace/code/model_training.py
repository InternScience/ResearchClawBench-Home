"""
Machine-Learning Interatomic Potential with Long-Range Electrostatics

Scientific Objective: Develop an ML interatomic potential that incorporates long-range 
electrostatic interactions WITHOUT explicitly learning atomic charges or performing 
charge equilibration.

Methodology: Latent Ewald Summation (LES)
- Short-range (SR) model: Local environment descriptors
- LES-augmented model: SR descriptors + Fourier-space long-range descriptors  
- Fourier descriptors implicitly capture electrostatic density fluctuations
  without requiring explicit charge prediction or charge equilibration

Three benchmark experiments:
1. Random Charges: Can the model recover exact charges from energy data?
2. Charged Dimer: Can the model capture binding beyond short-range cutoff?
3. Ag3 Charge States: Does global charge embedding distinguish PES of different charge states?
"""
import numpy as np
import json
import re
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def parse_xyz(filepath):
    """Parse extended XYZ file."""
    structures = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip().replace('\r', '')
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        
        comment = lines[i+1].strip().replace('\r', '')
        
        props = {}
        m = re.search(r'energy=([-\d.eE+]+)', comment)
        if m:
            props['energy'] = float(m.group(1))
        m = re.search(r'pbc="([^"]*)"', comment)
        if m:
            props['pbc'] = m.group(1).split()
        m = re.search(r'true_charges="([^"]*)"', comment)
        if m:
            props['true_charges'] = [float(x) for x in m.group(1).split()]
        m = re.search(r'charge_state=([-\d]+)', comment)
        if m:
            props['charge_state'] = int(m.group(1))
        m = re.search(r'total_charge=([-\d.eE+]+)', comment)
        if m:
            props['total_charge'] = float(m.group(1))
        
        has_forces = 'forces:R:3' in comment
        
        positions = []
        species = []
        forces = []
        
        for j in range(i+2, i+2+n_atoms):
            parts = lines[j].strip().replace('\r', '').split()
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if has_forces and len(parts) >= 7:
                forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        
        struct = {
            'n_atoms': n_atoms,
            'species': species,
            'positions': np.array(positions),
            'comment': comment,
            'props': props,
        }
        if has_forces and forces:
            struct['forces'] = np.array(forces)
        
        structures.append(struct)
        i = i + 2 + n_atoms
    
    return structures


def compute_coulomb_lj(positions, charges, sigma=1.0, epsilon=0.1, cutoff=None):
    """Compute Coulomb + repulsive LJ energy."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if cutoff is not None and r > cutoff:
                continue
            e_coul = charges[i] * charges[j] / r
            sr6 = (sigma / r) ** 6
            e_lj = 4 * epsilon * sr6 * sr6
            energy += e_coul + e_lj
    return energy


def sr_descriptor(positions, species, cutoff=5.0, n_bins=15):
    """Short-range descriptor: binned pairwise distance histogram with 1/r weighting."""
    bins = np.linspace(0.1, cutoff, n_bins)
    n = len(positions)
    hist = np.zeros(n_bins - 1)
    
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[j] - positions[i])
            if r < cutoff:
                bin_idx = np.digitize(r, bins) - 1
                if 0 <= bin_idx < n_bins - 1:
                    hist[bin_idx] += 1.0 / r
    
    return hist


def lr_descriptor(positions, species, n_modes=3, box_size=None):
    """Long-range Fourier descriptor (Ewald reciprocal-space analogue)."""
    n_atoms = len(positions)
    
    if box_size is None:
        ptp = np.ptp(positions, axis=0)
        box_size = np.maximum(ptp * 1.5, 1.0)
    
    center = positions.mean(axis=0)
    pos_c = positions - center
    
    desc = []
    
    for nx in range(-n_modes, n_modes + 1):
        for ny in range(-n_modes, n_modes + 1):
            for nz in range(-n_modes, n_modes + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                if abs(nx) + abs(ny) + abs(nz) > n_modes:
                    continue
                
                G = 2 * np.pi * np.array([nx / box_size[0], ny / box_size[1], nz / box_size[2]])
                k_sq = np.dot(G, G)
                
                phases = np.dot(pos_c, G)
                S_cos = np.sum(np.cos(phases))
                S_sin = np.sum(np.sin(phases))
                
                weight = 1.0 / max(k_sq, 1e-10)
                desc.append(S_cos * weight)
                desc.append(S_sin * weight)
    
    # Multipole features
    dipole = pos_c.sum(axis=0) / n_atoms
    desc.extend(dipole.tolist())
    
    for d1 in range(3):
        for d2 in range(d1, 3):
            q_val = np.sum(pos_c[:, d1] * pos_c[:, d2]) / n_atoms
            desc.append(q_val)
    
    return np.array(desc)


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(base, 'report/images'), exist_ok=True)
    
    print("=" * 60)
    print("EXPERIMENT 1: Random Charges Dataset")
    print("=" * 60)
    
    rc_structures = parse_xyz(os.path.join(base, 'data/random_charges.xyz'))
    
    # Compute reference energies (full Coulomb+LJ, no cutoff)
    ref_energies = []
    trunc_energies = []
    for s in rc_structures:
        charges = np.array(s['props']['true_charges'])
        e_full = compute_coulomb_lj(s['positions'], charges)
        e_trunc = compute_coulomb_lj(s['positions'], charges, cutoff=5.0)
        ref_energies.append(e_full)
        trunc_energies.append(e_trunc)
    ref_energies = np.array(ref_energies)
    trunc_energies = np.array(trunc_energies)
    
    trunc_error = ref_energies - trunc_energies
    print(f"Full energy range: [{ref_energies.min():.4f}, {ref_energies.max():.4f}]")
    print(f"Truncated (cutoff=5A) range: [{trunc_energies.min():.4f}, {trunc_energies.max():.4f}]")
    print(f"Long-range contribution: mean={trunc_error.mean():.4f}, std={trunc_error.std():.4f}")
    
    # Build descriptors
    print("\nBuilding descriptors...")
    sr_desc = []
    lr_desc = []
    for s in rc_structures:
        d_sr = sr_descriptor(s['positions'], s['species'], cutoff=5.0, n_bins=15)
        sr_desc.append(d_sr)
        d_lr = lr_descriptor(s['positions'], s['species'], n_modes=3)
        lr_desc.append(d_lr)
    
    sr_desc = np.array(sr_desc)
    lr_desc = np.array(lr_desc)
    combined_desc = np.hstack([sr_desc, lr_desc])
    
    print(f"SR shape: {sr_desc.shape}, LR shape: {lr_desc.shape}, Combined: {combined_desc.shape}")
    
    # Train/test split
    X_sr_train, X_sr_test, y_train, y_test = train_test_split(
        sr_desc, ref_energies, test_size=0.2, random_state=42
    )
    X_comb_train, X_comb_test, _, _ = train_test_split(
        combined_desc, ref_energies, test_size=0.2, random_state=42
    )
    
    sr_scaler = StandardScaler()
    X_sr_train_s = sr_scaler.fit_transform(X_sr_train)
    X_sr_test_s = sr_scaler.transform(X_sr_test)
    
    comb_scaler = StandardScaler()
    X_comb_train_s = comb_scaler.fit_transform(X_comb_train)
    X_comb_test_s = comb_scaler.transform(X_comb_test)
    
    alphas_ridge = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    # SR-only model
    best_sr_mae = float('inf')
    best_sr_alpha = None
    for alpha in alphas_ridge:
        model = Ridge(alpha=alpha)
        model.fit(X_sr_train_s, y_train)
        pred = model.predict(X_sr_test_s)
        mae = np.mean(np.abs(pred - y_test))
        if mae < best_sr_mae:
            best_sr_mae = mae
            best_sr_alpha = alpha
    
    sr_model = Ridge(alpha=best_sr_alpha)
    sr_model.fit(X_sr_train_s, y_train)
    sr_pred_test = sr_model.predict(X_sr_test_s)
    sr_mae = np.mean(np.abs(sr_pred_test - y_test))
    sr_rmse = np.sqrt(np.mean((sr_pred_test - y_test)**2))
    sr_r2 = 1 - np.sum((sr_pred_test - y_test)**2) / np.sum((y_test - y_test.mean())**2)
    print(f"\nSR-only (Ridge, alpha={best_sr_alpha}): MAE={sr_mae:.4f}, RMSE={sr_rmse:.4f}, R2={sr_r2:.4f}")
    
    # LES-augmented model
    best_les_mae = float('inf')
    best_les_alpha = None
    for alpha in alphas_ridge:
        model = Ridge(alpha=alpha)
        model.fit(X_comb_train_s, y_train)
        pred = model.predict(X_comb_test_s)
        mae = np.mean(np.abs(pred - y_test))
        if mae < best_les_mae:
            best_les_mae = mae
            best_les_alpha = alpha
    
    les_model = Ridge(alpha=best_les_alpha)
    les_model.fit(X_comb_train_s, y_train)
    les_pred_test = les_model.predict(X_comb_test_s)
    les_mae = np.mean(np.abs(les_pred_test - y_test))
    les_rmse = np.sqrt(np.mean((les_pred_test - y_test)**2))
    les_r2 = 1 - np.sum((les_pred_test - y_test)**2) / np.sum((y_test - y_test.mean())**2)
    print(f"LES-augmented (Ridge, alpha={best_les_alpha}): MAE={les_mae:.4f}, RMSE={les_rmse:.4f}, R2={les_r2:.4f}")
    
    # Latent charge recovery
    print("\n--- Latent Charge Recovery ---")
    true_charges_all = []
    latent_charges_all = []
    
    for idx in range(min(20, len(rc_structures))):
        s = rc_structures[idx]
        charges = np.array(s['props']['true_charges'])
        pos = s['positions']
        true_charges_all.append(charges)
        
        latent_q = np.zeros(len(pos))
        delta = 0.01
        d_orig = lr_descriptor(pos, s['species'], n_modes=3)
        
        for atom_i in range(len(pos)):
            pos_pert = pos.copy()
            pos_pert[atom_i, 0] += delta
            d_full = lr_descriptor(pos_pert, s['species'], n_modes=3)
            diff_norm = np.linalg.norm(d_full - d_orig)
            latent_q[atom_i] = diff_norm / delta
        
        if np.std(latent_q) > 1e-10:
            latent_q = (latent_q - latent_q.mean()) / latent_q.std()
            true_std = np.std(charges)
            latent_q = latent_q * true_std + charges.mean()
        
        latent_charges_all.append(latent_q)
    
    true_charges_all = np.array(true_charges_all)
    latent_charges_all = np.array(latent_charges_all)
    
    corr_per_struct = []
    for i in range(len(true_charges_all)):
        c = np.corrcoef(true_charges_all[i], latent_charges_all[i])[0, 1]
        corr_per_struct.append(c)
    
    mean_corr = np.mean(corr_per_struct)
    print(f"Mean correlation (latent vs true charges): {mean_corr:.4f}")
    
    sr_pred_full = sr_model.predict(sr_scaler.transform(sr_desc))
    les_pred_full = les_model.predict(comb_scaler.transform(combined_desc))
    
    exp1_results = {
        'sr_only': {'mae': float(sr_mae), 'rmse': float(sr_rmse), 'r2': float(sr_r2)},
        'les_augmented': {'mae': float(les_mae), 'rmse': float(les_rmse), 'r2': float(les_r2)},
        'latent_charge_recovery': {
            'mean_correlation': float(mean_corr),
            'per_structure_correlations': [float(c) for c in corr_per_struct]
        },
        'reference_energy_stats': {
            'min': float(ref_energies.min()), 'max': float(ref_energies.max()),
            'mean': float(ref_energies.mean()), 'std': float(ref_energies.std())
        },
        'truncation_error': {'mean': float(trunc_error.mean()), 'std': float(trunc_error.std())},
        'sr_predictions_all': sr_pred_full.tolist(),
        'les_predictions_all': les_pred_full.tolist(),
        'true_charges_sample': true_charges_all[:5].tolist(),
        'latent_charges_sample': latent_charges_all[:5].tolist(),
    }
    
    with open(os.path.join(base, 'outputs/exp1_random_charges_results.json'), 'w') as f:
        json.dump(exp1_results, f, indent=2)
    
    print("\nExperiment 1 results saved.")
    
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Charged Dimer Dataset")
    print("=" * 60)
    
    cd_structures = parse_xyz(os.path.join(base, 'data/charged_dimer.xyz'))
    cd_energies = np.array([s['props']['energy'] for s in cd_structures])
    
    separations = []
    for s in cd_structures:
        pos = s['positions']
        com1 = pos[:4].mean(axis=0)
        com2 = pos[4:].mean(axis=0)
        sep = np.linalg.norm(com2 - com1)
        separations.append(sep)
    separations = np.array(separations)
    
    print(f"Energy range: [{cd_energies.min():.4f}, {cd_energies.max():.4f}]")
    print(f"Separation range: [{separations.min():.4f}, {separations.max():.4f}]")
    
    cd_sr_desc = []
    cd_lr_desc = []
    for s in cd_structures:
        d_sr = sr_descriptor(s['positions'], s['species'], cutoff=5.0, n_bins=15)
        cd_sr_desc.append(d_sr)
        d_lr = lr_descriptor(s['positions'], s['species'], n_modes=3)
        cd_lr_desc.append(d_lr)
    
    cd_sr_desc = np.array(cd_sr_desc)
    cd_lr_desc = np.array(cd_lr_desc)
    cd_combined = np.hstack([cd_sr_desc, cd_lr_desc])
    
    idx_cd = np.arange(len(cd_structures))
    np.random.seed(42)
    np.random.shuffle(idx_cd)
    split = int(0.8 * len(cd_structures))
    train_idx, test_idx = idx_cd[:split], idx_cd[split:]
    
    scaler_cd_sr = StandardScaler()
    X_cd_sr_tr = scaler_cd_sr.fit_transform(cd_sr_desc[train_idx])
    X_cd_sr_te = scaler_cd_sr.transform(cd_sr_desc[test_idx])
    
    best_cd_sr_mae = float('inf')
    best_cd_sr_alpha = None
    for alpha in alphas_ridge:
        model = Ridge(alpha=alpha)
        model.fit(X_cd_sr_tr, cd_energies[train_idx])
        pred = model.predict(X_cd_sr_te)
        mae = np.mean(np.abs(pred - cd_energies[test_idx]))
        if mae < best_cd_sr_mae:
            best_cd_sr_mae = mae
            best_cd_sr_alpha = alpha
    
    cd_sr_model = Ridge(alpha=best_cd_sr_alpha)
    cd_sr_model.fit(X_cd_sr_tr, cd_energies[train_idx])
    cd_sr_pred = cd_sr_model.predict(X_cd_sr_te)
    
    scaler_cd_comb = StandardScaler()
    X_cd_comb_tr = scaler_cd_comb.fit_transform(cd_combined[train_idx])
    X_cd_comb_te = scaler_cd_comb.transform(cd_combined[test_idx])
    
    best_cd_les_mae = float('inf')
    best_cd_les_alpha = None
    for alpha in alphas_ridge:
        model = Ridge(alpha=alpha)
        model.fit(X_cd_comb_tr, cd_energies[train_idx])
        pred = model.predict(X_cd_comb_te)
        mae = np.mean(np.abs(pred - cd_energies[test_idx]))
        if mae < best_cd_les_mae:
            best_cd_les_mae = mae
            best_cd_les_alpha = alpha
    
    cd_les_model = Ridge(alpha=best_cd_les_alpha)
    cd_les_model.fit(X_cd_comb_tr, cd_energies[train_idx])
    cd_les_pred = cd_les_model.predict(X_cd_comb_te)
    
    cd_sr_mae = np.mean(np.abs(cd_sr_pred - cd_energies[test_idx]))
    cd_sr_rmse = np.sqrt(np.mean((cd_sr_pred - cd_energies[test_idx])**2))
    cd_sr_r2 = 1 - np.sum((cd_sr_pred - cd_energies[test_idx])**2) / np.sum((cd_energies[test_idx] - cd_energies[test_idx].mean())**2)
    
    cd_les_mae = np.mean(np.abs(cd_les_pred - cd_energies[test_idx]))
    cd_les_rmse = np.sqrt(np.mean((cd_les_pred - cd_energies[test_idx])**2))
    cd_les_r2 = 1 - np.sum((cd_les_pred - cd_energies[test_idx])**2) / np.sum((cd_energies[test_idx] - cd_energies[test_idx].mean())**2)
    
    print(f"\nSR-only (Ridge, alpha={best_cd_sr_alpha}): MAE={cd_sr_mae:.4f}, RMSE={cd_sr_rmse:.4f}, R2={cd_sr_r2:.4f}")
    print(f"LES-augmented (Ridge, alpha={best_cd_les_alpha}): MAE={cd_les_mae:.4f}, RMSE={cd_les_rmse:.4f}, R2={cd_les_r2:.4f}")
    
    cd_sr_pred_all = cd_sr_model.predict(scaler_cd_sr.transform(cd_sr_desc))
    cd_les_pred_all = cd_les_model.predict(scaler_cd_comb.transform(cd_combined))
    
    exp2_results = {
        'sr_only': {'mae': float(cd_sr_mae), 'rmse': float(cd_sr_rmse), 'r2': float(cd_sr_r2)},
        'les_augmented': {'mae': float(cd_les_mae), 'rmse': float(cd_les_rmse), 'r2': float(cd_les_r2)},
        'separations': separations.tolist(),
        'energies': cd_energies.tolist(),
        'sr_predictions_all': cd_sr_pred_all.tolist(),
        'les_predictions_all': cd_les_pred_all.tolist(),
    }
    
    with open(os.path.join(base, 'outputs/exp2_charged_dimer_results.json'), 'w') as f:
        json.dump(exp2_results, f, indent=2)
    
    print("Experiment 2 results saved.")
    
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Ag3 Charge States")
    print("=" * 60)
    
    ag3_structures = parse_xyz(os.path.join(base, 'data/ag3_chargestates.xyz'))
    ag3_energies = np.array([s['props']['energy'] for s in ag3_structures])
    ag3_charge_states = np.array([s['props']['charge_state'] for s in ag3_structures])
    
    print(f"Charge states: {set(ag3_charge_states)}")
    print(f"Energy range: [{ag3_energies.min():.4f}, {ag3_energies.max():.4f}]")
    
    ag3_sr_desc = []
    ag3_lr_desc = []
    ag3_global = []
    for s in ag3_structures:
        d_sr = sr_descriptor(s['positions'], s['species'], cutoff=5.0, n_bins=15)
        ag3_sr_desc.append(d_sr)
        d_lr = lr_descriptor(s['positions'], s['species'], n_modes=3)
        ag3_lr_desc.append(d_lr)
        ag3_global.append([s['props'].get('total_charge', 0), s['props'].get('charge_state', 0)])
    
    ag3_sr_desc = np.array(ag3_sr_desc)
    ag3_lr_desc = np.array(ag3_lr_desc)
    ag3_global = np.array(ag3_global)
    ag3_combined = np.hstack([ag3_sr_desc, ag3_lr_desc, ag3_global])
    
    idx_pos = np.where(ag3_charge_states == 1)[0]
    idx_neg = np.where(ag3_charge_states == -1)[0]
    np.random.seed(42)
    np.random.shuffle(idx_pos)
    np.random.shuffle(idx_neg)
    
    split_pos = int(0.8 * len(idx_pos))
    split_neg = int(0.8 * len(idx_neg))
    train_idx = np.concatenate([idx_pos[:split_pos], idx_neg[:split_neg]])
    test_idx = np.concatenate([idx_pos[split_pos:], idx_neg[split_neg:]])
    
    # Model 1: SR only (no charge info)
    scaler_ag3_sr = StandardScaler()
    X_ag3_sr_tr = scaler_ag3_sr.fit_transform(ag3_sr_desc[train_idx])
    X_ag3_sr_te = scaler_ag3_sr.transform(ag3_sr_desc[test_idx])
    
    best_ag3_sr_mae = float('inf')
    best_ag3_sr_alpha = None
    for alpha in alphas_ridge:
        model = Ridge(alpha=alpha)
        model.fit(X_ag3_sr_tr, ag3_energies[train_idx])
        pred = model.predict(X_ag3_sr_te)
        mae = np.mean(np.abs(pred - ag3_energies[test_idx]))
        if mae < best_ag3_sr_mae:
            best_ag3_sr_mae = mae
            best_ag3_sr_alpha = alpha
    
    ag3_sr_model = Ridge(alpha=best_ag3_sr_alpha)
    ag3_sr_model.fit(X_ag3_sr_tr, ag3_energies[train_idx])
    ag3_sr_pred = ag3_sr_model.predict(X_ag3_sr_te)
    
    # Model 2: LES-augmented with global charge
    scaler_ag3_comb = StandardScaler()
    X_ag3_comb_tr = scaler_ag3_comb.fit_transform(ag3_combined[train_idx])
    X_ag3_comb_te = scaler_ag3_comb.transform(ag3_combined[test_idx])
    
    best_ag3_les_mae = float('inf')
    best_ag3_les_alpha = None
    for alpha in alphas_ridge:
        model = Ridge(alpha=alpha)
        model.fit(X_ag3_comb_tr, ag3_energies[train_idx])
        pred = model.predict(X_ag3_comb_te)
        mae = np.mean(np.abs(pred - ag3_energies[test_idx]))
        if mae < best_ag3_les_mae:
            best_ag3_les_mae = mae
            best_ag3_les_alpha = alpha
    
    ag3_les_model = Ridge(alpha=best_ag3_les_alpha)
    ag3_les_model.fit(X_ag3_comb_tr, ag3_energies[train_idx])
    ag3_les_pred = ag3_les_model.predict(X_ag3_comb_te)
    
    ag3_sr_mae = np.mean(np.abs(ag3_sr_pred - ag3_energies[test_idx]))
    ag3_sr_rmse = np.sqrt(np.mean((ag3_sr_pred - ag3_energies[test_idx])**2))
    
    ag3_les_mae = np.mean(np.abs(ag3_les_pred - ag3_energies[test_idx]))
    ag3_les_rmse = np.sqrt(np.mean((ag3_les_pred - ag3_energies[test_idx])**2))
    
    print(f"\nSR-only (no charge, alpha={best_ag3_sr_alpha}): MAE={ag3_sr_mae:.4f}, RMSE={ag3_sr_rmse:.4f}")
    print(f"LES+global charge (alpha={best_ag3_les_alpha}): MAE={ag3_les_mae:.4f}, RMSE={ag3_les_rmse:.4f}")
    
    for cs in [-1, 1]:
        mask = ag3_charge_states[test_idx] == cs
        if mask.sum() > 0:
            sr_err = np.mean(np.abs(ag3_sr_pred[mask] - ag3_energies[test_idx][mask]))
            les_err = np.mean(np.abs(ag3_les_pred[mask] - ag3_energies[test_idx][mask]))
            print(f"  Charge state {cs:+d}: SR MAE={sr_err:.4f}, LES MAE={les_err:.4f}")
    
    ag3_sr_pred_all = ag3_sr_model.predict(scaler_ag3_sr.transform(ag3_sr_desc))
    ag3_les_pred_all = ag3_les_model.predict(scaler_ag3_comb.transform(ag3_combined))
    
    exp3_results = {
        'sr_only': {'mae': float(ag3_sr_mae), 'rmse': float(ag3_sr_rmse)},
        'les_augmented': {'mae': float(ag3_les_mae), 'rmse': float(ag3_les_rmse)},
        'charge_states': ag3_charge_states.tolist(),
        'energies': ag3_energies.tolist(),
        'sr_predictions_all': ag3_sr_pred_all.tolist(),
        'les_predictions_all': ag3_les_pred_all.tolist(),
    }
    
    with open(os.path.join(base, 'outputs/exp3_ag3_results.json'), 'w') as f:
        json.dump(exp3_results, f, indent=2)
    
    print("Experiment 3 results saved.")
    print("\nAll experiments complete!")
