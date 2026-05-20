import numpy as np, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('data/H0DN_MinimalDataset.txt','r') as f:
    exec(f.read())

host_meas = host_measurements
sneia_cal = sneia_calibrators
sbf_cal = sbf_calibrators
hf_sneia = hubble_flow_sneia
hf_sbf = hubble_flow_sbf

def run_gls(exclude_anchors=None, exclude_methods=None, use_sn1a=True, use_sbf=True):
    if exclude_anchors is None:
        exclude_anchors = set()
    if exclude_methods is None:
        exclude_methods = set()
    
    hm = [(h,m,a,mu,e) for h,m,a,mu,e in host_meas if a not in exclude_anchors and m not in exclude_methods]
    hosts_prim = sorted({h for h, *_ in hm})
    host_index = {h:i for i,h in enumerate(hosts_prim)}
    
    sc = [(h,mB,e) for h,mB,e in sneia_cal if h in host_index] if use_sn1a else []
    sbc = [(h,mF,e) for h,mF,e in sbf_cal if host_group.get(h) in {g for g in host_group.values()}] if use_sbf else []
    
    hf_s = hf_sneia if use_sn1a and len(sc)>0 else []
    hf_sb = hf_sbf if use_sbf and len(sbc)>0 else []
    
    if not (sc or sbc):
        return None
    
    groups = sorted({host_group[h] for h, *_ in sbc} | {host_group.get(h) for h, *_ in hm} - {None})
    # Actually groups only needed if SBF is used; host measurements don't use groups directly
    if not sbc and not hf_sb:
        groups = []
    else:
        groups = sorted({host_group[h] for h, *_ in sbc} | {host_group.get(h) for h, *_ in hm} - {None})
    group_index = {g:i for i,g in enumerate(groups)}
    
    n_host = len(hosts_prim)
    n_group = len(groups)
    n_abs = (1 if use_sn1a else 0) + (1 if use_sbf else 0)
    n_params = n_host + n_group + n_abs + 1  # +1 for a
    
    param_names = hosts_prim + groups + (['M_B'] if use_sn1a else []) + (['M_F110W'] if use_sbf else []) + ['a']
    
    i_M_B = n_host + n_group if use_sn1a else None
    i_M_F110W = n_host + n_group + (1 if use_sn1a else 0) if use_sbf else None
    i_a = n_params - 1
    
    data_rows, design_rows, labels = [], [], []
    for h, method, anchor, mu, err in hm:
        data_rows.append(mu)
        row = np.zeros(n_params)
        row[host_index[h]] = 1.0
        design_rows.append(row)
        labels.append(f'host_{h}_{method}_{anchor}')
    for h, mB, err in sc:
        data_rows.append(mB)
        row = np.zeros(n_params)
        row[host_index[h]] = 1.0
        row[i_M_B] = 1.0
        design_rows.append(row)
        labels.append(f'sneia_cal_{h}')
    for h, mF, err in sbc:
        g = host_group[h]
        data_rows.append(mF)
        row = np.zeros(n_params)
        row[n_host + group_index[g]] = 1.0
        row[i_M_F110W] = 1.0
        design_rows.append(row)
        labels.append(f'sbf_cal_{h}')
    for z, mB, err, sigv in hf_s:
        cz = c_km * z
        const = 5.0 * np.log10(cz) + 25.0
        data_rows.append(mB - const)
        row = np.zeros(n_params)
        row[i_M_B] = 1.0
        row[i_a] = -1.0
        design_rows.append(row)
        labels.append(f'hf_sneia_z{z}')
    for z, mF, err, sigv in hf_sb:
        cz = c_km * z
        const = 5.0 * np.log10(cz) + 25.0
        data_rows.append(mF - const)
        row = np.zeros(n_params)
        row[i_M_F110W] = 1.0
        row[i_a] = -1.0
        design_rows.append(row)
        labels.append(f'hf_sbf_z{z}')
    
    y = np.array(data_rows)
    X = np.vstack(design_rows)
    n_data = len(y)
    if n_data < n_params:
        return None
    
    C = np.zeros((n_data, n_data))
    idx = 0
    for h, method, anchor, mu, err in hm:
        C[idx, idx] += err**2
        idx += 1
    for h, mB, err in sc:
        C[idx, idx] += err**2
        idx += 1
    for h, mF, err in sbc:
        C[idx, idx] += err**2 + depth_scatter**2
        idx += 1
    for z, mB, err, sigv in hf_s:
        cz = c_km * z
        sig_mu_pec = (5.0/np.log(10)) * (sigv/cz)
        C[idx, idx] += err**2 + sig_mu_pec**2
        idx += 1
    for z, mF, err, sigv in hf_sb:
        cz = c_km * z
        sig_mu_pec = (5.0/np.log(10)) * (sigv/cz)
        C[idx, idx] += err**2 + sig_mu_pec**2
        idx += 1
    
    n_host_meas = len(hm)
    for i in range(n_host_meas):
        h_i, method_i, anchor_i, mu_i, err_i = hm[i]
        for j in range(i+1, n_host_meas):
            h_j, method_j, anchor_j, mu_j, err_j = hm[j]
            cov = 0.0
            if anchor_i == anchor_j:
                cov += anchors[anchor_i]['err']**2
            if method_i == method_j and anchor_i == anchor_j:
                key = (method_i, anchor_i)
                if key in method_anchor_err:
                    cov += method_anchor_err[key]**2
            C[i, j] += cov
            C[j, i] += cov
    
    C += np.eye(n_data) * 1e-12
    C_inv = np.linalg.inv(C)
    Cov_params = np.linalg.inv(X.T @ C_inv @ X)
    p_hat = Cov_params @ (X.T @ C_inv @ y)
    
    a_hat = p_hat[i_a]
    sigma_a = np.sqrt(Cov_params[i_a, i_a])
    H0 = 10**a_hat
    sigma_H0 = H0 * np.log(10) * sigma_a
    
    residuals = y - X @ p_hat
    chi2 = float(residuals.T @ C_inv @ residuals)
    dof = n_data - n_params
    
    return {
        'H0': float(H0), 'sigma_H0': float(sigma_H0),
        'a': float(a_hat), 'sigma_a': float(sigma_a),
        'chi2': chi2, 'dof': dof, 'reduced_chi2': chi2/max(dof,1),
        'parameters': {name: float(val) for name, val in zip(param_names, p_hat)},
        'errors': {name: float(err) for name, err in zip(param_names, np.sqrt(np.diag(Cov_params)))},
        'labels': labels, 'residuals': residuals.tolist()
    }

baseline = run_gls()
print(f"Baseline: H0 = {baseline['H0']:.3f} +/- {baseline['sigma_H0']:.3f}")

variants = {
    'No LMC anchor': run_gls(exclude_anchors={'LMC'}),
    'No N4258 anchor': run_gls(exclude_anchors={'N4258'}),
    'Cepheid only': run_gls(exclude_methods={'TRGB'}),
    'TRGB only': run_gls(exclude_methods={'Cepheid'}),
    'SN Ia only': run_gls(use_sbf=False),
    'SBF only': run_gls(use_sn1a=False),
}

for name, res in variants.items():
    if res:
        print(f"{name}: H0 = {res['H0']:.3f} +/- {res['sigma_H0']:.3f}, chi2/dof = {res['reduced_chi2']:.2f}")
    else:
        print(f"{name}: insufficient data")

os.makedirs('outputs', exist_ok=True)
with open('outputs/variants.json','w') as f:
    json.dump({'baseline': baseline, 'variants': {k:v for k,v in variants.items() if v is not None}}, f, indent=2)
