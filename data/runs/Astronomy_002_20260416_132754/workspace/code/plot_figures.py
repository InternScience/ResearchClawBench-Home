import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and functions
# (Copying the fit_h0 code)
anchors = {
    'N4258': {'mu': 29.397, 'err': 0.032},
    'LMC': {'mu': 18.477, 'err': 0.024},
    'MW': {'mu': 0.0, 'err': 0.0}
}
host_measurements = [
    ('NGC1309', 'Cepheid', 'N4258', 32.50, 0.10),
    ('NGC1365', 'Cepheid', 'N4258', 31.33, 0.08),
    ('NGC1448', 'Cepheid', 'N4258', 31.31, 0.09),
    ('NGC1559', 'Cepheid', 'N4258', 31.42, 0.07),
    ('M101', 'Cepheid', 'N4258', 29.12, 0.06),
    ('NGC1316', 'TRGB', 'N4258', 31.39, 0.10),
    ('NGC1365', 'TRGB', 'N4258', 31.32, 0.12),
    ('NGC5643', 'TRGB', 'N4258', 30.53, 0.09),
    ('M101', 'TRGB', 'N4258', 29.13, 0.08),
    ('NGC1309', 'Cepheid', 'LMC', 32.51, 0.11),
    ('NGC1365', 'Cepheid', 'LMC', 31.34, 0.09)
]
sneia_calibrators = [
    ('NGC1309', 12.10, 0.05),
    ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05),
    ('NGC1559', 12.22, 0.05),
    ('M101', 9.85, 0.04),
    ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06)
]
sbf_calibrators = [
    ('NGC1399', 28.35, 0.10),
    ('NGC1404', 28.33, 0.10),
    ('NGC4472', 28.56, 0.12)
]
hubble_flow_sneia = [
    (0.034, 15.12, 0.06, 250),
    (0.042, 15.68, 0.05, 250),
    (0.055, 16.35, 0.05, 250),
    (0.068, 17.02, 0.05, 250),
    (0.082, 17.55, 0.06, 250)
]
hubble_flow_sbf = [
    (0.023, 30.45, 0.15, 250),
    (0.031, 31.02, 0.15, 250),
    (0.045, 31.89, 0.16, 250)
]
method_anchor_err = {
    ('Cepheid', 'N4258'): 0.04,
    ('Cepheid', 'LMC'): 0.03,
    ('Cepheid', 'MW'): 0.02,
    ('TRGB', 'N4258'): 0.05
}
host_group = {
    'NGC1399': 'Fornax',
    'NGC1404': 'Fornax',
    'NGC4472': 'Virgo',
    'NGC1316': 'Fornax',
    'NGC1365': 'Fornax'
}

depth_scatter = 0.10
c_km = 299792.458
q0 = -0.55

def get_group(h): return host_group.get(h, h)

def fit_h0(use_sneia=True, use_sbf=True, use_cepheid=True, use_trgb=True, return_params=False):
    hosts = set()
    for h, method, anchor, _, _ in host_measurements:
        if (method == 'Cepheid' and use_cepheid) or (method == 'TRGB' and use_trgb):
            hosts.add(get_group(h))
    if use_sneia:
        for h, _, _ in sneia_calibrators: hosts.add(get_group(h))
    if use_sbf:
        for h, _, _ in sbf_calibrators: hosts.add(get_group(h))
        
    params = ['mu_' + a for a in anchors] + ['mu_' + h for h in hosts]
    if use_sneia: params.append('MB')
    if use_sbf: params.append('MF110W')
    params.append('H0_param')
    
    p_idx = {p: i for i, p in enumerate(params)}
    N_params = len(params)
    
    Y, A, C = [], [], []
    sys_methods = []
    
    for a, data in anchors.items():
        row = np.zeros(N_params)
        row[p_idx['mu_' + a]] = 1.0
        Y.append(data['mu'])
        A.append(row)
        C.append(data['err']**2 if data['err'] > 0 else 1e-10)
            
    idx_start_sys = len(Y)
    
    for h, method, anchor, mu_meas, err_meas in host_measurements:
        if not ((method == 'Cepheid' and use_cepheid) or (method == 'TRGB' and use_trgb)):
            continue
        row = np.zeros(N_params)
        row[p_idx['mu_' + get_group(h)]] = 1.0
        row[p_idx['mu_' + anchor]] = -1.0
        Y.append(mu_meas - anchors[anchor]['mu'])
        A.append(row)
        C.append(err_meas**2)
        sys_methods.append((method, anchor))
        
    if use_sneia:
        for h, mB, err_mB in sneia_calibrators:
            if get_group(h) not in hosts: continue
            row = np.zeros(N_params)
            row[p_idx['mu_' + get_group(h)]] = 1.0
            row[p_idx['MB']] = 1.0
            Y.append(mB)
            A.append(row)
            err = np.sqrt(err_mB**2 + depth_scatter**2) if h in host_group else err_mB
            C.append(err**2)
            sys_methods.append(None)
            
        for z, mB, err_mB, v_pec in hubble_flow_sneia:
            row = np.zeros(N_params)
            row[p_idx['MB']] = 1.0
            row[p_idx['H0_param']] = -1.0
            dl_h0 = z * (1 + 0.5 * (1 - q0) * z)
            y_val = mB - 5 * np.log10(c_km * dl_h0) - 25
            Y.append(y_val)
            A.append(row)
            err_v = (5 / np.log(10)) * (v_pec / (c_km * z))
            C.append(err_mB**2 + err_v**2)
            sys_methods.append(None)
            
    if use_sbf:
        for h, mF110W, err_mF110W in sbf_calibrators:
            if get_group(h) not in hosts: continue
            row = np.zeros(N_params)
            row[p_idx['mu_' + get_group(h)]] = 1.0
            row[p_idx['MF110W']] = 1.0
            Y.append(mF110W)
            A.append(row)
            err = np.sqrt(err_mF110W**2 + depth_scatter**2) if h in host_group else err_mF110W
            C.append(err**2)
            sys_methods.append(None)
            
        for z, mF110W, err_mF110W, v_pec in hubble_flow_sbf:
            row = np.zeros(N_params)
            row[p_idx['MF110W']] = 1.0
            row[p_idx['H0_param']] = -1.0
            dl_h0 = z * (1 + 0.5 * (1 - q0) * z)
            y_val = mF110W - 5 * np.log10(c_km * dl_h0) - 25
            Y.append(y_val)
            A.append(row)
            err_v = (5 / np.log(10)) * (v_pec / (c_km * z))
            C.append(err_mF110W**2 + err_v**2)
            sys_methods.append(None)
            
    Y = np.array(Y)
    A = np.array(A)
    C_matrix = np.diag(C)
    
    for i in range(len(sys_methods)):
        for j in range(len(sys_methods)):
            if sys_methods[i] is not None and sys_methods[j] is not None:
                if sys_methods[i] == sys_methods[j]:
                    err_sys = method_anchor_err.get(sys_methods[i], 0.0)
                    C_matrix[idx_start_sys + i, idx_start_sys + j] += err_sys**2
                    
    try:
        C_inv = la.inv(C_matrix)
        AT_Cinv = A.T @ C_inv
        Cov = la.inv(AT_Cinv @ A)
        X = Cov @ AT_Cinv @ Y
    except la.LinAlgError:
        return None, None
        
    H0_param = X[p_idx['H0_param']]
    H0_err = np.sqrt(Cov[p_idx['H0_param'], p_idx['H0_param']])
    H0 = 10**(H0_param / 5)
    H0_sigma = H0 * np.log(10) / 5 * H0_err
    
    if return_params:
        return H0, H0_sigma, X, Cov, params
    return H0, H0_sigma

# Variants
variants = {
    'Baseline (All Data)': fit_h0(),
    'Cepheid + SNe Ia': fit_h0(use_sbf=False, use_trgb=False),
    'TRGB + SNe Ia': fit_h0(use_sbf=False, use_cepheid=False),
    'Cepheid + SBF': fit_h0(use_sneia=False, use_trgb=False),
    'TRGB + SBF': fit_h0(use_sneia=False, use_cepheid=False)
}

# Plot H0 variants
plt.figure(figsize=(8, 5))
labels = list(variants.keys())
h0_vals = [v[0] for v in variants.values()]
h0_errs = [v[1] for v in variants.values()]

plt.errorbar(h0_vals, range(len(labels)), xerr=h0_errs, fmt='o', color='black', capsize=5)
plt.yticks(range(len(labels)), labels)
plt.axvline(x=73.50, color='red', linestyle='--', label='Literature Baseline (73.50 km/s/Mpc)')
plt.axvspan(73.50 - 0.81, 73.50 + 0.81, color='red', alpha=0.2)
plt.xlabel('$H_0$ (km/s/Mpc)')
plt.title('Hubble Constant Estimates from Different Analysis Variants')
plt.legend()
plt.tight_layout()
plt.savefig('report/images/h0_variants.png')

# Hubble Diagram
H0_base, H0_sigma, X, Cov, params = fit_h0(return_params=True)
p_idx = {p: i for i, p in enumerate(params)}
MB = X[p_idx['MB']]
MF110W = X[p_idx['MF110W']]

plt.figure(figsize=(8, 6))
# SNe Ia
z_sneia = [v[0] for v in hubble_flow_sneia]
mu_sneia = [v[1] - MB for v in hubble_flow_sneia]
mu_err_sneia = [np.sqrt(v[2]**2 + ((5/np.log(10))*(v[3]/(c_km*v[0])))**2) for v in hubble_flow_sneia]

plt.errorbar(z_sneia, mu_sneia, yerr=mu_err_sneia, fmt='o', color='blue', label='SNe Ia')

# SBF
z_sbf = [v[0] for v in hubble_flow_sbf]
mu_sbf = [v[1] - MF110W for v in hubble_flow_sbf]
mu_err_sbf = [np.sqrt(v[2]**2 + ((5/np.log(10))*(v[3]/(c_km*v[0])))**2) for v in hubble_flow_sbf]

plt.errorbar(z_sbf, mu_sbf, yerr=mu_err_sbf, fmt='s', color='green', label='SBF')

# Model line
z_range = np.linspace(0.015, 0.09, 100)
dl_h0_range = z_range * (1 + 0.5 * (1 - q0) * z_range)
mu_model = 5 * np.log10(c_km * dl_h0_range / H0_base) + 25
plt.plot(z_range, mu_model, color='black', linestyle='-', label=f'Model ($H_0$={H0_base:.1f})')

plt.xscale('log')
plt.xlabel('Redshift $z$')
plt.ylabel('Distance Modulus $\mu$')
plt.title('Hubble Diagram')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('report/images/hubble_diagram.png')

