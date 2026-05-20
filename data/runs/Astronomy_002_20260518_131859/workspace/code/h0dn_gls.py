"""
Generalized Least Squares fit to the H0 Distance Network minimal dataset.
"""
import numpy as np
import json

# Load dataset
with open('data/H0DN_MinimalDataset.txt', 'r') as f:
    exec(f.read())

# --- Assemble data ---

# Host measurements
host_meas = host_measurements  # list of tuples
# SNe Ia calibrators
sneia_cal = sneia_calibrators
# SBF calibrators
sbf_cal = sbf_calibrators
# Hubble flow SNe Ia
hf_sneia = hubble_flow_sneia
# Hubble flow SBF
hf_sbf = hubble_flow_sbf

# Identify unique hosts (with primary measurements)
hosts_prim = sorted({h for h, *_ in host_meas})
host_index = {h: i for i, h in enumerate(hosts_prim)}

# Identify groups
groups = sorted({g for g in host_group.values()})
group_index = {g: i for i, g in enumerate(groups)}

# Parameters:
# 0..N_host-1 : mu_host for each host with primary measurement
# N_host .. N_host+N_group-1 : mu_group for each group
# N_host+N_group : M_B (SN Ia absolute magnitude)
# N_host+N_group+1 : M_F110W (SBF absolute magnitude)
# N_host+N_group+2 : a = 5*log10(H0)

n_host = len(hosts_prim)
n_group = len(groups)
n_params = n_host + n_group + 3

param_names = hosts_prim + groups + ['M_B', 'M_F110W', 'a']

# Data vector and design matrix rows
data_rows = []
design_rows = []
data_labels = []

# 1) Host measurements
for h, method, anchor, mu, err in host_meas:
    data_rows.append(mu)
    row = np.zeros(n_params)
    row[host_index[h]] = 1.0
    design_rows.append(row)
    data_labels.append(f'host_{h}_{method}_{anchor}')

# 2) SNe Ia calibrators
for h, mB, err in sneia_cal:
    data_rows.append(mB)
    row = np.zeros(n_params)
    row[host_index[h]] = 1.0
    row[n_host + n_group] = 1.0  # M_B
    design_rows.append(row)
    data_labels.append(f'sneia_cal_{h}')

# 3) SBF calibrators
for h, mF, err in sbf_cal:
    g = host_group[h]
    data_rows.append(mF)
    row = np.zeros(n_params)
    row[n_host + group_index[g]] = 1.0  # mu_group
    row[n_host + n_group + 1] = 1.0  # M_F110W
    design_rows.append(row)
    data_labels.append(f'sbf_cal_{h}')

# 4) Hubble flow SNe Ia
for z, mB, err, sigv in hf_sneia:
    # constant term: 5*log10(c*z) + 25
    cz = c_km * z
    const = 5.0 * np.log10(cz) + 25.0
    data_rows.append(mB - const)
    row = np.zeros(n_params)
    row[n_host + n_group] = 1.0  # M_B
    row[n_host + n_group + 2] = -1.0  # -a
    design_rows.append(row)
    data_labels.append(f'hf_sneia_z{z}')

# 5) Hubble flow SBF
for z, mF, err, sigv in hf_sbf:
    cz = c_km * z
    const = 5.0 * np.log10(cz) + 25.0
    data_rows.append(mF - const)
    row = np.zeros(n_params)
    row[n_host + n_group + 1] = 1.0  # M_F110W
    row[n_host + n_group + 2] = -1.0  # -a
    design_rows.append(row)
    data_labels.append(f'hf_sbf_z{z}')

y = np.array(data_rows)
X = np.vstack(design_rows)
n_data = len(y)

# --- Covariance matrix ---
C = np.zeros((n_data, n_data))

# Diagonal noise
# Host measurements: err_meas^2
idx = 0
for h, method, anchor, mu, err in host_meas:
    C[idx, idx] += err**2
    idx += 1
# SNe Ia calibrators
for h, mB, err in sneia_cal:
    C[idx, idx] += err**2
    idx += 1
# SBF calibrators
for h, mF, err in sbf_cal:
    C[idx, idx] += err**2 + depth_scatter**2
    idx += 1
# Hubble flow SNe Ia
for z, mB, err, sigv in hf_sneia:
    cz = c_km * z
    sig_mu_pec = (5.0 / np.log(10)) * (sigv / cz)
    C[idx, idx] += err**2 + sig_mu_pec**2
    idx += 1
# Hubble flow SBF
for z, mF, err, sigv in hf_sbf:
    cz = c_km * z
    sig_mu_pec = (5.0 / np.log(10)) * (sigv / cz)
    C[idx, idx] += err**2 + sig_mu_pec**2
    idx += 1

# Off-diagonal covariances for host measurements
# Shared anchor uncertainties and method-anchor systematics
n_host_meas = len(host_meas)
for i in range(n_host_meas):
    h_i, method_i, anchor_i, mu_i, err_i = host_meas[i]
    for j in range(i+1, n_host_meas):
        h_j, method_j, anchor_j, mu_j, err_j = host_meas[j]
        cov = 0.0
        # Shared anchor uncertainty
        if anchor_i == anchor_j:
            cov += anchors[anchor_i]['err']**2
        # Shared method-anchor systematic
        if method_i == method_j and anchor_i == anchor_j:
            key = (method_i, anchor_i)
            if key in method_anchor_err:
                cov += method_anchor_err[key]**2
        C[i, j] += cov
        C[j, i] += cov

# Ensure matrix is positive definite (add tiny jitter if needed)
C += np.eye(n_data) * 1e-12

# --- GLS solution ---
C_inv = np.linalg.inv(C)
XtCinv = X.T @ C_inv
Cov_params = np.linalg.inv(XtCinv @ X)
p_hat = Cov_params @ (XtCinv @ y)

# Extract results
a_hat = p_hat[n_host + n_group + 2]
sigma_a = np.sqrt(Cov_params[n_host + n_group + 2, n_host + n_group + 2])

H0 = 10**a_hat
sigma_H0 = H0 * np.log(10) * sigma_a

print(f"H0 = {H0:.3f} ± {sigma_H0:.3f} km/s/Mpc")
print(f"a = 5 log10(H0) = {a_hat:.5f} ± {sigma_a:.5f}")

# Print other parameters
for name, val, err in zip(param_names, p_hat, np.sqrt(np.diag(Cov_params))):
    print(f"{name:12s}: {val:.4f} ± {err:.4f}")

# Save results
results = {
    'H0': float(H0),
    'sigma_H0': float(sigma_H0),
    'a': float(a_hat),
    'sigma_a': float(sigma_a),
    'parameters': {name: float(val) for name, val in zip(param_names, p_hat)},
    'parameter_errors': {name: float(err) for name, err in zip(param_names, np.sqrt(np.diag(Cov_params)))},
}
with open('outputs/h0dn_gls_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# --- Residuals and chi2 ---
residuals = y - X @ p_hat
chi2 = float(residuals.T @ C_inv @ residuals)
dof = n_data - n_params
print(f"chi2 = {chi2:.3f}, dof = {dof}, reduced chi2 = {chi2/dof:.3f}")

# Save residuals
np.savetxt('outputs/residuals.txt', residuals, fmt='%.5f')

# Save covariance matrix of parameters
np.savetxt('outputs/parameter_covariance.txt', Cov_params, fmt='%.6f')
