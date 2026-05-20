import numpy as np

with open('data/H0DN_MinimalDataset.txt', 'r') as f:
    exec(f.read())

host_meas = host_measurements
sneia_cal = sneia_calibrators
sbf_cal = sbf_calibrators
hf_sneia = hubble_flow_sneia
hf_sbf = hubble_flow_sbf

hosts_prim = sorted({h for h, *_ in host_meas})
host_index = {h: i for i, h in enumerate(hosts_prim)}

groups = sorted({g for g in host_group.values()})
group_index = {g: i for i, g in enumerate(groups)}

n_host = len(hosts_prim)
n_group = len(groups)
n_params = n_host + n_group + 3

param_names = hosts_prim + groups + ['M_B', 'M_F110W', 'a']

data_rows = []
design_rows = []
labels = []

for h, method, anchor, mu, err in host_meas:
    data_rows.append(mu)
    row = np.zeros(n_params)
    row[host_index[h]] = 1.0
    design_rows.append(row)
    labels.append(f'host_{h}_{method}_{anchor}')

for h, mB, err in sneia_cal:
    data_rows.append(mB)
    row = np.zeros(n_params)
    row[host_index[h]] = 1.0
    row[n_host + n_group] = 1.0
    design_rows.append(row)
    labels.append(f'sneia_cal_{h}')

for h, mF, err in sbf_cal:
    g = host_group[h]
    data_rows.append(mF)
    row = np.zeros(n_params)
    row[n_host + group_index[g]] = 1.0
    row[n_host + n_group + 1] = 1.0
    design_rows.append(row)
    labels.append(f'sbf_cal_{h}')

for z, mB, err, sigv in hf_sneia:
    cz = c_km * z
    const = 5.0 * np.log10(cz) + 25.0
    data_rows.append(mB - const)
    row = np.zeros(n_params)
    row[n_host + n_group] = 1.0
    row[n_host + n_group + 2] = -1.0
    design_rows.append(row)
    labels.append(f'hf_sneia_z{z}')

for z, mF, err, sigv in hf_sbf:
    cz = c_km * z
    const = 5.0 * np.log10(cz) + 25.0
    data_rows.append(mF - const)
    row = np.zeros(n_params)
    row[n_host + n_group + 1] = 1.0
    row[n_host + n_group + 2] = -1.0
    design_rows.append(row)
    labels.append(f'hf_sbf_z{z}')

y = np.array(data_rows)
X = np.vstack(design_rows)

C = np.zeros((len(y), len(y)))
idx = 0
for h, method, anchor, mu, err in host_meas:
    C[idx, idx] += err**2
    idx += 1
for h, mB, err in sneia_cal:
    C[idx, idx] += err**2
    idx += 1
for h, mF, err in sbf_cal:
    C[idx, idx] += err**2 + depth_scatter**2
    idx += 1
for z, mB, err, sigv in hf_sneia:
    cz = c_km * z
    sig_mu_pec = (5.0 / np.log(10)) * (sigv / cz)
    C[idx, idx] += err**2 + sig_mu_pec**2
    idx += 1
for z, mF, err, sigv in hf_sbf:
    cz = c_km * z
    sig_mu_pec = (5.0 / np.log(10)) * (sigv / cz)
    C[idx, idx] += err**2 + sig_mu_pec**2
    idx += 1

n_host_meas = len(host_meas)
for i in range(n_host_meas):
    h_i, method_i, anchor_i, mu_i, err_i = host_meas[i]
    for j in range(i+1, n_host_meas):
        h_j, method_j, anchor_j, mu_j, err_j = host_meas[j]
        cov = 0.0
        if anchor_i == anchor_j:
            cov += anchors[anchor_i]['err']**2
        if method_i == method_j and anchor_i == anchor_j:
            key = (method_i, anchor_i)
            if key in method_anchor_err:
                cov += method_anchor_err[key]**2
        C[i, j] += cov
        C[j, i] += cov

C += np.eye(len(y)) * 1e-12
C_inv = np.linalg.inv(C)
Cov_params = np.linalg.inv(X.T @ C_inv @ X)
p_hat = Cov_params @ (X.T @ C_inv @ y)

res = y - X @ p_hat
chi2 = res.T @ C_inv @ res
print(f"chi2 = {chi2:.2f}, dof = {len(y)-n_params}")
for label, r, c_diag in zip(labels, res, np.diag(C)):
    print(f"{label:30s} residual = {r:8.3f}  sigma = {np.sqrt(c_diag):6.3f}")

a_hat = p_hat[-1]
print(f"\na = {a_hat:.5f}, H0 = {10**(a_hat/5):.3f}")
