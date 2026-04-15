import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

# Load structured data
with open('../outputs/structured_data.pkl', 'rb') as f:
    structured = pickle.load(f)

anchors = structured['anchors']
host_meas = structured['host_measurements']
snia_cal = structured['sneia_calibrators']
sbf_cal = structured['sbf_calibrators']
flow_snia = structured['hubble_flow_sneia']
flow_sbf = structured['hubble_flow_sbf']
method_anchor_err = structured['method_anchor_err']
host_group = structured['host_group']
depth_scatter = structured['depth_scatter']
c_km = structured['c_km']

int_sn = 0.1
int_sbf = 0.15

print('Full GLS with fixed model.')

# Param indices
anchor_names = ['LMC', 'N4258']
anchor_prior_mu = np.array([anchors[a]['mu'] for a in anchor_names])
anchor_prior_sigma = np.array([anchors[a]['err'] for a in anchor_names])

primary_hosts = sorted(np.unique(host_meas[:,0]))
primary_host_idx = {h: len(anchor_names) + i for i, h in enumerate(primary_hosts)}

M_B_idx = len(anchor_names) + len(primary_hosts)

group_names = sorted(np.unique(list(host_group.values())))
group_idx = {g: M_B_idx + 1 + i for i, g in enumerate(group_names)}

sbf_host_start = M_B_idx + 1 + len(group_names)
sbf_hosts_sorted = sorted(sbf_cal[:,0])
sbf_host_idx = {h: sbf_host_start + i for i, h in enumerate(sbf_hosts_sorted)}

M_SBF_idx = sbf_host_start + len(sbf_hosts_sorted)
logH0_idx = M_SBF_idx + 1

param_names = anchor_names + primary_hosts + ['M_B'] + group_names + sbf_hosts_sorted + ['M_SBF', 'log10_H0']
N_param = len(param_names)
param_to_idx = {p:i for i,p in enumerate(param_names)}

print('N_param:', N_param)
print('logH0_idx:', logH0_idx)

# y, sigma, X
y_obs = []
sigma_obs = []
X_obs = []

# 1. Anchors prior
for i, a in enumerate(anchor_names):
    y_obs.append(anchor_prior_mu[i])
    sigma_obs.append(anchor_prior_sigma[i])
    x_row = np.zeros(N_param)
    x_row[param_to_idx[a]] = 1
    X_obs.append(x_row)

# 2. Primary : mu_meas = mu_host - mu_a
ma_groups = {}
for row in host_meas:
    h, m, a, mu_meas, err_meas = row[0], row[1], row[2], float(row[3]), float(row[4])
    y_obs.append(mu_meas)
    sigma_obs.append(err_meas)
    x_row = np.zeros(N_param)
    x_row[param_to_idx[h]] = 1
    x_row[param_to_idx[a]] = -1
    X_obs.append(x_row)
    ma = (m, a)
    ma_groups.setdefault(ma, []).append(len(y_obs)-1)

# 3. SNIa cal mB = mu_host + M_B
for row in snia_cal:
    h, mB, err_mB = row[0], float(row[1]), float(row[2])
    y_obs.append(mB)
    sigma_obs.append(np.sqrt(err_mB**2 + int_sn**2))
    x_row = np.zeros(N_param)
    x_row[param_to_idx[h]] = 1
    x_row[M_B_idx] = 1
    X_obs.append(x_row)

# 4. SBF depth
for row in sbf_cal:
    h = row[0]
    g = host_group[h]
    y_obs.append(0)
    sigma_obs.append(depth_scatter)
    x_row = np.zeros(N_param)
    x_row[param_to_idx[h]] = 1
    x_row[param_to_idx[g]] = -1
    X_obs.append(x_row)

# 5. SBF cal mF110W = mu_host + M_SBF
for row in sbf_cal:
    h, mF, err_mF = row[0], float(row[1]), float(row[2])
    y_obs.append(mF)
    sigma_obs.append(np.sqrt(err_mF**2 + int_sbf**2))
    x_row = np.zeros(N_param)
    x_row[param_to_idx[h]] = 1
    x_row[M_SBF_idx] = 1
    X_obs.append(x_row)

# 6. Flow SNIa
for row in flow_snia:
    z, mB, err_mB, vp_km = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    cz = c_km * z
    sigma_vp_mu = (5 / np.log(10)) * vp_km / cz
    sigma_tot = np.sqrt(err_mB**2 + sigma_vp_mu**2 + int_sn**2)
    const = 5 * np.log10(cz) + 25
    y_obs.append(mB - const)
    sigma_obs.append(sigma_tot)
    x_row = np.zeros(N_param)
    x_row[M_B_idx] = 1
    x_row[logH0_idx] = -5
    X_obs.append(x_row)

# 7. Flow SBF
for row in flow_sbf:
    z, mF, err_mF, vp_km = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    cz = c_km * z
    sigma_vp_mu = (5 / np.log(10)) * vp_km / cz
    sigma_tot = np.sqrt(err_mF**2 + sigma_vp_mu**2 + int_sbf**2)
    const = 5 * np.log10(cz) + 25
    y_obs.append(mF - const)
    sigma_obs.append(sigma_tot)
    x_row = np.zeros(N_param)
    x_row[M_SBF_idx] = 1
    x_row[logH0_idx] = -5
    X_obs.append(x_row)

y = np.array(y_obs)
X = np.array(X_obs)
sigma_stat = np.array(sigma_obs)

N_obs = len(y)

C = np.diag(sigma_stat**2)

# Syst ma for prim
prim_start = len(anchor_names)
for ma, idxs in ma_groups.items():
    sys_err = method_anchor_err.get(ma, 0.0)
    sys2 = sys_err**2
    for i in idxs:
        C[i,i] += sys2  # diagonal only for simplicity, or full common
        # for common sys, full C[i,j] += sys2 for all i,j in group

# For common sys, since sys_ma common bias for the group, add full matrix.
# Yes, better for cov.
    for i in idxs:
        for j in idxs:
            C[i,j] += sys2

C_inv = np.linalg.inv(C)
XtCX = X.T @ C_inv @ X
Xty = X.T @ C_inv @ y

theta = np.linalg.solve(XtCX, Xty)
cov_theta = np.linalg.inv(XtCX)

logH0 = theta[logH0_idx]
sigma_log = np.sqrt(cov_theta[logH0_idx,logH0_idx])
H0 = 10**logH0
sigma_H0 = H0 * np.log(10) * sigma_log

print('Consensus H0 = %.2f +/- %.2f km/s/Mpc' % (H0, sigma_H0))

res = y - X @ theta
chi2 = res.T @ C_inv @ res
dof = N_obs - N_param
redchi = chi2 / dof
print('chi2/dof = %.2f / %d (%.2f)' % (chi2, dof, redchi))

res_dict = {
 'H0': float(H0),
 'sigma_H0': float(sigma_H0),
 'chi2': float(chi2),
 'dof': dof,
 'param_names': param_names,
 'theta': theta.tolist()
}
with open('../outputs/H0_baseline.json', 'w') as f:
    json.dump(res_dict, f, indent=2)

# Plots
plt.figure(figsize=(12,10))
plt.imshow(cov_theta, cmap='RdYlBu_r', aspect='auto')
plt.colorbar()
ticks = np.arange(0, N_param, max(1, N_param//15))
plt.xticks(ticks, [param_names[int(t)] for t in ticks], rotation=90)
plt.yticks(ticks, [param_names[int(t)] for t in ticks])
plt.title('Covariance Matrix')
plt.savefig('../report/images/cov_matrix.png')
plt.close()

plt.figure(figsize=(12,6))
theta_std = np.sqrt(np.diag(cov_theta))
plt.errorbar(range(N_param), theta, theta_std, fmt='o')
plt.xticks(range(0,N_param,2), param_names[::2], rotation=90)
plt.title('Parameters +/- sigma')
plt.savefig('../report/images/param_values.png')
plt.close()

plt.figure(figsize=(8,6))
plt.errorbar(['Network', 'CMB'], [H0, 67.4], yerr=[sigma_H0, 0.5], fmt='o', capsize=5)
plt.ylabel('H_0 (km/s/Mpc)')
plt.title('H_0 Comparison')
plt.savefig('../report/images/h0_compare.png')
plt.close()

print('Done, figures saved.')