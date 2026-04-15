import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

data = pickle.load(open('../outputs/structured_data.pkl', 'rb'))

anchors = data['anchors']
host_meas = data['host_measurements']
snia_cal = data['sneia_calibrators']
flow_snia = data['hubble_flow_sneia']
method_anchor_err = data['method_anchor_err']
c_km = data['c_km']

sn_int_scatter = 0.08  # mag

print('SNIa ladder GLS.')

# Params: LMC, N4258, hosts..., M_B, logH0
anchor_idx = {'LMC': 0, 'N4258': 1}
hosts = sorted(np.unique(host_meas[:,0]))
host_idx = {h: 2 + i for i, h in enumerate(hosts)}
M_B_idx = 2 + len(hosts)
logH0_idx = M_B_idx + 1

param_names = ['LMC', 'N4258'] + hosts + ['M_B', 'log10_H0']
N_param = len(param_names)
param_to_idx = {p:i for i,p in enumerate(param_names)}

y = []
sigma = []
X = []

# Anchors prior
for a in anchor_idx:
    mu = anchors[a]['mu']
    err = anchors[a]['err']
    y.append(mu)
    sigma.append(err)
    xr = np.zeros(N_param)
    xr[anchor_idx[a]] = 1
    X.append(xr)

# Primaries
ma_groups = {}
for row in host_meas:
    h, _, a, mu_m, err_stat = row[0], row[1], row[2], float(row[3]), float(row[4])
    ma = (row[1], row[2])
    y.append(mu_m)
    sigma.append(err_stat)
    xr = np.zeros(N_param)
    xr[param_to_idx[h]] = 1
    X.append(xr)
    ma_groups.setdefault(ma, []).append(len(y)-1)

# SN cal
for row in snia_cal:
    h, mB, err_stat = row[0], float(row[1]), float(row[2])
    y.append(mB)
    sigma.append(np.sqrt(err_stat**2 + sn_int_scatter**2))
    xr = np.zeros(N_param)
    xr[param_to_idx[h]] = 1
    xr[M_B_idx] = 1
    X.append(xr)

# Flow SNIa
for row in flow_snia:
    z, mB, err_stat, vp_sigma = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    v = c_km * z
    sigma_vp_mu = 5 / np.log(10) * vp_sigma / v
    sigma_tot = np.sqrt(err_stat**2 + sigma_vp_mu**2 + sn_int_scatter**2)
    const = 5 * np.log10(c_km * z) + 25
    y.append(mB - const)
    sigma.append(sigma_tot)
    xr = np.zeros(N_param)
    xr[M_B_idx] = 1
    xr[logH0_idx] = -5
    X.append(xr)

y = np.array(y)
X = np.array(X)
sigma = np.array(sigma)

N_obs = len(y)
print('N_obs:', N_obs, 'N_param:', N_param)

C = np.diag(sigma**2)

# Syst for prim
prim_start = 2
for ma, ilist in ma_groups.items():
    sys_err = method_anchor_err.get(ma, 0.0)
    sys2 = sys_err**2
    for i in ilist:
        for j in ilist:
            C[i,j] += sys2

C_inv = np.linalg.inv(C)
beta = X.T @ C_inv @ X
y_beta = X.T @ C_inv @ y

theta = np.linalg.solve(beta, y_beta)
cov_th = np.linalg.inv(beta)

logH0 = theta[logH0_idx]
sigma_logH0 = np.sqrt(cov_th[logH0_idx,logH0_idx])
H0 = 10**logH0
sigma_H0 = H0 * np.log(10) * sigma_logH0

print('H0 = %.2f +/- %.2f km/s/Mpc' % (H0, sigma_H0))

res = y - X @ theta
chi2 = res.T @ C_inv @ res
dof = N_obs - N_param
print('chi2 = %.1f (dof=%d), reduced=%.2f' % (chi2, dof, chi2/dof))

# Save
out = {
    'H0': H0,
    'sigma_H0': sigma_H0,
    'chi2': chi2,
    'dof': dof,
    'theta': theta.tolist(),
    'param_names': param_names
}
with open('../outputs/H0_snia.json', 'w') as f:
    json.dump(out, f)

print('Saved H0_snia.json')