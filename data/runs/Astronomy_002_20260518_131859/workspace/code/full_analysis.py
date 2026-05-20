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
    group_index = {g:i for i,g in enumerate(groups)}
    
    n_host = len(hosts_prim)
    n_group = len(groups)
    n_abs = (1 if use_sn1a else 0) + (1 if use_sbf else 0)
    n_params = n_host + n_group + n_abs + 1
    
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
    
    for h, g in host_group.items():
        if h in host_index and g in group_index:
            data_rows.append(0.0)
            row = np.zeros(n_params)
            row[host_index[h]] = 1.0
            row[n_host + group_index[g]] = -1.0
            design_rows.append(row)
            labels.append(f'group_{h}_{g}')
    
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
    for h, g in host_group.items():
        if h in host_index and g in group_index:
            C[idx, idx] += depth_scatter**2
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
    info_mat = X.T @ C_inv @ X
    try:
        Cov_params = np.linalg.inv(info_mat)
    except np.linalg.LinAlgError:
        Cov_params = np.linalg.pinv(info_mat)
    p_hat = Cov_params @ (X.T @ C_inv @ y)
    
    a_hat = p_hat[i_a]
    sigma_a = np.sqrt(Cov_params[i_a, i_a])
    # a = 5 * log10(H0), so H0 = 10^(a/5)
    H0 = 10**(a_hat / 5.0)
    sigma_H0 = H0 * np.log(10) * sigma_a / 5.0
    
    residuals = y - X @ p_hat
    chi2 = float(residuals.T @ C_inv @ residuals)
    dof = n_data - n_params
    
    return {
        'H0': float(H0), 'sigma_H0': float(sigma_H0),
        'a': float(a_hat), 'sigma_a': float(sigma_a),
        'chi2': chi2, 'dof': dof, 'reduced_chi2': chi2/max(dof,1),
        'parameters': {name: float(val) for name, val in zip(param_names, p_hat)},
        'errors': {name: float(err) for name, err in zip(param_names, np.sqrt(np.diag(Cov_params)))},
        'labels': labels, 'residuals': residuals.tolist(),
        'hosts': hosts_prim, 'groups': groups
    }

# Run all variants
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

# ============================================================
# Figures
# ============================================================
os.makedirs('report/images', exist_ok=True)

# 1. Hubble diagram for SNe Ia
fig, ax = plt.subplots(figsize=(8,6))
M_B = baseline['parameters']['M_B']
for z, mB, err, sigv in hf_sneia:
    cz = c_km * z
    mu_obs = mB - M_B
    mu_theo = 5*np.log10(cz/baseline['H0']) + 25
    ax.scatter(mu_theo, mu_obs, c='steelblue', s=80, zorder=3)
    ax.errorbar(mu_theo, mu_obs, yerr=err, fmt='none', c='steelblue', alpha=0.5)
ax.plot([30, 40], [30, 40], 'k--', lw=1)
ax.set_xlabel(r'$\mu_{\rm theo} = 5\log_{10}(cz/H_0)+25$')
ax.set_ylabel(r'$\mu_{\rm obs} = m_B - M_B$')
h0_str = f"{baseline['H0']:.1f}"
ax.set_title(r'SNe Ia Hubble Diagram ($H_0=' + h0_str + r'$ km/s/Mpc)')
fig.tight_layout()
fig.savefig('report/images/hubble_diagram_sneia.png', dpi=150)
plt.close(fig)

# 2. Residuals plot
fig, ax = plt.subplots(figsize=(10,6))
res = np.array(baseline['residuals'])
labels = baseline['labels']
errors = []
idx = 0
for h, method, anchor, mu, err in host_meas:
    errors.append(err)
    idx += 1
for h, g in host_group.items():
    if h in baseline['hosts'] and g in baseline['groups']:
        errors.append(depth_scatter)
        idx += 1
for h, mB, err in sneia_cal:
    errors.append(err)
    idx += 1
for h, mF, err in sbf_cal:
    errors.append(np.sqrt(err**2 + depth_scatter**2))
    idx += 1
for z, mB, err, sigv in hf_sneia:
    cz = c_km * z
    sig_mu_pec = (5.0/np.log(10)) * (sigv/cz)
    errors.append(np.sqrt(err**2 + sig_mu_pec**2))
    idx += 1
for z, mF, err, sigv in hf_sbf:
    cz = c_km * z
    sig_mu_pec = (5.0/np.log(10)) * (sigv/cz)
    errors.append(np.sqrt(err**2 + sig_mu_pec**2))
    idx += 1

errors = np.array(errors)
colors = []
for l in labels:
    if l.startswith('host'):
        colors.append('tab:blue')
    elif l.startswith('group'):
        colors.append('tab:green')
    elif l.startswith('sneia_cal'):
        colors.append('tab:red')
    elif l.startswith('sbf_cal'):
        colors.append('tab:orange')
    elif l.startswith('hf_sneia'):
        colors.append('tab:purple')
    else:
        colors.append('tab:brown')

ax.barh(range(len(res)), res/errors, color=colors, alpha=0.7)
ax.axvline(0, color='k', lw=0.5)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel('Residual / Error')
ax.set_title('Standardized Residuals from GLS Fit')
fig.tight_layout()
fig.savefig('report/images/residuals.png', dpi=150)
plt.close(fig)

# 3. Comparison of H0 variants
fig, ax = plt.subplots(figsize=(8,5))
names = ['Baseline'] + list(variants.keys())
vals = [baseline['H0']] + [variants[k]['H0'] if variants[k] else np.nan for k in variants]
errs = [baseline['sigma_H0']] + [variants[k]['sigma_H0'] if variants[k] else np.nan for k in variants]
mask = ~np.isnan(vals)
names_f = [n for n,m in zip(names, mask) if m]
vals_f = [v for v,m in zip(vals, mask) if m]
errs_f = [e for e,m in zip(errs, mask) if m]
ax.errorbar(vals_f, range(len(vals_f)), xerr=errs_f, fmt='o', capsize=4, color='darkred')
ax.set_yticks(range(len(vals_f)))
ax.set_yticklabels(names_f)
ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
ax.set_title(r'$H_0$ from Analysis Variants')
ax.axvline(73.5, color='k', ls='--', lw=1, label='Literature baseline (73.50)')
ax.legend()
fig.tight_layout()
fig.savefig('report/images/h0_variants.png', dpi=150)
plt.close(fig)

# 4. Host distance moduli comparison
fig, ax = plt.subplots(figsize=(8,5))
hosts = baseline['hosts']
mu_fit = [baseline['parameters'][h] for h in hosts]
mu_err = [baseline['errors'][h] for h in hosts]
mu_meas = {}
for h, method, anchor, mu, err in host_meas:
    if h not in mu_meas:
        mu_meas[h] = []
    mu_meas[h].append((method, anchor, mu, err))

x = np.arange(len(hosts))
width = 0.2
for i, h in enumerate(hosts):
    ax.errorbar(i, mu_fit[i], yerr=mu_err[i], fmt='s', color='darkblue', markersize=8, capsize=4, label='GLS fit' if i==0 else '')
    if h in mu_meas:
        for j, (method, anchor, mu, err) in enumerate(mu_meas[h]):
            offset = (j - len(mu_meas[h])/2 + 0.5) * width
            color = 'tab:orange' if method=='Cepheid' else 'tab:green'
            ax.errorbar(i+offset, mu, yerr=err, fmt='o', color=color, markersize=5, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(hosts, rotation=45, ha='right')
ax.set_ylabel(r'Distance Modulus $\mu$ [mag]')
ax.set_title('Host Galaxy Distance Moduli: Measurements vs. GLS Fit')
ax.legend()
fig.tight_layout()
fig.savefig('report/images/host_mu_comparison.png', dpi=150)
plt.close(fig)

# 5. Calibrator absolute magnitudes
fig, ax = plt.subplots(figsize=(6,4))
M_B_val = baseline['parameters']['M_B']
M_B_err = baseline['errors']['M_B']
M_F_val = baseline['parameters'].get('M_F110W', None)
M_F_err = baseline['errors'].get('M_F110W', None)

labels_mag = []
vals_mag = []
errs_mag = []
if M_B_val is not None:
    labels_mag.append(r'$M_B$ (SNe Ia)')
    vals_mag.append(M_B_val)
    errs_mag.append(M_B_err)
if M_F_val is not None:
    labels_mag.append(r'$M_{\rm F110W}$ (SBF)')
    vals_mag.append(M_F_val)
    errs_mag.append(M_F_err)

x = np.arange(len(labels_mag))
ax.errorbar(x, vals_mag, yerr=errs_mag, fmt='o', capsize=4, color='darkgreen', markersize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels_mag)
ax.set_ylabel('Absolute Magnitude [mag]')
ax.set_title('Calibrator Absolute Magnitudes from GLS Fit')
fig.tight_layout()
fig.savefig('report/images/absolute_magnitudes.png', dpi=150)
plt.close(fig)

# 6. Covariance matrix heatmap of host measurements
fig, ax = plt.subplots(figsize=(8,6))
with open('data/H0DN_MinimalDataset.txt','r') as f:
    exec(f.read())
hm = host_measurements
n_hm = len(hm)
C_hm = np.zeros((n_hm, n_hm))
for i in range(n_hm):
    h_i, method_i, anchor_i, mu_i, err_i = hm[i]
    C_hm[i,i] += err_i**2
    for j in range(i+1, n_hm):
        h_j, method_j, anchor_j, mu_j, err_j = hm[j]
        cov = 0.0
        if anchor_i == anchor_j:
            cov += anchors[anchor_i]['err']**2
        if method_i == method_j and anchor_i == anchor_j:
            key = (method_i, anchor_i)
            if key in method_anchor_err:
                cov += method_anchor_err[key]**2
        C_hm[i,j] += cov
        C_hm[j,i] += cov

labels_hm = [f"{h}_{m}_{a}" for h,m,a,mu,err in hm]
im = ax.imshow(C_hm, cmap='coolwarm', vmin=-0.01, vmax=0.01)
ax.set_xticks(range(n_hm))
ax.set_yticks(range(n_hm))
ax.set_xticklabels(labels_hm, rotation=90, fontsize=6)
ax.set_yticklabels(labels_hm, fontsize=6)
ax.set_title('Host Measurement Covariance Matrix')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig('report/images/covariance_matrix.png', dpi=150)
plt.close(fig)

print("Figures saved to report/images/")
