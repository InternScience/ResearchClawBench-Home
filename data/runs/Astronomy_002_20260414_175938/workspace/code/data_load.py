import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data via exec
data_file = Path('../data/H0DN_MinimalDataset.txt')
with open(data_file) as f:
    ns = {}
    exec(f.read(), ns)
data = ns

# Save raw data
with open('../outputs/raw_data.pkl', 'wb') as f:
    pickle.dump(data, f)

# Extract
anchors = data['anchors']
host_meas = np.array(data['host_measurements'])
snia_cal = np.array(data['sneia_calibrators'])
sbf_cal = np.array(data['sbf_calibrators'])
flow_snia = np.array(data['hubble_flow_sneia'])
flow_sbf = np.array(data['hubble_flow_sbf'])
method_anchor_sys = data['method_anchor_err']
host_group = data['host_group']
depth_scatter = data['depth_scatter']
c_km = data['c_km']

# To float
host_meas[:, 3:] = host_meas[:, 3:].astype(float)
snia_cal[:, 1:] = snia_cal[:, 1:].astype(float)
sbf_cal[:, 1:] = sbf_cal[:, 1:].astype(float)
flow_snia[:, 1:] = flow_snia[:, 1:].astype(float)
flow_sbf[:, 1:] = flow_sbf[:, 1:].astype(float)

structured = {
    'anchors': anchors,
    'host_measurements': host_meas,
    'sneia_calibrators': snia_cal,
    'sbf_calibrators': sbf_cal,
    'hubble_flow_sneia': flow_snia,
    'hubble_flow_sbf': flow_sbf,
    'method_anchor_err': method_anchor_sys,
    'host_group': host_group,
    'depth_scatter': depth_scatter,
    'c_km': c_km
}
with open('../outputs/structured_data.pkl', 'wb') as f:
    pickle.dump(structured, f)

print('Data loaded successfully.')
print('Unique primary hosts:', np.unique(host_meas[:,0]))
print('SNe Ia cal hosts:', np.unique(snia_cal[:,0]))
print('SBF cal hosts:', np.unique(sbf_cal[:,0]))
print('Anchor values:')
for a, vals in anchors.items():
    print('  %s: %.3f +/- %.3f' % (a, vals['mu'], vals['err']))

# Plots 1
fig, axs = plt.subplots(2, 2, figsize=(12,10))
axs = axs.flatten()

# Anchors
a_names = list(anchors)
a_mu = np.array([anchors[a]['mu'] for a in a_names])
a_err = np.array([anchors[a]['err'] for a in a_names])
axs[0].errorbar(range(len(a_names)), a_mu, yerr=a_err, fmt='o')
axs[0].set_xticks(range(len(a_names)))
axs[0].set_xticklabels(a_names)
axs[0].set_title('Geometric Anchors')
axs[0].set_ylabel(r'$\mu$')

# Host primary meas
methods = host_meas[:,1]
anch_h = host_meas[:,2]
mu_h = host_meas[:,3]
err_h = host_meas[:,4]
for m in np.unique(methods):
    idx = methods == m
    col = 'blue' if m=='Cepheid' else 'orange'
    axs[1].errorbar(range(np.sum(idx)), mu_h[idx], yerr=err_h[idx], fmt='o', label=m, color=col)
axs[1].legend()
axs[1].set_title('Host primary $\mu$')
axs[1].set_xlabel('measurements')
axs[1].set_ylabel(r'$\mu_\mathrm{meas}$')

# SN cal
mB = snia_cal[:,1]
errB = snia_cal[:,2]
hosts_sn = snia_cal[:,0]
axs[2].errorbar(range(len(mB)), mB, yerr=errB, fmt='s', color='red')
for i,h in enumerate(hosts_sn):
    axs[2].annotate(h[:4], (i, mB[i]), xytext=(5,5), textcoords='offset points')
axs[2].set_title('SNe Ia calibrators')
axs[2].set_xlabel('hosts')
axs[2].set_ylabel('m_B')
axs[2].set_xticks(range(len(mB)))
axs[2].set_xticklabels([h.replace('NGC','N').replace('M101','M1') for h in hosts_sn], rotation=45)

# Hubble flow SNe Ia
z_sn = flow_snia[:,0]
mB_sn = flow_snia[:,1]
err_sn = flow_snia[:,2]
axs[3].errorbar(z_sn, mB_sn, yerr=err_sn, fmt='^', color='red', label='data')
H0_test = 73.5
mu_test = 5*np.log10(c_km * np.array(z_sn) / H0_test) + 25
axs[3].plot(z_sn, mu_test + 11.0, 'k--', label=r'$\mu$(H0=73.5, M=11)')
axs[3].legend()
axs[3].set_xlabel('z')
axs[3].set_title('Hubble flow SNe Ia')
axs[3].set_ylabel('m_B')

plt.tight_layout()
plt.savefig('../report/images/overview1.png', dpi=150)
plt.close()

# Plots 2
fig, axs = plt.subplots(1,2, figsize=(12,5))

# SBF cal
mF = sbf_cal[:,1]
errF = sbf_cal[:,2]
hosts_sbf = sbf_cal[:,0]
groups = np.array([host_group[h] for h in hosts_sbf])
axs[0].errorbar(range(len(mF)), mF, yerr=errF, fmt='D', color='green')
for i,h in enumerate(hosts_sbf):
    axs[0].annotate('%s\\n%s' % (h[:6], groups[i]), (i, mF[i]))
axs[0].set_title('SBF calibrators')
axs[0].set_ylabel('m_F110W')
axs[0].set_xticks(range(len(mF)))
axs[0].set_xticklabels(groups, rotation=45)

# Flow SBF
z_sbf = flow_sbf[:,0]
mF_sbf = flow_sbf[:,1]
errF_sbf = flow_sbf[:,2]
axs[1].errorbar(z_sbf, mF_sbf, yerr=errF_sbf, fmt='v', color='green', label='data')
mu_test_sbf = 5*np.log10(c_km * np.array(z_sbf) / H0_test) + 25
axs[1].plot(z_sbf, mu_test_sbf + 28.4, 'k--', label=r'$\mu$(H0=73.5, M=28.4)')
axs[1].legend()
axs[1].set_xlabel('z')
axs[1].set_title('Hubble flow SBF')
axs[1].set_ylabel('m_F110W')

plt.tight_layout()
plt.savefig('../report/images/overview2.png', dpi=150)
plt.close()

print('Overview plots saved to report/images/')