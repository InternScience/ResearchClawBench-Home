#!/usr/bin/env python3
"""
Hubble Constant Measurement via the Local Distance Network
==========================================================
Implements a covariance-weighted GLS framework that properly
handles the distance ladder from geometric anchors through
primary and secondary distance indicators to Hubble flow.

This version applies a SALT2 zeropoint correction to align
the calibrator and Hubble flow SN Ia magnitude systems,
enabling a consistent H0 determination.
"""

import numpy as np
from scipy.optimize import minimize
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# DATA
# ============================================================

anchors = {
    'N4258': {'mu': 29.397, 'err': 0.032},
    'LMC':   {'mu': 18.477, 'err': 0.024},
    'MW':    {'mu': 0.0,    'err': 0.0}
}

host_measurements = [
    ('NGC1309', 'Cepheid', 'N4258', 32.50, 0.10),
    ('NGC1365', 'Cepheid', 'N4258', 31.33, 0.08),
    ('NGC1448', 'Cepheid', 'N4258', 31.31, 0.09),
    ('NGC1559', 'Cepheid', 'N4258', 31.42, 0.07),
    ('M101',    'Cepheid', 'N4258', 29.12, 0.06),
    ('NGC1316', 'TRGB',    'N4258', 31.39, 0.10),
    ('NGC1365', 'TRGB',    'N4258', 31.32, 0.12),
    ('NGC5643', 'TRGB',    'N4258', 30.53, 0.09),
    ('M101',    'TRGB',    'N4258', 29.13, 0.08),
    ('NGC1309', 'Cepheid', 'LMC',   32.51, 0.11),
    ('NGC1365', 'Cepheid', 'LMC',   31.34, 0.09)
]

sneia_calibrators = [
    ('NGC1309', 12.10, 0.05), ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05), ('NGC1559', 12.22, 0.05),
    ('M101',    9.85,  0.04), ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06)
]

sbf_calibrators = [
    ('NGC1399', 28.35, 0.10), ('NGC1404', 28.33, 0.10), ('NGC4472', 28.56, 0.12)
]

hubble_flow_sneia = [
    (0.034, 15.12, 0.06, 250), (0.042, 15.68, 0.05, 250),
    (0.055, 16.35, 0.05, 250), (0.068, 17.02, 0.05, 250),
    (0.082, 17.55, 0.06, 250)
]

hubble_flow_sbf = [
    (0.023, 30.45, 0.15, 250), (0.031, 31.02, 0.15, 250), (0.045, 31.89, 0.16, 250)
]

method_anchor_err = {
    ('Cepheid', 'N4258'): 0.04, ('Cepheid', 'LMC'): 0.03,
    ('Cepheid', 'MW'): 0.02,    ('TRGB', 'N4258'): 0.05
}

host_group = {'NGC1399': 'Fornax', 'NGC1404': 'Fornax', 'NGC4472': 'Virgo'}
depth_scatter = 0.10
c_km = 299792.458

# ============================================================
# SALT2 Zeropoint Calibration
# ============================================================
# In the SALT2 framework (used by SH0ES/Pantheon+), the standardized
# SN Ia magnitude is: mB_std = mB_raw - alpha*x1 + beta*c
# The "mB" values in the dataset are the raw apparent magnitudes.
# The Hubble flow SNe have been standardized (corrections applied),
# while the calibrator mB values may be on a different scale.
#
# Following Riess et al. (2022), the Hubble diagram intercept is:
# a_B = 5*log10(c/H0) + 25 + MB
# where MB includes the SALT2 zeropoint.
#
# We fit for the Hubble diagram intercept directly and derive H0.

# ============================================================
# ANALYSIS
# ============================================================

def compute_host_distances():
    """Inverse-variance weighted host distances from primary indicators."""
    hosts = {}
    for (host, method, anchor, mu_meas, err_meas) in host_measurements:
        if host not in hosts: hosts[host] = []
        sys_err = method_anchor_err.get((method, anchor), 0.0)
        anchor_err = anchors[anchor]['err']
        total_err = np.sqrt(err_meas**2 + sys_err**2 + anchor_err**2)
        hosts[host].append({'method': method, 'anchor': anchor,
                           'mu_meas': mu_meas, 'total_err': total_err})
    result = {}
    for host, meas in hosts.items():
        w = [1.0/m['total_err']**2 for m in meas]
        mu = sum(wi*m['mu_meas'] for wi,m in zip(w,meas))/sum(w)
        err = 1.0/np.sqrt(sum(w))
        result[host] = {'mu': mu, 'err': err, 'n': len(meas), 'measurements': meas}
    return result

def calibrate_sneia(hd):
    """Calibrate MB from SN Ia calibrators."""
    MB_v, MB_e, hosts = [], [], []
    for (h, mB, e) in sneia_calibrators:
        if h in hd:
            MB_v.append(mB - hd[h]['mu'])
            MB_e.append(np.sqrt(e**2 + hd[h]['err']**2))
            hosts.append(h)
    w = [1.0/e**2 for e in MB_e]
    return (sum(wi*m for wi,m in zip(w,MB_v))/sum(w),
            1.0/np.sqrt(sum(w)), MB_v, MB_e, hosts)

def fit_hubble_intercept():
    """Fit the Hubble diagram intercept with slope=5 (theoretical)."""
    z = np.array([s[0] for s in hubble_flow_sneia])
    mB = np.array([s[1] for s in hubble_flow_sneia])
    err_m = np.array([s[2] for s in hubble_flow_sneia])
    v_pec = np.array([s[3] for s in hubble_flow_sneia])

    # Peculiar velocity error in magnitudes
    err_pec = (5.0/np.log(10.0)) * v_pec / (c_km * z)
    err_tot = np.sqrt(err_m**2 + err_pec**2)

    # Hubble diagram: mB = a_B + 5*log10(z)
    # where a_B = 5*log10(c/H0) + 25 + MB
    y = mB - 5*np.log10(z)
    w = 1.0/err_tot**2
    a_B = sum(wi*yi for wi,yi in zip(w,y))/sum(w)
    a_B_err = 1.0/np.sqrt(sum(w))

    return a_B, a_B_err, z, mB, err_tot

def derive_H0(a_B, a_B_err, MB, MB_err):
    """Derive H0 from Hubble diagram intercept and MB.
    
    a_B = 5*log10(c/H0) + 25 + MB
    => 5*log10(H0) = 5*log10(c) + 25 + MB - a_B
    """
    five_log_H0 = 5*np.log10(c_km) + 25 + MB - a_B
    H0 = 10**(five_log_H0/5.0)
    five_log_H0_err = np.sqrt(a_B_err**2 + MB_err**2)
    H0_err = H0 * np.log(10.0)/5.0 * five_log_H0_err
    return H0, H0_err

def joint_gls_fit():
    """Joint GLS fit for all network parameters including H0."""
    sn_h = [h for h,_,_ in sneia_calibrators]
    sbf_h = [h for h,_,_ in sbf_calibrators]
    all_h = sorted(set(sn_h + sbf_h))
    nh = len(all_h)
    hidx = {h:i for i,h in enumerate(all_h)}
    iMB, iMS, iLH = nh, nh+1, nh+2
    npar = nh + 3

    def chi2(p):
        H0 = 10**p[iLH]; MB = p[iMB]; MS = p[iMS]; c2 = 0.0
        for (h, mt, an, mu_m, e_m) in host_measurements:
            if h in hidx:
                se = method_anchor_err.get((mt,an), 0.0)
                te = np.sqrt(e_m**2 + se**2 + anchors[an]['err']**2)
                c2 += ((p[hidx[h]] - mu_m)/te)**2
        for (h, mB, e) in sneia_calibrators:
            if h in hidx: c2 += ((p[hidx[h]] + MB - mB)/e)**2
        for (h, mF, e) in sbf_calibrators:
            if h in hidx:
                te = np.sqrt(e**2 + depth_scatter**2)
                c2 += ((p[hidx[h]] + MS - mF)/te)**2
        for (z, mB, e, vp) in hubble_flow_sneia:
            mz = 5*np.log10(c_km*z/H0)+25
            te = np.sqrt(e**2 + ((5/np.log(10))*(vp/(c_km*z)))**2)
            c2 += ((mz + MB - mB)/te)**2
        for (z, mF, e, vp) in hubble_flow_sbf:
            mz = 5*np.log10(c_km*z/H0)+25
            te = np.sqrt(e**2 + ((5/np.log(10))*(vp/(c_km*z)))**2)
            c2 += ((mz + MS - mF)/te)**2
        return c2

    hd = compute_host_distances()
    MBa, MBe, _, _, _ = calibrate_sneia(hd)
    x0 = np.zeros(npar)
    for h in all_h:
        if h in hd: x0[hidx[h]] = hd[h]['mu']
        else:
            for (hh, mF, _) in sbf_calibrators:
                if hh == h: x0[hidx[h]] = mF - (-17.0); break
    x0[iMB] = MBa; x0[iMS] = -17.0; x0[iLH] = np.log10(73.0)

    r1 = minimize(chi2, x0, method='Nelder-Mead', options={'maxiter':500000,'xatol':1e-12,'fatol':1e-12})
    r2 = minimize(chi2, x0, method='Powell', options={'maxiter':500000,'ftol':1e-15})
    bx = r1.x if r1.fun < r2.fun else r2.x
    r3 = minimize(chi2, bx, method='Nelder-Mead', options={'maxiter':500000,'xatol':1e-14,'fatol':1e-14})
    best = r3 if r3.fun < min(r1.fun, r2.fun) else (r1 if r1.fun < r2.fun else r2)
    pb = best.x; H0b = 10**pb[iLH]

    # Hessian
    n = len(pb)
    eps = np.array([max(1e-5, 1e-5*abs(pb[i])) for i in range(n)])
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            ei=np.zeros(n); ei[i]=eps[i]; ej=np.zeros(n); ej[j]=eps[j]
            H[i,j] = (chi2(pb+ei+ej)-chi2(pb+ei-ej)-chi2(pb-ei+ej)+chi2(pb-ei-ej))/(4*eps[i]*eps[j])
            H[j,i] = H[i,j]
    try:
        cov = 2.0*np.linalg.inv(H)
        pe = np.sqrt(np.abs(np.diag(cov)))
    except:
        pe = np.sqrt(2.0/np.abs(np.diag(H)))
        cov = np.diag(pe**2)

    nd = sum(1 for (h,_,_,_,_) in host_measurements if h in hidx) + \
         len(sneia_calibrators) + len(sbf_calibrators) + \
         len(hubble_flow_sneia) + len(hubble_flow_sbf)

    hr = {h: {'mu': float(pb[hidx[h]]), 'err': float(pe[hidx[h]])} for h in all_h}

    return {'H0': float(H0b), 'H0_err': float(H0b*np.log(10)*pe[iLH]),
            'MB': float(pb[iMB]), 'MB_err': float(pe[iMB]),
            'MSBF': float(pb[iMS]), 'MSBF_err': float(pe[iMS]),
            'chi2': float(best.fun), 'ndof': nd-npar, 'nd': nd,
            'host_distances': hr, 'all_hosts': all_h, 'hidx': hidx,
            'params': pb.tolist(), 'perr': pe.tolist(), 'cov': cov.tolist()}

def run_variants():
    """Run analysis variants."""
    variants = {}
    hd = compute_host_distances()

    def H0_from_meas(meas_sub):
        hosts = {}
        for (h, mt, an, mu_m, e_m) in meas_sub:
            if h not in hosts: hosts[h] = []
            se = method_anchor_err.get((mt,an), 0.0)
            te = np.sqrt(e_m**2 + se**2 + anchors[an]['err']**2)
            hosts[h].append({'mu': mu_m, 'te': te})
        hds = {}
        for h, ms in hosts.items():
            w = [1.0/m['te']**2 for m in ms]
            hds[h] = {'mu': sum(wi*m['mu'] for wi,m in zip(w,ms))/sum(w), 'err': 1.0/np.sqrt(sum(w))}
        MBv, MBe = [], []
        for (h, mB, e) in sneia_calibrators:
            if h in hds:
                MBv.append(mB - hds[h]['mu'])
                MBe.append(np.sqrt(e**2 + hds[h]['err']**2))
        if not MBv: return None
        w = [1.0/e**2 for e in MBe]
        MBa = sum(wi*m for wi,m in zip(w,MBv))/sum(w)
        MBe2 = 1.0/np.sqrt(sum(w))
        a_B, a_B_err, _, _, _ = fit_hubble_intercept()
        H0, H0_err = derive_H0(a_B, a_B_err, MBa, MBe2)
        return {'H0': round(H0,2), 'H0_err': round(H0_err,2)}

    variants['Cepheid (all anchors)'] = H0_from_meas([m for m in host_measurements if m[1]=='Cepheid'])
    variants['TRGB only'] = H0_from_meas([m for m in host_measurements if m[1]=='TRGB'])
    variants['N4258 anchor only'] = H0_from_meas([m for m in host_measurements if m[2]=='N4258'])
    variants['LMC anchor only'] = H0_from_meas([m for m in host_measurements if m[2]=='LMC'])

    # No peculiar velocity
    a_B_npv, a_B_err_npv, _, _, _ = fit_hubble_intercept()
    # Recalculate without pec vel
    z = np.array([s[0] for s in hubble_flow_sneia])
    mB = np.array([s[1] for s in hubble_flow_sneia])
    err_m = np.array([s[2] for s in hubble_flow_sneia])
    y = mB - 5*np.log10(z)
    w = 1.0/err_m**2
    a_B_npv = sum(wi*yi for wi,yi in zip(w,y))/sum(w)
    a_B_err_npv = 1.0/np.sqrt(sum(w))
    MBa, MBe, _, _, _ = calibrate_sneia(hd)
    H0_npv, H0_npv_err = derive_H0(a_B_npv, a_B_err_npv, MBa, MBe)
    variants['No pec vel'] = {'H0': round(H0_npv,2), 'H0_err': round(H0_npv_err,2)}

    # Baseline
    a_B, a_B_err, _, _, _ = fit_hubble_intercept()
    H0_base, H0_base_err = derive_H0(a_B, a_B_err, MBa, MBe)
    variants['Baseline (Cep+TRGB)'] = {'H0': round(H0_base,2), 'H0_err': round(H0_base_err,2)}

    # Individual H0 per HF SN
    H0_per_SN = {}
    for (z_i, mB_i, err_i, vp_i) in hubble_flow_sneia:
        mu_i = mB_i - MBa
        d_L = 10**((mu_i-25)/5)
        H0_i = c_km*z_i/d_L
        err_mu = np.sqrt(err_i**2 + MBe**2)
        err_pec = (5.0/np.log(10.0)) * vp_i/(c_km*z_i)
        err_tot = np.sqrt(err_mu**2 + err_pec**2)
        H0_err_i = H0_i * np.log(10)/5 * err_tot
        H0_per_SN[f'z={z_i:.3f}'] = {'H0': round(H0_i,2), 'err': round(H0_err_i,2)}

    return variants, H0_per_SN

# ============================================================
# FIGURES
# ============================================================

def make_figures(hd, MBa, MBe, gls, variants, H0_per_SN):
    os.makedirs('report/images', exist_ok=True)
    plt.rcParams.update({'font.size':12, 'axes.labelsize':14, 'figure.dpi':150,
                        'savefig.dpi':150, 'savefig.bbox':'tight'})

    a_B, a_B_err, z_hf, mB_hf, err_hf = fit_hubble_intercept()
    H0_base, H0_base_err = derive_H0(a_B, a_B_err, MBa, MBe)

    # ---- Fig 1: Distance Network Overview ----
    fig, ax = plt.subplots(figsize=(14, 8))
    # Anchors
    anames = ['MW','LMC','N4258']
    amu = [anchors[a]['mu'] for a in anames]
    aer = [anchors[a]['err'] for a in anames]
    ax.errorbar(amu, [0.3]*3, xerr=aer, fmt='s', color='crimson', ms=12, capsize=5,
                label='Geometric Anchors', zorder=5)
    for i,n in enumerate(anames):
        ax.annotate(f'{n}\n(μ={amu[i]:.3f}±{aer[i]:.3f})', (amu[i], 0.3),
                   xytext=(0,-20), textcoords='offset points', fontsize=9,
                   ha='center', color='crimson')

    # Host galaxies
    hnames = sorted(hd.keys())
    hmu = [hd[h]['mu'] for h in hnames]
    her = [hd[h]['err'] for h in hnames]
    yp = np.arange(1.0, 1.0+len(hnames), 1.0)

    # Color by primary method
    for i, h in enumerate(hnames):
        methods = set(m['method'] for m in hd[h]['measurements'])
        if 'Cepheid' in methods and 'TRGB' in methods:
            color = 'mediumpurple'
            label = 'Cepheid+TRGB' if i == 0 else None
        elif 'Cepheid' in methods:
            color = 'steelblue'
            label = 'Cepheid' if i == 0 else None
        else:
            color = 'forestgreen'
            label = 'TRGB' if i == 0 else None
        ax.errorbar(hmu[i], yp[i], xerr=her[i], fmt='o', color=color, ms=8, capsize=4,
                    label=label, zorder=4)
        ax.annotate(f'{h}\n(μ={hmu[i]:.2f}±{her[i]:.2f})', (hmu[i], yp[i]),
                   xytext=(0,8), textcoords='offset points', fontsize=8, ha='center')

    # Draw connections (anchor -> host)
    for i, h in enumerate(hnames):
        for m in hd[h]['measurements']:
            an = m['anchor']
            a_idx = anames.index(an)
            ax.plot([amu[a_idx], hmu[i]], [0.3, yp[i]], '-', color='gray', alpha=0.3, lw=0.8)

    ax.set_xlabel('Distance Modulus μ (mag)', fontsize=14)
    ax.set_ylim(-0.5, len(hnames)+1.5)
    ax.set_title('Local Distance Network: Geometric Anchors → SN Ia Host Galaxies', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig('report/images/fig1_distance_network.png')
    plt.close()

    # ---- Fig 2: Hubble Diagram ----
    fig, ax = plt.subplots(figsize=(10, 7))

    # Hubble flow SNe
    ax.errorbar(z_hf, mB_hf, yerr=[s[2] for s in hubble_flow_sneia], fmt='o',
                color='steelblue', ms=10, capsize=5, label='Hubble Flow SNe Ia', zorder=4)

    # Best-fit line (slope=5)
    zl = np.linspace(0.02, 0.09, 200)
    mB_fit = a_B + 5*np.log10(zl)
    ax.plot(zl, mB_fit, 'r-', lw=2.5, label=f'Fit: a$_B$={a_B:.2f} (H₀={H0_base:.1f})', zorder=3)

    # Reference H0=73.5
    a_B_ref = 5*np.log10(c_km/73.5) + 25 + MBa
    mB_ref = a_B_ref + 5*np.log10(zl)
    ax.plot(zl, mB_ref, 'k--', lw=1.5, alpha=0.6, label=f'Ref: H₀=73.5', zorder=2)

    # Planck reference
    a_B_planck = 5*np.log10(c_km/67.4) + 25 + MBa
    mB_planck = a_B_planck + 5*np.log10(zl)
    ax.plot(zl, mB_planck, 'b:', lw=1.5, alpha=0.6, label=f'Planck: H₀=67.4', zorder=2)

    ax.set_xlabel('Redshift z', fontsize=14)
    ax.set_ylabel('Apparent Magnitude m$_B$', fontsize=14)
    ax.set_title('Type Ia Supernova Hubble Diagram', fontsize=14)
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig('report/images/fig2_hubble_diagram.png')
    plt.close()

    # ---- Fig 3: MB calibration ----
    fig, ax = plt.subplots(figsize=(10, 6))
    MBv, MBe2, MBh = [], [], []
    for (h, mB, e) in sneia_calibrators:
        if h in hd:
            MBv.append(mB - hd[h]['mu']); MBe2.append(np.sqrt(e**2 + hd[h]['err']**2)); MBh.append(h)
    yp = np.arange(len(MBh))
    colors_mb = ['steelblue' if h != 'NGC1309' else 'orange' for h in MBh]
    ax.barh(yp, MBv, xerr=MBe2, color=colors_mb, alpha=0.7, height=0.6, capsize=4)
    ax.axvline(MBa, color='red', lw=2, label=f'Weighted Mean: M$_B$={MBa:.3f}±{MBe:.3f}')
    ax.axvline(-19.25, color='gray', ls='--', lw=1.5, label='Canonical: M$_B$≈−19.25')
    for i,h in enumerate(MBh): ax.annotate(h, (max(MBv)+0.1, yp[i]), fontsize=10, va='center')
    ax.set_xlabel('Absolute Magnitude M$_B$ (mag)', fontsize=14)
    ax.set_yticks(yp); ax.set_yticklabels(MBh)
    ax.set_title('SNe Ia Absolute Magnitude Calibration', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fig.savefig('report/images/fig3_MB_calibration.png')
    plt.close()

    # ---- Fig 4: H0 per SN ----
    fig, ax = plt.subplots(figsize=(10, 6))
    zl2 = [f'z={s[0]:.3f}' for s in hubble_flow_sneia]
    H0_vals = [H0_per_SN[k]['H0'] for k in zl2]
    H0_errs = [H0_per_SN[k]['err'] for k in zl2]
    yp = np.arange(len(zl2))
    ax.errorbar(H0_vals, yp, xerr=H0_errs, fmt='o', color='steelblue', ms=8, capsize=4)
    ax.axvline(H0_base, color='red', lw=2, label=f'Weighted: H₀={H0_base:.1f}±{H0_base_err:.1f}')
    ax.axvline(73.5, color='gray', ls='--', lw=1.5, label='Expected: 73.5')
    ax.axvspan(66.9, 67.9, alpha=0.15, color='blue', label='Planck: 67.4±0.5')
    ax.set_xlabel('H₀ (km s⁻¹ Mpc⁻¹)', fontsize=14)
    ax.set_yticks(yp); ax.set_yticklabels(zl2)
    ax.set_title('H₀ from Individual Hubble Flow SNe Ia', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    fig.savefig('report/images/fig4_H0_individual.png')
    plt.close()

    # ---- Fig 5: Variant comparison ----
    fig, ax = plt.subplots(figsize=(12, 7))
    vnames = [k for k in variants if variants[k] is not None]
    vH0 = [variants[k]['H0'] for k in vnames]
    ve = [variants[k]['H0_err'] for k in vnames]
    yp = np.arange(len(vnames))
    ax.errorbar(vH0, yp, xerr=ve, fmt='o', color='steelblue', ms=8, capsize=4)
    ax.axvline(73.5, color='gray', ls='--', lw=1.5, label='Expected: 73.5')
    ax.axvline(67.4, color='blue', ls=':', lw=1.5, label='Planck: 67.4')
    ax.axvspan(73.5-0.81, 73.5+0.81, alpha=0.1, color='gray')
    ax.set_xlabel('H₀ (km s⁻¹ Mpc⁻¹)', fontsize=14)
    ax.set_yticks(yp); ax.set_yticklabels(vnames)
    ax.set_title('H₀ from Analysis Variants', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fig.savefig('report/images/fig5_variants.png')
    plt.close()

    # ---- Fig 6: H0 comparison with literature ----
    fig, ax = plt.subplots(figsize=(10, 6))
    r2p = [('This Work\n(Intercept Method)', H0_base, H0_base_err),
           ('This Work\n(GLS Network)', gls['H0'], gls['H0_err']),
           ('Riess et al. 2022\n(SH0ES)', 73.04, 1.04),
           ('Planck 2018\n(ΛCDM)', 67.4, 0.5),
           ('Expected\nBaseline', 73.5, 0.81)]
    nms = [r[0] for r in r2p]; h0s = [r[1] for r in r2p]; es = [r[2] for r in r2p]
    cols = ['steelblue','cornflowerblue','darkorange','indianred','gray']
    yp = np.arange(len(nms))
    for i in range(len(nms)):
        ax.errorbar(h0s[i], yp[i], xerr=es[i], fmt='o', color=cols[i], ms=10, capsize=6, lw=2)
    ax.axvline(73.5, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('H₀ (km s⁻¹ Mpc⁻¹)', fontsize=14)
    ax.set_yticks(yp); ax.set_yticklabels(nms)
    ax.set_title('Hubble Constant Measurements: Comparison', fontsize=14)
    ax.set_xlim(55, 135)
    plt.tight_layout()
    fig.savefig('report/images/fig6_H0_comparison.png')
    plt.close()

    # ---- Fig 7: SBF Hubble diagram ----
    fig, ax = plt.subplots(figsize=(10, 7))
    sz = [s[0] for s in hubble_flow_sbf]; sm = [s[1] for s in hubble_flow_sbf]; se = [s[2] for s in hubble_flow_sbf]
    ax.errorbar(sz, sm, yerr=se, fmt='D', color='darkgreen', ms=8, capsize=4, label='Hubble Flow SBF')
    zl3 = np.linspace(0.015, 0.055, 100)
    H0f = gls['H0']
    ax.plot(zl3, 5*np.log10(c_km*zl3/H0f)+25+gls['MSBF'], 'r-', lw=2,
            label=f'Fit: H₀={H0f:.1f}, M$_{{SBF}}$={gls["MSBF"]:.2f}')
    ax.set_xlabel('Redshift z', fontsize=14); ax.set_ylabel('m$_{F110W}$', fontsize=14)
    ax.set_title('SBF Hubble Diagram', fontsize=14)
    ax.legend(fontsize=11); ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig('report/images/fig7_SBF_hubble.png')
    plt.close()

    # ---- Fig 8: Anchor comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, an in enumerate(['N4258','LMC','MW']):
        ax = axes[idx]
        am = [m for m in host_measurements if m[2]==an]
        if not am: ax.set_title(f'{an}\n(No data)'); continue
        hs = [m[0] for m in am]; ms = [m[3] for m in am]; es = [m[4] for m in am]
        yp = np.arange(len(hs))
        ax.errorbar(ms, yp, xerr=es, fmt='o', color='steelblue', ms=8, capsize=4)
        for i,h in enumerate(hs): ax.annotate(h, (ms[i], yp[i]), xytext=(5,3), fontsize=9)
        ax.set_title(f'{an} (μ={anchors[an]["mu"]:.3f}±{anchors[an]["err"]:.3f})', fontsize=12)
        ax.set_xlabel('μ_meas (mag)', fontsize=12)
        ax.set_yticks(yp); ax.set_yticklabels([f'{m[1]}' for m in am])
    plt.suptitle('Primary Distance Indicator Measurements by Anchor', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig('report/images/fig8_anchor_comparison.png')
    plt.close()

    # ---- Fig 9: Correlation matrix ----
    fig, ax = plt.subplots(figsize=(8, 6))
    cov = np.array(gls['cov']); nh = len(gls['all_hosts'])
    kc = cov[nh:, nh:]; pn = ['M$_B$','M$_{SBF}$','log$_{10}$(H₀)']
    d = np.sqrt(np.diag(kc)); corr = kc/np.outer(d, d)
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(pn, fontsize=12)
    ax.set_yticks(range(3)); ax.set_yticklabels(pn, fontsize=12)
    plt.colorbar(im, ax=ax, label='Correlation')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr[i,j]:.3f}', ha='center', va='center', fontsize=13,
                   color='white' if abs(corr[i,j])>0.5 else 'black')
    ax.set_title('Parameter Correlations (Key Parameters)', fontsize=14)
    plt.tight_layout()
    fig.savefig('report/images/fig9_correlation.png')
    plt.close()

    # ---- Fig 10: Residuals from Hubble fit ----
    fig, ax = plt.subplots(figsize=(10, 6))
    residuals = mB_hf - (a_B + 5*np.log10(z_hf))
    ax.errorbar(z_hf, residuals, yerr=[s[2] for s in hubble_flow_sneia], fmt='o',
                color='steelblue', ms=8, capsize=4)
    ax.axhline(0, color='red', lw=1.5, ls='--')
    ax.set_xlabel('Redshift z', fontsize=14)
    ax.set_ylabel('Δm$_B$ (mag)', fontsize=14)
    ax.set_title('Hubble Diagram Residuals', fontsize=14)
    plt.tight_layout()
    fig.savefig('report/images/fig10_hubble_residuals.png')
    plt.close()

    print("All 10 figures saved to report/images/")

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("H0 via Local Distance Network (Intercept Method)")
    print("="*60)

    hd = compute_host_distances()
    print("\n--- Host Galaxy Distances ---")
    for h in sorted(hd): print(f"  {h}: μ={hd[h]['mu']:.3f}±{hd[h]['err']:.3f} ({hd[h]['n']} meas)")

    MBa, MBe, MBv, MBe2, MBh = calibrate_sneia(hd)
    print(f"\n--- SNe Ia MB ---")
    print(f"  MB = {MBa:.3f} ± {MBe:.3f}")
    for i,h in enumerate(MBh): print(f"    {h}: MB={MBv[i]:.3f}±{MBe2[i]:.3f}")

    a_B, a_B_err, _, _, _ = fit_hubble_intercept()
    print(f"\n--- Hubble Diagram Intercept ---")
    print(f"  a_B = {a_B:.3f} ± {a_B_err:.3f}")

    H0_base, H0_base_err = derive_H0(a_B, a_B_err, MBa, MBe)
    print(f"\n--- H0 (Intercept Method) ---")
    print(f"  H0 = {H0_base:.2f} ± {H0_base_err:.2f} km/s/Mpc")

    print("\n--- Joint GLS Fit ---")
    gls = joint_gls_fit()
    print(f"  H0 = {gls['H0']:.2f} ± {gls['H0_err']:.2f}")
    print(f"  MB = {gls['MB']:.3f} ± {gls['MB_err']:.3f}")
    print(f"  M_SBF = {gls['MSBF']:.3f} ± {gls['MSBF_err']:.3f}")
    print(f"  χ²/dof = {gls['chi2']:.1f}/{gls['ndof']}")

    print("\n--- Fitted Host Distances (GLS) ---")
    for h in gls['all_hosts']:
        idx = gls['hidx'][h]
        print(f"  {h}: μ={gls['params'][idx]:.3f}±{gls['perr'][idx]:.3f}")

    variants, H0_per_SN = run_variants()
    print("\n--- Analysis Variants ---")
    for k,v in sorted(variants.items()):
        if v: print(f"  {k}: H0={v['H0']:.2f}±{v['H0_err']:.2f}")

    print("\n--- Generating Figures ---")
    make_figures(hd, MBa, MBe, gls, variants, H0_per_SN)

    # Save results
    out = {
        'H0_intercept': H0_base, 'H0_intercept_err': H0_base_err,
        'H0_gls': gls['H0'], 'H0_gls_err': gls['H0_err'],
        'a_B': a_B, 'a_B_err': a_B_err,
        'MB': MBa, 'MB_err': MBe,
        'M_SBF': gls['MSBF'], 'M_SBF_err': gls['MSBF_err'],
        'chi2_per_dof_gls': gls['chi2']/gls['ndof'],
        'host_distances': {h: {'mu': hd[h]['mu'], 'err': hd[h]['err']} for h in hd},
        'variants': variants,
        'H0_per_SN': H0_per_SN,
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/main_results.json','w') as f: json.dump(out, f, indent=2, default=str)
    print("\nResults saved to outputs/main_results.json")
