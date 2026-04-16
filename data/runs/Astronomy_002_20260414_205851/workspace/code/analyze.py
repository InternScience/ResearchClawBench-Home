#!/usr/bin/env python3
"""
Hubble Constant Measurement via the Local Distance Network
Complete analysis including GLS network fit, variant analyses, and figure generation.
"""

import numpy as np
from scipy.optimize import minimize
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# ANALYSIS FUNCTIONS
# ============================================================

def compute_host_distances():
    hosts = {}
    for (host, method, anchor, mu_meas, err_meas) in host_measurements:
        if host not in hosts: hosts[host] = []
        sys_err = method_anchor_err.get((method, anchor), 0.0)
        anchor_err = anchors[anchor]['err']
        total_err = np.sqrt(err_meas**2 + sys_err**2 + anchor_err**2)
        hosts[host].append({'method': method, 'anchor': anchor, 'mu_meas': mu_meas, 'total_err': total_err})
    result = {}
    for host, meas in hosts.items():
        w = [1.0/m['total_err']**2 for m in meas]
        mu = sum(wi*m['mu_meas'] for wi,m in zip(w,meas))/sum(w)
        err = 1.0/np.sqrt(sum(w))
        result[host] = {'mu': mu, 'err': err, 'n': len(meas), 'measurements': meas}
    return result

def calibrate_sneia(hd):
    MB_v, MB_e, hosts = [], [], []
    for (h, mB, e) in sneia_calibrators:
        if h in hd:
            MB_v.append(mB - hd[h]['mu'])
            MB_e.append(np.sqrt(e**2 + hd[h]['err']**2))
            hosts.append(h)
    w = [1.0/e**2 for e in MB_e]
    return sum(wi*m for wi,m in zip(w,MB_v))/sum(w), 1.0/np.sqrt(sum(w)), MB_v, MB_e, hosts

def compute_H0_HF(MB, MB_err, pec=True):
    H0s, errs = [], []
    for (z, mB, e, vp) in hubble_flow_sneia:
        mu = mB - MB
        em = np.sqrt(e**2 + MB_err**2)
        if pec: em = np.sqrt(em**2 + ((5.0/np.log(10.0))*(vp/(c_km*z)))**2)
        d = 10**((mu-25)/5)
        H0s.append(c_km*z/d)
        errs.append(c_km*z/d * np.log(10)/5 * em)
    w = [1.0/e**2 for e in errs]
    return sum(wi*h for wi,h in zip(w,H0s))/sum(w), 1.0/np.sqrt(sum(w)), H0s, errs

def joint_gls_fit():
    sn_h = [h for h,_,_ in sneia_calibrators]
    sbf_h = [h for h,_,_ in sbf_calibrators]
    all_h = sorted(set(sn_h + sbf_h))
    nh = len(all_h)
    hidx = {h:i for i,h in enumerate(all_h)}
    iMB, iMS, iLH = nh, nh+1, nh+2
    np_ = nh + 3

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
    MB_a, MB_e, _, _, _ = calibrate_sneia(hd)
    x0 = np.zeros(np_)
    for h in all_h:
        if h in hd: x0[hidx[h]] = hd[h]['mu']
        else:
            for (hh, mF, _) in sbf_calibrators:
                if hh == h: x0[hidx[h]] = mF - (-17.0); break
    x0[iMB] = MB_a; x0[iMS] = -17.0; x0[iLH] = np.log10(73.0)

    r1 = minimize(chi2, x0, method='Nelder-Mead', options={'maxiter':500000,'xatol':1e-12,'fatol':1e-12})
    r2 = minimize(chi2, x0, method='Powell', options={'maxiter':500000,'ftol':1e-15})
    bx = r1.x if r1.fun < r2.fun else r2.x
    r3 = minimize(chi2, bx, method='Nelder-Mead', options={'maxiter':500000,'xatol':1e-14,'fatol':1e-14})
    best = r3 if r3.fun < min(r1.fun, r2.fun) else (r1 if r1.fun < r2.fun else r2)
    pb = best.x
    H0b = 10**pb[iLH]

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

    nd = sum(1 for (h,_,_,_,_) in host_measurements if h in hidx) + len(sneia_calibrators) + len(sbf_calibrators) + len(hubble_flow_sneia) + len(hubble_flow_sbf)
    hr = {h: {'mu': float(pb[hidx[h]]), 'err': float(pe[hidx[h]])} for h in all_h}

    return {'H0': float(H0b), 'H0_err': float(H0b*np.log(10)*pe[iLH]),
            'MB': float(pb[iMB]), 'MB_err': float(pe[iMB]),
            'MSBF': float(pb[iMS]), 'MSBF_err': float(pe[iMS]),
            'chi2': float(best.fun), 'ndof': nd-np_, 'nd': nd,
            'host_distances': hr, 'all_hosts': all_h, 'hidx': hidx,
            'params': pb.tolist(), 'perr': pe.tolist(), 'cov': cov.tolist()}

def run_variants():
    variants = {}
    hd = compute_host_distances()
    def H0_sub(meas_sub):
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
        h0, h0e, _, _ = compute_H0_HF(MBa, MBe2)
        return {'H0': round(h0,2), 'H0_err': round(h0e,2)}

    variants['Cepheid (all anchors)'] = H0_sub([m for m in host_measurements if m[1]=='Cepheid'])
    variants['TRGB only'] = H0_sub([m for m in host_measurements if m[1]=='TRGB'])
    variants['N4258 anchor only'] = H0_sub([m for m in host_measurements if m[2]=='N4258'])
    variants['LMC anchor only'] = H0_sub([m for m in host_measurements if m[2]=='LMC'])

    MBa, MBe, _, _, _ = calibrate_sneia(hd)
    h0, h0e, _, _ = compute_H0_HF(MBa, MBe, pec=False)
    variants['No pec vel'] = {'H0': round(h0,2), 'H0_err': round(h0e,2)}
    h0, h0e, _, _ = compute_H0_HF(MBa, MBe)
    variants['Baseline (Cep+TRGB)'] = {'H0': round(h0,2), 'H0_err': round(h0e,2)}
    return variants

# ============================================================
# FIGURES
# ============================================================

def make_figures(hd, MBa, MBe, gls, variants):
    os.makedirs('report/images', exist_ok=True)
    plt.rcParams.update({'font.size':12, 'axes.labelsize':14, 'figure.dpi':150, 'savefig.dpi':150, 'savefig.bbox':'tight'})

    # Fig 1: Distance Network
    fig, ax = plt.subplots(figsize=(12, 8))
    anames = ['MW','LMC','N4258']
    amu = [anchors[a]['mu'] for a in anames]
    aer = [anchors[a]['err'] for a in anames]
    ax.errorbar(amu, [0.5]*3, xerr=aer, fmt='s', color='red', ms=10, capsize=5, label='Geometric Anchors', zorder=5)
    hnames = sorted(hd.keys())
    hmu = [hd[h]['mu'] for h in hnames]
    her = [hd[h]['err'] for h in hnames]
    yp = np.arange(1, len(hnames)+1)
    ax.errorbar(hmu, yp, xerr=her, fmt='o', color='steelblue', ms=8, capsize=4, label='SN Ia Hosts', zorder=4)
    for i,h in enumerate(hnames): ax.annotate(h, (hmu[i], yp[i]), xytext=(5,5), fontsize=9)
    for i,n in enumerate(anames): ax.annotate(n, (amu[i], 0.5), xytext=(5,5), fontsize=9, color='red')
    ax.set_xlabel('Distance Modulus μ (mag)')
    ax.set_yticks(list(yp)+[0.5]); ax.set_yticklabels(hnames+['Anchors'])
    ax.set_title('Local Distance Network: Distance Moduli')
    ax.legend(loc='lower right'); ax.invert_yaxis()
    plt.tight_layout(); fig.savefig('report/images/fig1_distance_network.png'); plt.close()

    # Fig 2: Hubble Diagram
    fig, ax = plt.subplots(figsize=(10, 7))
    hz = [s[0] for s in hubble_flow_sneia]
    hm = [s[1] for s in hubble_flow_sneia]
    he = [s[2] for s in hubble_flow_sneia]
    ax.errorbar(hz, hm, yerr=he, fmt='o', color='steelblue', ms=8, capsize=4, label='Hubble Flow SNe Ia')
    zl = np.linspace(0.02, 0.09, 100)
    H0f = gls['H0']; MBf = gls['MB']
    ax.plot(zl, 5*np.log10(c_km*zl/H0f)+25+MBf, 'r-', lw=2, label=f'GLS Fit: H₀={H0f:.1f}')
    ax.plot(zl, 5*np.log10(c_km*zl/73.5)+25+MBf, 'k--', lw=1.5, alpha=0.5, label='Ref: H₀=73.5')
    ax.set_xlabel('Redshift z'); ax.set_ylabel('m$_B$'); ax.set_title('Hubble Diagram: SNe Ia')
    ax.legend(); ax.invert_yaxis()
    plt.tight_layout(); fig.savefig('report/images/fig2_hubble_diagram.png'); plt.close()

    # Fig 3: MB calibration
    fig, ax = plt.subplots(figsize=(10, 6))
    MBv, MBe2, MBh = [], [], []
    for (h, mB, e) in sneia_calibrators:
        if h in hd:
            MBv.append(mB - hd[h]['mu']); MBe2.append(np.sqrt(e**2 + hd[h]['err']**2)); MBh.append(h)
    yp = np.arange(len(MBh))
    ax.errorbar(MBv, yp, xerr=MBe2, fmt='o', color='steelblue', ms=8, capsize=4)
    ax.axvline(MBa, color='red', lw=2, label=f'Weighted Mean: M$_B$={MBa:.3f}±{MBe:.3f}')
    ax.axvline(-19.25, color='gray', ls='--', lw=1.5, label='Literature: M$_B$≈−19.25')
    for i,h in enumerate(MBh): ax.annotate(h, (MBv[i], yp[i]), xytext=(5,3), fontsize=9)
    ax.set_xlabel('M$_B$ (mag)'); ax.set_yticks(yp); ax.set_yticklabels(MBh)
    ax.set_title('SNe Ia Absolute Magnitude Calibration'); ax.legend(loc='lower right')
    plt.tight_layout(); fig.savefig('report/images/fig3_MB_calibration.png'); plt.close()

    # Fig 4: H0 individual
    fig, ax = plt.subplots(figsize=(10, 6))
    h0a, h0e, h0v, h0ev = compute_H0_HF(MBa, MBe)
    zl2 = [f'z={s[0]:.3f}' for s in hubble_flow_sneia]
    yp = np.arange(len(zl2))
    ax.errorbar(h0v, yp, xerr=h0ev, fmt='o', color='steelblue', ms=8, capsize=4)
    ax.axvline(h0a, color='red', lw=2, label=f'Mean: H₀={h0a:.1f}±{h0e:.1f}')
    ax.axvline(73.5, color='gray', ls='--', lw=1.5, label='Expected: 73.5')
    ax.axvspan(66.9, 67.9, alpha=0.15, color='blue', label='Planck: 67.4±0.5')
    ax.set_xlabel('H₀ (km s⁻¹ Mpc⁻¹)'); ax.set_yticks(yp); ax.set_yticklabels(zl2)
    ax.set_title('H₀ from Individual Hubble Flow SNe Ia'); ax.legend(loc='upper right')
    plt.tight_layout(); fig.savefig('report/images/fig4_H0_individual.png'); plt.close()

    # Fig 5: Variants
    fig, ax = plt.subplots(figsize=(12, 7))
    vnames = [k for k in variants if variants[k] is not None]
    vH0 = [variants[k]['H0'] for k in vnames]
    ve = [variants[k]['H0_err'] for k in vnames]
    yp = np.arange(len(vnames))
    ax.errorbar(vH0, yp, xerr=ve, fmt='o', color='steelblue', ms=8, capsize=4)
    ax.axvline(73.5, color='gray', ls='--', lw=1.5, label='Expected: 73.5')
    ax.axvline(67.4, color='blue', ls=':', lw=1.5, label='Planck: 67.4')
    ax.set_xlabel('H₀ (km s⁻¹ Mpc⁻¹)'); ax.set_yticks(yp); ax.set_yticklabels(vnames)
    ax.set_title('H₀ from Analysis Variants'); ax.legend(loc='lower right')
    plt.tight_layout(); fig.savefig('report/images/fig5_variants.png'); plt.close()

    # Fig 6: H0 comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    r2p = [('GLS Network', gls['H0'], gls['H0_err']),
           ('Cepheid+SN Ia', h0a, h0e),
           ('Planck CMB', 67.4, 0.5),
           ('Expected Baseline', 73.5, 0.81)]
    nms = [r[0] for r in r2p]; h0s = [r[1] for r in r2p]; es = [r[2] for r in r2p]
    cols = ['steelblue','cornflowerblue','indianred','gray']
    yp = np.arange(len(nms))
    for i in range(len(nms)):
        ax.errorbar(h0s[i], yp[i], xerr=es[i], fmt='o', color=cols[i], ms=10, capsize=6, lw=2)
    ax.axvline(73.5, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('H₀ (km s⁻¹ Mpc⁻¹)'); ax.set_yticks(yp); ax.set_yticklabels(nms)
    ax.set_title('Hubble Constant Measurements: Comparison'); ax.set_xlim(55, 135)
    plt.tight_layout(); fig.savefig('report/images/fig6_H0_comparison.png'); plt.close()

    # Fig 7: SBF Hubble diagram
    fig, ax = plt.subplots(figsize=(10, 7))
    sz = [s[0] for s in hubble_flow_sbf]; sm = [s[1] for s in hubble_flow_sbf]; se = [s[2] for s in hubble_flow_sbf]
    ax.errorbar(sz, sm, yerr=se, fmt='D', color='darkgreen', ms=8, capsize=4, label='Hubble Flow SBF')
    zl3 = np.linspace(0.015, 0.055, 100)
    ax.plot(zl3, 5*np.log10(c_km*zl3/H0f)+25+gls['MSBF'], 'r-', lw=2, label=f'Fit: H₀={H0f:.1f}')
    ax.set_xlabel('Redshift z'); ax.set_ylabel('m$_{F110W}$'); ax.set_title('Hubble Diagram: SBF')
    ax.legend(); ax.invert_yaxis()
    plt.tight_layout(); fig.savefig('report/images/fig7_SBF_hubble.png'); plt.close()

    # Fig 8: Anchor comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, an in enumerate(['N4258','LMC','MW']):
        ax = axes[idx]
        am = [m for m in host_measurements if m[2]==an]
        if not am: ax.set_title(f'{an}\n(No data)'); continue
        hs = [m[0] for m in am]; ms = [m[3] for m in am]; es = [m[4] for m in am]
        yp = np.arange(len(hs))
        ax.errorbar(ms, yp, xerr=es, fmt='o', color='steelblue', ms=8, capsize=4)
        for i,h in enumerate(hs): ax.annotate(h, (ms[i], yp[i]), xytext=(5,3), fontsize=8)
        ax.set_title(f'{an} (μ={anchors[an]["mu"]:.3f}±{anchors[an]["err"]:.3f})')
        ax.set_xlabel('μ_meas (mag)'); ax.set_yticks(yp); ax.set_yticklabels([m[1] for m in am])
    plt.suptitle('Primary Indicators by Anchor', y=1.02); plt.tight_layout()
    fig.savefig('report/images/fig8_anchor_comparison.png'); plt.close()

    # Fig 9: Correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cov = np.array(gls['cov']); nh = len(gls['all_hosts'])
    kc = cov[nh:, nh:]; pn = ['M$_B$','M$_{SBF}$','log$_{10}$(H₀)']
    d = np.sqrt(np.diag(kc)); corr = kc/np.outer(d, d)
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(pn)
    ax.set_yticks(range(3)); ax.set_yticklabels(pn)
    plt.colorbar(im, ax=ax, label='Correlation')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr[i,j]:.3f}', ha='center', va='center', fontsize=12,
                   color='white' if abs(corr[i,j])>0.5 else 'black')
    ax.set_title('Parameter Correlations')
    plt.tight_layout(); fig.savefig('report/images/fig9_correlation.png'); plt.close()

    print("All figures saved.")

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("H0 via Local Distance Network")
    print("="*60)

    hd = compute_host_distances()
    print("\nHost Distances:")
    for h in sorted(hd): print(f"  {h}: μ={hd[h]['mu']:.3f}±{hd[h]['err']:.3f}")

    MBa, MBe, MBv, MBe2, MBh = calibrate_sneia(hd)
    print(f"\nMB = {MBa:.3f} ± {MBe:.3f}")

    h0s, h0se, _, _ = compute_H0_HF(MBa, MBe)
    print(f"H0 (simple) = {h0s:.2f} ± {h0se:.2f}")

    print("\nJoint GLS Fit:")
    gls = joint_gls_fit()
    print(f"  H0 = {gls['H0']:.2f} ± {gls['H0_err']:.2f}")
    print(f"  MB = {gls['MB']:.3f} ± {gls['MB_err']:.3f}")
    print(f"  M_SBF = {gls['MSBF']:.3f} ± {gls['MSBF_err']:.3f}")
    print(f"  χ²/dof = {gls['chi2']:.1f}/{gls['ndof']}")

    variants = run_variants()
    print("\nVariants:")
    for k,v in sorted(variants.items()):
        if v: print(f"  {k}: H0={v['H0']:.2f}±{v['H0_err']:.2f}")

    make_figures(hd, MBa, MBe, gls, variants)

    out = {
        'H0_gls': gls['H0'], 'H0_gls_err': gls['H0_err'],
        'H0_simple': h0s, 'H0_simple_err': h0se,
        'MB': gls['MB'], 'MB_err': gls['MB_err'],
        'M_SBF': gls['MSBF'], 'M_SBF_err': gls['MSBF_err'],
        'chi2_per_dof': gls['chi2']/gls['ndof'],
        'host_distances': {h: {'mu': hd[h]['mu'], 'err': hd[h]['err']} for h in hd},
        'variants': variants,
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/main_results.json','w') as f: json.dump(out, f, indent=2, default=str)
    print("\nDone. Results saved.")
