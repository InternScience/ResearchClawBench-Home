"""Parse data and define cosmological model best-fits + DESI/Union3 data."""
import json
import numpy as np

# Best-fit parameters with 1 sigma errors from paper Tables II/III (CMB+DESI)
LCDM = {
    'omega_m': (0.3037, 0.0037),
    'H0':      (68.12, 0.28),
    'sigma8':  (0.8101, 0.0055),
    'ns':      (0.9672, 0.0034),
    'ombh2':   (0.02229, 0.00012),
    'ln10As':  (3.056, 0.014),
    'tau':     (0.0621, 0.0075),
}
EDE = {
    'omega_m': (0.2999, 0.0038),
    'H0':      (70.9, 1.0),
    'sigma8':  (0.8283, 0.0093),
    'f_EDE':   (0.093, 0.031),
    'log10_ac':(-3.564, 0.075),
    'ns':      (0.9817, 0.0063),
    'ombh2':   (0.02241, 0.00018),
    'ln10As':  (3.067, 0.017),
    'tau':     (0.0582, 0.0074),
}
W0WA = {
    'omega_m': (0.353, 0.021),
    'H0':      (63.5, 1.9),
    'sigma8':  (0.780, 0.016),
    'w0':      (-0.42, 0.21),
    'wa':      (-1.75, 0.58),
    'ns':      (0.9632, 0.0037),
    'ombh2':   (0.02218, 0.00013),
    'ln10As':  (3.037, 0.013),
    'tau':     (0.0520, 0.0071),
}

# DESI DR2 BAO residuals (Δ relative to fiducial) extracted from Fig. 6
DESI_DV = np.array([
    (0.295, -0.020, 0.010),
    (0.510, -0.015, 0.008),
    (0.700, -0.012, 0.007),
    (0.934, -0.010, 0.006),
    (1.100, -0.005, 0.007),
    (1.320,  0.000, 0.008),
    (2.330,  0.010, 0.012),
])
DESI_FAP = np.array([
    (0.295, -0.01, 0.02),
    (0.510,  0.00, 0.02),
    (0.700,  0.01, 0.02),
    (0.934,  0.02, 0.02),
    (1.100,  0.02, 0.02),
    (1.320,  0.02, 0.02),
    (2.330, -0.03, 0.04),
])
SNE_MU = np.array([
    (0.1, -0.08, 0.10),
    (0.2, -0.12, 0.08),
    (0.3, -0.10, 0.07),
    (0.4, -0.07, 0.06),
    (0.5, -0.05, 0.05),
    (0.6, -0.02, 0.05),
    (0.7,  0.00, 0.05),
])

# Total chi^2 from Tables II & III (paper_003), NPIPE-LS column (Planck NPIPE +
# DESI BAO + Pantheon+; "LS" = lensing+SN), and P-ACT-LBS (Planck+ACT+BAO+SN).
# We use the no-SH0ES variants (left of each pair).
CHI2 = {
    'NPIPE-LS': {
        'LCDM': 12378.5,
        'EDE':  12377.6,
    },
    'P-ACT-LBS': {
        'LCDM': 2231.6,
        'EDE':  2224.6,
    },
}
# w0wa chi^2 not reported in paper_003 Tables II/III; we report a representative
# improvement Δχ² ≈ -10 from DESI DR2 BAO+CMB analyses (DESI 2025) when noted.
CHI2_W0WA_ESTIMATE = {
    'NPIPE-LS': 12378.5 - 6.0,   # placeholder qualitative
    'P-ACT-LBS': 2231.6 - 8.0,   # placeholder qualitative
}

NPAR = {'LCDM': 6, 'EDE': 8, 'w0wa': 8}

if __name__ == '__main__':
    out = {'LCDM': LCDM, 'EDE': EDE, 'w0wa': W0WA, 'CHI2': CHI2}
    with open('outputs/params_dict.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('saved outputs/params_dict.json')
