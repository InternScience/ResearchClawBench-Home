import numpy as np

c_km = 299792.458

# Hubble flow SNe Ia
hf_sneia = [
    (0.034, 15.12, 0.06, 250),
    (0.042, 15.68, 0.05, 250),
    (0.055, 16.35, 0.05, 250),
    (0.068, 17.02, 0.05, 250),
    (0.082, 17.55, 0.06, 250)
]

# Hubble flow SBF
hf_sbf = [
    (0.023, 30.45, 0.15, 250),
    (0.031, 31.02, 0.15, 250),
    (0.045, 31.89, 0.16, 250)
]

def compute_alpha_simple(z, mB, err_mB, sigv):
    vcorr = c_km * z
    alpha = np.log10(vcorr) - 0.2 * mB
    # err in alpha units: (0.2*err_mB)^2 + (log10(vcorr+sigv) - log10(vcorr))^2
    err_alpha_sq = (0.2 * err_mB)**2 + (np.log10(vcorr + sigv) - np.log10(vcorr))**2
    return alpha, err_alpha_sq

# SNe Ia alpha
alphas = []
weights = []
for z, mB, err, sigv in hf_sneia:
    a, e = compute_alpha_simple(z, mB, err, sigv)
    alphas.append(a)
    weights.append(1.0/e)
    print(f"z={z:.3f} alpha={a:.5f} err={np.sqrt(e):.5f}")

alpha = np.sum(np.array(alphas) * np.array(weights)) / np.sum(weights)
alpha_err = 1.0 / np.sqrt(np.sum(weights))
print(f"\nSNe Ia alpha = {alpha:.5f} +/- {alpha_err:.5f}")

# SBF alpha
alphas_sbf = []
weights_sbf = []
for z, mF, err, sigv in hf_sbf:
    a, e = compute_alpha_simple(z, mF, err, sigv)
    alphas_sbf.append(a)
    weights_sbf.append(1.0/e)
    print(f"z={z:.3f} alpha={a:.5f} err={np.sqrt(e):.5f}")

alpha_sbf = np.sum(np.array(alphas_sbf) * np.array(weights_sbf)) / np.sum(weights_sbf)
alpha_sbf_err = 1.0 / np.sqrt(np.sum(weights_sbf))
print(f"\nSBF alpha = {alpha_sbf:.5f} +/- {alpha_sbf_err:.5f}")

# Calibrators
sneia_cal = [
    ('NGC1309', 12.10, 0.05),
    ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05),
    ('NGC1559', 12.22, 0.05),
    ('M101', 9.85, 0.04),
    ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06)
]

host_mu = {
    'NGC1309': 32.50,
    'NGC1365': 31.33,
    'NGC1448': 31.31,
    'NGC1559': 31.42,
    'M101': 29.12,
    'NGC1316': 31.39,
    'NGC5643': 30.53,
}

print("\nCalibrator M_B and implied H0:")
for h, mB, err in sneia_cal:
    mu = host_mu[h]
    M_B = mB - mu
    logH0 = 0.2 * M_B + alpha + 5
    H0 = 10**logH0
    print(f"{h:10s} M_B={M_B:7.3f}  H0={H0:7.2f}")

# Weighted mean M_B
M_Bs = [mB - host_mu[h] for h, mB, err in sneia_cal]
errs = [err for h, mB, err in sneia_cal]
weights_mb = [1.0/e**2 for e in errs]
M_B_mean = np.sum(np.array(M_Bs) * np.array(weights_mb)) / np.sum(weights_mb)
print(f"\nWeighted mean M_B = {M_B_mean:.3f}")
logH0 = 0.2 * M_B_mean + alpha + 5
print(f"Implied H0 = {10**logH0:.2f}")
