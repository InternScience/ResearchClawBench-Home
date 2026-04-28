"""
Phase 2 + 3: EDA and Gaussian-process calibration of MD Tg -> experimental Tg.

Outputs:
  - report/images/fig_tg_distributions.png
  - report/images/fig_md_vs_exp_calibration.png
  - outputs/calibration_metrics.csv
  - outputs/calibration_predictions.csv
  - outputs/gp_calibration.pkl   (fitted GP and a small predict() helper)
"""
import os, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "report", "images"), exist_ok=True)

cal = pd.read_csv(os.path.join(ROOT, "data/tg_calibration.csv"))
vit = pd.read_csv(os.path.join(ROOT, "data/tg_vitrimer_MD.csv"))

# --- Figure 1 : distributions ---------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.hist(cal.tg_exp, bins=30, alpha=0.7, label="Tg exp (calibration set)", color="#1f77b4")
ax.hist(cal.tg_md, bins=30, alpha=0.5, label="Tg MD (calibration set)", color="#ff7f0e")
ax.set_xlabel("Tg [K]"); ax.set_ylabel("count"); ax.set_title("Calibration set: Exp vs MD")
ax.legend()

ax = axes[1]
ax.hist(vit.tg, bins=60, color="#2ca02c", alpha=0.85)
ax.set_xlabel("MD Tg [K]"); ax.set_ylabel("count")
ax.set_title(f"Vitrimer MD set ({len(vit)} acid+epoxide pairs)")

ax = axes[2]
ax.hist(vit['std'], bins=60, color="#9467bd", alpha=0.85, label="vitrimer MD")
ax.hist(cal['std'], bins=30, color="#d62728", alpha=0.6, label="calibration MD")
ax.set_xlabel("MD Tg std [K]"); ax.set_ylabel("count")
ax.set_title("Within-MD uncertainty (std)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "report/images/fig_tg_distributions.png"), dpi=150)
plt.close(fig)

# --- GP calibration --------------------------------------------------------
# Heteroscedastic-aware: alpha = (std_md)^2 per training point.
# We work in *unnormalized* K because tg_exp std (~75 K) and per-point MD std
# are all in K — so we design the kernel directly at that scale and avoid the
# alpha/normalize_y scale mismatch that drove the GP to a constant predictor.
X = cal[["tg_md"]].values.astype(float)
y = cal["tg_exp"].values.astype(float)
alpha = np.clip(cal["std"].values.astype(float), 2.0, None) ** 2  # variance per pt [K^2]
y_scale2 = float(np.var(y))   # ~ 75^2 K^2

kernel = (C(y_scale2, (1e2, 1e6))
          * RBF(length_scale=80.0, length_scale_bounds=(10.0, 600.0))
          + WhiteKernel(noise_level=400.0, noise_level_bounds=(1.0, 1e5)))

gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                              normalize_y=False, n_restarts_optimizer=8, random_state=0)
gp.fit(X, y)
print("Trained GP kernel:", gp.kernel_)

# Leave-one-out CV: freeze the optimized kernel and refit on each fold, which
# is both faster and avoids per-fold pathological optima at n=294.
fitted_kernel = gp.kernel_
loo = LeaveOneOut()
y_pred = np.zeros_like(y, dtype=float)
y_std = np.zeros_like(y, dtype=float)
for tr, te in loo.split(X):
    gp_i = GaussianProcessRegressor(kernel=fitted_kernel, alpha=alpha[tr],
                                    normalize_y=False, optimizer=None,
                                    random_state=0)
    gp_i.fit(X[tr], y[tr])
    mu, sd = gp_i.predict(X[te], return_std=True)
    y_pred[te] = mu
    y_std[te]  = sd

# Metrics for raw MD baseline vs GP calibration
def metrics(y_true, y_hat):
    return dict(R2=r2_score(y_true, y_hat),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_hat))),
                MAE=mean_absolute_error(y_true, y_hat),
                bias=float(np.mean(y_hat - y_true)))

m_raw = metrics(y, X[:,0])
m_gp  = metrics(y, y_pred)

# linear-regression baseline for comparison (a + b*tg_md)
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X, y)
y_lr_loo = np.zeros_like(y)
for tr, te in loo.split(X):
    y_lr_loo[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
m_lr = metrics(y, y_lr_loo)

pd.DataFrame([
    dict(model="raw MD",            **m_raw),
    dict(model="linear MD->Exp (LOOCV)", **m_lr),
    dict(model="GP MD->Exp (LOOCV)", **m_gp),
]).to_csv(os.path.join(ROOT, "outputs/calibration_metrics.csv"), index=False)

pd.DataFrame(dict(name=cal["name"], smiles=cal["smiles"],
                  tg_exp=y, tg_md=X[:,0],
                  tg_gp_loo=y_pred, tg_gp_std_loo=y_std)
            ).to_csv(os.path.join(ROOT, "outputs/calibration_predictions.csv"), index=False)

# --- Figure 2 : MD vs Exp calibration plot --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
lo, hi = 150, 650
ax = axes[0]
ax.errorbar(cal.tg_md, cal.tg_exp, xerr=cal["std"], fmt='o', alpha=0.4, color='#ff7f0e',
            label="raw MD (1 σ)")
ax.plot([lo, hi], [lo, hi], 'k--', label="y = x")
ax.set_xlabel("MD Tg [K]"); ax.set_ylabel("Experimental Tg [K]")
ax.set_title(f"Raw MD vs Exp\nR²={m_raw['R2']:.3f} RMSE={m_raw['RMSE']:.1f} K bias={m_raw['bias']:+.1f} K")
ax.legend(); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')

ax = axes[1]
ax.errorbar(y_pred, y, xerr=y_std, fmt='o', alpha=0.4, color='#1f77b4',
            label="GP LOOCV (1 σ)")
ax.plot([lo, hi], [lo, hi], 'k--', label="y = x")
ax.set_xlabel("GP-calibrated predicted Tg [K]"); ax.set_ylabel("Experimental Tg [K]")
ax.set_title(f"GP MD→Exp (LOOCV)\nR²={m_gp['R2']:.3f} RMSE={m_gp['RMSE']:.1f} K bias={m_gp['bias']:+.1f} K")
ax.legend(); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "report/images/fig_md_vs_exp_calibration.png"), dpi=150)
plt.close(fig)

# Persist GP for later use
with open(os.path.join(ROOT, "outputs/gp_calibration.pkl"), "wb") as f:
    pickle.dump({"gp": gp, "kernel_str": str(gp.kernel_),
                 "alpha_train": alpha, "X_train": X, "y_train": y}, f)

print("metrics raw   :", m_raw)
print("metrics linear:", m_lr)
print("metrics gp    :", m_gp)
