import pandas as pd
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load calibration data
calib = pd.read_csv('data/tg_calibration.csv')
X = calib.tg_md.values.reshape(-1, 1)
y = calib.tg_exp.values

# Train GP
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
gp.fit(X, y)

# Save model
joblib.dump(gp, 'outputs/gp_model.pkl')

# Evaluate
y_pred, y_std = gp.predict(X, return_std=True)
r2 = gp.score(X, y)
mae = np.mean(np.abs(y - y_pred))
print(f'GP R2: {r2:.4f}, MAE: {mae:.2f} K')

# Plot fit
plt.figure(figsize=(8,6))
plt.scatter(calib.tg_md, y, alpha=0.6, label='Data')
order = np.sort(X.flatten())
plt.plot(order, gp.predict(order), 'r-', label='GP Mean')
plt.fill_between(order.flatten(), gp.predict(order).flatten() - 1.96 * gp.predict(order.reshape(-1,1), return_std=True)[1], 
                 gp.predict(order).flatten() + 1.96 * gp.predict(order.reshape(-1,1), return_std=True)[1], alpha=0.3)
plt.plot([200,600],[200,600], 'k--')
plt.xlabel('Tg MD'); plt.ylabel('Tg Exp'); plt.legend(); plt.title('GP Calibration')
plt.savefig('report/images/gp_calibration.png', dpi=300, bbox_inches='tight')
plt.close()

# Apply to vitrimers
vit = pd.read_csv('data/tg_vitrimer_MD.csv')
vit_tg_md = vit.tg.values.reshape(-1,1)
vit['tg_calib'] = gp.predict(vit_tg_md)
vit['tg_calib_std'] = gp.predict(vit_tg_md, return_std=True)[1]
vit.to_csv('outputs/calibrated_vitrimers.csv', index=False)

print('Calibrated vitrimers saved. Stats:')
print(vit.tg_calib.describe())
print(f'Min Tg calib: {vit.tg_calib.min():.1f}, Max: {vit.tg_calib.max():.1f}')
