import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load data
cal = pd.read_csv('data/tg_calibration.csv')
X = cal[['tg_md']].values
y = cal['tg_exp'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GP model
kernel = RBF(length_scale=50.0) + WhiteKernel(noise_level=10.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp.fit(X_train, y_train)

# Predictions
y_pred, y_std = gp.predict(X_test, return_std=True)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"GP Calibration RMSE: {rmse:.2f}, R2: {r2:.3f}")

# Save model
joblib.dump(gp, 'outputs/gp_calibration_model.pkl')

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel('Experimental Tg (K)')
plt.ylabel('Calibrated GP Prediction (K)')
plt.title('Gaussian Process Calibration: MD vs Experimental Tg')
plt.savefig('report/images/gp_calibration.png', dpi=150, bbox_inches='tight')
print("Saved gp_calibration.png")
