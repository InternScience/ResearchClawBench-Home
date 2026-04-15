import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
with open('outputs/dataset.json', 'r') as f:
    data = json.load(f)

feat = np.array(data['property_prediction'][1])
cat = np.array(data['property_prediction'][2])
targ = np.array(data['property_prediction'][3])

# One-hot categories (short array, pad or use as is)
encoder = OneHotEncoder(sparse_output=False)
cat_encoded = encoder.fit_transform(cat.reshape(-1,1))

# Combine features
X = np.column_stack([feat, cat_encoded[:len(feat)]])  # align lengths
y = targ

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RF
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f'MAE: {mae:.4f}')

# Save model
joblib.dump(model, 'outputs/models/property_model.pkl')
joblib.dump(encoder, 'outputs/models/property_encoder.pkl')

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].scatter(y_test, y_pred, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('True')
axes[0].set_ylabel('Pred')
axes[0].set_title('Pred vs True')

residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6)
axes[1].axhline(0, color='r', linestyle='--')
axes[1].set_xlabel('Pred')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals')

plt.tight_layout()
plt.savefig('report/images/property_prediction.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
results = {'mae': float(mae), 'y_test': y_test.tolist(), 'y_pred': y_pred.tolist()}
with open('outputs/property_results.json', 'w') as f:
    json.dump(results, f)
