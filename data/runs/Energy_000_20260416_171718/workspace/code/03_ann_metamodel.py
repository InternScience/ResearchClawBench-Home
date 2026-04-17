#!/usr/bin/env python3
"""
ANN Meta-Model for Rapid Parameter Identification
Uses neural network to map voltage curve features to ECAT model parameters
"""

import os
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260416_171718"
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")

print("=" * 60)
print("ANN META-MODEL TRAINING FOR PARAMETER IDENTIFICATION")
print("=" * 60)

# ============================================================================
# Load Training Data
# ============================================================================
print("\n[1] Loading training data...")
with open(os.path.join(OUTPUTS_DIR, 'ann_training_data.json'), 'r') as f:
    train_data = json.load(f)

X_params = np.array(train_data['params'])
X_features = np.array(train_data['features'])

# Voltage curves have variable length - pad to max length
volt_lengths = [len(v) for v in train_data['voltage_curves']]
max_len = max(volt_lengths)
y_voltage = np.zeros((len(train_data['voltage_curves']), max_len))
for i, v in enumerate(train_data['voltage_curves']):
    y_voltage[i, :len(v)] = v

print(f"  Parameters shape: {X_params.shape}")
print(f"  Features shape: {X_features.shape}")
print(f"  Voltage curves shape: {y_voltage.shape} (padded)")

# Parameter names from ECAT model
PARAM_NAMES = [
    'R_p_n', 'R_p_p', 'D_s_n', 'D_s_p', 'k_n', 'k_p',
    'eps_s_n', 'eps_s_p', 'eps_e', 'h', 'rho_cp', 'k_SEI', 'R_SEI_0'
]

FEATURE_NAMES = [
    'V_initial', 'V_final', 'V_mean', 'V_std',
    't_discharge', 'delta_T', 'Q_capacity'
]

# ============================================================================
# Data Preprocessing
# ============================================================================
print("\n[2] Preprocessing data...")

# Log-transform parameters that span orders of magnitude
log_params = ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI', 'R_SEI_0']
X_params_processed = X_params.copy()

for i, name in enumerate(PARAM_NAMES):
    if name in log_params:
        X_params_processed[:, i] = np.log10(X_params[:, i])

# Scale features and parameters
scaler_features = StandardScaler()
scaler_params = StandardScaler()

X_features_scaled = scaler_features.fit_transform(X_features)
X_params_scaled = scaler_params.fit_transform(X_params_processed)

# Save scalers
with open(os.path.join(OUTPUTS_DIR, 'scaler_features.pkl'), 'wb') as f:
    pickle.dump(scaler_features, f)
with open(os.path.join(OUTPUTS_DIR, 'scaler_params.pkl'), 'wb') as f:
    pickle.dump(scaler_params, f)

print(f"  Features scaled: mean={X_features_scaled.mean():.3f}, std={X_features_scaled.std():.3f}")
print(f"  Parameters scaled: mean={X_params_scaled.mean():.3f}, std={X_params_scaled.std():.3f}")

# ============================================================================
# Build ANN Model
# ============================================================================
print("\n[3] Building ANN architecture...")

def build_ann_model(input_dim, output_dim):
    """Build multi-layer perceptron for parameter prediction"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

input_dim = X_features_scaled.shape[1]
output_dim = X_params_scaled.shape[1]

model = build_ann_model(input_dim, output_dim)
model.summary()

# ============================================================================
# Train ANN Model
# ============================================================================
print("\n[4] Training ANN model...")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_features_scaled, X_params_scaled, test_size=0.2, random_state=42
)

print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=50, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print(f"\n  Training completed at epoch {len(history.history['loss'])}")
print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
print(f"  Final val loss: {history.history['val_loss'][-1]:.6f}")

# Save model
model.save(os.path.join(OUTPUTS_DIR, 'ann_metamodel.h5'))
print(f"  Saved model to: outputs/ann_metamodel.h5")

# ============================================================================
# Evaluate Model Performance
# ============================================================================
print("\n[5] Evaluating model performance...")

# Predict on full dataset
y_pred_scaled = model.predict(X_features_scaled, verbose=0)

# Inverse transform
y_pred = scaler_params.inverse_transform(y_pred_scaled)
y_true = scaler_params.inverse_transform(X_params_scaled)

# Calculate errors
errors = np.abs(y_pred - y_true)
relative_errors = errors / (np.abs(y_true) + 1e-10) * 100

print("\n  Parameter Prediction Errors:")
print("  " + "-" * 60)
for i, name in enumerate(PARAM_NAMES):
    mae = np.mean(errors[:, i])
    mape = np.mean(relative_errors[:, i])
    print(f"  {name:12s}: MAE={mae:.4e}, MAPE={mape:.2f}%")

# Overall statistics
overall_mape = np.mean(relative_errors)
print(f"\n  Overall MAPE: {overall_mape:.2f}%")

# Save evaluation results
eval_results = {
    'param_names': PARAM_NAMES,
    'mae': np.mean(errors, axis=0).tolist(),
    'mape': np.mean(relative_errors, axis=0).tolist(),
    'overall_mape': float(overall_mape),
    'training_epochs': len(history.history['loss']),
    'final_train_loss': float(history.history['loss'][-1]),
    'final_val_loss': float(history.history['val_loss'][-1])
}

with open(os.path.join(OUTPUTS_DIR, 'ann_evaluation.json'), 'w') as f:
    json.dump(eval_results, f, indent=2)

# ============================================================================
# Generate Training Plots
# ============================================================================
print("\n[6] Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ANN Meta-Model Training Results', fontsize=14, fontweight='bold')

# Plot 1: Training history
ax = axes[0, 0]
ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Training History')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: MAE per parameter
ax = axes[0, 1]
x_pos = np.arange(len(PARAM_NAMES))
mae_values = np.mean(errors, axis=0)
bars = ax.bar(x_pos, mae_values, color='steelblue', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('MAE')
ax.set_title('Mean Absolute Error per Parameter')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: True vs Predicted (key parameters)
ax = axes[1, 0]
key_params = ['R_p_n', 'D_s_n', 'k_n', 'h']
colors = plt.cm.viridis(np.linspace(0, 1, len(key_params)))
for i, name in enumerate(key_params):
    idx = PARAM_NAMES.index(name)
    ax.scatter(y_true[:, idx], y_pred[:, idx], alpha=0.5, 
               color=colors[i], label=name, s=30)
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
        'k--', linewidth=2, label='Ideal')
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.set_title('True vs Predicted (Key Parameters)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: Relative error distribution
ax = axes[1, 1]
ax.boxplot(relative_errors, labels=PARAM_NAMES, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7))
ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Relative Error (%)')
ax.set_title('Relative Error Distribution per Parameter')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'ann_training_results.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {IMAGES_DIR}/ann_training_results.png")
plt.close()

# Additional plot: Feature importance (using permutation-based approach)
print("\n[7] Analyzing feature sensitivity...")

# Simple sensitivity analysis: perturb each feature and observe output change
base_features = X_features_scaled.mean(axis=0, keepdims=True)
base_pred = model.predict(base_features, verbose=0)

sensitivities = np.zeros((len(FEATURE_NAMES), output_dim))
delta = 0.1  # Perturbation size

for i in range(len(FEATURE_NAMES)):
    perturbed = base_features.copy()
    perturbed[0, i] += delta
    pert_pred = model.predict(perturbed, verbose=0)
    sensitivities[i, :] = np.abs(pert_pred - base_pred).flatten() / delta

# Plot sensitivity heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sensitivities.T, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Input Features')
ax.set_ylabel('Output Parameters')
ax.set_xticks(np.arange(len(FEATURE_NAMES)))
ax.set_yticks(np.arange(len(PARAM_NAMES)))
ax.set_xticklabels(FEATURE_NAMES, fontsize=9)
ax.set_yticklabels(PARAM_NAMES, fontsize=8)
ax.set_title('Feature Sensitivity Analysis', fontsize=12, fontweight='bold')

# Add value labels
for i in range(len(PARAM_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        text = ax.text(j, i, f'{sensitivities[j, i]:.3f}',
                      ha='center', va='center', color='black', fontsize=7)

plt.colorbar(im, label='Sensitivity')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'feature_sensitivity.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {IMAGES_DIR}/feature_sensitivity.png")
plt.close()

# Save sensitivity data
sensitivity_data = {
    'features': FEATURE_NAMES,
    'parameters': PARAM_NAMES,
    'sensitivities': sensitivities.tolist()
}
with open(os.path.join(OUTPUTS_DIR, 'feature_sensitivity.json'), 'w') as f:
    json.dump(sensitivity_data, f, indent=2)

print("\n" + "=" * 60)
print("ANN META-MODEL TRAINING COMPLETE")
print("=" * 60)
