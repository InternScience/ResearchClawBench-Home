"""
Phase 3: ANN Meta-model Construction

Train an Artificial Neural Network surrogate model that maps 
battery parameters -> discharge voltage curves.
This replaces expensive physical simulations during GA optimization.
"""
import numpy as np
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_000_20260415_130453"
OUTPUTS = os.path.join(WORKSPACE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

# Load simulation results
sim_data = np.load(os.path.join(OUTPUTS, "spm_simulation_results.npz"))
voltages = sim_data['voltages']
temperatures = sim_data['temperatures']
t_sim = sim_data['time']
I_app = float(sim_data['current'])

lhs_data = np.load(os.path.join(OUTPUTS, "lhs_samples.npz"))
samples = lhs_data['samples']
param_names = lhs_data['param_names'].tolist()

with open(os.path.join(OUTPUTS, "simulation_metadata.json"), 'r') as f:
    meta = json.load(f)

print("=" * 60)
print("Phase 3: ANN Meta-model Construction")
print("=" * 60)

# Filter successful simulations
success_mask = ~np.any(np.isnan(voltages), axis=1) & ~np.any(np.isinf(voltages), axis=1)
X_all = samples[success_mask]
Y_all = voltages[success_mask]

print(f"\nTotal LHS samples: {len(samples)}")
print(f"Successful simulations: {success_mask.sum()}")
print(f"Parameters: {param_names}")
print(f"Output dimension: {Y_all.shape[1]} time points")

# Log-transform parameters for better scaling
X_log = np.log10(X_all.copy())
for j, name in enumerate(param_names):
    lo, hi = meta['param_bounds'][name]
    if hi / lo > 100:
        X_log[:, j] = np.log10(X_all[:, j])

# Normalize inputs and outputs
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_norm = scaler_X.fit_transform(X_log)
Y_norm = scaler_Y.fit_transform(Y_all)

# Train/validation split
X_train, X_val, Y_train, Y_val = train_test_split(
    X_norm, Y_norm, test_size=0.15, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train_t = torch.FloatTensor(X_train).to(device)
Y_train_t = torch.FloatTensor(Y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
Y_val_t = torch.FloatTensor(Y_val).to(device)

train_dataset = TensorDataset(X_train_t, Y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class ANNSurrogate(nn.Module):
    """ANN surrogate model mapping parameters -> voltage curve."""
    
    def __init__(self, n_params, n_output, hidden_sizes=[128, 256, 256, 128]):
        super().__init__()
        layers = []
        prev_size = n_params
        
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = h_size
        
        layers.append(nn.Linear(prev_size, n_output))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Build model
n_params = X_train.shape[1]
n_output = Y_train.shape[1]

model = ANNSurrogate(n_params, n_output).to(device)
print(f"\nModel architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

# Training loop
n_epochs = 500
best_val_loss = float('inf')
best_model_state = None
train_losses = []
val_losses = []

print(f"\nTraining for {n_epochs} epochs...")
for epoch in range(n_epochs):
    # Training
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * len(batch_X)
    
    epoch_train_loss /= len(X_train)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        epoch_val_loss = criterion(val_pred, Y_val_t).item()
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    
    scheduler.step(epoch_val_loss)
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={epoch_train_loss:.6f}, val_loss={epoch_val_loss:.6f}")

# Load best model
model.load_state_dict(best_model_state)
print(f"\nBest validation loss: {best_val_loss:.6f}")

# Save model
torch.save({
    'model_state_dict': best_model_state,
    'scaler_X_mean': scaler_X.mean_,
    'scaler_X_scale': scaler_X.scale_,
    'scaler_Y_mean': scaler_Y.mean_,
    'scaler_Y_scale': scaler_Y.scale_,
    'param_names': param_names,
    'n_params': n_params,
    'n_output': n_output,
}, os.path.join(OUTPUTS, "ann_surrogate.pt"))

# Evaluate on validation set
model.eval()
with torch.no_grad():
    val_pred = model(X_val_t).cpu().numpy()
    val_true = Y_val_t.cpu().numpy()

# Denormalize
val_pred_denorm = scaler_Y.inverse_transform(val_pred)
val_true_denorm = scaler_Y.inverse_transform(val_true)

# Compute metrics
rmse_per_curve = np.sqrt(np.mean((val_pred_denorm - val_true_denorm)**2, axis=1))
mae_per_curve = np.mean(np.abs(val_pred_denorm - val_true_denorm), axis=1)
max_error_per_curve = np.max(np.abs(val_pred_denorm - val_true_denorm), axis=1)

print(f"\nValidation Metrics (denormalized):")
print(f"  Mean RMSE: {np.mean(rmse_per_curve):.4f} V")
print(f"  Median RMSE: {np.median(rmse_per_curve):.4f} V")
print(f"  Mean MAE: {np.mean(mae_per_curve):.4f} V")
print(f"  Max error (mean): {np.mean(max_error_per_curve):.4f} V")

# Save metrics
metrics = {
    'mean_rmse_V': float(np.mean(rmse_per_curve)),
    'median_rmse_V': float(np.median(rmse_per_curve)),
    'mean_mae_V': float(np.mean(mae_per_curve)),
    'mean_max_error_V': float(np.mean(max_error_per_curve)),
    'min_rmse_V': float(np.min(rmse_per_curve)),
    'max_rmse_V': float(np.max(rmse_per_curve)),
}
with open(os.path.join(OUTPUTS, "ann_metrics.json"), 'w') as f:
    json.dump(metrics, f, indent=2)

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Training Loss', alpha=0.7)
axes[0].plot(val_losses, label='Validation Loss', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('ANN Training Convergence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot sample predictions
n_show = min(5, len(val_true_denorm))
for i in range(n_show):
    axes[1].plot(t_sim, val_true_denorm[i], '--', alpha=0.5, label=f'True {i}')
    axes[1].plot(t_sim, val_pred_denorm[i], '-', alpha=0.7, label=f'Pred {i}')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Voltage (V)')
axes[1].set_title('Sample Predictions vs True')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "ann_training_results.png"), dpi=150)
plt.close()

print("\nANN meta-model training complete!")
print(f"Model saved to: {os.path.join(OUTPUTS, 'ann_surrogate.pt')}")
print(f"Training plot saved to: {os.path.join(OUTPUTS, 'ann_training_results.png')}")
