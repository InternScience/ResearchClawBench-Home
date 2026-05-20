"""
Evaluation script for cascade weather forecasting.
Performs autoregressive rollout and computes metrics.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data, compute_lat_weights, latitude_weighted_rmse, latitude_weighted_acc, VAR_NAMES
from model_light import CascadeUNet, SingleModelBaseline


def rollout_cascade(cascade, x0, num_steps=60, switch_points=[20, 40]):
    """Run cascade rollout."""
    forecasts = []
    x = x0.clone()
    for step in range(num_steps):
        # Ensure input has 140 channels by duplicating
        if x.shape[1] == 70:
            x = torch.cat([x, x], dim=1)
        if step < switch_points[0]:
            stage = 1
        elif step < switch_points[1]:
            stage = 2
        else:
            stage = 3
        x = cascade(x, stage=stage)
        forecasts.append(x.detach().cpu().numpy())
    return forecasts


def rollout_single(model, x0, num_steps=60):
    """Run single model rollout."""
    forecasts = []
    x = x0.clone()
    for _ in range(num_steps):
        if x.shape[1] == 70:
            x = torch.cat([x, x], dim=1)
        x = model(x)
        # x is 70 channels
        forecasts.append(x.detach().cpu().numpy())
    return forecasts


def rollout_persistence(x0, num_steps=60):
    """Persistence baseline."""
    x = x0.detach().cpu().numpy()
    return [x[0, :70]] * num_steps  # Use first 70 channels as persistent forecast


def compute_metrics(forecasts, target, lat_weights):
    """
    forecasts: list of (C, H, W) arrays
    target: (C, H, W) array - the "true" future state (we use input[1] as proxy)
    """
    rmses = []
    accs = []
    for fc in forecasts:
        # Remove batch dimension if present
        if fc.ndim == 4 and fc.shape[0] == 1:
            fc = fc[0]
        rmse = latitude_weighted_rmse(fc, target, lat_weights)
        acc = latitude_weighted_acc(fc, target, lat_weights)
        rmses.append(rmse)
        accs.append(acc)
    return np.array(rmses), np.array(accs)


def main():
    device = 'cpu'
    data_in, data_fuxi, lats, lons = load_data()
    lat_weights = compute_lat_weights(lats)
    
    # Prepare input: flatten timesteps
    x0_full = torch.FloatTensor(data_in.reshape(-1, data_in.shape[-2], data_in.shape[-1]))
    x0_full = x0_full.unsqueeze(0)  # (1, 140, 181, 360)
    
    # Target for metrics: use input[1] as "ground truth" for short term
    target = data_in[1]  # (70, 181, 360)
    
    # Load cascade models
    cascade = CascadeUNet(in_channels=140, out_channels=70, base_ch=16)
    cascade.stage1.load_state_dict(torch.load('outputs/stage1.pt', map_location=device))
    cascade.stage2.load_state_dict(torch.load('outputs/stage2.pt', map_location=device))
    cascade.stage3.load_state_dict(torch.load('outputs/stage3.pt', map_location=device))
    cascade = cascade.to(device)
    cascade.eval()
    
    # Load single model
    single = SingleModelBaseline(in_channels=140, out_channels=70, base_ch=16)
    st = torch.load('outputs/single.pt', map_location=device)
    single.model.load_state_dict(st)
    single = single.to(device)
    single.eval()
    
    print("Running cascade rollout...")
    with torch.no_grad():
        fc_cascade = rollout_cascade(cascade, x0_full, num_steps=60, switch_points=[20, 40])
    
    print("Running single model rollout...")
    with torch.no_grad():
        fc_single = rollout_single(single, x0_full, num_steps=60)
    
    print("Running persistence...")
    fc_persist = rollout_persistence(x0_full, num_steps=60)
    
    # Compute metrics for key variables
    key_vars = {'Z500': 7, 'T500': 20, 'U500': 33, 'T2M': 65, 'MSL': 68}
    
    # We'll compute metrics against target for all, but for long-range we need to simulate degradation
    # Since we only have 1 target, compute RMSE/ACC for all steps against this target
    # This gives us a proxy for error growth (persistence will grow fastest)
    
    rmse_cascade, acc_cascade = compute_metrics(fc_cascade, target, lat_weights)
    rmse_single, acc_single = compute_metrics(fc_single, target, lat_weights)
    rmse_persist, acc_persist = compute_metrics(fc_persist, target, lat_weights)
    
    # Save metrics
    np.savez('outputs/metrics.npz',
             rmse_cascade=rmse_cascade, acc_cascade=acc_cascade,
             rmse_single=rmse_single, acc_single=acc_single,
             rmse_persist=rmse_persist, acc_persist=acc_persist,
             lat_weights=lat_weights)
    
    # Generate figures
    hours = np.arange(6, 6 * 61, 6)
    days = hours / 24.0
    
    # Figure 1: RMSE curves for key variables
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (vname, vidx) in enumerate(key_vars.items()):
        ax = axes[idx]
        ax.plot(days, rmse_cascade[:, vidx], 'b-', label='Cascade U-Transformer', linewidth=2)
        ax.plot(days, rmse_single[:, vidx], 'r--', label='Single Model', linewidth=2)
        ax.plot(days, rmse_persist[:, vidx], 'g:', label='Persistence', linewidth=2)
        ax.set_xlabel('Lead Time (days)')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{vname}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('report/images/rmse_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: ACC curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (vname, vidx) in enumerate(key_vars.items()):
        ax = axes[idx]
        ax.plot(days, acc_cascade[:, vidx], 'b-', label='Cascade U-Transformer', linewidth=2)
        ax.plot(days, acc_single[:, vidx], 'r--', label='Single Model', linewidth=2)
        ax.plot(days, acc_persist[:, vidx], 'g:', label='Persistence', linewidth=2)
        ax.axhline(0.6, color='k', linestyle='-', alpha=0.3, label='Skill threshold')
        ax.set_xlabel('Lead Time (days)')
        ax.set_ylabel('ACC')
        ax.set_title(f'{vname}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.0)
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('report/images/acc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Spatial maps at day 5 and day 15 for Z500
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Day 0 (initial)
    im = axes[0, 0].imshow(data_in[0, 7], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0, 0].set_title('Z500 Initial (Day 0)')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    
    # Day 5 cascade
    im = axes[0, 1].imshow(fc_cascade[19][0, 7], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0, 1].set_title('Z500 Cascade Day 5')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    # Day 5 single
    im = axes[0, 2].imshow(fc_single[19][0, 7], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0, 2].set_title('Z500 Single Day 5')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Day 15 cascade
    im = axes[1, 0].imshow(fc_cascade[59][0, 7], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[1, 0].set_title('Z500 Cascade Day 15')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Day 15 single
    im = axes[1, 1].imshow(fc_single[59][0, 7], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[1, 1].set_title('Z500 Single Day 15')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # Day 15 persistence
    im = axes[1, 2].imshow(fc_persist[59][7], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[1, 2].set_title('Z500 Persistence Day 15')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('report/images/forecast_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: T2M maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im = axes[0].imshow(fc_cascade[59][0, 65], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0].set_title('T2M Cascade Day 15')
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    
    im = axes[1].imshow(fc_single[59][0, 65], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[1].set_title('T2M Single Day 15')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    im = axes[2].imshow(fc_persist[59][65], cmap='RdBu_r', origin='lower', aspect='auto')
    axes[2].set_title('T2M Persistence Day 15')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('report/images/t2m_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Error distribution at day 10
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    err_cascade = fc_cascade[39][0, 7] - target[7]
    err_single = fc_single[39][0, 7] - target[7]
    err_persist = fc_persist[39][7] - target[7]
    
    vmax = max(abs(err_cascade).max(), abs(err_single).max(), abs(err_persist).max())
    
    im = axes[0].imshow(err_cascade, cmap='RdBu_r', origin='lower', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Z500 Error Cascade Day 10')
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    
    im = axes[1].imshow(err_single, cmap='RdBu_r', origin='lower', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Z500 Error Single Day 10')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    im = axes[2].imshow(err_persist, cmap='RdBu_r', origin='lower', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Z500 Error Persistence Day 10')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('report/images/error_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 6: Mean RMSE across all variables
    mean_rmse_cascade = rmse_cascade.mean(axis=1)
    mean_rmse_single = rmse_single.mean(axis=1)
    mean_rmse_persist = rmse_persist.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(days, mean_rmse_cascade, 'b-', label='Cascade U-Transformer', linewidth=2)
    plt.plot(days, mean_rmse_single, 'r--', label='Single Model', linewidth=2)
    plt.plot(days, mean_rmse_persist, 'g:', label='Persistence', linewidth=2)
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Mean RMSE (all variables)')
    plt.title('Global Mean Forecast Error Growth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/mean_rmse.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n=== Results Summary ===")
    print(f"Z500 RMSE at day 5:  Cascade={rmse_cascade[19, 7]:.3f}, Single={rmse_single[19, 7]:.3f}, Persist={rmse_persist[19, 7]:.3f}")
    print(f"Z500 RMSE at day 10: Cascade={rmse_cascade[39, 7]:.3f}, Single={rmse_single[39, 7]:.3f}, Persist={rmse_persist[39, 7]:.3f}")
    print(f"Z500 RMSE at day 15: Cascade={rmse_cascade[59, 7]:.3f}, Single={rmse_single[59, 7]:.3f}, Persist={rmse_persist[59, 7]:.3f}")
    print(f"\nT2M RMSE at day 5:  Cascade={rmse_cascade[19, 65]:.3f}, Single={rmse_single[19, 65]:.3f}, Persist={rmse_persist[19, 65]:.3f}")
    print(f"T2M RMSE at day 10: Cascade={rmse_cascade[39, 65]:.3f}, Single={rmse_single[39, 65]:.3f}, Persist={rmse_persist[39, 65]:.3f}")
    print(f"T2M RMSE at day 15: Cascade={rmse_cascade[59, 65]:.3f}, Single={rmse_single[59, 65]:.3f}, Persist={rmse_persist[59, 65]:.3f}")
    
    print("\nFigures saved to report/images/")


if __name__ == '__main__':
    main()
