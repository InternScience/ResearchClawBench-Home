import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# We have the 006.nc which is the actual FuXi model output for 6-hour forecast
ds_fuxi = xr.open_dataset('data/006.nc')
ds_pred = xr.open_dataset('outputs/forecast_15days.nc')

fuxi_z500 = ds_fuxi['data'].values[0, 0, list(ds_fuxi['level'].values).index('Z500')]
pred_z500_6h = ds_pred['data'].values[0, 0, list(ds_pred['level'].values).index('Z500')]

# Calculate RMSE between our baseline and the actual FuXi forecast for the first 6h step
rmse = np.sqrt(np.mean((fuxi_z500 - pred_z500_6h)**2))
print(f"RMSE between our model and FuXi for Z500 at 6h: {rmse}")

# Let's plot the comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(fuxi_z500, cmap='coolwarm', origin='lower')
axes[0].set_title('FuXi Z500 (6h forecast)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pred_z500_6h, cmap='coolwarm', origin='lower')
axes[1].set_title('Our Model Z500 (6h forecast)')
plt.colorbar(im1, ax=axes[1])

diff = fuxi_z500 - pred_z500_6h
im2 = axes[2].imshow(diff, cmap='bwr', origin='lower', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
axes[2].set_title(f'Difference (RMSE: {rmse:.4f})')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('report/images/fuxi_comparison.png')
print("Saved fuxi_comparison.png")
