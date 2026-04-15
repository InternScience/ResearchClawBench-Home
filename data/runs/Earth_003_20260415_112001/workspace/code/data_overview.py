import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# Load data
input_ds = xr.open_dataset('data/20231012-06_input_netcdf.nc')
fuxi_ds = xr.open_dataset('data/006.nc')

levels = input_ds.level.values
print('Levels:', levels)

# Key vars indices
z500_idx = np.where(levels == 'Z500')[0][0]
t2m_idx = np.where(levels == 'T2M')[0][0]
tp_idx = np.where(levels == 'TP')[0][0]

def plot_global_var(da, title, fname):
    fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': 'cyl'})
    im = ax.contourf(da.lon, da.lat, da, cmap='RdBu_r', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.savefig(f'report/images/{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()

# But cartopy? Use pcolormesh if no cartopy

# Fallback plot without cartopy
def plot_global_simple(da, title, fname, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(12,5))
    lon, lat = np.meshgrid(da.lon, da.lat)
    if vmin is None:
        vmin, vmax = da.min(), da.max()
    im = ax.pcolormesh(lon, lat, da, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.savefig(f'report/images/{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Plot input t=0 Z500
plot_global_simple(input_ds.data.isel(time=0, level=z500_idx), 'Input Z500 t=0 (2023-10-12 00Z)', 'z500_t0')

# t=1 Z500
plot_global_simple(input_ds.data.isel(time=1, level=z500_idx), 'Input Z500 t=1 (06Z)', 'z500_t1')

# FuXi forecast Z500
plot_global_simple(fuxi_ds.data.isel(time=0, step=0, level=z500_idx), 'FuXi Z500 +6h forecast', 'fuxi_z500_6h')

# T2m input t=1
plot_global_simple(input_ds.data.isel(time=1, level=t2m_idx), 'Input T2m t=1', 't2m_t1')

# FuXi T2m
plot_global_simple(fuxi_ds.data.isel(time=0, step=0, level=t2m_idx), 'FuXi T2m +6h', 'fuxi_t2m_6h')

# TP input t=1 (precip accum? )
plot_global_simple(input_ds.data.isel(time=1, level=tp_idx)*1000, 'Input TP t=1 (mm?)', 'tp_t1', vmin=0, vmax=10)

# FuXi TP
plot_global_simple(fuxi_ds.data.isel(time=0, step=0, level=tp_idx)*1000, 'FuXi TP +6h (mm?)', 'fuxi_tp_6h', vmin=0, vmax=10)

print('Plots saved.')