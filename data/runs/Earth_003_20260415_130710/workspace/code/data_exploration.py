"""
Data exploration and visualization for cascade weather forecasting system.
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os

def load_data(filepath):
    """Load NetCDF data file."""
    ds = Dataset(filepath, 'r')
    data = ds.variables['data'][:]
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    level = ds.variables['level'][:]
    ds.close()
    return data, lat, lon, level

def get_variable_info():
    """Get information about variables in the dataset."""
    variables = {
        'upper_air': {
            'Z': {'name': 'Geopotential', 'levels': ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'], 'unit': 'm²/s²'},
            'T': {'name': 'Temperature', 'levels': ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'], 'unit': 'K'},
            'U': {'name': 'U-wind', 'levels': ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'], 'unit': 'm/s'},
            'V': {'name': 'V-wind', 'levels': ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'], 'unit': 'm/s'},
            'R': {'name': 'Relative Humidity', 'levels': ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'], 'unit': '%'}
        },
        'surface': {
            'T2M': {'name': '2m Temperature', 'unit': 'K'},
            'U10': {'name': '10m U-wind', 'unit': 'm/s'},
            'V10': {'name': '10m V-wind', 'unit': 'm/s'},
            'MSL': {'name': 'Mean Sea Level Pressure', 'unit': 'hPa'},
            'TP': {'name': 'Total Precipitation', 'unit': 'm'}
        }
    }
    return variables

def plot_global_map(data, lat, lon, title, cmap='RdBu_r', vmin=None, vmax=None, save_path=None):
    """Plot a global map of a variable."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Plot
    im = ax.contourf(lon_grid, lat_grid, data, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def plot_variable_at_level(data, lat, lon, var_name, level_idx, time_idx=0, save_path=None):
    """Plot a specific variable at a specific level."""
    # Variable indices: Z=0-12, T=13-25, U=26-38, V=39-51, R=52-64, surface=65-69
    var_map = {
        'Z500': 7, 'T2M': 65, 'U10': 66, 'V10': 67, 'MSL': 68, 'TP': 69
    }
    
    if var_name in var_map:
        idx = var_map[var_name]
    else:
        idx = level_idx
    
    var_data = data[time_idx, idx, :, :]
    
    # Set appropriate colormap and limits
    if var_name.startswith('Z'):
        cmap = 'viridis'
        title = f'{var_name} (Geopotential)'
    elif var_name.startswith('T') or var_name == 'T2M':
        cmap = 'RdYlBu_r'
        title = f'{var_name} (Temperature)'
    elif var_name.startswith('U') or var_name.startswith('V'):
        cmap = 'RdBu_r'
        title = f'{var_name} (Wind)'
    elif var_name == 'MSL':
        cmap = 'viridis'
        title = f'{var_name} (Mean Sea Level Pressure)'
    elif var_name == 'TP':
        cmap = 'Blues'
        title = f'{var_name} (Total Precipitation)'
    else:
        cmap = 'RdBu_r'
        title = var_name
    
    plot_global_map(var_data, lat, lon, title, cmap=cmap, save_path=save_path)

def plot_vertical_cross_section(data, lat, lon, var_type='T', time_idx=0, save_path=None):
    """Plot vertical cross-section (latitude vs pressure level) averaged over longitude."""
    # Map variable type to channel indices
    var_ranges = {
        'Z': (0, 13),
        'T': (13, 26),
        'U': (26, 39),
        'V': (39, 52),
        'R': (52, 65)
    }
    
    start, end = var_ranges[var_type]
    var_data = data[time_idx, start:end, :, :]  # (levels, lat, lon)
    
    # Average over longitude
    zonal_mean = np.mean(var_data, axis=2)  # (levels, lat)
    
    # Pressure levels in hPa (reversed to go from top to bottom)
    levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create contour plot
    contour = ax.contourf(lat, levels, zonal_mean, levels=20, cmap='RdBu_r')
    ax.invert_yaxis()
    ax.set_xlabel('Latitude', fontsize=12)
    ax.set_ylabel('Pressure (hPa)', fontsize=12)
    ax.set_title(f'Zonal Mean {var_type} - Vertical Cross Section', fontsize=14)
    
    cbar = plt.colorbar(contour, ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def create_data_overview(data, lat, lon, save_dir='report/images'):
    """Create comprehensive data overview plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot key variables
    variables_to_plot = ['Z500', 'T2M', 'U10', 'V10', 'MSL']
    
    for var in variables_to_plot:
        save_path = os.path.join(save_dir, f'input_{var}.png')
        plot_variable_at_level(data, lat, lon, var, 0, time_idx=0, save_path=save_path)
    
    # Plot vertical cross-sections
    for var_type in ['T', 'U', 'Z']:
        save_path = os.path.join(save_dir, f'vertical_cross_section_{var_type}.png')
        plot_vertical_cross_section(data, lat, lon, var_type, time_idx=0, save_path=save_path)
    
    # Plot comparison between two time steps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Z500 at t=0 and t=1
    for i, t in enumerate([0, 1]):
        z500 = data[t, 7, :, :]  # Z500 is index 7
        im = axes[0, i].contourf(lon, lat, z500, levels=20, cmap='viridis')
        axes[0, i].set_title(f'Z500 at t={t*6}h', fontsize=12)
        axes[0, i].set_xlabel('Longitude')
        axes[0, i].set_ylabel('Latitude')
        plt.colorbar(im, ax=axes[0, i], shrink=0.6)
    
    # T2M at t=0 and t=1
    for i, t in enumerate([0, 1]):
        t2m = data[t, 65, :, :]  # T2M is index 65
        im = axes[1, i].contourf(lon, lat, t2m, levels=20, cmap='RdYlBu_r')
        axes[1, i].set_title(f'T2M at t={t*6}h', fontsize=12)
        axes[1, i].set_xlabel('Longitude')
        axes[1, i].set_ylabel('Latitude')
        plt.colorbar(im, ax=axes[1, i], shrink=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'temporal_evolution_input.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def analyze_variable_statistics(data, save_dir='outputs'):
    """Compute and save statistics for all variables."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Variable names
    var_names = ['Z50', 'Z100', 'Z150', 'Z200', 'Z250', 'Z300', 'Z400', 'Z500', 'Z600', 'Z700', 'Z850', 'Z925', 'Z1000',
                 'T50', 'T100', 'T150', 'T200', 'T250', 'T300', 'T400', 'T500', 'T600', 'T700', 'T850', 'T925', 'T1000',
                 'U50', 'U100', 'U150', 'U200', 'U250', 'U300', 'U400', 'U500', 'U600', 'U700', 'U850', 'U925', 'U1000',
                 'V50', 'V100', 'V150', 'V200', 'V250', 'V300', 'V400', 'V500', 'V600', 'V700', 'V850', 'V925', 'V1000',
                 'R50', 'R100', 'R150', 'R200', 'R250', 'R300', 'R400', 'R500', 'R600', 'R700', 'R850', 'R925', 'R1000',
                 'T2M', 'U10', 'V10', 'MSL', 'TP']
    
    stats = {}
    for i, name in enumerate(var_names):
        var_data = data[:, i, :, :]
        stats[name] = {
            'mean': float(np.mean(var_data)),
            'std': float(np.std(var_data)),
            'min': float(np.min(var_data)),
            'max': float(np.max(var_data)),
            'median': float(np.median(var_data))
        }
    
    # Save statistics
    import json
    with open(os.path.join(save_dir, 'variable_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {save_dir}/variable_statistics.json")
    return stats

if __name__ == '__main__':
    # Load input data
    print("Loading input data...")
    input_data, lat, lon, level = load_data('data/20231012-06_input_netcdf.nc')
    print(f"Input data shape: {input_data.shape}")
    
    # Create data overview
    print("\nCreating data overview plots...")
    create_data_overview(input_data, lat, lon, save_dir='report/images')
    
    # Compute statistics
    print("\nComputing variable statistics...")
    analyze_variable_statistics(input_data, save_dir='outputs')
    
    print("\nData exploration complete!")
