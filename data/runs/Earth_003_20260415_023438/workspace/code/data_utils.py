"""
Data utilities for loading and processing ERA5 atmospheric data.
"""
import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.interpolate import interp1d


# Variable names at 13 pressure levels
LEVEL_NAMES = [
    'Z50', 'Z100', 'Z150', 'Z200', 'Z250', 'Z300', 'Z400', 'Z500', 'Z600', 'Z700', 'Z850', 'Z925', 'Z1000',
    'T50', 'T100', 'T150', 'T200', 'T250', 'T300', 'T400', 'T500', 'T600', 'T700', 'T850', 'T925', 'T1000',
    'U50', 'U100', 'U150', 'U200', 'U250', 'U300', 'U400', 'U500', 'U600', 'U700', 'U850', 'U925', 'U1000',
    'V50', 'V100', 'V150', 'V200', 'V250', 'V300', 'V400', 'V500', 'V600', 'V700', 'V850', 'V925', 'V1000',
    'R50', 'R100', 'R150', 'R200', 'R250', 'R300', 'R400', 'R500', 'R600', 'R700', 'R850', 'R925', 'R1000',
    'T2M', 'U10', 'V10', 'MSL', 'TP'
]

# Key variables for evaluation
KEY_VARS = {
    'Z500': 7,    # Geopotential at 500 hPa
    'Z850': 10,   # Geopotential at 850 hPa
    'T500': 20,   # Temperature at 500 hPa
    'T850': 23,   # Temperature at 850 hPa
    'U850': 36,   # U-wind at 850 hPa
    'V850': 49,   # V-wind at 850 hPa
    'R500': 56,   # Relative humidity at 500 hPa
    'T2M': 65,    # 2m temperature
    'U10': 66,    # 10m u-wind
    'V10': 67,    # 10m v-wind
    'MSL': 68,    # Mean sea level pressure
    'TP': 69,     # Total precipitation
}

# Variable groups
UPPER_AIR_VARS = {
    'Z': list(range(0, 13)),
    'T': list(range(13, 26)),
    'U': list(range(26, 39)),
    'V': list(range(39, 52)),
    'R': list(range(52, 65)),
}
SURFACE_VARS = {
    'T2M': 65, 'U10': 66, 'V10': 67, 'MSL': 68, 'TP': 69
}

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def load_input_data(path='data/20231012-06_input_netcdf.nc'):
    """Load input data (two consecutive time steps)."""
    ds = nc.Dataset(path)
    data = ds.variables['data'][:]  # (2, 70, 181, 360)
    lats = ds.variables['lat'][:]   # (181,)
    lons = ds.variables['lon'][:]   # (360,)
    times = ds.variables['time'][:] # (2,)
    ds.close()
    return np.array(data), np.array(lats), np.array(lons), np.array(times)


def load_fuxi_data(path='data/006.nc'):
    """Load FuXi forecast data."""
    ds = nc.Dataset(path)
    data = ds.variables['data'][:]  # (1, 1, 70, 181, 360)
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    ds.close()
    return np.array(data), np.array(lats), np.array(lons)


def compute_latitude_weights(lats):
    """Compute latitude weights for weighted RMSE/ACC."""
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    return weights


def latitude_weighted_rmse(pred, target, lats):
    """Compute latitude-weighted RMSE."""
    weights = compute_latitude_weights(lats)
    diff = pred - target
    sq_diff = diff ** 2
    # weights shape: (lat,), broadcast over (lat, lon)
    weighted_sq = sq_diff * weights[np.newaxis, :, np.newaxis]
    rmse = np.sqrt(np.mean(weighted_sq))
    return rmse


def latitude_weighted_rmse_per_step(pred, target, lats):
    """Compute latitude-weighted RMSE for each variable and step."""
    weights = compute_latitude_weights(lats)
    results = {}
    for var_name, var_idx in KEY_VARS.items():
        p = pred[:, var_idx] if pred.ndim == 3 else pred[var_idx]
        t = target[:, var_idx] if target.ndim == 3 else target[var_idx]
        diff = p - t
        sq_diff = diff ** 2
        weighted_sq = sq_diff * weights[:, np.newaxis]
        rmse = np.sqrt(np.mean(weighted_sq))
        results[var_name] = rmse
    return results


def compute_anomaly_correlation(pred, target, climatology, lats):
    """Compute Anomaly Correlation Coefficient (ACC)."""
    weights = compute_latitude_weights(lats)
    pred_anom = pred - climatology
    target_anom = target - climatology
    
    numerator = np.sum(weights[:, np.newaxis] * pred_anom * target_anom)
    denom_pred = np.sqrt(np.sum(weights[:, np.newaxis] * pred_anom ** 2))
    denom_target = np.sqrt(np.sum(weights[:, np.newaxis] * target_anom ** 2))
    
    acc = numerator / (denom_pred * denom_target + 1e-10)
    return acc


def compute_climatology(data, var_idx, lats):
    """Compute zonal-mean climatology for a variable."""
    # Simple zonal mean as climatology proxy
    return np.mean(data[var_idx], axis=1, keepdims=True)


if __name__ == "__main__":
    data, lats, lons, times = load_input_data()
    print(f"Input data shape: {data.shape}")
    print(f"Lat range: {lats.min()} to {lats.max()}")
    print(f"Lon range: {lons.min()} to {lons.max()}")
    print(f"Time values: {times}")
    
    fuxi_data, _, _ = load_fuxi_data()
    print(f"FuXi data shape: {fuxi_data.shape}")
    
    # Test RMSE
    rmse = latitude_weighted_rmse(data[0], data[1], lats)
    print(f"RMSE between t0 and t1: {rmse:.4f}")
