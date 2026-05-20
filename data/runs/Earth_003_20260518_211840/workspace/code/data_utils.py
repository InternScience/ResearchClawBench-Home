"""
Data loading and preprocessing utilities for cascade weather forecasting.
"""
import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset

VAR_NAMES = ['Z50', 'Z100', 'Z150', 'Z200', 'Z250', 'Z300', 'Z400', 'Z500', 'Z600', 'Z700',
             'Z850', 'Z925', 'Z1000', 'T50', 'T100', 'T150', 'T200', 'T250', 'T300', 'T400',
             'T500', 'T600', 'T700', 'T850', 'T925', 'T1000', 'U50', 'U100', 'U150', 'U200',
             'U250', 'U300', 'U400', 'U500', 'U600', 'U700', 'U850', 'U925', 'U1000', 'V50',
             'V100', 'V150', 'V200', 'V250', 'V300', 'V400', 'V500', 'V600', 'V700', 'V850',
             'V925', 'V1000', 'R50', 'R100', 'R150', 'R200', 'R250', 'R300', 'R400', 'R500',
             'R600', 'R700', 'R850', 'R925', 'R1000', 'T2M', 'U10', 'V10', 'MSL', 'TP']

# Group variables by modality
MODALITIES = {
    'geopotential': list(range(0, 13)),      # Z at 13 levels
    'temperature': list(range(13, 26)) + [65], # T at 13 levels + T2M
    'uwind': list(range(26, 39)) + [66],      # U at 13 levels + U10
    'vwind': list(range(39, 52)) + [67],      # V at 13 levels + V10
    'humidity': list(range(52, 65)),           # R at 13 levels
    'surface': [65, 66, 67, 68, 69]            # T2M, U10, V10, MSL, TP
}


def load_data():
    """Load input and FuXi forecast data."""
    nc = netCDF4.Dataset('data/20231012-06_input_netcdf.nc')
    data_in = nc.variables['data'][:]  # (2, 70, 181, 360)
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]
    nc.close()
    
    nc2 = netCDF4.Dataset('data/006.nc')
    data_fuxi = nc2.variables['data'][:]  # (1, 1, 70, 181, 360)
    nc2.close()
    data_fuxi = data_fuxi[0, 0]  # (70, 181, 360)
    
    return data_in, data_fuxi, lats, lons


def compute_lat_weights(lats):
    """Latitude weights for area-weighted metrics."""
    lats_rad = np.deg2rad(lats)
    weights = np.cos(lats_rad)
    return weights / weights.mean()


def latitude_weighted_rmse(pred, target, lat_weights):
    """
    pred, target: (C, H, W)
    lat_weights: (H,)
    """
    diff = pred - target
    mse = ((diff ** 2) * lat_weights[None, :, None]).mean(axis=(1, 2))
    return np.sqrt(mse)


def latitude_weighted_acc(pred, target, lat_weights, climatology=None):
    """
    Anomaly Correlation Coefficient.
    """
    if climatology is None:
        climatology = target.mean(axis=(1, 2), keepdims=True)
    
    pred_anom = pred - climatology
    target_anom = target - climatology
    
    num = (pred_anom * target_anom * lat_weights[None, :, None]).sum(axis=(1, 2))
    den1 = ((pred_anom ** 2) * lat_weights[None, :, None]).sum(axis=(1, 2))
    den2 = ((target_anom ** 2) * lat_weights[None, :, None]).sum(axis=(1, 2))
    
    acc = num / np.sqrt(den1 * den2 + 1e-8)
    return acc


class WeatherDataset(Dataset):
    """Synthetic weather dataset with data augmentation."""
    
    def __init__(self, data_in, num_augments=512, noise_scale=0.05):
        """
        data_in: (2, 70, 181, 360)
        Creates synthetic training pairs by perturbing the single sample.
        """
        self.data_in = data_in
        self.num_augments = num_augments
        self.noise_scale = noise_scale
        
    def __len__(self):
        return self.num_augments
    
    def __getitem__(self, idx):
        # Base input: both timesteps
        x = self.data_in.copy()  # (2, 70, 181, 360)
        
        # Target: second timestep (6h ahead)
        y = self.data_in[1].copy()  # (70, 181, 360)
        
        # Apply random perturbations
        np.random.seed(idx)
        
        # Random noise
        noise_x = np.random.randn(*x.shape).astype(np.float32) * self.noise_scale
        noise_y = np.random.randn(*y.shape).astype(np.float32) * self.noise_scale
        x = x + noise_x
        y = y + noise_y
        
        # Random longitude shift (cyclic)
        shift = np.random.randint(0, x.shape[-1])
        x = np.roll(x, shift, axis=-1)
        y = np.roll(y, shift, axis=-1)
        
        # Random latitude flip
        if np.random.rand() > 0.5:
            x = x[:, :, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        
        # Random small scale perturbation
        if np.random.rand() > 0.5:
            scale = 1.0 + np.random.randn() * 0.02
            x = x * scale
            y = y * scale
        
        # Flatten timesteps into channels: (2, 70, H, W) -> (140, H, W)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


class CascadeWeatherDataset(Dataset):
    """Dataset for training cascade models with accumulated error simulation."""
    
    def __init__(self, data_in, model_prev=None, num_steps=5, num_augments=256, 
                 noise_scale=0.05, error_scale=0.1):
        """
        For training cascade stage k, we simulate inputs that have accumulated errors
        from previous stages.
        """
        self.data_in = data_in
        self.model_prev = model_prev
        self.num_steps = num_steps
        self.num_augments = num_augments
        self.noise_scale = noise_scale
        self.error_scale = error_scale
        
    def __len__(self):
        return self.num_augments
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        
        # Start from perturbed initial condition
        x0 = self.data_in[0].copy()
        x0 = x0 + np.random.randn(*x0.shape) * self.noise_scale
        
        # Simulate error accumulation by adding structured noise
        # This simulates what happens after running previous cascade stages
        error = np.zeros_like(x0)
        for _ in range(self.num_steps):
            error += np.random.randn(*x0.shape) * self.error_scale
            # Smooth the error spatially
            from scipy.ndimage import gaussian_filter
            error = gaussian_filter(error, sigma=1.0)
        
        x = x0 + error
        
        # Target: future state with same error characteristics
        y = self.data_in[1].copy() + error * 0.5 + np.random.randn(*x0.shape) * self.noise_scale
        
        # Use both timesteps concatenated like the main dataset
        x_full = np.stack([x, y], axis=0)  # (2, 70, 181, 360)
        x_seq = x_full.reshape(-1, x_full.shape[-2], x_full.shape[-1])  # (140, 181, 360)
        
        return torch.from_numpy(x_seq.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
