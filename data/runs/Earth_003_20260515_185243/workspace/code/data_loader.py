import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

class ERA5Dataset(Dataset):
    def __init__(self, input_path, fuxi_path):
        self.input_ds = xr.open_dataset(input_path)
        self.fuxi_ds = xr.open_dataset(fuxi_path)
        # Shapes: (time=2, channel=70, lat=181, lon=360)
        self.input_data = self.input_ds['data'].values.astype(np.float32)
        # FuXi: (time=1, step=1, channel=70, lat=181, lon=360)
        self.target_data = self.fuxi_ds['data'].values.astype(np.float32)

    def __len__(self):
        return 1  # single sample for demo

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_data)  # (2, 70, 181, 360)
        y = torch.from_numpy(self.target_data[0, 0])  # (70, 181, 360)
        return x, y

def get_data_loader(input_path, fuxi_path, batch_size=1):
    ds = ERA5Dataset(input_path, fuxi_path)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)