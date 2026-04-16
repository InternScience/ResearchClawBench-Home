import xarray as xr
import numpy as np

ds_in = xr.open_dataset('data/20231012-06_input_netcdf.nc')
ds_out = xr.open_dataset('data/006.nc')

print("Input data shape:", ds_in.data.shape)
print("Output data shape:", ds_out.data.shape)
