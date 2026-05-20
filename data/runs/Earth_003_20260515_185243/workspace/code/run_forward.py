import torch
import xarray as xr
from u_transformer import UTransformer

# Load input
ds = xr.open_dataset('data/20231012-06_input_netcdf.nc')
inp = torch.from_numpy(ds['data'].values).float()  # (2,70,181,360)
inp = inp.unsqueeze(0)  # (1,2,70,181,360)
inp = inp.view(1, 140, 181, 360)  # merge time into channels

model = UTransformer(in_channels=140, out_channels=70)
model.eval()
with torch.no_grad():
    out = model(inp)
print('Input shape:', inp.shape)
print('Output shape:', out.shape)
torch.save(out.cpu().numpy(), 'outputs/forecast.npy')
print('Success: forward pass completed.')
print('Saved forecast to outputs/forecast.npy')