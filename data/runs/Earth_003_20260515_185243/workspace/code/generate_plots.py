import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

os.makedirs('report/images', exist_ok=True)

# Load forecast
f = np.load('outputs/forecast.npy', allow_pickle=True)
raw = f['forecast/data.pkl']
forecast = pickle.loads(raw)  # shape (1, 70, 181, 360)

print("Forecast shape:", forecast.shape)

# Channel indices (approximate, based on typical ERA5 ordering)
T2M_IDX = 0      # 2m temperature (surface)
MSLP_IDX = 3     # MSLP (surface)
U10_IDX = 1      # 10m u-wind
V10_IDX = 2      # 10m v-wind
Z500_IDX = 4     # geopotential 500 hPa (upper air example)

# Extract fields
t2m = forecast[0, T2M_IDX]
mslp = forecast[0, MSLP_IDX]
u10 = forecast[0, U10_IDX]
v10 = forecast[0, V10_IDX]
z500 = forecast[0, Z500_IDX]

# Figure 1: Surface temperature
plt.figure(figsize=(10, 6))
plt.imshow(t2m, cmap='RdBu_r', origin='lower')
plt.colorbar(label='Temperature (K)')
plt.title('Forecast Surface Temperature (2m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('report/images/figure1_surface_temp.png', dpi=150)
plt.close()

# Figure 2: Mean Sea Level Pressure
plt.figure(figsize=(10, 6))
plt.imshow(mslp, cmap='viridis', origin='lower')
plt.colorbar(label='MSLP (Pa)')
plt.title('Forecast Mean Sea Level Pressure')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('report/images/figure2_mslp.png', dpi=150)
plt.close()

# Figure 3: 10m Wind Speed
wind_speed = np.sqrt(u10**2 + v10**2)
plt.figure(figsize=(10, 6))
plt.imshow(wind_speed, cmap='plasma', origin='lower')
plt.colorbar(label='Wind Speed (m/s)')
plt.title('Forecast 10m Wind Speed')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('report/images/figure3_wind_speed.png', dpi=150)
plt.close()

# Figure 4: 500 hPa Geopotential
plt.figure(figsize=(10, 6))
plt.imshow(z500, cmap='coolwarm', origin='lower')
plt.colorbar(label='Geopotential (m²/s²)')
plt.title('Forecast 500 hPa Geopotential')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('report/images/figure4_z500.png', dpi=150)
plt.close()

print("All figures saved successfully to report/images/")
