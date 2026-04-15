#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('report/images', exist_ok=True)

df = pd.read_csv('outputs/mangroves_matched.csv')

# Normalize risks 0-1
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

scens = ['SSP245', 'SSP370', 'SSP585']
for scen in scens:
    slr_col = f'slr_2100_mm_{scen}'
    df[f'slr_risk_{scen}'] = normalize(df[slr_col])
df['tc_risk'] = normalize(df['tc_freq_yr'])

# Composite: average
for scen in scens:
    df[f'risk_{scen}'] = 0.5 * df[f'slr_risk_{scen}'] + 0.5 * df['tc_risk']

df_risk = df[['uid', 'lon', 'lat', 'area_ha', 'tc_risk', 'slr_risk_SSP585', 'risk_SSP585']].copy()
df_risk.to_csv('outputs/risk_index.csv', index=False)

# Plots
fig, axes = plt.subplots(2,2, figsize=(15,12))
# TC risk hist
axes[0,0].hist(df.tc_risk, bins=50)
axes[0,0].set_title('TC Risk Distribution')

# SLR risk SSP585
axes[0,1].hist(df['slr_risk_SSP585'], bins=50)
axes[0,1].set_title('SLR Risk SSP5-8.5')

# Composite
axes[1,0].hist(df['risk_SSP585'], bins=50)
axes[1,0].set_title('Composite Risk SSP5-8.5')

# Area weighted
axes[1,1].scatter(df['risk_SSP585'], df.area_ha, alpha=0.5)
axes[1,1].set_xlabel('Risk'); axes[1,1].set_ylabel('Area (ha)')
plt.tight_layout()
plt.savefig('report/images/risk_histograms.png', dpi=300)
plt.close()

# Risk map SSP585
import cartopy.crs as ccrs
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
sc = ax.scatter(df.lon, df.lat, c=df['risk_SSP585'], s=df.area_ha.clip(upper=100)/1000, cmap='Reds', transform=ccrs.PlateCarree(), alpha=0.6)
ax.coastlines()
ax.set_global()
ax.set_title('Composite Risk Index SSP5-8.5 (size ~ area)')
plt.colorbar(sc, ax=ax, label='Risk (0-1)')
plt.savefig('report/images/risk_map_SSP585.png', dpi=300, bbox_inches='tight')
plt.close()

print('Risk computation complete!')