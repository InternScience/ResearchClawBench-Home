#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

os.makedirs('report/images', exist_ok=True)

df = pd.read_csv('outputs/mangroves_matched.csv')

# Normalize 0-1
def norm(col):
    return (col - col.min()) / (col.max() - col.min())

df['tc_risk'] = norm(df['tc_freq_yr'])

scens = ['SSP245', 'SSP370', 'SSP585']
for s in scens:
    slr = 'slr_2100_mm_' + s
    df['slr_risk_' + s] = norm(df[slr])
    df['composite_risk_' + s] = 0.5 * df['slr_risk_' + s] + 0.5 * df['tc_risk']

df[['uid', 'lon', 'lat', 'weight', 'tc_risk', 'slr_risk_SSP585', 'composite_risk_SSP585']].to_csv('outputs/risk_index.csv', index=False)

# Histograms
fig, axes = plt.subplots(1,3, figsize=(15,5))
axes[0].hist(df.tc_risk, bins=50)
axes[0].set_title('TC Risk')
axes[1].hist(df['slr_risk_SSP585'], bins=50)
axes[1].set_title('SLR Risk SSP585')
axes[2].hist(df['composite_risk_SSP585'], bins=50)
axes[2].set_title('Composite Risk SSP585')
plt.tight_layout()
plt.savefig('report/images/risk_hists.png', dpi=300)
plt.close()

# Risk map
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
sc = ax.scatter(df.lon, df.lat, c=df['composite_risk_SSP585'], s=5, cmap='Reds', transform=ccrs.PlateCarree(), alpha=0.6)
ax.add_feature(cfeature.COASTLINE)
ax.set_global()
ax.set_title('Composite Risk (SSP5-8.5, size proportional to sample weight)')
plt.colorbar(sc, ax=ax, shrink=0.6, label='Risk Index (0-1)')
plt.savefig('report/images/composite_risk_map.png', dpi=300, bbox_inches='tight')
plt.close()

# Scenario comparison table
risk_mean = df[['composite_risk_SSP245', 'composite_risk_SSP370', 'composite_risk_SSP585']].mean()
risk_mean.to_csv('outputs/risk_summary.csv')
print(risk_mean)

print('Risk figures saved.')