import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load map
africa = gpd.read_file('data/africa_map/ne_10m_admin_0_countries.shp')
africa = africa[africa['CONTINENT'] == 'Africa']

# Load results
df = pd.read_csv('outputs/results.csv')
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

fig, ax = plt.subplots(1, 2, figsize=(15, 7))

africa.plot(ax=ax[0], color='lightgrey', edgecolor='white')
gdf.plot(ax=ax[0], column='LCOA_Base', cmap='viridis', legend=True,
         legend_kwds={'label': "Delivered Cost (€/kgH2) - 8% WACC"})
ax[0].set_title('Delivered Cost of Green Hydrogen (via Ammonia)\nBaseline Financing (8% WACC)')

africa.plot(ax=ax[1], color='lightgrey', edgecolor='white')
gdf.plot(ax=ax[1], column='LCOA_Derisked', cmap='viridis', legend=True,
         legend_kwds={'label': "Delivered Cost (€/kgH2) - 4% WACC"})
ax[1].set_title('Delivered Cost of Green Hydrogen (via Ammonia)\nDe-risked Financing (4% WACC)')

plt.tight_layout()
plt.savefig('report/images/lcoa_map.png', dpi=300)

print("Map saved to report/images/lcoa_map.png")
