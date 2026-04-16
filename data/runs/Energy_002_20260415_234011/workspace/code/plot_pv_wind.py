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
gdf.plot(ax=ax[0], column='theo_pv', cmap='plasma', legend=True,
         legend_kwds={'label': "PV Potential (Capacity Factor)"})
ax[0].set_title('Solar PV Potential')

africa.plot(ax=ax[1], color='lightgrey', edgecolor='white')
gdf.plot(ax=ax[1], column='theo_wind', cmap='viridis', legend=True,
         legend_kwds={'label': "Wind Potential (Capacity Factor)"})
ax[1].set_title('Wind Potential')

plt.tight_layout()
plt.savefig('report/images/potential_map.png', dpi=300)

print("Potential map saved to report/images/potential_map.png")
