import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os

# Ensure output dirs
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
csv_path = 'data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv'
df = pd.read_csv(csv_path)

# Data overview
print('Data loaded: shape', df.shape)
min_year = df['year'].min()
max_year = df['year'].max()
summary_stats = {
    'total_records': len(df),
    'years_range': '{}-{}'.format(min_year, max_year),
    'states_nunique': df['state'].nunique(),
    'purposes_nunique': df['purpose'].nunique(),
    'agents_nunique': df['agent'].nunique(),
    'apparatus_nunique': df['apparatus'].nunique()
}
summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('outputs/data_summary.csv', index=False)
print(summary_stats)

# Spatial concentration: state counts
state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'projects']
state_counts.to_csv('outputs/state_counts.csv', index=False)

# Map
gdf_states = gpd.read_file('data/dataset1_cloud_seeding_records/us_states.geojson')
state_counts['state_lower'] = state_counts['state'].str.lower()
gdf_states['name_lower'] = gdf_states['name'].str.lower()
gdf = gdf_states.merge(state_counts, left_on='name_lower', right_on='state_lower', how='left')
gdf['projects'] = gdf['projects'].fillna(0).astype(int)

fig, ax = plt.subplots(1,1, figsize=(12,8))
gdf.plot(column='projects', ax=ax, legend=True, cmap='Blues', 
         legend_kwds={'label': "Number of Projects", 'orientation': "horizontal"})
ax.set_title('Spatial Concentration of Cloud Seeding Projects by State (2000-2025)')
ax.axis('off')
plt.tight_layout()
plt.savefig('report/images/state_map.png', dpi=300, bbox_inches='tight')
plt.close()

# Annual dynamics
yearly_counts = df.groupby('year').size().reset_index(name='projects')
yearly_counts.to_csv('outputs/yearly_counts.csv', index=False)

plt.figure(figsize=(12,6))
plt.plot(yearly_counts['year'], yearly_counts['projects'], marker='o')
plt.title('Annual Cloud Seeding Activity (Projects per Year)')
plt.xlabel('Year')
plt.ylabel('Number of Projects')
plt.grid(True)
plt.savefig('report/images/annual_activity.png', dpi=300, bbox_inches='tight')
plt.close()

# Purpose composition
purpose_counts = df['purpose'].value_counts().reset_index()
purpose_counts.columns = ['purpose', 'projects']
purpose_counts.to_csv('outputs/purpose_counts.csv', index=False)

plt.figure(figsize=(12,8))
sns.barplot(data=purpose_counts.head(10), y='purpose', x='projects')
plt.title('Top 10 Stated Purposes of Cloud Seeding Projects')
plt.xlabel('Number of Projects')
plt.tight_layout()
plt.savefig('report/images/purpose_composition.png', dpi=300, bbox_inches='tight')
plt.close()

# Agent x Apparatus crosstab
agent_app_crosstab = pd.crosstab(df['agent'], df['apparatus'])
agent_app_crosstab.to_csv('outputs/agent_apparatus_crosstab.csv')

plt.figure(figsize=(10,8))
sns.heatmap(agent_app_crosstab, annot=True, cmap='YlOrRd', fmt='d')
plt.title('Agent-Apparatus Deployment Patterns (Counts)')
plt.xlabel('Apparatus')
plt.ylabel('Seeding Agent')
plt.tight_layout()
plt.savefig('report/images/agent_apparatus_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Season counts
season_counts = df['season'].value_counts().reset_index()
season_counts.columns = ['season', 'projects']
season_counts.to_csv('outputs/season_counts.csv', index=False)

print('All artifacts saved successfully.')