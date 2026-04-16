import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Load data
df = pd.read_csv('../data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv')

# 1. Spatial concentration
plt.figure(figsize=(10, 6))
state_counts = df['state'].value_counts()
sns.barplot(x=state_counts.values, y=state_counts.index, hue=state_counts.index, palette="viridis", legend=False)
plt.title('Spatial Concentration of Cloud Seeding Projects (2000-2025)')
plt.xlabel('Number of Projects')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('../report/images/spatial_concentration.png', dpi=300)
plt.close()

# Also save a table
state_counts.to_csv('../outputs/spatial_concentration.csv')

# 2. Annual activity dynamics
plt.figure(figsize=(12, 6))
year_counts = df['year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color='b', linewidth=2)
plt.title('Annual Cloud Seeding Activity Dynamics (2000-2025)')
plt.xlabel('Year')
plt.ylabel('Number of Projects')
plt.xticks(np.arange(2000, 2026, 2))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('../report/images/annual_dynamics.png', dpi=300)
plt.close()

year_counts.to_csv('../outputs/annual_dynamics.csv')

# 3. Purpose composition
plt.figure(figsize=(10, 8))
purpose_counts = df['purpose'].value_counts()
plt.pie(purpose_counts.values, labels=purpose_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3", len(purpose_counts)))
plt.title('Composition of Cloud Seeding Purposes')
plt.axis('equal')
plt.tight_layout()
plt.savefig('../report/images/purpose_composition.png', dpi=300)
plt.close()

purpose_counts.to_csv('../outputs/purpose_composition.csv')

# 4. Agent-apparatus deployment patterns
# Clean up apparatus and agent strings if necessary
df['agent'] = df['agent'].str.lower().str.strip()
df['apparatus'] = df['apparatus'].str.lower().str.strip()

agent_apparatus = pd.crosstab(df['agent'], df['apparatus'])
plt.figure(figsize=(12, 8))
sns.heatmap(agent_apparatus, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Number of Projects'})
plt.title('Agent-Apparatus Deployment Patterns')
plt.xlabel('Apparatus')
plt.ylabel('Seeding Agent')
plt.tight_layout()
plt.savefig('../report/images/agent_apparatus_heatmap.png', dpi=300)
plt.close()

agent_apparatus.to_csv('../outputs/agent_apparatus.csv')

# Try mapping with geopandas
try:
    usa = gpd.read_file('../data/dataset1_cloud_seeding_records/us_states.geojson')
    # Map state names to title case to match
    df['state_title'] = df['state'].str.title()
    state_counts_df = df['state_title'].value_counts().reset_index()
    state_counts_df.columns = ['name', 'Project_Count']
    
    merged = usa.merge(state_counts_df, on='name', how='left')
    merged['Project_Count'] = merged['Project_Count'].fillna(0)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged.plot(column='Project_Count', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, 
                legend_kwds={'label': "Number of Projects", 'orientation': "horizontal"})
    # Exclude Alaska and Hawaii for better continental view if desired, but let's just plot all
    ax.set_xlim([-130, -65])
    ax.set_ylim([24, 50])
    plt.title('Geographic Distribution of Cloud Seeding Projects')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../report/images/spatial_map.png', dpi=300)
    plt.close()
except Exception as e:
    print(f"Geo map failed: {e}")

print("Analysis complete.")
