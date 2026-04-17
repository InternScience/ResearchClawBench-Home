#!/usr/bin/env python3
"""
Comprehensive analysis of NOAA cloud-seeding records (2000-2025).
Reproduces spatial concentration, annual dynamics, purpose composition,
and agent-apparatus deployment patterns.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.figsize': (10, 6),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

# Paths
BASE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_001_20260417_103256"
DATA_PATH = os.path.join(BASE, "data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv")
GEO_PATH = os.path.join(BASE, "data/dataset1_cloud_seeding_records/us_states.geojson")
IMG_DIR = os.path.join(BASE, "report/images")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print(f"States: {df['state'].nunique()}")

# ============================================================
# 1. DATA OVERVIEW
# ============================================================
overview = {
    "total_records": int(df.shape[0]),
    "columns": list(df.columns),
    "year_range": [int(df['year'].min()), int(df['year'].max())],
    "n_states": int(df['state'].nunique()),
    "states": sorted(df['state'].unique().tolist()),
    "n_unique_projects": int(df['project'].nunique()),
    "n_unique_operators": int(df['operator_affiliation'].nunique()),
    "missing_values": df.isnull().sum().to_dict()
}
with open(os.path.join(OUT_DIR, "data_overview.json"), 'w') as f:
    json.dump(overview, f, indent=2)
print("\n[1] Data overview saved.")

# ============================================================
# 2. SPATIAL CONCENTRATION
# ============================================================
state_counts = df['state'].value_counts().sort_values(ascending=False)
state_df = state_counts.reset_index()
state_df.columns = ['state', 'project_count']
state_df['percentage'] = (state_df['project_count'] / state_df['project_count'].sum() * 100).round(1)
state_df['cumulative_pct'] = state_df['percentage'].cumsum().round(1)
state_df.to_csv(os.path.join(OUT_DIR, "state_project_counts.csv"), index=False)
print("\n[2] State project counts:")
print(state_df.to_string(index=False))

# Figure 1: State-level bar chart
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(state_df)))[::-1]
bars = ax.barh(range(len(state_df)), state_df['project_count'], color=colors)
ax.set_yticks(range(len(state_df)))
ax.set_yticklabels([s.title() for s in state_df['state']])
ax.invert_yaxis()
ax.set_xlabel('Number of Projects')
ax.set_title('Cloud-Seeding Projects by State (2000–2025)')
for i, (cnt, pct) in enumerate(zip(state_df['project_count'], state_df['percentage'])):
    ax.text(cnt + 2, i, f'{cnt} ({pct}%)', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig1_state_distribution.png"))
plt.close()
print("Figure 1 saved: state distribution")

# Figure 1b: Choropleth map
try:
    import geopandas as gpd
    gdf = gpd.read_file(GEO_PATH)
    # Standardize state names
    gdf['state_lower'] = gdf['NAME'].str.lower()
    state_map = state_df.set_index('state')['project_count'].to_dict()
    gdf['project_count'] = gdf['state_lower'].map(state_map).fillna(0)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    # Plot all states in light gray
    gdf.plot(ax=ax, color='#f0f0f0', edgecolor='gray', linewidth=0.5)
    # Plot states with data
    gdf_active = gdf[gdf['project_count'] > 0]
    gdf_active.plot(ax=ax, column='project_count', cmap='YlOrRd', edgecolor='gray',
                    linewidth=0.5, legend=True,
                    legend_kwds={'label': 'Number of Projects', 'shrink': 0.6})
    # Add state labels for active states
    for idx, row in gdf_active.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(f"{row['NAME']}\n({int(row['project_count'])})", 
                    xy=(centroid.x, centroid.y), fontsize=7, ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_xlim(-130, -95)
    ax.set_ylim(25, 50)
    ax.set_title('Spatial Distribution of U.S. Cloud-Seeding Projects (2000–2025)', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "fig1b_choropleth_map.png"))
    plt.close()
    print("Figure 1b saved: choropleth map")
except Exception as e:
    print(f"Choropleth map error: {e}")

# ============================================================
# 3. ANNUAL ACTIVITY DYNAMICS
# ============================================================
year_counts = df.groupby('year').size().reset_index(name='project_count')
year_counts.to_csv(os.path.join(OUT_DIR, "annual_project_counts.csv"), index=False)
print("\n[3] Annual project counts:")
print(year_counts.to_string(index=False))

# Figure 2: Annual time series
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(year_counts['year'], year_counts['project_count'], color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
ax.plot(year_counts['year'], year_counts['project_count'], 'o-', color='darkred', markersize=5, linewidth=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Projects')
ax.set_title('Annual Cloud-Seeding Activity in the U.S. (2000–2025)')
ax.set_xticks(range(2000, 2026, 2))
ax.set_xticklabels(range(2000, 2026, 2), rotation=45)
ax.grid(axis='y', alpha=0.3)
# Add mean line
mean_val = year_counts['project_count'].mean()
ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.6, label=f'Mean = {mean_val:.1f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig2_annual_activity.png"))
plt.close()
print("Figure 2 saved: annual activity")

# Figure 2b: State-Year heatmap
state_year = df.groupby(['state', 'year']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(state_year, cmap='YlOrRd', annot=True, fmt='d', linewidths=0.5,
            ax=ax, cbar_kws={'label': 'Number of Projects'}, annot_kws={'size': 7})
ax.set_title('Cloud-Seeding Projects by State and Year (2000–2025)')
ax.set_xlabel('Year')
ax.set_ylabel('State')
ax.set_yticklabels([s.title() for s in state_year.index], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig2b_state_year_heatmap.png"))
plt.close()
print("Figure 2b saved: state-year heatmap")
state_year.to_csv(os.path.join(OUT_DIR, "state_year_matrix.csv"))

# ============================================================
# 4. PURPOSE COMPOSITION
# ============================================================
# Normalize purposes - extract primary purposes
def extract_primary_purposes(purpose_str):
    """Extract individual purposes from comma-separated strings."""
    purposes = [p.strip().lower() for p in purpose_str.split(',')]
    return purposes

# Get all individual purposes
all_purposes = []
for p in df['purpose']:
    all_purposes.extend(extract_primary_purposes(p))

purpose_series = pd.Series(all_purposes)
purpose_counts = purpose_series.value_counts()
purpose_df = purpose_counts.reset_index()
purpose_df.columns = ['purpose', 'count']
purpose_df['percentage'] = (purpose_df['count'] / purpose_df['count'].sum() * 100).round(1)
purpose_df.to_csv(os.path.join(OUT_DIR, "purpose_composition.csv"), index=False)
print("\n[4] Purpose composition (individual purposes):")
print(purpose_df.to_string(index=False))

# Also get raw (combined) purpose counts
raw_purpose = df['purpose'].value_counts().reset_index()
raw_purpose.columns = ['purpose_combination', 'count']
raw_purpose['percentage'] = (raw_purpose['count'] / raw_purpose['count'].sum() * 100).round(1)
raw_purpose.to_csv(os.path.join(OUT_DIR, "purpose_combinations.csv"), index=False)

# Figure 3: Purpose composition
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Bar chart of individual purposes
ax1 = axes[0]
colors_p = plt.cm.Set2(np.linspace(0, 1, len(purpose_df)))
ax1.barh(range(len(purpose_df)), purpose_df['count'], color=colors_p)
ax1.set_yticks(range(len(purpose_df)))
ax1.set_yticklabels([p.title() for p in purpose_df['purpose']], fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel('Count (Individual Purpose Mentions)')
ax1.set_title('Individual Purpose Frequency')
for i, (cnt, pct) in enumerate(zip(purpose_df['count'], purpose_df['percentage'])):
    ax1.text(cnt + 1, i, f'{cnt} ({pct}%)', va='center', fontsize=8)

# Pie chart of top categories
ax2 = axes[1]
top_purposes = purpose_df.head(6).copy()
other_count = purpose_df.iloc[6:]['count'].sum() if len(purpose_df) > 6 else 0
if other_count > 0:
    top_purposes = pd.concat([top_purposes, pd.DataFrame({'purpose': ['other'], 'count': [other_count], 'percentage': [round(other_count/purpose_df['count'].sum()*100, 1)]})], ignore_index=True)
wedges, texts, autotexts = ax2.pie(top_purposes['count'], labels=[p.title() for p in top_purposes['purpose']],
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
ax2.set_title('Purpose Distribution')

plt.suptitle('Cloud-Seeding Purpose Composition (2000–2025)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig3_purpose_composition.png"))
plt.close()
print("Figure 3 saved: purpose composition")

# Figure 3b: Purpose trends over time
# Group into major categories
def categorize_purpose(p):
    p = p.lower()
    if 'snowpack' in p:
        return 'Augment Snowpack'
    elif 'hail' in p:
        return 'Suppress Hail'
    elif 'fog' in p:
        return 'Suppress Fog'
    elif 'research' in p:
        return 'Research'
    elif 'precipitation' in p or 'runoff' in p:
        return 'Increase Precipitation/Runoff'
    else:
        return 'Other'

df['purpose_category'] = df['purpose'].apply(categorize_purpose)
purpose_year = df.groupby(['year', 'purpose_category']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 7))
purpose_year.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='gray', linewidth=0.3)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Projects')
ax.set_title('Purpose Composition Over Time (2000–2025)')
ax.legend(title='Purpose Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels([str(y) for y in purpose_year.index], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig3b_purpose_trends.png"))
plt.close()
print("Figure 3b saved: purpose trends")

purpose_year.to_csv(os.path.join(OUT_DIR, "purpose_year_matrix.csv"))

# ============================================================
# 5. AGENT DEPLOYMENT PATTERNS
# ============================================================
# Normalize agents - extract primary agents
def extract_agents(agent_str):
    """Categorize agent strings into primary agent categories."""
    a = agent_str.lower()
    if 'silver iodide' in a:
        return 'Silver Iodide (+ variants)'
    elif 'dry ice' in a or 'carbon dioxide' in a:
        return 'Dry Ice / CO2'
    elif 'calcium chloride' in a:
        return 'Calcium Chloride'
    elif 'water' in a:
        return 'Water'
    elif 'sulfur dioxide' in a:
        return 'Sulfur Dioxide'
    elif 'ionized air' in a:
        return 'Ionized Air'
    elif 'ammonium iodide' in a:
        return 'Ammonium Iodide'
    else:
        return 'Other'

df['agent_category'] = df['agent'].apply(extract_agents)
agent_counts = df['agent_category'].value_counts()
agent_df = agent_counts.reset_index()
agent_df.columns = ['agent_category', 'count']
agent_df['percentage'] = (agent_df['count'] / agent_df['count'].sum() * 100).round(1)
agent_df.to_csv(os.path.join(OUT_DIR, "agent_categories.csv"), index=False)
print("\n[5] Agent categories:")
print(agent_df.to_string(index=False))

# Raw agent counts
raw_agent = df['agent'].value_counts().reset_index()
raw_agent.columns = ['agent', 'count']
raw_agent['percentage'] = (raw_agent['count'] / raw_agent['count'].sum() * 100).round(1)
raw_agent.to_csv(os.path.join(OUT_DIR, "agent_raw_counts.csv"), index=False)

# Figure 4: Agent deployment
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Category-level
ax1 = axes[0]
colors_a = plt.cm.Paired(np.linspace(0, 1, len(agent_df)))
bars = ax1.barh(range(len(agent_df)), agent_df['count'], color=colors_a)
ax1.set_yticks(range(len(agent_df)))
ax1.set_yticklabels(agent_df['agent_category'])
ax1.invert_yaxis()
ax1.set_xlabel('Number of Projects')
ax1.set_title('Seeding Agent Categories')
for i, (cnt, pct) in enumerate(zip(agent_df['count'], agent_df['percentage'])):
    ax1.text(cnt + 2, i, f'{cnt} ({pct}%)', va='center', fontsize=9)

# Top raw agents
ax2 = axes[1]
top_raw = raw_agent.head(10)
ax2.barh(range(len(top_raw)), top_raw['count'], color='coral')
ax2.set_yticks(range(len(top_raw)))
ax2.set_yticklabels(top_raw['agent'], fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel('Number of Projects')
ax2.set_title('Top 10 Specific Agent Formulations')

plt.suptitle('Seeding Agent Deployment Patterns (2000–2025)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig4_agent_deployment.png"))
plt.close()
print("Figure 4 saved: agent deployment")

# ============================================================
# 6. APPARATUS DISTRIBUTION
# ============================================================
apparatus_counts = df['apparatus'].value_counts().reset_index()
apparatus_counts.columns = ['apparatus', 'count']
apparatus_counts['percentage'] = (apparatus_counts['count'] / apparatus_counts['count'].sum() * 100).round(1)
apparatus_counts.to_csv(os.path.join(OUT_DIR, "apparatus_distribution.csv"), index=False)
print("\n[6] Apparatus distribution:")
print(apparatus_counts.to_string(index=False))

# Figure 5: Apparatus distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
colors_ap = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
wedges, texts, autotexts = ax1.pie(apparatus_counts['count'], 
                                     labels=apparatus_counts['apparatus'].str.title(),
                                     autopct='%1.1f%%', startangle=90, colors=colors_ap[:len(apparatus_counts)])
ax1.set_title('Apparatus Type Distribution')

# Apparatus over time
ax2 = axes[1]
app_year = df.groupby(['year', 'apparatus']).size().unstack(fill_value=0)
app_year.plot(kind='area', stacked=True, ax=ax2, alpha=0.7, colormap='Set1')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Projects')
ax2.set_title('Apparatus Usage Over Time')
ax2.legend(title='Apparatus', fontsize=8)

plt.suptitle('Deployment Apparatus Patterns (2000–2025)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig5_apparatus_distribution.png"))
plt.close()
print("Figure 5 saved: apparatus distribution")

# ============================================================
# 7. AGENT-APPARATUS CROSS-TABULATION
# ============================================================
agent_app = pd.crosstab(df['agent_category'], df['apparatus'])
agent_app.to_csv(os.path.join(OUT_DIR, "agent_apparatus_crosstab.csv"))
print("\n[7] Agent-Apparatus cross-tabulation:")
print(agent_app.to_string())

# Figure 6: Agent-apparatus heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(agent_app, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Number of Projects'})
ax.set_title('Agent–Apparatus Cross-Tabulation (2000–2025)')
ax.set_xlabel('Apparatus Type')
ax.set_ylabel('Agent Category')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig6_agent_apparatus_heatmap.png"))
plt.close()
print("Figure 6 saved: agent-apparatus heatmap")

# ============================================================
# 8. SEASON ANALYSIS
# ============================================================
# Normalize seasons
def categorize_season(s):
    s = s.lower().replace(',', ', ')
    seasons = set()
    for token in s.split(','):
        token = token.strip()
        if token in ['winter', 'spring', 'summer', 'fall']:
            seasons.add(token)
    if len(seasons) == 1:
        return list(seasons)[0].title()
    elif len(seasons) > 1:
        return 'Multi-Season'
    return 'Unknown'

df['season_category'] = df['season'].apply(categorize_season)
season_counts = df['season_category'].value_counts().reset_index()
season_counts.columns = ['season', 'count']
season_counts['percentage'] = (season_counts['count'] / season_counts['count'].sum() * 100).round(1)
season_counts.to_csv(os.path.join(OUT_DIR, "season_distribution.csv"), index=False)
print("\n[8] Season distribution:")
print(season_counts.to_string(index=False))

# Figure 7: Season distribution
fig, ax = plt.subplots(figsize=(8, 6))
colors_s = {'Winter': '#2196F3', 'Summer': '#FF9800', 'Spring': '#4CAF50', 
            'Fall': '#795548', 'Multi-Season': '#9C27B0', 'Unknown': '#9E9E9E'}
bar_colors = [colors_s.get(s, '#9E9E9E') for s in season_counts['season']]
ax.bar(range(len(season_counts)), season_counts['count'], color=bar_colors)
ax.set_xticks(range(len(season_counts)))
ax.set_xticklabels(season_counts['season'], rotation=30)
ax.set_xlabel('Season')
ax.set_ylabel('Number of Projects')
ax.set_title('Seasonal Distribution of Cloud-Seeding Projects (2000–2025)')
for i, (cnt, pct) in enumerate(zip(season_counts['count'], season_counts['percentage'])):
    ax.text(i, cnt + 1, f'{cnt}\n({pct}%)', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig7_season_distribution.png"))
plt.close()
print("Figure 7 saved: season distribution")

# ============================================================
# 9. OPERATOR AFFILIATION ANALYSIS
# ============================================================
operator_counts = df['operator_affiliation'].value_counts().reset_index()
operator_counts.columns = ['operator', 'count']
operator_counts['percentage'] = (operator_counts['count'] / operator_counts['count'].sum() * 100).round(1)
operator_counts.to_csv(os.path.join(OUT_DIR, "operator_affiliation.csv"), index=False)
print("\n[9] Top operators:")
print(operator_counts.head(15).to_string(index=False))

# Figure 8: Top operators
fig, ax = plt.subplots(figsize=(12, 7))
top_ops = operator_counts.head(15)
ax.barh(range(len(top_ops)), top_ops['count'], color='teal', alpha=0.8)
ax.set_yticks(range(len(top_ops)))
ax.set_yticklabels(top_ops['operator'].str.title(), fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of Projects')
ax.set_title('Top 15 Operator Affiliations (2000–2025)')
for i, (cnt, pct) in enumerate(zip(top_ops['count'], top_ops['percentage'])):
    ax.text(cnt + 1, i, f'{cnt} ({pct}%)', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig8_operator_affiliation.png"))
plt.close()
print("Figure 8 saved: operator affiliation")

# ============================================================
# 10. STATE-PURPOSE CROSS-TABULATION
# ============================================================
state_purpose = pd.crosstab(df['state'], df['purpose_category'])
state_purpose.to_csv(os.path.join(OUT_DIR, "state_purpose_crosstab.csv"))

fig, ax = plt.subplots(figsize=(12, 7))
state_purpose.plot(kind='barh', stacked=True, ax=ax, colormap='Set2', edgecolor='gray', linewidth=0.3)
ax.set_xlabel('Number of Projects')
ax.set_ylabel('')
ax.set_yticklabels([s.title() for s in state_purpose.index], rotation=0)
ax.set_title('Purpose Composition by State (2000–2025)')
ax.legend(title='Purpose', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig9_state_purpose.png"))
plt.close()
print("Figure 9 saved: state-purpose composition")

# ============================================================
# 11. PROJECT DURATION ANALYSIS
# ============================================================
df['start_dt'] = pd.to_datetime(df['start_date'], format='mixed', errors='coerce')
df['end_dt'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
df['duration_days'] = (df['end_dt'] - df['start_dt']).dt.days

valid_dur = df[df['duration_days'].notna() & (df['duration_days'] > 0)]
print(f"\n[11] Project duration stats (n={len(valid_dur)}):")
print(valid_dur['duration_days'].describe())

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(valid_dur['duration_days'], bins=40, color='steelblue', edgecolor='navy', alpha=0.7)
ax.set_xlabel('Project Duration (Days)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Cloud-Seeding Project Durations (2000–2025)')
ax.axvline(valid_dur['duration_days'].median(), color='red', linestyle='--', 
           label=f'Median = {valid_dur["duration_days"].median():.0f} days')
ax.axvline(valid_dur['duration_days'].mean(), color='green', linestyle='--', 
           label=f'Mean = {valid_dur["duration_days"].mean():.0f} days')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig10_project_duration.png"))
plt.close()
print("Figure 10 saved: project duration")

# ============================================================
# 12. SUMMARY STATISTICS TABLE
# ============================================================
summary = {
    "total_records": int(df.shape[0]),
    "year_range": f"{df['year'].min()}-{df['year'].max()}",
    "n_years": int(df['year'].nunique()),
    "n_states": int(df['state'].nunique()),
    "n_unique_projects": int(df['project'].nunique()),
    "n_unique_operators": int(df['operator_affiliation'].nunique()),
    "top_state": state_counts.index[0],
    "top_state_count": int(state_counts.iloc[0]),
    "top_state_pct": round(state_counts.iloc[0] / len(df) * 100, 1),
    "top_3_states": list(state_counts.index[:3]),
    "top_3_states_pct": round(state_counts.iloc[:3].sum() / len(df) * 100, 1),
    "dominant_agent": "silver iodide",
    "silver_iodide_pct": round(df['agent'].str.contains('silver iodide', case=False).sum() / len(df) * 100, 1),
    "dominant_apparatus": apparatus_counts.iloc[0]['apparatus'],
    "dominant_apparatus_pct": float(apparatus_counts.iloc[0]['percentage']),
    "mean_annual_projects": round(year_counts['project_count'].mean(), 1),
    "max_annual_projects": int(year_counts['project_count'].max()),
    "max_annual_year": int(year_counts.loc[year_counts['project_count'].idxmax(), 'year']),
    "min_annual_projects": int(year_counts['project_count'].min()),
    "min_annual_year": int(year_counts.loc[year_counts['project_count'].idxmin(), 'year']),
    "dominant_purpose": "augment snowpack",
    "winter_season_pct": round(df['season_category'].value_counts().get('Winter', 0) / len(df) * 100, 1),
    "median_duration_days": float(valid_dur['duration_days'].median()),
    "mean_duration_days": round(float(valid_dur['duration_days'].mean()), 1)
}
with open(os.path.join(OUT_DIR, "summary_statistics.json"), 'w') as f:
    json.dump(summary, f, indent=2)
print("\n[12] Summary statistics saved.")
print(json.dumps(summary, indent=2))

# ============================================================
# 13. AGENT-STATE HEATMAP
# ============================================================
agent_state = pd.crosstab(df['state'], df['agent_category'])
agent_state.to_csv(os.path.join(OUT_DIR, "agent_state_crosstab.csv"))

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(agent_state, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Number of Projects'})
ax.set_title('Agent Category by State (2000–2025)')
ax.set_xlabel('Agent Category')
ax.set_ylabel('State')
ax.set_yticklabels([s.title() for s in agent_state.index], rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig11_agent_state_heatmap.png"))
plt.close()
print("Figure 11 saved: agent-state heatmap")

# ============================================================
# 14. CONCENTRATION METRICS
# ============================================================
# Herfindahl-Hirschman Index for spatial concentration
shares = state_counts / state_counts.sum()
hhi = (shares ** 2).sum()
top3_share = state_counts.iloc[:3].sum() / state_counts.sum()
top5_share = state_counts.iloc[:5].sum() / state_counts.sum()

concentration = {
    "herfindahl_hirschman_index": round(float(hhi), 4),
    "top_3_state_share": round(float(top3_share), 4),
    "top_5_state_share": round(float(top5_share), 4),
    "gini_interpretation": "High spatial concentration - few states dominate"
}
with open(os.path.join(OUT_DIR, "concentration_metrics.json"), 'w') as f:
    json.dump(concentration, f, indent=2)
print("\n[14] Concentration metrics:")
print(json.dumps(concentration, indent=2))

print("\n=== Analysis Complete ===")
print(f"Total figures generated: 11")
print(f"Total output files: {len(os.listdir(OUT_DIR))}")
