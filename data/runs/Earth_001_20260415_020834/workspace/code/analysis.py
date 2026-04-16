"""
Comprehensive analysis of NOAA cloud-seeding records (US, 2000-2025).
Produces all figures, tables, and intermediate outputs for the report.
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
from collections import Counter

# ── Paths ──
DATA_PATH = 'data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv'
GEOJSON_PATH = 'data/dataset1_cloud_seeding_records/us_states.geojson'
OUT_DIR = 'outputs'
IMG_DIR = 'report/images'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ── Load data ──
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Standardize state names to title case for mapping
df['state'] = df['state'].str.strip().str.title()

# ── Helper: save JSON ──
def save_json(obj, name):
    with open(os.path.join(OUT_DIR, name), 'w') as f:
        json.dump(obj, f, indent=2, default=str)

# ──────────────────────────────────────────────
# 1. DATA OVERVIEW
# ──────────────────────────────────────────────
overview = {
    "total_records": len(df),
    "year_range": [int(df['year'].min()), int(df['year'].max())],
    "unique_projects": int(df['project'].nunique()),
    "unique_states": int(df['state'].nunique()),
    "unique_operators": int(df['operator_affiliation'].nunique()),
    "unique_agents": int(df['agent'].nunique()),
    "unique_apparatus": int(df['apparatus'].nunique()),
    "unique_purposes": int(df['purpose'].nunique()),
}
save_json(overview, 'data_overview.json')
print("Data overview:", overview)

# ──────────────────────────────────────────────
# 2. SPATIAL CONCENTRATION (Figure 1)
# ──────────────────────────────────────────────
state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'n_records']
state_counts['pct'] = (state_counts['n_records'] / state_counts['n_records'].sum() * 100).round(2)
save_json(state_counts.to_dict(orient='records'), 'spatial_concentration.json')

# Figure 1a: Choropleth map
try:
    import geopandas as gpd
    gdf = gpd.read_file(GEOJSON_PATH)
    # Normalize state names for merging
    gdf['name'] = gdf['name'].str.strip().str.title()
    merged = gdf.merge(state_counts, left_on='name', right_on='state', how='left')
    merged['n_records'] = merged['n_records'].fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8))
    merged.plot(column='n_records', cmap='YlOrRd', linewidth=0.6, edgecolor='0.4',
                legend=True, ax=ax, missing_kwds={'color': 'lightgrey', 'label': 'No data'})
    ax.set_title('Spatial Distribution of Cloud-Seeding Projects by State (2000–2025)', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig1a_choropleth.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure 1a saved.")
except Exception as e:
    print(f"Choropleth failed: {e}")
    # Fallback: bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=state_counts, y='state', x='n_records', palette='YlOrRd_r', ax=ax)
    ax.set_title('Number of Cloud-Seeding Records by State (2000–2025)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Records')
    ax.set_ylabel('State')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig1a_state_bar.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure 1a (bar fallback) saved.")

# Figure 1b: Top-10 states bar chart
fig, ax = plt.subplots(figsize=(10, 5))
top10 = state_counts.head(10)
colors = sns.color_palette('YlOrRd_r', len(top10))
bars = ax.barh(top10['state'][::-1], top10['n_records'][::-1], color=colors[::-1])
for bar, val in zip(bars, top10['n_records'][::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(val),
            va='center', fontsize=10)
ax.set_xlabel('Number of Records')
ax.set_title('Top 10 States by Cloud-Seeding Records (2000–2025)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1b_top10_states.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 1b saved.")

# ──────────────────────────────────────────────
# 3. ANNUAL ACTIVITY DYNAMICS (Figure 2)
# ──────────────────────────────────────────────
yearly = df.groupby('year').size().reset_index(name='n_records')
save_json(yearly.to_dict(orient='records'), 'annual_dynamics.json')

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(yearly['year'], yearly['n_records'], marker='o', linewidth=2, color='#d62728')
ax.fill_between(yearly['year'], yearly['n_records'], alpha=0.15, color='#d62728')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Records')
ax.set_title('Annual Cloud-Seeding Activity in the U.S. (2000–2025)', fontsize=13, fontweight='bold')
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.set_xlim(1999.5, 2025.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_annual_trend.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# Figure 2b: Annual by state (stacked area, top 5 states)
top5_states = state_counts['state'].head(5).tolist()
yearly_state = df[df['state'].isin(top5_states)].groupby(['year', 'state']).size().unstack(fill_value=0)
yearly_state = yearly_state.reindex(range(2000, 2026), fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
yearly_state.plot.area(ax=ax, alpha=0.7, colormap='Set2')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Records')
ax.set_title('Annual Cloud-Seeding Activity by Top 5 States (2000–2025)', fontsize=13, fontweight='bold')
ax.legend(title='State', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.set_xlim(2000, 2025)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2b_annual_by_state.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2b saved.")

# ──────────────────────────────────────────────
# 4. PURPOSE COMPOSITION (Figure 3)
# ──────────────────────────────────────────────
# Normalize purpose into broader categories
def categorize_purpose(p):
    p = p.lower().strip()
    if 'suppress hail' in p and 'increase precipitation' in p:
        return 'Hail Suppression + Precipitation Enhancement'
    elif 'suppress hail' in p:
        return 'Hail Suppression'
    elif 'augment snowpack' in p and 'increase precipitation' in p:
        return 'Snowpack Augmentation + Precipitation Enhancement'
    elif 'augment snowpack' in p and 'increase runoff' in p:
        return 'Snowpack Augmentation + Runoff Enhancement'
    elif 'augment snowpack' in p and 'suppress fog' in p:
        return 'Snowpack Augmentation + Fog Suppression'
    elif 'augment snowpack' in p and 'research' in p:
        return 'Snowpack Augmentation + Research'
    elif 'augment snowpack' in p:
        return 'Snowpack Augmentation'
    elif 'increase precipitation' in p and 'increase runoff' in p:
        return 'Precipitation + Runoff Enhancement'
    elif 'increase precipitation' in p:
        return 'Precipitation Enhancement'
    elif 'suppress fog' in p:
        return 'Fog Suppression'
    elif 'research' in p:
        return 'Research'
    else:
        return 'Other'

df['purpose_category'] = df['purpose'].apply(categorize_purpose)
purpose_cat = df['purpose_category'].value_counts().reset_index()
purpose_cat.columns = ['purpose_category', 'n_records']
purpose_cat['pct'] = (purpose_cat['n_records'] / purpose_cat['n_records'].sum() * 100).round(2)
save_json(purpose_cat.to_dict(orient='records'), 'purpose_composition.json')

# Figure 3a: Pie chart of purpose categories
fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    purpose_cat['n_records'], labels=purpose_cat['purpose_category'],
    autopct='%1.1f%%', startangle=140, pctdistance=0.8,
    colors=sns.color_palette('Set3', len(purpose_cat)))
for t in autotexts:
    t.set_fontsize(8)
for t in texts:
    t.set_fontsize(8)
ax.set_title('Stated Purpose Composition of U.S. Cloud-Seeding Projects (2000–2025)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3a_purpose_pie.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 3a saved.")

# Figure 3b: Purpose by season
purpose_season = df.groupby(['purpose_category', 'season']).size().unstack(fill_value=0)
# Simplify season labels
def simplify_season(s):
    s = s.lower().strip()
    if s == 'winter':
        return 'Winter'
    elif s == 'summer':
        return 'Summer'
    elif 'winter' in s and 'spring' in s and 'summer' in s and 'fall' in s:
        return 'Year-round'
    elif 'spring' in s and 'summer' in s and 'fall' in s:
        return 'Spring–Fall'
    elif 'winter' in s and 'spring' in s:
        return 'Winter–Spring'
    elif 'spring' in s and 'summer' in s:
        return 'Spring–Summer'
    elif 'summer' in s and 'fall' in s:
        return 'Summer–Fall'
    elif 'fall' in s and 'winter' in s:
        return 'Fall–Winter'
    elif 'spring' in s:
        return 'Spring'
    elif 'fall' in s:
        return 'Fall'
    else:
        return 'Other'

df['season_simplified'] = df['season'].apply(simplify_season)
purpose_season_simple = df.groupby(['purpose_category', 'season_simplified']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
purpose_season_simple.plot(kind='barh', stacked=True, ax=ax, colormap='Set2')
ax.set_xlabel('Number of Records')
ax.set_ylabel('Purpose Category')
ax.set_title('Purpose Composition by Season (2000–2025)', fontsize=13, fontweight='bold')
ax.legend(title='Season', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3b_purpose_by_season.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 3b saved.")

# ──────────────────────────────────────────────
# 5. AGENT–APPARATUS DEPLOYMENT PATTERNS (Figure 4)
# ──────────────────────────────────────────────
# Simplify agent into primary agent type
def simplify_agent(a):
    a = a.lower().strip()
    if 'silver iodide' in a and 'hygroscopic' in a:
        return 'Silver Iodide + Hygroscopic'
    elif 'silver iodide' in a and 'sodium iodide' in a:
        return 'Silver Iodide + Sodium Iodide'
    elif 'silver iodide' in a and 'ammonium iodide' in a:
        return 'Silver Iodide + Ammonium Iodide'
    elif 'silver iodide' in a and 'calcium chloride' in a:
        return 'Silver Iodide + Calcium Chloride'
    elif 'silver iodide' in a and 'dry ice' in a:
        return 'Silver Iodide + Dry Ice'
    elif 'silver iodide' in a and 'acetone' in a:
        return 'Silver Iodide + Acetone'
    elif 'silver iodide' in a:
        return 'Silver Iodide (pure)'
    elif 'ionized air' in a:
        return 'Ionized Air'
    elif 'calcium chloride' in a:
        return 'Calcium Chloride'
    elif 'carbon dioxide' in a:
        return 'Carbon Dioxide'
    elif 'sodium chloride' in a:
        return 'Sodium Chloride'
    elif 'hygroscopic' in a:
        return 'Hygroscopic Agents'
    else:
        return 'Other'

df['agent_simplified'] = df['agent'].apply(simplify_agent)
agent_apparatus = df.groupby(['agent_simplified', 'apparatus']).size().unstack(fill_value=0)
save_json(agent_apparatus.to_dict(), 'agent_apparatus_matrix.json')

# Figure 4a: Heatmap of agent × apparatus
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(agent_apparatus, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('Seeding Agent × Deployment Apparatus Cross-Tabulation (2000–2025)', fontsize=12, fontweight='bold')
ax.set_xlabel('Apparatus')
ax.set_ylabel('Seeding Agent (Simplified)')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4a_agent_apparatus_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 4a saved.")

# Figure 4b: Apparatus distribution over time
apparatus_year = df.groupby(['year', 'apparatus']).size().unstack(fill_value=0)
apparatus_year = apparatus_year.reindex(range(2000, 2026), fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
apparatus_year.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Records')
ax.set_title('Deployment Apparatus Composition by Year (2000–2025)', fontsize=13, fontweight='bold')
ax.legend(title='Apparatus')
# Reduce x-tick labels
ax.set_xticks(range(0, 26, 2))
ax.set_xticklabels(range(2000, 2026, 2), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4b_apparatus_by_year.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 4b saved.")

# Figure 4c: Agent composition over time (top agents)
top_agents = df['agent_simplified'].value_counts().head(6).index.tolist()
agent_year = df[df['agent_simplified'].isin(top_agents)].groupby(['year', 'agent_simplified']).size().unstack(fill_value=0)
agent_year = agent_year.reindex(range(2000, 2026), fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
agent_year.plot(ax=ax, marker='o', linewidth=1.5, colormap='tab10')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Records')
ax.set_title('Top Seeding Agent Types Over Time (2000–2025)', fontsize=13, fontweight='bold')
ax.legend(title='Agent', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.set_xlim(1999.5, 2025.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4c_agent_over_time.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 4c saved.")

# ──────────────────────────────────────────────
# 6. OPERATOR ANALYSIS (Figure 5)
# ──────────────────────────────────────────────
operator_counts = df['operator_affiliation'].value_counts().head(10).reset_index()
operator_counts.columns = ['operator', 'n_records']
save_json(operator_counts.to_dict(orient='records'), 'top_operators.json')

fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette('Blues_d', len(operator_counts))
bars = ax.barh(operator_counts['operator'][::-1], operator_counts['n_records'][::-1], color=colors[::-1])
for bar, val in zip(bars, operator_counts['n_records'][::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, str(val),
            va='center', fontsize=9)
ax.set_xlabel('Number of Records')
ax.set_title('Top 10 Operator Affiliations (2000–2025)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig5_top_operators.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# ──────────────────────────────────────────────
# 7. SEASONAL PATTERNS (Figure 6)
# ──────────────────────────────────────────────
season_order = ['Winter', 'Winter–Spring', 'Spring', 'Spring–Summer', 'Summer',
                'Summer–Fall', 'Fall', 'Fall–Winter', 'Spring–Fall', 'Year-round', 'Other']
season_counts = df['season_simplified'].value_counts().reindex(season_order, fill_value=0)
save_json(season_counts.to_dict(), 'seasonal_patterns.json')

fig, ax = plt.subplots(figsize=(10, 5))
season_counts.plot(kind='bar', ax=ax, color=sns.color_palette('coolwarm', len(season_counts)))
ax.set_xlabel('Season')
ax.set_ylabel('Number of Records')
ax.set_title('Seasonal Distribution of Cloud-Seeding Activities (2000–2025)', fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig6_seasonal.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ──────────────────────────────────────────────
# 8. CROSS-TABULATIONS FOR REPORT TABLES
# ──────────────────────────────────────────────

# Table: State × Purpose
state_purpose = df.groupby(['state', 'purpose_category']).size().unstack(fill_value=0)
state_purpose.to_csv(os.path.join(OUT_DIR, 'table_state_purpose.csv'))

# Table: State × Apparatus
state_apparatus = df.groupby(['state', 'apparatus']).size().unstack(fill_value=0)
state_apparatus.to_csv(os.path.join(OUT_DIR, 'table_state_apparatus.csv'))

# Table: Year × Purpose Category
year_purpose = df.groupby(['year', 'purpose_category']).size().unstack(fill_value=0)
year_purpose.to_csv(os.path.join(OUT_DIR, 'table_year_purpose.csv'))

# Table: Agent × Apparatus
agent_apparatus.to_csv(os.path.join(OUT_DIR, 'table_agent_apparatus.csv'))

# Summary statistics
summary = {
    "total_records": len(df),
    "year_range": [int(df['year'].min()), int(df['year'].max())],
    "peak_year": int(yearly.loc[yearly['n_records'].idxmax(), 'year']),
    "peak_year_records": int(yearly['n_records'].max()),
    "min_year": int(yearly.loc[yearly['n_records'].idxmin(), 'year']),
    "min_year_records": int(yearly['n_records'].min()),
    "top_state": state_counts.iloc[0]['state'],
    "top_state_records": int(state_counts.iloc[0]['n_records']),
    "top_state_pct": float(state_counts.iloc[0]['pct']),
    "top3_states_pct": float(state_counts.head(3)['pct'].sum()),
    "top5_states_pct": float(state_counts.head(5)['pct'].sum()),
    "dominant_purpose": purpose_cat.iloc[0]['purpose_category'],
    "dominant_purpose_pct": float(purpose_cat.iloc[0]['pct']),
    "dominant_agent": df['agent_simplified'].value_counts().index[0],
    "dominant_agent_pct": round(float(df['agent_simplified'].value_counts().iloc[0] / len(df) * 100), 2),
    "dominant_apparatus": df['apparatus'].value_counts().index[0],
    "dominant_apparatus_pct": round(float(df['apparatus'].value_counts().iloc[0] / len(df) * 100), 2),
    "winter_dominant_pct": round(float(season_counts.get('Winter', 0) / len(df) * 100), 2),
    "mean_annual_records": round(float(yearly['n_records'].mean()), 1),
    "std_annual_records": round(float(yearly['n_records'].std()), 1),
}
save_json(summary, 'summary_statistics.json')
print("Summary statistics:", json.dumps(summary, indent=2))

# ──────────────────────────────────────────────
# 9. ADDITIONAL: State-level concentration metrics
# ──────────────────────────────────────────────
# Herfindahl-Hirschman Index for spatial concentration
state_shares = state_counts['n_records'] / state_counts['n_records'].sum()
hhi = float((state_shares ** 2).sum())
summary['spatial_hhi'] = round(hhi, 4)

# Effective number of states
ens = 1 / hhi
summary['effective_n_states'] = round(ens, 2)

save_json(summary, 'summary_statistics.json')
print(f"Spatial HHI: {hhi:.4f}, Effective N states: {ens:.2f}")

# ──────────────────────────────────────────────
# 10. FIGURE 7: State × Year heatmap for top states
# ──────────────────────────────────────────────
top_states_list = state_counts['state'].head(8).tolist()
state_year = df[df['state'].isin(top_states_list)].groupby(['state', 'year']).size().unstack(fill_value=0)
state_year = state_year.reindex(columns=range(2000, 2026), fill_value=0)

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(state_year, cmap='YlOrRd', annot=False, ax=ax, linewidths=0.3,
            cbar_kws={'label': 'Number of Records'})
ax.set_title('Annual Cloud-Seeding Records by State (Top 8 States, 2000–2025)', fontsize=13, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('State')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_state_year_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

print("\n=== ALL ANALYSIS COMPLETE ===")
