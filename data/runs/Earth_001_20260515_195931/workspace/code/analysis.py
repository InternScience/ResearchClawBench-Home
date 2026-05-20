#!/usr/bin/env python3
"""
Comprehensive analysis of NOAA cloud-seeding records (2000–2025)
U.S. weather modification activities: spatial, temporal, purpose, and agent-apparatus patterns.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import Counter
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUT_DIR = 'outputs'
IMG_DIR = 'report/images'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================
print("=== Loading data ===")
df = pd.read_csv('data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv')
print(f"Raw records: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Strip whitespace from string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Normalize state names to title case
df['state'] = df['state'].str.title()

# Parse year as integer
df['year'] = df['year'].astype(int)

# Create primary agent (first listed agent)
df['primary_agent'] = df['agent'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) else x)

# Create agent count
df['agent_count'] = df['agent'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)

# Normalize apparatus
df['apparatus_clean'] = df['apparatus'].str.lower().str.strip()

# Create apparatus categories
def classify_apparatus(app):
    if pd.isna(app):
        return 'Unknown'
    app = app.lower().strip()
    if 'ground' in app and 'airborne' in app:
        return 'Ground + Airborne'
    elif 'ground' in app:
        return 'Ground-based'
    elif 'airborne' in app:
        return 'Airborne'
    else:
        return app.title()

df['apparatus_category'] = df['apparatus'].apply(classify_apparatus)

# Parse purpose into individual items
df['purpose_list'] = df['purpose'].apply(lambda x: [p.strip() for p in x.split(',')] if pd.notna(x) else [])

# Create primary purpose
df['primary_purpose'] = df['purpose'].apply(lambda x: x.split(',')[0].strip() if pd.notna(x) else x)

print(f"\nYear range: {df['year'].min()} – {df['year'].max()}")
print(f"States: {df['state'].nunique()}")
print(f"Unique projects: {df['project'].nunique()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ============================================================
# 2. SUMMARY STATISTICS
# ============================================================
print("\n=== Computing summary statistics ===")

# Total records
total_records = len(df)

# State distribution
state_counts = df['state'].value_counts()
top_states = state_counts.head(10)

# Year distribution
year_counts = df['year'].value_counts().sort_index()

# Season distribution
season_counts = df['season'].value_counts()

# Purpose breakdown (explode multi-purpose)
all_purposes = []
for purposes in df['purpose_list']:
    all_purposes.extend(purposes)
purpose_counts = Counter(all_purposes)

# Agent distribution
agent_counts = df['primary_agent'].value_counts()

# Apparatus distribution
apparatus_counts = df['apparatus_category'].value_counts()

# Operator distribution
operator_counts = df['operator_affiliation'].value_counts()

print(f"\nTotal records: {total_records}")
print(f"\nTop 10 states by project count:")
for state, count in top_states.items():
    print(f"  {state}: {count}")

print(f"\nYear range: {df['year'].min()}-{df['year'].max()}")
print(f"\nSeasons:")
for season, count in season_counts.items():
    print(f"  {season}: {count}")

print(f"\nPrimary purposes:")
for purpose, count in purpose_counts.most_common():
    print(f"  {purpose}: {count}")

print(f"\nPrimary agents:")
for agent, count in agent_counts.items():
    print(f"  {agent}: {count}")

print(f"\nApparatus categories:")
for app, count in apparatus_counts.items():
    print(f"  {app}: {count}")

# ============================================================
# 3. SAVE SUMMARY TABLES
# ============================================================
print("\n=== Saving summary tables ===")

# Table 1: State summary
table1 = pd.DataFrame({
    'State': state_counts.index,
    'Project Count': state_counts.values,
    'Percentage': (state_counts.values / total_records * 100).round(1)
})
table1.to_csv(f'{OUT_DIR}/table1_state_distribution.csv', index=False)
print("Table 1 saved: table1_state_distribution.csv")

# Table 2: Annual activity
table2 = pd.DataFrame({
    'Year': year_counts.index,
    'Project Count': year_counts.values
})
table2.to_csv(f'{OUT_DIR}/table2_annual_activity.csv', index=False)
print("Table 2 saved: table2_annual_activity.csv")

# Table 3: Purpose composition
table3 = pd.DataFrame({
    'Purpose': [p for p, c in purpose_counts.most_common()],
    'Count': [c for p, c in purpose_counts.most_common()],
    'Percentage': [round(c/total_records*100, 1) for p, c in purpose_counts.most_common()]
})
table3.to_csv(f'{OUT_DIR}/table3_purpose_composition.csv', index=False)
print("Table 3 saved: table3_purpose_composition.csv")

# Table 4: Agent-apparatus cross-tabulation
cross_tab = pd.crosstab(df['primary_agent'], df['apparatus_category'])
cross_tab.to_csv(f'{OUT_DIR}/table4_agent_apparatus.csv')
print("Table 4 saved: table4_agent_apparatus.csv")

# Table 5: Season distribution
table5 = pd.DataFrame({
    'Season': season_counts.index,
    'Count': season_counts.values,
    'Percentage': (season_counts.values / total_records * 100).round(1)
})
table5.to_csv(f'{OUT_DIR}/table5_season_distribution.csv', index=False)
print("Table 5 saved: table5_season_distribution.csv")

# Table 6: Top operators
table6 = pd.DataFrame({
    'Operator': operator_counts.head(10).index,
    'Project Count': operator_counts.head(10).values,
    'Percentage': (operator_counts.head(10).values / total_records * 100).round(1)
})
table6.to_csv(f'{OUT_DIR}/table6_top_operators.csv', index=False)
print("Table 6 saved: table6_top_operators.csv")

# ============================================================
# 4. GENERATE FIGURES
# ============================================================
print("\n=== Generating figures ===")

# --- Figure 1: State-level geographic distribution (bar chart) ---
fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette("viridis", len(top_states))
bars = ax.barh(top_states.index[::-1], top_states.values[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Number of Cloud-Seeding Projects', fontsize=12)
ax.set_ylabel('State', fontsize=12)
ax.set_title('Figure 1: Geographic Distribution of Cloud-Seeding Projects by U.S. State (2000–2025)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, top_states.values[::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=10)
ax.set_xlim(0, top_states.max() * 1.15)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure1_state_distribution.png')
plt.close()
print("Figure 1 saved")

# --- Figure 2: Annual activity time series ---
fig, ax = plt.subplots(figsize=(14, 5))
years = year_counts.index.values
counts = year_counts.values
ax.plot(years, counts, 'o-', color='#2171b5', linewidth=2, markersize=6, markerfacecolor='white', markeredgecolor='#2171b5', markeredgewidth=2)
ax.fill_between(years, counts, alpha=0.15, color='#2171b5')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Reported Projects', fontsize=12)
ax.set_title('Figure 2: Annual Cloud-Seeding Activity in the United States (2000–2025)', fontsize=13, fontweight='bold')
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45, fontsize=9)
ax.set_xlim(years.min()-0.5, years.max()+0.5)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure2_annual_activity.png')
plt.close()
print("Figure 2 saved")

# --- Figure 3: Seasonal distribution (pie chart) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart for seasons
season_labels = season_counts.index.tolist()
season_values = season_counts.values
colors_pie = ['#2166ac', '#67a9cf', '#fddbc7', '#ef8a62', '#b2182b', '#d6604d']
# Extend colors if needed
while len(colors_pie) < len(season_labels):
    colors_pie.append('#999999')
wedges, texts, autotexts = ax1.pie(season_values, labels=season_labels, autopct='%1.1f%%', 
                                     colors=colors_pie[:len(season_labels)], startangle=90,
                                     textprops={'fontsize': 11})
ax1.set_title('Seasonal Distribution', fontsize=13, fontweight='bold')

# Horizontal bar for seasons
bars2 = ax2.barh(season_labels[::-1], season_values[::-1], color=colors_pie[:len(season_labels)][::-1], edgecolor='white')
ax2.set_xlabel('Number of Projects', fontsize=12)
ax2.set_title('Season Distribution by Count', fontsize=13, fontweight='bold')
for bar, val in zip(bars2, season_values[::-1]):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=10)

plt.suptitle('Figure 3: Seasonal Distribution of Cloud-Seeding Projects (2000–2025)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure3_seasonal_distribution.png')
plt.close()
print("Figure 3 saved")

# --- Figure 4: Purpose composition ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top purposes as horizontal bar
top_purposes = purpose_counts.most_common(8)
purp_labels = [p for p, c in top_purposes]
purp_values = [c for p, c in top_purposes]
colors_purpose = sns.color_palette("RdYlBu_r", len(purp_labels))
bars3 = ax1.barh(purp_labels[::-1], purp_values[::-1], color=colors_purpose[::-1], edgecolor='white')
ax1.set_xlabel('Number of Mentions', fontsize=12)
ax1.set_title('Purpose Frequency (Multi-label)', fontsize=13, fontweight='bold')
for bar, val in zip(bars3, purp_values[::-1]):
    ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=10)

# Primary purpose by year (stacked area)
purpose_by_year = pd.crosstab(df['year'], df['primary_purpose'])
purpose_by_year_pct = purpose_by_year.div(purpose_by_year.sum(axis=1), axis=0) * 100
top_primary = purpose_by_year_pct.sum().nlargest(4).index
purpose_by_year_top = purpose_by_year_pct[top_primary]
purpose_by_year_top.plot.area(ax=ax2, colormap='Set2', alpha=0.8)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Primary Purpose Composition Over Time', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_xlim(purpose_by_year_top.index.min(), purpose_by_year_top.index.max())

plt.suptitle('Figure 4: Purpose Composition of Cloud-Seeding Projects (2000–2025)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure4_purpose_composition.png')
plt.close()
print("Figure 4 saved")

# --- Figure 5: Agent and Apparatus patterns ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 5a: Agent distribution
ax = axes[0, 0]
agent_top = agent_counts.head(8)
bars4 = ax.barh(agent_top.index[::-1], agent_top.values[::-1], color=sns.color_palette("Blues_r", len(agent_top))[::-1], edgecolor='white')
ax.set_xlabel('Number of Projects', fontsize=11)
ax.set_title('(a) Seeding Agent Distribution', fontsize=12, fontweight='bold')
for bar, val in zip(bars4, agent_top.values[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)

# 5b: Apparatus distribution
ax = axes[0, 1]
app_colors = {'Ground-based': '#2166ac', 'Airborne': '#ef8a62', 'Ground + Airborne': '#67a9cf', 'Unknown': '#cccccc'}
app_labels = apparatus_counts.index.tolist()
app_vals = apparatus_counts.values
bars5 = ax.bar(app_labels, app_vals, color=[app_colors.get(l, '#999') for l in app_labels], edgecolor='white', width=0.6)
ax.set_ylabel('Number of Projects', fontsize=11)
ax.set_title('(b) Deployment Apparatus Distribution', fontsize=12, fontweight='bold')
for bar, val in zip(bars5, app_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val), ha='center', fontsize=10)
ax.tick_params(axis='x', rotation=20)

# 5c: Agent × Apparatus heatmap
ax = axes[1, 0]
top_agents_for_heatmap = agent_counts.head(6).index.tolist()
cross_sub = df[df['primary_agent'].isin(top_agents_for_heatmap)]
cross_matrix = pd.crosstab(cross_sub['primary_agent'], cross_sub['apparatus_category'])
sns.heatmap(cross_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5, cbar_kws={'label': 'Count'})
ax.set_title('(c) Agent × Apparatus Cross-Tabulation', fontsize=12, fontweight='bold')
ax.set_ylabel('Seeding Agent', fontsize=11)
ax.set_xlabel('Apparatus', fontsize=11)
ax.tick_params(axis='x', rotation=25)
ax.tick_params(axis='y', rotation=0)

# 5d: Purpose by apparatus (stacked bar)
ax = axes[1, 1]
purpose_app = pd.crosstab(df['apparatus_category'], df['primary_purpose'], normalize='index') * 100
top_4_purp = purpose_by_year.sum().nlargest(4).index.tolist()
purpose_app_top = purpose_app[[p for p in top_4_purp if p in purpose_app.columns]]
purpose_app_top.plot.bar(stacked=True, ax=ax, colormap='Set2', edgecolor='white')
ax.set_ylabel('Percentage (%)', fontsize=11)
ax.set_title('(d) Purpose Composition by Apparatus', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
ax.tick_params(axis='x', rotation=20)

plt.suptitle('Figure 5: Seeding Agent and Deployment Apparatus Patterns (2000–2025)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure5_agent_apparatus_patterns.png')
plt.close()
print("Figure 5 saved")

# --- Figure 6: Heatmap of activity by state and year (top states) ---
fig, ax = plt.subplots(figsize=(16, 7))
top_n_states = state_counts.head(8).index.tolist()
state_year = df[df['state'].isin(top_n_states)].groupby(['state', 'year']).size().unstack(fill_value=0)
# Reorder by total
state_year = state_year.loc[state_year.sum(axis=1).sort_values(ascending=False).index]
sns.heatmap(state_year, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=0.5, 
            cbar_kws={'label': 'Number of Projects'}, annot_kws={'size': 9})
ax.set_title('Figure 6: Cloud-Seeding Activity by State and Year (Top 8 States)', fontsize=13, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('State', fontsize=12)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure6_state_year_heatmap.png')
plt.close()
print("Figure 6 saved")

# --- Figure 7: Agent × Year trends ---
fig, ax = plt.subplots(figsize=(14, 6))
top_5_agents = agent_counts.head(5).index.tolist()
for agent in top_5_agents:
    subset = df[df['primary_agent'] == agent]
    agent_year = subset.groupby('year').size()
    ax.plot(agent_year.index, agent_year.values, 'o-', label=agent, linewidth=2, markersize=5)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Projects', fontsize=12)
ax.set_title('Figure 7: Temporal Trends by Seeding Agent (2000–2025)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(years.min()-0.5, years.max()+0.5)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure7_agent_temporal_trends.png')
plt.close()
print("Figure 7 saved")

# --- Figure 8: State × Purpose heatmap ---
fig, ax = plt.subplots(figsize=(12, 7))
top_8_states = state_counts.head(8).index.tolist()
state_purpose = df[df['state'].isin(top_8_states)].groupby(['state', 'primary_purpose']).size().unstack(fill_value=0)
# Sort by total
state_purpose = state_purpose.loc[state_purpose.sum(axis=1).sort_values(ascending=False).index]
sns.heatmap(state_purpose, annot=True, fmt='d', cmap='RdYlBu_r', ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Number of Projects'}, annot_kws={'size': 9})
ax.set_title('Figure 8: Purpose Distribution Across Top States', fontsize=13, fontweight='bold')
ax.set_xlabel('Primary Purpose', fontsize=12)
ax.set_ylabel('State', fontsize=12)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/figure8_state_purpose_heatmap.png')
plt.close()
print("Figure 8 saved")

# ============================================================
# 5. COMPREHENSIVE DATA SUMMARY JSON
# ============================================================
print("\n=== Saving data summary ===")

summary = {
    'total_records': int(total_records),
    'year_range': [int(df['year'].min()), int(df['year'].max())],
    'n_states': int(df['state'].nunique()),
    'n_unique_projects': int(df['project'].nunique()),
    'n_operators': int(df['operator_affiliation'].nunique()),
    'state_distribution': {k: int(v) for k, v in state_counts.head(10).items()},
    'annual_activity': {int(k): int(v) for k, v in year_counts.items()},
    'season_distribution': {k: int(v) for k, v in season_counts.items()},
    'purpose_frequency': {k: int(v) for k, v in purpose_counts.most_common()},
    'agent_distribution': {k: int(v) for k, v in agent_counts.items()},
    'apparatus_distribution': {k: int(v) for k, v in apparatus_counts.items()},
    'top_operators': {k: int(v) for k, v in operator_counts.head(10).items()}
}

with open(f'{OUT_DIR}/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved: data_summary.json")

print("\n=== Analysis complete ===")
print(f"Outputs saved to: {OUT_DIR}/")
print(f"Figures saved to: {IMG_DIR}/")
