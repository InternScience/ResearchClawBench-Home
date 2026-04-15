#!/usr/bin/env python3
"""
Generate all figures for the cloud seeding research report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os

# ---- Paths ----
OUTPUTS_DIR = 'outputs'
IMAGES_DIR = 'report/images'
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load pre-computed tables
state_counts = pd.read_csv(os.path.join(OUTPUTS_DIR, 'state_concentration.csv'))
annual_counts = pd.read_csv(os.path.join(OUTPUTS_DIR, 'annual_activity.csv'))
purpose_counts = pd.read_csv(os.path.join(OUTPUTS_DIR, 'purpose_composition.csv'))
agent_counts = pd.read_csv(os.path.join(OUTPUTS_DIR, 'agent_distribution.csv'))
apparatus_counts = pd.read_csv(os.path.join(OUTPUTS_DIR, 'apparatus_distribution.csv'))
agent_apparatus_ct = pd.read_csv(os.path.join(OUTPUTS_DIR, 'agent_apparatus_crosstab.csv'), index_col=0)
season_simple = pd.read_csv(os.path.join(OUTPUTS_DIR, 'season_simple_distribution.csv'))
annual_state = pd.read_csv(os.path.join(OUTPUTS_DIR, 'annual_by_state.csv'))
annual_purpose = pd.read_csv(os.path.join(OUTPUTS_DIR, 'annual_by_purpose.csv'))

# Reload raw data for some plots
df = pd.read_csv('data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv')
df['year'] = df['year'].astype(int)

# Color palette
palette = sns.color_palette("Set2", 10)
palette_dark = sns.color_palette("dark", 10)

# ============================================================
# FIGURE 1: Spatial Concentration - Horizontal Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
state_sorted = state_counts.sort_values('count', ascending=True)
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(state_sorted)))
bars = ax.barh(state_sorted['state'], state_sorted['count'], color=colors, edgecolor='gray', linewidth=0.5)

for bar, pct in zip(bars, state_sorted['pct']):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Number of Project Records', fontsize=12)
ax.set_title('Spatial Concentration of US Cloud-Seeding Projects by State (2000-2025)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig1_spatial_concentration.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig1_spatial_concentration.png")

# ============================================================
# FIGURE 2: Annual Activity Dynamics
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

# Main: annual counts with trend line
ax1 = axes[0]
ax1.bar(annual_counts['year'], annual_counts['count'], color='steelblue', edgecolor='white', linewidth=0.5, alpha=0.8)
# Rolling average
rolling = annual_counts['count'].rolling(window=5, center=True).mean()
ax1.plot(annual_counts['year'], rolling, 'r-', linewidth=2.5, label='5-year rolling mean', marker='o', markersize=4)
ax1.axhline(annual_counts['count'].mean(), color='gray', linestyle='--', linewidth=1.5, label=f'Mean ({annual_counts["count"].mean():.0f}/yr)')
ax1.set_ylabel('Number of Projects', fontsize=12)
ax1.set_title('Annual Cloud-Seeding Activity in the United States (2000-2025)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xlim(1999.5, 2025.5)

# Bottom: stacked by top states
ax2 = axes[1]
top_states = state_counts.head(6)['state'].tolist()
pivot = annual_state[annual_state['state'].isin(top_states)].pivot(index='year', columns='state', values='count').fillna(0)
pivot = pivot.reindex(columns=top_states)
pivot.plot(kind='bar', stacked=True, ax=ax2, colormap='YlOrRd', width=0.8, edgecolor='none')
ax2.set_ylabel('Projects', fontsize=11)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_title('Annual Activity by Top 6 States', fontsize=12, fontweight='bold')
ax2.legend(title='State', fontsize=8, loc='upper right', ncol=2)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig2_annual_trends.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig2_annual_trends.png")

# ============================================================
# FIGURE 3: Purpose Composition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Donut chart for primary purposes
ax1 = axes[0]
purp_top = purpose_counts.head(6)
colors_donut = plt.cm.Set3(np.linspace(0.1, 0.9, len(purp_top)))
wedges, texts, autotexts = ax1.pie(purp_top['count'], labels=purp_top['purpose_primary'],
                                     autopct='%1.1f%%', colors=colors_donut,
                                     startangle=90, pctdistance=0.75,
                                     textprops={'fontsize': 9},
                                     wedgeprops=dict(width=0.5, edgecolor='white'))
for t in autotexts:
    t.set_fontweight('bold')
    t.set_fontsize(9)
ax1.set_title('Purpose Composition (Top Categories)', fontsize=13, fontweight='bold')

# Right: Purpose over time
ax2 = axes[1]
purp_top_names = purp_top['purpose_primary'].tolist()
pivot_purp = annual_purpose[annual_purpose['purpose'].isin(purp_top_names)].pivot(
    index='year', columns='purpose', values='count').fillna(0)
pivot_purp = pivot_purp.reindex(columns=purp_top_names)
pivot_purp.plot(kind='area', stacked=True, ax=ax2, colormap='Set2', alpha=0.8)
ax2.set_ylabel('Projects', fontsize=11)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_title('Purpose Trends Over Time', fontsize=12, fontweight='bold')
ax2.legend(fontsize=7, loc='upper left')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig3_purpose_composition.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_purpose_composition.png")

# ============================================================
# FIGURE 4: Agent-Apparatus Deployment Heatmap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Agent x Apparatus heatmap
ax1 = axes[0]
aa_data = agent_apparatus_ct.copy()
# Remove rows/cols with all zeros
aa_data = aa_data.loc[(aa_data != 0).any(axis=1)]
aa_data = aa_data.loc[:, (aa_data != 0).any(axis=0)]
sns.heatmap(aa_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Count'},
            linewidths=0.5, linecolor='white')
ax1.set_title('Seeding Agent × Deployment Apparatus', fontsize=13, fontweight='bold')
ax1.set_xlabel('')
ax1.set_ylabel('')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax1.get_yticklabels(), fontsize=9)

# Right: Agent distribution bar chart
ax2 = axes[1]
ag_top = agent_counts.head(8)
colors_bar = plt.cm.GnBu(np.linspace(0.3, 0.9, len(ag_top)))
bars = ax2.barh(ag_top['agent_simplified'][::-1], ag_top['count'][::-1], color=colors_bar[::-1], edgecolor='gray', linewidth=0.5)
for bar, pct in zip(bars, ag_top['pct'][::-1]):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')
ax2.set_xlabel('Number of Project Records', fontsize=11)
ax2.set_title('Seeding Agent Distribution', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig4_agent_apparatus_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig4_agent_apparatus_heatmap.png")

# ============================================================
# FIGURE 5: Seasonal Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Pie chart
ax1 = axes[0]
season_top = season_simple.head(5)
colors_season = ['#4a90d9', '#e8c840', '#5cb85c', '#d9534f', '#6f42c1']
wedges, texts, autotexts = ax1.pie(season_top['count'], labels=season_top['season_simple'],
                                     autopct='%1.1f%%', colors=colors_season[:len(season_top)],
                                     startangle=90, pctdistance=0.75,
                                     textprops={'fontsize': 10},
                                     wedgeprops=dict(width=0.5, edgecolor='white'))
for t in autotexts:
    t.set_fontweight('bold')
    t.set_fontsize(10)
ax1.set_title('Seasonal Distribution (Primary Season)', fontsize=13, fontweight='bold')

# Right: Season by state
ax2 = axes[1]
df['season_simple'] = df['season'].apply(lambda s: str(s).lower().replace(' ', '').split(',')[0])
top_states_for_season = state_counts.head(8)['state'].tolist()
season_state = df[df['state'].isin(top_states_for_season)].groupby(['state', 'season_simple']).size().reset_index(name='count')
pivot_ss = season_state.pivot(index='state', columns='season_simple', values='count').fillna(0)
# Reorder by total
pivot_ss['total'] = pivot_ss.sum(axis=1)
pivot_ss = pivot_ss.sort_values('total', ascending=True).drop('total', axis=1)
pivot_ss.plot(kind='barh', stacked=True, ax=ax2, colormap='viridis', edgecolor='none')
ax2.set_xlabel('Projects', fontsize=11)
ax2.set_title('Season Distribution by Top 8 States', fontsize=13, fontweight='bold')
ax2.legend(title='Season', fontsize=8)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig5_seasonal_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig5_seasonal_distribution.png")

# ============================================================
# FIGURE 6: State-Year Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
top_states_heat = state_counts.head(10)['state'].tolist()
heat_data = annual_state[annual_state['state'].isin(top_states_heat)].pivot(
    index='state', columns='year', values='count').fillna(0)
# Reorder states by total
heat_data['total'] = heat_data.sum(axis=1)
heat_data = heat_data.sort_values('total', ascending=False).drop('total', axis=1)

sns.heatmap(heat_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Number of Projects'},
            linewidths=0.5, linecolor='white')
ax.set_title('Cloud-Seeding Activity by State and Year (2000-2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax.get_yticklabels(), fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig6_state_year_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig6_state_year_heatmap.png")

# ============================================================
# FIGURE 7: Apparatus usage over time
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
df_clean = df[df['apparatus'].notna() & (df['apparatus'] != 'nan')]
app_annual = df_clean.groupby(['year', 'apparatus']).size().reset_index(name='count')
pivot_app = app_annual.pivot(index='year', columns='apparatus', values='count').fillna(0)
pivot_app = pivot_app.reindex(columns=['ground', 'airborne', 'ground, airborne'])
pivot_app.plot(kind='bar', stacked=True, ax=ax, colormap='Blues', width=0.8, edgecolor='none')
ax.set_ylabel('Projects', fontsize=12)
ax.set_xlabel('Year', fontsize=12)
ax.set_title('Deployment Apparatus Usage Over Time', fontsize=14, fontweight='bold')
ax.legend(title='Apparatus', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig7_apparatus_timeline.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig7_apparatus_timeline.png")

# ============================================================
# FIGURE 8: Operator concentration
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
operator_counts = df.groupby('operator_affiliation').size().reset_index(name='count')
operator_counts = operator_counts.sort_values('count', ascending=True).tail(12)
colors_op = plt.cm.Oranges(np.linspace(0.3, 0.9, len(operator_counts)))
bars = ax.barh(operator_counts['operator_affiliation'], operator_counts['count'], color=colors_op, edgecolor='gray', linewidth=0.5)
for bar, count in zip(bars, operator_counts['count']):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{count}', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Number of Project Records', fontsize=12)
ax.set_title('Top 12 Cloud-Seeding Operators (2000-2025)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig8_operator_concentration.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig8_operator_concentration.png")

print("\nAll figures generated successfully.")
