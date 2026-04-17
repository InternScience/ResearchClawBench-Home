#!/usr/bin/env python3
"""
Cloud Seeding Analysis Script
Analyzes NOAA weather-modification records for U.S. cloud-seeding projects 2000-2025
"""

import pandas as pd
import numpy as np
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = 'data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv'
OUTPUTS_DIR = 'outputs'
IMAGES_DIR = 'report/images'

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records")

# Basic data summary
print("\n=== DATA SUMMARY ===")
print(f"Columns: {list(df.columns)}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print(f"\nUnique states: {df['state'].nunique()}")
print(f"States: {sorted(df['state'].unique())}")
print(f"\nUnique purposes: {df['purpose'].nunique()}")
print(f"Purposes: {df['purpose'].unique()}")
print(f"\nUnique agents: {df['agent'].nunique()}")
print(f"Agents: {df['agent'].unique()}")
print(f"\nUnique apparatus: {df['apparatus'].nunique()}")
print(f"Apparatus: {df['apparatus'].unique()}")
print(f"\nUnique seasons: {df['season'].nunique()}")
print(f"Seasons: {df['season'].unique()}")

# Save basic summary
summary = {
    'total_records': len(df),
    'year_range': [int(df['year'].min()), int(df['year'].max())],
    'num_states': df['state'].nunique(),
    'states': sorted(df['state'].unique()),
    'num_purposes': df['purpose'].nunique(),
    'purposes': list(df['purpose'].unique()),
    'num_agents': df['agent'].nunique(),
    'agents': list(df['agent'].unique()),
    'num_apparatus': df['apparatus'].nunique(),
    'apparatus': list(df['apparatus'].unique()),
    'num_seasons': df['season'].nunique(),
    'seasons': list(df['season'].unique()),
    'num_operators': df['operator_affiliation'].nunique()
}

with open(f'{OUTPUTS_DIR}/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary to {OUTPUTS_DIR}/data_summary.json")

# ============================================
# ANALYSIS 1: Spatial Concentration (by state)
# ============================================
print("\n=== SPATIAL CONCENTRATION ===")
state_counts = df['state'].value_counts().to_dict()
print("Records by state:")
for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
    print(f"  {state}: {count}")

with open(f'{OUTPUTS_DIR}/state_counts.json', 'w') as f:
    json.dump(state_counts, f, indent=2)

# ============================================
# ANALYSIS 2: Annual Activity Dynamics
# ============================================
print("\n=== ANNUAL ACTIVITY DYNAMICS ===")
year_counts = df['year'].value_counts().sort_index().to_dict()
print("Records by year:")
for year, count in year_counts.items():
    print(f"  {year}: {count}")

with open(f'{OUTPUTS_DIR}/yearly_activity.json', 'w') as f:
    json.dump({str(k): v for k, v in year_counts.items()}, f, indent=2)

# Seasonal patterns by year
year_season = df.groupby(['year', 'season']).size().unstack(fill_value=0)
print("\nYear-Season breakdown:")
print(year_season)

# ============================================
# ANALYSIS 3: Purpose Composition
# ============================================
print("\n=== PURPOSE COMPOSITION ===")
purpose_counts = df['purpose'].value_counts().to_dict()
print("Records by purpose:")
for purpose, count in sorted(purpose_counts.items(), key=lambda x: -x[1]):
    print(f"  {purpose}: {count}")

with open(f'{OUTPUTS_DIR}/purpose_composition.json', 'w') as f:
    json.dump(purpose_counts, f, indent=2)

# Purpose by state
purpose_state = df.groupby(['state', 'purpose']).size().unstack(fill_value=0)
print("\nPurpose by state matrix saved...")
purpose_state.to_csv(f'{OUTPUTS_DIR}/purpose_by_state.csv')

# ============================================
# ANALYSIS 4: Agent-Apparatus Deployment Patterns
# ============================================
print("\n=== AGENT-APPARATUS DEPLOYMENT PATTERNS ===")
agent_apparatus = df.groupby(['agent', 'apparatus']).size().unstack(fill_value=0)
print("Agent-Apparatus matrix:")
print(agent_apparatus)

agent_apparatus_dict = agent_apparatus.to_dict()
with open(f'{OUTPUTS_DIR}/agent_apparatus_matrix.json', 'w') as f:
    # Convert to serializable format
    serializable = {str(k): dict(v) for k, v in agent_apparatus_dict.items()}
    json.dump(serializable, f, indent=2)

# Operator analysis
print("\n=== OPERATOR ANALYSIS ===")
operator_counts = df['operator_affiliation'].value_counts().to_dict()
print("Top operators:")
for op, count in list(operator_counts.items())[:10]:
    print(f"  {op}: {count}")

with open(f'{OUTPUTS_DIR}/operator_counts.json', 'w') as f:
    json.dump(operator_counts, f, indent=2)

# ============================================
# FIGURE GENERATION
# ============================================
print("\n=== GENERATING FIGURES ===")

# Figure 1: Spatial Distribution (Map-like bar chart)
fig, ax = plt.subplots(figsize=(12, 8))
state_order = df['state'].value_counts().index
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(state_order)))
bars = ax.barh(state_order, df['state'].value_counts().values, color=colors)
ax.set_xlabel('Number of Projects')
ax.set_ylabel('State')
ax.set_title('Spatial Concentration of Cloud Seeding Projects (2000-2025)')
ax.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, df['state'].value_counts().values)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
            str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/spatial_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/spatial_distribution.png")

# Figure 2: Annual Activity Trends
fig, ax = plt.subplots(figsize=(12, 6))
years = sorted(year_counts.keys())
counts = [year_counts[y] for y in years]
ax.plot(years, counts, marker='o', linewidth=2, markersize=8, color='steelblue')
ax.fill_between(years, counts, alpha=0.3, color='steelblue')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Projects')
ax.set_title('Annual Cloud Seeding Activity (2000-2025)')
ax.set_xticks(years)
ax.grid(True, alpha=0.3)
for x, y in zip(years, counts):
    ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0,5), ha='center')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/annual_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/annual_trends.png")

# Figure 3: Purpose Composition
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(purpose_counts)))
wedges, texts, autotexts = axes[0].pie(purpose_counts.values(), labels=purpose_counts.keys(),
                                        autopct='%1.1f%%', colors=colors_pie)
axes[0].set_title('Purpose Composition (All Years)')

# Bar chart
purpose_order = sorted(purpose_counts.keys(), key=lambda x: -purpose_counts[x])
axes[1].bar(purpose_order, [purpose_counts[p] for p in purpose_order], color=colors_pie)
axes[1].set_xlabel('Purpose')
axes[1].set_ylabel('Number of Projects')
axes[1].set_title('Purpose Distribution')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/purpose_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/purpose_chart.png")

# Figure 4: Agent-Apparatus Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(agent_apparatus, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
            cbar_kws={'label': 'Number of Projects'})
ax.set_xlabel('Apparatus')
ax.set_ylabel('Agent')
ax.set_title('Agent-Apparatus Deployment Matrix')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/agent_apparatus_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/agent_apparatus_heatmap.png")

# Figure 5: Seasonal Patterns
fig, ax = plt.subplots(figsize=(12, 6))
season_year = df.groupby(['season', 'year']).size().unstack(fill_value=0)
sns.heatmap(season_year, annot=True, fmt='d', cmap='Blues', ax=ax,
            cbar_kws={'label': 'Number of Projects'})
ax.set_xlabel('Year')
ax.set_ylabel('Season')
ax.set_title('Seasonal Activity Patterns by Year')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/seasonal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/seasonal_patterns.png")

# Figure 6: State-Purpose Matrix
fig, ax = plt.subplots(figsize=(14, 10))
# Pivot table for top states only
top_states = df['state'].value_counts().head(15).index
purpose_state_top = df[df['state'].isin(top_states)].groupby(['state', 'purpose']).size().unstack(fill_value=0)
sns.heatmap(purpose_state_top, annot=True, fmt='d', cmap='Greens', ax=ax,
            cbar_kws={'label': 'Number of Projects'})
ax.set_xlabel('Purpose')
ax.set_ylabel('State (Top 15)')
ax.set_title('Purpose Distribution by State')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/state_purpose_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/state_purpose_matrix.png")

# Figure 7: Operator Analysis
fig, ax = plt.subplots(figsize=(12, 8))
top_operators = df['operator_affiliation'].value_counts().head(10)
colors_op = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_operators)))
bars = ax.barh(range(len(top_operators)), top_operators.values, color=colors_op)
ax.set_yticks(range(len(top_operators)))
ax.set_yticklabels([op[:40] + '...' if len(op) > 40 else op for op in top_operators.index])
ax.set_xlabel('Number of Projects')
ax.set_title('Top 10 Operators by Project Count')
ax.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars, top_operators.values)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
            str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/operator_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/operator_analysis.png")

# Figure 8: Year-State Activity (comprehensive temporal-spatial)
fig, ax = plt.subplots(figsize=(16, 10))
# Aggregate by year and state
year_state = df.groupby(['year', 'state']).size().unstack(fill_value=0)
# Only show top states
top_states = df['state'].value_counts().head(12).index
year_state_top = year_state[top_states]
sns.heatmap(year_state_top.T, annot=True, fmt='d', cmap='viridis', ax=ax,
            cbar_kws={'label': 'Number of Projects'})
ax.set_xlabel('Year')
ax.set_ylabel('State (Top 12)')
ax.set_title('Temporal-Spatial Activity Matrix')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/year_state_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {IMAGES_DIR}/year_state_matrix.png")

print("\n=== ANALYSIS COMPLETE ===")
print(f"Output files saved to: {OUTPUTS_DIR}/")
print(f"Figures saved to: {IMAGES_DIR}/")
