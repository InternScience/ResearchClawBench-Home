"""
Cloud Seeding Analysis Script
==============================
Analyzes NOAA weather modification records for U.S. cloud-seeding projects (2000-2025)
Produces reproducible tables and figures for:
- Spatial concentration
- Annual activity dynamics
- Purpose composition
- Agent-apparatus deployment patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv')

print(f"Dataset loaded: {len(df)} records")
print(f"Columns: {list(df.columns)}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# Data cleaning and preparation
df['state'] = df['state'].str.lower().str.strip()
df['purpose'] = df['purpose'].str.lower().str.strip()
df['agent'] = df['agent'].str.lower().str.strip()
df['apparatus'] = df['apparatus'].str.lower().str.strip()
df['operator_affiliation'] = df['operator_affiliation'].str.lower().str.strip()

# Parse season - handle multiple seasons
seasons_expanded = []
for idx, row in df.iterrows():
    seasons = [s.strip() for s in str(row['season']).split(',')]
    for season in seasons:
        new_row = row.copy()
        new_row['season_single'] = season.strip().lower()
        seasons_expanded.append(new_row)

df_expanded = pd.DataFrame(seasons_expanded)

# ===============================
# 1. SPATIAL CONCENTRATION ANALYSIS
# ===============================

# State-level aggregation
state_counts = df['state'].value_counts()
state_pct = (state_counts / len(df) * 100).round(2)

print("\n=== SPATIAL CONCENTRATION ===")
print("Top 10 States by Project Count:")
print(state_counts.head(10))

# Save state summary
state_summary = pd.DataFrame({
    'state': state_counts.index,
    'project_count': state_counts.values,
    'percentage': state_pct.values
})
state_summary.to_csv('outputs/state_concentration.csv', index=False)

# ===============================
# 2. ANNUAL ACTIVITY DYNAMICS
# ===============================

yearly_counts = df['year'].value_counts().sort_index()

print("\n=== ANNUAL ACTIVITY DYNAMICS ===")
print("Projects per year:")
print(yearly_counts)

# Save yearly summary
yearly_summary = pd.DataFrame({
    'year': yearly_counts.index,
    'project_count': yearly_counts.values
})
yearly_summary.to_csv('outputs/annual_dynamics.csv', index=False)

# ===============================
# 3. PURPOSE COMPOSITION
# ===============================

purpose_counts = df['purpose'].value_counts()
purpose_pct = (purpose_counts / len(df) * 100).round(2)

print("\n=== PURPOSE COMPOSITION ===")
print("Top purposes:")
print(purpose_counts.head(10))

purpose_summary = pd.DataFrame({
    'purpose': purpose_counts.index,
    'count': purpose_counts.values,
    'percentage': purpose_pct.values
})
purpose_summary.to_csv('outputs/purpose_composition.csv', index=False)

# ===============================
# 4. AGENT-APPARATUS DEPLOYMENT
# ===============================

# Agent analysis
agent_counts = df['agent'].value_counts()
print("\n=== SEEDING AGENTS ===")
print("Top agents:")
print(agent_counts.head(10))

agent_summary = pd.DataFrame({
    'agent': agent_counts.index,
    'count': agent_counts.values
})
agent_summary.to_csv('outputs/agent_deployment.csv', index=False)

# Apparatus analysis
apparatus_counts = df['apparatus'].value_counts()
print("\n=== DEPLOYMENT APPARATUS ===")
print("Apparatus types:")
print(apparatus_counts)

apparatus_summary = pd.DataFrame({
    'apparatus': apparatus_counts.index,
    'count': apparatus_counts.values
})
apparatus_summary.to_csv('outputs/apparatus_deployment.csv', index=False)

# Agent-Apparatus cross-tabulation
agent_apparatus_crosstab = pd.crosstab(df['agent'], df['apparatus'])
agent_apparatus_crosstab.to_csv('outputs/agent_apparatus_crosstab.csv')

# ===============================
# 5. OPERATOR ANALYSIS
# ===============================

operator_counts = df['operator_affiliation'].value_counts()
print("\n=== OPERATOR AFFILIATIONS ===")
print("Top operators:")
print(operator_counts.head(10))

operator_summary = pd.DataFrame({
    'operator': operator_counts.index,
    'count': operator_counts.values
})
operator_summary.to_csv('outputs/operator_summary.csv', index=False)

# ===============================
# 6. SEASONAL PATTERNS
# ===============================

season_counts = df_expanded['season_single'].value_counts()
print("\n=== SEASONAL PATTERNS ===")
print("Projects by season:")
print(season_counts)

season_summary = pd.DataFrame({
    'season': season_counts.index,
    'count': season_counts.values
})
season_summary.to_csv('outputs/seasonal_patterns.csv', index=False)

# ===============================
# GENERATE FIGURES
# ===============================

# Figure 1: Spatial Concentration Map (Bar Chart)
fig, ax = plt.subplots(figsize=(14, 8))
top_15_states = state_counts.head(15)
bars = ax.bar(range(len(top_15_states)), top_15_states.values, color='steelblue', edgecolor='navy', alpha=0.8)
ax.set_xticks(range(len(top_15_states)))
ax.set_xticklabels([s.title() for s in top_15_states.index], rotation=45, ha='right')
ax.set_xlabel('State', fontsize=12)
ax.set_ylabel('Number of Projects', fontsize=12)
ax.set_title('Spatial Concentration: Top 15 States by Cloud Seeding Projects (2000-2025)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, top_15_states.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val), 
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/fig1_spatial_concentration.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Annual Activity Dynamics (Time Series)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=6, color='darkgreen')
ax.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='green')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Projects', fontsize=12)
ax.set_title('Annual Activity Dynamics: Cloud Seeding Projects (2000-2025)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(2000, 2025)

# Add trend line
z = np.polyfit(yearly_counts.index, yearly_counts.values, 1)
p = np.poly1d(z)
ax.plot(yearly_counts.index, p(yearly_counts.index), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.2f})')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/fig2_annual_dynamics.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Purpose Composition (Pie Chart)
fig, ax = plt.subplots(figsize=(12, 8))
top_purposes = purpose_counts.head(8)
others_count = purpose_counts.iloc[8:].sum()
if others_count > 0:
    plot_purposes = list(top_purposes.index) + ['Other']
    plot_values = list(top_purposes.values) + [others_count]
else:
    plot_purposes = top_purposes.index
    plot_values = top_purposes.values

colors = plt.cm.Set3(np.linspace(0, 1, len(plot_purposes)))
wedges, texts, autotexts = ax.pie(plot_values, labels=[p.title() for p in plot_purposes], 
                                   autopct='%1.1f%%', startangle=90, colors=colors,
                                   textprops={'fontsize': 10})
ax.set_title('Purpose Composition of Cloud Seeding Projects (2000-2025)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig3_purpose_composition.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Agent-Apparatus Deployment (Stacked Bar)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Agents
ax1 = axes[0]
top_agents = agent_counts.head(8)
ax1.barh(range(len(top_agents)), top_agents.values, color='coral', edgecolor='darkred', alpha=0.8)
ax1.set_yticks(range(len(top_agents)))
ax1.set_yticklabels([a.title() for a in top_agents.index])
ax1.set_xlabel('Number of Projects', fontsize=11)
ax1.set_title('Seeding Agent Types', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Apparatus
ax2 = axes[1]
ax2.barh(range(len(apparatus_counts)), apparatus_counts.values, color='skyblue', edgecolor='navy', alpha=0.8)
ax2.set_yticks(range(len(apparatus_counts)))
ax2.set_yticklabels([a.title() for a in apparatus_counts.index])
ax2.set_xlabel('Number of Projects', fontsize=11)
ax2.set_title('Deployment Apparatus Types', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.suptitle('Agent and Apparatus Deployment Patterns (2000-2025)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig4_agent_apparatus.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Seasonal Distribution
fig, ax = plt.subplots(figsize=(10, 6))
season_colors = {'winter': 'steelblue', 'spring': 'lightgreen', 'summer': 'gold', 
                 'fall': 'orange', 'autumn': 'coral', 'year-round': 'purple'}
colors = [season_colors.get(s, 'gray') for s in season_counts.index]
bars = ax.bar(season_counts.index, season_counts.values, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Season', fontsize=12)
ax.set_ylabel('Number of Projects', fontsize=12)
ax.set_title('Seasonal Distribution of Cloud Seeding Projects (2000-2025)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, season_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/fig5_seasonal_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Multi-panel Summary Dashboard
fig = plt.figure(figsize=(16, 12))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top left: Top 10 states
ax1 = fig.add_subplot(gs[0, :2])
top_10 = state_counts.head(10)
bars1 = ax1.barh(range(len(top_10)), top_10.values, color='steelblue', alpha=0.8)
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels([s.title() for s in top_10.index])
ax1.set_xlabel('Number of Projects')
ax1.set_title('Top 10 States by Project Count', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Top right: Key stats
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
KEY STATISTICS

Total Projects: {len(df)}
Year Range: {df['year'].min()}-{df['year'].max()}
States: {df['state'].nunique()}
Operators: {df['operator_affiliation'].nunique()}

Peak Year: {yearly_counts.idxmax()} ({yearly_counts.max()} projects)
Peak State: {state_counts.index[0].title()} ({state_counts.iloc[0]} projects)

Most Common:
• Agent: {agent_counts.index[0].title()}
• Apparatus: {apparatus_counts.index[0].title()}
• Purpose: {purpose_counts.index[0].title()}
• Season: {season_counts.index[0].title()}
"""
ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Middle row: Time series
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=5, color='darkgreen')
ax3.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='green')
ax3.set_xlabel('Year')
ax3.set_ylabel('Number of Projects')
ax3.set_title('Annual Project Count Over Time', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Bottom left: Purpose
ax4 = fig.add_subplot(gs[2, 0])
purpose_top = purpose_counts.head(6)
ax4.bar(range(len(purpose_top)), purpose_top.values, color='coral', alpha=0.8)
ax4.set_xticks(range(len(purpose_top)))
ax4.set_xticklabels([p[:15] + '...' if len(p) > 15 else p for p in purpose_top.index], rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Count')
ax4.set_title('Top Purposes', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Bottom middle: Apparatus
ax5 = fig.add_subplot(gs[2, 1])
ax5.bar(range(len(apparatus_counts)), apparatus_counts.values, color='skyblue', alpha=0.8)
ax5.set_xticks(range(len(apparatus_counts)))
ax5.set_xticklabels([a[:10] for a in apparatus_counts.index], rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Count')
ax5.set_title('Apparatus Types', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# Bottom right: Top operators
ax6 = fig.add_subplot(gs[2, 2])
operator_top = operator_counts.head(6)
ax6.barh(range(len(operator_top)), operator_top.values, color='mediumpurple', alpha=0.8)
ax6.set_yticks(range(len(operator_top)))
ax6.set_yticklabels([op[:20] + '...' if len(op) > 20 else op for op in operator_top.index], fontsize=8)
ax6.set_xlabel('Count')
ax6.set_title('Top Operators', fontweight='bold')
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)

plt.suptitle('Cloud Seeding in the United States (2000-2025): Comprehensive Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('report/images/fig6_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 7: State-Year Heatmap
pivot_state_year = df.pivot_table(index='state', columns='year', aggfunc='size', fill_value=0)
# Select top states for readability
top_states = state_counts.head(15).index
pivot_subset = pivot_state_year.loc[top_states]

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(pivot_subset.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot_subset.columns)))
ax.set_xticklabels(pivot_subset.columns, rotation=45)
ax.set_yticks(range(len(pivot_subset.index)))
ax.set_yticklabels([s.title() for s in pivot_subset.index])
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('State', fontsize=12)
ax.set_title('State-Year Activity Heatmap: Top 15 States (2000-2025)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Number of Projects')
plt.tight_layout()
plt.savefig('report/images/fig7_state_year_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 8: Purpose by State (Stacked bar for top states)
fig, ax = plt.subplots(figsize=(14, 8))
top_10_states = state_counts.head(10).index
df_top = df[df['state'].isin(top_10_states)]
purpose_by_state = pd.crosstab(df_top['state'], df_top['purpose'])
# Simplify purposes for visualization
purpose_by_state.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
ax.set_xlabel('State', fontsize=12)
ax.set_ylabel('Number of Projects', fontsize=12)
ax.set_title('Purpose Composition by State (Top 10 States)', fontsize=14, fontweight='bold')
ax.legend(title='Purpose', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.set_xticklabels([s.title() for s in purpose_by_state.index], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('report/images/fig8_purpose_by_state.png', dpi=300, bbox_inches='tight')
plt.close()

# ===============================
# GENERATE SUMMARY STATISTICS JSON
# ===============================

summary_stats = {
    "total_projects": int(len(df)),
    "year_range": {"start": int(df['year'].min()), "end": int(df['year'].max())},
    "unique_states": int(df['state'].nunique()),
    "unique_operators": int(df['operator_affiliation'].nunique()),
    "peak_year": {"year": int(yearly_counts.idxmax()), "count": int(yearly_counts.max())},
    "top_state": {"state": state_counts.index[0], "count": int(state_counts.iloc[0])},
    "most_common_agent": agent_counts.index[0],
    "most_common_apparatus": apparatus_counts.index[0],
    "most_common_purpose": purpose_counts.index[0],
    "most_common_season": season_counts.index[0],
    "annual_trend_slope": float(z[0]),
    "projects_per_year_avg": float(yearly_counts.mean()),
    "projects_per_year_std": float(yearly_counts.std()),
    "state_concentration_gini": float(1 - 2 * np.sum((np.arange(1, len(state_counts)+1) * state_counts.values)) / (len(state_counts) * np.sum(state_counts.values)) + 1/len(state_counts))
}

with open('outputs/summary_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\n=== ANALYSIS COMPLETE ===")
print(f"Generated outputs:")
print(f"  - outputs/state_concentration.csv")
print(f"  - outputs/annual_dynamics.csv")
print(f"  - outputs/purpose_composition.csv")
print(f"  - outputs/agent_deployment.csv")
print(f"  - outputs/apparatus_deployment.csv")
print(f"  - outputs/agent_apparatus_crosstab.csv")
print(f"  - outputs/operator_summary.csv")
print(f"  - outputs/seasonal_patterns.csv")
print(f"  - outputs/summary_statistics.json")
print(f"\nGenerated figures:")
print(f"  - report/images/fig1_spatial_concentration.png")
print(f"  - report/images/fig2_annual_dynamics.png")
print(f"  - report/images/fig3_purpose_composition.png")
print(f"  - report/images/fig4_agent_apparatus.png")
print(f"  - report/images/fig5_seasonal_distribution.png")
print(f"  - report/images/fig6_dashboard.png")
print(f"  - report/images/fig7_state_year_heatmap.png")
print(f"  - report/images/fig8_purpose_by_state.png")
