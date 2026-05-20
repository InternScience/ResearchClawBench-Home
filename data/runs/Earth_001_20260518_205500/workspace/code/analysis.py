#!/usr/bin/env python3
"""
Comprehensive analysis of U.S. cloud seeding activities (2000-2025)
Reproduces key empirical findings from the target paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
print("Loading data...")
df = pd.read_csv('data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv')
print(f"Loaded {len(df)} records")

# ============================================================================
# 1. DATA OVERVIEW AND CLEANING
# ============================================================================
print("\n=== DATA OVERVIEW ===")
print(f"Records: {len(df)}")
print(f"Years: {df['year'].min()} - {df['year'].max()}")
print(f"States: {df['state'].nunique()}")
print(f"Unique projects: {df['project'].nunique()}")

# Standardize text columns
for col in ['state', 'season', 'agent', 'apparatus', 'purpose', 'operator_affiliation']:
    df[col] = df[col].astype(str).str.lower().str.strip()

# Handle missing/NaN
for col in ['agent', 'apparatus', 'purpose', 'operator_affiliation']:
    df[col] = df[col].replace('nan', 'unknown')

# ============================================================================
# 2. SPATIAL CONCENTRATION ANALYSIS
# ============================================================================
print("\n=== SPATIAL CONCENTRATION ===")
state_counts = df['state'].value_counts()
print(state_counts)

# Compute concentration metrics
total_records = len(df)
top3_share = state_counts.head(3).sum() / total_records * 100
top5_share = state_counts.head(5).sum() / total_records * 100
top8_share = state_counts.head(8).sum() / total_records * 100

print(f"\nTop 3 states (CA, CO, UT) account for {top3_share:.1f}% of activities")
print(f"Top 5 states account for {top5_share:.1f}% of activities")
print(f"Top 8 states account for {top8_share:.1f}% of activities")

spatial_metrics = {
    'total_records': int(total_records),
    'num_states': int(df['state'].nunique()),
    'top3_states': state_counts.head(3).index.tolist(),
    'top3_counts': state_counts.head(3).values.tolist(),
    'top3_share_pct': round(top3_share, 2),
    'top5_share_pct': round(top5_share, 2),
    'top8_share_pct': round(top8_share, 2),
    'state_counts': state_counts.to_dict()
}

with open('outputs/spatial_concentration.json', 'w') as f:
    json.dump(spatial_metrics, f, indent=2)

# ============================================================================
# 3. ANNUAL ACTIVITY DYNAMICS
# ============================================================================
print("\n=== ANNUAL ACTIVITY DYNAMICS ===")
yearly_counts = df['year'].value_counts().sort_index()
print(yearly_counts)

# Identify peak and decline periods
peak_years = yearly_counts.loc[2003:2005]
print(f"\nPeak period (2003-2005): {peak_years.sum()} activities ({peak_years.sum()/total_records*100:.1f}%)")

decline_years = yearly_counts.loc[2016:2020]
print(f"Decline period (2016-2020): {decline_years.sum()} activities ({decline_years.sum()/total_records*100:.1f}%)")

recovery_years = yearly_counts.loc[2021:2025]
print(f"Recovery period (2021-2025): {recovery_years.sum()} activities ({recovery_years.sum()/total_records*100:.1f}%)")

# Pre-2020 vs post-2020
pre_2020 = yearly_counts.loc[2000:2019].sum()
post_2020 = yearly_counts.loc[2020:2025].sum()
print(f"Pre-2020 (2000-2019): {pre_2020} activities")
print(f"Post-2020 (2020-2025): {post_2020} activities")

# Compute moving average
yearly_df = yearly_counts.reset_index()
yearly_df.columns = ['year', 'count']
yearly_df['moving_avg_3yr'] = yearly_df['count'].rolling(window=3, center=True, min_periods=1).mean()

yearly_metrics = {
    'yearly_counts': yearly_counts.to_dict(),
    'peak_2003_2005': int(peak_years.sum()),
    'decline_2016_2020': int(decline_years.sum()),
    'recovery_2021_2025': int(recovery_years.sum()),
    'pre_2020_total': int(pre_2020),
    'post_2020_total': int(post_2020),
    'lowest_year': int(yearly_counts.idxmin()),
    'lowest_count': int(yearly_counts.min()),
    'highest_year': int(yearly_counts.idxmax()),
    'highest_count': int(yearly_counts.max()),
}

with open('outputs/annual_dynamics.json', 'w') as f:
    json.dump(yearly_metrics, f, indent=2)

# ============================================================================
# 4. PURPOSE COMPOSITION
# ============================================================================
print("\n=== PURPOSE COMPOSITION ===")
purpose_counts = df['purpose'].value_counts()
print(purpose_counts)

# Categorize purposes into primary categories
def categorize_purpose(p):
    p = str(p).lower()
    categories = []
    if 'snowpack' in p:
        categories.append('augment snowpack')
    if 'precipitation' in p or 'rain' in p:
        categories.append('increase precipitation')
    if 'hail' in p:
        categories.append('suppress hail')
    if 'fog' in p:
        categories.append('suppress fog')
    if 'runoff' in p:
        categories.append('increase runoff')
    if len(categories) == 0:
        categories.append('other')
    return categories

df['purpose_categories'] = df['purpose'].apply(categorize_purpose)

# Flatten for counting
from collections import Counter
purpose_cat_counts = Counter()
for cats in df['purpose_categories']:
    for c in cats:
        purpose_cat_counts[c] += 1

print("\nPrimary purpose categories:")
for p, c in purpose_cat_counts.most_common():
    print(f"  {p}: {c} ({c/total_records*100:.1f}%)")

purpose_metrics = {
    'raw_purpose_counts': purpose_counts.head(10).to_dict(),
    'primary_category_counts': dict(purpose_cat_counts.most_common()),
    'snowpack_pct': round(purpose_cat_counts['augment snowpack'] / total_records * 100, 2),
    'precipitation_pct': round(purpose_cat_counts['increase precipitation'] / total_records * 100, 2),
    'hail_pct': round(purpose_cat_counts.get('suppress hail', 0) / total_records * 100, 2),
}

with open('outputs/purpose_composition.json', 'w') as f:
    json.dump(purpose_metrics, f, indent=2)

# ============================================================================
# 5. AGENT-APPARATUS DEPLOYMENT PATTERNS
# ============================================================================
print("\n=== AGENT-APPARATUS DEPLOYMENT ===")

# Agent categorization
def categorize_agent(a):
    a = str(a).lower()
    if 'silver iodide' in a and 'sodium iodide' in a:
        return 'AgI + NaI'
    elif 'silver iodide' in a and 'ammonium iodide' in a:
        return 'AgI + NH4I'
    elif 'silver iodide' in a and 'calcium chloride' in a:
        return 'AgI + CaCl2'
    elif 'silver iodide' in a and 'hygroscopic' in a:
        return 'AgI + hygroscopic'
    elif 'silver iodide' in a and 'dry ice' in a:
        return 'AgI + dry ice'
    elif 'silver iodide' in a:
        return 'AgI only'
    elif 'ionized air' in a:
        return 'ionized air'
    else:
        return 'other'

df['agent_category'] = df['agent'].apply(categorize_agent)
agent_counts = df['agent_category'].value_counts()
print("Agent categories:")
print(agent_counts)

# Apparatus
apparatus_counts = df['apparatus'].value_counts()
print("\nApparatus:")
print(apparatus_counts)

# Cross-tabulation
print("\nAgent-Apparatus cross-tabulation:")
crosstab = pd.crosstab(df['agent_category'], df['apparatus'])
print(crosstab)

# Silver iodide dominance
agi_records = df[df['agent'].str.contains('silver iodide', na=False)]
agi_share = len(agi_records) / total_records * 100
print(f"\nSilver iodide (any form): {len(agi_records)} records ({agi_share:.1f}%)")

# Ground vs airborne
ground_records = df[df['apparatus'].str.contains('ground', na=False)]
airborne_records = df[df['apparatus'].str.contains('airborne', na=False)]
print(f"Ground-based: {len(ground_records)} ({len(ground_records)/total_records*100:.1f}%)")
print(f"Airborne: {len(airborne_records)} ({len(airborne_records)/total_records*100:.1f}%)")

agent_apparatus_metrics = {
    'agent_counts': agent_counts.to_dict(),
    'apparatus_counts': apparatus_counts.to_dict(),
    'crosstab': crosstab.to_dict(),
    'silver_iodide_any_pct': round(agi_share, 2),
    'ground_any_pct': round(len(ground_records)/total_records*100, 2),
    'airborne_any_pct': round(len(airborne_records)/total_records*100, 2),
}

with open('outputs/agent_apparatus.json', 'w') as f:
    json.dump(agent_apparatus_metrics, f, indent=2)

# ============================================================================
# 6. SEASONAL PATTERNS
# ============================================================================
print("\n=== SEASONAL PATTERNS ===")
season_counts = df['season'].value_counts()
print(season_counts)

# Simplify seasons
def simplify_season(s):
    s = str(s).lower()
    if 'winter' in s and 'spring' in s and 'summer' in s and 'fall' in s:
        return 'multi-season'
    elif 'winter' in s and ('spring' in s or 'summer' in s or 'fall' in s):
        return 'winter + other'
    elif s == 'winter':
        return 'winter only'
    elif 'summer' in s:
        return 'summer'
    else:
        return 'other'

df['season_simple'] = df['season'].apply(simplify_season)
season_simple_counts = df['season_simple'].value_counts()
print("\nSimplified seasons:")
print(season_simple_counts)

# ============================================================================
# 7. OPERATOR CONCENTRATION
# ============================================================================
print("\n=== OPERATOR CONCENTRATION ===")
operator_counts = df['operator_affiliation'].value_counts()
print(operator_counts.head(10))

top3_ops = operator_counts.head(3).sum()
print(f"\nTop 3 operators account for {top3_ops} records ({top3_ops/total_records*100:.1f}%)")

# ============================================================================
# 8. GENERATE ALL FIGURES
# ============================================================================
print("\n=== GENERATING FIGURES ===")

# ---- Figure 1: Spatial concentration by state ----
fig, ax = plt.subplots(figsize=(12, 6))
states = state_counts.index
values = state_counts.values
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(states)))
bars = ax.bar(range(len(states)), values, color=colors)
ax.set_xticks(range(len(states)))
ax.set_xticklabels([s.title() for s in states], rotation=45, ha='right')
ax.set_ylabel('Number of Cloud Seeding Activities')
ax.set_title('U.S. Cloud Seeding Activity by State (2000–2025)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# Add value labels
for i, v in enumerate(values):
    ax.text(i, v + 3, str(v), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('report/images/figure1_state_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure1_state_distribution.png")

# ---- Figure 2: Annual activity dynamics ----
fig, ax = plt.subplots(figsize=(12, 6))
years = yearly_df['year']
counts = yearly_df['count']
moving_avg = yearly_df['moving_avg_3yr']

ax.bar(years, counts, color='steelblue', alpha=0.7, label='Annual Count', width=0.8)
ax.plot(years, moving_avg, color='darkred', linewidth=2.5, marker='o', markersize=4, label='3-Year Moving Average')

# Highlight periods
ax.axvspan(2003, 2005, alpha=0.15, color='green', label='Peak Period (2003–2005)')
ax.axvspan(2016, 2020, alpha=0.15, color='orange', label='Decline Period (2016–2020)')
ax.axvspan(2021, 2025, alpha=0.15, color='purple', label='Recovery Period (2021–2025)')

ax.set_xlabel('Year')
ax.set_ylabel('Number of Cloud Seeding Activities')
ax.set_title('Cloud Seeding Activity by Year in the United States (2000–2025)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(range(2000, 2026, 2))
plt.tight_layout()
plt.savefig('report/images/figure2_annual_dynamics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure2_annual_dynamics.png")

# ---- Figure 3: State over time heatmap ----
state_year = df.groupby(['state', 'year']).size().unstack(fill_value=0)
# Keep top 10 states
top_states = state_counts.head(10).index
state_year_top = state_year.loc[top_states]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(state_year_top, cmap='YlOrRd', linewidths=0.5, annot=True, fmt='d', 
            cbar_kws={'label': 'Number of Activities'}, ax=ax)
ax.set_title('Cloud Seeding Activity by U.S. State over Time (2000–2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('State')
ax.set_yticklabels([s.title() for s in state_year_top.index], rotation=0)
plt.tight_layout()
plt.savefig('report/images/figure3_state_year_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure3_state_year_heatmap.png")

# ---- Figure 4: Purpose composition ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Top raw purposes
top_purposes = purpose_counts.head(8)
ax1 = axes[0]
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(top_purposes)))
wedges, texts, autotexts = ax1.pie(top_purposes.values, labels=[p.title() for p in top_purposes.index], 
                                    autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                    textprops={'fontsize': 9})
ax1.set_title('Stated Purpose of Cloud Seeding Activity (2000–2025)', fontsize=12, fontweight='bold')

# Right: Primary categories
ax2 = axes[1]
cat_labels = [p.replace('augment ', 'Augment ').replace('increase ', 'Increase ').replace('suppress ', 'Suppress ').title() 
              for p, _ in purpose_cat_counts.most_common()]
cat_values = [c for _, c in purpose_cat_counts.most_common()]
colors_bar = plt.cm.Paired(np.linspace(0, 1, len(cat_labels)))
bars = ax2.barh(cat_labels, cat_values, color=colors_bar)
ax2.set_xlabel('Number of Activities')
ax2.set_title('Primary Purpose Categories (2000–2025)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, cat_values):
    ax2.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val} ({val/total_records*100:.1f}%)', 
             va='center', fontsize=9)
ax2.set_xlim(0, max(cat_values) * 1.3)

plt.tight_layout()
plt.savefig('report/images/figure4_purpose_composition.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure4_purpose_composition.png")

# ---- Figure 5: Agent and Apparatus ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Agent categories
ax1 = axes[0]
agent_colors = plt.cm.tab10(np.linspace(0, 1, len(agent_counts)))
bars1 = ax1.barh(range(len(agent_counts)), agent_counts.values, color=agent_colors)
ax1.set_yticks(range(len(agent_counts)))
ax1.set_yticklabels(agent_counts.index, fontsize=10)
ax1.set_xlabel('Number of Activities')
ax1.set_title('Seeding Agent Usage (2000–2025)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(agent_counts.values):
    ax1.text(v + 3, i, f'{v} ({v/total_records*100:.1f}%)', va='center', fontsize=9)
ax1.set_xlim(0, max(agent_counts.values) * 1.25)

# Right: Apparatus
ax2 = axes[1]
app_colors = ['#2ca02c', '#ff7f0e', '#d62728']
bars2 = ax2.bar(range(len(apparatus_counts)), apparatus_counts.values, color=app_colors)
ax2.set_xticks(range(len(apparatus_counts)))
ax2.set_xticklabels([a.title() for a in apparatus_counts.index], fontsize=10)
ax2.set_ylabel('Number of Activities')
ax2.set_title('Deployment Apparatus (2000–2025)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(apparatus_counts.values):
    ax2.text(i, v + 5, f'{v}\n({v/total_records*100:.1f}%)', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/figure5_agent_apparatus.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure5_agent_apparatus.png")

# ---- Figure 6: Agent-Apparatus cross-tabulation heatmap ----
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Number of Activities'})
ax.set_title('Agent-Apparatus Deployment Matrix (2000–2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('Deployment Apparatus')
ax.set_ylabel('Seeding Agent Category')
ax.set_xticklabels([t.get_text().title() for t in ax.get_xticklabels()], rotation=0)
ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], rotation=0)
plt.tight_layout()
plt.savefig('report/images/figure6_agent_apparatus_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure6_agent_apparatus_matrix.png")

# ---- Figure 7: Seasonal distribution ----
fig, ax = plt.subplots(figsize=(10, 6))
season_order = ['winter only', 'winter + other', 'multi-season', 'summer', 'other']
season_data = {s: season_simple_counts.get(s, 0) for s in season_order}
season_labels = [s.title() for s in season_order]
season_vals = list(season_data.values())
colors_season = plt.cm.Spectral(np.linspace(0, 1, len(season_labels)))
bars = ax.bar(season_labels, season_vals, color=colors_season)
ax.set_ylabel('Number of Activities')
ax.set_title('Seasonal Distribution of Cloud Seeding Activities (2000–2025)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, season_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val}\n({val/total_records*100:.1f}%)', 
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('report/images/figure7_seasonal_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure7_seasonal_distribution.png")

# ---- Figure 8: Purpose by State (top states) ----
fig, ax = plt.subplots(figsize=(12, 7))
# Create purpose-state matrix for top states
top5 = state_counts.head(5).index
purpose_state = pd.crosstab(df[df['state'].isin(top5)]['state'], 
                            df[df['state'].isin(top5)]['purpose'].apply(
                                lambda x: 'snowpack' if 'snowpack' in str(x) and 'precipitation' not in str(x)
                                else ('precipitation' if 'precipitation' in str(x) and 'snowpack' not in str(x) 
                                and 'hail' not in str(x) and 'fog' not in str(x)
                                else ('snowpack+precip' if 'snowpack' in str(x) and 'precipitation' in str(x)
                                else ('hail' if 'hail' in str(x) else 'other')))))
purpose_state = purpose_state.reindex(top5)
purpose_state.plot(kind='barh', stacked=True, ax=ax, colormap='tab10', width=0.7)
ax.set_xlabel('Number of Activities')
ax.set_ylabel('State')
ax.set_title('Purpose Composition by Top 5 States (2000–2025)', fontsize=14, fontweight='bold')
ax.legend(title='Purpose Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_yticklabels([s.title() for s in purpose_state.index], rotation=0)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure8_purpose_by_state.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure8_purpose_by_state.png")

# ---- Figure 9: Operator concentration ----
fig, ax = plt.subplots(figsize=(12, 6))
top_ops = operator_counts.head(10)
bars = ax.barh(range(len(top_ops)), top_ops.values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top_ops))))
ax.set_yticks(range(len(top_ops)))
ax.set_yticklabels([o.title() for o in top_ops.index], fontsize=9)
ax.set_xlabel('Number of Activities')
ax.set_title('Top 10 Cloud Seeding Operators by Activity Count (2000–2025)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_ops.values):
    ax.text(v + 1, i, f'{v}', va='center', fontsize=9)
ax.set_xlim(0, max(top_ops.values) * 1.15)
plt.tight_layout()
plt.savefig('report/images/figure9_operator_concentration.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure9_operator_concentration.png")

# ============================================================================
# 9. SUMMARY STATISTICS TABLE
# ============================================================================
print("\n=== SUMMARY STATISTICS ===")
summary = {
    'dataset': {
        'total_records': int(total_records),
        'year_range': f"{df['year'].min()}-{df['year'].max()}",
        'unique_projects': int(df['project'].nunique()),
        'num_states': int(df['state'].nunique()),
    },
    'spatial': {
        'top_state': state_counts.index[0].title(),
        'top_state_count': int(state_counts.iloc[0]),
        'top3_share_pct': round(top3_share, 2),
        'top5_share_pct': round(top5_share, 2),
    },
    'temporal': {
        'peak_period': '2003-2005',
        'peak_count': int(peak_years.sum()),
        'decline_period': '2016-2020',
        'decline_count': int(decline_years.sum()),
        'recovery_period': '2021-2025',
        'recovery_count': int(recovery_years.sum()),
        'lowest_year': int(yearly_counts.idxmin()),
        'lowest_count': int(yearly_counts.min()),
    },
    'purpose': {
        'primary_purpose': 'augment snowpack',
        'primary_count': int(purpose_cat_counts['augment snowpack']),
        'primary_pct': round(purpose_cat_counts['augment snowpack']/total_records*100, 2),
        'secondary_purpose': 'increase precipitation',
        'secondary_count': int(purpose_cat_counts['increase precipitation']),
        'secondary_pct': round(purpose_cat_counts['increase precipitation']/total_records*100, 2),
    },
    'agent_apparatus': {
        'dominant_agent': 'silver iodide (any form)',
        'agent_count': int(len(agi_records)),
        'agent_pct': round(agi_share, 2),
        'dominant_apparatus': 'ground',
        'ground_count': int(len(ground_records)),
        'ground_pct': round(len(ground_records)/total_records*100, 2),
        'airborne_count': int(len(airborne_records)),
        'airborne_pct': round(len(airborne_records)/total_records*100, 2),
    }
}

with open('outputs/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Create summary CSV
summary_df = pd.DataFrame([
    ['Total Records', total_records, ''],
    ['Year Range', '2000-2025', ''],
    ['Unique Projects', df['project'].nunique(), ''],
    ['States Covered', df['state'].nunique(), ''],
    ['Top State', state_counts.index[0].title(), state_counts.iloc[0]],
    ['Top 3 States Share', f'{top3_share:.1f}%', f'({state_counts.head(3).index.tolist()})'],
    ['Top 5 States Share', f'{top5_share:.1f}%', ''],
    ['Peak Period (2003-2005)', peak_years.sum(), f'{peak_years.sum()/total_records*100:.1f}%'],
    ['Decline Period (2016-2020)', decline_years.sum(), f'{decline_years.sum()/total_records*100:.1f}%'],
    ['Recovery Period (2021-2025)', recovery_years.sum(), f'{recovery_years.sum()/total_records*100:.1f}%'],
    ['Lowest Year', yearly_counts.idxmin(), yearly_counts.min()],
    ['Primary Purpose', 'Augment Snowpack', f'{purpose_cat_counts["augment snowpack"]} ({purpose_cat_counts["augment snowpack"]/total_records*100:.1f}%)'],
    ['Secondary Purpose', 'Increase Precipitation', f'{purpose_cat_counts["increase precipitation"]} ({purpose_cat_counts["increase precipitation"]/total_records*100:.1f}%)'],
    ['Dominant Agent', 'Silver Iodide', f'{len(agi_records)} ({agi_share:.1f}%)'],
    ['Dominant Apparatus', 'Ground', f'{len(ground_records)} ({len(ground_records)/total_records*100:.1f}%)'],
    ['Airborne Apparatus', 'Airborne', f'{len(airborne_records)} ({len(airborne_records)/total_records*100:.1f}%)'],
    ['Combined Ground+Airborne', 'Both', f'{apparatus_counts.get("ground, airborne", 0)} ({apparatus_counts.get("ground, airborne", 0)/total_records*100:.1f}%)'],
], columns=['Metric', 'Value', 'Detail'])

summary_df.to_csv('outputs/summary_table.csv', index=False)
print("\nSaved outputs/summary_table.csv")

print("\n=== ANALYSIS COMPLETE ===")
