#!/usr/bin/env python3
"""
Comprehensive analysis of US cloud-seeding records (2000-2025).
Produces tables, figures, and intermediate outputs for the research report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import json
import os
from collections import Counter

# ---- Paths ----
DATA_PATH = 'data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv'
OUTPUTS_DIR = 'outputs'
IMAGES_DIR = 'report/images'
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---- Load Data ----
df = pd.read_csv(DATA_PATH)
df['year'] = df['year'].astype(int)

print(f"Loaded {len(df)} records")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print(f"Columns: {list(df.columns)}")

# ============================================================
# 1. DATA OVERVIEW TABLE
# ============================================================
overview = {
    "total_records": int(len(df)),
    "year_range": f"{int(df['year'].min())}-{int(df['year'].max())}",
    "n_states": int(df['state'].nunique()),
    "states": sorted(df['state'].unique().tolist()),
    "n_operators": int(df['operator_affiliation'].nunique()),
    "n_unique_projects": int(df['project'].nunique()),
    "n_agents": int(df['agent'].nunique()),
    "agents": sorted(df['agent'].unique().tolist()),
    "apparatus_types": sorted(df['apparatus'].dropna().unique().tolist()),
    "n_purposes": int(df['purpose'].nunique()),
    "purposes": sorted(df['purpose'].unique().tolist()),
    "seasons": sorted(df['season'].unique().tolist()),
}
with open(os.path.join(OUTPUTS_DIR, 'data_overview.json'), 'w') as f:
    json.dump(overview, f, indent=2)
print("\n=== Data Overview ===")
for k, v in overview.items():
    if isinstance(v, list) and len(v) > 10:
        print(f"  {k}: {len(v)} items")
    else:
        print(f"  {k}: {v}")

# ============================================================
# 2. SPATIAL CONCENTRATION ANALYSIS
# ============================================================
state_counts = df.groupby('state').size().reset_index(name='count')
state_counts = state_counts.sort_values('count', ascending=False).reset_index(drop=True)
state_counts['pct'] = (state_counts['count'] / len(df) * 100).round(1)

state_counts.to_csv(os.path.join(OUTPUTS_DIR, 'state_concentration.csv'), index=False)
print("\n=== State Concentration ===")
print(state_counts.to_string(index=False))

# Top states concentration
top5 = state_counts.head(5)
top5_pct = top5['pct'].sum()
print(f"\nTop 5 states account for {top5_pct:.1f}% of all projects")

# ============================================================
# 3. ANNUAL ACTIVITY DYNAMICS
# ============================================================
annual_counts = df.groupby('year').size().reset_index(name='count')
annual_counts.to_csv(os.path.join(OUTPUTS_DIR, 'annual_activity.csv'), index=False)

# Annual by state
annual_state = df.groupby(['year', 'state']).size().reset_index(name='count')
annual_state.to_csv(os.path.join(OUTPUTS_DIR, 'annual_by_state.csv'), index=False)

# Annual by purpose
annual_purpose = df.groupby(['year', 'purpose']).size().reset_index(name='count')
annual_purpose.to_csv(os.path.join(OUTPUTS_DIR, 'annual_by_purpose.csv'), index=False)

print("\n=== Annual Activity ===")
print(annual_counts.to_string(index=False))

# Compute trends
early = annual_counts[annual_counts['year'] <= 2012]['count'].mean()
late = annual_counts[annual_counts['year'] > 2012]['count'].mean()
print(f"\nMean annual projects (2000-2012): {early:.1f}")
print(f"Mean annual projects (2013-2025): {late:.1f}")

# ============================================================
# 4. PURPOSE COMPOSITION
# ============================================================
# Normalize purposes into primary categories
def classify_purpose(p):
    p_lower = str(p).lower()
    cats = []
    if 'snowpack' in p_lower:
        cats.append('augment snowpack')
    if 'precipitation' in p_lower:
        cats.append('increase precipitation')
    if 'runoff' in p_lower:
        cats.append('increase runoff')
    if 'hail' in p_lower:
        cats.append('suppress hail')
    if 'fog' in p_lower:
        cats.append('suppress fog')
    if 'research' in p_lower:
        cats.append('research')
    return '; '.join(cats) if cats else 'other'

df['purpose_primary'] = df['purpose'].apply(classify_purpose)

# Count by primary purpose category
purpose_counts = df.groupby('purpose_primary').size().reset_index(name='count')
purpose_counts = purpose_counts.sort_values('count', ascending=False).reset_index(drop=True)
purpose_counts['pct'] = (purpose_counts['count'] / len(df) * 100).round(1)
purpose_counts.to_csv(os.path.join(OUTPUTS_DIR, 'purpose_composition.csv'), index=False)

print("\n=== Purpose Composition (Primary Categories) ===")
print(purpose_counts.to_string(index=False))

# Also save raw purpose distribution
raw_purpose = df.groupby('purpose').size().reset_index(name='count')
raw_purpose = raw_purpose.sort_values('count', ascending=False).reset_index(drop=True)
raw_purpose['pct'] = (raw_purpose['count'] / len(df) * 100).round(1)
raw_purpose.to_csv(os.path.join(OUTPUTS_DIR, 'purpose_raw.csv'), index=False)

# ============================================================
# 5. AGENT-APPARATUS DEPLOYMENT PATTERNS
# ============================================================
# Simplify agent names
def simplify_agent(a):
    a_lower = str(a).lower()
    if 'silver iodide' in a_lower:
        if 'hygroscopic' in a_lower:
            return 'silver iodide + hygroscopic'
        elif 'dry ice' in a_lower:
            return 'silver iodide + dry ice'
        elif 'cesium' in a_lower:
            return 'silver iodide + cesium iodide'
        elif 'calcium chloride' in a_lower:
            return 'silver iodide + calcium chloride'
        elif 'ammonium iodide' in a_lower or 'ammonium' in a_lower:
            return 'silver iodide + ammonium compounds'
        else:
            return 'silver iodide'
    elif 'dry ice' in a_lower:
        return 'dry ice'
    elif 'calcium chloride' in a_lower:
        return 'calcium chloride'
    elif 'carbon dioxide' in a_lower:
        return 'carbon dioxide'
    elif 'ionized air' in a_lower:
        return 'ionized air'
    elif 'ammonium iodide' in a_lower:
        return 'ammonium iodide'
    else:
        return 'other'

df['agent_simplified'] = df['agent'].apply(simplify_agent)

# Agent counts
agent_counts = df.groupby('agent_simplified').size().reset_index(name='count')
agent_counts = agent_counts.sort_values('count', ascending=False).reset_index(drop=True)
agent_counts['pct'] = (agent_counts['count'] / len(df) * 100).round(1)
agent_counts.to_csv(os.path.join(OUTPUTS_DIR, 'agent_distribution.csv'), index=False)

print("\n=== Agent Distribution (Simplified) ===")
print(agent_counts.to_string(index=False))

# Apparatus counts
apparatus_counts = df.groupby('apparatus').size().reset_index(name='count')
apparatus_counts = apparatus_counts.sort_values('count', ascending=False).reset_index(drop=True)
apparatus_counts['pct'] = (apparatus_counts['count'] / len(df) * 100).round(1)
apparatus_counts.to_csv(os.path.join(OUTPUTS_DIR, 'apparatus_distribution.csv'), index=False)

print("\n=== Apparatus Distribution ===")
print(apparatus_counts.to_string(index=False))

# Agent x Apparatus cross-tabulation
agent_apparatus_ct = pd.crosstab(df['agent_simplified'], df['apparatus'])
agent_apparatus_ct.to_csv(os.path.join(OUTPUTS_DIR, 'agent_apparatus_crosstab.csv'))
print("\n=== Agent x Apparatus Crosstab ===")
print(agent_apparatus_ct)

# Agent x Purpose crosstab
agent_purpose_ct = pd.crosstab(df['agent_simplified'], df['purpose_primary'])
agent_purpose_ct.to_csv(os.path.join(OUTPUTS_DIR, 'agent_purpose_crosstab.csv'))

# State x Agent crosstab
state_agent_ct = pd.crosstab(df['state'], df['agent_simplified'])
state_agent_ct.to_csv(os.path.join(OUTPUTS_DIR, 'state_agent_crosstab.csv'))

# ============================================================
# 6. SEASONAL DISTRIBUTION
# ============================================================
season_counts = df.groupby('season').size().reset_index(name='count')
season_counts = season_counts.sort_values('count', ascending=False).reset_index(drop=True)
season_counts['pct'] = (season_counts['count'] / len(df) * 100).round(1)
season_counts.to_csv(os.path.join(OUTPUTS_DIR, 'seasonal_distribution.csv'), index=False)

print("\n=== Seasonal Distribution ===")
print(season_counts.to_string(index=False))

# Simplify seasons
def simplify_season(s):
    s_lower = str(s).lower().replace(' ', '')
    if ',' not in s_lower:
        return s_lower
    # Multi-season: use first season as primary
    return s_lower.split(',')[0]

df['season_simple'] = df['season'].apply(simplify_season)
simple_season = df.groupby('season_simple').size().reset_index(name='count')
simple_season = simple_season.sort_values('count', ascending=False).reset_index(drop=True)
simple_season['pct'] = (simple_season['count'] / len(df) * 100).round(1)
simple_season.to_csv(os.path.join(OUTPUTS_DIR, 'season_simple_distribution.csv'), index=False)

# ============================================================
# 7. OPERATOR ANALYSIS
# ============================================================
operator_counts = df.groupby('operator_affiliation').size().reset_index(name='count')
operator_counts = operator_counts.sort_values('count', ascending=False).reset_index(drop=True)
operator_counts['pct'] = (operator_counts['count'] / len(df) * 100).round(1)
operator_counts.to_csv(os.path.join(OUTPUTS_DIR, 'operator_distribution.csv'), index=False)

print("\n=== Top 15 Operators ===")
print(operator_counts.head(15).to_string(index=False))

# ============================================================
# 8. SAVE SUMMARY STATISTICS
# ============================================================
summary_stats = {
    "total_records": int(len(df)),
    "years_covered": int(df['year'].nunique()),
    "states_active": int(df['state'].nunique()),
    "top_state": state_counts.iloc[0]['state'],
    "top_state_count": int(state_counts.iloc[0]['count']),
    "top_state_pct": float(state_counts.iloc[0]['pct']),
    "top5_states_pct": float(top5_pct),
    "peak_year": int(annual_counts.loc[annual_counts['count'].idxmax(), 'year']),
    "peak_year_count": int(annual_counts['count'].max()),
    "mean_annual_projects": float(annual_counts['count'].mean()),
    "median_annual_projects": float(annual_counts['count'].median()),
    "dominant_purpose": purpose_counts.iloc[0]['purpose_primary'],
    "dominant_purpose_pct": float(purpose_counts.iloc[0]['pct']),
    "dominant_agent": agent_counts.iloc[0]['agent_simplified'],
    "dominant_agent_pct": float(agent_counts.iloc[0]['pct']),
    "dominant_apparatus": apparatus_counts.iloc[0]['apparatus'],
    "dominant_apparatus_pct": float(apparatus_counts.iloc[0]['pct']),
    "dominant_season": simple_season.iloc[0]['season_simple'],
    "dominant_season_pct": float(simple_season.iloc[0]['pct']),
}
with open(os.path.join(OUTPUTS_DIR, 'summary_statistics.json'), 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\n=== Summary Statistics ===")
for k, v in summary_stats.items():
    print(f"  {k}: {v}")

print("\nAll intermediate outputs saved successfully.")
