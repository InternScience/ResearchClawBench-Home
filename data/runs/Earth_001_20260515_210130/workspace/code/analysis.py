#!/usr/bin/env python3
"""
Cloud Seeding US 2000-2025 — Reproducible Analysis
Analyzes NOAA weather-modification records to test whether the target paper's
central empirical conclusions can be independently recovered.
"""

import csv
import json
import os
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────
DATA = 'data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv'
GEOJSON = 'data/dataset1_cloud_seeding_records/us_states.geojson'
OUTPUTS = 'outputs'
REPORT_IMG = 'report/images'
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(REPORT_IMG, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open(DATA) as f:
    reader = csv.DictReader(f)
    records = list(reader)

print(f"Loaded {len(records)} records")

# ── Normalize fields ───────────────────────────────────────────────────────
for r in records:
    r['year'] = int(r['year'])
    r['state'] = r['state'].strip().lower()
    r['operator_affiliation'] = r['operator_affiliation'].strip()
    r['agent'] = r['agent'].strip()
    r['apparatus'] = r['apparatus'].strip()
    r['purpose'] = r['purpose'].strip()
    r['season'] = r['season'].strip()

# ── State name standardization ─────────────────────────────────────────────
state_map = {
    'california': 'California', 'colorado': 'Colorado', 'idaho': 'Idaho',
    'kansas': 'Kansas', 'montana': 'Montana', 'nevada': 'Nevada',
    'north dakota': 'North Dakota', 'oklahoma': 'Oklahoma',
    'oregon': 'Oregon', 'south dakota': 'South Dakota', 'texas': 'Texas',
    'utah': 'Utah', 'wyoming': 'Wyoming'
}

# ═══════════════════════════════════════════════════════════════════════════
# 1. SPATIAL CONCENTRATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SPATIAL CONCENTRATION ===")

state_counts = Counter(r['state'] for r in records)
state_total = sum(state_counts.values())

# Export
state_dist = []
for state_name, sname in sorted(state_map.items()):
    count = state_counts.get(state_name, 0)
    pct = 100.0 * count / state_total
    state_dist.append({
        'state': sname,
        'state_key': state_name,
        'count': count,
        'pct': round(pct, 2)
    })

with open(os.path.join(OUTPUTS, 'state_distribution.json'), 'w') as f:
    json.dump(state_dist, f, indent=2)

print("Top 5 states by project count:")
for s in sorted(state_dist, key=lambda x: -x['count'])[:5]:
    print(f"  {s['state']}: {s['count']} ({s['pct']:.1f}%)")

# Load GeoJSON and build choropleth data
with open(GEOJSON) as f:
    geojson = json.load(f)

state_abbrev_map = {
    'California': 'CA', 'Colorado': 'CO', 'Idaho': 'ID',
    'Kansas': 'KS', 'Montana': 'MT', 'Nevada': 'NV',
    'North Dakota': 'ND', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'South Dakota': 'SD', 'Texas': 'TX', 'Utah': 'UT', 'Wyoming': 'WY'
}

# Build FIPS to count mapping
fips_to_count = {}
for feature in geojson['features']:
    state_name = feature['properties']['name']
    fips = feature['id']
    key = state_name.lower()
    cnt = state_counts.get(key, 0)
    fips_to_count[fips] = cnt

# Figure: State-level choropleth
fig, ax = plt.subplots(figsize=(12, 7))
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

patches = []
values = []
state_labels = {}
for feature in geojson['features']:
    fips = feature['id']
    for geom in (feature['geometry']['coordinates'] if feature['geometry']['type'] == 'Polygon'
                 else feature['geometry']['coordinates'][0]):
        if feature['geometry']['type'] == 'MultiPolygon':
            for poly_coords in feature['geometry']['coordinates']:
                xy = np.array(poly_coords[0])
                if len(xy.shape) == 2:
                    polygon = Polygon(xy, closed=True)
                    patches.append(polygon)
                    values.append(fips_to_count.get(fips, 0))
        else:
            xy = np.array(geom)
            if len(xy.shape) == 2:
                polygon = Polygon(xy, closed=True)
                patches.append(polygon)
                values.append(fips_to_count.get(fips, 0))
    # Store centroid for labeling
    if feature['geometry']['type'] == 'Polygon':
        coords = np.array(feature['geometry']['coordinates'][0])
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
        state_labels[fips] = (cx, cy)
    else:
        coords = np.array(feature['geometry']['coordinates'][0][0])
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
        state_labels[fips] = (cx, cy)

# Only keep contiguous US states with data or nearby
cmap = plt.cm.YlOrRd
norm = plt.Normalize(0, max(values) + 1)

pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='#333333', linewidth=0.5)
pc.set_array(np.array(values))
ax.add_collection(pc)

# Color all states grey, then overlay colored patches
# Actually simpler: just do the patch collection approach
ax.autoscale_view()
ax.set_xlim(-125, -66)
ax.set_ylim(24, 50)

# Add labels
for fips, (cx, cy) in state_labels.items():
    cnt = fips_to_count.get(fips, 0)
    if cnt > 0:
        abbr = state_abbrev_map.get(
            next((f['properties']['name'] for f in geojson['features'] if f['id'] == fips), ''), '')
        ax.text(cx, cy, f"{abbr}\n{cnt}", ha='center', va='center', fontsize=7,
                fontweight='bold', color='black' if cnt < 80 else 'white')

cbar = fig.colorbar(pc, ax=ax, shrink=0.7, label='Number of Projects')
ax.set_title('Spatial Distribution of Cloud Seeding Projects\nUnited States, 2000–2025', fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.axis('off')

fig.tight_layout()
fig.savefig(os.path.join(REPORT_IMG, 'figure1_spatial_concentration.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure1_spatial_concentration.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ANNUAL ACTIVITY DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== ANNUAL ACTIVITY DYNAMICS ===")

yearly = Counter(r['year'] for r in records)
years = sorted(yearly.keys())
year_range = list(range(min(years), max(years) + 1))

annual_data = []
for y in year_range:
    cnt = yearly.get(y, 0)
    annual_data.append({'year': y, 'count': cnt})

with open(os.path.join(OUTPUTS, 'annual_activity.json'), 'w') as f:
    json.dump(annual_data, f, indent=2)

# Also by state
yearly_state = defaultdict(Counter)
for r in records:
    yearly_state[r['year']][r['state']] += 1

# Export yearly by state
ys_export = {}
for y in year_range:
    ys_export[str(y)] = dict(yearly_state[y])
with open(os.path.join(OUTPUTS, 'annual_by_state.json'), 'w') as f:
    json.dump(ys_export, f, indent=2)

# Figure: Annual activity
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel A: Total annual projects
ax = axes[0]
counts = [yearly.get(y, 0) for y in year_range]
bars = ax.bar(year_range, counts, color='steelblue', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Projects', fontsize=12)
ax.set_title('A. Annual Cloud Seeding Project Count (2000–2025)', fontsize=13, fontweight='bold')
ax.set_xlim(1999.5, 2025.5)
ax.grid(axis='y', alpha=0.3)
# Add trend line
z = np.polyfit(year_range, counts, 1)
p = np.poly1d(z)
ax.plot(year_range, p(year_range), '--', color='darkred', linewidth=2, label=f'Trend (slope={z[0]:.2f}/yr)')
ax.legend(fontsize=9)

# Panel B: Stacked area by state
ax = axes[1]
top_states = [s['state_key'] for s in sorted(state_dist, key=lambda x: -x['count'])[:6]]
other_states = [s for s in state_map if s not in top_states]

state_series = {}
for sk in top_states + ['other']:
    state_series[sk] = []

for y in year_range:
    for sk in top_states:
        state_series[sk].append(yearly_state[y].get(sk, 0))
    other_cnt = sum(yearly_state[y].get(s, 0) for s in other_states)
    state_series['other'].append(other_cnt)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f']
labels_display = [state_map[s] for s in top_states] + ['Other States']

ax.stackplot(year_range,
             [state_series[top_states[0]], state_series[top_states[1]],
              state_series[top_states[2]], state_series[top_states[3]],
              state_series[top_states[4]], state_series[top_states[5]],
              state_series['other']],
             labels=labels_display, colors=colors, alpha=0.85)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Projects', fontsize=12)
ax.set_title('B. Project Composition by State (2000–2025)', fontsize=13, fontweight='bold')
ax.set_xlim(1999.5, 2025.5)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(REPORT_IMG, 'figure2_annual_dynamics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure2_annual_dynamics.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PURPOSE COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== PURPOSE COMPOSITION ===")

# Parse multi-purpose strings
purpose_counter = Counter()
for r in records:
    purposes = [p.strip() for p in r['purpose'].split(',')]
    for p in purposes:
        purpose_counter[p] += 1

# Consolidate
purpose_map = {
    'augment snowpack': 'Snowpack Augmentation',
    'increase precipitation': 'Precip. Enhancement',
    'suppress hail': 'Hail Suppression',
    'increase runoff': 'Runoff Increase',
    'research': 'Research',
    'suppress fog': 'Fog Suppression',
    'increase precipitation': 'Precip. Enhancement',
}

consolidated = Counter()
for p, c in purpose_counter.items():
    consolidated[purpose_map.get(p, p)] += c

purpose_data = [{'purpose': k, 'count': v} for k, v in consolidated.most_common()]
with open(os.path.join(OUTPUTS, 'purpose_composition.json'), 'w') as f:
    json.dump(purpose_data, f, indent=2)

print("Purpose breakdown:")
for pd in purpose_data:
    print(f"  {pd['purpose']}: {pd['count']}")

# Also raw multi-purpose combinations
raw_purpose = Counter(r['purpose'] for r in records)
raw_purpose_data = [{'purpose': k, 'count': v} for k, v in raw_purpose.most_common()]
with open(os.path.join(OUTPUTS, 'purpose_raw_combinations.json'), 'w') as f:
    json.dump(raw_purpose_data, f, indent=2)

# Figure: Purpose composition
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Individual purpose tags (bar)
ax = axes[0]
labels = [pd['purpose'] for pd in purpose_data]
values = [pd['count'] for pd in purpose_data]
colors_bar = ['#2166ac', '#4393c3', '#92c5de', '#d6604d', '#b2182b', '#f4a582']
bars = ax.barh(range(len(labels)), values, color=colors_bar[:len(labels)], edgecolor='white')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Number of Project-Purpose Mentions', fontsize=12)
ax.set_title('A. Purpose Tag Frequency', fontsize=13, fontweight='bold')
for i, (bar, v) in enumerate(zip(bars, values)):
    ax.text(v + 3, bar.get_y() + bar.get_height()/2, str(v), va='center', fontsize=10)
ax.set_xlim(0, max(values) * 1.15)
ax.grid(axis='x', alpha=0.3)

# Panel B: Top multi-purpose combinations
ax = axes[1]
top_combos = raw_purpose_data[:8]
labels = [pd['purpose'][:60] for pd in top_combos]
values = [pd['count'] for pd in top_combos]
bars = ax.barh(range(len(labels)), values, color='#4393c3', edgecolor='white')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Number of Projects', fontsize=12)
ax.set_title('B. Top Purpose Combinations', fontsize=13, fontweight='bold')
for bar, v in zip(bars, values):
    ax.text(v + 1, bar.get_y() + bar.get_height()/2, str(v), va='center', fontsize=9)
ax.set_xlim(0, max(values) * 1.2)
ax.grid(axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(REPORT_IMG, 'figure3_purpose_composition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure3_purpose_composition.png")

# Purpose by state
purpose_by_state = defaultdict(Counter)
for r in records:
    purposes = [p.strip() for p in r['purpose'].split(',')]
    for p in purposes:
        purpose_by_state[purpose_map.get(p, p)][r['state']] += 1

with open(os.path.join(OUTPUTS, 'purpose_by_state.json'), 'w') as f:
    json.dump({k: dict(v) for k, v in purpose_by_state.items()}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# 4. AGENT-APPARATUS DEPLOYMENT PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== AGENT-APPARATUS DEPLOYMENT PATTERNS ===")

# Normalize agents
def normalize_agent(agent_str):
    """Classify agent into main categories."""
    parts = [a.strip().lower() for a in agent_str.split(',')]
    has_si = any('silver iodide' in p for p in parts)
    has_na = any('sodium iodide' in p for p in parts)
    has_ca = any('calcium chloride' in p for p in parts)
    has_hygro = any('hygroscopic' in p for p in parts)
    has_dryice = any('dry ice' in p for p in parts)
    has_ammonium = any('ammonium' in p for p in parts)
    has_propane = any('propane' in p for p in parts)
    has_ionized = any('ionized' in p for p in parts)
    
    cats = []
    if has_si: cats.append('Silver Iodide')
    if has_na: cats.append('Sodium Iodide')
    if has_ca: cats.append('Calcium Chloride')
    if has_hygro: cats.append('Hygroscopic')
    if has_dryice: cats.append('Dry Ice')
    if has_ammonium: cats.append('Ammonium')
    if has_propane: cats.append('Propane')
    if has_ionized: cats.append('Ionized Air')
    
    if not cats:
        return agent_str[:40]
    
    # Simplify
    if 'Silver Iodide' in cats:
        extras = [c for c in cats if c != 'Silver Iodide']
        if extras:
            return f"Silver Iodide + {extras[0]}" if len(extras) == 1 else "Silver Iodide + others"
        return "Silver Iodide only"
    
    return ' + '.join(cats)

# Normalize apparatus
def normalize_apparatus(app_str):
    parts = [a.strip().lower() for a in app_str.split(',')]
    has_ground = any('ground' in p for p in parts)
    has_airborne = any('airborne' in p for p in parts)
    if has_ground and has_airborne:
        return 'Ground + Airborne'
    elif has_ground:
        return 'Ground only'
    elif has_airborne:
        return 'Airborne only'
    else:
        return app_str[:20]

for r in records:
    r['agent_cat'] = normalize_agent(r['agent'])
    r['apparatus_cat'] = normalize_apparatus(r['apparatus'])

# Agent categories
agent_cats = Counter(r['agent_cat'] for r in records)
apparatus_cats = Counter(r['apparatus_cat'] for r in records)

# Cross-tabulation
cross_tab = defaultdict(Counter)
for r in records:
    cross_tab[r['agent_cat']][r['apparatus_cat']] += 1

# Export
agent_data = [{'agent': k, 'count': v, 'pct': round(100*v/len(records), 2)}
              for k, v in agent_cats.most_common()]
apparatus_data = [{'apparatus': k, 'count': v, 'pct': round(100*v/len(records), 2)}
                  for k, v in apparatus_cats.most_common()]

with open(os.path.join(OUTPUTS, 'agent_distribution.json'), 'w') as f:
    json.dump(agent_data, f, indent=2)
with open(os.path.join(OUTPUTS, 'apparatus_distribution.json'), 'w') as f:
    json.dump(apparatus_data, f, indent=2)

# Cross-tabulation matrix
agent_labels = [a['agent'] for a in agent_data]
app_labels = ['Ground only', 'Airborne only', 'Ground + Airborne']
matrix = np.zeros((len(agent_labels), len(app_labels)))
for i, ag in enumerate(agent_labels):
    for j, ap in enumerate(app_labels):
        matrix[i, j] = cross_tab.get(ag, Counter()).get(ap, 0)

crosstab_export = []
for i, ag in enumerate(agent_labels):
    row = {'agent': ag}
    for j, ap in enumerate(app_labels):
        row[ap] = int(matrix[i, j])
    crosstab_export.append(row)
with open(os.path.join(OUTPUTS, 'agent_apparatus_crosstab.json'), 'w') as f:
    json.dump(crosstab_export, f, indent=2)

print("Agent categories:")
for a in agent_data:
    print(f"  {a['agent']}: {a['count']} ({a['pct']}%)")
print("Apparatus categories:")
for a in apparatus_data:
    print(f"  {a['apparatus']}: {a['count']} ({a['pct']}%)")

# Figure: Agent-Apparatus Patterns
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Agent distribution
ax = axes[0]
labels_a = [a['agent'] for a in agent_data]
values_a = [a['count'] for a in agent_data]
colors_a = plt.cm.Set2(np.linspace(0, 1, len(labels_a)))
wedges, texts, autotexts = ax.pie(values_a, labels=labels_a, autopct='%1.1f%%',
                                   colors=colors_a, startangle=140,
                                   textprops={'fontsize': 8})
ax.set_title('A. Seeding Agent Distribution', fontsize=13, fontweight='bold')

# Panel B: Apparatus distribution
ax = axes[1]
labels_b = [a['apparatus'] for a in apparatus_data]
values_b = [a['count'] for a in apparatus_data]
colors_b = ['#66c2a5', '#fc8d62', '#8da0cb']
wedges, texts, autotexts = ax.pie(values_b, labels=labels_b, autopct='%1.1f%%',
                                   colors=colors_b[:len(labels_b)], startangle=140,
                                   textprops={'fontsize': 10})
ax.set_title('B. Deployment Apparatus Distribution', fontsize=13, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(REPORT_IMG, 'figure4_agent_apparatus.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure4_agent_apparatus.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. CROSS-TABULATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=app_labels, yticklabels=agent_labels,
            cbar_kws={'label': 'Number of Projects'}, ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_xlabel('Deployment Apparatus', fontsize=12)
ax.set_ylabel('Seeding Agent', fontsize=12)
ax.set_title('Agent–Apparatus Cross-Tabulation\nCloud Seeding Projects, 2000–2025', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(REPORT_IMG, 'figure5_crosstab_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure5_crosstab_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. SEASONAL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SEASONAL PATTERNS ===")

season_counter = Counter()
for r in records:
    seasons = [s.strip() for s in r['season'].split(',')]
    for s in seasons:
        season_counter[s] += 1

season_data = [{'season': k, 'count': v} for k, v in season_counter.most_common()]
with open(os.path.join(OUTPUTS, 'season_distribution.json'), 'w') as f:
    json.dump(season_data, f, indent=2)

print("Season breakdown:")
for s in season_data:
    print(f"  {s['season']}: {s['count']}")

# Season by state
season_by_state = defaultdict(Counter)
for r in records:
    seasons = [s.strip() for s in r['season'].split(',')]
    for s in seasons:
        season_by_state[s][r['state']] += 1


# ═══════════════════════════════════════════════════════════════════════════
# 7. OPERATOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== OPERATOR ANALYSIS ===")

operator_counter = Counter(r['operator_affiliation'] for r in records)
operator_data = [{'operator': k, 'count': v, 'pct': round(100*v/len(records), 2)}
                 for k, v in operator_counter.most_common()]
with open(os.path.join(OUTPUTS, 'operator_distribution.json'), 'w') as f:
    json.dump(operator_data, f, indent=2)

# Figure: Top operators bar chart
fig, ax = plt.subplots(figsize=(12, 6))
top_ops = operator_data[:12]
labels = [o['operator'].title()[:50] for o in top_ops]
values = [o['count'] for o in top_ops]
bars = ax.barh(range(len(labels)), values, color='steelblue', edgecolor='white')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Number of Projects', fontsize=12)
ax.set_title('Top Operator Affiliations (2000–2025)', fontsize=14, fontweight='bold')
for bar, v in zip(bars, values):
    ax.text(v + 1, bar.get_y() + bar.get_height()/2, str(v), va='center', fontsize=9)
ax.set_xlim(0, max(values) * 1.2)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(REPORT_IMG, 'figure6_operator_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure6_operator_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8. SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SUMMARY STATISTICS ===")

summary = {
    'total_records': len(records),
    'year_range': f"{min(years)}–{max(years)}",
    'num_years': len(years),
    'num_states': len(state_counts),
    'states': sorted(state_map.keys()),
    'unique_projects': len(set(r['project'].strip().lower() for r in records)),
    'unique_operators': len(operator_counter),
    'avg_annual_projects': round(np.mean(counts), 1),
    'std_annual_projects': round(np.std(counts), 1),
    'max_annual': max(counts),
    'max_year': years[counts.index(max(counts))],
    'min_annual': min(counts),
    'min_year': years[counts.index(min(counts))],
    'top_state': state_dist[0]['state'],
    'top_state_count': state_dist[0]['count'],
    'top_state_pct': state_dist[0]['pct'],
    'dominant_purpose': purpose_data[0]['purpose'],
    'dominant_agent': agent_data[0]['agent'],
    'dominant_apparatus': apparatus_data[0]['apparatus'],
}

with open(os.path.join(OUTPUTS, 'summary_statistics.json'), 'w') as f:
    json.dump(summary, f, indent=2)

for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n=== ANALYSIS COMPLETE ===")
print(f"Outputs saved to: {OUTPUTS}/")
print(f"Figures saved to: {REPORT_IMG}/")
