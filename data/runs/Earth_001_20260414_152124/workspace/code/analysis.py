#!/usr/bin/env python3
"""Analysis of NOAA cloud-seeding records 2000-2025."""

import csv
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Ensure output dirs
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
with open('data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Loaded {len(rows)} records")

# ---- Normalize fields ----
def normalize_purpose(p):
    """Normalize purpose to primary categories."""
    p = p.lower().strip()
    parts = [x.strip() for x in p.replace(';', ',').split(',')]
    return parts

def normalize_agent(a):
    """Normalize agent to primary categories."""
    a = a.lower().strip()
    if 'silver iodide' in a:
        return 'silver iodide'
    if 'water' in a:
        return 'water'
    if 'dry ice' in a or 'carbon dioxide' in a:
        return 'dry ice / CO2'
    if 'calcium chloride' in a:
        return 'calcium chloride'
    if 'ammonium iodide' in a:
        return 'ammonium iodide'
    return a

def normalize_apparatus(ap):
    if ap == 'nan' or not ap:
        return 'unknown'
    return ap.strip().lower()

def primary_purpose(p):
    """Get primary purpose from comma-separated list."""
    parts = normalize_purpose(p)
    for part in parts:
        if 'augment snowpack' in part or 'increase runoff' in part:
            return 'augment snowpack / runoff'
        if 'increase precipitation' in part:
            return 'increase precipitation'
        if 'suppress hail' in part:
            return 'suppress hail'
        if 'suppress fog' in part:
            return 'suppress fog'
        if 'research' in part:
            return 'research'
    return parts[0] if parts else 'other'

# ---- 1. Records per year ----
year_counts = Counter(int(r['year']) for r in rows)
years = sorted(year_counts.keys())
counts = [year_counts[y] for y in years]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(years, counts, color='#2196F3', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title('Cloud-Seeding Project Records by Year (2000–2025)', fontsize=14)
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig1_annual_records.png', dpi=150)
plt.close()
print("Saved fig1_annual_records.png")

# ---- 2. Records per state ----
state_counts = Counter(r['state'].strip().lower() for r in rows)
states_sorted = sorted(state_counts.items(), key=lambda x: -x[1])
state_names = [s[0].title() for s in states_sorted]
state_vals = [s[1] for s in states_sorted]

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(state_names)))
ax.barh(state_names[::-1], state_vals[::-1], color=colors)
ax.set_xlabel('Number of Records', fontsize=12)
ax.set_title('Cloud-Seeding Records by State (2000–2025)', fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig2_state_distribution.png', dpi=150)
plt.close()
print("Saved fig2_state_distribution.png")

# ---- 3. Purpose composition ----
purpose_counter = Counter()
for r in rows:
    p = primary_purpose(r['purpose'])
    purpose_counter[p] += 1

purposes_sorted = purpose_counter.most_common()
purpose_labels = [p[0] for p in purposes_sorted]
purpose_vals = [p[1] for p in purposes_sorted]

fig, ax = plt.subplots(figsize=(8, 6))
colors_pie = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', '#795548']
wedges, texts, autotexts = ax.pie(purpose_vals, labels=purpose_labels, autopct='%1.1f%%',
                                   colors=colors_pie[:len(purpose_labels)], startangle=140)
for t in autotexts:
    t.set_fontsize(10)
ax.set_title('Purpose Composition of Cloud-Seeding Projects', fontsize=14)
plt.tight_layout()
plt.savefig('report/images/fig3_purpose_composition.png', dpi=150)
plt.close()
print("Saved fig3_purpose_composition.png")

# ---- 4. Apparatus deployment ----
apparatus_counter = Counter(normalize_apparatus(r['apparatus']) for r in rows)
app_sorted = apparatus_counter.most_common()
app_labels = [a[0].title() for a in app_sorted]
app_vals = [a[1] for a in app_sorted]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(app_labels, app_vals, color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0'][:len(app_labels)])
ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title('Deployment Apparatus Distribution', fontsize=14)
for i, v in enumerate(app_vals):
    ax.text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig4_apparatus_distribution.png', dpi=150)
plt.close()
print("Saved fig4_apparatus_distribution.png")

# ---- 5. Seeding agent distribution ----
agent_counter = Counter(normalize_agent(r['agent']) for r in rows)
agent_sorted = agent_counter.most_common(10)
agent_labels = [a[0].title() for a in agent_sorted]
agent_vals = [a[1] for a in agent_sorted]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(agent_labels[::-1], agent_vals[::-1], color='#4CAF50')
ax.set_xlabel('Number of Records', fontsize=12)
ax.set_title('Top Seeding Agents Used (2000–2025)', fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig5_agent_distribution.png', dpi=150)
plt.close()
print("Saved fig5_agent_distribution.png")

# ---- 6. State-year heatmap ----
all_states = sorted(set(r['state'].strip().lower() for r in rows))
all_years = sorted(set(int(r['year']) for r in rows))
heatmap_data = np.zeros((len(all_states), len(all_years)))
for r in rows:
    si = all_states.index(r['state'].strip().lower())
    yi = all_years.index(int(r['year']))
    heatmap_data[si, yi] += 1

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xticks(range(len(all_years)))
ax.set_xticklabels(all_years, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(all_states)))
ax.set_yticklabels([s.title() for s in all_states], fontsize=10)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('State', fontsize=12)
ax.set_title('Cloud-Seeding Activity Heatmap: State × Year', fontsize=14)
plt.colorbar(im, ax=ax, label='Number of Records')
plt.tight_layout()
plt.savefig('report/images/fig6_state_year_heatmap.png', dpi=150)
plt.close()
print("Saved fig6_state_year_heatmap.png")

# ---- 7. Purpose by state ----
purpose_state = defaultdict(lambda: Counter())
for r in rows:
    st = r['state'].strip().lower().title()
    p = primary_purpose(r['purpose'])
    purpose_state[st][p] += 1

top_states = [s[0].title() for s in states_sorted[:8]]
all_purposes = sorted(set(primary_purpose(r['purpose']) for r in rows))
stacked_data = {p: [] for p in all_purposes}
for st in top_states:
    for p in all_purposes:
        stacked_data[p].append(purpose_state[st][p])

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(top_states))
bottom = np.zeros(len(top_states))
colors_stack = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', '#795548']
for i, p in enumerate(all_purposes):
    vals = stacked_data[p]
    ax.bar(x, vals, bottom=bottom, label=p.title(), color=colors_stack[i % len(colors_stack)])
    bottom += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels(top_states, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title('Purpose Composition by State (Top 8 States)', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig7_purpose_by_state.png', dpi=150)
plt.close()
print("Saved fig7_purpose_by_state.png")

# ---- 8. Apparatus by state ----
app_state = defaultdict(lambda: Counter())
for r in rows:
    st = r['state'].strip().lower().title()
    ap = normalize_apparatus(r['apparatus']).title()
    app_state[st][ap] += 1

app_types = sorted(set(normalize_apparatus(r['apparatus']).title() for r in rows))
stacked_app = {a: [] for a in app_types}
for st in top_states:
    for a in app_types:
        stacked_app[a].append(app_state[st][a])

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(top_states))
for i, a in enumerate(app_types):
    vals = stacked_app[a]
    ax.bar(x, vals, bottom=bottom, label=a, color=colors_stack[i % len(colors_stack)])
    bottom += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels(top_states, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title('Apparatus Deployment by State (Top 8 States)', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig8_apparatus_by_state.png', dpi=150)
plt.close()
print("Saved fig8_apparatus_by_state.png")

# ---- Save summary tables to outputs ----
summary = {
    'total_records': len(rows),
    'year_range': f"{min(int(r['year']) for r in rows)}-{max(int(r['year']) for r in rows)}",
    'unique_states': len(set(r['state'] for r in rows)),
    'states': sorted(set(r['state'].strip().lower() for r in rows)),
    'records_per_year': {str(y): year_counts[y] for y in years},
    'records_per_state': {s: state_counts[s] for s in sorted(state_counts.keys())},
    'purpose_distribution': {p: v for p, v in purpose_counter.most_common()},
    'apparatus_distribution': {a: v for a, v in apparatus_counter.most_common()},
    'agent_distribution': {a: v for a, v in agent_counter.most_common()},
}

with open('outputs/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved outputs/summary_statistics.json")

# ---- Season distribution ----
season_counter = Counter()
for r in rows:
    for s in r['season'].split(','):
        season_counter[s.strip().lower()] += 1

with open('outputs/season_distribution.json', 'w') as f:
    json.dump(dict(season_counter.most_common()), f, indent=2)

# ---- Operator distribution ----
op_counter = Counter(r['operator_affiliation'].strip().lower() for r in rows)
with open('outputs/operator_distribution.json', 'w') as f:
    json.dump({k: v for k, v in op_counter.most_common(20)}, f, indent=2)

print("\nAll analysis complete.")
