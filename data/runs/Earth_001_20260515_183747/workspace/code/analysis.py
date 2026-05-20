#!/usr/bin/env python3
"""
NOAA Cloud Seeding Records Analysis (2000-2025)
Reproduces spatial concentration, annual dynamics, purpose composition,
and agent-apparatus deployment patterns from the published dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'

# Paths
DATA_PATH = Path("data/dataset1_cloud_seeding_records/cloud_seeding_us_2000_2025.csv")
OUTPUT_DIR = Path("outputs")
IMAGE_DIR = Path("report/images")
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records with columns: {list(df.columns)}")

# Basic cleaning
df['year'] = df['year'].astype(int)
df['state'] = df['state'].str.strip().str.upper()
df['purpose'] = df['purpose'].str.strip()
df['agent'] = df['agent'].str.strip()
df['apparatus'] = df['apparatus'].fillna("Unknown")

# ============================================
# 1. ANNUAL ACTIVITY DYNAMICS
# ============================================
annual_counts = df.groupby('year').size().reset_index(name='projects')
annual_counts.to_csv(OUTPUT_DIR / "annual_activity.csv", index=False)

plt.figure(figsize=(10, 5))
sns.lineplot(data=annual_counts, x='year', y='projects', marker='o', linewidth=2)
plt.title("Annual Cloud-Seeding Projects in the United States (2000–2025)")
plt.xlabel("Year")
plt.ylabel("Number of Projects")
plt.xticks(range(2000, 2026, 2), rotation=45)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure1_annual_dynamics.png")
plt.close()
print("Saved figure1_annual_dynamics.png")

# ============================================
# 2. SPATIAL CONCENTRATION (Top 10 States)
# ============================================
state_counts = df.groupby('state').size().sort_values(ascending=False).reset_index(name='projects')
state_counts.to_csv(OUTPUT_DIR / "state_distribution.csv", index=False)

top_states = state_counts.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_states, x='projects', y='state', palette='viridis')
plt.title("Top 10 States by Number of Cloud-Seeding Projects (2000–2025)")
plt.xlabel("Number of Projects")
plt.ylabel("State")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure2_spatial_concentration.png")
plt.close()
print("Saved figure2_spatial_concentration.png")

# ============================================
# 3. PURPOSE COMPOSITION
# ============================================
purpose_counts = df.groupby('purpose').size().sort_values(ascending=False).reset_index(name='projects')
purpose_counts.to_csv(OUTPUT_DIR / "purpose_composition.csv", index=False)

plt.figure(figsize=(9, 6))
sns.barplot(data=purpose_counts, x='projects', y='purpose', palette='Set2')
plt.title("Distribution of Cloud-Seeding Projects by Stated Purpose")
plt.xlabel("Number of Projects")
plt.ylabel("Purpose")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure3_purpose_composition.png")
plt.close()
print("Saved figure3_purpose_composition.png")

# ============================================
# 4. AGENT-APPARATUS DEPLOYMENT PATTERNS
# ============================================
# Cross-tabulation
deploy_matrix = pd.crosstab(df['agent'], df['apparatus'])
deploy_matrix.to_csv(OUTPUT_DIR / "agent_apparatus_matrix.csv")

# Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(deploy_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.title("Agent × Apparatus Deployment Matrix")
plt.xlabel("Deployment Apparatus")
plt.ylabel("Seeding Agent")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(IMAGE_DIR / "figure4_agent_apparatus.png")
plt.close()
print("Saved figure4_agent_apparatus.png")

# ============================================
# 5. SUMMARY STATISTICS
# ============================================
summary = {
    "total_projects": len(df),
    "unique_states": df['state'].nunique(),
    "year_range": f"{df['year'].min()}-{df['year'].max()}",
    "most_common_purpose": purpose_counts.iloc[0]['purpose'],
    "most_common_agent": df['agent'].value_counts().idxmax(),
    "most_common_apparatus": df['apparatus'].value_counts().idxmax(),
    "top_state": state_counts.iloc[0]['state']
}
pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
print("Analysis complete. All tables and figures generated.")