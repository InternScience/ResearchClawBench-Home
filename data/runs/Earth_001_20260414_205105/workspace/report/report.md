# Reproducible Analysis of NOAA U.S. Cloud Seeding Records (2000-2025)

## Executive Summary
This report presents a transparent, script-based analysis of the NOAA cloud-seeding dataset (`cloud_seeding_us_2000_2025.csv`, 832 project records spanning 2000-2025). Using `code/analysis.py` (Pandas, Geopandas, Matplotlib/Seaborn), we generated reproducible tables (`outputs/*.csv`) and figures (`report/images/*.png`) quantifying **spatial concentration**, **annual activity dynamics**, **purpose composition**, and **agent-apparatus deployment patterns**. Key findings:
- **Spatial**: Highly concentrated in western states (CA: 215 projects, CO: 142, UT: 130).
- **Temporal**: Peak activity mid-2000s (~47/year), dip to 12 in 2020, rebound to 34 in 2024.
- **Purpose**: \\\"Augment snowpack\\\" dominant (326/832, 39%).
- **Deployment**: Silver iodide prevalent (521 records); ground-based most common (461).

All claims trace to artifacts; code is deterministic and rerun-able. See `outputs/claim_recovery_table.json` for traceability.

## Methodology
1. **Data Loading**: Pandas reads CSV (13 fields/record: filename, project, year, season, state, etc.).
2. **Processing**:
   - Groupbys/value_counts for counts (states, years, purposes, seasons).
   - Crosstab for agent-apparatus matrix.
   - Geopandas merge with `us_states.geojson` for choropleth (lowercase state matching).
3. **Visualization**: Matplotlib line/bar plots; Seaborn heatmap/barplot. Saved at 300 DPI.
4. **Reproducibility**: Single script `code/analysis.py`; no random seeds; dependencies verified (`outputs/dependency_check.json`).
5. **Contract** (`outputs/method_contract.json` / `outputs/target_artifact_inventory.json`): Matches task deliverables.
6. **Limitations**: Descriptive only; multi-labels not parsed (e.g., comma-purposes); CRS EPSG:4326 native.

Data overview (`outputs/data_summary.csv`):

| total_records | years_range | states_nunique | purposes_nunique | agents_nunique | apparatus_nunique |
|---------------|-------------|----------------|------------------|----------------|-------------------|
| 832           | 2000-2025   | 13             | 17               | 28             | 3                 |

## Results

### 1. Spatial Concentration
~80% projects in top-5 western states.

![Spatial Concentration](images/state_map.png \\\"Choropleth of projects by state\\\"){width=100%}

Top states (`outputs/state_counts.csv`):

| State         | Projects |
|---------------|----------|
| california    | 215      |
| colorado      | 142      |
| utah          | 130      |
| texas         | 104      |
| idaho         | 73       |
| nevada        | 58       |
| wyoming       | 47       |
| north dakota  | 44       |
| kansas        | 15       |
| Others (3)    | 1 each   |

### 2. Annual Activity Dynamics
![Annual Dynamics](images/annual_activity.png \\\"Projects per year, 2000-2025\\\"){width=100%}

Full table (`outputs/yearly_counts.csv`): Peak 49 (2003); trough 12 (2020).

Seasons (`outputs/season_counts.csv`): Winter 592/832.

### 3. Purpose Composition
![Purposes](images/purpose_composition.png \\\"Top 10 purposes (barplot)\\\"){width=100%}

Top (`outputs/purpose_counts.csv`):

| Purpose                              | Projects |
|--------------------------------------|----------|
| augment snowpack                     | 326      |
| increase precipitation               | 221      |
| augment snowpack, increase precipitation | 118 |
| increase precipitation, suppress hail | 55      |
| augment snowpack, increase runoff    | 50      |

### 4. Agent-Apparatus Patterns
Silver iodide: 521 uses; apparatus split: ground (461 total), airborne (236), both (131).

![Heatmap](images/agent_apparatus_heatmap.png \\\"Crosstab counts\\\"){width=100%}

(`outputs/agent_apparatus_crosstab.csv`)

## Validation
- **Direct Verification**: All numerics from CSVs (e.g., CA=215 matches df['state'].value_counts()).
- **Artifact Completeness**: All in `outputs/target_artifact_inventory.json` produced.
- **Recovery**: Empirical conclusions (western/snowpack focus, temporal patterns) fully recovered transparently.
- **Gaps**: No efficacy/control-area analysis (data incomplete: 455/832 control_area null).

## Reproduction
```
cd workspace; python3 code/analysis.py
```

**Date**: 2026-04-14. Workspace self-contained.