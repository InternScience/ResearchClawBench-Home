# Experiment Plan

**Problem**: Reproduce the target paper's empirical claims about U.S. cloud-seeding activity from the released NOAA project records alone.
**Method Thesis**: Transparent descriptive analysis of the published structured records should recover the paper's core findings on spatial concentration, annual dynamics, operational purposes, and agent-apparatus deployment patterns without external data.
**Date**: 2026-04-09

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1 | Reported activity is geographically concentrated rather than evenly distributed. | State-level counts and a map showing strong skew toward a few western states. | B1 |
| C2 | Annual activity varies materially over time rather than remaining flat. | Yearly counts with a clear peak-trough range and year-over-year changes. | B2 |
| C3 | Operational purposes and delivery methods are dominated by a limited set of patterns. | Purpose shares plus agent-apparatus cross-tab showing repeated dominant combinations. | B3, B4 |

## Paper Storyline
- Main paper must prove: the released dataset alone reproduces the paper's descriptive empirical structure.
- Appendix can support: operator concentration and project-duration context.
- Experiments intentionally cut: causal inference, meteorological effectiveness, and any claims requiring external validation data.

## Experiment Blocks

### Block 1: Spatial concentration
- Claim tested: activity records are concentrated in a small number of states.
- Why this block exists: this is one of the task's mandatory empirical dimensions.
- Dataset / split / task: full CSV, grouped by state, joined with provided U.S. states GeoJSON.
- Compared systems: not comparative; descriptive reproduction from source data.
- Metrics: state counts, state shares, top-3 concentration share.
- Setup details: lowercase-normalized state names, no record filtering beyond null handling.
- Success criterion: top states dominate the distribution and the map is visually concentrated.
- Failure interpretation: if shares are diffuse, the paper's concentration claim is not reproduced.
- Table / figure target: `outputs/state_counts.csv`, `report/images/state_concentration_map.png`
- Priority: MUST-RUN

### Block 2: Annual dynamics
- Claim tested: activity volume changes over the 2000-2025 period.
- Why this block exists: the benchmark requires annual activity evidence.
- Dataset / split / task: full CSV grouped by year.
- Compared systems: not comparative; descriptive trend recovery.
- Metrics: yearly counts, peak year, trough year, year-over-year change.
- Setup details: integer year aggregation from structured column.
- Success criterion: substantial variation across years with an interpretable trajectory.
- Failure interpretation: if counts are nearly flat, dynamic claims should be weakened.
- Table / figure target: `outputs/annual_activity.csv`, `report/images/annual_activity.png`
- Priority: MUST-RUN

### Block 3: Purpose composition
- Claim tested: a limited number of purposes dominate reported activities.
- Why this block exists: purpose composition is a mandatory benchmark deliverable.
- Dataset / split / task: full CSV grouped by purpose string.
- Compared systems: not comparative; direct composition audit.
- Metrics: purpose counts and shares.
- Setup details: preserve original combined-purpose strings; additionally interpret comma-separated tokens in narrative.
- Success criterion: one or a few purpose categories account for most records.
- Failure interpretation: if purpose shares are fragmented, composition claims should be toned down.
- Table / figure target: `outputs/purpose_counts.csv`, `report/images/purpose_composition.png`
- Priority: MUST-RUN

### Block 4: Agent-apparatus deployment patterns
- Claim tested: a small set of seeding agents and apparatus combinations dominate operations.
- Why this block exists: the task explicitly requires agent-apparatus deployment evidence.
- Dataset / split / task: comma-split agent and apparatus fields, then cross-tabulate.
- Compared systems: not comparative; pattern extraction from released structured fields.
- Metrics: token counts for agents/apparatus and pair-frequency matrix.
- Setup details: normalize tokens to lowercase and whitespace-collapse before pairing.
- Success criterion: silver iodide and ground/airborne deployment dominate counts and pair matrix.
- Failure interpretation: if no dominant combinations emerge, deployment-pattern claims should be narrowed.
- Table / figure target: `outputs/agent_apparatus_pairs.csv`, `report/images/agent_apparatus_heatmap.png`
- Priority: MUST-RUN

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Validate schema and local inputs | CSV inspection, PDF triage | Columns and years align with task brief | Low | Low |
| M1 | Generate core descriptive tables | Analysis script aggregation | Required tables are written to `outputs/` | Low | Low |
| M2 | Generate figure-level evidence | Map, time-series, bar chart, heatmap | At least one PNG figure exists and renders | Low | Medium |
| M3 | Claim gate | Compare recovered evidence to intended claims | Claims remain descriptive and data-supported | Low | Medium |
| M4 | Report writing | Write benchmark report with relative figure refs | `report/report.md` complete | Low | Low |

## Compute and Data Budget
- Total estimated GPU-hours: 0
- Data preparation needs: CSV parsing and lightweight token normalization
- Human evaluation needs: none
- Biggest bottleneck: possible schema irregularities in multi-valued text fields

## Risks and Mitigations
- Risk: purpose and agent fields contain inconsistent combined strings.
- Mitigation: report both raw grouped categories and normalized token-level summaries.
- Risk: local literature corpus may be noisy or only partly related to the target paper.
- Mitigation: use local PDFs only for high-level framing and keep all substantive claims grounded in the dataset.

## Final Checklist
- [x] Main paper tables are covered
- [x] Novelty is isolated
- [x] Simplicity is defended
- [x] Frontier contribution is explicitly not claimed
- [x] Nice-to-have runs are separated from must-run runs
