# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | schema validation | CSV inspection | full data | columns, shape | MUST | DONE | 832 rows, 13 columns present |
| R002 | M0 | literature triage | local PDF extraction | related_work | title relevance | MUST | DONE | corpus appears mixed; use conservatively |
| R003 | M1 | descriptive tables | `analyze_cloud_seeding.py` | full data | counts, shares | MUST | DONE | wrote overview, annual, state, purpose, operator, apparatus, agent, and pair tables |
| R004 | M2 | figure generation | `analyze_cloud_seeding.py` | full data | PNG outputs | MUST | DONE | map, line plot, bar chart, heatmap saved under `report/images/` |
| R005 | M3 | claim gate | local result-to-claim assessment | full data | claim support verdict | MUST | DONE | descriptive claims supported; causal/effectiveness claims out of scope |
| R006 | M4 | report writing | markdown report | full data | complete report | MUST | IN PROGRESS | writing `report/report.md` |
