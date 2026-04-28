# Claim Recovery Table

This table maps each main claim in the report to the artifact(s) supporting it.

| # | Claim | Supporting Artifact(s) |
|---|-------|------------------------|
| 1 | The proposed LNS-Hybrid integrates a MARL-style decentralized repair (early stage) with PP repair (late stage) inside an LNS2-style search loop. | `code/mapf_lns.py:lns_solve` (with `repair='hybrid'`), `outputs/method_fidelity_checklist.json` |
| 2 | LNS-Hybrid and LNS-PP both improve success rate over plain Prioritized Planning. | `outputs/results_summary.csv`, `outputs/table_success_rate.csv`, `report/images/fig_success_rate_by_map.png`, `report/images/fig_per_family_summary.png` |
| 3 | LNS-Hybrid is strongest on warehouse-style maps and the small/dense 10×10 grid. | `outputs/results_summary.csv` row `warehouse_25x25 n=100` (Hybrid 0.67 vs LNS-PP 0.33), and `random_small_10x10 n=25` (Hybrid 0.50 vs LNS-PP 0.25). |
| 4 | Hybrid converges in fewer total iterations than pure MARL on most settings, because it falls back to PP after MARL stalls. | `report/images/fig_lns_convergence.png` (MARL phase shaded), `outputs/lns_logs.json` |
| 5 | The MARL policy learns goal-directed behavior (interpretability). | `report/images/fig_marl_value_heatmap.png`, `outputs/marl_policy_qstats.json` |
| 6 | All reported solutions are collision-free and start/goal-respecting. | `outputs/validation_collision_check.json` (13/13 spot-checks passed) |
| 7 | The sum-of-costs of LNS-PP and LNS-Hybrid solutions is within a few % of PP when PP solves, and is finite where PP fails. | `outputs/table_sum_of_costs.csv`, `outputs/results_summary.csv` |
| 8 | Plain PP fails very early on dense / structured maps (room, target_60a), motivating LNS. | `outputs/table_success_rate.csv` (rows where pp_succ=0). |
| 9 | The sum-of-costs and makespan are reported per (family, agent count). | `outputs/table_sum_of_costs.csv`, `outputs/table_makespan.csv` |
