| Claim | Evidence Artifact | Verified? |
|-------|-------------------|-----------|
| LLMs achieve avg score 1.7/2 on HF Hamiltonian derivation steps | outputs/task_stats.json (mean~1.7 across categories) | [Y] |
| Strong on math_derivation/physics_logic (avg>1.8), weaker on final_accuracy (~1.5) | report/images/score_bar.png, task_scores.csv | [Y] |
| Bottleneck: particle-hole transformation (in_paper=0) | task_scores.csv row 'Particle-hole transformation' | [Y] |
| Single-particle H_tau topology C_+K=-1, C_-K=+1 from paper | data/2111.01152 PDF text/Fig1 | [Y] |
| HF opens gap at nu=2 for Z2 TI | PDF Fig2, dependency_check.json (numerical repro blocked) | [N] compute-intensive |
| Phases at nu=1: VP-ChI, SDW, FMx | PDF Fig3 | [Y] |
| 9 tasks parsed successfully | task_stats.json num_tasks=9 | [Y]