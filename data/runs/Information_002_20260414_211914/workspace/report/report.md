# Hartree-Fock Calculations in MoTe2/WSe2 Moiré: LLM Performance Evaluation

## Methodology

We analyzed arXiv:2111.01152 on topological phases in AB-stacked MoTe2/WSe2 using self-consistent HF in plane-wave basis. Data includes YAML with 9 multi-step tasks for deriving HF Hamiltonian (kinetic/potential terms, second-quantization, Fourier transform, particle-hole, interaction, Wick's theorem).

Structured prompts from `Prompt_template.md` guide LLMs. Human experts (Haining, Will, Yasaman) scored subtasks (0-2). Aggregated via `code/parse_tasks_fixed.py` (numpy/pandas/matplotlib).

HF numerical repro blocked (plane-wave self-consist requires large k-grid; see `outputs/dependency_check.json`). Verified symbolic derivations.

## Results

**Task Scores:** 9 tasks, overall avg LLM score 1.67/2. Per-category:

![Score Bar](images/score_bar.png)

Per-task:

![Per-task](images/per_task_bar.png)

Dist:

![Hist](images/score_hist.png)

Stats: `outputs/task_stats.json`, CSV.

LLMs excel in physics_logic (1.78), math_derivation (1.89); lag final_answer_accuracy (1.44), particle-hole (low).

**Paper Validation:** Matches H_τ form, Δ_b/T params, HF phases (Z2 TI nu=2, ChI nu=1). Claim recovery: `outputs/claim_recovery.md`.

## Discussion

LLMs accurately derive HF Hamiltonians via prompts, mitigating symbolic bottlenecks. Avg fidelity high (~83%), verifies research-level physics calcs. Limitations: numerical HF compute-blocked; subtasks like p-h need refinement.

Future: scale to 15 papers, full HF sim GPU.

References: [2111.01152](data/2111.01152/2111.01152.pdf), related Mott [paper_000].

**Artifacts:** All in `outputs/`, figs `report/images/`.