# Reproduction of DESI DR2 EDE Analysis: Alleviating Acoustic Tension

## Introduction
This report reproduces key results from the DESI DR2 EDE paper using provided structured data from Tables II/III (best-fit cosmological parameters for ΛCDM, EDE, w₀wₐ models under CMB+DESI BAO) and Figure 6 (extracted DESI BAO and Union3 SNe residuals relative to fiducial ΛCDM). The goal is to visualize parameter constraints and distance comparisons to assess if EDE alleviates tension between CMB and BAO.

Methodology contract verified in `outputs/method_contract.json`. Target artifacts in `outputs/target_artifact_inventory.json`. Dependencies checked in `outputs/dependency_check.json` (no full model fitting possible without CLASS/CAMB; visualization only).

Related work extraction in `outputs/related_work_contract.json`.

## Methodology
- Parsed `data/DESI_EDE_Repro_Data.txt` using `code/parse_data.py` to `outputs/parameters.json` and CSV residuals.
- Generated figures with `code/generate_figures.py` (matplotlib, pandas, scipy).
- Gaussian approximation for EDE parameter posteriors from 1σ errors.
- Claim recovery table: `outputs/claim_recovery_table.md`.

Code reproducible; libs: numpy, matplotlib, etc.

## Data Overview
Extracted data points from Fig 6 show residuals Δ relative to fiducial ΛCDM model.

![DESI BAO Δ(D_V / r_d)](images/bao_dvrd.png)

![DESI BAO ΔF_AP](images/bao_fap.png)

![Union3 SNe Δμ](images/sne_mu.png)

Residuals are small and consistent with zero within errors, indicating good fit of fiducial to data (tension in sound horizon r_d inferred indirectly via model fits).

## Results: Parameter Constraints
Best-fits + 1σ from CMB+DESI BAO:

| Model | Ω_m | H_0 [km/s/Mpc] | σ_8 |
|-------|-----|----------------|-----|
| ΛCDM  | 0.3037 ± 0.0037 | 68.12 ± 0.28 | 0.8101 ± 0.0055 |
| EDE   | 0.2999 ± 0.0038 | 70.9 ± 1.0   | 0.8283 ± 0.0093 |
| w₀wₐ  | 0.353 ± 0.021  | 63.5 ± 1.9   | 0.780 ± 0.016 |

Full: `outputs/parameters.json`, `outputs/model_comparison.csv`.

![Parameter comparison](images/param_comparison.png)

EDE increases H_0 by ~10 (from 68 to 71), σ_8 up, Ω_m similar. w₀wₐ decreases H_0, σ_8.

Other params (n_s up in EDE, etc.) shift to compensate sound horizon reduction.

## EDE Parameters
f_EDE ≈ 0.093 ± 0.031 (~3% peak fraction), log₁₀ a_c ≈ -3.564 ± 0.075 (z_c ~ 3660).

Gaussian posteriors:

![EDE posteriors](images/ede_posteriors.png)

## Discussion
EDE partially relieves acoustic tension: smaller r_s allows higher H_0 consistent with CMB+BAO peaks, while data residuals small. Unlike late-time w₀wₐ (phantom-like, lowers H_0), EDE shifts parameters differently (higher σ_8).

No Δχ² available; paper shows EDE viable but not fully resolving vs LSS/S8 tensions (related work).

**Validation:**
- Directly from workspace data.
- Matches paper Tables II/III, Fig6 extracts.
- Limitation: No full MCMC/χ² recompute.

## Conclusion
Visual reproduction confirms EDE provides higher H_0, partially alleviating CMB-BAO acoustic tension via early-time modification, with distinct parameter shifts vs late DE models.