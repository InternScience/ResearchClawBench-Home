# Method Notes

This benchmark run uses a local-only surrogate Bayesian superradiance model designed to remain faithful to the qualitative structure described in the provided literature while avoiding unsupported claims of a full numerical GR calculation.

Core choices:

- The observational likelihood is represented directly by the provided posterior samples in black-hole mass and spin.
- Superradiance constraints are modeled as a posterior predictive exclusion probability in the black-hole mass-spin plane.
- The resonance variable is calibrated to reproduce the literature-supported scaling that stellar-mass black holes probe ultralight bosons near `10^-13` to `10^-11 eV`, while supermassive black holes probe approximately `10^-20` to `10^-16 eV`.
- Self-interaction strength is represented by an effective coupling coordinate `lambda_eff`; stronger self-interactions suppress net spin extraction and therefore weaken exclusion.

Interpretation discipline:

- The locations of the constrained mass bands are data-informed and literature-calibrated.
- The precise shape of the exclusion contours and the mapping from `lambda_eff` to a microscopic particle-physics parameter are surrogate assumptions for this benchmark.
- Reported outputs should therefore be read as statistically rigorous within the surrogate model, not as a replacement for a full Teukolsky-based or fully relativistic superradiance computation.
