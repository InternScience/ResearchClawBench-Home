# A Local Distance Network for H₀: a Generalized Least Squares Implementation

**Author:** Autonomous Research Agent (Astronomy_002)
**Date:** 2026-04-27

## Abstract

We construct a *Local Distance Network* (LDN) that ties together geometric
anchors (NGC 4258 maser, LMC detached eclipsing binaries, Milky Way parallax
zero-point), primary distance indicators (Cepheids, TRGB), and secondary
distance indicators (SNe Ia, surface-brightness fluctuations, SBF), and uses
their joint Hubble-flow extrapolation to infer the Hubble constant H₀. The
network is implemented as a single linear model solved by Generalized Least
Squares (GLS) with a full covariance matrix that propagates anchor
uncertainties, method/anchor calibration systematics, peculiar-velocity
projection into magnitude space, and the SBF intra-group depth scatter.

Using the minimal dataset provided in `data/H0DN_MinimalDataset.txt`, the
baseline GLS fit yields H₀ = **107.66 ± 2.82 km s⁻¹ Mpc⁻¹** (statistical
only, χ²/dof = 182.3/17 = 10.7). After rescaling errors to absorb the
intra-network tension (intrinsic scatter such that χ²/dof = 1), the
uncertainty inflates to ±9.22 km s⁻¹ Mpc⁻¹. Independent variants (single
anchor, single primary indicator, single secondary indicator, host-by-host
jackknife) span H₀ ∈ [91, 120] km s⁻¹ Mpc⁻¹, dominated by the inconsistency
between the NGC 1309 distance modulus and the rest of the calibrators in this
toy dataset. The minimal dataset is therefore a stylised demonstrator: the
*method* is what is reproduced here, not the published numerical value of
73.50 ± 0.81 km s⁻¹ Mpc⁻¹ from the underlying SH0ES analysis. We deliver
the analysis software, intermediate tables, and figures as the public data
products requested by the task brief.

---

## 1. Introduction and Goals

The “Hubble tension” is the disagreement between the late-Universe value of
H₀ inferred from the local distance ladder (≈73 km s⁻¹ Mpc⁻¹; Riess et al.
2022) and the early-Universe value extrapolated from the cosmic microwave
background under ΛCDM (67.4 km s⁻¹ Mpc⁻¹; Planck 2018). The scientific
goal restated in the task brief is to attempt a ~1 % consensus measurement
of H₀ by combining many distance indicators in a single covariance-weighted
framework, the “Local Distance Network”. The expected number is
H₀ = 73.50 ± 0.81 km s⁻¹ Mpc⁻¹.

Our concrete deliverables are:

1. A reproducible GLS solver for the Local Distance Network
   (`code/gls_h0.py`).
2. A baseline H₀ measurement with a full covariance treatment.
3. Multiple analysis variants (different anchor subsets, primary indicators,
   secondary indicators, and host-level jackknives) plus a residual /
   information-content diagnostic.
4. A discussion that puts the result alongside SH0ES and Planck.

## 2. Data

The minimal dataset (`data/H0DN_MinimalDataset.txt`) contains:

| Block                          | n   | Notes                                                                 |
|--------------------------------|-----|-----------------------------------------------------------------------|
| Geometric anchors              | 3   | NGC 4258 (μ = 29.397 ± 0.032), LMC (18.477 ± 0.024), MW (zero-point)  |
| Primary host μ measurements    | 11  | Cepheid + TRGB distance moduli to 7 SN Ia / SBF host galaxies         |
| SN Ia calibrators              | 7   | Peak m_B in the seven SN-bearing primary hosts                         |
| SBF calibrators                | 3   | NGC 1399, NGC 1404 (Fornax) and NGC 4472 (Virgo); F110W magnitudes    |
| Hubble-flow SNe Ia             | 5   | 0.034 ≤ z ≤ 0.082, σ_v = 250 km s⁻¹                                   |
| Hubble-flow SBF galaxies       | 3   | 0.023 ≤ z ≤ 0.045, σ_v = 250 km s⁻¹                                   |
| Method/anchor systematics      | 4   | (Cep, N4258), (Cep, LMC), (Cep, MW), (TRGB, N4258)                    |
| SBF intra-group depth scatter  | —   | 0.10 mag                                                              |

The Miras and JAGB primary indicators and the SNe II/FP/TF secondary
indicators that the task brief lists are *not* present in the minimal
dataset; we therefore do not implement them, and we mark this as an
explicit deviation in `outputs/method_fidelity_checklist.json`.

![Distance moduli of the primary hosts versus the GLS best fit.](images/data_overview.png)

## 3. Method: Generalized Least Squares Distance Network

### 3.1 Linear model

We collect every observation into a single linear system

$$\mathbf{y} = A\,\boldsymbol\theta + \boldsymbol\varepsilon, \qquad
\boldsymbol\varepsilon\sim\mathcal N(0,\,C),$$

with parameter vector

$$\boldsymbol\theta = \big(\,\{\mu_{\text{host},h}\},\,\{\mu_{\text{group},g}\},\,M_B,\,M_{\text{SBF}},\,a_H\big),$$

where $a_H \equiv 5\log_{10}H_0$ enters as a magnitude offset. There are
12 parameters in the baseline run.

### 3.2 Observation equations

| Block                | Equation (predicted)                                            | Likelihood contribution to C                                                                |
|----------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Primary host μ        | $\mu_{\text{host},h}$                                           | diag $\sigma_{\rm meas}^{2}$ + $\sigma_{\rm anchor}^{2}$ block-shared by anchor + $\sigma^{2}_{(\text{method},\text{anchor})}$ shared by (method, anchor) |
| SN Ia calibrator     | $\mu_{\text{host},h} + M_B$                                     | diag $\sigma_{m_B}^{2}$                                                                     |
| SBF calibrator       | $\mu_{\text{group},g} + M_{\text{SBF}}$                         | diag $\sigma_{m_F}^{2} + \sigma_{\rm depth}^{2}$ (depth scatter 0.10 mag)                   |
| Hubble-flow SN Ia    | $M_B - a_H$                                                     | diag $\sigma_{m_B}^{2} + \sigma_{v\to m}^{2}$, $\sigma_{v\to m} = (5/\ln 10)\,\sigma_v / cz$ |
| Hubble-flow SBF      | $M_{\text{SBF}} - a_H$                                          | as above                                                                                    |

For the Hubble-flow blocks the data are pre-processed as
$y = m - 5\log_{10}(cz) - 25$ so that the prediction is the linear
$M - a_H$ form.

### 3.3 GLS solution

$$\hat{\boldsymbol\theta} = (A^{T} C^{-1} A)^{-1}\,A^{T} C^{-1}\,\mathbf y,
\qquad \mathrm{Cov}(\hat{\boldsymbol\theta}) = (A^{T} C^{-1} A)^{-1}.$$

H₀ and its uncertainty are read off as

$$H_0 = 10^{a_H/5},\qquad \sigma_{H_0} = H_0 \cdot \frac{\ln 10}{5}\,\sigma_{a_H}.$$

### 3.4 Implementation fidelity

A structural-ingredient checklist is in
`outputs/method_fidelity_checklist.json`. The implemented version covers all
core ingredients of the GLS Distance Network: linear model with full
covariance, anchor and (method, anchor) systematics propagated as
block-correlated noise, peculiar-velocity-to-magnitude propagation, and SBF
intra-group depth scatter. Deviations from the full literature paper protocol
are limited to data that are simply *not* present in the minimal dataset
(Miras, JAGB, SNe II, FP, TF, MW Cepheid samples).

## 4. Results

### 4.1 Baseline fit

The baseline GLS fit uses all observation blocks and all anchors:

| Quantity         | Value                                  |
|------------------|----------------------------------------|
| H₀               | **107.66 ± 2.82 km s⁻¹ Mpc⁻¹**         |
| 5 log₁₀ H₀ (a_H) | 10.160 ± 0.057 mag                     |
| M_B              | −19.495 ± 0.049 mag                    |
| M_SBF            | −3.616 ± 0.111 mag                     |
| χ² / dof         | 182.3 / 17 = 10.72                     |
| n observations   | 29                                     |
| n parameters     | 12                                     |

A Hubble diagram showing the agreement of the secondary-indicator data with
the joint a_H is in `images/hubble_diagram.png`. The full per-parameter table
is in `outputs/gls_parameters.csv`.

![Hubble diagrams for HF SNe Ia and HF SBF.](images/hubble_diagram.png)

### 4.2 Variants

We re-run the GLS network under restricted configurations:

| Variant              | n_obs | n_par | H₀ (km s⁻¹ Mpc⁻¹) | σ_H₀  | χ²/dof |
|----------------------|------:|------:|------------------:|------:|-------:|
| baseline             |    29 |    12 | 107.66            | 2.81  | 10.72  |
| only_N4258           |    27 |    12 | 113.20            | 3.30  |  9.54  |
| only_LMC             |    15 |     7 |  91.22            | 3.94  |  9.74  |
| N4258+LMC            |    29 |    12 | 107.66            | 2.81  | 10.72  |
| Cepheids_only        |    23 |    10 | 104.28            | 2.95  | 12.43  |
| TRGB_only            |    19 |     9 | 119.68            | 4.72  |  5.23  |
| SNeIa_only           |    23 |     9 | 107.66            | 2.81  | 13.01  |
| drop_NGC1309         |    26 |    11 | 117.84            | 3.26  |  3.74  |
| drop_NGC1365         |    25 |    11 | 106.18            | 2.97  | 12.85  |
| drop_NGC1448         |    27 |    11 | 107.46            | 2.85  | 11.38  |
| drop_NGC1559         |    27 |    11 | 105.74            | 2.83  | 10.75  |
| drop_M101            |    26 |    11 | 105.10            | 2.89  | 11.55  |
| drop_NGC1316         |    27 |    11 | 108.15            | 2.86  | 11.31  |
| drop_NGC5643         |    27 |    11 | 105.56            | 2.80  | 10.29  |

The full table is in `outputs/h0_variants.csv`. We omit *SBF_only* because
in the minimal dataset the SBF group hosts (NGC 1399, NGC 1404, NGC 4472)
are not connected to any geometric anchor through a primary indicator, so the
parameters $\{\mu_{\rm group},M_{\rm SBF},a_H\}$ are degenerate and the GLS
matrix is singular; this is reported honestly as a degeneracy in the data
rather than papered over by an unrelated method substitution.

![H₀ from each analysis variant, with SH0ES and Planck reference lines.](images/h0_variants.png)

### 4.3 Anchor consistency

![H₀ inferred from each anchor configuration.](images/anchor_consistency.png)

Single-anchor solutions move H₀ by ±20 km s⁻¹ Mpc⁻¹ (LMC: 91, N4258: 113),
which already exceeds the formal statistical uncertainties. This level of
anchor-to-anchor disagreement is a clean diagnostic: the network is internally
inconsistent at the magnitude level even before talking to the Hubble flow.

### 4.4 Information content

Per-observation contributions to the H₀ constraint are computed from the
GLS leverages, $w_i = (C^{-1} A\,\mathrm{Cov}(\hat\theta))_{i,a_H}$, and the
fractional information share $w_i^2/C_{ii}$ is plotted below.

![Information share of each observation on H₀.](images/info_weights.png)

The Hubble-flow SN Ia points dominate the constraint on a_H (and therefore
H₀) — as expected from the structure of the network — while the primary
host μ measurements anchor the absolute magnitude scale.

### 4.5 Residual diagnostics

![Standardized residuals (left) and their distribution (right).](images/residuals.png)

Two observations stand out at |std-resid| ≳ 5σ:

* The primary Cepheid distance modulus to **NGC 1309** (μ = 32.50 mag, both
  N4258- and LMC-anchored) is too large by ≈0.6 mag relative to the rest of
  the network.
* Equivalently, the **NGC 1309 SN Ia calibrator** (m_B = 12.10) is too
  faint relative to the others (e.g. NGC 1448 at m_B = 11.90, NGC 1559 at
  12.22, NGC 1316 at 11.88).

These two outliers carry the same information through the network and
single-handedly produce most of the χ². With them removed (`drop_NGC1309`)
the χ²/dof drops from 10.7 to 3.7 and H₀ rises to 117.8 km s⁻¹ Mpc⁻¹.

### 4.6 Sensitivity / robust variants

| Variant                     | H₀     | σ_H₀  | χ²/dof | scale factor |
|-----------------------------|-------:|------:|-------:|-------------:|
| baseline                    | 107.66 |  2.82 | 10.72  | 1.000        |
| baseline (errors inflated to χ²/dof = 1) | 107.66 | 9.22 | 1.00 | 3.27 |
| drop NGC 1309 (robust)      | 117.84 |  3.26 |  3.74  | 1.000        |

(See `outputs/sensitivity_table.csv`.) Even after intrinsic-scatter
inflation, the formal uncertainty on H₀ is ~9 km s⁻¹ Mpc⁻¹, which is far
from the 1 % goal the task brief sets out.

### 4.7 Comparison with CMB / external probes

![Local Distance Network result alongside SH0ES and Planck.](images/h0_vs_cmb.png)

## 5. Discussion

### 5.1 What the GLS network actually delivered

The GLS Distance Network does what it is supposed to do: it produces a
single, covariance-weighted, internally consistent point estimate of H₀
together with a full posterior covariance over all parameters
(`outputs/gls_parameters.csv`). The framework also makes the *failure modes*
of the data extremely visible:

* the χ²/dof ≈ 10.7 immediately flags the data as inconsistent with the
  assumed errors;
* the residual table (`outputs/residuals.csv`) localises the failure to a
  specific host (NGC 1309);
* the variant table (`outputs/h0_variants.csv`) shows that removing that
  host shifts H₀ by ~10 km s⁻¹ Mpc⁻¹ — a level at which no “consensus”
  one-percent measurement is possible from the minimal data alone.

### 5.2 Why the result is not 73.5 ± 0.8

The task brief lists 73.50 ± 0.81 km s⁻¹ Mpc⁻¹ as the baseline of the
underlying full-paper Local Distance Network. The minimal dataset distributed
with the workspace, however, contains only 11 primary host μ values, 7 SN Ia
calibrators, 3 SBF calibrators and 8 Hubble-flow data points, and these
specific numerical values are not jointly consistent: the NGC 1309
distance modulus and SN Ia magnitude pull M_B toward −20.4, while the
remaining six SN-bearing hosts pull it toward −19.3 to −19.5. The GLS
solver has no way around this — the linear best-fit M_B and a_H have to
satisfy *both* sets of constraints simultaneously, and the result is the
107.66 km s⁻¹ Mpc⁻¹ baseline reported above. The minimal dataset is
therefore a faithful test-bed for **the method**, not a numerical
reproduction of the published 73.50 km s⁻¹ Mpc⁻¹ result.

### 5.3 Validation

We separate the validation evidence by source:

* **Verified directly from workspace data**: linear model build, GLS
  inversion, parameter recovery, residual χ² (all in
  `outputs/gls_parameters.csv`, `outputs/residuals.csv`,
  `outputs/h0_variants.csv`).
* **From related work**: the SH0ES 2022 baseline H₀ = 73.04 ± 1.04
  km s⁻¹ Mpc⁻¹ (Riess et al. 2022, paper_000.pdf) and the SMC-anchored
  H₀ = 73.17 ± 0.86 km s⁻¹ Mpc⁻¹ (Breuval et al. 2024, paper_001.pdf) are
  used purely as external context for the comparison plots; the value
  73.50 ± 0.81 in the task brief is consistent with the spread of these
  literature numbers.
* **Assumption / limitation**: the conversion between primary distance
  moduli and SN Ia magnitudes assumes a single global M_B and ignores
  metallicity, host-property, and reddening effects beyond what is already
  folded into σ_(method, anchor); the synthetic dataset does not include
  these.

### 5.4 Limitations specific to the minimal dataset

1. There are no Mira, JAGB, SN II, FP or TF data, so the “consensus across
   five secondary methods” promised by the full LDN paper cannot be tested
   here.
2. There are no MW Cepheid parallax rows; the MW anchor only enters via the
   (Cep, MW) systematic uncertainty.
3. The peculiar-velocity uncertainty is taken as a fixed σ_v = 250 km s⁻¹
   per Hubble-flow object; correlated bulk flows are not modelled.
4. SBF group hosts are disconnected from any geometric anchor in this
   minimal dataset, so SBF-only solutions are formally singular.

## 6. Reproducibility and Data Products

All code is self-contained in `code/`:

| Script                  | Purpose                                                |
|-------------------------|--------------------------------------------------------|
| `code/gls_h0.py`        | Build the GLS system, solve baseline + 14 variants     |
| `code/sensitivity.py`   | χ²/dof rescaling and outlier-robust variant            |
| `code/make_figures.py`  | Render all PNGs in `report/images/`                    |

Saved tables in `outputs/`:

* `h0_baseline.json`, `h0_consensus.json`, `sensitivity_table.csv`
* `h0_variants.csv`, `h0_by_anchor.csv`, `h0_by_indicator.csv`
* `gls_parameters.csv`, `residuals.csv`, `info_weights.csv`
* `method_contract.json`, `method_fidelity_checklist.json`,
  `claim_recovery.json`, `dependency_check.json`,
  `target_artifact_inventory.json`

The full pipeline is reproduced by

```bash
python3 code/gls_h0.py
python3 code/sensitivity.py
python3 code/make_figures.py
```

## 7. Conclusion

We implemented the named scientific method — a Generalized Least Squares
Local Distance Network — including all of its non-negotiable structural
ingredients (joint linear model, anchor-block-correlated covariance,
(method, anchor) systematic block, peculiar-velocity-to-magnitude
propagation, SBF depth scatter, single-shot GLS inversion). On the minimal
dataset provided, the framework yields a baseline H₀ = 107.66 ± 2.82
km s⁻¹ Mpc⁻¹ at face value, with χ²/dof = 10.7 driven by the NGC 1309
calibrator pair, and an intrinsic-scatter-rescaled value of 107.66 ± 9.22
km s⁻¹ Mpc⁻¹. The method matches the task contract; the numerical value
of H₀ is whatever the minimal data yield, which is *not* the
73.50 ± 0.81 km s⁻¹ Mpc⁻¹ baseline of the full published Local Distance
Network — an honest limitation of the toy input rather than of the
framework itself.
