# Research Plan

## Phase 1: Understand the Data and Problem
- The goal is to compute $H_0$ with ~1% precision using a "Local Distance Network".
- We have a minimal dataset (`data/H0DN_MinimalDataset.txt`) containing:
  - Anchors (geometric distance moduli): N4258, LMC, MW
  - Primary indicators: Cepheids and TRGB measurements for various host galaxies
  - Secondary calibrators: SNe Ia and SBF
  - Hubble flow measurements: SNe Ia and SBF
  - Calibration uncertainties and physical constants
- The method is a covariance-weighted approach (generalized least squares).
- We need to formulate a linear system or a likelihood function to jointly fit the distances to all galaxies, the absolute magnitudes of the standard candles (SNe Ia, SBF), and the Hubble constant $H_0$.

## Phase 2: Formulate the Generalized Least Squares (GLS) Model
- Parameters to fit:
  - $\mu_i$: Distance moduli for each host galaxy (NGC1309, NGC1365, etc.) and anchors (which have prior measurements).
  - $M_{B}$: Absolute magnitude of SNe Ia.
  - $M_{F110W}$: Absolute magnitude of SBF.
  - $H_0$: Hubble constant (or $a_B = \log_{10}(c z) - 0.2 m_B$ intercept). Actually, from Hubble flow SNe Ia and SBF, we can relate $m$ to $\mu$ and $H_0$.
    - For a source in the Hubble flow: $\mu = 5 \log_{10}(c z / H_0) + 25 = m - M$.
    - So $m = M + 5 \log_{10}(c z) - 5 \log_{10}(H_0) + 25$.
- We will construct a large vector of measurements $Y$, a design matrix $A$, and a covariance matrix $C$.
- The GLS solution is $\hat{X} = (A^T C^{-1} A)^{-1} A^T C^{-1} Y$.

## Phase 3: Implement the Code
- Write a Python script to parse `H0DN_MinimalDataset.txt`.
- Construct $Y$, $A$, and $C$.
- Solve for $\hat{X}$ and its covariance $\Sigma = (A^T C^{-1} A)^{-1}$.
- Extract $H_0$ and its uncertainty.
- Perform variants (e.g., only Cepheids, only TRGB, exclude SBF, different anchors).

## Phase 4: Generate Figures
- Huble diagram (distance vs redshift) for SNe Ia and SBF.
- Distance ladder plot (Anchors -> Hosts -> Hubble flow).
- Corner plot or error bar plot for $H_0$ from different variants.

## Phase 5: Write Report
- Methodology: Describe the GLS framework and the distance network.
- Results: Present the baseline $H_0$ and variants.
- Discussion: Compare with CMB and discuss the tension.
