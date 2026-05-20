# Can Early Dark Energy Alleviate the Acoustic Tension Between CMB and BAO?

## A Study Using DESI DR2, Planck, and ACT Data

---

## Abstract

We investigate whether an Early Dark Energy (EDE) model can alleviate the acoustic tension between measurements from the cosmic microwave background (CMB) and baryon acoustic oscillations (BAO). Using the best-fit cosmological parameters from DESI DR2 combined with CMB data from Planck and ACT, we compare the performance of three cosmological models: the standard ΛCDM model, the EDE extension, and the late-time dark energy parametrization w₀wₐ. Our analysis demonstrates that the EDE model achieves a Hubble constant of H₀ = 70.9 ± 1.0 km/s/Mpc, reducing the tension with SH0ES from 3.3σ (ΛCDM) to 1.4σ, while maintaining competitive fits to BAO data. The EDE model requires a peak energy fraction of f_EDE = 0.093 ± 0.031 at a critical scale log₁₀(a_c) = −3.564 ± 0.075, consistent with the epoch of matter-radiation equality. In contrast, the w₀wₐ model pushes H₀ to lower values (63.5 ± 1.9 km/s/Mpc), exacerbating the Hubble tension (4.0σ). These results indicate that EDE provides a physically motivated mechanism to partially resolve the CMB-BAO acoustic tension, though at the cost of shifting other cosmological parameters in ways that are increasingly constrained by large-scale structure data.

---

## 1. Introduction

### 1.1 The Hubble and Acoustic Tensions

The standard cosmological model, ΛCDM, has been remarkably successful in describing the large-scale structure and evolution of the Universe. However, a persistent discrepancy has emerged between the value of the Hubble constant H₀ inferred from early-Universe probes (particularly CMB anisotropies) and that measured through local distance-ladder techniques using Type Ia supernovae (SNe Ia). The Planck 2018 CMB analysis yields H₀ = 67.36 ± 0.54 km/s/Mpc [1], while the SH0ES collaboration measures H₀ = 73.04 ± 1.04 km/s/Mpc [2], representing a tension at the 4–5σ level.

This tension is not merely a disagreement in a single parameter. It reflects a deeper inconsistency in the acoustic scale calibration: the angular size of the sound horizon at decoupling, θ_s = r_s(z_*) / D_A(z_*), is measured to 0.03% precision by Planck, creating tight constraints on any modification to the early-Universe expansion history. Any resolution must simultaneously reduce the sound horizon r_s while preserving the measured angular scale θ_s, a requirement that severely limits possible explanations.

The second data release (DR2) from the Dark Energy Spectroscopic Instrument (DESI) has provided new baryon acoustic oscillation (BAO) measurements spanning the redshift range 0.3 < z < 2.3 [3]. These data test the expansion history at multiple epochs and have revealed intriguing hints that the dark energy equation of state may deviate from the cosmological constant (w = −1), with the DESI team reporting preferences for dynamical dark energy models at 2–3σ significance [3].

### 1.2 Early Dark Energy as a Resolution

Early Dark Energy (EDE) represents a class of models where a scalar field contributes a significant fraction of the total energy density at early times (near matter-radiation equality, z ~ 3000–5000) before rapidly diluting away [4, 5]. The key mechanism is as follows:

1. The additional energy density at early times increases the Hubble rate H(z) at z ≫ z_*, which reduces the comoving sound horizon r_s = ∫ c_s dz / H(z).
2. This reduction in r_s allows a proportionally larger H₀ (and correspondingly smaller D_A) while keeping the angular scale θ_s = r_s / D_A fixed.
3. After the EDE field begins oscillating and its energy density dilutes faster than matter, the late-time expansion history returns to that of standard ΛCDM.

The canonical EDE model is described by an axion-like potential V(θ) = m²f²[1 − cos(θ)]³, with three additional parameters beyond ΛCDM: the peak energy fraction f_EDE, the critical redshift z_c (or equivalently a_c = 1/(1 + z_c)), and the initial field value θ_i [4]. The field becomes dynamical when H(z_c) ≈ m, after which the energy density dilutes as a^{−6} (for n = 3 in the potential).

### 1.3 Current Status of EDE Constraints

Recent analyses have painted a complex picture of EDE viability:

- **Poulin et al. (2019)** [4]: First detailed analysis showing that EDE can bring CMB-inferred H₀ into agreement with SH0ES while maintaining good fits to BAO and SNe data, with f_EDE ≈ 0.10 needed.
- **Ivanov et al. (2020)** [6]: Showed that large-scale structure data from BOSS, using EFT-based full-shape analysis, constrains f_EDE < 0.072 (95% CL), below the benchmark value.
- **McDonough et al. (2024)** [7]: Comprehensive review finding tight upper bounds on f_EDE from Planck NPIPE combined with BOSS and weak lensing data, with Bayesian analyses preferring ΛCDM.
- **Poulin et al. (2025)** [8]: Using ACT DR6 and DESI DR2, showed that EDE partially relieves the tension, with residual tension with SH0ES of only ~2σ for the combined P-ACT + lensing + Pantheon-plus + DESI DR2 dataset.

This paper contributes to this ongoing discussion by systematically comparing EDE, ΛCDM, and w₀wₐ models using the DESI DR2 cosmological parameter constraints.

---

## 2. Data and Methodology

### 2.1 Datasets

Our analysis is based on the following data, as compiled in the DESI DR2 EDE paper (Poulin et al. 2025):

**CMB Data:**
- Planck PR4 (NPIPE) high-ℓ TT, TE, EE power spectra [1]
- Planck low-ℓ TT (Commander) and EE (SROLL2) likelihoods
- ACT DR6 high-ℓ TTTEEE data [9]
- Planck and ACT DR6 gravitational lensing reconstructions

**BAO Data (DESI DR2):**
- BAO measurements at seven redshift bins: z = 0.295, 0.510, 0.700, 0.934, 1.100, 1.320, 2.330 [3]
- Measurements include the spherically-averaged distance D_V/r_d and the Alcock-Paczynski parameter F_AP

**Supernovae Data:**
- Pantheon-plus compilation [10]
- Union3 dataset [11] (used in some analyses)

### 2.2 Models

We compare three cosmological models:

1. **ΛCDM**: The standard model with 6 parameters: ω_b, ω_cdm, θ_s, A_s, n_s, τ_reio.
2. **EDE**: ΛCDM plus three EDE parameters: f_EDE, log₁₀(a_c), θ_i.
3. **w₀wₐ**: ΛCDM with a time-varying dark energy equation of state: w(a) = w₀ + wₐ(1 − a).

### 2.3 Parameter Estimation

We use the best-fit parameter values and 1σ uncertainties from the combined CMB + DESI BAO analysis, as reported in Tables II and III of the DESI DR2 EDE paper [8]. The parameters and their constraints are summarized below.

### 2.4 Derived Quantities

From the base parameters, we compute several derived quantities:

- **S₈ ≡ σ₈(Ω_m/0.3)^{0.5}**: The amplitude of matter fluctuations, which is sensitive to both σ₈ and Ω_m and is strongly constrained by weak lensing surveys.
- **Hubble tension significance**: The discrepancy between model-predicted and SH0ES-measured H₀, quantified in units of combined uncertainty σ.

---

## 3. Results

### 3.1 Cosmological Parameter Constraints

Table 1 summarizes the best-fit parameters and 1σ uncertainties for each model from the CMB + DESI DR2 analysis.

| Parameter | ΛCDM | EDE | w₀wₐ |
|-----------|------|-----|------|
| Ω_m | 0.3037 ± 0.0037 | 0.2999 ± 0.0038 | 0.353 ± 0.021 |
| H₀ (km/s/Mpc) | 68.12 ± 0.28 | 70.9 ± 1.0 | 63.5 ± 1.9 |
| σ₈ | 0.8101 ± 0.0055 | 0.8283 ± 0.0093 | 0.780 ± 0.016 |
| n_s | 0.9672 ± 0.0034 | 0.9817 ± 0.0063 | 0.9632 ± 0.0037 |
| ω_b | 0.02229 ± 0.00012 | 0.02241 ± 0.00018 | 0.02218 ± 0.00013 |
| ln(10¹⁰A_s) | 3.056 ± 0.014 | 3.067 ± 0.017 | 3.037 ± 0.013 |
| τ | 0.0621 ± 0.0075 | 0.0582 ± 0.0074 | 0.0520 ± 0.0071 |
| f_EDE | — | 0.093 ± 0.031 | — |
| log₁₀(a_c) | — | −3.564 ± 0.075 | — |
| w₀ | — | — | −0.42 ± 0.21 |
| wₐ | — | — | −1.75 ± 0.58 |

*Table 1: Best-fit cosmological parameters with 1σ uncertainties from the CMB + DESI DR2 analysis.*

**Key observations:**

1. **Hubble constant**: The EDE model yields H₀ = 70.9 ± 1.0 km/s/Mpc, which is 2.8 km/s/Mpc higher than ΛCDM. This shift is achieved through the reduction in the sound horizon at decoupling. The w₀wₐ model gives H₀ = 63.5 ± 1.9 km/s/Mpc, moving in the opposite direction.

2. **Matter density**: Ω_m is similar between ΛCDM (0.3037) and EDE (0.2999), but substantially higher in w₀wₐ (0.353). This reflects the different physical mechanisms: EDE modifies the early Universe, while w₀wₐ changes the late-time expansion.

3. **Spectral index**: n_s is higher in the EDE model (0.9817 vs 0.9672), reflecting the increased early-universe expansion rate which affects the tilt of the primordial power spectrum needed to fit the CMB peaks.

4. **EDE parameters**: The preferred f_EDE = 0.093 is close to the benchmark value of ~0.10 identified by Poulin et al. (2019) as needed to resolve the Hubble tension. The critical scale a_c corresponds to z_c ≈ 3600, consistent with matter-radiation equality.

### 3.2 Derived Quantities

From the base parameters, we compute:

| Derived Quantity | ΛCDM | EDE | w₀wₐ |
|-----------------|------|-----|------|
| S₈ | 0.8151 ± 0.0074 | 0.8282 ± 0.0107 | 0.8461 ± 0.0306 |

*Table 2: Derived S₈ values for the three models.*

The EDE model has a slightly higher S₈ than ΛCDM, reflecting the increased σ₈ that accompanies the EDE-induced parameter shifts. This is a key tension point: while EDE alleviates the Hubble tension, the associated increase in S₈ can create tension with weak lensing measurements, which prefer S₈ ≈ 0.76–0.79 [7].

### 3.3 Hubble Tension with SH0ES

We quantify the tension between each model's H₀ prediction and the SH0ES measurement (H₀ = 73.52 ± 1.62 km/s/Mpc):

| Model | H₀ (km/s/Mpc) | |ΔH₀|/σ_combined | Tension |
|-------|---------------|-----------------|---------|
| ΛCDM | 68.12 ± 0.28 | 5.40/1.64 | **3.3σ** |
| EDE | 70.9 ± 1.0 | 2.62/1.90 | **1.4σ** |
| w₀wₐ | 63.5 ± 1.9 | 10.02/2.52 | **4.0σ** |

*Table 3: Tension between model H₀ and SH0ES measurement.*

**Figure 4** visualizes these results, showing that EDE significantly reduces the Hubble tension from 3.3σ to 1.4σ—a factor of ~2.4 improvement. In contrast, the w₀wₐ model exacerbates the tension to 4.0σ. This demonstrates that early-Universe modifications (EDE) and late-Universe modifications (w₀wₐ) can have qualitatively different effects on the Hubble tension.

### 3.4 EDE Parameter Posterior

**Figure 3** shows the posterior distributions of the two primary EDE parameters:

- **f_EDE = 0.093 ± 0.031**: The preferred EDE energy fraction at the critical epoch. This is consistent with the benchmark value of ~0.10 identified in the original EDE proposal [4], and represents the amount of extra radiation needed to reduce the sound horizon sufficiently.

- **log₁₀(a_c) = −3.564 ± 0.075**: The critical scale at which the EDE field becomes dynamical, corresponding to z_c ≈ 3600 ± 600. This is near matter-radiation equality (z_eq ≈ 3400), a physically motivated location: the EDE must be active before recombination to affect the sound horizon, but its effects on structure formation are minimized if it dilutes near or before matter-radiation equality.

### 3.5 BAO Data Comparison

**Figure 2** compares the DESI DR2 BAO measurements with the predictions of the three models. The data points show Δ(D_V/r_d) relative to the fiducial model. All three models provide reasonable fits to the low-to-intermediate redshift data (z < 1.5), but differences become apparent at the highest redshift bin (z = 2.330), which probes the Lyman-α forest and is particularly sensitive to the early-universe physics.

The EDE model produces a distinctive signature in the BAO measurements: the reduced sound horizon systematically shifts the distance-redshift relation, but the effect is partially compensated by the increased H₀, resulting in a net improvement in the overall fit to the BAO data compared to ΛCDM.

### 3.6 Goodness-of-Fit

Following the DESI DR2 analysis [8], we report the Δχ² values (model − ΛCDM) for the combined CMB + DESI BAO + lensing + Pantheon-plus dataset:

| Model Combination | Δχ² (vs ΛCDM) | Interpretation |
|-------------------|-----------------|----------------|
| EDE (CMB+DESI) | −8.5 | Mild preference for EDE |
| EDE (+SH0ES) | −35.4 | Strong preference for EDE |
| w₀wₐ (CMB+DESI) | −3.2 | Marginal preference for w₀wₐ |

*Table 4: Goodness-of-fit comparison. Negative Δχ² indicates the alternative model fits better.*

**Figure 5** illustrates these results. The key finding is that EDE provides a notably better fit than ΛCDM when SH0ES data are included (Δχ² = −35.4), representing very strong statistical preference. Even without SH0ES, EDE shows a mild preference (Δχ² = −8.5). The w₀wₐ model shows only marginal improvement.

This difference in Δχ² values reflects the fundamental distinction between early-time and late-time solutions:
- EDE modifies the sound horizon directly, allowing the model to fit both CMB and local H₀ measurements simultaneously.
- w₀wₐ modifies the late-time expansion but cannot change the sound horizon, so it cannot simultaneously fit CMB acoustic scale data and a high H₀.

### 3.7 EDE-H₀ Degeneracy

**Figure 6** illustrates the key degeneracy in the EDE model: the correlation between f_EDE and H₀. This degeneracy is the physical basis for EDE's ability to resolve the Hubble tension:

- Larger f_EDE → more early-time energy density → smaller sound horizon r_s → larger H₀ consistent with θ_s.
- The SH0ES measurement (H₀ ≈ 73 km/s/Mpc) is accessible for f_EDE ≈ 0.09–0.12.
- The ΛCDM limit (f_EDE = 0) recovers the standard H₀ ≈ 68 km/s/Mpc.

The degeneracy direction is approximately linear: H₀ ≈ 68.1 + 32 × f_EDE km/s/Mpc, indicating that each 1% increase in f_EDE raises H₀ by approximately 0.32 km/s/Mpc.

---

## 4. Discussion

### 4.1 EDE as a Partial Resolution

Our analysis confirms that the EDE model provides a physically motivated partial resolution to the acoustic tension between CMB and local H₀ measurements. The key results are:

1. **H₀ improvement**: EDE raises the inferred H₀ from 68.1 to 70.9 km/s/Mpc, reducing the SH0ES tension from 3.3σ to 1.4σ.

2. **CMB fit maintained**: The EDE model maintains or slightly improves the fit to CMB data by exploiting the f_EDE-n_s degeneracy: larger f_EDE requires larger n_s, which better matches the observed CMB peak structure.

3. **BAO consistency**: The EDE model provides a competitive fit to DESI DR2 BAO data, with the altered expansion history partially compensating the reduced sound horizon.

4. **Preferred parameter values**: The posterior peaks at f_EDE ≈ 0.09 and z_c ≈ 3600, consistent with the theoretical expectation that EDE should be active near matter-radiation equality.

### 4.2 Comparison with w₀wₐ

The contrast between EDE and w₀wₐ illustrates a fundamental point about the nature of the tensions:

- **EDE (early-time)**: Modifies the sound horizon → can resolve the CMB-H₀ tension → but shifts other parameters (n_s, S₈) that may create new tensions with LSS data.

- **w₀wₐ (late-time)**: Modifies the distance-redshift relation → cannot change the sound horizon → instead prefers lower H₀ and higher Ω_m to fit the same CMB acoustic scale → exacerbates the Hubble tension.

This asymmetry highlights that the CMB acoustic scale is a "locking mechanism" that constrains late-time modifications more tightly than early-time modifications.

### 4.3 Challenges for EDE

Despite its successes, the EDE model faces several challenges:

1. **S₈ tension**: The EDE model predicts S₈ = 0.828 ± 0.011, which is higher than the ΛCDM value and in mild tension with weak lensing measurements (S₈ ≈ 0.76–0.79 from DES, HSC, and KV-450 [7, 12]). This is because the increased n_s and ω_cdm needed to fit CMB data in the EDE model enhance structure growth.

2. **Prior-volume effects**: As discussed by McDonough et al. (2024) [7], Bayesian analyses of EDE can be affected by prior-volume effects, particularly when SH0ES is not included. The profile likelihood analysis by Poulin et al. (2025) [8] found that f_EDE = 0.09 ± 0.03 from the profile likelihood, compared to weaker Bayesian constraints, suggesting prior-volume effects are non-negligible.

3. **Theoretical naturalness**: The EDE model requires fine-tuning of the scalar field mass and potential to produce the desired energy density at the right epoch. While UV completions in string theory have been proposed [13], the model remains a phenomenological parametrization rather than a fully motivated particle physics scenario.

4. **Large-scale structure constraints**: BOSS full-shape and Lyman-α data from eBOSS place strong constraints on f_EDE, with upper limits of f_EDE < 0.053–0.072 (95% CL) depending on the analysis [6, 7], which is below the benchmark value of ~0.10.

### 4.4 Implications of DESI DR2

The DESI DR2 BAO data provide crucial new information for discriminating between models:

1. **Extended redshift coverage**: DESI DR2 spans z = 0.3–2.3, compared to BOSS DR12 at z = 0.3–0.6. This wider range breaks degeneracies between different dark energy models more effectively.

2. **Hints of dynamical dark energy**: DESI DR2 has reported 2–3σ evidence for w₀ ≠ −1 or wₐ ≠ 0 when combined with CMB and SNe data [3]. However, our analysis shows that this preference does not resolve the Hubble tension—in fact, the w₀wₐ model prefers H₀ = 63.5 km/s/Mpc, further from SH0ES than ΛCDM.

3. **Consistency between early and late probes**: The DESI DR2 data improve consistency between CMB and BAO measurements in the EDE model, as shown by Poulin et al. (2025) [8], with EDE providing better concordance between these probes than ΛCDM.

### 4.5 Future Prospects

Several upcoming observations will further constrain the EDE scenario:

- **DESI full survey**: The complete DESI dataset will significantly improve BAO precision, particularly at z > 1.5, where the EDE signature is most distinct.
- **Euclid**: The Euclid satellite will provide weak lensing measurements that tightly constrain S₈, potentially ruling out the EDE parameter space where S₈ is enhanced.
- **CMB-S4**: Next-generation CMB experiments will measure the damping tail with unprecedented precision, constraining the EDE effects on perturbation growth.
- **Roman**: The Nancy Grace Roman Space Telescope will provide independent SNe Ia and BAO measurements at z > 1.

As shown by Ivanov et al. (2020) [6], Euclid-like surveys could constrain f_EDE < 0.01 (95% CL), which would effectively rule out the EDE resolution of the Hubble tension.

---

## 5. Conclusions

We have investigated whether Early Dark Energy can alleviate the acoustic tension between CMB and BAO measurements using data from DESI DR2, Planck, and ACT. Our main findings are:

1. **EDE partially resolves the Hubble tension**: The EDE model yields H₀ = 70.9 ± 1.0 km/s/Mpc, reducing the discrepancy with SH0ES from 3.3σ (ΛCDM) to 1.4σ. The preferred EDE fraction is f_EDE = 0.093 ± 0.031, consistent with the theoretical benchmark.

2. **w₀wₐ exacerbates the tension**: Late-time dark energy dynamics push H₀ in the wrong direction (63.5 ± 1.9 km/s/Mpc, 4.0σ tension with SH0ES), demonstrating that early-time and late-time modifications have qualitatively different effects on the Hubble tension.

3. **EDE fits the data competitively**: The EDE model provides a better fit to combined CMB+BAO data than ΛCDM (Δχ² ≈ −8.5), with dramatically improved concordance when SH0ES is included (Δχ² ≈ −35.4).

4. **Challenges remain**: The EDE-induced shifts in other cosmological parameters (particularly the increase in S₈ and n_s) create tension with large-scale structure data, and the model faces theoretical concerns about naturalness.

5. **DESI DR2 is decisive for model comparison**: The extended redshift coverage of DESI DR2 breaks degeneracies between early-time and late-time models, showing that while DESI hints at dynamical dark energy, only early-time modifications like EDE can address the Hubble tension.

In summary, EDE represents a viable and physically motivated mechanism to partially alleviate the CMB-BAO acoustic tension, though it is not a complete resolution. The tension between CMB and local H₀ measurements remains the strongest evidence for physics beyond ΛCDM, and future surveys will be decisive in determining whether EDE or alternative models provide the correct explanation.

---

## References

[1] Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters," *Astron. Astrophys.* 641, A6 (2020).

[2] A. G. Riess et al., "A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team," *ApJL* 934, L7 (2022).

[3] DESI Collaboration, "DESI 2024 IV: Baryon Acoustic Oscillations from Galaxies and Quasars," arXiv:2404.03000 (2024); DESI DR2 (2025).

[4] V. Poulin, T. L. Smith, T. Karwal, and M. Kamionkowski, "Early Dark Energy Can Resolve The Hubble Tension," *Phys. Rev. Lett.* 122, 221301 (2019).

[5] T. L. Smith, V. Poulin, and M. A. Amin, "Early Dark Energy constraints from the evolution of cosmic perturbations," *Phys. Rev. D* 101, 063523 (2020).

[6] M. M. Ivanov, E. McDonough, J. C. Hill, M. Simonović, M. W. Toomey, S. Alexander, and M. Zaldarriaga, "Constraining Early Dark Energy with Large-Scale Structure," *Phys. Rev. D* 102, 103502 (2020).

[7] E. McDonough, J. C. Hill, M. M. Ivanov, A. La Posta, and M. W. Toomey, "Observational Constraints on Early Dark Energy," arXiv:2401.10900 (2024).

[8] V. Poulin, T. L. Smith, R. Calderón, and T. Simon, "Impact of ACT DR6 and DESI DR2 for Early Dark Energy and the Hubble Tension," arXiv:2503.xxxxx (2025).

[9] ACT Collaboration, "The Atacama Cosmology Telescope: DR6 Power Spectra," arXiv:2503.xxxxx (2025).

[10] D. Scolnic et al., "The Pantheon+ Analysis: The Full Dataset and Light-Curve Release," *ApJ* 938, 113 (2022).

[11] M. Rubin et al., "Cosmological constraints from the Union3 supernova compilation," arXiv:2311.xxxxx (2023).

[12] DES Collaboration, "Dark Energy Survey Year 3 Results: Cosmological Constraints from Galaxy Clustering and Weak Lensing," *Phys. Rev. D* 105, 023520 (2022).

[13] V. Poulin, T. L. Smith, D. Grin, and T. Karwal, "Probing the Dark Energy Axion," arXiv:1911.xxxxx (2019).

---

## Appendix: Figure Descriptions

**Figure 1** ([figure1_parameter_comparison.png](images/figure1_parameter_comparison.png)): Comparison of key cosmological parameters (Ω_m, H₀, σ₈) across the three models, showing the Planck fiducial values and SH0ES H₀ constraint.

**Figure 2** ([figure2_bao_comparison.png](images/figure2_bao_comparison.png)): DESI DR2 BAO measurements (Δ(D_V/r_d) and ΔF_AP) compared with the predictions of the three models across the full redshift range.

**Figure 3** ([figure3_ede_posteriors.png](images/figure3_ede_posteriors.png)): Posterior distributions of the EDE parameters f_EDE and log₁₀(a_c), showing the preferred values and their relationship to the matter-radiation equality epoch.

**Figure 4** ([figure4_hubble_tension.png](images/figure4_hubble_tension.png)): Visual comparison of the Hubble constant predicted by each model versus the SH0ES measurement, with tension significance indicated.

**Figure 5** ([figure5_goodness_of_fit.png](images/figure5_goodness_of_fit.png)): (Left) Parameter shifts of EDE and w₀wₐ relative to ΛCDM in units of ΛCDM 1σ; (Right) Goodness-of-fit comparison showing Δχ² values.

**Figure 6** ([figure6_ede_degeneracy.png](images/figure6_ede_degeneracy.png)): The f_EDE-H₀ degeneracy direction, showing how increasing EDE energy fraction systematically raises H₀, with the SH0ES band indicating the target range.

**Figure 7** ([figure7_sne_comparison.png](images/figure7_sne_comparison.png)): Union3 supernovae distance modulus residuals compared with model predictions at low redshifts.
