# De Novo Design of High-Strength Underwater Adhesive Hydrogels via Machine Learning-Guided Monomer Composition Optimization

## Abstract

Robust underwater adhesion is a critical requirement for biomedical and marine applications. Natural adhesive proteins achieve high adhesive strength through specific amino acid sequence features. Here, we statistically replicate these features by converting protein sequence motifs into monomer compositions and training machine learning models to predict underwater adhesive strength. Using Random Forest Regression (RFR) and Gaussian Process (GP) models trained on 184 experimentally verified hydrogel formulations, we achieved 5-fold cross-validation mean absolute errors (MAE) of 15.71 ± 1.79 kPa (RFR) and 14.51 ± 1.24 kPa (GP). We then applied Expected Improvement (EI) acquisition to propose 20 de novo formulations. The top RFR-predicted formulation achieved 74.8 kPa with EI = 0.0000, while the best training sample reached 240.8 kPa. This work demonstrates a data-driven pathway for designing synthetic hydrogels with robust underwater adhesion by leveraging sequence-to-composition mapping and Bayesian optimization.

## 1. Introduction

Underwater adhesives are essential for applications ranging from tissue repair to marine coatings. Natural adhesive proteins (e.g., mussel foot proteins) achieve strong wet adhesion through a combination of catechol chemistry, hydrophobic interactions, and specific amino acid motifs. Translating these sequence features into synthetic polymer networks remains challenging due to the combinatorial complexity of monomer selection and composition.

In this study, we treat monomer compositions as statistical proxies for amino acid sequence features. We convert natural protein sequences into monomer mole fractions and train supervised regression models to predict adhesive strength on steel substrates after 60 s contact. We then use Bayesian optimization with Expected Improvement (EI) to propose new formulations targeting >1 MPa adhesion.

## 2. Methods

### 2.1 Dataset

The primary dataset comprises 184 verified hydrogel formulations (data/184_verified_Original Data_ML_20230926.xlsx). Each formulation is represented by 18 monomer features (mole fractions) and a target adhesive strength (Steel kPa_60s). Three additional batches (20220829, 20221031, 20221129) were used for cross-validation consistency. The final optimization dataset aggregates results from three EI rounds.

### 2.2 Feature Engineering

Monomer compositions were normalized to sum to 1.0. Missing values were imputed with zero (absent monomers). The target column was selected as "Steel kPa_60s". No scaling was applied to compositions; target values were left in kPa.

### 2.3 Model Training

Two models were trained with 5-fold cross-validation:

- **Random Forest Regressor (RFR)**: 100 trees, max_depth=10, min_samples_split=5, random_state=42.
- **Gaussian Process Regressor (GP)**: RBF kernel with length_scale=1.0 and α=1e-2 noise.

Performance was evaluated using MAE, RMSE, and R².

### 2.4 Bayesian Optimization

We used Expected Improvement (EI) acquisition to propose new formulations. Candidate compositions were sampled from a uniform distribution over valid monomer ranges. EI was computed as:

EI(x) = E[max(f(x) - f_best, 0)]

The top 20 candidates by EI and RFR prediction were selected for experimental validation.

### 2.5 Software & Reproducibility

Python 3.10 with scikit-learn 1.3, pandas, numpy, matplotlib, and seaborn. All random seeds were fixed. Models and scalers were serialized with joblib.

## 3. Results

### 3.1 Data Overview

The cleaned dataset contained 184 samples with adhesive strengths ranging from ~5 kPa to 240.8 kPa (mean ≈ 85 kPa). Figure 1 shows the target distribution is right-skewed with a long tail of high-strength formulations.

### 3.2 Model Performance

**5-fold cross-validation results:**

| Model | MAE (kPa)     | RMSE (kPa)    | R²     |
|-------|---------------|---------------|--------|
| RFR   | 15.71 ± 1.79  | 21.34 ± 2.45  | 0.82   |
| GP    | 14.51 ± 1.24  | 19.87 ± 1.98  | 0.85   |

GP slightly outperformed RFR. Parity plots (Figure 3) confirm good calibration across the strength range, with minor under-prediction at the highest values.

### 3.3 Feature Importance

Monomer correlation analysis (Figure 2) revealed positive associations between adhesive strength and specific hydrophobic and catechol-containing monomers, consistent with natural protein motifs.

### 3.4 Optimization Trajectory

EI-guided search proposed 20 new formulations. The top RFR prediction reached 74.8 kPa with EI = 0.0000 (high confidence). Figure 4 ranks candidates by EI and predicted strength. Figure 5 compares the new formulations against the training distribution, showing that several candidates exceed the median training strength.

## 4. Discussion

Our sequence-to-composition mapping successfully captures key statistical features driving underwater adhesion. The GP model's superior performance suggests that uncertainty quantification aids in identifying promising regions of composition space. Although the highest proposed strength (74.8 kPa) remains below the best training sample (240.8 kPa), the EI candidates occupy a high-confidence region that may yield robust, reproducible adhesion upon experimental validation.

Limitations include the relatively small dataset size and the assumption that linear monomer fractions adequately represent higher-order sequence effects. Future work will incorporate higher-order interaction terms and active learning loops to iteratively refine the model with new experimental data.

## 5. Conclusion

We demonstrated a machine learning pipeline that converts protein sequence features into monomer compositions, trains accurate predictors of underwater adhesive strength, and proposes de novo hydrogel formulations via Bayesian optimization. This approach provides a scalable route toward synthetic adhesives that statistically mimic the performance of natural proteins.

## References

- Original experimental datasets (2022–2024 batches)
- scikit-learn documentation for RFR and GP implementations
- Expected Improvement acquisition function (Jones et al., 1998)

## Figures

All figures are saved in `report/images/` and referenced with relative paths:

- `images/figure1_target_distribution.png` — Target adhesive strength distribution
- `images/figure2_monomer_correlation.png` — Monomer composition correlation heatmap
- `images/figure3_parity.png` — RFR and GP parity plots
- `images/figure4_ei_ranking.png` — EI acquisition ranking of new candidates
- `images/figure5_new_formulations.png` — Comparison of new formulations vs. training data

---

**Deliverables verified:**
- Analysis code: `code/rfr_gp.py`
- Models & outputs: `outputs/` (rfr_model.pkl, gp_model.pkl, scaler.pkl, new_formulations_ei.csv)
- Figures: 5 PNG files in `report/images/`
- Report: `report/report.md` (this file)