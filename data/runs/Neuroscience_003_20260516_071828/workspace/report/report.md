# Dynamic Feature Selection for Single-Cell Trajectory Preservation in Retinal Pigment Epithelium

## Abstract

Single-cell profiling of protein expression dynamics offers a powerful window into cellular state transitions relevant to neural lineage progression, glial activation, and neurodegeneration. However, high-dimensional feature spaces introduce confounding variation that can obscure continuous cellular trajectories. We present a systematic feature selection framework that identifies a compact subset of dynamically expressed molecular features optimal for preserving continuous cellular trajectories in single-cell protein imaging data. Applied to a dataset of 2,759 retinal pigment epithelium (RPE) cells profiled across 241 protein features via iterative indirect immunofluorescence imaging, our method identifies a minimal set of 20 features that achieves superior trajectory preservation (Spearman ρ = 0.878) compared to the full 241-feature set (ρ = 0.813), representing a 7.9% improvement while reducing dimensionality by 91.7%. The selected features are enriched for nuclear-localized cell cycle regulators (Cyclin A, Cyclin B1, CDK2, PCNA, Skp2, E2F1) and DNA content markers, consistent with the known role of cell cycle progression as a dominant axis of variation in proliferating RPE cells. Subgroup analysis reveals robust trajectory preservation in cycling cells (ρ = 0.907) with moderate performance in quiescent G0/arrested cells (ρ = 0.642), highlighting the method's sensitivity to active state transitions.

---

## 1. Introduction

Single-cell technologies have revolutionized our ability to study cellular heterogeneity and dynamic state transitions in complex tissues. In neuroscience-related contexts, understanding continuous trajectories of cell state—whether during neural lineage specification, glial activation cascades, or neurodegenerative progression—is critical for identifying disease mechanisms and therapeutic targets. However, high-dimensional single-cell measurements often contain substantial technical and biological noise that confounds trajectory inference.

Feature selection—identifying a subset of molecular features that optimally captures the underlying biological signal—is an essential preprocessing step that can reduce confounding variation, improve computational efficiency, and enhance interpretability. In the context of retinal biology, the retinal pigment epithelium (RPE) serves as a model system for studying cell cycle dynamics, senescence, and state transitions relevant to retinal degenerative diseases.

Here, we leverage a preprocessed single-cell dataset of RPE cells profiled with protein iterative indirect immunofluorescence imaging (4i) to develop and validate a dynamic feature selection framework. Our approach integrates multiple complementary metrics—Spearman correlation with ground-truth pseudotime, polynomial regression fit quality, and mutual information—into a unified dynamic score, and evaluates trajectory preservation through PCA-based pseudotime reconstruction.

---

## 2. Methods

### 2.1 Data Description

The dataset consists of 2,759 single RPE cells, each characterized by 241 protein features measured via iterative indirect immunofluorescence imaging. Each cell is annotated with four metadata variables:

- **Cell cycle phase**: G0 (n=402), G1 (n=1,128), S (n=891), G2 (n=338)
- **Cell state**: Cycling (n=2,174), Arrested (n=402), Unlabeled (n=183)
- **Annotated age**: A continuous pseudotime variable (range 0–25.07 hours, mean 6.76 ± 5.32)
- **Batch**: Two experimental batches (1: n=1,025, 2: n=1,734)

The 241 features span four cellular compartments: whole-cell edge intensity (n=96), nuclear intensity (n=49), cytoplasmic intensity (n=48), and ring/peripheral intensity (n=48). Proteins profiled include cell cycle regulators (Cyclin A, Cyclin B1, Cyclin D1, Cyclin E, CDK2, CDK4, CDK6), tumor suppressors (p53, p21, p27, p16, p14ARF, RB), signaling molecules (AKT, ERK, GSK3β, STAT3, RSK), DNA damage markers (pH2AX, pCHK1), and structural/cell adhesion proteins (Cdh1, β-Catenin).

### 2.2 Data Preprocessing

The dataset was provided in a preprocessed state with features normalized to the [0, 1] range. For downstream analysis, we applied standard scaling (z-score normalization per feature) prior to dimensionality reduction. PCA was computed on the full feature set (50 components), and a k-nearest neighbor graph (k=15) was constructed using 30 PCs for UMAP visualization.

### 2.3 Dynamic Feature Scoring

We defined a composite "dynamic score" to quantify each feature's association with the ground-truth pseudotime trajectory. For each of the 241 features, we computed:

1. **Spearman rank correlation (ρ)** with annotated age, capturing monotonic association (weight: 0.4)
2. **Polynomial regression R²** (degree 3), capturing non-linear trajectory patterns (weight: 0.3)
3. **Mutual information** with annotated age, capturing general statistical dependence (weight: 0.3, max-normalized)

The dynamic score was computed as the weighted sum:

$$\text{Dynamic Score}_i = 0.4 \cdot |\rho_i| + 0.3 \cdot R^2_i + 0.3 \cdot \frac{\text{MI}_i}{\max(\text{MI})}$$

Multiple testing correction was applied using the Benjamini-Hochberg FDR procedure on Spearman p-values.

### 2.4 Trajectory Preservation Evaluation

To evaluate how well a feature subset preserves the cellular trajectory, we implemented the following pipeline:

1. Select the top-k features based on the ranking strategy
2. Standard-scale the selected feature matrix
3. Apply PCA (up to 10 components) to the selected features
4. Train a linear regression model to predict ground-truth pseudotime from the PCA embedding
5. Evaluate using Spearman correlation between predicted and ground-truth pseudotime, and 5-fold cross-validated R²

We compared five feature selection strategies at subset sizes k ∈ {5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 241}:

- **Dynamic Score**: Composite metric (our method)
- **Absolute Correlation**: Ranked by |Spearman ρ|
- **Variance**: Ranked by feature variance
- **Mutual Information**: Ranked by MI with pseudotime
- **Random**: Random feature ordering (baseline)

### 2.5 Diffusion Pseudotime

As an orthogonal trajectory inference method, we computed Diffusion Pseudotime (DPT) using Scanpy's implementation. The root cell was set to the cell with minimum annotated age. DPT was computed using 10 diffusion components.

### 2.6 Interpretability Analysis

We performed permutation importance analysis on the PCA-regression model, projected linear model coefficients back to the original feature space for feature-level importance scoring, and conducted subgroup analyses stratified by cell cycle phase and proliferation state.

---

## 3. Results

### 3.1 Data Overview

UMAP visualization of the full 241-feature space reveals a clear gradient structure aligned with the annotated age pseudotime, confirming that the data captures a continuous cellular trajectory (Figure 1). Cell cycle phases form overlapping but distinguishable distributions along this gradient, with G0/arrested cells concentrated at low pseudotime values, G1 cells in the early-to-mid range, S-phase cells in the mid range, and G2 cells at higher values, consistent with cell cycle progression as the primary biological process captured.

![UMAP overview colored by metadata](images/figure_umap_overview.png)

**Figure 1: UMAP overview of the RPE dataset.** Cells colored by (A) cell cycle phase, (B) cell state (cycling vs. arrested), (C) annotated age pseudotime, and (D) experimental batch. The continuous gradient in pseudotime is well-captured by the UMAP embedding, and batch effects are minimal.

Feature variances span several orders of magnitude, with DNA content-related features (Int_Intg_DNA_nuc) and nuclear area showing the highest variance across cells (Figure 2). This is consistent with DNA replication during S-phase and nuclear expansion during cell cycle progression as major sources of variation.

![Feature variance](images/figure_feature_variance.png)

**Figure 2: Top 30 features by variance.** DNA content and nuclear morphology features dominate the variance ranking.

### 3.2 Dynamic Feature Identification

Of the 241 features, 177 (73.4%) showed statistically significant Spearman correlation with annotated age after FDR correction (q < 0.05). The top-ranked feature by dynamic score was integrated nuclear DNA intensity (Int_Intg_DNA_nuc, ρ = 0.736, dynamic score = 0.793), followed by nuclear Cyclin A (Int_Med_cycA_nuc, ρ = 0.732, dynamic score = 0.767) and nuclear area (AreaShape_Area_nuc, ρ = 0.715, dynamic score = 0.679).

The top 12 dynamically expressed features, plotted as a function of pseudotime with cubic polynomial fits, show clear temporal patterns (Figure 3). DNA content, Cyclin A, Skp2, and PCNA increase monotonically with pseudotime—reflecting DNA replication and S-phase progression—while Cdt1 and p27 show decreasing trends, consistent with their known degradation and downregulation during cell cycle entry.

![Top dynamic features](images/figure_top_dynamic_features.png)

**Figure 3: Top 12 dynamically expressed features along pseudotime.** Each panel shows single-cell measurements (scatter, α=0.3) with cubic polynomial fit (red line). Spearman ρ values are indicated.

A heatmap of the top 30 dynamic features across 10 pseudotime bins further illustrates coordinated expression programs (Figure 4). Features cluster into two main groups: those increasing with pseudotime (DNA replication and mitotic entry markers) and those decreasing (cell cycle inhibitors and adhesion proteins).

![Dynamic heatmap](images/figure_dynamic_heatmap.png)

**Figure 4: Heatmap of top 30 dynamic features.** Z-scored mean expression per pseudotime bin shows coordinated temporal programs.

### 3.3 Feature Selection Performance

Our dynamic score-based feature selection substantially outperforms random selection and variance-based selection across all subset sizes (Figure 5). At k = 20 features, the dynamic score strategy achieves Spearman ρ = 0.878 (CV R² = 0.716), comparable to the absolute correlation strategy (ρ = 0.885, CV R² = 0.724) and mutual information strategy (ρ = 0.875, CV R² = 0.715). All three information-theoretic strategies dramatically outperform variance-based selection (ρ = 0.794) and random selection (ρ = 0.581).

![Feature selection curves](images/figure_feature_selection_curves.png)

**Figure 5: Feature selection performance curves.** (A) Spearman correlation between reconstructed and ground-truth pseudotime as a function of selected feature count. (B) Cross-validated R². (C) |Spearman ρ| between first PC and ground truth. Dynamic score-based selection (green) closely tracks absolute correlation-based selection (blue).

Notably, the full 241-feature set achieves only ρ = 0.813, indicating that including all features introduces noise that degrades trajectory reconstruction. The optimal performance is observed at k = 75 features (ρ = 0.892), representing a 69% feature reduction while improving trajectory preservation by 9.7% over the full set.

An ablation study of the dynamic score strategy demonstrates that trajectory preservation improves rapidly with the first 5–10 features, approaches near-optimal performance by k = 20–30 features, and then gradually plateaus (Figure 6).

![Ablation study](images/figure_ablation_k.png)

**Figure 6: Ablation study of feature subset size.** Spearman ρ as a function of k for the dynamic score strategy. The recommended k = 20 (green dashed line) achieves near-optimal performance while maximizing dimensionality reduction.

### 3.4 Trajectory Reconstruction Quality

Comparison of pseudotime reconstruction using the full feature set versus the selected 20 features confirms the benefit of feature selection (Figure 7). The selected features achieve tighter alignment with the ground truth (ρ = 0.878 vs. ρ = 0.813 for the full set), particularly in the mid-to-high pseudotime range where cell cycle progression is most active.

![Trajectory reconstruction](images/figure_trajectory_reconstruction.png)

**Figure 7: Pseudotime reconstruction comparison.** Ground truth (annotated age) vs. reconstructed pseudotime using (A) all 241 features and (B) the selected 20 features. The red dashed line represents perfect reconstruction. The selected feature set achieves better correlation.

UMAP embeddings computed from the selected 20 features preserve the continuous pseudotime gradient and cell cycle phase structure observed in the full-feature UMAP (Figure 8). The main trajectory structure is retained, with arrested cells (G0) forming a distinct cluster at low pseudotime values and cycling cells distributed along the progression axis.

![UMAP comparison](images/figure_umap_comparison.png)

**Figure 8: UMAP comparison between full and selected features.** Top row: UMAP from all 241 features. Bottom row: UMAP from 20 selected features. Both representations preserve pseudotime gradient and cell cycle structure.

### 3.5 Selected Feature Composition

The 20 selected features (Table 1) are dominated by nuclear-localized proteins (14/20, 70%), with contributions from the ring/periphery (2), cytoplasmic (2), and whole-cell edge (2) compartments. This nuclear enrichment is consistent with the centrality of nuclear events (DNA replication, transcription factor regulation, CDK activity) in cell cycle progression.

**Table 1: Selected Features and Their Biological Functions**

| Feature | Protein | Compartment | Dynamic Score | Biological Role |
|---------|---------|-------------|---------------|-----------------|
| Int_Intg_DNA_nuc | DNA | Nucleus | 0.793 | DNA content (replication) |
| Int_Med_cycA_nuc | Cyclin A | Nucleus | 0.767 | S/G2-M phase regulator |
| AreaShape_Area_nuc | — | Nucleus | 0.679 | Nuclear morphology |
| Int_Med_Skp2_nuc | Skp2 | Nucleus | 0.625 | p27 ubiquitination, S-phase entry |
| Int_Med_pH2AX_nuc | pH2AX | Nucleus | 0.522 | DNA damage response |
| Int_Med_cycB1_ring | Cyclin B1 | Ring | 0.513 | G2-M transition |
| Int_Med_cycB1_cyto | Cyclin B1 | Cytoplasm | 0.513 | G2-M transition |
| Int_Med_CDK2_nuc | CDK2 | Nucleus | 0.501 | G1-S transition |
| Int_Std_PCNA_nuc | PCNA | Nucleus | 0.494 | DNA replication clamp |
| Int_Med_cycB1_cell | Cyclin B1 | Whole Cell | 0.489 | G2-M transition |
| Int_Med_E2F1_nuc | E2F1 | Nucleus | 0.471 | S-phase transcription factor |
| Int_Med_Cdt1_nuc | Cdt1 | Nucleus | 0.421 | DNA licensing factor |
| Int_Med_Cdh1_nuc | Cdh1 | Nucleus | 0.396 | APC/C activator |
| Int_Med_pp21_nuc | pp21 | Nucleus | 0.379 | CDK inhibitor (phosphorylated) |
| Int_Med_p27_nuc | p27 | Nucleus | 0.379 | CDK inhibitor |
| Int_Med_cycB1_nuc | Cyclin B1 | Nucleus | 0.323 | G2-M transition |
| Int_Med_cycA_cyto | Cyclin A | Cytoplasm | 0.300 | S/G2-M phase regulator |
| Int_Med_cycA_ring | Cyclin A | Ring | 0.300 | S/G2-M phase regulator |
| Int_Med_p21_nuc | p21 | Nucleus | 0.282 | CDK inhibitor |
| Int_Med_cycA_cell | Cyclin A | Whole Cell | 0.269 | S/G2-M phase regulator |

The selected proteins reflect all major phases of the cell cycle: Cyclin A and Cyclin B1 (each appearing in 4 compartments) are key regulators of S-phase through mitosis; CDK2 drives G1-S transition; PCNA is the DNA replication processivity factor; Skp2 targets p27 for degradation to enable cell cycle entry; E2F1 is the master transcription factor for S-phase genes; Cdt1 is involved in DNA replication licensing; and Cdh1 (not to be confused with E-cadherin) is the APC/C co-activator that targets mitotic cyclins for degradation. The presence of CDK inhibitors (p21, p27, pp21) reflects the balance between proliferation-promoting and proliferation-inhibiting signals.

Permutation importance analysis confirms that the first two principal components carry the majority of trajectory-relevant information, with PC1 alone accounting for the dominant signal (Figure 9A). Feature-level coefficient analysis highlights Cyclin A and Cyclin B1 variants as the most influential predictors (Figure 9B).

![Permutation importance and feature importance](images/figure_permutation_importance.png)
![Feature importance](images/figure_feature_importance.png)

**Figure 9: Model interpretability.** (A) Permutation importance of PCA components for trajectory prediction. (B) Feature importance scores derived from linear model coefficients projected back to original feature space.

### 3.6 Subgroup Performance

Trajectory preservation varies across cell cycle phases (Figure 10). Reconstruction is strongest for S-phase cells (ρ = 0.715), followed by G1 (ρ = 0.653) and G0 (ρ = 0.642), with G2 showing the weakest correlation (ρ = 0.316). The reduced performance in G2 may reflect the relatively small sample size (n = 338) and the fact that G2 is a transitional phase with subtler protein expression changes.

When stratified by proliferation state, cycling cells show excellent trajectory preservation (ρ = 0.907, n = 2,174), substantially higher than the overall population. Arrested (G0) cells show moderate preservation (ρ = 0.642, n = 402), consistent with the expectation that quiescent cells exhibit less continuous variation along the pseudotime axis.

![Subgroup performance](images/figure_subgroup_performance.png)

**Figure 10: Subgroup trajectory preservation.** (A) Spearman ρ by cell cycle phase. (B) Spearman ρ by proliferation state (cycling vs. arrested).

### 3.7 Compartment-Level Analysis

Nuclear features show the highest mean dynamic score across all compartments (Figure 11), followed by ring/peripheral and cytoplasmic features. The top 30 most dynamic features are predominantly nuclear (16/30), with ring/periphery (7/30) and cytoplasmic (5/30) features also represented. This nuclear dominance aligns with the cell cycle as the primary trajectory, where nuclear events (DNA synthesis, transcription, CDK localization) are the most dynamically regulated processes.

![Compartment analysis](images/figure_compartment_analysis.png)

**Figure 11: Compartment-level analysis.** (A) Mean dynamic score by cellular compartment. (B) Compartment distribution among top 30 dynamic features. Nuclear features dominate the dynamic landscape.

### 3.8 Diffusion Pseudotime Validation

Diffusion Pseudotime (DPT), computed independently of the annotated age, shows strong agreement with the ground-truth pseudotime (Spearman ρ = 0.803, Figure 12). This orthogonal validation confirms that the data indeed contains a robust continuous trajectory structure that can be recovered by unsupervised methods. Phase-specific analysis shows that the DPT-ground truth correlation is strongest for S-phase cells.

![DPT comparison](images/figure_dpt_comparison.png)

**Figure 12: Diffusion Pseudotime validation.** (A) DPT vs. ground-truth annotated age shows strong concordance (ρ = 0.803). (B) Phase-stratified comparison reveals that all phases contribute to the continuous trajectory.

---

## 4. Discussion

### 4.1 Summary of Findings

We have demonstrated that a small subset of dynamically expressed protein features (20 out of 241, 8.3%) can capture continuous cellular trajectories in RPE cells with higher fidelity than the full feature set. Our dynamic scoring approach, which integrates correlation, non-linear fit quality, and information-theoretic criteria, identifies features that are not merely variable but genuinely informative about the underlying temporal progression.

The key findings are:

1. **Feature selection improves trajectory preservation**: Selected features (ρ = 0.878) outperform the full feature set (ρ = 0.813), demonstrating that noise reduction through feature selection enhances biological signal detection.

2. **Cell cycle regulators dominate the selected features**: Cyclin A, Cyclin B1, CDK2, PCNA, Skp2, E2F1, and their regulators (p21, p27) constitute the core selected set, consistent with cell cycle progression as the primary trajectory in these proliferating RPE cells.

3. **Nuclear compartment enrichment**: 70% of selected features are nuclear-localized, reflecting the nuclear-centric nature of cell cycle regulation.

4. **Subgroup specificity**: Trajectory preservation is strongest in actively cycling cells (ρ = 0.907) and weakest in G2 phase (ρ = 0.316), suggesting that feature selection should be tuned to the specific biological process of interest.

### 4.2 Relevance to Neuroscience Applications

While this analysis was performed on RPE cells, the methodological framework is directly applicable to neuroscience contexts. The selected features—cell cycle regulators, DNA damage markers, and nuclear morphology—are relevant to:

- **Neural lineage progression**: Neural progenitor proliferation and differentiation are governed by the same cell cycle machinery (Cyclins, CDKs, CDK inhibitors) identified here. Feature selection could help isolate the proliferative component from differentiation signals in neural stem cell trajectories.

- **Glial activation**: Reactive gliosis involves re-entry into the cell cycle and upregulation of proliferation markers. Our feature selection approach could identify the minimal protein panel needed to track glial activation states over time.

- **Neurodegeneration-related state transitions**: DNA damage markers (pH2AX) and cell cycle re-entry in post-mitotic neurons are hallmarks of neurodegeneration. The dynamic scoring framework could be adapted to identify features that track disease progression trajectories.

### 4.3 Methodological Considerations

Several aspects of our approach merit discussion:

**Composite scoring**: The dynamic score combines three complementary metrics. While absolute Spearman correlation alone performs comparably in this dataset, the composite score provides a more robust ranking when different features exhibit different types of pseudotime dependence (monotonic vs. non-monotonic, linear vs. non-linear).

**Optimal subset size**: The "elbow" in the performance curve at k ≈ 20–30 features suggests a natural trade-off between information content and noise. This likely reflects the intrinsic dimensionality of the cell cycle trajectory in these cells, where a limited number of coordinated protein changes capture the essential dynamics.

**PCA-based reconstruction**: Using PCA followed by linear regression provides a simple yet effective evaluation framework. However, non-linear reconstruction methods (e.g., kernel regression, neural networks) might further improve trajectory preservation, particularly for features with complex pseudotime dependencies.

**Limitations**: The analysis is limited to a single dataset and a single trajectory axis (cell cycle). The generalizability of the selected features to other cell types and biological processes requires further validation. Additionally, the ground-truth pseudotime (annotated age) may itself contain measurement error, though the strong DPT concordance supports its reliability.

### 4.4 Comparison with Related Work

Our feature selection framework is inspired by methods developed for scRNA-seq trajectory analysis, including diffusion pseudotime (Haghverdi et al., 2016) and Monocle (Trapnell et al., 2014; Cao et al., 2019), which use the full transcriptome for trajectory inference. The key innovation here is the explicit optimization of feature subsets for trajectory preservation in protein imaging data, where the feature space is more structured (known proteins in known compartments) and the measurement modality differs fundamentally from transcriptomics.

The finding that feature selection can improve trajectory inference aligns with recent work showing that highly variable gene selection improves single-cell analyses. Our results extend this principle to protein-level measurements and demonstrate that information-theoretic feature ranking can identify a near-optimal subset.

---

## 5. Conclusion

We present a systematic framework for selecting dynamically expressed molecular features that optimally preserve continuous cellular trajectories in single-cell data. Applied to RPE protein imaging data, the method identifies a compact set of 20 features—dominated by nuclear cell cycle regulators—that achieves superior trajectory preservation compared to the full feature set. The framework is readily generalizable to other single-cell modalities and biological contexts, including neural lineage progression, glial activation, and neurodegeneration-related state transitions. Future work should validate the selected feature panels in independent datasets and extend the approach to multi-trajectory settings where cells may follow branching differentiation paths.

---

## Data Availability

All analysis code is available in the `code/` directory. Intermediate results and processed data are stored in `outputs/`. The input dataset (`data/adata_RPE.h5ad`) is a preprocessed single-cell protein imaging dataset from retinal pigment epithelium cells.

## Code Repository Structure

```
code/
├── 01_data_exploration.py       # Data loading, preprocessing, PCA, UMAP
├── 02_trajectory_analysis.py    # Dynamic feature scoring, DPT computation
├── 03_feature_selection.py      # Feature selection strategies and evaluation
└── 04_interpretability.py       # SHAP-style analysis, subgroup validation
outputs/
├── adata_processed.h5ad         # Processed AnnData object
├── feature_dynamism_scores.csv  # Dynamic scores for all 241 features
├── feature_selection_evaluation.csv  # Performance across strategies and k
├── selected_features.txt        # Final 20 selected features
├── summary.json                 # Key metrics summary
└── ...                          # Additional intermediate outputs
report/
├── report.md                    # This report
└── images/                      # All figures (14 PNG files)
```
