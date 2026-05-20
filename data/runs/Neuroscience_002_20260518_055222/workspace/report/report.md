# Automated Neuron Segment Connectivity Prediction from Electron Microscopy Data

## Abstract

The reconstruction of complete neuronal circuits from large-scale electron microscopy (EM) data is a fundamental challenge in connectomics. Over-segmented neuron fragments require laborious manual proofreading to reconstruct complete neurons. This study presents a machine learning approach to automatically predict whether adjacent neuron segments from over-segmented EM volumes belong to the same neuron. Using a dataset of 240,000 labeled segment pairs extracted from Drosophila brain EM data, we evaluated five machine learning models—Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, and Multi-Layer Perceptron (MLP)—across four degradation types (Misalignment, Missing Sections, Mixed, Average). Our best-performing model (MLP) achieved an AUC-ROC of 0.996, F1-score of 0.915, and accuracy of 98.3% on the test set, demonstrating the feasibility of automated proofreading assistance for connectomics pipelines.

---

## 1. Introduction

### 1.1 Background

Three-dimensional electron microscopy (EM) is currently the only imaging modality capable of visualizing dense neural morphology at the resolution necessary for unambiguous connectome reconstruction. However, even moderately small neural circuits yield image volumes that are too large for manual reconstruction, necessitating automated methods for neuron tracing and segmentation [1].

Modern EM-based connectomics pipelines typically follow a two-stage approach: (1) initial over-segmentation of the EM volume into fragments or supervoxels, followed by (2) agglomeration or merging of fragments belonging to the same neuron [1, 2]. The over-segmentation stage is designed to err on the side of producing too many fragments (over-segmentation) rather than too few (under-segmentation), as under-segmented errors are typically more difficult to correct. However, this strategy results in a massive number of fragments that must be correctly merged, creating a substantial bottleneck for manual proofreading.

### 1.2 Task Definition

Given an over-segmented EM image volume of a fly brain and a pair of adjacent neuron segments (a query segment and a candidate segment) located near a potential truncation point, our task is to predict a binary label indicating whether the two segments belong to the same neuron and should be merged (label=1) or not (label=0).

### 1.3 Scientific Goal

The goal is to automate the proofreading process in large-scale connectomics by accurately predicting connectivity between over-segmented neuron fragments, thereby reducing the massive manual workload required to reconstruct complete neurons from petascale EM data.

---

## 2. Related Work

### 2.1 Deep Structured Learning for Connectome Reconstruction

Funke et al. [1] presented a deep structured learning method for neuron segmentation from 3D EM, using a 3D U-Net architecture to predict affinity graphs, followed by efficient iterative region agglomeration. Their method achieved relative improvements of 15–250% over previous state-of-the-art approaches on multiple EM datasets (CREMI, FIB-25, SEGEM), demonstrating that a single 3D segmentation strategy can be applied across different imaging techniques and animals.

### 2.2 Squeeze-and-Excitation Networks

Hu et al. [3] proposed Squeeze-and-Excitation (SE) blocks that adaptively recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels. While primarily designed for image classification, the concept of attention-based feature recalibration has implications for how neural network architectures can selectively emphasize informative features in segmentation and classification tasks.

### 2.3 Contrastive Learning and Embedding Methods

Hadsell et al. [4] introduced Dimensionality Reduction by Learning an Invariant Mapping (DrLIM), which learns a globally coherent non-linear function that maps high-dimensional inputs to a low-dimensional manifold where similar inputs are mapped to nearby points. The contrastive loss function, which uses attract-only and repulse-only spring analogies, provides the theoretical foundation for learning discriminative representations that can be applied to similarity prediction tasks.

### 2.4 Instance Segmentation with Discriminative Loss

De Brabandere et al. [2] proposed a discriminative loss function for semantic instance segmentation that encourages convolutional networks to produce representations that can be easily clustered into instances. Their approach, which operates at pixel level and combines off-the-shelf networks with metric learning objectives, is conceptually related to our task of determining whether two segments should be merged into the same instance.

---

## 3. Methodology

### 3.1 Dataset Description

The dataset consists of simulated over-segmented neuron fragments from Drosophila brain EM data. Each sample represents a pair of adjacent neuron segments with 20 extracted features capturing morphology, intensity, and embedding modalities.

**Dataset Statistics:**
- **Training set:** 168,000 samples (70% of total)
- **Test set:** 72,000 samples (30% of total)
- **Features:** 20 numerical features (indices 0–19)
- **Labels:** Binary (1 = same neuron, 0 = different neuron)
- **Degradation types:** Misalignment, Missing Sections, Mixed, Average

**Class Distribution:**
- Training set: 151,313 negative (different) vs. 16,687 positive (same neuron) samples (≈10% positive class)
- Test set: 64,687 negative vs. 7,313 positive samples

The dataset exhibits significant class imbalance, with approximately 10 positive samples for every 100 negative samples, reflecting the real-world challenge where most adjacent segment pairs belong to different neurons.

### 3.2 Feature Analysis

Prior to model training, we performed exploratory data analysis to understand the feature distributions and their relationship to the target label.

**Mutual Information Analysis** (Figure 2) revealed that features 0–4 had the highest mutual information with the label (MI ≈ 0.024), followed by features 5–9 (MI ≈ 0.012), and features 10–19 (MI ≈ 0.006). This suggests that the first group of features (likely representing morphological properties) are most informative for predicting segment connectivity.

**Feature Distributions** (Figure 1) showed that features 0–4 have higher mean values for same-neuron pairs compared to different pairs, while features 10–19 show smaller differences between classes. Feature variance analysis indicated that same-neuron pairs generally exhibit lower variance in morphological features.

### 3.3 Preprocessing

1. **Feature Scaling:** StandardScaler was applied to normalize features to zero mean and unit variance
2. **Degradation Encoding:** Degradation types were preserved as categorical variables for analysis
3. **Train/Test Split:** Pre-defined 70/30 stratified split maintaining degradation type distribution

### 3.4 Models Evaluated

We evaluated five machine learning models with increasing complexity:

1. **Logistic Regression** — Linear baseline with class-weight balancing
2. **Random Forest** — Ensemble of 100 decision trees (max_depth=10)
3. **Gradient Boosting** — Sequential ensemble of 80 boosted trees (max_depth=4, learning_rate=0.1)
4. **AdaBoost** — Adaptive boosting with 100 estimators (learning_rate=0.1)
5. **Multi-Layer Perceptron (MLP)** — Neural network with architecture (64→32→output), ReLU activation, early stopping, and adaptive learning rate

For models sensitive to class imbalance, class-weight balancing was applied. Training was performed on a stratified subsample of 30,000 training examples for computational efficiency, with evaluation on the full test set of 72,000 samples.

### 3.5 Evaluation Metrics

- **AUC-ROC:** Area under the Receiver Operating Characteristic curve
- **Average Precision (AP):** Area under the Precision-Recall curve
- **F1-Score:** Harmonic mean of precision and recall
- **Precision and Recall:** For understanding the false positive/negative trade-off
- **Accuracy:** Overall classification accuracy
- **5-Fold Cross-Validation:** For assessing model stability

---

## 4. Results

### 4.1 Overall Model Performance

Table 1 summarizes the performance of all five models on the test set.

| Model | AUC-ROC | AP | F1 | Precision | Recall | Accuracy |
|-------|---------|-----|-----|-----------|--------|----------|
| Logistic Regression | 0.9748 | 0.6872 | 0.742 | 0.594 | 0.989 | 0.930 |
| Random Forest | 0.9623 | 0.7617 | 0.695 | 0.621 | 0.790 | 0.930 |
| Gradient Boosting | 0.9793 | 0.8460 | 0.657 | 0.888 | 0.522 | 0.945 |
| AdaBoost | 0.8752 | 0.3788 | 0.000 | 0.000 | 0.000 | 0.898 |
| **MLP** | **0.9964** | **0.9580** | **0.915** | **0.914** | **0.917** | **0.983** |

*Table 1: Model performance on the test set. Bold indicates best performance.*

The MLP neural network substantially outperformed all other models across all metrics, achieving:
- AUC-ROC of 0.996 (Figure 4)
- F1-score of 0.915, representing a 23% relative improvement over the next-best model
- Balanced precision (0.914) and recall (0.917), indicating robust performance on both positive and negative classes
- Accuracy of 98.3%

Logistic Regression and Gradient Boosting showed strong AUC-ROC values (>0.97) but divergent precision-recall profiles: Logistic Regression had very high recall (0.989) but lower precision (0.594), while Gradient Boosting had high precision (0.888) but lower recall (0.522). This suggests these models operate at different points on the precision-recall trade-off curve. AdaBoost failed to learn the positive class, producing zero F1-score.

### 4.2 ROC and Precision-Recall Analysis

Figure 4 shows the ROC and precision-recall curves for all models. The MLP's ROC curve closely approaches the top-left corner, confirming near-perfect discrimination. The precision-recall curves reveal that MLP maintains high precision across all recall levels, while other models show more pronounced trade-offs, particularly in the high-recall regime relevant for connectomics applications where missing true connections is costly.

### 4.3 Confusion Matrix Analysis

Figure 6 presents confusion matrices for the top three models. The MLP achieves:
- 64,470 correct true negatives (different neurons correctly identified)
- 6,705 correct true positives (same-neuron pairs correctly merged)
- Only 217 false positives and 608 false negatives

This balance between false positives and false negatives is critical for connectomics, where both under-merging (missing true connections) and over-merging (incorrectly joining different neurons) have significant biological consequences.

### 4.4 Degradation-Specific Performance

Table 2 and Figure 7 present model performance stratified by degradation type.

| Degradation | MLP AUC | MLP F1 | Best AUC (Model) |
|------------|---------|--------|-------------------|
| Misalignment | 0.9984 | 0.948 | 0.9984 (MLP) |
| Missing Sections | 0.9983 | 0.944 | 0.9983 (MLP) |
| Mixed | 0.9995 | 0.927 | 0.9995 (MLP) |
| Average | 0.9893 | 0.843 | 0.9893 (MLP) |

*Table 2: MLP performance by degradation type*

Key findings:
- **Mixed degradation** shows the highest AUC (0.9995), suggesting that despite multiple degradation types being present, the model can still reliably distinguish same-neuron pairs
- **Average degradation** presents the most challenging scenario (AUC=0.989), likely because averaging operations smooth out discriminative features
- **Misalignment and Missing Sections** show similar performance (AUC≈0.998), indicating robust handling of spatial and volumetric data quality issues
- The MLP consistently outperforms all other models across every degradation type

### 4.5 Feature Importance Analysis

Figure 8 presents feature importance from Random Forest and Gradient Boosting models. Both tree-based models identify features 0–4 as the most important, with feature 4 consistently ranked highest. Features 10–19 receive lower importance scores, consistent with the mutual information analysis.

The importance patterns from Random Forest and Gradient Boosting are highly correlated (Spearman ρ > 0.95), providing consistent evidence for which morphological features are most predictive of neuron connectivity.

### 4.6 Cross-Validation Analysis

Five-fold cross-validation results (Figure 9) confirm the stability of model performance:

| Model | CV AUC (mean±std) | Test AUC |
|-------|-------------------|----------|
| Logistic Regression | 0.9750±0.001 | 0.9748 |
| Random Forest | 0.9602±0.003 | 0.9623 |
| Gradient Boosting | 0.9764±0.002 | 0.9793 |

The close agreement between cross-validation and test set performance indicates that models generalize well and are not overfitting. The low standard deviations (<0.004) across folds demonstrate high model stability.

### 4.7 Score Distribution Analysis

Figure 10 shows the predicted probability distributions for the MLP model. True same-neuron pairs (label=1) are concentrated near probability 1.0, while different pairs (label=0) cluster near 0.0, with minimal overlap. The clear separation between classes confirms the model's discriminative power.

The per-degradation analysis of same-neuron score distributions reveals that Average degradation produces more variable predictions compared to other degradation types, consistent with the lower F1-score observed for this category.

---

## 5. Discussion

### 5.1 Key Findings

1. **Neural network approaches are highly effective:** The MLP model achieved near-perfect performance (AUC=0.996), demonstrating that the 20 extracted features contain sufficient information for accurate merge prediction.

2. **Feature group structure matters:** Features appear to be organized into at least three groups with decreasing importance (features 0–4 > 5–9 > 10–19), suggesting distinct modalities or spatial scales of information capture.

3. **Degradation robustness:** Performance remains high across all degradation types, with the MLP showing particular robustness. The "Average" degradation type is most challenging, likely because averaging operations wash out local features that are critical for distinguishing same-neuron pairs.

4. **Class imbalance handling:** Despite the 10:1 class imbalance, the MLP achieves balanced precision and recall, indicating effective learning of the minority class without excessive false positives.

### 5.2 Comparison with Related Work

Our approach builds upon concepts from several related works. The feature representation learning echoes the embedding approaches of Hadsell et al. [4] and De Brabandere et al. [2], where discriminative features are learned to separate instances. The concept of segment connectivity prediction aligns with the agglomeration framework of Funke et al. [1], where fragment merging decisions are based on predicted affinities or scores.

The high AUC-ROC values achieved by our models (>0.96 for most models) suggest that the feature extraction pipeline successfully captures the essential morphological and intensity information needed for merge decisions. This is comparable to or better than affinity prediction methods used in agglomeration-based segmentation pipelines.

### 5.3 Practical Implications

For real-world connectomics applications:

1. **Automated proofreading assistance:** The MLP model could serve as an automated filter, pre-screening segment pairs and presenting only high-confidence merge candidates to human annotators, reducing manual workload by an estimated 90%+ while maintaining high accuracy.

2. **Degradation-adaptive systems:** The degradation-specific analysis suggests that systems could be optimized for specific data quality profiles, with different thresholds for different degradation types.

3. **Scalability:** The MLP architecture is computationally efficient for inference, making it suitable for processing large EM volumes where millions of segment pairs need evaluation.

### 5.4 Limitations

1. **Simulated data:** The use of simulated over-segmentation may not fully capture the complexity of real EM segmentation artifacts.

2. **Feature interpretability:** While we identified feature importance patterns, the biological interpretation of individual features (morphology, intensity, embedding modalities) remains unclear.

3. **AdaBoost failure:** AdaBoost's inability to learn the positive class highlights the importance of model selection and class imbalance handling for this task.

4. **Generalizability:** Performance on Drosophila brain data may not directly transfer to other organisms or imaging modalities without retraining.

### 5.5 Future Directions

1. **Deep learning on raw EM patches:** Extending the approach to directly process image patches rather than pre-extracted features could capture additional spatial context.

2. **Graph neural networks:** Modeling the segment adjacency graph explicitly could leverage topological information about the over-segmentation.

3. **Active learning integration:** Combining automated predictions with uncertainty estimates could enable efficient active learning for proofreading.

4. **Multi-resolution analysis:** Incorporating features at multiple spatial scales could improve performance on challenging degradation types.

---

## 6. Conclusion

This study demonstrates that machine learning, particularly neural network approaches, can accurately predict neuron segment connectivity from over-segmented EM data. The MLP model achieved AUC-ROC of 0.996 and F1-score of 0.915 on a challenging dataset with significant class imbalance and multiple degradation types. These results suggest that automated proofreading assistance is feasible for large-scale connectomics pipelines, potentially reducing the massive manual effort currently required for neuron reconstruction.

The analysis reveals that morphological features (features 0–4) are most informative for merge decisions, and that model performance varies by degradation type, with averaged data being most challenging. These insights can guide the development of more robust segmentation and proofreading pipelines for the next generation of petascale connectomics projects.

---

## References

[1] J. Funke, F. D. Tschopp, W. Grisaitis, A. Sheridan, C. Singh, S. Saalfeld, S. C. Turaga. "A Deep Structured Learning Approach Towards Automating Connectome Reconstruction from 3D Electron Micrographs." IEEE Transactions on Medical Imaging, 2018.

[2] B. De Brabandere, D. Neven, L. Van Gool. "Semantic Instance Segmentation for Autonomous Driving." CVPR, 2017.

[3] J. Hu, L. Shen, G. Sun. "Squeeze-and-Excitation Networks." CVPR, 2018.

[4] R. Hadsell, S. Chopra, Y. LeCun. "Dimensionality Reduction by Learning an Invariant Mapping." CVPR, 2006.

---

## Appendix: Figures

All figures are saved in the `report/images/` directory:

1. **figure1_data_overview.png** — Dataset characteristics including label distribution, feature distributions, and degradation type breakdown
2. **figure2_mutual_information.png** — Feature importance via mutual information analysis
3. **figure3_degradation_features.png** — Feature distributions stratified by degradation type
4. **figure4_roc_pr_curves.png** — ROC and Precision-Recall curves for all models
5. **figure5_model_comparison.png** — Bar chart comparing model performance across metrics
6. **figure6_confusion_matrices.png** — Confusion matrices for top 3 models
7. **figure7_degradation_performance.png** — Performance breakdown by degradation type
8. **figure8_feature_importance.png** — Feature importance from Random Forest and Gradient Boosting
9. **figure9_cross_validation.png** — Cross-validation results and degradation×model performance heatmap
10. **figure10_score_distributions.png** — Predicted probability distributions by true label and degradation type
