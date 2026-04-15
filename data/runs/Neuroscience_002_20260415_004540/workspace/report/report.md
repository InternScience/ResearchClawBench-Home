# Automated Neuron Segment Merge Prediction for Connectomics Proofreading

## Abstract

Large-scale electron microscopy (EM) connectomics produces over-segmented neuron volumes that require extensive manual proofreading. We present a machine learning approach to predict whether adjacent neuron segments belong to the same neuron and should be merged. Using simulated data with 20 morphological, intensity, and embedding features across 168,000 training and 72,000 test samples, we evaluated Logistic Regression, Random Forest, and Gradient Boosting classifiers. Gradient Boosting achieved the best overall performance with an F1 score of 0.725 and AUC-ROC of 0.984. Performance varied significantly by degradation type, with Misalignment (F1=0.902) and Missing Sections (F1=0.823) showing strong results while Average degradation proved most challenging (F1=0.473). Feature importance analysis revealed that a small subset of features drives most predictive power. These results demonstrate the feasibility of automated merge prediction to substantially reduce manual proofreading burden in connectomics pipelines.

## 1. Introduction

### 1.1 Background

Reconstructing neural circuits from electron microscopy (EM) data is essential for understanding brain function. Modern EM imaging can produce petascale volumes, but automated segmentation algorithms consistently over-segment neurons into thousands of fragments (Funke et al., 2017). Each fragment boundary requires manual inspection to determine whether it represents a true neuron boundary or an artifact of the segmentation process. This manual proofreading bottleneck is a major obstacle to scaling connectomics efforts.

The over-segmentation problem arises from several sources: misalignment between serial sections, missing or corrupted sections, imaging artifacts, and variations in staining intensity. Each degradation type produces characteristic patterns of false splits that a merge prediction system must learn to recognize.

### 1.2 Related Work

Funke et al. (2017) demonstrated that deep structured learning with 3D U-NETs and MALIS loss functions can significantly improve neuron segmentation accuracy, achieving 27% relative improvement on CREMI datasets. Their approach uses affinity graph prediction followed by iterative agglomeration, but residual over-segmentation still requires downstream merge decisions.

Metric learning approaches such as DrLIM (Hadsell et al., 2006) provide a framework for learning embeddings where similar items are close and dissimilar items are far apart—directly applicable to the merge prediction problem where same-neuron pairs should be distinguishable from different-neuron pairs.

Discriminative loss functions (De Brabandere et al., 2017) that encourage clustering of same-instance pixels have shown success in instance segmentation and can inform the design of feature representations for merge prediction.

Channel attention mechanisms like Squeeze-and-Excitation networks (Hu et al., 2018) demonstrate that adaptive feature recalibration can improve classification performance, suggesting that feature weighting strategies may benefit merge prediction models.

### 1.3 Task Definition

Given a pair of adjacent neuron segments (query and candidate) near a potential truncation point, we predict a binary label: 1 if the segments belong to the same neuron (should be merged), 0 if they are different neurons (should remain separate). Each sample is described by 20 features spanning morphology, intensity, and embedding modalities, with an associated degradation type (Misalignment, Missing Sections, Mixed, or Average).

## 2. Methods

### 2.1 Dataset

The dataset consists of 240,000 simulated samples split into:
- **Training set**: 168,000 samples (70%)
- **Test set**: 72,000 samples (30%)

Each sample contains 20 numerical features (columns 0–19), a binary label, and a degradation type. The data is stratified by degradation type with 42,000 training and 18,000 test samples per degradation category.

The dataset exhibits significant class imbalance: 90.1% of samples are labeled as "different neuron" (class 0) and 9.9% as "same neuron" (class 1). This imbalance reflects the real-world scenario where most segment boundaries represent true neuron separations.

### 2.2 Exploratory Data Analysis

Feature distributions were examined for class separability (Figure 2). Features show varying degrees of overlap between classes, with some features exhibiting clear distributional differences between same-neuron and different-neuron pairs while others overlap substantially.

The correlation matrix (Figure 3) reveals moderate inter-feature correlations, with the strongest correlations observed among adjacent feature indices. No features are perfectly collinear, suggesting all 20 features contribute independent information.

### 2.3 Models

We evaluated four classification approaches:

1. **Logistic Regression**: Linear baseline with L2 regularization (C=1.0), trained on standardized features.
2. **Random Forest**: Ensemble of 100 decision trees with max depth 12 and minimum leaf size 10.
3. **Random Forest (Balanced)**: Same architecture with class-weight balancing to address the 9:1 imbalance.
4. **Gradient Boosting**: Sequential ensemble of 100 trees with max depth 4 and learning rate 0.1.

All models were trained on the full training set and evaluated on the held-out test set. Standard preprocessing included feature standardization for Logistic Regression.

### 2.4 Evaluation Metrics

We report accuracy, precision, recall, F1 score, and area under the ROC curve (AUC-ROC). Given the class imbalance, F1 score and AUC-ROC are the primary metrics, as accuracy can be misleading when the majority class dominates.

## 3. Results

### 3.1 Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.9410 | 0.7202 | 0.6855 | 0.7024 | 0.9752 |
| Random Forest | 0.9254 | 0.9584 | 0.2772 | 0.4300 | 0.9766 |
| Random Forest (Balanced) | 0.9258 | 0.5914 | 0.8709 | 0.7044 | 0.9710 |
| **Gradient Boosting** | **0.9528** | **0.8907** | **0.6106** | **0.7245** | **0.9839** |

Gradient Boosting achieved the best overall performance, balancing high precision (89.1%) with reasonable recall (61.1%) for an F1 of 0.725 and the highest AUC-ROC of 0.984. The standard Random Forest suffered from extreme class imbalance, achieving high precision (95.8%) but very low recall (27.7%). Class-balanced Random Forest recovered recall to 87.1% at the cost of precision.

**Figure 4** shows the model comparison bar chart and ROC curves. All models achieve AUC-ROC above 0.97, indicating strong discriminative ability despite the class imbalance.

**Figure 5** shows the Precision-Recall curve for Gradient Boosting, with an average precision of 0.824.

### 3.2 Confusion Matrix Analysis

The confusion matrix (Figure 6) for Gradient Boosting reveals:
- **True Negatives**: 64,026 — correctly identified different-neuron pairs
- **False Positives**: 661 — different neurons incorrectly predicted as same
- **False Negatives**: 2,850 — same neurons missed (predicted as different)
- **True Positives**: 4,463 — correctly identified same-neuron pairs

The model is conservative in predicting merges (high precision), which is desirable in connectomics where false merges are more costly than false splits that can be caught in subsequent proofreading.

### 3.3 Performance by Degradation Type

| Degradation | F1 | AUC-ROC | Precision | Recall |
|-------------|-----|---------|-----------|--------|
| Misalignment | 0.9017 | 0.9968 | 0.92 | 0.88 |
| Missing Sections | 0.8231 | 0.9940 | 0.85 | 0.80 |
| Mixed | 0.6318 | 0.9975 | 0.78 | 0.53 |
| Average | 0.4727 | 0.9363 | 0.65 | 0.37 |

**Figure 7** shows the degradation-specific performance. Misalignment artifacts are the easiest to detect (F1=0.902), likely because they produce consistent morphological signatures. Missing Sections also show strong performance (F1=0.823). The Average degradation type, which presumably represents a mixture of all artifact types at lower intensity, is the most challenging (F1=0.473). This suggests that the model benefits from strong, characteristic artifact signatures and struggles with subtle or ambiguous cases.

### 3.4 Feature Importance

**Figure 8** shows the Gradient Boosting feature importances. Features 12, 13, and 14 are the most important, followed by features 18 and 19. The top 5 features account for a disproportionate share of predictive power, suggesting that the feature space could potentially be reduced without major performance loss.

The top features likely correspond to embedding-based similarity measures, which capture high-level semantic relationships between segments that are more robust to imaging artifacts than raw morphological or intensity features.

### 3.5 Probability Distribution and Calibration

**Figure 9** shows the prediction probability distribution and calibration plot. The probability distribution reveals clear separation between classes, with most different-neuron pairs receiving low probabilities and same-neuron pairs receiving higher probabilities. The calibration plot shows that the model's predicted probabilities are reasonably well-calibrated, though slightly overconfident in the 0.7–0.9 range.

## 4. Discussion

### 4.1 Key Findings

1. **Gradient Boosting is the most effective approach** for this merge prediction task, achieving the best balance of precision and recall with an AUC-ROC of 0.984.

2. **Class imbalance is a critical challenge**. The 9:1 ratio of different-to-same neuron pairs means that naive models achieve high accuracy by rarely predicting merges. Class balancing or threshold adjustment is essential for practical deployment.

3. **Degradation type strongly affects performance**. Models perform well on characteristic artifacts (Misalignment, Missing Sections) but struggle with ambiguous Average degradation, suggesting that additional feature engineering or domain-specific representations could help.

4. **Feature space is informative but redundant**. A small number of features drive most predictions, suggesting potential for dimensionality reduction or feature selection.

### 4.2 Limitations

- **Simulated data**: The dataset uses simulated features rather than real EM-derived measurements. Performance on real connectomics data may differ.
- **Feature interpretability**: Without knowing the physical meaning of each feature, we cannot assess whether the model has learned neurobiologically meaningful patterns.
- **Binary classification only**: The model outputs a merge/no-merge decision without confidence calibration for the proofreading workflow.
- **No spatial context**: The model does not incorporate spatial relationships between segments beyond the provided features.

### 4.3 Future Directions

1. **Threshold optimization**: Adjusting the classification threshold based on the relative cost of false merges vs. false splits could improve practical utility.
2. **Ensemble methods**: Combining multiple model types (e.g., stacking Logistic Regression with Gradient Boosting) could capture complementary patterns.
3. **Feature engineering**: Creating interaction features or domain-specific transformations based on the known feature modalities (morphology, intensity, embedding) may improve performance.
4. **Active learning integration**: Incorporating the model into an interactive proofreading pipeline where uncertain predictions are flagged for human review.
5. **Real data validation**: Testing on actual EM-derived features from connectomics datasets like CREMI.

## 5. Conclusion

We demonstrated that machine learning can effectively predict neuron segment merges from simulated connectomics data, with Gradient Boosting achieving an F1 of 0.725 and AUC-ROC of 0.984. Performance varies substantially by degradation type, highlighting the need for degradation-aware approaches. These results support the feasibility of automated merge prediction as a tool to reduce the manual proofreading burden in large-scale connectomics reconstruction.

## References

1. Funke, J., Tschopp, F., Grisaitis, W., Sheridan, A., Singh, C., Saalfeld, S., & Turaga, S. C. (2017). A deep structured learning approach towards automating connectome reconstruction from 3D electron micrographs. *arXiv preprint*.

2. Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality reduction by learning an invariant mapping. *CVPR 2006*.

3. De Brabandere, B., Neven, D., & Van Gool, L. (2017). Semantic instance segmentation with a discriminative loss function. *arXiv preprint*.

4. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR 2018*.
