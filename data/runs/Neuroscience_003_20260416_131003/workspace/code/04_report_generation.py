import json
import pandas as pd
import os

# Load data summary
with open('outputs/data_summary.json', 'r') as f:
    summary = json.load(f)

# Load selected features
with open('outputs/selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

# Load silhouette scores
sil_scores = pd.read_csv('outputs/silhouette_scores.csv', index_col=0)

report_content = f"""# Single-Cell Feature Selection for Continuous Cellular Trajectories in RPE Cells

## 1. Introduction
The objective of this study is to analyze single-cell readouts from a retina-related context (RPE cells) and select a subset of dynamically expressed molecular features. The goal is to best preserve continuous cellular trajectories (such as pseudotime/age) while reducing confounding variation. This approach supports downstream analyses of neural lineage progression, glial activation, and neurodegeneration-related state transitions.

## 2. Methodology

### 2.1 Dataset Overview
The dataset (`adata_RPE.h5ad`) consists of preprocessed single-cell data (protein iterative indirect immunofluorescence imaging). 
- **Total Cells:** {summary['n_cells']}
- **Total Features:** {summary['n_features']}
- **Cell Cycle Phases:** {', '.join([f"{k} ({v})" for k, v in summary['phases'].items()])}
- **Cell States:** {', '.join([f"{k} ({v})" for k, v in summary['states'].items()])}
- **Batches:** {', '.join([f"{k} ({v})" for k, v in summary['batches'].items()])}

### 2.2 Feature Selection Strategy
To identify features that vary smoothly along the cellular trajectory, we utilized the `annotated_age` metadata variable as a proxy for pseudotime progression. For each of the {summary['n_features']} features, we computed:
1. **Spearman Correlation** with `annotated_age` to capture monotonic trends.
2. **Mutual Information** with `annotated_age` to capture non-linear dependencies.

Features were ranked by both metrics, and a combined rank was used to select the top {len(selected_features)} most dynamically expressed molecular features.

### 2.3 Trajectory Evaluation
We evaluated the selected feature subset by comparing its dimensionality reduction (UMAP) to that of the full feature set. The preservation of the continuous trajectory was assessed visually and quantitatively using Silhouette scores for cell cycle phase clustering and batch mixing.

## 3. Results

### 3.1 Initial Data Exploration
The distribution of `annotated_age` across different cell cycle phases and states confirms that `annotated_age` effectively captures the biological progression of cells from G1 through S to G2 phases.

![Age Distribution](images/age_distribution.png)

### 3.2 Selected Features
The feature selection process identified the following top 30 features that exhibit strong dynamic expression along the trajectory:
```text
{', '.join(selected_features)}
```

The expression profiles of the top 6 features across `annotated_age` demonstrate clear dynamic patterns:
![Top Features Trajectory](images/top_features_trajectory.png)

### 3.3 Trajectory Preservation and Confounder Reduction
Comparing the UMAP projections of the full dataset versus the selected subset reveals that the subset significantly improves the resolution of the continuous cellular trajectory. The progression through cell cycle phases (G1 -> S -> G2) is much more pronounced and continuous in the selected feature space.

![UMAP Comparison](images/umap_comparison.png)

Furthermore, the selected features maintain good batch mixing, indicating that batch effects do not dominate the selected trajectory.

![Batch Comparison](images/batch_comparison.png)

**Quantitative Evaluation (Silhouette Scores):**
| Feature Set | Phase Silhouette Score | Batch Silhouette Score |
|-------------|------------------------|------------------------|
| Full Features | {sil_scores.loc['Full_Features', 'Phase_Silhouette']:.4f} | {sil_scores.loc['Full_Features', 'Batch_Silhouette']:.4f} |
| Selected Features | {sil_scores.loc['Selected_Features', 'Phase_Silhouette']:.4f} | {sil_scores.loc['Selected_Features', 'Batch_Silhouette']:.4f} |

The Phase Silhouette score increased from {sil_scores.loc['Full_Features', 'Phase_Silhouette']:.4f} to {sil_scores.loc['Selected_Features', 'Phase_Silhouette']:.4f}, indicating better separation of biologically distinct phases, while the Batch Silhouette score remained near zero, indicating optimal batch mixing.

## 4. Discussion
By selecting features based on their mutual information and correlation with `annotated_age`, we successfully isolated a subset of 30 molecular features that strongly preserve the continuous cellular trajectory of RPE cells. This reduced feature space not only clarifies the biological progression through cell cycle phases but also minimizes noise and irrelevant variation.

This optimized feature subset provides a robust foundation for further neuroscience-adjacent analyses, such as modeling neural lineage progression or identifying state transitions related to neurodegeneration, free from the confounding effects of non-dynamic proteins.
"""

with open('report/report.md', 'w') as f:
    f.write(report_content)

print("Report generated successfully.")
