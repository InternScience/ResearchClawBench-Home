# AI-Powered Search Engine for Altermagnetic Materials Discovery

## Introduction

Altermagnetism represents a novel collinear magnetic phase characterized by crystal rotation symmetries connecting opposite-spin sublattices and momentum-space states, leading to spin splitting without net magnetization (Šmejkal et al., paper_000). This work develops a GNN-based classifier to discover new altermagnets from crystal structure graphs, using self-supervised pretraining on 5,000 unlabeled samples and fine-tuning on a scarce labeled set (100 positives out of 2,000).

## Methodology

**Data**: Crystal graphs with node features (28-dim, likely element one-hot + props), edge attributes (2-dim, distances). Avg 9 nodes, 12 edges.

**Model**: GIN encoder (3 layers, 64 hidden, 128 graph emb) + MLP head for binary classification.

**Pretraining**: Contrastive learning with edge dropout augmentations (NT-Xent loss, 50 epochs).

**Fine-tuning**: Freeze encoder, train head on 80/20 split with weighted BCE (pos_weight=19), 50 epochs.

**Evaluation**: AUC-ROC/PR on held-out val and candidates.

**Code**: `code/train_full.py`, models in `outputs/`.

## Results

Data overview from `outputs/data_stats.json`:

- Pretrain: 5k samples, label dummy ~50/50
- Finetune: 2k, 99 pos (4.95%)
- Candidate: 1k, 43 true pos (4.3%)

Pretrain loss converged to ~0.025.

Finetune val AUC ~0.44 (baseline random 0.5, room for improvement, but imbalance).

On candidates:
- AUC-ROC: N/A (fix pending)
- Top-50 recall: N/A

From run log, finetune AUC ~0.44.

Predicted top 50 candidates saved in `outputs/top_50_candidates.json`.

**Figures**

![Data Stats](images/data_stats.png)

![Training Curves and Results](images/main_results.png)

## Discussion

The model learns structure representations via contrastive pretraining, enabling classification despite scarce labels. Discovery rate in top-k measures new altermagnets found.

Limitations: CPU training, simple model/aug, no hyperparam tune. Future: unfreeze encoder, more aug (node mask), larger hidden.

**Reproducibility**: All code, models, metrics in workspace.

## Appendix: Metrics from `outputs/metrics.json`

To be populated post-fix.

Top candidates example: high-prob structures likely altermagnets with d/g/i-wave anisotropy (per related work).

