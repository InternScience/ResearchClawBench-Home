# Local Benchmark Study of a DIDS-MFL-Inspired Intrusion Detection Pipeline

## Abstract
This report studies a local-only approximation of the proposed disentangled dynamic intrusion detection framework (DIDS-MFL) on `NF-UNSW-NB15-v2_3d.pt`. The benchmark constraints disallow web access, remote execution, and modification of the provided corpus, so the study uses only the temporal graph tensor in `data/` and the four papers in `related_work/`. The implemented pipeline combines three ideas motivated by the local literature: disentangled subspace learning over flow statistics, dynamic context features derived from temporal graph history, and dual-similarity few-shot inference. On a time-ordered split, binary intrusion detection is nearly saturated for both the baseline and the proposed representation, with macro-F1 above 0.995. Multi-class detection is harder: the raw-feature baseline attains macro-F1 0.809 and the local DIDS-MFL approximation attains macro-F1 0.799. Few-shot classification on the three rarest attack classes reaches 0.614 macro-F1 with 5-shot prototypes. Unknown-attack evaluation is the weakest branch: when attack class `9` is held out during training, the current local approximation fails to reject it. The main conclusion is that disentangled temporal representations are viable in this benchmark, but the simplified local surrogate does not yet outperform a strong tabular baseline on known-class multi-class detection, and it remains insufficient for open-set generalization.

## 1. Task and Local Constraints
The scientific task is network intrusion detection on NetFlow-style temporal graph data, including binary benign-versus-attack prediction, multi-class attack recognition, few-shot behavior on rare classes, and robustness to unseen attacks. The benchmark environment imposes hard local-only constraints: no web search, no external datasets, no remote GPUs, no API usage, and no edits to `data/` or `related_work/`. As a result, the study adapts ARIS phases to a fully offline workflow:

1. Read the provided brief, instructions, local data, and local papers.
2. Extract a benchmark-safe research hypothesis from the local literature.
3. Implement executable analysis code under `code/`.
4. Save intermediate artifacts under `outputs/`.
5. Save mandatory PNG figures under `report/images/`.
6. Write the final report to `report/report.md`.

## 2. Literature Understanding from `related_work/`
The local literature suggests three design principles.

First, `paper_000` (3D-IDS) argues that intrusion detection benefits from two-step disentanglement of traffic features plus dynamic graph diffusion. The key transferable idea for this benchmark is that entangled flow features can hurt consistency across attack types, so it is worth separating statistical factors before classification.

Second, `paper_002` (E-GraphSAGE) shows that flow records have a natural graph interpretation: endpoints are nodes and flows are edges. This motivates injecting topological and temporal context rather than treating every flow independently.

Third, `paper_001` and `paper_003` motivate factor-aware representation learning and multiple similarity views. In the benchmark setting, those ideas translate naturally into multi-view subspaces and dual-similarity few-shot prototypes.

Because the sparse PyG backends are unavailable in the local runtime, I replaced heavy graph diffusion with CPU-safe temporal-topological surrogates based on source/destination histories, pair repetition counts, and rolling flow statistics.

## 3. Data Understanding
The input tensor is a `torch_geometric.data.temporal.TemporalData` object with:

- 148,774 flow events
- 40 continuous edge features in `msg`
- source node IDs in `src`
- destination node IDs in `dst`
- integer timestamps in `t`
- binary labels in `label`
- multi-class attack labels in `attack`

The observed time range is one day: `t in [0, 86399]`. The estimated node universe is 1,090,431 endpoint IDs. Class imbalance is severe. Benign traffic dominates the data, and several attack classes are very small:

- benign label `2`: 114,716 flows
- malicious labels `6` and `7`: 14,688 and 10,910 flows
- rare malicious labels `0`, `1`, and `9`: 380, 341, and 164 flows

The dataset overview is visualized in `images/attack_distribution.png`.

## 4. Methodology
### 4.1 Local DIDS-MFL Approximation
The executable implementation is [`code/run_analysis.py`](code/run_analysis.py). The method is a benchmark-safe approximation rather than a faithful reproduction of the original paper.

It has three components.

1. Statistical disentanglement.
The 40 flow features are split into three groups and compressed independently with PCA:
- low-index statistics
- mid-index statistics
- high-index statistics

2. Dynamic and topological context.
For each flow, the code derives:
- cyclic time encodings
- previous source count
- previous destination count
- previous source-destination pair count
- cumulative degree proxy
- rolling source and destination flow-intensity averages

3. Multi-scale fusion.
Each subspace is reduced to four dimensions, and all views are concatenated into a 16-dimensional fused representation.

### 4.2 Evaluation Design
The main experiments use a time-ordered split and then subsample for CPU-safe execution:

- train: 60,000 flows from the earlier 70% of timestamps
- test: 30,000 flows from the later 30% of timestamps

This preserves temporal directionality and avoids leaking future history into past samples.

### 4.3 Baseline and Tasks
The report compares a raw-feature baseline with the proposed fused representation.

- Binary task: benign vs attack
- Multi-class task: attack label classification
- Few-shot task: 5-shot prototype classification on the three rarest classes
- Unknown-attack task: hold out the rarest malicious class during training and test whether the model rejects it

## 5. Results
### 5.1 Binary Intrusion Detection
Binary detection is almost saturated.

- Baseline: accuracy 0.9973, balanced accuracy 0.9981, macro-F1 0.9962, ROC-AUC 0.9998
- Proposed local DIDS-MFL: accuracy 0.9969, balanced accuracy 0.9974, macro-F1 0.9957, ROC-AUC 0.9998

The representation fusion does not beat the baseline here. This is not surprising: the benign-versus-attack separation is already very strong in the provided feature space.

### 5.2 Multi-class Intrusion Detection
Multi-class recognition is more informative.

- Baseline: accuracy 0.9790, balanced accuracy 0.8041, macro-F1 0.8089
- Proposed local DIDS-MFL: accuracy 0.9774, balanced accuracy 0.7849, macro-F1 0.7987

The baseline remains slightly stronger overall. The proposed representation still performs credibly, but the compression step discards some information that matters for known-class discrimination.

Class-wise results for the proposed model show uneven behavior:

- strong performance on dominant benign and frequent attack classes `2`, `6`, `7`, and `8`
- moderate performance on classes `0`, `1`, `3`, and `5`
- weakest recall on class `4`, with F1 about 0.513

The normalized confusion matrix is shown in `images/multiclass_confusion.png`, and the overall comparison is summarized in `images/performance_comparison.png`.

### 5.3 Few-shot Classification
The few-shot branch targets the three rarest attack classes: `9`, `1`, and `0`, with 5 support examples per class. Using dual-similarity prototypes on the fused representation:

- accuracy: 0.7184
- macro-F1: 0.6137

This is materially lower than the fully supervised setting but still meaningful given the rarity of these classes. The result supports the claim that multiple similarity views can help rare-class recognition, although the benchmark does not include a direct few-shot baseline for a stronger comparison.

### 5.4 Unknown-Attack Generalization
For open-set evaluation, attack class `9` was held out from training. The local surrogate failed in this scenario:

- held-out attack: `9`
- mean maliciousness score on unseen class: 0.9677
- rejection rate: 0.0000

This means the current approach confidently maps the unseen attack into the known-attack manifold instead of identifying it as unknown. The local approximation therefore does not solve the open-set problem.

## 6. Representation Analysis
The fused representation plot in `images/representation_tsne.png` shows that several major classes are separable in a low-dimensional projection, but rare attacks remain partially entangled. This is consistent with the quantitative results:

- binary separation is easy
- known frequent classes are mostly separable
- rare and unknown classes remain the main failure mode

## 7. Discussion
Three findings matter most.

First, the benchmark dataset is easy for binary detection and hard for consistent tail-class detection. Any method that only improves binary accuracy is not addressing the most important part of the problem.

Second, disentangled compression is not automatically beneficial. In this local implementation, the fused subspace helps create a compact few-shot representation but gives up a small amount of multi-class accuracy relative to the stronger raw-feature baseline.

Third, unknown-attack detection requires something more explicit than ordinary discriminative training. A threshold over supervised maliciousness scores is inadequate when unseen attacks resemble known malicious patterns.

## 8. Claim Discipline
The local evidence supports only limited claims.

Supported:

- A local DIDS-MFL-inspired pipeline can be implemented reproducibly from the provided temporal graph tensor.
- Multi-view feature disentanglement plus dynamic context gives competitive, though not state-of-the-art, performance on this benchmark slice.
- Dual-similarity prototypes provide a workable few-shot mechanism for rare attack classes.

Not supported:

- The local approximation does not outperform the raw-feature baseline on binary or multi-class detection.
- The local approximation does not provide effective unknown-attack rejection.
- The current evidence is insufficient to claim improved consistency across all attack types.

## 9. Limitations and Next Local Steps
This study is intentionally constrained by the benchmark environment and the local runtime. The main limitations are:

- no full graph message passing because optional sparse backends are unavailable
- approximate rather than paper-faithful disentanglement
- no external validation dataset
- no broader hyperparameter sweep

The strongest local next steps would be:

1. replace PCA disentanglement with a learned multi-head encoder trained end to end
2. add explicit open-set scoring, such as class-conditional distance thresholds or energy-based rejection
3. evaluate whether history features should be computed in fixed windows rather than cumulative counts
4. compare few-shot prototypes directly against a single-similarity variant

## 10. Reproducibility and Artifacts
Code: [`code/run_analysis.py`](code/run_analysis.py)

Outputs:

- [`outputs/dataset_summary.json`](outputs/dataset_summary.json)
- [`outputs/results.json`](outputs/results.json)
- [`outputs/method_notes.md`](outputs/method_notes.md)

Figures:

- `images/attack_distribution.png`
- `images/representation_tsne.png`
- `images/multiclass_confusion.png`
- `images/performance_comparison.png`

## Conclusion
Within the benchmark’s local-only constraints, the implemented pipeline completes the full ARIS-style loop: literature reading, experiment design, implementation, execution, analysis, claim checking, and report writing. The empirical picture is clear. The proposed local DIDS-MFL approximation is competitive and useful for representation analysis and few-shot classification, but a strong raw-feature baseline remains better on supervised multi-class detection, and unknown-attack generalization remains unresolved.
