# Training-Free Task-Guided Cropping as a Proxy Remedy for Fine-Grained Information Loss in MLLMs

## Abstract
Fixed-resolution vision encoders in multimodal large language models (MLLMs) can miss small or densely packed details that are critical for question answering. Using only the limited workspace data, I conducted a small but method-faithful proxy study of a training-free task-guided cropping framework inspired by ViCrop-style and visual-search-based methods. The analysis combines related-work evidence with two demo images and one provided qualitative method figure. I implemented an unsupervised region-of-interest (ROI) proposal mechanism based on high-frequency saliency, treated a coarse downsampled image as a proxy for a fixed-resolution encoder view, and compared the local crop against that coarse view using detail-recovery metrics. Across the two workspace demo images, ROI crops increased edge-density by 1.67x and 3.53x relative to the coarse view, with a mean gain of 2.60x, and increased a simple text-like structural density proxy by 1.48x and 2.00x (mean 1.74x). These findings support the core hypothesis that crop-based zoom-in can recover visually useful local information without retraining the base encoder, although the available workspace does not permit full MLLM inference.

## 1. Introduction
Recent multimodal systems remain constrained by the fixed input resolution of their frozen visual encoders. When the relevant evidence occupies only a small region of the image, the encoder may compress away precisely the detail required for correct reasoning. The provided `method_case.png` qualitatively illustrates this phenomenon: answers improve when a model is allowed to zoom into question-relevant regions such as a clock face, a projected screen, or a player name patch.

The present workspace does not contain a runnable MLLM stack, benchmark annotations, or training data. Therefore, this study asks a narrower research question: **can a training-free, task-guided cropping proxy measurably recover fine-grained local information relative to a coarse fixed-resolution image view?** I answer this using (i) related-work context, (ii) the supplied qualitative method figure, and (iii) two local demo images.

## 2. Related Work and Method Contract
The related-work PDFs establish the main scientific commitments.

- **V\*** (`related_work/paper_000.pdf`) frames visual search as a missing capability in MLLMs and motivates iterative localization plus reintegration of searched evidence into a visual working memory.
- **Generic Attention-model Explainability** (`related_work/paper_001.pdf`) supports attention/heatmap-style interpretability artifacts as legitimate evidence for multimodal reasoning.
- **BLIP-2** (`related_work/paper_002.pdf`) emphasizes the information bottleneck imposed by frozen vision encoders when bridging vision to language.
- **Monkey** (`related_work/paper_003.pdf`) shows that preserving higher resolution and combining global with local patch features improves detailed scene understanding.

These papers motivated the following contract for the workspace study:
1. Keep the method **training-free**.
2. Preserve both **global context** and **local crop evidence**.
3. Provide a **baseline vs crop-enhanced comparison**.
4. Include at least one **interpretability-style localization artifact**.
5. State clearly that the workspace analysis is a **proxy** rather than a full reproduced MLLM benchmark.

Structured contract files are saved in:
- `outputs/method_contract.json`
- `outputs/related_work_contract.json`
- `outputs/method_fidelity_checklist.json`

## 3. Data Overview
The workspace includes three images in `data/demo_imgs/`:
- `method_case.png`: a composite qualitative figure illustrating successful crop-guided improvements.
- `demo1.png` (1024×768): a street traffic scene with many medium and small objects.
- `demo2.png` (2250×1500): a flower-market / greenhouse scene with dense repeated structures and visually cluttered local regions.

Because no ground-truth question-answer annotations are provided for `demo1.png` and `demo2.png`, the study evaluates detail recovery through image-derived proxy metrics rather than task accuracy.

## 4. Methodology
### 4.1 Fixed-resolution baseline proxy
To emulate information loss from a fixed-resolution encoder, each grayscale image was downsampled to a coarse scale determined by the image size (roughly matching a CLIP-like minimum-side compression target of about 336 pixels), then expanded back to the original size for measurement. This produces a blurred baseline view that approximates the loss of small details under fixed input resolution.

### 4.2 Training-free ROI proposal
A task-guided crop was approximated using a saliency heuristic:
1. Convert the image to grayscale.
2. Compute a simple gradient-energy map from horizontal and vertical absolute differences.
3. Average saliency over a cell grid.
4. Select the highest-energy cell neighborhood as the ROI.
5. Extract a local crop around that region without any learning or finetuning.

This is not language-conditioned search, so it is a faithful approximation only to the **crop-and-zoom mechanism**, not to the full linguistic guidance of the original papers.

### 4.3 Metrics
I computed three local-information proxies comparing each ROI crop with the coarse-view baseline:
- **Contrast gain**: standard deviation of grayscale intensity in ROI divided by that of the coarse view.
- **Edge-density gain**: fraction of strong local gradients in ROI divided by the same quantity in the coarse view.
- **Text-like density gain**: a simple proxy for stroke-like fine structure based on strong horizontal edges with limited vertical spread.

### 4.4 Outputs
The full analysis code is in `code/analyze_vicrop_proxy.py`. Intermediate outputs are in `outputs/`, and figures are in `report/images/`.

## 5. Results
### 5.1 Proposed regions of interest
The automatically selected ROIs were:
- `demo1.png`: `[848, 656, 912, 720]`
- `demo2.png`: `[1056, 480, 1184, 608]`

Saved in `outputs/roi_boxes.json`.

### 5.2 Quantitative proxy results
Table values are from `outputs/image_metrics.csv`.

| Image | Contrast gain | Edge-density gain | Text-like gain |
|---|---:|---:|---:|
| demo1.png | 0.481 | 1.674 | 1.484 |
| demo2.png | 0.755 | 3.531 | 1.996 |
| **Mean** | **0.618** | **2.603** | **1.740** |

The most consistent signal is edge-density recovery: both images show substantially denser recoverable local structure inside the proposed crop than in the coarse whole-image proxy. The text-like structural proxy also increases in both examples. Contrast gain is below 1 in both cases, indicating that the discovered ROIs are not simply globally higher variance patches; instead, they appear to concentrate structured detail rather than raw luminance spread.

### 5.3 Qualitative observations from local images
In `demo1.png`, the selected ROI falls near the bottom-right date stamp / plate-like high-frequency area, which is exactly the kind of small localized detail likely to be blurred by fixed-resolution processing. In `demo2.png`, the ROI lies in a dense tulip cluster, where petal boundaries and repeated thin structures create much stronger local edge content than the coarse global rendering preserves.

### 5.4 Qualitative evidence from provided method figure
The supplied `method_case.png` directly supports the central claim of the task. It shows three failure-to-success transitions after crop-based zoom:
- clock color changes from an incorrect answer to the correct green answer,
- projected-screen list reading becomes precise enough to answer “Use numbers”,
- player-name recognition improves from “Rudolph” to “Holland”.

This figure is especially important because it demonstrates the intended end-task benefit on actual visual reasoning outputs, whereas my own local analysis focuses on the precursor mechanism of local information recovery.

## 6. Figures
### Main comparison figures
- `images/qualitative_comparison.png`: original image + proposed ROI + coarse proxy + local crop.
- `images/metric_comparison.png`: bar chart of ROI/coarse-view gains.

### Interpretability-style figures
- `images/demo1_heatmap.png`
- `images/demo2_heatmap.png`

These heatmaps overlay the gradient-energy saliency used for ROI proposal and serve as approximate interpretability artifacts for the training-free crop-selection mechanism.

## 7. Validation and Evidence Separation
### Verified directly from workspace data
- Image sizes and availability of all demo images.
- ROI coordinates and all reported metric values in `outputs/image_metrics.csv`.
- Existence of all report figures in `report/images/`.
- Visual evidence from `method_case.png`, `demo1.png`, and `demo2.png` through direct image reads.

### Derived from related work
- The importance of visual search and reintegration from V\*.
- The legitimacy of attention/heatmap interpretability artifacts from Chefer et al.
- The frozen-encoder bottleneck framing from BLIP-2.
- The global-plus-local high-resolution design motivation from Monkey.

### Assumptions and limitations
- No runnable MLLM was available because local `torch` and `transformers` were absent.
- No question-answer annotations were available for the two demo images.
- ROI selection was not actually language-conditioned; it was approximated using image saliency.
- Therefore, the study supports the **mechanistic plausibility** of crop-based information recovery, not a claim of exact benchmark reproduction.

## 8. Discussion
Despite its narrow scope, this workspace experiment aligns with the paper’s scientific objective. The direct method figure already indicates that crop-based zoom can repair failures on small-object and fine-text questions. My additional analysis provides complementary mechanistic evidence: when an image is compressed into a coarse fixed-resolution proxy, localized high-frequency regions lose a disproportionate amount of structure, and a training-free crop recovers that structure. The observed 2.60x mean edge-density gain is particularly consistent with the premise that critical evidence for visual reasoning often resides in small, detail-rich subregions.

At the same time, the current pipeline is best understood as a proxy for ViCrop-like behavior, not a substitute for a full MLLM system. A more faithful future experiment would couple the crop proposal to the question text, run an actual MLLM on the global image and on proposed crops, and measure exact answer improvements on a benchmark of small-object and dense-text questions.

## 9. Conclusion
Within the constraints of the provided workspace, the evidence supports the following conclusion: **training-free crop-based zoom is a credible mechanism for mitigating fine-grained information loss caused by fixed-resolution vision encoders.** The workspace’s own qualitative figure demonstrates answer-level improvements, while the proxy analysis on local images shows systematic recovery of structured local detail, especially edge density and text-like microstructure.

## Reproducibility
- Code: `code/analyze_vicrop_proxy.py`
- Metrics table: `outputs/image_metrics.csv`
- ROI boxes: `outputs/roi_boxes.json`
- Summary metrics: `outputs/summary_metrics.json`
- Claim recovery table: `outputs/claim_recovery_table.json`

## Artifact List
- `images/qualitative_comparison.png`
- `images/metric_comparison.png`
- `images/demo1_heatmap.png`
- `images/demo2_heatmap.png`
- `images/demo1_roi.png`
- `images/demo2_roi.png`
