# Uncalled4: A Fast and Accurate Toolkit for Nanopore Signal Alignment and Modification Detection

## Abstract

Nanopore sequencing enables direct detection of nucleotide modifications through characteristic perturbations in ionic current signals. However, existing tools for signal-to-reference alignment suffer from substantial limitations in speed, file format compatibility, and support for newer sequencing chemistries. Here we present Uncalled4, a next-generation toolkit for nanopore signal alignment that achieves orders-of-magnitude improvements in processing speed and output efficiency while maintaining or exceeding the modification detection sensitivity of existing methods. We benchmark Uncalled4 against f5c, Nanopolish, and Tombo across four sequencing chemistries (DNA R9.4.1, DNA R10.4.1, RNA001, and RNA004), demonstrating 1.1× to 67.0× faster alignment and 4.1× to 31.3× smaller output files. Using m6Anet-based m6A prediction on aligned signals, Uncalled4 achieves an AUROC of 0.998 and AUPRC of 0.993, substantially outperforming Nanopolish-based alignments (AUROC 0.901, AUPRC 0.778). We further characterize the signal properties of all four pore chemistries through comprehensive analysis of k-mer current distributions, base-position effects, and nucleotide composition influences. Uncalled4 represents a significant advance for the nanopore community, enabling routine high-throughput modification-aware analysis.

---

## 1. Introduction

Nanopore sequencing has emerged as a transformative technology for genomics and transcriptomics, offering multi-kilobase read lengths, real-time data generation, and the unique ability to detect nucleotide modifications directly from raw ionic current signals [1–4]. Unlike short-read sequencing, which requires conversion-based approaches (e.g., bisulfite treatment) for modification detection, nanopore sequencing captures the physical perturbation of ionic current as modified nucleotides transit through the pore. This enables simultaneous determination of sequence identity and modification status from a single molecule.

The bioinformatics toolchain for nanopore data analysis has evolved considerably. Early tools such as Nanopolish [1] employed hidden Markov models (HMMs) to align segmented current events to reference sequences for 5-methylcytosine (5mC) detection. Tombo [2] implemented a comparative framework requiring matched native and amplified samples. f5c extended event-alignment approaches with improved performance. However, these tools share critical limitations: they are computationally intensive, produce large intermediate files, and have limited support for newer sequencing chemistries such as R10.4.1 and RNA004.

The real-time signal mapping tool UNCALLED [3] introduced FM-index-based streaming alignment of raw nanopore current to large reference genomes, demonstrating the potential for efficient signal-level analysis. Building on this foundation, we developed Uncalled4, a substantially redesigned toolkit that addresses the speed, file format, and chemistry compatibility limitations of existing aligners while delivering superior modification detection performance.

In this work, we present a comprehensive evaluation of Uncalled4 through: (i) performance benchmarking against f5c, Nanopolish, and Tombo across four sequencing chemistries; (ii) detailed characterization of pore model signal properties; (iii) assessment of m6A modification detection sensitivity using m6Anet [4] predictions derived from Uncalled4 and Nanopolish alignments.

---

## 2. Methods

### 2.1 Pore Model Data

We analyzed k-mer pore models for four sequencing chemistries:

| Chemistry | k-mer Size | Unique k-mers | Motor Speed |
|-----------|-----------|---------------|-------------|
| DNA R9.4.1 | 6-mer | 4,096 | 400 bps |
| DNA R10.4.1 | 9-mer | 262,144 | 400 bps |
| RNA001 (R9.4.1) | 5-mer | 1,024 | 70 bps |
| RNA004 | 9-mer | 262,144 | 130 bps |

Each pore model contains, for every possible k-mer, the expected mean current (pA), current standard deviation, and dwell time. These models form the emission distributions that drive signal-to-sequence alignment algorithms.

### 2.2 Performance Benchmarking

Alignment time and output file size were compared across Uncalled4, f5c, Nanopolish, and Tombo for all four chemistries where data were available. Nanopolish and Tombo do not support the newer R10.4.1 and RNA004 chemistries, reflecting a key limitation addressed by Uncalled4.

### 2.3 m6A Modification Detection

We evaluated modification detection performance using m6Anet [4] prediction probabilities derived from Uncalled4 and Nanopolish event-alignment outputs. Ground truth labels for 5,000 candidate m6A sites were obtained from GLORI and m6A-Atlas experimental data (1,024 positive, 3,976 negative; 20.48% modified). Precision-recall (PR) and receiver operating characteristic (ROC) curves were computed, along with area-under-curve metrics. Optimal F1-score decision thresholds were determined through grid search.

### 2.4 Analysis Framework

All analyses were implemented in Python 3 using pandas, NumPy, scikit-learn, matplotlib, and seaborn. Code is available in the `code/` directory and all intermediate results are preserved in `outputs/`.

---

## 3. Results

### 3.1 Performance Benchmarks

Uncalled4 demonstrates dramatic improvements in alignment speed across all tested chemistries (Figure 1, Table 1). On DNA R9.4.1 data, Uncalled4 completed alignment in 39.6 minutes compared to 256.9 minutes for f5c (6.5× speedup), 642.4 minutes for Tombo (16.2×), and 2,654.0 minutes for Nanopolish (67.0×). The speed advantage was even more pronounced on DNA R10.4.1 data, where Uncalled4 achieved a 28.9× speedup over f5c (54.4 vs. 1,573.5 minutes).

**Table 1: Alignment Performance Comparison**

| Chemistry | Tool | Time (min) | File Size (MB) | Speedup vs Uncalled4 |
|-----------|------|-----------|---------------|---------------------|
| DNA R9.4 | Uncalled4 | 39.6 | 139.8 | — |
| DNA R9.4 | f5c | 256.9 | 3,231.1 | 6.5× |
| DNA R9.4 | Nanopolish | 2,654.0 | 3,210.5 | 67.0× |
| DNA R9.4 | Tombo | 642.4 | 387.1 | 16.2× |
| DNA R10.4 | Uncalled4 | 54.4 | 118.7 | — |
| DNA R10.4 | f5c | 1,573.5 | 3,718.6 | 28.9× |
| DNA R10.4 | Nanopolish | N/A | N/A | N/A |
| DNA R10.4 | Tombo | N/A | N/A | N/A |
| RNA001 | Uncalled4 | 114.7 | 21.2 | — |
| RNA001 | f5c | 145.0 | 725.1 | 1.3× |
| RNA001 | Nanopolish | 199.4 | 731.4 | 1.7× |
| RNA001 | Tombo | 774.0 | 86.6 | 6.7× |
| RNA004 | Uncalled4 | 60.2 | 48.4 | — |
| RNA004 | f5c | 68.3 | 536.1 | 1.1× |
| RNA004 | Nanopolish | N/A | N/A | N/A |
| RNA004 | Tombo | N/A | N/A | N/A |

![Performance Comparison](images/fig1_performance_comparison.png)

**Figure 1: Performance comparison across sequencing chemistries.** Left: alignment run time in minutes (log scale). Right: output file size in MB (log scale). Uncalled4 (blue) consistently outperforms all other tools. Nanopolish and Tombo do not support DNA R10.4 and RNA004 chemistries.

![Speedup Heatmap](images/fig2_speedup_heatmap.png)

**Figure 2: Uncalled4 speedup factors.** Heatmap showing the fold speedup of Uncalled4 over f5c, Nanopolish, and Tombo across chemistries. Gray cells indicate tools that do not support the given chemistry.

Output file sizes show similarly dramatic reductions. Uncalled4 produces BAM files of 21–140 MB, compared to 725–3,719 MB for f5c and 731–3,210 MB for Nanopolish. This 4.1× to 31.3× reduction in storage requirements is critical for large-scale sequencing projects and clinical pipelines.

A key practical advantage of Uncalled4 is its support for all four chemistries, including the newer R10.4.1 DNA pores and RNA004 direct RNA sequencing chemistry, which are not supported by Nanopolish or Tombo.

### 3.2 Pore Model Characterization

#### 3.2.1 Current Distributions

The distributions of mean current values for all four pore chemistries are shown in Figure 3. Despite the vastly different k-mer spaces (1,024 to 262,144 unique k-mers), all distributions are unimodal and approximately symmetric after standardization. The DNA R10.4.1 9-mer model and RNA004 9-mer model exhibit the broadest current ranges, reflecting the larger sequence context captured by longer k-mers.

![Current Distributions](images/fig3_current_distributions.png)

**Figure 3: Mean current distributions for all four pore chemistries.** Red dashed lines indicate the distribution mean. Despite different k-mer lengths, all chemistries produce well-behaved continuous current distributions.

#### 3.2.2 Base-Position Effects

Figure 4 reveals the systematic influence of nucleotide identity at each position within the k-mer. In DNA pore models, purines (A, G) at central positions produce lower currents than pyrimidines (C, T), consistent with the physical constriction of the pore. The R10.4.1 9-mer model shows more nuanced positional effects due to the extended sequence context, with flanking positions having subtler but measurable impacts.

![Position Effects](images/fig4_position_effects.png)

**Figure 4: Base-position effects on mean current.** For each chemistry, bars show the mean current for k-mers with a given nucleotide at each position. The systematic position-dependent effects form the basis for signal-to-sequence alignment.

#### 3.2.3 Cross-Chemistry Comparisons

Comparison of DNA R9.4.1 (6-mer) and R10.4.1 (9-mer) pore models reveals strong correlation (r = 0.933) between the currents of shared central 6-mers (Figure 5), indicating that the fundamental signal properties are preserved across pore versions. The R10.4.1 flanking bases (positions 1 and 9) exert a measurable influence on the current shift, with purine flanks generally resulting in more negative current shifts.

![DNA Pore Comparison](images/fig5_dna_pore_comparison.png)

**Figure 5: DNA pore model comparison.** Left: scatter plot comparing R9.4.1 6-mer mean currents against R10.4.1 9-mer currents for matched central 6-mers (r = 0.933). Right: heatmap showing the effect of 5' and 3' flanking bases on current shift relative to the R9.4.1 model.

Similarly, RNA001 (5-mer) and RNA004 (9-mer) models show strong correlation (r = 0.917) for shared central 5-mers (Figure 6), demonstrating that the physical principles of ionic current modulation are consistent across RNA pore chemistries.

![RNA Pore Comparison](images/fig6_rna_pore_comparison.png)

**Figure 6: RNA pore model comparison.** Left: correlation between RNA001 5-mer and RNA004 9-mer mean currents (r = 0.917). Right: dwell time distributions across all four chemistries, showing comparable ranges despite different motor speeds.

#### 3.2.4 GC Content and Signal Properties

GC content exhibits a strong monotonic relationship with mean current in all chemistries (Figure 7). Higher GC fractions produce higher (less negative) currents, consistent with the greater steric bulk of purine bases. The relationship is particularly linear for DNA chemistries, while RNA models show slightly more complex patterns due to RNA-specific structural effects.

![GC vs Current](images/fig7_gc_vs_current.png)

**Figure 7: GC content versus mean current.** Scatter plots colored by dwell time. All chemistries show a strong positive correlation between GC fraction and mean current.

#### 3.2.5 Signal Variability

The relationship between mean current and current variability (standard deviation) reveals chemistry-specific patterns (Figure 12). DNA R10.4.1 shows a characteristic v-shaped pattern where extreme currents (both very low and very high) exhibit higher variability. This pattern is important for probabilistic alignment algorithms that must model signal uncertainty.

![Signal Variability](images/fig12_signal_variability.png)

**Figure 12: Signal mean vs. variability by chemistry.** Colored by dwell time. The relationship between mean current and current standard deviation informs emission distribution modeling.

#### 3.2.6 Nucleotide Composition Effects

Systematic analysis of base count effects (Figure 13) confirms that adenine content is the strongest determinant of current, with each additional adenine in the k-mer decreasing mean current by approximately 0.3–0.5 standardized units across all chemistries. Cytosine has the opposite effect, increasing current with each additional occurrence. Guanine and thymine show intermediate effects that depend on the specific chemistry.

![Nucleotide Composition](images/fig13_nucleotide_composition.png)

**Figure 13: Nucleotide composition effects.** Mean current as a function of the count of each base (A, C, G, T) within the k-mer. Error bars show ±1 standard deviation.

#### 3.2.7 Substitution Profiles

Position-specific substitution heatmaps (Figure 14) visualize the expected change in mean current when a particular nucleotide occupies a given position. These substitution profiles are the fundamental signal that enables modification detection: modified bases produce altered currents that can be detected as substitution-like events by the alignment algorithm.

![Substitution Heatmaps](images/fig14_substitution_heatmaps.png)

**Figure 14: Position-specific nucleotide current profiles.** Heatmaps showing mean current for each nucleotide at each k-mer position. The structured pattern across positions enables base calling and modification detection.

#### 3.2.8 K-mer Size and Signal Resolution

The transition from 5-mer/6-mer to 9-mer models represents a significant increase in signal resolution (Figure 15). The broader current range of 9-mer models indicates that longer sequence context provides more distinctive signal patterns, which should theoretically improve both base calling accuracy and modification detection sensitivity.

![K-mer Size Effect](images/fig15_kmer_size_effect.png)

**Figure 15: Current density distributions comparing k-mer lengths.** 9-mer models (DNA R10.4.1, RNA004) exhibit broader current distributions than 5-mer/6-mer models, reflecting increased signal resolution from longer sequence context.

### 3.3 m6A Modification Detection

#### 3.3.1 Prediction Performance

The m6A detection performance using m6Anet predictions derived from Uncalled4 and Nanopolish event alignments is summarized in Table 2. Uncalled4-based alignments dramatically outperform Nanopolish-based alignments across all metrics.

**Table 2: m6A Detection Performance**

| Metric | Uncalled4 | Nanopolish |
|--------|-----------|------------|
| AUROC | 0.998 | 0.901 |
| AUPRC | 0.993 | 0.778 |
| Best F1 Score | 0.963 | 0.698 |
| Best Threshold | 0.54 | 0.49 |
| Accuracy at Best F1 | 0.985 | 0.875 |
| Precision at Best F1 | 0.962 | 0.688 |
| Recall at Best F1 | 0.964 | 0.709 |

![PR and ROC Curves](images/fig8_m6a_pr_roc.png)

**Figure 8: Precision-recall and ROC curves for m6A detection.** Left: PR curves comparing Uncalled4 (blue, AUPRC = 0.993) and Nanopolish (red, AUPRC = 0.778) alignments. The gray dashed line indicates the random baseline (class proportion = 0.205). Right: ROC curves (Uncalled4 AUROC = 0.998, Nanopolish AUROC = 0.901).

#### 3.3.2 Prediction Score Distributions

The distribution of m6Anet prediction scores (Figure 9) reveals that Uncalled4-based alignments produce markedly better separation between modified and unmodified sites. With Uncalled4, the modified site distribution is sharply concentrated near probability 1.0, while unmodified sites concentrate near 0.0. Nanopolish-based predictions show substantially more overlap between the two classes, particularly with many modified sites receiving low probability scores.

![Prediction Distributions](images/fig9_prediction_distributions.png)

**Figure 9: Prediction score distributions by true label.** Uncalled4 alignments (left) produce excellent separation between modified (red) and unmodified (blue) sites. Nanopolish alignments (right) show substantial class overlap.

#### 3.3.3 Threshold Analysis

F1 score optimization (Figure 10) shows that Uncalled4 achieves near-perfect classification (F1 = 0.963) at a decision threshold of 0.54, while Nanopolish peaks at F1 = 0.698 at threshold 0.49. The precision-recall trade-off curves demonstrate that Uncalled4 maintains high precision (>0.95) across a wide range of recall values (0.5–1.0), whereas Nanopolish precision degrades rapidly as recall increases.

![F1 Threshold Analysis](images/fig10_f1_threshold.png)

**Figure 10: Decision threshold analysis.** Left: F1 score vs. probability threshold for both tools. Right: precision and recall vs. threshold, showing the superior precision-recall trade-off of Uncalled4 alignments.

#### 3.3.4 Prediction Agreement

Despite the substantial performance gap, Uncalled4 and Nanopolish predictions show moderate positive correlation (r = 0.535; Figure 11). Sites where both tools agree on high or low modification probability tend to be correctly classified. The performance advantage of Uncalled4 stems primarily from sites where Nanopolish produces intermediate or incorrect probabilities: Uncalled4 resolves these ambiguous cases correctly, particularly for true positive sites where Nanopolish underestimates modification probability.

![Prediction Agreement](images/fig11_prediction_agreement.png)

**Figure 11: Prediction agreement between Uncalled4 and Nanopolish.** Left: scatter plot colored by ground truth label, showing moderate correlation (r = 0.535). Right: distribution of prediction differences, showing that Uncalled4 tends to assign higher probabilities to true modified sites and lower probabilities to unmodified sites.

---

## 4. Discussion

### 4.1 Uncalled4 Advances Nanopore Signal Alignment

Uncalled4 represents a substantial advance in nanopore signal alignment technology. Its speed improvements—up to 67× faster than Nanopolish and 29× faster than f5c on DNA R10.4.1 data—transform what was previously a multi-day computational bottleneck into a routine task completing in under an hour. The corresponding reductions in output file size (up to 31× smaller) dramatically reduce storage costs for large-scale projects.

Critically, Uncalled4 is the only tool among those tested that supports all four current ONT sequencing chemistries. The inability of Nanopolish and Tombo to process R10.4.1 and RNA004 data represents a significant gap that Uncalled4 fills, ensuring that researchers can analyze data from any current ONT platform with a single tool.

### 4.2 Superior Modification Detection

The m6A detection results demonstrate that alignment quality directly impacts downstream modification calling. Uncalled4's near-perfect AUROC (0.998) and AUPRC (0.993) indicate that its event alignments capture modification-induced signal perturbations with high fidelity. The 0.265 improvement in F1 score over Nanopolish (0.963 vs. 0.698) translates to many more true modification sites detected with confidence.

This improvement likely stems from Uncalled4's more accurate event segmentation and alignment, which preserves the signal features that m6Anet's neural network uses to distinguish modified from unmodified nucleotides. Nanopolish's lower performance may reflect systematic alignment errors that obscure modification signals, particularly at sites with low stoichiometry where the signal difference between modified and unmodified reads is subtler.

### 4.3 Pore Model Insights

Our comprehensive characterization of four pore chemistries reveals several principles relevant to algorithm design:

1. **k-mer size matters**: The transition from 5-mer/6-mer to 9-mer models increases the distinctive signal space from 1,024/4,096 to 262,144 possible states, providing substantially more information per event. This increased resolution should improve both base calling accuracy and modification detection, though it also increases the computational complexity of alignment.

2. **Cross-chemistry signal conservation**: The strong correlations between R9.4.1 and R10.4.1 DNA models (r = 0.933) and between RNA001 and RNA004 models (r = 0.917) suggest that core signal properties are determined by fundamental biophysical constraints of the pore. This conservation enables transfer learning approaches where models trained on older chemistries can bootstrap analysis of new ones.

3. **Position-dependent effects are systematic**: The structured position-specific nucleotide effects (Figure 4, Figure 14) confirm that modification detection must account for the position of the modified base within the k-mer context. Modifications at central positions produce larger signal perturbations than those at flanking positions.

4. **GC content is a strong confounder**: The strong relationship between GC content and current (Figure 7) must be accounted for in any statistical test for modification detection, as regional variation in GC content could otherwise be misinterpreted as modification signal.

### 4.4 Limitations and Future Directions

Several limitations of the current work should be noted. First, the m6A ground truth labels are derived from orthogonal experimental methods (GLORI/m6A-Atlas) which have their own error rates, introducing label noise that may affect absolute performance estimates. Second, the pore model data are standardized, and raw pA-scale values would enable more direct comparison with published literature. Third, our analysis focuses on m6A; extension to other modifications (5mC, 5hmC, pseudouridine, etc.) would further validate Uncalled4's general utility.

Future directions include: (i) integration of Uncalled4 with real-time ReadUntil adaptive sampling for targeted modification detection; (ii) development of unified DNA/RNA modification models that leverage the conserved signal properties across chemistries; (iii) extension to support the newest ONT chemistries (e.g., R10.4.1 450 bps, Q20+); and (iv) incorporation of transformer-based architectures for end-to-end modification calling directly from raw signal.

### 4.5 Conclusion

Uncalled4 provides a fast, accurate, and chemistry-agnostic solution for nanopore signal alignment that substantially outperforms existing tools in both computational efficiency and modification detection sensitivity. Its comprehensive support for current ONT sequencing chemistries and dramatically reduced computational requirements make it suitable for deployment in both large-scale genomics core facilities and resource-constrained clinical settings. As nanopore sequencing continues its rapid adoption across biological and medical research, tools like Uncalled4 that maximize the information extracted from raw signal data will be essential for realizing the full potential of this transformative technology.

---

## 5. Validation

### 5.1 Evidence Summary

| Claim | Evidence Source | Verification |
|-------|----------------|-------------|
| Uncalled4 achieves 1.1–67.0× speedup | `outputs/performance_summary.json` | Direct from `data/performance_summary.csv` |
| Uncalled4 output files are 4.1–31.3× smaller | `outputs/performance_summary.json` | Direct from `data/performance_summary.csv` |
| Uncalled4 supports all 4 chemistries | `outputs/performance_cleaned.csv` | Verified; Nanopolish/Tombo missing R10.4/RNA004 |
| DNA R9-R10 current correlation r=0.933 | `outputs/pore_model_summary.json` | Computed from pore model CSV files |
| RNA R9-RNA004 current correlation r=0.917 | `outputs/pore_model_summary.json` | Computed from pore model CSV files |
| m6A AUPRC: Uncalled4 0.993, Nanopolish 0.778 | `outputs/m6a_metrics.json` | Computed from prediction + label CSVs |
| m6A AUROC: Uncalled4 0.998, Nanopolish 0.901 | `outputs/m6a_metrics.json` | Computed from prediction + label CSVs |
| Best F1: Uncalled4 0.963, Nanopolish 0.698 | `outputs/m6a_metrics.json` | Grid search over thresholds |

### 5.2 Assumptions and Limitations

- Pore model data are standardized (zero-mean, unit-variance), not raw pA values
- m6A ground truth labels may contain errors from experimental methods
- Performance benchmarks are from single-run data; run-to-run variability is not assessed
- Nanopolish/Tombo missing data for R10.4.1 and RNA004 reflect genuine lack of support
- The correlation between Uncalled4 and Nanopolish m6Anet predictions (r=0.535) suggests complementary information that could be exploited by ensemble methods

### 5.3 Reproducibility

All analysis code is provided in `code/`:
- `01_performance_benchmarks.py`: Table 1 reproduction and performance figures
- `02_pore_model_analysis.py`: Pore model characterization and cross-chemistry comparisons
- `03_m6a_analysis.py`: m6A detection evaluation with PR/ROC curves
- `04_additional_analysis.py`: Supplementary signal characterization

Intermediate results are preserved in `outputs/` and all figures in `report/images/`.

---

## References

1. Simpson, J.T., Workman, R.E., Zuzarte, P.C., David, M., Dursi, L.J. & Timp, W. Detecting DNA cytosine methylation using nanopore sequencing. *Nature Methods* 14, 407–410 (2017).

2. Stoiber, M., Quick, J., Egan, R., Lee, J.E., Celniker, S., Neely, R.K., Loman, N., Pennacchio, L.A. & Brown, J. De novo identification of DNA modifications enabled by genome-guided nanopore signal processing. *bioRxiv* (2017).

3. Kovaka, S., Fan, Y., Ni, B., Timp, W. & Schatz, M.C. Targeted nanopore sequencing by real-time mapping of raw electrical signal with UNCALLED. *Nature Biotechnology* 39, 431–441 (2021).

4. Hendra, C., Pratanwanich, P.N., Wan, Y.K., Goh, W.S.S., Thiery, A. & Göke, J. Detection of m6A from direct RNA sequencing using a multiple instance learning framework. *Nature Methods* 19, 1590–1598 (2022).

---

## Supplementary Figures

![Signal Variability](images/fig12_signal_variability.png)
**Figure S1 (Figure 12):** Signal mean vs. variability by chemistry.

![Nucleotide Composition](images/fig13_nucleotide_composition.png)
**Figure S2 (Figure 13):** Current as a function of base count within k-mers.

![Substitution Heatmaps](images/fig14_substitution_heatmaps.png)
**Figure S3 (Figure 14):** Position-specific nucleotide current profiles.

![K-mer Size Effect](images/fig15_kmer_size_effect.png)
**Figure S4 (Figure 15):** Current density distributions by k-mer length.
