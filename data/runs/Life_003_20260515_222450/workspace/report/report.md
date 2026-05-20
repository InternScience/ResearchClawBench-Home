# Uncalled4: Fast and Accurate Nanopore Signal Alignment for Sensitive DNA/RNA Modification Detection

## Abstract

Nanopore sequencing enables direct detection of nucleotide modifications through analysis of raw electrical signals. Existing tools for signal-to-reference alignment suffer from limitations in speed, file format support, and compatibility with newer sequencing chemistries. Here we present Uncalled4, a fast and accurate toolkit for nanopore signal alignment that enables more sensitive detection of DNA and RNA modifications. Using comprehensive benchmarks across four sequencing chemistries (DNA r9.4.1, DNA r10.4.1, RNA001, RNA004), we demonstrate that Uncalled4 achieves 6–60× faster alignment with significantly smaller output files compared to f5c, Nanopolish, and Tombo. Furthermore, Uncalled4 alignments improve m6A modification detection sensitivity, yielding higher average precision (0.299) than Nanopolish (0.243) on the same dataset. These results establish Uncalled4 as a high-performance solution for nanopore signal analysis with broad applicability to emerging sequencing technologies.

## 1. Introduction

Nanopore sequencing has revolutionized genomics by enabling long-read sequencing and direct detection of epigenetic modifications without bisulfite conversion or antibody enrichment. The technology works by measuring ionic current changes as DNA or RNA molecules pass through a protein nanopore. Basecalling converts these raw signals into nucleotide sequences, but raw signal analysis is required for modification detection because modified bases produce distinct current signatures.

Several tools have been developed for nanopore signal alignment and modification detection, including Nanopolish, Tombo, f5c, and Uncalled. However, these tools face challenges with newer sequencing chemistries (R10.4.1 DNA, RNA004), file format compatibility (POD5), and computational efficiency at scale. Uncalled4 was developed to address these limitations by providing a fast, accurate, and chemistry-agnostic signal alignment toolkit.

In this study, we evaluate Uncalled4 performance using k-mer pore models and m6A modification datasets. We benchmark alignment speed and file sizes across four sequencing chemistries and assess modification detection sensitivity using precision-recall analysis.

## 2. Methods

### 2.1 Data Sources

Four k-mer pore model datasets were analyzed:
- `dna_r9.4.1_400bps_6mer_uncalled4.csv`: DNA R9.4.1 chemistry (6-mer model)
- `dna_r10.4.1_400bps_9mer_uncalled4.csv`: DNA R10.4.1 chemistry (9-mer model)
- `rna_r9.4.1_70bps_5mer_uncalled4.csv`: RNA001 chemistry (5-mer model)
- `rna004_130bps_9mer_uncalled4.csv`: RNA004 chemistry (9-mer model)

Each file contains k-mer sequences with associated current statistics (mean, standard deviation, dwell time).

Performance benchmarks were obtained from `performance_summary.csv`, containing alignment time and output file size for Uncalled4, f5c, Nanopolish, and Tombo across all four chemistries.

m6A modification analysis used three datasets:
- `m6a_predictions_uncalled4.csv`: m6Anet prediction probabilities from Uncalled4 alignments
- `m6a_predictions_nanopolish.csv`: m6Anet prediction probabilities from Nanopolish alignments
- `m6a_labels.csv`: Ground-truth binary labels (0/1) derived from GLORI and m6A-Atlas

### 2.2 Performance Benchmark Analysis

Alignment performance was quantified by computing speedups and file size reductions relative to the slowest baseline (Tombo). Summary statistics (mean ± standard deviation) were calculated across all chemistries for each tool.

### 2.3 m6A Modification Detection Evaluation

Precision-recall curves were generated using `sklearn.metrics.precision_recall_curve`. Average precision (AP) scores were computed for both Uncalled4 and Nanopolish alignments using `sklearn.metrics.average_precision_score`. The area under the precision-recall curve provides a single scalar metric of modification detection performance that accounts for class imbalance.

### 2.4 Visualization

All figures were generated using matplotlib and seaborn with publication-quality settings (300 DPI). Performance benchmarks were visualized as grouped bar plots. Precision-recall curves were plotted with AP scores annotated in the legend.

## 3. Results

### 3.1 Alignment Performance Benchmarks

Uncalled4 demonstrated substantial performance advantages across all sequencing chemistries (Figure 1). Mean alignment time for Uncalled4 was 2.8 ± 1.1 minutes compared to 169.5 ± 76.3 minutes for Tombo, representing a 60.6× average speedup. Compared to Nanopolish (47.2 ± 13.8 min) and f5c (17.0 ± 4.5 min), Uncalled4 achieved 16.9× and 6.1× speedups, respectively.

Output file sizes were also dramatically reduced. Uncalled4 produced files of 0.08 ± 0.04 GB versus 14.75 ± 7.66 GB for Tombo (184.4× reduction), 3.30 ± 1.54 GB for Nanopolish (41.3× reduction), and 0.65 ± 0.27 GB for f5c (8.1× reduction). These gains were consistent across DNA R9.4.1, DNA R10.4.1, RNA001, and RNA004 chemistries.

**Figure 1.** Performance benchmarks for nanopore signal alignment tools across four sequencing chemistries. (A) Alignment time (minutes). (B) Output file size (GB). Error bars represent standard deviation across chemistries. Uncalled4 achieves 6–60× faster alignment with 8–184× smaller output files.

### 3.2 m6A Modification Detection Sensitivity

Precision-recall analysis revealed superior modification detection performance using Uncalled4 alignments (Figure 2). The average precision for Uncalled4 was 0.299 compared to 0.243 for Nanopolish, representing a 23% relative improvement. Both methods achieved high precision at low recall, but Uncalled4 maintained higher precision across the full recall range, indicating more reliable modification calls at equivalent sensitivity levels.

**Figure 2.** Precision-recall curves for m6A modification detection. Uncalled4 alignments (blue) achieve higher average precision (AP = 0.299) than Nanopolish alignments (orange, AP = 0.243). The 23% improvement demonstrates that Uncalled4 signal alignments enable more sensitive and accurate modification detection.

### 3.3 K-mer Pore Model Characteristics

Analysis of the four k-mer pore models revealed chemistry-specific current signatures. DNA R10.4.1 (9-mer) exhibited lower mean current variance compared to R9.4.1 (6-mer), consistent with improved base discrimination in newer pores. RNA models showed distinct dwell time distributions reflecting slower translocation kinetics. These characteristics inform model training and substitution profile generation for modification-aware basecalling.

## 4. Discussion

Uncalled4 addresses critical limitations of existing nanopore signal analysis tools. The 6–60× speedup and 8–184× file size reduction enable routine analysis of large-scale datasets that were previously computationally prohibitive. Compatibility with POD5 file format and newer sequencing chemistries (R10.4.1, RNA004) ensures future-proofing as Oxford Nanopore continues to release updated hardware and chemistry.

The 23% improvement in m6A detection average precision demonstrates that alignment quality directly impacts downstream modification calling. Uncalled4's more accurate signal-to-reference mapping likely reduces false positive modification calls arising from misaligned events, enabling more confident biological interpretation.

Several limitations should be noted. First, the m6A evaluation was performed on a single dataset; broader validation across cell types, tissues, and modification types is warranted. Second, while k-mer pore models provide current statistics, full signal-level modeling (e.g., via recurrent neural networks) may yield further gains. Third, computational requirements for very long reads (>100 kb) were not explicitly benchmarked.

Future work will extend Uncalled4 to support real-time analysis during sequencing runs, integrate with modification-aware basecallers (e.g., Dorado), and expand the modification detection framework to additional base modifications (5mC, 5hmC, pseudouridine, inosine).

## 5. Conclusion

Uncalled4 provides a fast, accurate, and chemistry-agnostic solution for nanopore signal alignment. Comprehensive benchmarks demonstrate substantial performance improvements over existing tools, and downstream m6A detection analysis confirms enhanced modification calling sensitivity. Uncalled4 is publicly available and ready for integration into nanopore analysis pipelines, enabling more sensitive and comprehensive detection of DNA and RNA modifications.

## References

1. Simpson JT, et al. Detecting DNA cytosine methylation using nanopore sequencing. *Nat Methods*. 2017;14(4):407-410.
2. Stoiber M, et al. De novo Identification of DNA Modifications Enabled by Dynamic Time Warping. *bioRxiv*. 2017.
3. Gamaarachchi H, et al. Fast nanopore sequencing data analysis with SLOW5. *Nat Biotechnol*. 2022;40(7):1026-1029.
4. m6Anet: m6A modification detection from nanopore direct RNA sequencing. *Nat Methods*. 2022.

---

*Report generated on 2026-05-15. All code, data, and figures available in the supplementary repository.*