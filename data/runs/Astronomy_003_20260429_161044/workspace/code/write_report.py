#!/usr/bin/env python3
from pathlib import Path
from string import Template
import json
import pandas as pd
fig6=pd.read_csv('outputs/fig6_summary.csv').iloc[0]
fig7=pd.read_csv('outputs/fig7_mode_summary.csv')
fig8=pd.read_csv('outputs/fig8_extrapolation_summary.csv')
th=pd.read_csv('outputs/threshold_fractions.csv')
val=json.loads(Path('outputs/data_validation.json').read_text())
paired=fig8[fig8.diagnostic=='paired_N2N4_over_N2N3'].iloc[0]
n23=fig8[fig8.diagnostic=='N2_vs_N3'].iloc[0]
n24=fig8[fig8.diagnostic=='N2_vs_N4'].iloc[0]
ell2=fig7[fig7.ell==2].iloc[0]
ell8=fig7[fig7.ell==8].iloc[0]
mode_table=fig7[['ell','median','q90','q95','frac_le_0.001','frac_le_0.01','max']].copy()
mode_table.columns=['ell','median','90th pct','95th pct','frac <=1e-3','frac <=1e-2','max']
mode_md=mode_table.to_markdown(index=False, floatfmt='.3e')
ext_table=pd.DataFrame([
    {'comparison':'N=2 vs N=3','median':n23['median'],'95th pct':n23['q95'],'frac <=1e-4':n23['frac_le_0.0001'],'frac <=1e-3':n23['frac_le_0.001'],'max':n23['max']},
    {'comparison':'N=2 vs N=4','median':n24['median'],'95th pct':n24['q95'],'frac <=1e-4':n24['frac_le_0.0001'],'frac <=1e-3':n24['frac_le_0.001'],'max':n24['max']},
])
ext_md=ext_table.to_markdown(index=False, floatfmt='.3e')
subs={
 'fig6_median':f"{fig6['median']:.3e}", 'fig6_geom':f"{fig6['geom_mean']:.3e}", 'fig6_mean':f"{fig6['mean']:.3e}",
 'fig6_q90':f"{fig6['q90']:.3e}", 'fig6_q95':f"{fig6['q95']:.3e}", 'fig6_q99':f"{fig6['q99']:.3e}", 'fig6_max':f"{fig6['max']:.3e}",
 'fig6_le1e3':f"{fig6['frac_le_0.001']:.1%}", 'fig6_le1e2':f"{fig6['frac_le_0.01']:.1%}", 'fig6_gt1e2':f"{fig6['frac_gt_0.01']:.1%}",
 'ell2_med':f"{ell2['median']:.3e}", 'ell8_med':f"{ell8['median']:.3e}", 'ell_factor':f"{ell8['median']/ell2['median']:.1f}",
 'ell2_q95':f"{ell2['q95']:.3e}", 'ell8_q95':f"{ell8['q95']:.3e}", 'ell2_le1e3':f"{ell2['frac_le_0.001']:.1%}", 'ell8_le1e3':f"{ell8['frac_le_0.001']:.1%}", 'ell8_le1e2':f"{ell8['frac_le_0.01']:.1%}",
 'n23_med':f"{n23['median']:.3e}", 'n24_med':f"{n24['median']:.3e}", 'n23_le1e4':f"{n23['frac_le_0.0001']:.1%}", 'n24_le1e4':f"{n24['frac_le_0.0001']:.1%}",
 'paired_frac':f"{paired['fraction_N2N4_gt_N2N3']:.1%}", 'paired_ratio':f"{paired['median_ratio']:.2f}", 'ratio_q05':f"{paired['q05_ratio']:.2f}", 'ratio_q95':f"{paired['q95_ratio']:.2f}", 'wilcox_p':f"{paired['wilcoxon_pvalue']:.2e}", 'rho':f"{paired['spearman_rho']:.3f}",
 'fig6_n':val['fig6_data.csv']['shape'][0], 'fig7_n':val['fig7_data.csv']['shape'][0], 'fig8_n':val['fig8_data.csv']['shape'][0], 'fig7_cols':val['fig7_data.csv']['shape'][1],
 'mode_md':mode_md, 'ext_md':ext_md,
}
template=Template(r'''# Numerical-Relativity Waveform Accuracy Diagnostics for a Synthetic SXS Binary-Black-Hole Catalog

## Abstract

This report analyzes three workspace datasets that emulate accuracy diagnostics from the SXS binary-black-hole (BBH) numerical-relativity catalog: a catalog-wide highest-resolution waveform difference (`fig6_data.csv`), mode-resolved differences for spherical-harmonic degrees $\ell=2,\ldots,8$ (`fig7_data.csv`), and paired finite-radius extrapolation-order comparisons (`fig8_data.csv`). The data contain summary uncertainty diagnostics rather than raw strain, Weyl scalar, horizon trajectories, or metadata; therefore the study focuses on reproducible catalog-quality assessment rather than waveform generation. The principal results are: (i) the catalog-wide median resolution difference is $fig6_median, close to the stated SXS-scale target of 4e-4; (ii) $fig6_le1e3 of catalog-wide differences are at or below 1e-3, but a measurable long tail remains; (iii) mode-wise medians increase monotonically from $ell2_med at ell=2 to $ell8_med at ell=8; and (iv) the N=2 versus N=4 extrapolation comparison is larger than the N=2 versus N=3 comparison in $paired_frac of paired simulations, with a median ratio of $paired_ratio.

## 1. Scientific context and scope

High-accuracy BBH catalogs support gravitational-wave searches, parameter estimation, waveform-model calibration, surrogate modeling, and tests of strong-field gravity. The related-work PDFs in `related_work/` reinforce three methodological points used here. First, SXS/SpEC waveforms are treated as highly accurate reference data but still require explicit error diagnostics, including numerical truncation/resolution error, finite-radius extraction or extrapolation error, and gauge-related effects such as center-of-mass motion. Second, comparing the two highest numerical resolutions is a standard conservative way to estimate numerical waveform error in SXS-related analyses. Third, higher harmonics are scientifically relevant: mode mixing, nonlinear ringdown content, and surrogate-model training all motivate preserving the ell-resolved structure rather than reporting only a pooled mismatch.

The workspace does not include the initial BBH parameters, strain time series, Weyl scalar Psi4, apparent-horizon masses/spins/trajectories, or detailed metadata described in the broad scientific goal. Consequently, this report evaluates the provided uncertainty diagnostics as a catalog-quality audit. All numeric claims below are traceable to CSV/JSON artifacts in `outputs/`, and all figures are PNG files in `report/images/`.

## 2. Data and methods

### 2.1 Input datasets

Validation found no missing, nonpositive, or non-finite values. The files have the expected dimensions:

- `fig6_data.csv`: $fig6_n simulations x 1 catalog-wide resolution-difference column.
- `fig7_data.csv`: $fig7_n simulations x $fig7_cols modal columns, renamed `ell_2` through `ell_8`.
- `fig8_data.csv`: $fig8_n simulations x 2 paired extrapolation comparisons (`N2_vs_N3`, `N2_vs_N4`).

### 2.2 Analysis protocol

The script `code/analyze_waveform_uncertainty.py` performs the complete analysis. For each diagnostic it computes sample size, missing count, minimum, mean, standard deviation, geometric mean, median, 1/5/10/90/95/99% quantiles, maximum, and pass/fail fractions at thresholds 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, and 1e-1. The paired extrapolation analysis also computes the ratio `(N2-N4)/(N2-N3)`, the paired log10 difference, a one-sided Wilcoxon signed-rank test on log differences, and a Spearman correlation between paired log values. Log scales are used in figures because all three datasets are positive and strongly right-skewed.

Primary saved artifacts are `outputs/data_validation.json`, `outputs/fig6_summary.csv`, `outputs/fig7_mode_summary.csv`, `outputs/fig8_extrapolation_summary.csv`, `outputs/threshold_fractions.csv`, source tables for figures, and `outputs/claim_recovery_table.csv`.

## 3. Results

### 3.1 Catalog-wide resolution differences

![Catalog-wide highest-resolution waveform-difference distribution](images/fig6_catalog_distribution.png)

**Figure 1.** Histogram and empirical cumulative distribution of the catalog-wide highest-resolution waveform difference. Vertical guides mark the median and selected accuracy thresholds.

The catalog-wide resolution diagnostic is concentrated near the stated SXS-like scale but retains a long upper tail. The median is $fig6_median, the geometric mean is $fig6_geom, and the arithmetic mean is $fig6_mean, larger than the median because of right skew. The central/tail quantiles are: 90th percentile $fig6_q90, 95th percentile $fig6_q95, and 99th percentile $fig6_q99; the maximum is $fig6_max. Threshold coverage is high at practical mismatch-like levels: $fig6_le1e3 of entries are at or below 1e-3, $fig6_le1e2 are at or below 1e-2, and $fig6_gt1e2 exceed 1e-2. No entry exceeds 1e-1.

These numbers support the interpretation that most synthetic catalog simulations achieve high resolution consistency, while a small high-difference tail should be flagged for waveform-model calibration or downstream data-analysis applications.

### 3.2 Mode-resolved uncertainty across ell=2--8

![Mode-wise waveform-difference distributions](images/fig7_mode_distributions.png)

**Figure 2.** Left: log-scale distribution of waveform differences for each spherical-harmonic degree. Right: median, 10--90% band, and 95th percentile versus ell.

The modal data show a systematic degradation with increasing harmonic degree. The median rises monotonically by a factor of $ell_factor, from $ell2_med at ell=2 to $ell8_med at ell=8. The high-percentile tail also expands: the 95th percentile grows from $ell2_q95 to $ell8_q95. This is consistent with the task description and with the related-work motivation for monitoring higher harmonics separately.

$mode_md

The threshold table reveals a practical transition. At ell=2, $ell2_le1e3 of entries are below 1e-3; by ell=8, only $ell8_le1e3 are below that threshold. Nevertheless, even at ell=8, $ell8_le1e2 remain at or below 1e-2. This supports using ell-dependent uncertainty budgets or mode-truncation checks rather than one global tolerance.

### 3.3 Extrapolation-order consistency

![Extrapolation-order waveform-difference comparison](images/fig8_extrapolation_comparison.png)

**Figure 3.** Left: distributions of the two extrapolation-order comparisons. Right: paired scatter; points above the diagonal have larger N=2 vs N=4 differences than N=2 vs N=3 differences.

The finite-radius extrapolation diagnostic behaves as expected: the broader separation in extrapolation order (N=2 vs N=4) gives larger differences than N=2 vs N=3. The median N=2 vs N=3 difference is $n23_med, while the median N=2 vs N=4 difference is $n24_med. In paired simulations, N=2 vs N=4 is larger in $paired_frac of cases; the median ratio is $paired_ratio, with a 5--95% ratio interval of [$ratio_q05, $ratio_q95]. A one-sided Wilcoxon test on log differences gives p=$wilcox_p, confirming that the median paired log difference is positive in this synthetic dataset.

$ext_md

The paired Spearman correlation is low (rho=$rho), indicating that the two extrapolation diagnostics are not simply a fixed multiplicative rescaling per simulation. Catalog validation should therefore retain both order comparisons when available.

### 3.4 Cross-diagnostic threshold coverage

![Threshold coverage heatmap](images/summary_threshold_heatmap.png)

**Figure 4.** Fraction of simulations at or below selected mismatch-like thresholds for every diagnostic family.

The heatmap summarizes the main practical outcome. Resolution errors and low-ell modal errors are mostly below 1e-3, whereas high-ell modes frequently sit between 1e-3 and 1e-2. Extrapolation differences are substantially smaller: $n23_le1e4 of N=2 vs N=3 values and $n24_le1e4 of N=2 vs N=4 values are at or below 1e-4, and nearly all are at or below 1e-3.

## 4. Validation and traceability

### 4.1 Directly verified from workspace data

- File shapes, positivity, and missing-value checks are stored in `outputs/data_validation.json`.
- Catalog-wide summary statistics are stored in `outputs/fig6_summary.csv`.
- Mode-wise summaries are stored in `outputs/fig7_mode_summary.csv` and the long-form figure source table is `outputs/fig7_mode_long_source.csv`.
- Extrapolation summaries and paired statistics are stored in `outputs/fig8_extrapolation_summary.csv` and `outputs/fig8_pair_source.csv`.
- Threshold fractions used in Figure 4 are stored in `outputs/threshold_fractions.csv`.
- A claim-to-artifact mapping is stored in `outputs/claim_recovery_table.csv`.

### 4.2 Related-work inputs

The local related-work PDFs were read with `pypdf` after the `ReadPDF` tool failed. The extraction is saved in `outputs/related_work_raw_extract.json`, and the task-relevant synthesis is saved in `outputs/related_work_contract.json`. Related work was used only to motivate diagnostic choices: highest-resolution comparisons, extrapolation-error checks, and mode-resolved treatment of higher harmonics.

### 4.3 Assumptions and limitations

- The datasets are synthetic positive waveform-difference summaries. They are treated as mismatch-like diagnostics, but the report does not claim they are detector-noise-weighted mismatches unless explicitly labeled by the source data.
- The analysis cannot reconstruct strain h, Weyl scalar Psi4, horizon mass/spin trajectories, recoil velocities, or detailed metadata because those raw products are absent.
- No simulation parameters such as mass ratio, spin vectors, eccentricity, resolution labels, extraction radii, or sky locations are present, so subgroup accuracy by physical parameter cannot be assessed.
- The extrapolation-order comparison has paired rows, but row identity across the three different files is not documented; cross-file per-simulation joins were therefore not performed.

## 5. Discussion

The synthetic catalog diagnostics are broadly consistent with a high-accuracy BBH numerical-relativity catalog. The resolution-difference median near 4e-4 and the $fig6_le1e3 fraction below 1e-3 indicate that most simulations are suitable as calibration-quality waveform data under this diagnostic. However, the right tail matters scientifically: even a small number of simulations with differences above 1e-2 may dominate model-training residuals or bias validation if not down-weighted, excluded, or inspected individually.

The clearest structural finding is mode dependence. The factor-$ell_factor increase in median difference from ell=2 to ell=8 means that a single catalog-wide tolerance can hide important higher-mode uncertainty. This is particularly relevant for asymmetric, precessing, eccentric, or high-signal-to-noise systems where subdominant harmonics carry astrophysical information. A robust catalog release should therefore expose per-mode error estimates and document the recommended maximum ell for different applications.

The extrapolation-order data show smaller absolute differences than the resolution and high-ell modal diagnostics, but the paired N=2 vs N=4 excess demonstrates that extrapolation uncertainty is not negligible. Because the paired correlation between the two extrapolation comparisons is weak, retaining multiple extrapolation diagnostics is preferable to assuming one order-pair fully predicts another.

## 6. Conclusions

1. The catalog-wide highest-resolution waveform difference has median $fig6_median; $fig6_le1e3 of simulations are at or below 1e-3, while $fig6_gt1e2 exceed 1e-2.
2. Mode-wise differences increase with ell: the median grows from $ell2_med at ell=2 to $ell8_med at ell=8, and the high-mode tail can approach 1e-1.
3. Extrapolation differences are mostly small but order-pair dependent: N=2 vs N=4 has median $n24_med, compared with $n23_med for N=2 vs N=3, and is larger in $paired_frac of paired simulations.
4. The available data support a catalog-quality audit, not full waveform/horizon catalog construction. Future work would require raw waveforms, extraction radii, resolution labels, physical BBH parameters, and horizon metadata to connect these uncertainty diagnostics to source-parameter coverage and waveform-model calibration performance.
''')
Path('report/report.md').write_text(template.safe_substitute(subs))
print('wrote report/report.md', Path('report/report.md').stat().st_size)
