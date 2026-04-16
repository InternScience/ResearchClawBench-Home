#!/usr/bin/env python3
"""
Save method contract, target artifact inventory, and dependency check.
"""

import json
import os

# Method contract
method_contract = {
    "task": "Evaluation of computational power of random quantum circuit sampling (RCS) on arbitrary geometries",
    "named_methods": [
        "XEB (Cross-Entropy Benchmarking) fidelity estimation",
        "MB (Measurement Benchmarking / Patch) survival probability",
        "Transport 1QRB (Randomized Benchmarking for transport gates)",
        "Gate-count error propagation model"
    ],
    "key_formulas": {
        "XEB_fidelity": "F_XEB = 2^n * <P(x_i)>_i - 1",
        "MB_survival": "p_survival = count(ideal_bitstring) / total_samples",
        "Gate_count_model": "F_pred = (1-e_1q)^{n_sq} * (1-e_2q)^{n_2q} * (1-e_readout)^{n_ro}"
    },
    "comparison_axes": [
        "Depth scan at fixed N (N=40, N=56)",
        "Qubit scan at fixed depth (d=12)",
        "Gap between experimental fidelity and classical approximability"
    ],
    "core_conclusion_to_validate": "Gap between experimental fidelity and classical approximability under arbitrary-geometry/high-connectivity random circuits"
}

with open('outputs/method_contract.json', 'w') as f:
    json.dump(method_contract, f, indent=2)

# Target artifact inventory
target_artifacts = {
    "primary_quantitative_answers": {
        "fidelity_per_N_d_r": "outputs/instance_fidelity_data.json",
        "aggregated_fidelity": "outputs/fidelity_results.json"
    },
    "required_comparison_tables": {
        "N40_depth_scan_XEB": "computed - see fidelity_results.json",
        "N40_depth_scan_MB": "computed - see fidelity_results.json",
        "N40_depth_scan_model": "computed - see fidelity_results.json",
        "Nscan_d12_XEB": "computed - see fidelity_results.json",
        "Nscan_d12_MB": "computed - see fidelity_results.json",
        "Nscan_d12_model": "computed - see fidelity_results.json"
    },
    "figure_families": {
        "fig1_n40_depth_xeb": "report/images/fig1_n40_depth_xeb.png - SATISFIED",
        "fig2_n40_depth_mb": "report/images/fig2_n40_depth_mb.png - SATISFIED",
        "fig3_n40_depth_combined": "report/images/fig3_n40_depth_combined.png - SATISFIED",
        "fig4_nscan_xeb": "report/images/fig4_nscan_xeb.png - SATISFIED",
        "fig5_nscan_mb": "report/images/fig5_nscan_mb.png - SATISFIED",
        "fig6_nscan_combined": "report/images/fig6_nscan_combined.png - SATISFIED",
        "fig7_transport_depth": "report/images/fig7_transport_depth.png - SATISFIED",
        "fig8_gap_analysis": "report/images/fig8_gap_analysis.png - SATISFIED",
        "fig9_logscale_comparison": "report/images/fig9_logscale_comparison.png - SATISFIED"
    },
    "interpretability_artifacts": {
        "gap_analysis_figure": "report/images/fig8_gap_analysis.png - SATISFIED"
    }
}

with open('outputs/target_artifact_inventory.json', 'w') as f:
    json.dump(target_artifacts, f, indent=2)

# Dependency check
dep_check = {
    "numpy": "available",
    "matplotlib": "available",
    "scipy": "available (curve_fit used in transport figure)",
    "json": "available (standard library)",
    "re": "available (standard library)",
    "limitations": [
        "No amplitude data available for N=48, N=56 XEB computations",
        "No amplitude data for N=56 depth scan XEB computations",
        "MB fidelity computed as raw survival probability (not normalized like XEB)",
        "Gate-count model uses default Sycamore error rates (e_1q=0.16%, e_2q=0.62%, e_readout=1.8%)"
    ]
}

with open('outputs/dependency_check.json', 'w') as f:
    json.dump(dep_check, f, indent=2)

print("Contract, inventory, and dependency files saved.")