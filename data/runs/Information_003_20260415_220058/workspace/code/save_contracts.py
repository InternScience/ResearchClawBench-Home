"""
DIDS-MFL: Method Contract and Target Artifact Inventory
"""
import json, os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')

# Method Contract
method_contract = {
    "task": "Network Intrusion Detection with DIDS-MFL framework",
    "named_method": "DIDS-MFL (Disentangled Dynamic Intrusion Detection with Multi-scale Fusion Learning)",
    "key_components": [
        {
            "name": "Statistical Disentanglement",
            "description": "Non-parametric MI-based feature weighting to separate entangled feature distributions",
            "implementation": "mutual_info_classif + correlation-based weighting",
            "fidelity_notes": "Approximated SMT optimization with MI/correlation ratio; faithful to 3D-IDS concept"
        },
        {
            "name": "Representational Disentanglement",
            "description": "Memory module + decorrelation loss to highlight attack-specific features",
            "implementation": "MemoryModule (read from class-specific memory) + corrcoef-based decorrelation loss",
            "fidelity_notes": "Faithful to 3D-IDS representational disentanglement concept"
        },
        {
            "name": "Dynamic Graph Diffusion",
            "description": "Spatiotemporal aggregation via graph diffusion on evolving data streams",
            "implementation": "Temporal encoding + spatial diffusion layer combination",
            "fidelity_notes": "Simplified from full multi-layer graph diffusion; uses node-level temporal encoding"
        },
        {
            "name": "Multi-scale Representation Fusion",
            "description": "Multiple representation scales for few-shot learning enhancement",
            "implementation": "MultiScaleFusion with 3 scale branches (32, 64, 96 dims) + fusion layer",
            "fidelity_notes": "Inspired by BSNet bi-similarity concept; extended to multi-scale"
        }
    ],
    "comparison_baselines": ["LogisticRegression", "LightGBM"],
    "evaluation_scenarios": [
        "Binary classification (benign vs attack)",
        "Multi-class classification (10 attack types)",
        "Few-shot attack detection (<1500 samples)",
        "Unknown attack detection (removed from training)"
    ]
}

with open(os.path.join(OUTPUT_DIR, 'method_contract.json'), 'w') as f:
    json.dump(method_contract, f, indent=2)

# Target Artifact Inventory
target_artifacts = {
    "data_overview_figures": {
        "fig1_class_distribution.png": "satisfied",
        "fig2_feature_distributions.png": "satisfied",
        "fig3_temporal_patterns.png": "satisfied",
        "fig4_correlation_heatmaps.png": "satisfied",
        "fig5_entangled_distribution.png": "satisfied"
    },
    "method_artifacts": {
        "fig9_feature_weights.png": "satisfied - statistical disentanglement weights",
        "fig10_disentangled_vs_original.png": "satisfied - representation comparison"
    },
    "result_comparison_figures": {
        "fig6_binary_comparison.png": "satisfied",
        "fig7_multi_comparison.png": "satisfied",
        "fig8_per_type_f1.png": "satisfied",
        "fig11_fewshot_comparison.png": "satisfied",
        "fig12_heatmap_comparison.png": "satisfied",
        "fig13_unknown_attack_detection.png": "satisfied"
    },
    "quantitative_results": {
        "baseline_binary_results.json": "satisfied",
        "baseline_multi_results.json": "satisfied",
        "baseline_per_type_results.json": "satisfied",
        "dids_binary_results.json": "satisfied",
        "dids_multi_results.json": "satisfied",
        "dids_per_type_results.json": "satisfied",
        "unknown_attack_results.json": "satisfied",
        "feature_weights.npy": "satisfied",
        "mi_scores.npy": "satisfied",
        "h_disentangled.npy": "satisfied"
    },
    "report": {
        "report/report.md": "pending"
    }
}

with open(os.path.join(OUTPUT_DIR, 'target_artifact_inventory.json'), 'w') as f:
    json.dump(target_artifacts, f, indent=2)

# Dependency Check
dependency_check = {
    "torch": "available",
    "numpy": "available",
    "sklearn": "available",
    "lightgbm": "available",
    "matplotlib": "available",
    "seaborn": "available",
    "torch_geometric": "available (for data loading)",
    "z3/SMT_solver": "not available - approximated with MI/correlation ratio",
    "cuda": "not available - using CPU"
}

with open(os.path.join(OUTPUT_DIR, 'dependency_check.json'), 'w') as f:
    json.dump(dependency_check, f, indent=2)

print("Contract and inventory files saved.")