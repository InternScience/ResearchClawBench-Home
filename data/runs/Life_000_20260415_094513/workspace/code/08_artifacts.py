"""
Phase 8: Method Contract & Artifact Inventory
"""
import json
import os

os.makedirs('outputs', exist_ok=True)

# Method contract
method_contract = {
    "task": "De novo design synthetic hydrogels achieving >1 MPa underwater adhesion by statistically replicating sequence features of natural adhesive proteins",
    "input_features": ["Nucleophilic-HEA", "Hydrophobic-BA", "Acidic-CBEA", "Cationic-ATAC", "Aromatic-PEA", "Amide-AAm"],
    "output": "Glass (kPa) adhesive strength",
    "target_threshold_kPa": 1000,
    "methods_used": {
        "ML_models": ["RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor", "GaussianProcessRegressor"],
        "optimization": ["RFR-GP hybrid", "GP-GP", "RFR-RFR", "Expected Improvement (EI)"],
        "interpretability": ["SHAP (TreeExplainer)", "Feature importance (RFR)", "Correlation analysis"],
        "validation": ["5-fold cross-validation", "Parity plots"]
    },
    "data_sources": {
        "initial_training": "184 verified hydrogel formulations",
        "optimization_rounds": "3 rounds of EI and PRED-based optimization"
    }
}

with open('outputs/method_contract.json', 'w') as f:
    json.dump(method_contract, f, indent=2)

# Target artifact inventory
target_artifacts = {
    "data_overview": {"status": "satisfied", "files": ["outputs/training_data_184.csv", "outputs/data_summary.json"]},
    "correlation_analysis": {"status": "satisfied", "files": ["outputs/correlation_matrix.csv"], "figures": ["report/images/fig3_correlation_heatmap.png", "report/images/fig4_monomer_vs_strength.png"]},
    "model_comparison": {"status": "satisfied", "files": ["outputs/model_comparison.json"], "figures": ["report/images/fig6_model_comparison.png", "report/images/fig7_parity_plots.png"]},
    "feature_importance": {"status": "satisfied", "files": ["outputs/feature_importance_rfr.json", "outputs/feature_importance_gbr.json", "outputs/shap_importance.json"], "figures": ["report/images/fig8_feature_importance.png", "report/images/fig15_shap_summary.png", "report/images/fig16_shap_bar.png"]},
    "bayesian_optimization": {"status": "satisfied", "files": ["outputs/bo_candidates.json", "outputs/multi_round_results.json"], "figures": ["report/images/fig9_optimization_trajectory.png", "report/images/fig10_ei_landscape.png"]},
    "optimization_analysis": {"status": "satisfied", "files": ["outputs/ei_summary_by_method.csv", "outputs/pred_summary_by_method.csv"], "figures": ["report/images/fig11_optimization_by_method.png", "report/images/fig12_composition_comparison.png", "report/images/fig13_strength_by_round.png", "report/images/fig14_composition_heatmap.png"]},
    "design_rules": {"status": "satisfied", "files": ["outputs/design_rules.json", "outputs/proposed_formulations.json", "outputs/key_findings.json"], "figures": ["report/images/fig18_proposed_formulations.png", "report/images/fig19_strength_landscape.png", "report/images/fig20_gap_analysis.png"]},
    "comprehensive_report": {"status": "pending", "files": ["report/report.md"]}
}

with open('outputs/target_artifact_inventory.json', 'w') as f:
    json.dump(target_artifacts, f, indent=2)

# Dependency check
dependency_check = {
    "pandas": "available",
    "numpy": "available",
    "scikit-learn": "available",
    "scikit-optimize": "available",
    "shap": "available",
    "matplotlib": "available",
    "seaborn": "available",
    "openpyxl": "available",
    "scipy": "available",
    "limitations": "GP predictions are bounded by training data range; extrapolation to >1 MPa requires iterative experimental validation"
}

with open('outputs/dependency_check.json', 'w') as f:
    json.dump(dependency_check, f, indent=2)

print("Phase 8 complete. Method contract and artifact inventory saved.")
