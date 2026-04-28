"""
06_claim_recovery.py
Build a small claim-recovery table that ties report claims to artifact paths
and to numeric values pulled directly from the saved JSON/CSV files.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"

m_a = json.load(open(OUT / "metrics_attack.json"))
m_s = json.load(open(OUT / "metrics_sniffing.json"))

def fmt(d, k):
    v = d["models"][k]
    return f"ROC-AUC={v['roc_auc']:.3f}, PR-AUC={v['pr_auc']:.3f}, F1={v['f1']:.3f}"

claims = [
    {
        "claim": "Random Forest reproduces SimBA-style Attack classification with strong discrimination on the 1738-frame sample (5-fold stratified CV).",
        "evidence_artifact": "outputs/metrics_attack.json (RandomForest_CV)",
        "value": fmt(m_a, "RandomForest_CV"),
    },
    {
        "claim": "Random Forest reproduces SimBA-style Sniffing classification with strong discrimination (5-fold stratified CV).",
        "evidence_artifact": "outputs/metrics_sniffing.json (RandomForest_CV)",
        "value": fmt(m_s, "RandomForest_CV"),
    },
    {
        "claim": "Threshold tuning on the precision-recall curve substantially improves F1 over the default 0.5 cutoff.",
        "evidence_artifact": "outputs/metrics_*.json (RandomForest_CV_best_threshold)",
        "value": (f"Attack: F1 {m_a['models']['RandomForest_CV']['f1']:.3f} -> "
                  f"{m_a['models']['RandomForest_CV_best_threshold']['best_f1']:.3f} "
                  f"@ thr={m_a['models']['RandomForest_CV_best_threshold']['best_threshold']:.2f}; "
                  f"Sniffing: F1 {m_s['models']['RandomForest_CV']['f1']:.3f} -> "
                  f"{m_s['models']['RandomForest_CV_best_threshold']['best_f1']:.3f} "
                  f"@ thr={m_s['models']['RandomForest_CV_best_threshold']['best_threshold']:.2f}"),
    },
    {
        "claim": "Gradient Boosting performs comparably to Random Forest, indicating SimBA's RF default is not arbitrary.",
        "evidence_artifact": "outputs/model_comparison.csv",
        "value": (f"Attack RF/GB ROC-AUC: {m_a['models']['RandomForest_CV']['roc_auc']:.3f}/"
                  f"{m_a['models']['GradientBoosting_CV']['roc_auc']:.3f}; "
                  f"Sniffing RF/GB ROC-AUC: {m_s['models']['RandomForest_CV']['roc_auc']:.3f}/"
                  f"{m_s['models']['GradientBoosting_CV']['roc_auc']:.3f}"),
    },
    {
        "claim": "Random K-fold CV is over-optimistic relative to a temporally-honest chronological hold-out for these long bouts; this is a known caveat for bout-level ethograms and exposes a transparent limitation.",
        "evidence_artifact": "outputs/metrics_*.json (RandomForest_chronological_holdout)",
        "value": (f"Attack hold-out F1={m_a['models']['RandomForest_chronological_holdout']['f1']:.3f}, "
                  f"PR-AUC={m_a['models']['RandomForest_chronological_holdout']['pr_auc']:.3f}; "
                  f"Sniffing hold-out F1={m_s['models']['RandomForest_chronological_holdout']['f1']:.3f}, "
                  f"PR-AUC={m_s['models']['RandomForest_chronological_holdout']['pr_auc']:.3f}"),
    },
    {
        "claim": "Top features are biologically interpretable: inter-animal Tail_base/Center distances dominate Sniffing; body-orientation, bbox geometry and within-animal distances dominate Attack.",
        "evidence_artifact": "outputs/feature_importance_*.csv, outputs/perm_importance_*.csv, outputs/feature_group_importance.csv",
        "value": "see the figure 05/09/10 panels and feature_group_importance.csv",
    },
    {
        "claim": "The reproduced classifier outputs are auditable: probabilities, thresholds, fold-level metrics and feature-importance rankings are all exported as plain CSV/JSON.",
        "evidence_artifact": "outputs/predictions_*.csv, outputs/cv_fold_metrics_*.csv, outputs/metrics_*.json, outputs/feature_importance_*.csv",
        "value": "8 main figures + 11 JSON/CSV evidence files",
    },
]
pd.DataFrame(claims).to_csv(OUT / "claim_recovery_table.csv", index=False)
print("Saved outputs/claim_recovery_table.csv")
print(json.dumps(claims, indent=2)[:2000])
