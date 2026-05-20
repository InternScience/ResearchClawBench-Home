#!/usr/bin/env python3
"""
SimBA-style supervised classifier reproduction on open sample data.
- Loads Together_1_features_extracted.csv (X) and Together_1_targets_inserted.csv (y)
- Trains RandomForest for Attack and Sniffing separately
- Produces: classification reports, PR curves, confusion matrices, feature importances
- All figures saved as PNG under report/images/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ----------------------------- Paths -----------------------------
DATA_FEAT = "data/Together_1_features_extracted.csv"
DATA_TARG = "data/Together_1_targets_inserted.csv"
OUT_DIR = "outputs"
FIG_DIR = "report/images"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------- Load & clean -----------------------------
feat = pd.read_csv(DATA_FEAT)
targ = pd.read_csv(DATA_TARG)

# Drop index column if present
if "Unnamed: 0" in feat.columns:
    feat = feat.drop(columns=["Unnamed: 0"])
if "Unnamed: 0" in targ.columns:
    targ = targ.drop(columns=["Unnamed: 0"])

# Keep only engineered features (drop raw pose coords)
feature_cols = [c for c in feat.columns if c.startswith("Feature_")]
X = feat[feature_cols].copy()

# Targets
y_attack = targ["Attack"].astype(int)
y_sniff = targ["Sniffing"].astype(int)

print(f"Feature matrix X: {X.shape}")
print(f"Attack positive rate: {y_attack.mean():.3f}")
print(f"Sniffing positive rate: {y_sniff.mean():.3f}")

# ----------------------------- Train/test split (stratified) -----------------------------
X_train, X_test, y_att_train, y_att_test = train_test_split(
    X, y_attack, test_size=0.25, random_state=42, stratify=y_attack
)
_, _, y_sniff_train, y_sniff_test = train_test_split(
    X, y_sniff, test_size=0.25, random_state=42, stratify=y_sniff
)

# ----------------------------- Model pipeline -----------------------------
def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

# ----------------------------- Attack classifier -----------------------------
print("\n=== Training Attack classifier ===")
model_att = build_model()
model_att.fit(X_train, y_att_train)

y_att_pred = model_att.predict(X_test)
y_att_proba = model_att.predict_proba(X_test)[:, 1]

print(classification_report(y_att_test, y_att_pred, target_names=["no_attack", "attack"]))
print("F1 (attack):", f1_score(y_att_test, y_att_pred))

# PR curve
prec, rec, _ = precision_recall_curve(y_att_test, y_att_proba)
ap = average_precision_score(y_att_test, y_att_proba)

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Attack - Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/attack_pr_curve.png", dpi=150)
plt.close()

# Confusion matrix
cm_att = confusion_matrix(y_att_test, y_att_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_att, annot=True, fmt="d", cmap="Blues",
            xticklabels=["no_attack", "attack"],
            yticklabels=["no_attack", "attack"])
plt.title("Attack - Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/attack_confusion.png", dpi=150)
plt.close()

# Feature importance
importances_att = model_att.named_steps["rf"].feature_importances_
feat_imp_att = pd.DataFrame({"feature": X.columns, "importance": importances_att})
feat_imp_att = feat_imp_att.sort_values("importance", ascending=False)
feat_imp_att.to_csv(f"{OUT_DIR}/attack_feature_importance.csv", index=False)

plt.figure(figsize=(6, 4))
sns.barplot(data=feat_imp_att.head(10), x="importance", y="feature", color="#2E86AB")
plt.title("Attack - Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/attack_feature_importance.png", dpi=150)
plt.close()

# ----------------------------- Sniffing classifier -----------------------------
print("\n=== Training Sniffing classifier ===")
model_sniff = build_model()
model_sniff.fit(X_train, y_sniff_train)

y_sniff_pred = model_sniff.predict(X_test)
y_sniff_proba = model_sniff.predict_proba(X_test)[:, 1]

print(classification_report(y_sniff_test, y_sniff_pred, target_names=["no_sniff", "sniff"]))
print("F1 (sniff):", f1_score(y_sniff_test, y_sniff_pred))

# PR curve
prec_s, rec_s, _ = precision_recall_curve(y_sniff_test, y_sniff_proba)
ap_s = average_precision_score(y_sniff_test, y_sniff_proba)

plt.figure(figsize=(6, 5))
plt.plot(rec_s, prec_s, label=f"AP = {ap_s:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Sniffing - Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sniffing_pr_curve.png", dpi=150)
plt.close()

# Confusion matrix
cm_sniff = confusion_matrix(y_sniff_test, y_sniff_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_sniff, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["no_sniff", "sniff"],
            yticklabels=["no_sniff", "sniff"])
plt.title("Sniffing - Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sniffing_confusion.png", dpi=150)
plt.close()

# Feature importance
importances_sniff = model_sniff.named_steps["rf"].feature_importances_
feat_imp_sniff = pd.DataFrame({"feature": X.columns, "importance": importances_sniff})
feat_imp_sniff = feat_imp_sniff.sort_values("importance", ascending=False)
feat_imp_sniff.to_csv(f"{OUT_DIR}/sniffing_feature_importance.csv", index=False)

plt.figure(figsize=(6, 4))
sns.barplot(data=feat_imp_sniff.head(10), x="importance", y="feature", color="#E8871E")
plt.title("Sniffing - Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sniffing_feature_importance.png", dpi=150)
plt.close()

# ----------------------------- Summary table -----------------------------
summary = pd.DataFrame({
    "behavior": ["Attack", "Sniffing"],
    "positive_rate": [y_attack.mean(), y_sniff.mean()],
    "test_f1": [f1_score(y_att_test, y_att_pred), f1_score(y_sniff_test, y_sniff_pred)],
    "test_ap": [ap, ap_s],
    "test_support_pos": [y_att_test.sum(), y_sniff_test.sum()]
})
summary.to_csv(f"{OUT_DIR}/summary_metrics.csv", index=False)
print("\n=== Summary ===")
print(summary.to_string(index=False))

print("\nAll artifacts saved to outputs/ and report/images/")
