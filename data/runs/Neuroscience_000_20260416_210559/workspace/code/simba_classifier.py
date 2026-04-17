#!/usr/bin/env python3
"""
SimBA-Style Behavior Classification Pipeline

This script implements a supervised behavior classification pipeline using
pose-derived features from the SimBA sample project to classify Attack and
Sniffing behaviors in mice.

Author: Research Pipeline
Date: 2026-04-16
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
WORKSPACE_ROOT = Path("/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_000_20260416_210559")
DATA_DIR = WORKSPACE_ROOT / "data"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
REPORT_DIR = WORKSPACE_ROOT / "report"
IMAGES_DIR = REPORT_DIR / "images"

# Create output directories
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load feature and target data from CSV files."""
    print("Loading data...")
    
    # Load features
    features_path = DATA_DIR / "Together_1_features_extracted.csv"
    features_df = pd.read_csv(features_path, index_col=0)
    
    # Load targets
    targets_path = DATA_DIR / "Together_1_targets_inserted.csv"
    targets_df = pd.read_csv(targets_path, index_col=0)
    
    print(f"Features shape: {features_df.shape}")
    print(f"Targets shape: {targets_df.shape}")
    
    return features_df, targets_df


def explore_data(features_df, targets_df):
    """Explore and summarize the data."""
    print("\n=== Data Exploration ===")
    
    # Feature columns
    feature_cols = features_df.columns.tolist()
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"Feature columns (first 10): {feature_cols[:10]}")
    print(f"Feature columns (last 10): {feature_cols[-10:]}")
    
    # Check for missing values
    missing_features = features_df.isnull().sum().sum()
    missing_targets = targets_df.isnull().sum().sum()
    print(f"\nMissing values in features: {missing_features}")
    print(f"Missing values in targets: {missing_targets}")
    
    # Target columns (Attack and Sniffing)
    target_cols = ["Attack", "Sniffing"]
    for col in target_cols:
        if col in targets_df.columns:
            n_positive = (targets_df[col] == 1).sum()
            n_negative = (targets_df[col] == 0).sum()
            total = len(targets_df)
            print(f"\n{col}:")
            print(f"  Positive frames: {n_positive} ({100*n_positive/total:.2f}%)")
            print(f"  Negative frames: {n_negative} ({100*n_negative/total:.2f}%)")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    print(features_df.describe().round(2))
    
    return {
        "n_samples": len(features_df),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "missing_features": int(missing_features),
        "missing_targets": int(missing_targets),
    }


def prepare_data(features_df, targets_df):
    """Prepare data for training."""
    print("\n=== Data Preparation ===")
    
    # Extract target columns
    target_cols = ["Attack", "Sniffing"]
    
    # Ensure we have the right columns
    X = features_df.values
    y_attack = targets_df["Attack"].values if "Attack" in targets_df.columns else None
    y_sniffing = targets_df["Sniffing"].values if "Sniffing" in targets_df.columns else None
    
    # Check class balance
    if y_attack is not None:
        print(f"Attack class distribution: {np.bincount(y_attack.astype(int))}")
    if y_sniffing is not None:
        print(f"Sniffing class distribution: {np.bincount(y_sniffing.astype(int))}")
    
    # Split data (stratified by each target)
    # Use Attack for stratification since it's the primary behavior
    stratify_col = y_attack if y_attack is not None else None
    
    X_train, X_test, y_attack_train, y_attack_test, y_sniffing_train, y_sniffing_test = train_test_split(
        X, 
        y_attack, 
        y_sniffing,
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=stratify_col
    )
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_attack_train": y_attack_train,
        "y_attack_test": y_attack_test,
        "y_sniffing_train": y_sniffing_train,
        "y_sniffing_test": y_sniffing_test,
        "scaler": scaler,
        "feature_names": features_df.columns.tolist(),
    }


def train_classifiers(data):
    """Train Random Forest classifiers for each behavior."""
    print("\n=== Model Training ===")
    
    models = {}
    cv_scores = {}
    
    # Train Attack classifier
    print("\nTraining Attack classifier...")
    rf_attack = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"  # Handle class imbalance
    )
    rf_attack.fit(data["X_train_scaled"], data["y_attack_train"])
    models["Attack"] = rf_attack
    
    # Cross-validation for Attack
    cv_scores_attack = cross_val_score(
        rf_attack, 
        data["X_train_scaled"], 
        data["y_attack_train"], 
        cv=5, 
        scoring="average_precision"
    )
    cv_scores["Attack"] = cv_scores_attack
    print(f"Attack CV AP scores: {cv_scores_attack}")
    print(f"Attack CV AP mean: {cv_scores_attack.mean():.4f} (+/- {cv_scores_attack.std()*2:.4f})")
    
    # Train Sniffing classifier
    print("\nTraining Sniffing classifier...")
    rf_sniffing = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf_sniffing.fit(data["X_train_scaled"], data["y_sniffing_train"])
    models["Sniffing"] = rf_sniffing
    
    # Cross-validation for Sniffing
    cv_scores_sniffing = cross_val_score(
        rf_sniffing, 
        data["X_train_scaled"], 
        data["y_sniffing_train"], 
        cv=5, 
        scoring="average_precision"
    )
    cv_scores["Sniffing"] = cv_scores_sniffing
    print(f"Sniffing CV AP scores: {cv_scores_sniffing}")
    print(f"Sniffing CV AP mean: {cv_scores_sniffing.mean():.4f} (+/- {cv_scores_sniffing.std()*2:.4f})")
    
    return models, cv_scores


def evaluate_models(models, data):
    """Evaluate trained models on test set."""
    print("\n=== Model Evaluation ===")
    
    results = {}
    
    for behavior, model in models.items():
        print(f"\n--- {behavior} Classifier ---")
        
        y_true = data[f"y_{behavior.lower()}_test"]
        y_pred = model.predict(data["X_test_scaled"])
        y_prob = model.predict_proba(data["X_test_scaled"])[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Precision-Recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # AUC-ROC (if both classes present)
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_prob)
        else:
            auc_roc = np.nan
        
        results[behavior] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "avg_precision": float(avg_precision),
            "auc_roc": float(auc_roc) if not np.isnan(auc_roc) else None,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "confusion_matrix": cm,
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        if not np.isnan(auc_roc):
            print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
    
    return results


def extract_feature_importance(models, feature_names):
    """Extract and analyze feature importance from trained models."""
    print("\n=== Feature Importance ===")
    
    importance_data = {}
    
    for behavior, model in models.items():
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importance_data[behavior] = {
            "feature_names": [feature_names[i] for i in indices],
            "importances": importances[indices].tolist(),
            "sorted_indices": indices.tolist(),
        }
        
        print(f"\nTop 10 features for {behavior}:")
        for i in range(min(10, len(indices))):
            feat_idx = indices[i]
            print(f"  {i+1}. {feature_names[feat_idx]}: {importances[feat_idx]:.4f}")
    
    return importance_data


def save_results(data_summary, data, models, cv_scores, results, importance_data):
    """Save all results to output files."""
    print("\n=== Saving Results ===")
    
    # Save data summary
    summary_path = OUTPUTS_DIR / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(data_summary, f, indent=2)
    print(f"Saved: {summary_path}")
    
    # Save train/test split info
    split_info = {
        "train_size": len(data["X_train"]),
        "test_size": len(data["X_test"]),
        "attack_train_positive": int(sum(data["y_attack_train"])),
        "attack_train_negative": int(sum(1 - data["y_attack_train"])),
        "attack_test_positive": int(sum(data["y_attack_test"])),
        "attack_test_negative": int(sum(1 - data["y_attack_test"])),
        "sniffing_train_positive": int(sum(data["y_sniffing_train"])),
        "sniffing_train_negative": int(sum(1 - data["y_sniffing_train"])),
        "sniffing_test_positive": int(sum(data["y_sniffing_test"])),
        "sniffing_test_negative": int(sum(1 - data["y_sniffing_test"])),
    }
    split_path = OUTPUTS_DIR / "train_test_split.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved: {split_path}")
    
    # Save CV scores
    cv_results = {
        "Attack": {
            "scores": cv_scores["Attack"].tolist(),
            "mean": float(cv_scores["Attack"].mean()),
            "std": float(cv_scores["Attack"].std()),
        },
        "Sniffing": {
            "scores": cv_scores["Sniffing"].tolist(),
            "mean": float(cv_scores["Sniffing"].mean()),
            "std": float(cv_scores["Sniffing"].std()),
        },
    }
    cv_path = OUTPUTS_DIR / "cross_validation_scores.json"
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"Saved: {cv_path}")
    
    # Save evaluation results
    eval_results = {}
    for behavior, res in results.items():
        eval_results[behavior] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "avg_precision": res["avg_precision"],
            "auc_roc": res["auc_roc"],
            "confusion_matrix": res["confusion_matrix"].tolist(),
        }
    eval_path = OUTPUTS_DIR / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved: {eval_path}")
    
    # Save feature importance
    importance_path = OUTPUTS_DIR / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(importance_data, f, indent=2)
    print(f"Saved: {importance_path}")
    
    # Save feature importance as CSV for easier reading
    for behavior in importance_data:
        df_imp = pd.DataFrame({
            "feature": importance_data[behavior]["feature_names"],
            "importance": importance_data[behavior]["importances"],
        })
        csv_path = OUTPUTS_DIR / f"feature_importance_{behavior.lower()}.csv"
        df_imp.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


def create_figures(data, results, importance_data, data_summary):
    """Create all figures for the report."""
    print("\n=== Creating Figures ===")
    
    figure_paths = {}
    
    # Figure 1: Data Overview - Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    behaviors = ["Attack", "Sniffing"]
    colors = ["#d62728", "#1f77b4"]
    
    for idx, behavior in enumerate(behaviors):
        y_train = data[f"y_{behavior.lower()}_train"]
        y_test = data[f"y_{behavior.lower()}_test"]
        
        train_counts = [sum(y_train == 0), sum(y_train == 1)]
        test_counts = [sum(y_test == 0), sum(y_test == 1)]
        
        ax = axes[idx]
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_counts, width, label="Train", color=colors[idx], alpha=0.8)
        bars2 = ax.bar(x + width/2, test_counts, width, label="Test", color=colors[idx], alpha=0.5)
        
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Frames")
        ax.set_title(f"{behavior} - Class Distribution")
        ax.set_xticks(x)
        ax.set_xticklabels(["Negative (0)", "Positive (1)"])
        ax.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig1_path = IMAGES_DIR / "figure1_class_distribution.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close()
    figure_paths["figure1"] = str(fig1_path)
    print(f"Saved: {fig1_path}")
    
    # Figure 2: Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        res = results[behavior]
        
        ax.plot(res["recall_curve"], res["precision_curve"], linewidth=2, color=colors[idx])
        ax.axhline(y=sum(res["y_true"]) / len(res["y_true"]), linestyle="--", 
                   label=f"No-skill (baseline: {sum(res['y_true'])/len(res['y_true']):.3f})", 
                   color="gray", alpha=0.5)
        
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{behavior} - Precision-Recall Curve\nAP = {res['avg_precision']:.3f}")
        ax.legend(loc="lower left")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = IMAGES_DIR / "figure2_pr_curves.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    figure_paths["figure2"] = str(fig2_path)
    print(f"Saved: {fig2_path}")
    
    # Figure 3: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        cm = results[behavior]["confusion_matrix"]
        
        # Normalize for percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        im = ax.imshow(cm_norm, cmap="Blues", aspect='auto', vmin=0, vmax=100)
        
        # Add text annotations
        thresh = 50
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1f}%)',
                       ha="center", va="center",
                       color="white" if cm_norm[i, j] > thresh else "darkblue",
                       fontsize=11)
        
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"{behavior} - Confusion Matrix")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        
        plt.colorbar(im, ax=ax, label="% of True Class")
    
    plt.tight_layout()
    fig3_path = IMAGES_DIR / "figure3_confusion_matrices.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close()
    figure_paths["figure3"] = str(fig3_path)
    print(f"Saved: {fig3_path}")
    
    # Figure 4: Feature Importance (Top 15 for each behavior)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        imp_data = importance_data[behavior]
        
        # Top 15 features
        top_n = 15
        features = imp_data["feature_names"][:top_n]
        importances = imp_data["importances"][:top_n]
        
        # Reverse for horizontal bar plot (highest at top)
        features = features[::-1]
        importances = importances[::-1]
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color=colors[idx], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{behavior} - Top {top_n} Features")
        ax.set_xlim([0, max(importances) * 1.1])
        
        # Add value labels
        for i, v in enumerate(importances):
            ax.text(v + max(importances)*0.01, i, f'{v:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    fig4_path = IMAGES_DIR / "figure4_feature_importance.png"
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close()
    figure_paths["figure4"] = str(fig4_path)
    print(f"Saved: {fig4_path}")
    
    # Figure 5: Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ["Accuracy", "Precision", "Recall", "F1", "Avg Precision"]
    x = np.arange(len(metrics))
    width = 0.35
    
    attack_metrics = [
        results["Attack"]["accuracy"],
        results["Attack"]["precision"],
        results["Attack"]["recall"],
        results["Attack"]["f1"],
        results["Attack"]["avg_precision"],
    ]
    sniffing_metrics = [
        results["Sniffing"]["accuracy"],
        results["Sniffing"]["precision"],
        results["Sniffing"]["recall"],
        results["Sniffing"]["f1"],
        results["Sniffing"]["avg_precision"],
    ]
    
    bars1 = ax.bar(x - width/2, attack_metrics, width, label="Attack", color=colors[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, sniffing_metrics, width, label="Sniffing", color=colors[1], alpha=0.8)
    
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig5_path = IMAGES_DIR / "figure5_metrics_comparison.png"
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    plt.close()
    figure_paths["figure5"] = str(fig5_path)
    print(f"Saved: {fig5_path}")
    
    # Figure 6: Feature Category Analysis (if applicable)
    # Group features by type based on naming convention
    feature_categories = {}
    for feat in data["feature_names"]:
        if "Mouse_1" in feat or "_1_" in feat.split("_")[0]:
            cat = "Mouse 1 Pose"
        elif "Mouse_2" in feat or "_2_" in feat.split("_")[0]:
            cat = "Mouse 2 Pose"
        elif "Movement" in feat:
            cat = "Movement Features"
        elif "Distance" in feat:
            cat = "Distance Features"
        elif "poly" in feat.lower() or "hull" in feat.lower():
            cat = "Geometric Features"
        else:
            cat = "Other Features"
        
        if cat not in feature_categories:
            feature_categories[cat] = []
        feature_categories[cat].append(feat)
    
    # Calculate average importance per category
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        imp_data = importance_data[behavior]
        
        category_importance = {}
        for cat, feats in feature_categories.items():
            cat_imps = []
            for feat in feats:
                if feat in imp_data["feature_names"]:
                    feat_idx = imp_data["feature_names"].index(feat)
                    cat_imps.append(imp_data["importances"][feat_idx])
            if cat_imps:
                category_importance[cat] = np.mean(cat_imps)
        
        # Sort by importance
        sorted_cats = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        cats = [c[0] for c in sorted_cats]
        imps = [c[1] for c in sorted_cats]
        
        bars = ax.bar(range(len(cats)), imps, color=colors[idx], alpha=0.8)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Average Feature Importance")
        ax.set_title(f"{behavior} - Feature Category Importance")
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, imps):
            height = bar.get_height()
            ax.annotate(f'{imp:.4f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig6_path = IMAGES_DIR / "figure6_category_importance.png"
    plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
    plt.close()
    figure_paths["figure6"] = str(fig6_path)
    print(f"Saved: {fig6_path}")
    
    return figure_paths


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("SimBA-Style Behavior Classification Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    features_df, targets_df = load_data()
    
    # Step 2: Explore data
    data_summary = explore_data(features_df, targets_df)
    
    # Step 3: Prepare data
    data = prepare_data(features_df, targets_df)
    
    # Step 4: Train classifiers
    models, cv_scores = train_classifiers(data)
    
    # Step 5: Evaluate models
    results = evaluate_models(models, data)
    
    # Step 6: Extract feature importance
    importance_data = extract_feature_importance(models, data["feature_names"])
    
    # Step 7: Save results
    save_results(data_summary, data, models, cv_scores, results, importance_data)
    
    # Step 8: Create figures
    figure_paths = create_figures(data, results, importance_data, data_summary)
    
    # Save figure paths
    figure_paths_path = OUTPUTS_DIR / "figure_paths.json"
    with open(figure_paths_path, "w") as f:
        json.dump(figure_paths, f, indent=2)
    print(f"\nSaved: {figure_paths_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return results, importance_data, figure_paths


if __name__ == "__main__":
    results, importance_data, figure_paths = main()
