"""
SimBA-style Supervised Behavior Classification: Full Pipeline
==============================================================
Reproduces and evaluates the SimBA workflow on open pose-derived features
for Attack and Sniffing behavior classification.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
IMAGES_DIR = "report/images"
CODE_DIR = "code"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

feat_df = pd.read_csv(os.path.join(DATA_DIR, "Together_1_features_extracted.csv"), index_col=0)
tgt_df = pd.read_csv(os.path.join(DATA_DIR, "Together_1_targets_inserted.csv"), index_col=0)
ref_df = pd.read_csv(os.path.join(DATA_DIR, "Together_1_machine_results_reference.csv"), index_col=0)

print(f"Features shape: {feat_df.shape}")
print(f"Targets shape: {tgt_df.shape}")
print(f"Reference shape: {ref_df.shape}")

# ============================================================
# 2. FEATURE ENGINEERING (SimBA-style)
# ============================================================
# The raw features file has 50 columns: 48 pose coordinates (x,y,p for 12 keypoints × 2 mice)
# plus Feature_1 (frame index) and Feature_2 (reverse frame count).
# We need to engineer meaningful behavioral features from these raw pose signals.

def engineer_features(df):
    """Engineer SimBA-style features from raw pose coordinates."""
    features = pd.DataFrame(index=df.index)
    
    # Mouse 1 keypoints
    m1_nose = df[['Nose_1_x', 'Nose_1_y']].values
    m1_ear_l = df[['Ear_left_1_x', 'Ear_left_1_y']].values
    m1_ear_r = df[['Ear_right_1_x', 'Ear_right_1_y']].values
    m1_center = df[['Center_1_x', 'Center_1_y']].values
    m1_lat_l = df[['Lat_left_1_x', 'Lat_left_1_y']].values
    m1_lat_r = df[['Lat_right_1_x', 'Lat_right_1_y']].values
    m1_tail_base = df[['Tail_base_1_x', 'Tail_base_1_y']].values
    m1_tail_end = df[['Tail_end_1_x', 'Tail_end_1_y']].values
    
    # Mouse 2 keypoints
    m2_nose = df[['Nose_2_x', 'Nose_2_y']].values
    m2_ear_l = df[['Ear_left_2_x', 'Ear_left_2_y']].values
    m2_ear_r = df[['Ear_right_2_x', 'Ear_right_2_y']].values
    m2_center = df[['Center_2_x', 'Center_2_y']].values
    m2_lat_l = df[['Lat_left_2_x', 'Lat_left_2_y']].values
    m2_lat_r = df[['Lat_right_2_x', 'Lat_right_2_y']].values
    m2_tail_base = df[['Tail_base_2_x', 'Tail_base_2_y']].values
    m2_tail_end = df[['Tail_end_2_x', 'Tail_end_2_y']].values
    
    def euclidean(a, b):
        return np.sqrt(np.sum((a - b)**2, axis=1))
    
    # --- Intra-animal distances ---
    features['Mouse_1_nose_to_tail'] = euclidean(m1_nose, m1_tail_base)
    features['Mouse_2_nose_to_tail'] = euclidean(m2_nose, m2_tail_base)
    features['Mouse_1_Ear_distance'] = euclidean(m1_ear_l, m1_ear_r)
    features['Mouse_2_Ear_distance'] = euclidean(m2_ear_l, m2_ear_r)
    features['Mouse_1_Nose_to_centroid'] = euclidean(m1_nose, m1_center)
    features['Mouse_2_Nose_to_centroid'] = euclidean(m2_nose, m2_center)
    features['Mouse_1_Nose_to_lateral_left'] = euclidean(m1_nose, m1_lat_l)
    features['Mouse_2_Nose_to_lateral_left'] = euclidean(m2_nose, m2_lat_l)
    features['Mouse_1_Nose_to_lateral_right'] = euclidean(m1_nose, m1_lat_r)
    features['Mouse_2_Nose_to_lateral_right'] = euclidean(m2_nose, m2_lat_r)
    features['Mouse_1_Centroid_to_lateral_left'] = euclidean(m1_center, m1_lat_l)
    features['Mouse_2_Centroid_to_lateral_left'] = euclidean(m2_center, m2_lat_l)
    features['Mouse_1_Centroid_to_lateral_right'] = euclidean(m1_center, m1_lat_r)
    features['Mouse_2_Centroid_to_lateral_right'] = euclidean(m2_center, m2_lat_r)
    
    # --- Inter-animal distances ---
    features['Centroid_distance'] = euclidean(m1_center, m2_center)
    features['Nose_to_nose_distance'] = euclidean(m1_nose, m2_nose)
    features['M1_Nose_to_M2_lat_left'] = euclidean(m1_nose, m2_lat_l)
    features['M1_Nose_to_M2_lat_right'] = euclidean(m1_nose, m2_lat_r)
    features['M2_Nose_to_M1_lat_left'] = euclidean(m2_nose, m1_lat_l)
    features['M2_Nose_to_M1_lat_right'] = euclidean(m2_nose, m1_lat_r)
    features['M1_Nose_to_M2_tail_base'] = euclidean(m1_nose, m2_tail_base)
    features['M2_Nose_to_M1_tail_base'] = euclidean(m2_nose, m1_tail_base)
    
    # --- Movement features (frame-to-frame displacement) ---
    for name, pts in [
        ('Movement_mouse_1_centroid', m1_center),
        ('Movement_mouse_2_centroid', m2_center),
        ('Movement_mouse_1_nose', m1_nose),
        ('Movement_mouse_2_nose', m2_nose),
        ('Movement_mouse_1_tail_base', m1_tail_base),
        ('Movement_mouse_2_tail_base', m2_tail_base),
        ('Movement_mouse_1_tail_end', m1_tail_end),
        ('Movement_mouse_2_tail_end', m2_tail_end),
        ('Movement_mouse_1_left_ear', m1_ear_l),
        ('Movement_mouse_2_left_ear', m2_ear_l),
        ('Movement_mouse_1_right_ear', m1_ear_r),
        ('Movement_mouse_2_right_ear', m2_ear_r),
        ('Movement_mouse_1_lateral_left', m1_lat_l),
        ('Movement_mouse_2_lateral_left', m2_lat_l),
        ('Movement_mouse_1_lateral_right', m1_lat_r),
        ('Movement_mouse_2_lateral_right', m2_lat_r),
    ]:
        movement = np.zeros(len(pts))
        movement[1:] = euclidean(pts[1:], pts[:-1])
        movement[0] = 0
        features[name] = movement
    
    # --- Polygon area (convex hull area approximation using shoelace formula) ---
    def poly_area(points_list):
        """Compute polygon area for each frame given a list of keypoint arrays."""
        areas = np.zeros(len(points_list[0]))
        for i in range(len(points_list[0])):
            pts = np.array([p[i] for p in points_list])
            n = len(pts)
            if n < 3:
                areas[i] = 0
                continue
            # Shoelace formula
            x = pts[:, 0]
            y = pts[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            areas[i] = area
        return areas
    
    m1_keypoints = [m1_nose, m1_ear_l, m1_ear_r, m1_center, m1_lat_l, m1_lat_r, m1_tail_base]
    m2_keypoints = [m2_nose, m2_ear_l, m2_ear_r, m2_center, m2_lat_l, m2_lat_r, m2_tail_base]
    
    features['Mouse_1_poly_area'] = poly_area(m1_keypoints)
    features['Mouse_2_poly_area'] = poly_area(m2_keypoints)
    
    # --- Width (ear-to-ear or lateral distance) ---
    features['Mouse_1_width'] = euclidean(m1_lat_l, m1_lat_r)
    features['Mouse_2_width'] = euclidean(m2_lat_l, m2_lat_r)
    
    # --- Polygon size change ---
    features['Mouse_1_polygon_size_change'] = np.zeros(len(features))
    features['Mouse_1_polygon_size_change'][1:] = np.abs(
        features['Mouse_1_poly_area'].values[1:] - features['Mouse_1_poly_area'].values[:-1]
    )
    features['Mouse_2_polygon_size_change'] = np.zeros(len(features))
    features['Mouse_2_polygon_size_change'][1:] = np.abs(
        features['Mouse_2_poly_area'].values[1:] - features['Mouse_2_poly_area'].values[:-1]
    )
    
    # --- Total movement ---
    features['Total_movement_centroids'] = features['Movement_mouse_1_centroid'] + features['Movement_mouse_2_centroid']
    features['Total_movement_tail_ends'] = features['Movement_mouse_1_tail_end'] + features['Movement_mouse_2_tail_end']
    
    # --- Rolling window features (median/mean/sum over windows of 2, 5, 15 frames) ---
    key_feature_cols = [
        'Centroid_distance', 'Nose_to_nose_distance',
        'Total_movement_centroids', 'Mouse_1_poly_area', 'Mouse_2_poly_area',
        'Mouse_1_nose_to_tail', 'Mouse_2_nose_to_tail',
        'Mouse_1_width', 'Mouse_2_width',
    ]
    
    for col in key_feature_cols:
        vals = features[col].values
        for w in [2, 5, 15]:
            roll = pd.Series(vals).rolling(window=w, min_periods=1)
            features[f'{col}_median_{w}'] = roll.median().values
            features[f'{col}_mean_{w}'] = roll.mean().values
            features[f'{col}_sum_{w}'] = roll.sum().values
    
    # --- Angle features ---
    def angle_between(a, b, c):
        """Angle at point b formed by rays ba and bc."""
        ba = a - b
        bc = c - b
        cos_angle = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.arccos(cos_angle)
    
    features['Mouse_1_angle'] = angle_between(m1_nose, m1_center, m1_tail_base)
    features['Mouse_2_angle'] = angle_between(m2_nose, m2_center, m2_tail_base)
    
    # --- Probability (confidence) features ---
    m1_probs = df[['Nose_1_p', 'Ear_left_1_p', 'Ear_right_1_p', 'Center_1_p',
                    'Lat_left_1_p', 'Lat_right_1_p', 'Tail_base_1_p', 'Tail_end_1_p']].values
    m2_probs = df[['Nose_2_p', 'Ear_left_2_p', 'Ear_right_2_p', 'Center_2_p',
                    'Lat_left_2_p', 'Lat_right_2_p', 'Tail_base_2_p', 'Tail_end_2_p']].values
    features['Sum_probabilities'] = np.sum(m1_probs, axis=1) + np.sum(m2_probs, axis=1)
    
    return features

print("Engineering features...")
X_engineered = engineer_features(feat_df)
y_attack = tgt_df['Attack'].values
y_sniffing = tgt_df['Sniffing'].values

print(f"Engineered features shape: {X_engineered.shape}")
print(f"Feature columns: {list(X_engineered.columns)}")

# Save engineered features
X_engineered.to_csv(os.path.join(OUTPUT_DIR, "engineered_features.csv"))

# ============================================================
# 3. CLASS DISTRIBUTION OVERVIEW FIGURE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, label, y, name in zip(axes, ['Attack', 'Sniffing'], [y_attack, y_sniffing], ['Attack', 'Sniffing']):
    counts = pd.Series(y).value_counts().sort_index()
    bars = ax.bar(['Absent (0)', 'Present (1)'], counts.values, color=['#4ECDC4', '#FF6B6B'])
    ax.set_title(f'{name} Class Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frame Count')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "class_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved class_distribution.png")

# ============================================================
# 4. TRAIN/TEST SPLIT & MODEL TRAINING
# ============================================================
RANDOM_STATE = 42

# Use stratified split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)

models_config = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
}

# IMPORTANT: Create fresh model instances per behavior to avoid shared state
def get_models():
    return {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
    }
behaviors = {
    'Attack': y_attack,
    'Sniffing': y_sniffing,
}

results = {}
all_predictions = {}

for behavior_name, y in behaviors.items():
    print(f"\n{'='*60}")
    print(f"Training models for: {behavior_name}")
    print(f"{'='*60}")
    
    for train_idx, test_idx in splitter.split(X_engineered, y):
        X_train_raw = X_engineered.iloc[train_idx]
        X_test_raw = X_engineered.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Apply SMOTE for imbalanced classes (only on training data)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    print(f"  Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"  After SMOTE: Train size: {len(y_train_sm)}, Positive: {np.sum(y_train_sm)}, Negative: {len(y_train_sm) - np.sum(y_train_sm)}")
    
    behavior_results = {}
    
    for model_name, model in get_models().items():
        print(f"\n  Training {model_name}...")
        
        # Use SMOTE-resampled data for RF and LR; GB handles imbalance via sampling
        if model_name == 'Gradient Boosting':
            model.fit(X_train, y_train)
        else:
            model.fit(X_train_sm, y_train_sm)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"    Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"    ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"    Confusion Matrix:\n{cm}")
        
        behavior_results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model': model,
            'scaler': scaler,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'X_train_raw': X_train_raw,
            'X_test_raw': X_test_raw,
        }
    
    results[behavior_name] = behavior_results

# ============================================================
# 5. SAVE QUANTITATIVE EVALUATION RESULTS
# ============================================================
eval_summary = {}
for behavior_name, behavior_results in results.items():
    eval_summary[behavior_name] = {}
    for model_name, res in behavior_results.items():
        eval_summary[behavior_name][model_name] = {
            'accuracy': round(res['accuracy'], 4),
            'precision': round(res['precision'], 4),
            'recall': round(res['recall'], 4),
            'f1': round(res['f1'], 4),
            'roc_auc': round(res['roc_auc'], 4),
            'pr_auc': round(res['pr_auc'], 4),
            'confusion_matrix': res['confusion_matrix'],
        }

with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), 'w') as f:
    json.dump(eval_summary, f, indent=2)
print("Saved evaluation_summary.json")

# ============================================================
# 6. GENERATE FIGURES
# ============================================================

# --- Confusion Matrices ---
for behavior_name, behavior_results in results.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (model_name, res) in enumerate(behavior_results.items()):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'])
        axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    fig.suptitle(f'Confusion Matrices — {behavior_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"confusion_matrix_{behavior_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion_matrix_{behavior_name.lower()}.png")

# --- ROC Curves ---
for behavior_name, behavior_results in results.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, res in behavior_results.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={res["roc_auc"]:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {behavior_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"roc_curve_{behavior_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved roc_curve_{behavior_name.lower()}.png")

# --- Precision-Recall Curves ---
for behavior_name, behavior_results in results.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, res in behavior_results.items():
        precision_arr, recall_arr, _ = precision_recall_curve(res['y_test'], res['y_prob'])
        ax.plot(recall_arr, precision_arr, label=f'{model_name} (AP={res["pr_auc"]:.3f})', linewidth=2)
    n_pos = np.sum(behavior_results['Random Forest']['y_test'])
    n_total = len(behavior_results['Random Forest']['y_test'])
    baseline = n_pos / n_total
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve — {behavior_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"pr_curve_{behavior_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pr_curve_{behavior_name.lower()}.png")

# --- Evaluation Metrics Comparison Bar Chart ---
for behavior_name, behavior_results in results.items():
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    model_names = list(behavior_results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for j, model_name in enumerate(model_names):
        values = [behavior_results[model_name][m] for m in metrics_names]
        ax.bar(x + j*width, values, width, label=model_name, alpha=0.85)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Performance Comparison — {behavior_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"metrics_comparison_{behavior_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics_comparison_{behavior_name.lower()}.png")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
feature_names = list(X_engineered.columns)

for behavior_name, behavior_results in results.items():
    # Random Forest feature importance
    rf_model = behavior_results['Random Forest']['model']
    rf_importances = rf_model.feature_importances_
    
    # Gradient Boosting feature importance
    gb_model = behavior_results['Gradient Boosting']['model']
    gb_importances = gb_model.feature_importances_
    
    # Top 20 features for each
    top_n = 20
    
    # RF
    rf_top_idx = np.argsort(rf_importances)[-top_n:]
    rf_top_features = [feature_names[i] for i in rf_top_idx]
    rf_top_values = rf_importances[rf_top_idx]
    
    # GB
    gb_top_idx = np.argsort(gb_importances)[-top_n:]
    gb_top_features = [feature_names[i] for i in gb_top_idx]
    gb_top_values = gb_importances[gb_top_idx]
    
    # Save feature importance tables
    rf_fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importances
    }).sort_values('importance', ascending=False)
    rf_fi_df.to_csv(os.path.join(OUTPUT_DIR, f"rf_feature_importance_{behavior_name.lower()}.csv"), index=False)
    
    gb_fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': gb_importances
    }).sort_values('importance', ascending=False)
    gb_fi_df.to_csv(os.path.join(OUTPUT_DIR, f"gb_feature_importance_{behavior_name.lower()}.csv"), index=False)
    
    # Plot feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].barh(range(top_n), rf_top_values, color='#2196F3', alpha=0.85)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(rf_top_features, fontsize=9)
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Random Forest — Top 20 Features', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    
    axes[1].barh(range(top_n), gb_top_values, color='#FF9800', alpha=0.85)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(gb_top_features, fontsize=9)
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Gradient Boosting — Top 20 Features', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    
    fig.suptitle(f'Feature Importance — {behavior_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"feature_importance_{behavior_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature_importance_{behavior_name.lower()}.png")

# ============================================================
# 8. SHAP ANALYSIS (for best model per behavior)
# ============================================================
for behavior_name, behavior_results in results.items():
    # Choose RF as the primary interpretable model
    rf_model = behavior_results['Random Forest']['model']
    scaler = behavior_results['Random Forest']['scaler']
    X_test_raw = behavior_results['Random Forest']['X_test_raw']
    X_test_scaled = scaler.transform(X_test_raw)
    
    print(f"\nComputing SHAP values for {behavior_name} (Random Forest)...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # For binary classification, shap_values is a list of two arrays
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]  # SHAP values for positive class
    else:
        shap_vals_pos = shap_values
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_vals_pos, X_test_scaled, feature_names=feature_names, show=False, max_display=20)
    plt.title(f'SHAP Feature Impact — {behavior_name} (RF)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"shap_summary_{behavior_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved shap_summary_{behavior_name.lower()}.png")
    
    # Save SHAP values
    # For TreeExplainer binary classification, shap_values returns shape (n_samples, n_features, n_classes)
    # We already extracted the positive class SHAP values above
    if shap_vals_pos.ndim == 3:
        shap_vals_pos = shap_vals_pos[:, :, 1]  # positive class
    shap_df = pd.DataFrame(shap_vals_pos, columns=feature_names)
    shap_mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    shap_mean_abs.to_csv(os.path.join(OUTPUT_DIR, f"shap_importance_{behavior_name.lower()}.csv"))

# ============================================================
# 9. PERMUTATION IMPORTANCE
# ============================================================
from sklearn.inspection import permutation_importance

for behavior_name, behavior_results in results.items():
    rf_model = behavior_results['Random Forest']['model']
    scaler = behavior_results['Random Forest']['scaler']
    X_test_scaled = scaler.transform(behavior_results['Random Forest']['X_test_raw'])
    y_test = behavior_results['Random Forest']['y_test']
    
    print(f"\nComputing permutation importance for {behavior_name}...")
    perm_imp = permutation_importance(rf_model, X_test_scaled, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring='f1')
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_imp.importances_mean,
        'importance_std': perm_imp.importances_std
    }).sort_values('importance_mean', ascending=False)
    perm_df.to_csv(os.path.join(OUTPUT_DIR, f"permutation_importance_{behavior_name.lower()}.csv"), index=False)
    
    top_n = 20
    top_perm = perm_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_perm['importance_mean'].values, 
            xerr=top_perm['importance_std'].values, color='#9C27B0', alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_perm['feature'].values, fontsize=9)
    ax.set_xlabel('Permutation Importance (F1 drop)')
    ax.set_title(f'Permutation Importance — {behavior_name} (RF)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"permutation_importance_{behavior_name.lower()}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved permutation_importance_{behavior_name.lower()}.png")

# ============================================================
# 10. REFERENCE COMPARISON
# ============================================================
# The reference file has 300 frames with predictions.
# We need to align these with our data (the reference corresponds to a subset).
# The reference indices may overlap with our full dataset.

# Check overlap between reference indices and our feature indices
ref_indices = ref_df.index.tolist()
our_indices = feat_df.index.tolist()
overlap = set(ref_indices).intersection(set(our_indices))
print(f"\nReference indices overlap with feature indices: {len(overlap)} out of {len(ref_indices)}")

# If there's overlap, compare our predictions with reference predictions
if len(overlap) > 0:
    # Get reference Attack/Sniffing labels and probabilities
    ref_attack_labels = ref_df.loc[list(overlap), 'Attack'].values
    ref_sniffing_labels = ref_df.loc[list(overlap), 'Sniffing'].values
    ref_attack_probs = ref_df.loc[list(overlap), 'Probability_Attack'].values
    ref_sniffing_probs = ref_df.loc[list(overlap), 'Probability_Sniffing'].values
    
    # Our predictions on the overlapping frames
    X_overlap = X_engineered.loc[list(overlap)]
    
    comparison_results = {}
    for behavior_name, behavior_results in results.items():
        rf_model = behavior_results['Random Forest']['model']
        scaler = behavior_results['Random Forest']['scaler']
        X_overlap_scaled = scaler.transform(X_overlap)
        our_probs = rf_model.predict_proba(X_overlap_scaled)[:, 1]
        our_preds = rf_model.predict(X_overlap_scaled)
        
        ref_key = f'Probability_{behavior_name}'
        ref_labels_key = behavior_name
        
        if behavior_name == 'Attack':
            ref_probs_arr = ref_attack_probs
            ref_labels_arr = ref_attack_labels
        else:
            ref_probs_arr = ref_sniffing_probs
            ref_labels_arr = ref_sniffing_labels
        
        # Agreement metrics
        prob_corr = np.corrcoef(our_probs, ref_probs_arr)[0, 1]
        pred_agreement = np.mean(our_preds == ref_labels_arr)
        
        comparison_results[behavior_name] = {
            'probability_correlation': round(prob_corr, 4),
            'prediction_agreement': round(pred_agreement, 4),
            'n_overlap_frames': len(overlap),
        }
        
        print(f"\n  {behavior_name} Reference Comparison:")
        print(f"    Probability correlation: {prob_corr:.4f}")
        print(f"    Prediction agreement: {pred_agreement:.4f}")
        
        # Scatter plot of probability comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(ref_probs_arr, our_probs, alpha=0.5, s=30, c='#2196F3')
        ax.set_xlabel(f'Reference Probability ({behavior_name})', fontsize=12)
        ax.set_ylabel(f'Our RF Probability ({behavior_name})', fontsize=12)
        ax.set_title(f'Probability Comparison with Reference — {behavior_name}\n(r = {prob_corr:.3f})', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        lims = [0, max(max(ref_probs_arr), max(our_probs)) * 1.05]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, 'k--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f"reference_comparison_{behavior_name.lower()}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reference_comparison_{behavior_name.lower()}.png")
    
    with open(os.path.join(OUTPUT_DIR, "reference_comparison.json"), 'w') as f:
        json.dump(comparison_results, f, indent=2)
else:
    print("No overlap found between reference and feature indices.")

# ============================================================
# 11. FEATURE CORRELATION HEATMAP (top features)
# ============================================================
# Select top 15 most important features across both behaviors
all_top_features = set()
for behavior_name in ['Attack', 'Sniffing']:
    rf_fi = pd.read_csv(os.path.join(OUTPUT_DIR, f"rf_feature_importance_{behavior_name.lower()}.csv"))
    for f in rf_fi.head(10)['feature'].values:
        all_top_features.add(f)

top_feat_list = sorted(list(all_top_features))[:15]
corr_matrix = X_engineered[top_feat_list].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix (Top Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "feature_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved feature_correlation_heatmap.png")

# ============================================================
# 12. BEHAVIOR TIMELINE VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
for ax, behavior_name, y in zip(axes, ['Attack', 'Sniffing'], [y_attack, y_sniffing]):
    ax.plot(range(len(y)), y, linewidth=0.8, color='#FF6B6B' if behavior_name == 'Attack' else '#4ECDC4')
    ax.set_ylabel(behavior_name, fontsize=12)
    ax.set_title(f'{behavior_name} Annotation Timeline', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Absent', 'Present'])
axes[1].set_xlabel('Frame Index', fontsize=12)
fig.suptitle('Behavior Annotation Timeline', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "behavior_timeline.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved behavior_timeline.png")

print("\n\nAll analysis complete!")
print("Evaluation summary saved to outputs/evaluation_summary.json")