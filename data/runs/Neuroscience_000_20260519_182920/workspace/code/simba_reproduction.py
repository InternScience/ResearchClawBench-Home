#!/usr/bin/env python3
"""
SimBA-style behavior classification reproduction script.
Loads raw pose data, engineers SimBA-like features, trains supervised classifiers,
evaluates them, and generates figures/tables for the report.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve, roc_curve)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
IMAGE_DIR = 'report/images'
for d in [OUTPUT_DIR, IMAGE_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURES_CSV = os.path.join(DATA_DIR, 'Together_1_features_extracted.csv')
TARGETS_CSV = os.path.join(DATA_DIR, 'Together_1_targets_inserted.csv')
REF_CSV = os.path.join(DATA_DIR, 'Together_1_machine_results_reference.csv')

# ------------------------------------------------------------------
# Load raw data
# ------------------------------------------------------------------
def load_data():
    feat = pd.read_csv(FEATURES_CSV)
    targ = pd.read_csv(TARGETS_CSV)
    ref = pd.read_csv(REF_CSV)
    if 'Unnamed: 0' in feat.columns:
        feat = feat.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in targ.columns:
        targ = targ.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in ref.columns:
        ref = ref.drop(columns=['Unnamed: 0'])
    y_attack = targ['Attack'].values
    y_sniff = targ['Sniffing'].values
    return feat, y_attack, y_sniff, ref

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def euclidean(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def angle_between(v1, v2):
    dot = np.sum(v1 * v2, axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cosang = np.clip(dot / (norm + 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def convex_hull_area(coords):
    try:
        hull = ConvexHull(coords)
        return hull.volume
    except Exception:
        return 0.0

def convex_hull_perimeter(coords):
    try:
        hull = ConvexHull(coords)
        return hull.area
    except Exception:
        return 0.0

def pairwise_distances(coords):
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    iu = np.triu_indices(dists.shape[0], k=1)
    return dists[iu]

# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------
def engineer_features(df):
    n = len(df)
    features = pd.DataFrame(index=df.index)
    parts = ['Nose', 'Ear_left', 'Ear_right', 'Center', 'Lat_left', 'Lat_right', 'Tail_base', 'Tail_end']

    # distances per mouse
    for m in [1, 2]:
        features[f'Mouse_{m}_nose_to_tail'] = euclidean(df[f'Nose_{m}_x'], df[f'Nose_{m}_y'],
                                                        df[f'Tail_base_{m}_x'], df[f'Tail_base_{m}_y'])
        features[f'Mouse_{m}_width'] = euclidean(df[f'Lat_left_{m}_x'], df[f'Lat_left_{m}_y'],
                                                  df[f'Lat_right_{m}_x'], df[f'Lat_right_{m}_y'])
        features[f'Mouse_{m}_Ear_distance'] = euclidean(df[f'Ear_left_{m}_x'], df[f'Ear_left_{m}_y'],
                                                        df[f'Ear_right_{m}_x'], df[f'Ear_right_{m}_y'])
        features[f'Mouse_{m}_Nose_to_centroid'] = euclidean(df[f'Nose_{m}_x'], df[f'Nose_{m}_y'],
                                                            df[f'Center_{m}_x'], df[f'Center_{m}_y'])
        features[f'Mouse_{m}_Nose_to_lat_left'] = euclidean(df[f'Nose_{m}_x'], df[f'Nose_{m}_y'],
                                                            df[f'Lat_left_{m}_x'], df[f'Lat_left_{m}_y'])
        features[f'Mouse_{m}_Nose_to_lat_right'] = euclidean(df[f'Nose_{m}_x'], df[f'Nose_{m}_y'],
                                                             df[f'Lat_right_{m}_x'], df[f'Lat_right_{m}_y'])
        features[f'Mouse_{m}_Centroid_to_lat_left'] = euclidean(df[f'Center_{m}_x'], df[f'Center_{m}_y'],
                                                                df[f'Lat_left_{m}_x'], df[f'Lat_left_{m}_y'])
        features[f'Mouse_{m}_Centroid_to_lat_right'] = euclidean(df[f'Center_{m}_x'], df[f'Center_{m}_y'],
                                                                 df[f'Lat_right_{m}_x'], df[f'Lat_right_{m}_y'])

    # cross-mouse distances
    features['Centroid_distance'] = euclidean(df['Center_1_x'], df['Center_1_y'],
                                              df['Center_2_x'], df['Center_2_y'])
    features['Nose_to_nose_distance'] = euclidean(df['Nose_1_x'], df['Nose_1_y'],
                                                  df['Nose_2_x'], df['Nose_2_y'])
    features['M1_Nose_to_M2_lat_left'] = euclidean(df['Nose_1_x'], df['Nose_1_y'],
                                                   df['Lat_left_2_x'], df['Lat_left_2_y'])
    features['M1_Nose_to_M2_lat_right'] = euclidean(df['Nose_1_x'], df['Nose_1_y'],
                                                    df['Lat_right_2_x'], df['Lat_right_2_y'])
    features['M2_Nose_to_M1_lat_left'] = euclidean(df['Nose_2_x'], df['Nose_2_y'],
                                                   df['Lat_left_1_x'], df['Lat_left_1_y'])
    features['M2_Nose_to_M1_lat_right'] = euclidean(df['Nose_2_x'], df['Nose_2_y'],
                                                    df['Lat_right_1_x'], df['Lat_right_1_y'])
    features['M1_Nose_to_M2_tail_base'] = euclidean(df['Nose_1_x'], df['Nose_1_y'],
                                                    df['Tail_base_2_x'], df['Tail_base_2_y'])
    features['M2_Nose_to_M1_tail_base'] = euclidean(df['Nose_2_x'], df['Nose_2_y'],
                                                    df['Tail_base_1_x'], df['Tail_base_1_y'])

    # movements
    for m in [1, 2]:
        for part in parts:
            dx = df[f'{part}_{m}_x'].diff()
            dy = df[f'{part}_{m}_y'].diff()
            features[f'Movement_{part}_{m}'] = np.sqrt(dx**2 + dy**2)
        features[f'Movement_mouse_{m}_centroid'] = features[f'Movement_Center_{m}']
        features[f'Movement_mouse_{m}_nose'] = features[f'Movement_Nose_{m}']
        features[f'Movement_mouse_{m}_tail_base'] = features[f'Movement_Tail_base_{m}']
        features[f'Movement_mouse_{m}_tail_end'] = features[f'Movement_Tail_end_{m}']
        features[f'Movement_mouse_{m}_left_ear'] = features[f'Movement_Ear_left_{m}']
        features[f'Movement_mouse_{m}_right_ear'] = features[f'Movement_Ear_right_{m}']
        features[f'Movement_mouse_{m}_lateral_left'] = features[f'Movement_Lat_left_{m}']
        features[f'Movement_mouse_{m}_lateral_right'] = features[f'Movement_Lat_right_{m}']

    m1_move_parts = [f'Movement_{p}_1' for p in parts]
    m2_move_parts = [f'Movement_{p}_2' for p in parts]
    features['Total_movement_all_bodyparts_M1'] = features[m1_move_parts].sum(axis=1)
    features['Total_movement_all_bodyparts_M2'] = features[m2_move_parts].sum(axis=1)
    features['Total_movement_all_bodyparts_both_mice'] = (
        features['Total_movement_all_bodyparts_M1'] + features['Total_movement_all_bodyparts_M2']
    )
    features['Total_movement_centroids'] = features['Movement_mouse_1_centroid'] + features['Movement_mouse_2_centroid']
    features['Total_movement_tail_ends'] = features['Movement_mouse_1_tail_end'] + features['Movement_mouse_2_tail_end']

    # polygon area / perimeter
    for m in [1, 2]:
        areas = []
        perims = []
        for i in range(n):
            coords = np.array([[df.loc[i, f'{p}_{m}_x'], df.loc[i, f'{p}_{m}_y']] for p in parts])
            areas.append(convex_hull_area(coords))
            perims.append(convex_hull_perimeter(coords))
        features[f'Mouse_{m}_poly_area'] = areas
        features[f'Mouse_{m}_poly_perim'] = perims
        features[f'Mouse_{m}_polygon_size_change'] = pd.Series(areas).diff().abs().values

    # hull pairwise stats
    for m in [1, 2]:
        largest = []
        smallest = []
        mean_d = []
        sum_d = []
        for i in range(n):
            coords = np.array([[df.loc[i, f'{p}_{m}_x'], df.loc[i, f'{p}_{m}_y']] for p in parts])
            pdists = pairwise_distances(coords)
            if len(pdists) == 0:
                largest.append(0); smallest.append(0); mean_d.append(0); sum_d.append(0)
            else:
                largest.append(pdists.max())
                smallest.append(pdists.min())
                mean_d.append(pdists.mean())
                sum_d.append(pdists.sum())
        features[f'M{m}_largest_euclidean_distance_hull'] = largest
        features[f'M{m}_smallest_euclidean_distance_hull'] = smallest
        features[f'M{m}_mean_euclidean_distance_hull'] = mean_d
        features[f'M{m}_sum_euclidean_distance_hull'] = sum_d

    # cross-mouse hull sum distances
    cross_sums = []
    for i in range(n):
        coords1 = np.array([[df.loc[i, f'{p}_1_x'], df.loc[i, f'{p}_1_y']] for p in parts])
        coords2 = np.array([[df.loc[i, f'{p}_2_x'], df.loc[i, f'{p}_2_y']] for p in parts])
        diffs = coords1[:, None, :] - coords2[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        cross_sums.append(dists.sum())
    features['Sum_euclidean_distance_hull_M1_M2'] = cross_sums

    # angles
    for m in [1, 2]:
        v_tail = np.stack([df[f'Tail_base_{m}_x'] - df[f'Center_{m}_x'],
                           df[f'Tail_base_{m}_y'] - df[f'Center_{m}_y']], axis=1)
        v_nose = np.stack([df[f'Nose_{m}_x'] - df[f'Center_{m}_x'],
                           df[f'Nose_{m}_y'] - df[f'Center_{m}_y']], axis=1)
        features[f'Mouse_{m}_angle'] = angle_between(v_tail, v_nose)
    features['Total_angle_both_mice'] = features['Mouse_1_angle'] + features['Mouse_2_angle']

    # tail end relative movement
    for m in [1, 2]:
        features[f'Tail_end_relative_to_tail_base_centroid_nose_M{m}'] = (
            features[f'Movement_mouse_{m}_tail_end']
            - (features[f'Movement_mouse_{m}_tail_base']
               + features[f'Movement_mouse_{m}_centroid']
               + features[f'Movement_mouse_{m}_nose'])
        )

    move_cols = [c for c in features.columns if 'Movement' in c or 'polygon_size_change' in c]
    features[move_cols] = features[move_cols].fillna(0)

    # rolling windows
    windows = [2, 5, 6, 8, 15]
    base_for_rolling = {
        'Centroid_distance': features['Centroid_distance'],
        'Movement_mouse_1_centroid': features['Movement_mouse_1_centroid'],
        'Movement_mouse_2_centroid': features['Movement_mouse_2_centroid'],
        'Total_movement_all_bodyparts_both_mice': features['Total_movement_all_bodyparts_both_mice'],
        'Sum_euclidean_distance_hull_M1_M2': features['Sum_euclidean_distance_hull_M1_M2'],
        'Mouse_1_width': features['Mouse_1_width'],
        'Mouse_2_width': features['Mouse_2_width'],
        'M1_mean_euclidean_distance_hull': features['M1_mean_euclidean_distance_hull'],
        'M2_mean_euclidean_distance_hull': features['M2_mean_euclidean_distance_hull'],
        'M1_smallest_euclidean_distance_hull': features['M1_smallest_euclidean_distance_hull'],
        'M2_smallest_euclidean_distance_hull': features['M2_smallest_euclidean_distance_hull'],
        'M1_largest_euclidean_distance_hull': features['M1_largest_euclidean_distance_hull'],
        'M2_largest_euclidean_distance_hull': features['M2_largest_euclidean_distance_hull'],
        'Total_movement_centroids': features['Total_movement_centroids'],
        'Total_movement_tail_ends': features['Total_movement_tail_ends'],
        'Movement_mouse_1_tail_base': features['Movement_mouse_1_tail_base'],
        'Movement_mouse_2_tail_base': features['Movement_mouse_2_tail_base'],
        'Movement_mouse_1_tail_end': features['Movement_mouse_1_tail_end'],
        'Movement_mouse_2_tail_end': features['Movement_mouse_2_tail_end'],
        'Movement_mouse_1_nose': features['Movement_mouse_1_nose'],
        'Movement_mouse_2_nose': features['Movement_mouse_2_nose'],
        'Total_angle_both_mice': features['Total_angle_both_mice'],
        'Mouse_1_poly_area': features['Mouse_1_poly_area'],
        'Mouse_2_poly_area': features['Mouse_2_poly_area'],
    }

    for name, ser in base_for_rolling.items():
        for w in windows:
            roll = ser.rolling(window=w, min_periods=1)
            features[f'{name}_median_{w}'] = roll.median()
            features[f'{name}_mean_{w}'] = roll.mean()
            features[f'{name}_sum_{w}'] = roll.sum()

    # deviation features
    for name, ser in base_for_rolling.items():
        for w in windows:
            roll_mean = ser.rolling(window=w, min_periods=1).mean()
            features[f'{name}_deviation_{w}'] = ser - roll_mean
        features[f'{name}_deviation'] = ser - ser.mean()

    # percentile ranks
    pr_features = [
        'Total_movement_centroids', 'Centroid_distance',
        'Movement_mouse_1_centroid', 'Movement_mouse_2_centroid',
        'Total_movement_all_bodyparts_both_mice', 'Sum_euclidean_distance_hull_M1_M2',
    ]
    for name in pr_features:
        if name in features.columns:
            features[f'{name}_percentile_rank'] = features[name].rank(pct=True) * 100

    prob_cols = [c for c in df.columns if c.endswith('_p')]
    if prob_cols:
        features['Sum_probabilities'] = df[prob_cols].sum(axis=1)
        features['Sum_probabilities_percentile_rank'] = features['Sum_probabilities'].rank(pct=True) * 100
        features['Sum_probabilities_deviation'] = features['Sum_probabilities'] - features['Sum_probabilities'].mean()
        features['Sum_probabilities_deviation_percentile_rank'] = features['Sum_probabilities_deviation'].rank(pct=True) * 100

    # append raw coordinates and likelihoods
    raw_cols = [c for c in df.columns if c not in ['Feature_1', 'Feature_2']]
    features = pd.concat([features, df[raw_cols]], axis=1)
    for c in ['Feature_1', 'Feature_2']:
        if c in df.columns:
            features[c] = df[c]

    features = features.dropna(axis=1, how='all')
    features = features.fillna(0)
    return features

# ------------------------------------------------------------------
# Modeling
# ------------------------------------------------------------------
def evaluate_model(clf, X_train, X_test, y_train, y_test, behavior_name, model_name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'behavior': behavior_name,
        'model': model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'pr_auc': float(average_precision_score(y_test, y_prob)),
    }
    return metrics, y_pred, y_prob, clf

def run_experiments(X, y_dict):
    """Stratified random 70/30 split."""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=12,
                                               class_weight='balanced', random_state=SEED, n_jobs=2),
        'LogisticRegression': make_pipeline(StandardScaler(),
                                            LogisticRegression(max_iter=1000, class_weight='balanced',
                                                               random_state=SEED)),
    }

    all_results = []
    predictions = {}
    fitted_models = {}
    for beh, y in y_dict.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=SEED)
        predictions[beh] = {}
        fitted_models[beh] = {}
        for mname, clf in models.items():
            metrics, y_pred, y_prob, fitted = evaluate_model(clf, X_train, X_test, y_train, y_test, beh, mname)
            all_results.append(metrics)
            predictions[beh][mname] = {'y_pred': y_pred, 'y_prob': y_prob, 'y_test': y_test, 'X_test': X_test}
            fitted_models[beh][mname] = fitted
            print(f"[{beh} | {mname}] Acc={metrics['accuracy']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}")
    return all_results, predictions, fitted_models

def run_experiments_temporal(X, y_dict):
    """Temporal 70/30 split as a robustness check."""
    n = len(X)
    split_idx = int(n * 0.7)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=12,
                                               class_weight='balanced', random_state=SEED, n_jobs=2),
    }
    all_results = []
    for beh, y in y_dict.items():
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        for mname, clf in models.items():
            metrics, _, _, _ = evaluate_model(clf, X_train, X_test, y_train, y_test, beh, mname)
            metrics['split'] = 'temporal'
            all_results.append(metrics)
            print(f"[TEMPORAL {beh} | {mname}] Acc={metrics['accuracy']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}")
    return all_results

# ------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------
def run_cv(X, y, beh_name, model, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
    scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=1)
    cv_summary = {k: float(np.mean(v)) for k, v in scores.items()}
    cv_summary['behavior'] = beh_name
    cv_summary['model'] = type(model).__name__ if hasattr(model, '__name__') else type(model).__name__
    return cv_summary

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
def plot_label_distribution(y_attack, y_sniff):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    for idx, (name, y) in enumerate([('Attack', y_attack), ('Sniffing', y_sniff)]):
        counts = pd.Series(y).value_counts().sort_index()
        counts.plot(kind='bar', ax=ax[idx], color=['steelblue', 'coral'])
        ax[idx].set_title(f'{name} label distribution')
        ax[idx].set_xlabel('Label')
        ax[idx].set_ylabel('Frame count')
        ax[idx].set_xticklabels(['0 (absent)', '1 (present)'], rotation=0)
        for p in ax[idx].patches:
            ax[idx].annotate(str(int(p.get_height())), (p.get_x() + p.get_width()/2., p.get_height()),
                             ha='center', va='bottom')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, 'label_distribution.png'), dpi=200)
    plt.close(fig)

def plot_confusion_matrices(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for idx, (beh, preds) in enumerate(predictions.items()):
        y_test = preds['RandomForest']['y_test']
        y_pred = preds['RandomForest']['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'])
        axes[idx].set_title(f'{beh} – Random Forest')
        axes[idx].set_ylabel('True label')
        axes[idx].set_xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, 'confusion_matrices.png'), dpi=200)
    plt.close(fig)

def plot_pr_curves(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for idx, (beh, preds) in enumerate(predictions.items()):
        ax = axes[idx]
        for mname in ['RandomForest', 'LogisticRegression']:
            y_test = preds[mname]['y_test']
            y_prob = preds[mname]['y_prob']
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            ax.plot(recall, precision, label=f"{mname} (AP={ap:.3f})")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall – {beh}')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, 'precision_recall_curves.png'), dpi=200)
    plt.close(fig)

def plot_roc_curves(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for idx, (beh, preds) in enumerate(predictions.items()):
        ax = axes[idx]
        for mname in ['RandomForest', 'LogisticRegression']:
            y_test = preds[mname]['y_test']
            y_prob = preds[mname]['y_prob']
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{mname} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC – {beh}')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, 'roc_curves.png'), dpi=200)
    plt.close(fig)

def plot_feature_importance(fitted_models, X, top_n=20):
    for beh, models in fitted_models.items():
        rf = models['RandomForest']
        imp = rf.feature_importances_
        feat_names = X.columns
        idx = np.argsort(imp)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(range(top_n), imp[idx][::-1], color='teal')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feat_names[i] for i in idx[::-1]], fontsize=7)
        ax.set_xlabel('Gini importance')
        ax.set_title(f'Top {top_n} features – Random Forest – {beh}')
        plt.tight_layout()
        fig.savefig(os.path.join(IMAGE_DIR, f'feature_importance_{beh}.png'), dpi=200)
        plt.close(fig)

def plot_probability_comparison(predictions, ref):
    if ref is None or ref.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for idx, beh in enumerate(['Attack', 'Sniffing']):
        ax = axes[idx]
        if beh in predictions and 'RandomForest' in predictions[beh]:
            y_prob = predictions[beh]['RandomForest']['y_prob']
            ax.hist(y_prob, bins=30, alpha=0.6, color='coral', label='Reproduced RF')
        ref_col = f'Probability_{beh}'
        if ref_col in ref.columns:
            ax.hist(ref[ref_col], bins=30, alpha=0.6, color='steelblue', label='Reference')
        ax.set_title(f'{beh} probability distribution')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Count')
        ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, 'probability_comparison.png'), dpi=200)
    plt.close(fig)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Loading data...")
    feat_raw, y_attack, y_sniff, ref = load_data()
    print(f"Raw features shape: {feat_raw.shape}")
    print("Engineering SimBA-style features...")
    X = engineer_features(feat_raw)
    print(f"Engineered features shape: {X.shape}")

    X.to_csv(os.path.join(OUTPUT_DIR, 'engineered_features.csv'), index=False)

    y_dict = {'Attack': y_attack, 'Sniffing': y_sniff}

    print("\nRunning train/test experiments (stratified random 70/30 split)...")
    all_results, predictions, fitted_models = run_experiments(X, y_dict)

    print("\nRunning temporal split robustness check...")
    temporal_results = run_experiments_temporal(X, y_dict)

    metrics_df = pd.DataFrame(all_results + temporal_results)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_metrics.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(all_results + temporal_results, f, indent=2)

    for beh, preds in predictions.items():
        pred_df = pd.DataFrame({
            'y_test': preds['RandomForest']['y_test'],
            'y_pred_rf': preds['RandomForest']['y_pred'],
            'y_prob_rf': preds['RandomForest']['y_prob'],
        })
        pred_df.to_csv(os.path.join(OUTPUT_DIR, f'predictions_{beh}.csv'), index=False)

    print("\nRunning 5-fold stratified CV (RandomForest)...")
    cv_results = []
    for beh, y in y_dict.items():
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced',
                                    random_state=SEED, n_jobs=2)
        cv_res = run_cv(X, y, beh, rf, cv=5)
        cv_results.append(cv_res)
        print(f"  {beh} CV -> Accuracy={cv_res['test_accuracy']:.3f}  F1={cv_res['test_f1']:.3f}  ROC-AUC={cv_res['test_roc_auc']:.3f}  PR-AUC={cv_res['test_average_precision']:.3f}")
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_metrics.csv'), index=False)

    print("\nComputing permutation importances (n_repeats=3)...")
    for beh, y in y_dict.items():
        rf = fitted_models[beh]['RandomForest']
        X_test_local = predictions[beh]['RandomForest']['X_test']
        y_test = predictions[beh]['RandomForest']['y_test']
        perm = permutation_importance(rf, X_test_local, y_test, n_repeats=3,
                                      random_state=SEED, scoring='roc_auc', n_jobs=1)
        perm_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm.importances_mean,
            'importance_std': perm.importances_std,
        }).sort_values('importance_mean', ascending=False)
        perm_df.to_csv(os.path.join(OUTPUT_DIR, f'permutation_importance_{beh}.csv'), index=False)

    print("\nGenerating figures...")
    plot_label_distribution(y_attack, y_sniff)
    plot_confusion_matrices(predictions)
    plot_pr_curves(predictions)
    plot_roc_curves(predictions)
    plot_feature_importance(fitted_models, X, top_n=20)
    plot_probability_comparison(predictions, ref)

    print("\nDone. Outputs saved to outputs/ and report/images/")

if __name__ == '__main__':
    main()
