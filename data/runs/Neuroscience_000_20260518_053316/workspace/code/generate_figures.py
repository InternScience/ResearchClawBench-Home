"""
Generate all figures for the SimBA reproducibility study
"""
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
features_df = pd.read_csv('outputs/engineered_features.csv')
full_preds = pd.read_csv('outputs/full_predictions.csv')
ref_compare = pd.read_csv('outputs/reference_comparison.csv')
ref_summary = json.load(open('outputs/reference_comparison_summary.json'))
eval_results = json.load(open('outputs/evaluation_results.json'))
cv_results = json.load(open('outputs/cross_validation_results.json'))
imp_attack = pd.read_csv('outputs/feature_importance_attack.csv')
imp_sniff = pd.read_csv('outputs/feature_importance_sniffing.csv')

labels_attack = full_preds['true_attack'].values
labels_sniffing = full_preds['true_sniffing'].values

os.makedirs('report/images', exist_ok=True)

# ============================================================
# FIGURE 1: Data Overview
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1a: Behavior distribution
ax = axes[0, 0]
behaviors = ['Attack\n(Positive)', 'Attack\n(Negative)', 'Sniffing\n(Positive)', 'Sniffing\n(Negative)']
counts = [labels_attack.sum(), (1-labels_attack).sum(), labels_sniffing.sum(), (1-labels_sniffing).sum()]
colors = ['#e74c3c', '#bdc3c7', '#3498db', '#bdc3c7']
bars = ax.bar(behaviors, counts, color=colors, edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15, 
            str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Frame Count', fontsize=12)
ax.set_title('Behavior Label Distribution', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(counts)*1.15)

# 1b: Class imbalance ratios
ax = axes[0, 1]
pos_ratio_attack = labels_attack.mean() * 100
pos_ratio_sniff = labels_sniffing.mean() * 100
ax.bar(['Attack', 'Sniffing'], [pos_ratio_attack, pos_ratio_sniff], 
       color=['#e74c3c', '#3498db'], edgecolor='black', linewidth=0.5)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
ax.set_ylabel('Positive Class (%)', fontsize=12)
ax.set_title('Positive Class Proportion', fontsize=13, fontweight='bold')
ax.legend()
for i, v in enumerate([pos_ratio_attack, pos_ratio_sniff]):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

# 1c: Feature count
ax = axes[1, 0]
feat_categories = {
    'Distance\n(within)': 14,
    'Distance\n(between)': 8,
    'Movement': 19,
    'Rolling\nStats': 48,
    'Deviation': 3,
    'Angle': 3,
    'Confidence': 4,
}
bars = ax.bar(feat_categories.keys(), feat_categories.values(), 
              color=sns.color_palette("Set2", len(feat_categories)),
              edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, feat_categories.values()):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylabel('Number of Features', fontsize=12)
ax.set_title(f'Engineered Feature Categories (Total: {features_df.shape[1]})', fontsize=13, fontweight='bold')

# 1d: Key feature distributions
ax = axes[1, 1]
key_features = ['centroid_distance', 'nose_to_nose_distance', 'total_mov_centroid', 'M1_angle']
for feat_name in key_features:
    vals = features_df[feat_name].values
    ax.hist(vals[labels_attack==0], bins=40, alpha=0.5, density=True, label=f'{feat_name} (non-attack)')
    ax.hist(vals[labels_attack==1], bins=40, alpha=0.5, density=True, label=f'{feat_name} (attack)')
    break  # Just show centroid distance for clarity
vals_cd = features_df['centroid_distance'].values
ax.hist(vals_cd[labels_attack==0], bins=40, alpha=0.5, density=True, label='centroid_dist (non-attack)', color='steelblue')
ax.hist(vals_cd[labels_attack==1], bins=40, alpha=0.5, density=True, label='centroid_dist (attack)', color='red')
vals_nn = features_df['nose_to_nose_distance'].values
ax.hist(vals_nn[labels_attack==0], bins=40, alpha=0.3, density=True, label='nose_dist (non-attack)', color='darkblue', histtype='step', linewidth=2)
ax.hist(vals_nn[labels_attack==1], bins=40, alpha=0.3, density=True, label='nose_dist (attack)', color='darkred', histtype='step', linewidth=2)
ax.set_xlabel('Feature Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Key Distance Features by Attack Label', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('report/images/figure_1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 1 saved: data overview")

# ============================================================
# FIGURE 2: ROC Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, beh_name in enumerate(['Attack', 'Sniffing']):
    ax = axes[idx]
    y_true = labels_attack if beh_name == 'Attack' else labels_sniffing
    
    for clf_name, color, ls in [('RF', '#2c3e50', '-'), ('GB', '#e67e22', '--')]:
        key = f'{beh_name}_{clf_name}'
        fpr = np.array(eval_results[key]['fpr'])
        tpr = np.array(eval_results[key]['tpr'])
        roc_auc_val = eval_results[key]['roc_auc']
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.5,
                label=f'{clf_name} (AUC = {roc_auc_val:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: {beh_name} Classification', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('report/images/figure_2_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 2 saved: ROC curves")

# ============================================================
# FIGURE 3: Precision-Recall Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, beh_name in enumerate(['Attack', 'Sniffing']):
    ax = axes[idx]
    y_true = labels_attack if beh_name == 'Attack' else labels_sniffing
    
    for clf_name, color, ls in [('RF', '#2c3e50', '-'), ('GB', '#e67e22', '--')]:
        beh_key = 'attack' if beh_name == 'Attack' else 'sniff'
        y_prob = full_preds[f'{beh_key}_{clf_name.lower()}_prob'].values
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec_arr, prec_arr, color=color, linestyle=ls, linewidth=2.5,
                label=f'{clf_name} (AP = {ap:.4f})')
    
    # Baseline
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='gray', linestyle=':', alpha=0.7, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve: {beh_name} Classification', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('report/images/figure_3_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 3 saved: PR curves")

# ============================================================
# FIGURE 4: Confusion Matrices
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, beh_name in enumerate(['Attack', 'Sniffing']):
    for j, clf_name in enumerate(['RF', 'GB']):
        ax = axes[i, j]
        key = f'{beh_name}_{clf_name}'
        cm = np.array(eval_results[key]['confusion_matrix'])
        
        # Normalize for display
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    linewidths=1, linecolor='black')
        
        # Add percentage annotations
        for k in range(2):
            for l in range(2):
                ax.text(l + 0.5, k + 0.7, f'({cm_pct[k,l]:.1f}%)', 
                       ha='center', va='center', fontsize=9, color='gray')
        
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'{beh_name} - {clf_name}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/figure_4_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 4 saved: confusion matrices")

# ============================================================
# FIGURE 5: Feature Importance (Top 15)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for idx, (beh_name, imp_df) in enumerate([('Attack', imp_attack), ('Sniffing', imp_sniff)]):
    ax = axes[idx]
    top = imp_df.nlargest(15, 'rf_gini')
    
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top['rf_gini'].values, color=sns.color_palette("viridis", len(top)),
            edgecolor='black', linewidth=0.5, label='Gini Importance')
    
    # Overlay permutation importance
    ax.errorbar(top['rf_permutation_mean'].values, y_pos, 
                xerr=top['rf_permutation_std'].values,
                fmt='o', color='red', markersize=6, capsize=3, 
                label='Permutation Importance')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top 15 Features: {beh_name} (RF)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/figure_5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 5 saved: feature importance")

# ============================================================
# FIGURE 6: Cross-Validation Box Plots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, beh_name in enumerate(['Attack', 'Sniffing']):
    ax = axes[idx]
    data = []
    labels_list = []
    for clf_name in ['RF', 'GB']:
        key = f'{beh_name}_{clf_name}'
        data.append(cv_results[key]['fold_scores'])
        labels_list.append(clf_name)
    
    bp = ax.boxplot(data, labels=labels_list, patch_artist=True,
                    widths=0.5, showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    colors_bp = ['#3498db', '#e67e22']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual fold points
    for i, (d, lbl) in enumerate(zip(data, labels_list)):
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax.scatter(x, d, color='black', alpha=0.6, s=30, zorder=3)
    
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'5-Fold CV: {beh_name}', fontsize=13, fontweight='bold')
    ax.set_ylim(0.5, 1.0)
    
    # Add mean text
    for i, (d, lbl) in enumerate(zip(data, labels_list)):
        key = f'{beh_name}_{lbl}'
        ax.text(i+1, cv_results[key]['mean_f1'] + 0.02, 
                f'{cv_results[key]["mean_f1"]:.3f} +/- {cv_results[key]["std_f1"]:.3f}',
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/figure_6_cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 6 saved: cross-validation")

# ============================================================
# FIGURE 7: Reference Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7a: Attack probability comparison scatter
ax = axes[0, 0]
ax.scatter(ref_compare['ref_attack_prob'], ref_compare['our_attack_prob'], 
           alpha=0.4, s=20, c=ref_compare['true_attack'].map({0: 'steelblue', 1: 'red'}),
           edgecolors='none')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('Reference P(Attack)', fontsize=12)
ax.set_ylabel('Our P(Attack)', fontsize=12)
ax.set_title(f'Attack Probability: Reference vs. Ours\n(Spearman r = {ref_summary["attack_prob_spearman"]:.3f})',
             fontsize=12, fontweight='bold')

# 7b: Sniffing probability comparison scatter
ax = axes[0, 1]
ax.scatter(ref_compare['ref_sniff_prob'], ref_compare['our_sniff_prob'],
           alpha=0.4, s=20, c=ref_compare['true_sniffing'].map({0: 'steelblue', 1: 'red'}),
           edgecolors='none')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel(' Reference P(Sniffing)', fontsize=12)
ax.set_ylabel('Our P(Sniffing)', fontsize=12)
ax.set_title(f'Sniffing Probability: Reference vs. Ours\n(Spearman r = {ref_summary["sniff_prob_spearman"]:.3f})',
             fontsize=12, fontweight='bold')

# 7c: Prediction agreement
ax = axes[1, 0]
agree_data = {
    'Attack': [ref_summary['attack_prediction_agreement']],
    'Sniffing': [ref_summary['sniff_prediction_agreement']]
}
x_pos = [0, 1]
bars = ax.bar(x_pos, [ref_summary['attack_prediction_agreement'], ref_summary['sniff_prediction_agreement']],
              color=['#e74c3c', '#3498db'], edgecolor='black', linewidth=0.5, width=0.5)
for bar, val in zip(bars, [ref_summary['attack_prediction_agreement'], ref_summary['sniff_prediction_agreement']]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.1%}', ha='center', fontweight='bold', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Attack', 'Sniffing'])
ax.set_ylabel('Prediction Agreement', fontsize=12)
ax.set_title('Binary Prediction Agreement\n(Reference vs. Our RF)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)

# 7d: Probability distributions
ax = axes[1, 1]
ax.hist(ref_compare['ref_attack_prob'], bins=20, alpha=0.5, label='Reference', color='steelblue', density=True)
ax.hist(ref_compare['our_attack_prob'], bins=20, alpha=0.5, label='Ours (RF)', color='red', density=True)
ax.set_xlabel('P(Attack)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Attack Probability Distribution\n(Reference subset, n=300)', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('report/images/figure_7_reference_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 7 saved: reference comparison")

# ============================================================
# FIGURE 8: Metrics Summary Table
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Build table data
headers = ['Model', 'Behavior', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'Avg Precision']
table_data = []
for beh_name in ['Attack', 'Sniffing']:
    for clf_name in ['RF', 'GB']:
        key = f'{beh_name}_{clf_name}'
        r = eval_results[key]
        table_data.append([
            clf_name, beh_name,
            f'{r["accuracy"]:.4f}', f'{r["precision"]:.4f}',
            f'{r["recall"]:.4f}', f'{r["f1"]:.4f}',
            f'{r["roc_auc"]:.4f}', f'{r["average_precision"]:.4f}'
        ])

table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                 cellLoc='center', colColours=['#ecf0f1']*len(headers))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color header
for j in range(len(headers)):
    table[0, j].set_facecolor('#34495e')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Color rows by behavior
for i in range(len(table_data)):
    if table_data[i][1] == 'Attack':
        for j in range(len(headers)):
            table[i+1, j].set_facecolor('#fadbd8')
    else:
        for j in range(len(headers)):
            table[i+1, j].set_facecolor('#d4e6f1')

ax.set_title('Classification Performance Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig('report/images/figure_8_metrics_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 8 saved: metrics table")

# ============================================================
# FIGURE 9: Temporal Prediction Plot
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

frames = full_preds['frame_idx'].values

for idx, beh_name in enumerate(['Attack', 'Sniffing']):
    ax = axes[idx]
    beh_key = 'attack' if beh_name == 'Attack' else 'sniff'
    y_true = full_preds[f'true_{beh_name.lower()}'].values
    y_prob_rf = full_preds[f'{beh_key}_rf_prob'].values
    y_prob_gb = full_preds[f'{beh_key}_gb_prob'].values
    
    # Ground truth
    ax.fill_between(frames, 0, y_true, alpha=0.3, color='green', label=f'True {beh_name}')
    
    # Predictions
    ax.plot(frames, y_prob_rf, color='#2c3e50', linewidth=1.2, alpha=0.8, label='RF Prob')
    ax.plot(frames, y_prob_gb, color='#e67e22', linewidth=1.2, alpha=0.8, label='GB Prob')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')
    
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'{beh_name} Classification Over Time', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc='upper right', ncol=2)

axes[1].set_xlabel('Frame Index', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/figure_9_temporal_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 9 saved: temporal predictions")

# ============================================================
# FIGURE 10: Feature Correlation Heatmap (top features)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))

# Select top 20 features from both behaviors
top_attack = imp_attack.nlargest(10, 'rf_gini')['feature'].tolist()
top_sniff = imp_sniff.nlargest(10, 'rf_gini')['feature'].tolist()
all_top = list(dict.fromkeys(top_attack + top_sniff))  # unique, preserve order

corr_data = features_df[all_top].corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool))

sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            annot_kws={'size': 7}, ax=ax,
            cbar_kws={'label': 'Pearson Correlation'})

ax.set_title('Feature Correlation Heatmap\n(Top Discriminative Features)', 
             fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('report/images/figure_10_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure 10 saved: feature correlation")

print("\nAll figures generated successfully!")
