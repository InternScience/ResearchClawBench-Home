import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import numpy as np

# Load Data
labels = pd.read_csv('data/m6a_labels.csv')
pred_uc4 = pd.read_csv('data/m6a_predictions_uncalled4.csv')
pred_np = pd.read_csv('data/m6a_predictions_nanopolish.csv')

# Merge
df1 = pd.merge(labels, pred_uc4, on='site_id')
df2 = pd.merge(labels, pred_np, on='site_id')

y_true = df1['label']
y_pred_uc4 = df1['probability']
y_pred_np = df2['probability']

# PR Curve
precision_uc4, recall_uc4, _ = precision_recall_curve(y_true, y_pred_uc4)
precision_np, recall_np, _ = precision_recall_curve(y_true, y_pred_np)

auc_pr_uc4 = average_precision_score(y_true, y_pred_uc4)
auc_pr_np = average_precision_score(y_true, y_pred_np)

plt.figure(figsize=(8, 6))
plt.plot(recall_uc4, precision_uc4, label=f'Uncalled4 (AUPRC = {auc_pr_uc4:.3f})', color='blue')
plt.plot(recall_np, precision_np, label=f'Nanopolish (AUPRC = {auc_pr_np:.3f})', color='orange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for m6A Detection')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/pr_curve.png', dpi=300)
plt.close()

# ROC Curve
fpr_uc4, tpr_uc4, _ = roc_curve(y_true, y_pred_uc4)
fpr_np, tpr_np, _ = roc_curve(y_true, y_pred_np)

auc_roc_uc4 = auc(fpr_uc4, tpr_uc4)
auc_roc_np = auc(fpr_np, tpr_np)

plt.figure(figsize=(8, 6))
plt.plot(fpr_uc4, tpr_uc4, label=f'Uncalled4 (AUROC = {auc_roc_uc4:.3f})', color='blue')
plt.plot(fpr_np, tpr_np, label=f'Nanopolish (AUROC = {auc_roc_np:.3f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for m6A Detection')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/roc_curve.png', dpi=300)
plt.close()

# Save metrics
with open('outputs/m6a_metrics.txt', 'w') as f:
    f.write(f"Uncalled4 AUPRC: {auc_pr_uc4:.4f}\n")
    f.write(f"Nanopolish AUPRC: {auc_pr_np:.4f}\n")
    f.write(f"Uncalled4 AUROC: {auc_roc_uc4:.4f}\n")
    f.write(f"Nanopolish AUROC: {auc_roc_np:.4f}\n")

print("m6A plots generated.")
