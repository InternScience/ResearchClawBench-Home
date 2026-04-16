import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load data
features_df = pd.read_csv('data/Together_1_features_extracted.csv', index_col=0)
targets_df = pd.read_csv('data/Together_1_targets_inserted.csv', index_col=0)

# Align features and targets
# The index should match, but let's ensure it.
assert (features_df.index == targets_df.index).all(), "Indexes do not match!"

# Extract features
X = features_df

# Extract targets
y_attack = targets_df['Attack']
y_sniffing = targets_df['Sniffing']

# Save basic info
with open('outputs/data_info.txt', 'w') as f:
    f.write(f"Number of samples: {len(X)}\n")
    f.write(f"Number of features: {X.shape[1]}\n")
    f.write(f"Attack positive instances: {y_attack.sum()}\n")
    f.write(f"Sniffing positive instances: {y_sniffing.sum()}\n")

def train_and_evaluate(X, y, label_name):
    # Train test split
    # Since it's sequential data, a time-series split or simple train_test_split could be used.
    # SimBA typically uses standard train_test_split or keeps videos separate. We'll use standard train_test_split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Quantitative evaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(f'outputs/{label_name}_eval_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {label_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'report/images/{label_name}_confusion_matrix.png')
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='.', label='Random Forest')
    plt.title(f'Precision-Recall Curve - {label_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'report/images/{label_name}_pr_curve.png')
    plt.close()
    
    # Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Save top 20 features
    top_n = 20
    top_features = X.columns[indices][:top_n]
    top_importances = importances[indices][:top_n]
    
    fi_df = pd.DataFrame({'Feature': top_features, 'Importance': top_importances})
    fi_df.to_csv(f'outputs/{label_name}_feature_importance.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title(f'Top {top_n} Feature Importances - {label_name}')
    plt.tight_layout()
    plt.savefig(f'report/images/{label_name}_feature_importance.png')
    plt.close()

# Run for both labels
if y_attack.sum() > 0:
    train_and_evaluate(X, y_attack, 'Attack')
else:
    print("No positive instances for Attack.")

if y_sniffing.sum() > 0:
    train_and_evaluate(X, y_sniffing, 'Sniffing')
else:
    print("No positive instances for Sniffing.")

print("Analysis complete.")
