"""
Simplest demonstration of Fourier-based KAN concept.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

def fourier_features(X, num_freqs=4):
    """Transform features using Fourier basis."""
    n_samples, n_features = X.shape
    result = []
    
    for k in range(1, num_freqs + 1):
        result.append(np.cos(k * X))
        result.append(np.sin(k * X))
    
    return np.hstack(result)

def generate_data(n=1000):
    """Generate synthetic data."""
    X = np.random.randn(n, 10)
    # Non-linear relationship
    y = ((np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.5 * X[:, 2] ** 2) > 0).astype(int)
    return X, y

def run_simple_demo():
    """Run simplest demo."""
    print("="*60)
    print("Simplest Demo: Fourier Features + MLP vs Plain MLP")
    print("="*60)
    
    # Generate data
    X, y = generate_data(2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"\nData: {len(X_train)} train, {len(X_test)} test")
    print(f"Features: {X.shape[1]}")
    print(f"Positive: {y.mean()*100:.1f}%")
    
    # MLP baseline
    print("\nTraining MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=200, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    preds_mlp = mlp.predict_proba(X_test)[:, 1]
    auc_mlp = roc_auc_score(y_test, preds_mlp)
    acc_mlp = accuracy_score(y_test, (preds_mlp > 0.5).astype(int))
    print(f"  AUC: {auc_mlp:.4f}, Acc: {acc_mlp:.4f}")
    
    # Fourier + MLP
    print("\nTraining Fourier KAN (Fourier features + MLP)...")
    X_train_f = fourier_features(X_train, num_freqs=4)
    X_test_f = fourier_features(X_test, num_freqs=4)
    
    kan = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=200, random_state=42, early_stopping=True)
    kan.fit(X_train_f, y_train)
    preds_kan = kan.predict_proba(X_test_f)[:, 1]
    auc_kan = roc_auc_score(y_test, preds_kan)
    acc_kan = accuracy_score(y_test, (preds_kan > 0.5).astype(int))
    print(f"  AUC: {auc_kan:.4f}, Acc: {acc_kan:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Method':<25} {'Test AUC':<12} {'Test Acc':<12}")
    print("-"*60)
    print(f"{'MLP':<25} {auc_mlp:<12.4f} {acc_mlp:<12.4f}")
    print(f"{'Fourier KAN (ours)':<25} {auc_kan:<12.4f} {acc_kan:<12.4f}")
    print("="*60)
    print(f"\nImprovement: {(auc_kan - auc_mlp)*100:.2f} percentage points")
    
    # Plot training curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp.loss_curve_, label='MLP', linewidth=2)
    ax.plot(kan.loss_curve_, label='Fourier KAN', linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('report/images', exist_ok=True)
    plt.savefig('report/images/simplest_demo.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to report/images/simplest_demo.png")
    
    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    results = {
        'MLP': {'test_auc': float(auc_mlp), 'test_acc': float(acc_mlp)},
        'Fourier_KAN': {'test_auc': float(auc_kan), 'test_acc': float(acc_kan)}
    }
    with open('outputs/results/simplest_demo.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    results = run_simple_demo()
