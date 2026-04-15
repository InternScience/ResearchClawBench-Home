"""
Property Prediction Workflow Analysis
Implements machine learning models for predicting material properties
from atomic structure features.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load parsed property prediction data."""
    with open('../outputs/parsed_data.json', 'r') as f:
        data = json.load(f)
    return data['property_prediction']


def prepare_features(pp_data):
    """
    Prepare feature matrix from property prediction data.
    """
    features = np.array(pp_data['features'])
    atomic_nums = np.array(pp_data['atomic_numbers'])
    targets = np.array(pp_data['targets'])
    
    n_samples = len(targets)
    feature_matrix = np.zeros((n_samples, 6))
    
    for i in range(n_samples):
        idx = i % len(features) if len(features) > 0 else 0
        feature_matrix[i, 0] = features[idx] if idx < len(features) else 0
        feature_matrix[i, 1] = atomic_nums[i % len(atomic_nums)] if len(atomic_nums) > 0 else 5
        feature_matrix[i, 2] = feature_matrix[i, 0] ** 2
        feature_matrix[i, 3] = np.sin(feature_matrix[i, 0])
        feature_matrix[i, 4] = np.cos(feature_matrix[i, 0])
        feature_matrix[i, 5] = np.abs(feature_matrix[i, 0])
    
    return feature_matrix, targets


def train_models(X, y):
    """Train multiple ML models and evaluate performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'y_test': y_test,
            'y_pred': y_pred
        }
        trained_models[name] = model
    
    return results, trained_models


def plot_results(results):
    """Generate visualization plots for property prediction."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Model Comparison - R² Scores
    ax1 = axes[0, 0]
    models = list(results.keys())
    r2_scores = [results[m]['r2'] for m in models]
    colors = ['#2ecc71', '#3498db']
    bars = ax1.bar(models, r2_scores, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance Comparison (R²)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Target (0.9)')
    ax1.tick_params(axis='x', rotation=15)
    ax1.legend()
    for bar, score in zip(bars, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # 2. Model Comparison - RMSE
    ax2 = axes[0, 1]
    rmse_scores = [results[m]['rmse'] for m in models]
    bars = ax2.bar(models, rmse_scores, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Model Performance Comparison (RMSE)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    for bar, score in zip(bars, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # 3. Prediction vs Actual (Best Model)
    ax3 = axes[1, 0]
    best_model = max(results, key=lambda x: results[x]['r2'])
    y_test = results[best_model]['y_test']
    y_pred = results[best_model]['y_pred']
    ax3.scatter(y_test, y_pred, alpha=0.6, c='steelblue', edgecolors='black', s=60)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Values', fontsize=12)
    ax3.set_ylabel('Predicted Values', fontsize=12)
    ax3.set_title(f'Prediction vs Actual ({best_model})', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add metrics text
    textstr = f'R² = {results[best_model]["r2"]:.4f}\nRMSE = {results[best_model]["rmse"]:.4f}\nMAE = {results[best_model]["mae"]:.4f}'
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. Residual Plot
    ax4 = axes[1, 1]
    residuals = y_test - y_pred
    ax4.scatter(y_pred, residuals, alpha=0.6, c='coral', edgecolors='black', s=60)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax4.set_xlabel('Predicted Values', fontsize=12)
    ax4.set_ylabel('Residuals', fontsize=12)
    ax4.set_title(f'Residual Plot ({best_model})', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../report/images/property_prediction.png', dpi=200, bbox_inches='tight')
    plt.savefig('../outputs/property_prediction.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    return best_model


def save_metrics(results, best_model):
    """Save metrics to JSON file."""
    metrics = {
        'best_model': best_model,
        'models': {}
    }
    for name, res in results.items():
        metrics['models'][name] = {
            'mse': res['mse'],
            'rmse': res['rmse'],
            'mae': res['mae'],
            'r2': res['r2'],
        }
    
    with open('../outputs/property_prediction_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to outputs/property_prediction_metrics.json")
    return metrics


def main():
    print("=" * 60)
    print("PROPERTY PREDICTION WORKFLOW ANALYSIS")
    print("=" * 60)
    
    # Load and prepare data
    pp_data = load_data()
    X, y = prepare_features(pp_data)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Train models
    print("\nTraining ML models...")
    results, models = train_models(X, y)
    
    # Print results
    print("\n" + "-" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("-" * 60)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  R² Score: {res['r2']:.4f}")
        print(f"  RMSE: {res['rmse']:.4f}")
        print(f"  MAE: {res['mae']:.4f}")
    
    # Generate plots
    print("\nGenerating property prediction plots...")
    best_model = plot_results(results)
    print(f"Best performing model: {best_model}")
    
    # Save metrics
    metrics = save_metrics(results, best_model)
    
    print("\n" + "=" * 60)
    print("Property prediction analysis complete!")
    print("Plots saved to: report/images/property_prediction.png")
    print("=" * 60)
    
    return results, best_model


if __name__ == '__main__':
    main()
