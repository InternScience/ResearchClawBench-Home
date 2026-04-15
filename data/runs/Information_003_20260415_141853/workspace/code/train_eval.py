"""
Training and Evaluation Script for DIDS-MFL and Baselines
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

from dids_mfl import DIDS_MFL
from baselines import (MLPBaseline, LSTMBaseline, GraphSAGEBaseline, 
                       EGraphSAGE, GATBaseline, TabNetBaseline)

# Set random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_and_preprocess_data(data_path='data/NF-UNSW-NB15-v2_3d.pt'):
    """Load and preprocess the NF-UNSW-NB15 data"""
    print("Loading data...")
    data = torch.load(data_path, weights_only=False)
    
    # Extract features and labels
    features = data.msg.numpy()
    binary_labels = data.label.numpy()
    attack_labels = data.attack.numpy()
    timestamps = data.t.numpy()
    
    print(f"Data shape: {features.shape}")
    print(f"Binary labels: {np.unique(binary_labels, return_counts=True)}")
    print(f"Attack types: {np.unique(attack_labels, return_counts=True)}")
    
    return {
        'features': features,
        'binary_labels': binary_labels,
        'attack_labels': attack_labels,
        'timestamps': timestamps,
    }


def create_data_splits(data_dict, test_size=0.2, val_size=0.1):
    """Create train/val/test splits with temporal awareness"""
    features = data_dict['features']
    attack_labels = data_dict['attack_labels']
    binary_labels = data_dict['binary_labels']
    timestamps = data_dict['timestamps']
    
    n_samples = len(features)
    
    # Temporal split: use earlier data for training, later for testing
    sorted_idx = np.argsort(timestamps)
    
    # Split: 70% train, 10% val, 20% test (temporal)
    train_end = int(0.7 * n_samples)
    val_end = int(0.8 * n_samples)
    
    train_idx = sorted_idx[:train_end]
    val_idx = sorted_idx[train_end:val_end]
    test_idx = sorted_idx[val_end:]
    
    # Standardize features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features[train_idx])
    features_val = scaler.transform(features[val_idx])
    features_test = scaler.transform(features[test_idx])
    
    splits = {
        'train': {
            'X': torch.tensor(features_train, dtype=torch.float32),
            'y_binary': torch.tensor(binary_labels[train_idx], dtype=torch.long),
            'y_attack': torch.tensor(attack_labels[train_idx], dtype=torch.long),
        },
        'val': {
            'X': torch.tensor(features_val, dtype=torch.float32),
            'y_binary': torch.tensor(binary_labels[val_idx], dtype=torch.long),
            'y_attack': torch.tensor(attack_labels[val_idx], dtype=torch.long),
        },
        'test': {
            'X': torch.tensor(features_test, dtype=torch.float32),
            'y_binary': torch.tensor(binary_labels[test_idx], dtype=torch.long),
            'y_attack': torch.tensor(attack_labels[test_idx], dtype=torch.long),
        }
    }
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    
    return splits, scaler


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_model(model, train_loader, val_loader, model_name, n_epochs=50, lr=0.001):
    """Train a model with early stopping"""
    print(f"\nTraining {model_name}...")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_f1 = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        scheduler.step(val_metrics['loss'])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.2f}%, Val F1={val_metrics['f1']:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def plot_training_curves(histories, output_dir='report/images'):
    """Plot training curves for all models"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for model_name, history in histories.items():
        axes[0, 0].plot(history['train_loss'], label=model_name)
        axes[0, 1].plot(history['train_acc'], label=model_name)
        axes[1, 0].plot(history['val_loss'], label=model_name)
        axes[1, 1].plot(history['val_f1'], label=model_name)
    
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Training Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Validation Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation F1 Score', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/training_curves.png")


def plot_confusion_matrices(results, n_classes=10, output_dir='report/images'):
    """Plot confusion matrices for all models"""
    os.makedirs(output_dir, exist_ok=True)
    
    attack_names = ['Benign', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS', 
                   'Exploits', 'Generic', 'Recon', 'Shellcode', 'Worms']
    
    n_models = len(results)
    fig, axes = plt.subplots(1, min(n_models, 4), figsize=(5 * min(n_models, 4), 4))
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, metrics) in enumerate(list(results.items())[:4]):
        cm = confusion_matrix(metrics['labels'], metrics['predictions'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=range(n_classes), yticklabels=range(n_classes),
                   ax=axes[idx], cbar=True)
        axes[idx].set_title(f'{model_name}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/confusion_matrices.png")


def plot_performance_comparison(results, output_dir='report/images'):
    """Plot performance comparison across models"""
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    
    data = {metric: [results[m][metric] for m in models] for metric in metrics_names}
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    for i, metric in enumerate(metrics_names):
        ax.bar(x + i * width, data[metric], width, label=metric.capitalize(), color=colors[i])
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/performance_comparison.png")


def plot_per_class_f1(results, n_classes=10, output_dir='report/images'):
    """Plot per-class F1 scores"""
    os.makedirs(output_dir, exist_ok=True)
    
    attack_names = ['Benign', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS', 
                   'Exploits', 'Generic', 'Recon', 'Shellcode', 'Worms']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(n_classes)
    width = 0.15
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        labels = metrics['labels']
        preds = metrics['predictions']
        
        # Calculate per-class F1
        f1_per_class = []
        for c in range(n_classes):
            true_c = (labels == c)
            pred_c = (preds == c)
            if true_c.sum() == 0:
                f1_per_class.append(0)
            else:
                tp = ((labels == c) & (preds == c)).sum()
                fp = ((labels != c) & (preds == c)).sum()
                fn = ((labels == c) & (preds != c)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_per_class.append(f1)
        
        ax.bar(x + idx * width, f1_per_class, width, label=model_name, color=colors[idx])
    
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(attack_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/per_class_f1.png")


def main():
    """Main training and evaluation pipeline"""
    # Load data
    data_dict = load_and_preprocess_data()
    splits, scaler = create_data_splits(data_dict)
    
    # Create dataloaders
    batch_size = 256
    train_dataset = TensorDataset(splits['train']['X'], splits['train']['y_attack'])
    val_dataset = TensorDataset(splits['val']['X'], splits['val']['y_attack'])
    test_dataset = TensorDataset(splits['test']['X'], splits['test']['y_attack'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    n_features = splits['train']['X'].shape[1]
    n_classes = 10
    
    # Define models
    models_dict = {
        'MLP': MLPBaseline(n_features, n_classes),
        'LSTM': LSTMBaseline(n_features, n_classes),
        'TabNet': TabNetBaseline(n_features, n_classes),
        'DIDS-MFL': DIDS_MFL(n_features, n_classes)
    }
    
    # Train models
    trained_models = {}
    histories = {}
    
    for name, model in models_dict.items():
        trained_model, history = train_model(
            model, train_loader, val_loader, name, n_epochs=30, lr=0.001
        )
        trained_models[name] = trained_model
        histories[name] = history
    
    # Evaluate on test set
    criterion = nn.CrossEntropyLoss()
    test_results = {}
    
    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate(model, test_loader, criterion, device)
        test_results[name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/test_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON
        results_json = {}
        for name, metrics in test_results.items():
            results_json[name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
        json.dump(results_json, f, indent=2)
    
    # Generate plots
    plot_training_curves(histories)
    plot_confusion_matrices(test_results)
    plot_performance_comparison(test_results)
    plot_per_class_f1(test_results)
    
    print("\nTraining and evaluation complete!")
    return test_results


if __name__ == '__main__':
    main()
