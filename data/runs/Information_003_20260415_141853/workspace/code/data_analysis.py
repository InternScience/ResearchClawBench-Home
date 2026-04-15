"""
Data Analysis for NF-UNSW-NB15 Network Intrusion Detection Dataset
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(path='data/NF-UNSW-NB15-v2_3d.pt'):
    """Load the temporal graph data"""
    data = torch.load(path, weights_only=False)
    return data

def analyze_data(data, output_dir='outputs'):
    """Perform comprehensive data analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    features = data.msg.numpy()
    labels = data.label.numpy()
    attacks = data.attack.numpy()
    timestamps = data.t.numpy()
    
    print("=" * 60)
    print("NF-UNSW-NB15 Dataset Analysis")
    print("=" * 60)
    
    # Basic statistics
    print(f"\nTotal samples: {len(labels)}")
    print(f"Feature dimensions: {features.shape[1]}")
    print(f"Time range: {timestamps.min()} - {timestamps.max()} seconds")
    
    # Binary classification stats
    benign_count = (labels == 0).sum()
    attack_count = (labels == 1).sum()
    print(f"\n--- Binary Classification ---")
    print(f"Benign (0): {benign_count} ({100*benign_count/len(labels):.2f}%)")
    print(f"Attack (1): {attack_count} ({100*attack_count/len(labels):.2f}%)")
    
    # Multi-class attack stats
    print(f"\n--- Multi-class Attack Distribution ---")
    attack_counts = Counter(attacks)
    attack_names = {
        0: 'Benign', 1: 'Fuzzers', 2: 'Analysis', 3: 'Backdoor',
        4: 'DoS', 5: 'Exploits', 6: 'Generic', 7: 'Reconnaissance',
        8: 'Shellcode', 9: 'Worms'
    }
    for atk_id in sorted(attack_counts.keys()):
        count = attack_counts[atk_id]
        pct = 100 * count / len(labels)
        print(f"  {atk_id} - {attack_names.get(atk_id, 'Unknown')}: {count} ({pct:.2f}%)")
    
    # Feature statistics
    print(f"\n--- Feature Statistics ---")
    print(f"Feature mean: {features.mean():.4f}")
    print(f"Feature std: {features.std():.4f}")
    print(f"Feature min: {features.min():.4f}")
    print(f"Feature max: {features.max():.4f}")
    
    return {
        'features': features,
        'labels': labels,
        'attacks': attacks,
        'timestamps': timestamps,
        'attack_counts': attack_counts,
        'attack_names': attack_names
    }

def plot_class_distribution(labels, attacks, attack_names, output_dir='report/images'):
    """Plot class distribution for binary and multi-class"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Binary distribution
    benign_count = (labels == 0).sum()
    attack_count = (labels == 1).sum()
    axes[0].bar(['Benign', 'Attack'], [benign_count, attack_count], 
                color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Binary Class Distribution', fontsize=14, fontweight='bold')
    for i, v in enumerate([benign_count, attack_count]):
        axes[0].text(i, v + 1000, f'{v}\n({100*v/len(labels):.1f}%)', 
                    ha='center', va='bottom', fontsize=10)
    
    # Multi-class distribution
    attack_counts = Counter(attacks)
    attack_ids = sorted(attack_counts.keys())
    counts = [attack_counts[i] for i in attack_ids]
    names = [attack_names.get(i, f'Class {i}') for i in attack_ids]
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_ids)))
    
    bars = axes[1].bar(attack_ids, counts, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Attack Type', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Multi-class Attack Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xticks(attack_ids)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/class_distribution.png")

def plot_feature_heatmap(features, labels, output_dir='report/images'):
    """Plot feature correlation heatmap"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample for faster computation
    sample_idx = np.random.choice(len(features), min(5000, len(features)), replace=False)
    sample_features = features[sample_idx]
    sample_labels = labels[sample_idx]
    
    # Separate benign and attack
    benign_features = sample_features[sample_labels == 0]
    attack_features = sample_features[sample_labels == 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Benign correlation
    corr_benign = np.corrcoef(benign_features.T)
    im1 = axes[0].imshow(corr_benign, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title('Feature Correlation - Benign Traffic', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature Index', fontsize=12)
    axes[0].set_ylabel('Feature Index', fontsize=12)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Attack correlation
    corr_attack = np.corrcoef(attack_features.T)
    im2 = axes[1].imshow(corr_attack, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_title('Feature Correlation - Attack Traffic', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature Index', fontsize=12)
    axes[1].set_ylabel('Feature Index', fontsize=12)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feature_correlation.png")

def plot_feature_distribution(features, labels, output_dir='report/images'):
    """Plot feature distribution comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select top 6 features with highest variance
    feature_vars = np.var(features, axis=0)
    top_features = np.argsort(feature_vars)[-6:]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_features):
        benign_vals = features[labels == 0, feat_idx]
        attack_vals = features[labels == 1, feat_idx]
        
        axes[idx].hist(benign_vals, bins=50, alpha=0.6, label='Benign', 
                      color='#2ecc71', density=True)
        axes[idx].hist(attack_vals, bins=50, alpha=0.6, label='Attack', 
                      color='#e74c3c', density=True)
        axes[idx].set_xlabel(f'Feature {feat_idx}', fontsize=11)
        axes[idx].set_ylabel('Density', fontsize=11)
        axes[idx].set_title(f'Feature {feat_idx} Distribution', fontsize=12, fontweight='bold')
        axes[idx].legend()
    
    plt.suptitle('Top 6 Feature Distributions by Variance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feature_distributions.png")

def plot_temporal_distribution(timestamps, labels, output_dir='report/images'):
    """Plot temporal distribution of traffic"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamps to hours
    hours = timestamps / 3600
    
    # Create histograms
    bins = np.linspace(0, 24, 25)
    benign_mask = labels == 0
    attack_mask = labels == 1
    
    ax.hist(hours[benign_mask], bins=bins, alpha=0.7, label='Benign', 
            color='#2ecc71', edgecolor='black')
    ax.hist(hours[attack_mask], bins=bins, alpha=0.7, label='Attack', 
            color='#e74c3c', edgecolor='black', bottom=np.histogram(hours[benign_mask], bins=bins)[0])
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Temporal Distribution of Network Traffic', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 25, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/temporal_distribution.png")

if __name__ == '__main__':
    # Load data
    data = load_data()
    
    # Analyze
    stats = analyze_data(data)
    
    # Generate plots
    plot_class_distribution(stats['labels'], stats['attacks'], stats['attack_names'])
    plot_feature_heatmap(stats['features'], stats['labels'])
    plot_feature_distribution(stats['features'], stats['labels'])
    plot_temporal_distribution(stats['timestamps'], stats['labels'])
    
    print("\nData analysis complete!")
