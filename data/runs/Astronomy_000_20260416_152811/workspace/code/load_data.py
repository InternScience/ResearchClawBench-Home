"""
Load and explore posterior samples from black hole observations.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Astronomy_000_20260416_152811"
DATA_DIR = os.path.join(WORKSPACE, "data")
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")

def load_posterior_samples(filepath):
    """Load posterior samples from .dat file, skipping comment lines."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                mass = float(parts[0])
                spin = float(parts[1])
                data.append([mass, spin])
    return np.array(data)

def main():
    # Load data
    iras_path = os.path.join(DATA_DIR, "IRAS_09149-6206_samples.dat")
    m33_path = os.path.join(DATA_DIR, "M33_X-7_samples.dat")
    
    print(f"Loading IRAS 09149-6206 samples from {iras_path}")
    iras_samples = load_posterior_samples(iras_path)
    print(f"  Loaded {len(iras_samples)} samples")
    print(f"  Mass range: {iras_samples[:, 0].min():.3e} - {iras_samples[:, 0].max():.3e} Msol")
    print(f"  Spin range: {iras_samples[:, 1].min():.3f} - {iras_samples[:, 1].max():.3f}")
    print(f"  Mass median: {np.median(iras_samples[:, 0]):.3e} Msol")
    print(f"  Spin median: {np.median(iras_samples[:, 1]):.3f}")
    
    print(f"\nLoading M33 X-7 samples from {m33_path}")
    m33_samples = load_posterior_samples(m33_path)
    print(f"  Loaded {len(m33_samples)} samples")
    print(f"  Mass range: {m33_samples[:, 0].min():.3f} - {m33_samples[:, 0].max():.3f} Msol")
    print(f"  Spin range: {m33_samples[:, 1].min():.3f} - {m33_samples[:, 1].max():.3f}")
    print(f"  Mass median: {np.median(m33_samples[:, 0]):.3f} Msol")
    print(f"  Spin median: {np.median(m33_samples[:, 1]):.3f}")
    
    # Save summary statistics
    summary = {
        'IRAS_09149-6206': {
            'n_samples': len(iras_samples),
            'mass_median': float(np.median(iras_samples[:, 0])),
            'mass_std': float(np.std(iras_samples[:, 0])),
            'mass_min': float(iras_samples[:, 0].min()),
            'mass_max': float(iras_samples[:, 0].max()),
            'spin_median': float(np.median(iras_samples[:, 1])),
            'spin_std': float(np.std(iras_samples[:, 1])),
            'spin_min': float(iras_samples[:, 1].min()),
            'spin_max': float(iras_samples[:, 1].max()),
        },
        'M33_X-7': {
            'n_samples': len(m33_samples),
            'mass_median': float(np.median(m33_samples[:, 0])),
            'mass_std': float(np.std(m33_samples[:, 0])),
            'mass_min': float(m33_samples[:, 0].min()),
            'mass_max': float(m33_samples[:, 0].max()),
            'spin_median': float(np.median(m33_samples[:, 1])),
            'spin_std': float(np.std(m33_samples[:, 1])),
            'spin_min': float(m33_samples[:, 1].min()),
            'spin_max': float(m33_samples[:, 1].max()),
        }
    }
    
    import json
    with open(os.path.join(OUTPUTS_DIR, "data_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary statistics to {os.path.join(OUTPUTS_DIR, 'data_summary.json')}")
    
    # Create data overview plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # IRAS 09149-6206
    ax = axes[0]
    h = ax.hist2d(np.log10(iras_samples[:, 0]), iras_samples[:, 1], 
                  bins=50, cmap='viridis', density=True)
    ax.set_xlabel('log10(Mass [Msol])')
    ax.set_ylabel('Dimensionless Spin a*')
    ax.set_title('IRAS 09149-6206 (SMBH)\nPosterior Distribution')
    ax.axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='a* = 0.99')
    ax.legend()
    plt.colorbar(h[3], ax=ax, label='Probability Density')
    
    # M33 X-7
    ax = axes[1]
    h = ax.hist2d(m33_samples[:, 0], m33_samples[:, 1], 
                  bins=50, cmap='viridis', density=True)
    ax.set_xlabel('Mass [Msol]')
    ax.set_ylabel('Dimensionless Spin a*')
    ax.set_title('M33 X-7 (Stellar-mass BH)\nPosterior Distribution')
    ax.axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='a* = 0.99')
    ax.legend()
    plt.colorbar(h[3], ax=ax, label='Probability Density')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "data_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved data overview plot to {os.path.join(IMAGES_DIR, 'data_overview.png')}")
    
    # Save samples for later use
    np.save(os.path.join(OUTPUTS_DIR, "iras_samples.npy"), iras_samples)
    np.save(os.path.join(OUTPUTS_DIR, "m33_samples.npy"), m33_samples)
    print(f"Saved samples to {OUTPUTS_DIR}")
    
    return iras_samples, m33_samples

if __name__ == "__main__":
    iras, m33 = main()
