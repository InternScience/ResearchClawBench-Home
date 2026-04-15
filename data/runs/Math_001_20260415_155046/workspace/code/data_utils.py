"""Data loading utilities for Lasso problem"""

import numpy as np
import matplotlib.pyplot as plt


def load_lasso_data(data_path='data/complex_optimization_data.npy'):
    """Load Lasso regression dataset"""
    data = np.load(data_path, allow_pickle=True).item()
    return data['A'], data['b'], data['x_true']


def plot_data_overview(A, b, x_true, save_path='report/images/data_overview.png'):
    """Create visualization of the Lasso problem structure"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Design matrix A heatmap (sample)
    ax = axes[0, 0]
    sample_size = min(200, A.shape[0])
    im = ax.imshow(A[:sample_size, :sample_size], cmap='RdBu_r', aspect='auto')
    ax.set_title('Design Matrix A (Sample)', fontsize=12)
    ax.set_xlabel('Features')
    ax.set_ylabel('Samples')
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Singular value distribution
    ax = axes[0, 1]
    from scipy.linalg import svd
    s = svd(A, compute_uv=False)
    ax.semilogy(s, 'b-', linewidth=1)
    ax.set_title('Singular Value Spectrum', fontsize=12)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value (log scale)')
    ax.axhline(y=s[-1], color='r', linestyle='--', label=f'σ_min = {s[-1]:.2e}')
    ax.axhline(y=s[0], color='g', linestyle='--', label=f'σ_max = {s[0]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Condition number info
    ax = axes[0, 2]
    cond_num = s[0] / s[-1]
    ax.bar(['Condition Number\nκ(A)'], [cond_num], color='steelblue')
    ax.set_title(f'Ill-conditioning: κ = {cond_num:.1e}', fontsize=12)
    ax.set_ylabel('Condition Number (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: True solution sparsity
    ax = axes[1, 0]
    nonzeros = np.count_nonzero(x_true)
    sparsity = nonzeros / len(x_true)
    ax.stem(np.arange(len(x_true)), x_true, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_title(f'True Solution (nnz={nonzeros}, sparsity={sparsity:.3f})', fontsize=12)
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('Value')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 5: Response vector b
    ax = axes[1, 1]
    ax.plot(b, 'g-', linewidth=0.5)
    ax.set_title('Response Vector b', fontsize=12)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Problem summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    Lasso Problem Summary:
    
    Dimensions:
    • Samples (m): {A.shape[0]}
    • Features (n): {A.shape[1]}
    • Overdetermined: {A.shape[0] < A.shape[1]}
    
    Properties:
    • Condition number: {cond_num:.2e}
    • True sparsity: {sparsity:.2%}
    • Non-zero coeffs: {nonzeros}
    
    Objective:
    min (1/2)||Ax - b||² + λ||x||₁
    
    Challenge:
    Ill-conditioned, high-dimensional
    sparse recovery problem
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Data overview saved to {save_path}")
    return fig


if __name__ == '__main__':
    A, b, x_true = load_lasso_data()
    plot_data_overview(A, b, x_true)
