#!/usr/bin/env python3
"""
Generate plots for XEB fidelity vs depth for N=40 RCS.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    with open('outputs/xeb_fidelities.json', 'r') as f:
        results = json.load(f)
    
    # Group by depth
    depths = sorted(set(r['d'] for r in results))
    fid_by_d = {d: [] for d in depths}
    for r in results:
        fid_by_d[r['d']].append(r['fidelity'])
    
    mean_fids = [np.mean(fid_by_d[d]) for d in depths]
    std_fids = [np.std(fid_by_d[d]) / np.sqrt(len(fid_by_d[d])) for d in depths]
    
    os.makedirs('report/images', exist_ok=True)
    
    # Plot 1: Fidelity vs Depth
    plt.figure(figsize=(8, 6))
    plt.errorbar(depths, mean_fids, yerr=std_fids, fmt='o-', capsize=5, label='Mean XEB Fidelity ± SEM')
    plt.xlabel('Circuit Depth d')
    plt.ylabel('XEB Fidelity F')
    plt.title('XEB Fidelity vs Circuit Depth (N=40 qubits)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/fidelity_vs_depth.png', dpi=150)
    plt.close()
    
    # Plot 2: Boxplot by depth
    plt.figure(figsize=(10, 6))
    data = [fid_by_d[d] for d in depths]
    plt.boxplot(data, labels=depths)
    plt.xlabel('Circuit Depth d')
    plt.ylabel('XEB Fidelity')
    plt.title('Distribution of XEB Fidelities by Depth (N=40)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fidelity_boxplot.png', dpi=150)
    plt.close()
    
    print("Saved plots to report/images/")

if __name__ == '__main__':
    main()