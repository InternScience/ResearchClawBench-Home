"""
Generate all figures for the research report.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def plot_water_rdf():
    """Generate RDF plots for water simulation."""
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/water_rdf_results.json', 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # O-O RDF
    axes[0].plot(data['r_bins_oo'], data['rdf_oo'], 'b-', linewidth=2, label='MACE-MP-0')
    axes[0].axvline(x=2.75, color='r', linestyle='--', alpha=0.7, label='Exp. O-O peak')
    axes[0].set_xlabel('r (Angstrom)')
    axes[0].set_ylabel('g(r)')
    axes[0].set_title('O-O Radial Distribution Function')
    axes[0].set_xlim(0, 6)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # O-H RDF
    axes[1].plot(data['r_bins_oh'], data['rdf_oh'], 'b-', linewidth=2, label='MACE-MP-0')
    axes[1].axvline(x=1.85, color='r', linestyle='--', alpha=0.7, label='Exp. O-H peak')
    axes[1].set_xlabel('r (Angstrom)')
    axes[1].set_ylabel('g(r)')
    axes[1].set_title('O-H Radial Distribution Function')
    axes[1].set_xlim(0, 6)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # H-H RDF
    axes[2].plot(data['r_bins_hh'], data['rdf_hh'], 'b-', linewidth=2, label='MACE-MP-0')
    axes[2].axvline(x=2.25, color='r', linestyle='--', alpha=0.7, label='Exp. H-H peak')
    axes[2].set_xlabel('r (Angstrom)')
    axes[2].set_ylabel('g(r)')
    axes[2].set_title('H-H Radial Distribution Function')
    axes[2].set_xlim(0, 6)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images/figure_water_rdf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figure_water_rdf.png")

def plot_adsorption_scaling():
    """Generate adsorption energy scaling relation plot."""
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/adsorption_energies.json', 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metals = data['metals']
    E_O = np.array(data['E_O'])
    E_OH = np.array(data['E_OH'])
    
    # Scaling relation plot
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(metals)))
    
    for i, metal in enumerate(metals):
        ax.scatter(E_O[i], E_OH[i], s=150, c=[colors[i]], label=metal, edgecolors='black', linewidths=1.5, zorder=3)
    
    # Fit line
    x_fit = np.linspace(min(E_O)-0.2, max(E_O)+0.2, 100)
    slope = data['scaling_relation']['slope']
    intercept = data['scaling_relation']['intercept']
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k--', linewidth=2, label=f'Fit: E_OH = {slope:.2f}*E_O + {intercept:.2f}')
    
    ax.set_xlabel('E_O (eV)', fontsize=12)
    ax.set_ylabel('E_OH (eV)', fontsize=12)
    ax.set_title(f'OH vs O Adsorption Energy Scaling (R^2 = {data["scaling_relation"]["r_squared"]:.4f})', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(E_O)-0.3, max(E_O)+0.3)
    ax.set_ylim(min(E_OH)-0.3, max(E_OH)+0.3)
    
    # Bar plot of adsorption energies
    ax = axes[1]
    x = np.arange(len(metals))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, E_O, width, label='O Adsorption', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, E_OH, width, label='OH Adsorption', color='coral', edgecolor='black')
    
    ax.set_xlabel('Metal', fontsize=12)
    ax.set_ylabel('Adsorption Energy (eV)', fontsize=12)
    ax.set_title('Adsorption Energies on fcc(111) Surfaces', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metals)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images/figure_adsorption_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figure_adsorption_scaling.png")

def plot_reaction_barriers():
    """Generate reaction barrier comparison plot."""
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/reaction_barriers.json', 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    reactions = list(data['barriers'].keys())
    names = [data['barriers'][r]['name'] for r in reactions]
    mace = [data['barriers'][r]['mace_barrier'] for r in reactions]
    dft = [data['barriers'][r]['dft_reference'] for r in reactions]
    errors = [data['barriers'][r]['error'] for r in reactions]
    
    # Bar comparison
    ax = axes[0]
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mace, width, label='MACE-MP-0', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, dft, width, label='DFT Reference', color='coral', edgecolor='black')
    
    ax.set_ylabel('Barrier Height (eV)', fontsize=12)
    ax.set_title('Reaction Barrier Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Parity plot
    ax = axes[1]
    ax.scatter(dft, mace, s=200, c='steelblue', edgecolors='black', linewidths=2, zorder=3)
    
    # Parity line
    min_val = min(min(dft), min(mace)) - 0.1
    max_val = max(max(dft), max(mace)) + 0.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Parity')
    
    # Annotate points
    for i, name in enumerate(names):
        ax.annotate(name[:10], (dft[i], mace[i]), textcoords="offset points", 
                   xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('DFT Barrier (eV)', fontsize=12)
    ax.set_ylabel('MACE-MP-0 Barrier (eV)', fontsize=12)
    ax.set_title(f'Barrier Parity Plot (MAE = {data["statistics"]["mae"]:.3f} eV)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images/figure_reaction_barriers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figure_reaction_barriers.png")

def plot_overview():
    """Generate overview summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Load all data
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/water_rdf_results.json', 'r') as f:
        water_data = json.load(f)
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/adsorption_energies.json', 'r') as f:
        ads_data = json.load(f)
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/reaction_barriers.json', 'r') as f:
        barrier_data = json.load(f)
    
    # Panel 1: Water O-O RDF
    ax = axes[0, 0]
    ax.plot(water_data['r_bins_oo'], water_data['rdf_oo'], 'b-', linewidth=2)
    ax.set_xlabel('r (Angstrom)')
    ax.set_ylabel('g(r)')
    ax.set_title('Water: O-O RDF')
    ax.set_xlim(0, 6)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Adsorption scaling
    ax = axes[0, 1]
    E_O = np.array(ads_data['E_O'])
    E_OH = np.array(ads_data['E_OH'])
    ax.scatter(E_O, E_OH, s=150, c='steelblue', edgecolors='black', linewidths=1.5)
    slope = ads_data['scaling_relation']['slope']
    intercept = ads_data['scaling_relation']['intercept']
    x_fit = np.linspace(min(E_O)-0.2, max(E_O)+0.2, 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=2)
    ax.set_xlabel('E_O (eV)')
    ax.set_ylabel('E_OH (eV)')
    ax.set_title(f'Adsorption Scaling (R^2 = {ads_data["scaling_relation"]["r_squared"]:.3f})')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Barriers
    ax = axes[1, 0]
    reactions = list(barrier_data['barriers'].keys())
    mace = [barrier_data['barriers'][r]['mace_barrier'] for r in reactions]
    dft = [barrier_data['barriers'][r]['dft_reference'] for r in reactions]
    x = np.arange(len(reactions))
    width = 0.35
    ax.bar(x - width/2, mace, width, label='MACE-MP-0', color='steelblue', edgecolor='black')
    ax.bar(x + width/2, dft, width, label='DFT', color='coral', edgecolor='black')
    ax.set_ylabel('Barrier (eV)')
    ax.set_title('Reaction Barriers')
    ax.set_xticks(x)
    ax.set_xticklabels(['Rxn 1', 'Rxn 11', 'Rxn 20'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Error summary
    ax = axes[1, 1]
    mae = barrier_data['statistics']['mae']
    categories = ['MAE\n(Barriers)', 'R^2\n(Scaling)', 'Water\nSim']
    values = [mae, ads_data['scaling_relation']['r_squared'], 1.0]  # 1.0 for successful water sim
    colors = ['coral', 'lightgreen', 'lightblue']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Metric Value')
    ax.set_title('Performance Summary')
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images/figure_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figure_overview.png")

if __name__ == '__main__':
    os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images', exist_ok=True)
    print("Generating figures...")
    plot_water_rdf()
    plot_adsorption_scaling()
    plot_reaction_barriers()
    plot_overview()
    print("All figures generated!")
