import re, json, math, csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'MACE-MP-0_Reproduction_Dataset.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

text = DATA.read_text()
metals = dict(re.findall(r'^\s*([A-Z][a-z]?):\s*([0-9]+\.[0-9]+)$', text, re.M))
metals = {k: float(v) for k, v in metals.items() if k in ['Ni', 'Cu', 'Rh', 'Pd', 'Ir', 'Pt']}
barriers = {f'Rxn {k}': float(v) for k, v in re.findall(r'Rxn\s+(\d+):\s*([0-9]+\.[0-9]+)', text)}
water = {
    'num_molecules': int(re.search(r'Number of water molecules:\s*(\d+)', text).group(1)),
    'box_size_A': float(re.search(r'Box size \(Å\):\s*([0-9.]+)', text).group(1)),
    'temperature_K': float(re.search(r'Temperature \(K\):\s*([0-9.]+)', text).group(1)),
    'time_step_fs': float(re.search(r'Time step \(fs\):\s*([0-9.]+)', text).group(1)),
    'md_steps': int(re.search(r'Total number of MD steps:\s*(\d+)', text).group(1)),
    'friction_fs_inv': float(re.search(r'Friction coefficient .*?:\s*([0-9.]+)', text).group(1)),
}
NA = 6.02214076e23
volume_cm3 = (water['box_size_A'] * 1e-8) ** 3
mass_g = (water['num_molecules'] / NA) * 18.01528
water['density_g_cm3'] = mass_g / volume_cm3
water['sim_time_ps'] = water['time_step_fs'] * water['md_steps'] / 1000.0
coords = {'O':[0.0,0.0,0.119262], 'H1':[0.0,0.763239,-0.477047], 'H2':[0.0,-0.763239,-0.477047]}
def dist(a,b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))
water['oh_bond_A'] = round(dist(coords['O'], coords['H1']), 4)
water['hh_distance_A'] = round(dist(coords['H1'], coords['H2']), 4)

(OUT / 'reproduction_dataset.json').write_text(json.dumps({'water': water, 'metals': metals, 'dft_barriers_eV': barriers}, indent=2))
overview = [
    {'benchmark':'water_rdf','items':water['num_molecules'],'quantity':'molecules','note':'32 H2O in cubic box'},
    {'benchmark':'adsorption_scaling','items':len(metals),'quantity':'metals','note':'fcc(111) surfaces with O/OH adsorbates'},
    {'benchmark':'reaction_barriers','items':len(barriers),'quantity':'reactions','note':'CRBH20 subset with DFT references'}
]
with open(OUT / 'data_overview.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['benchmark','items','quantity','note'])
    w.writeheader(); w.writerows(overview)

# Figures
fig, axes = plt.subplots(1, 3, figsize=(13,4))
overview_df = pd.DataFrame(overview)
sns.barplot(data=overview_df, x='benchmark', y='items', hue='benchmark', legend=False, palette='viridis', ax=axes[0])
axes[0].set_title('Benchmark coverage'); axes[0].set_xlabel(''); axes[0].tick_params(axis='x', rotation=20); axes[0].set_ylabel('Count')
metals_df = pd.DataFrame({'metal': list(metals.keys()), 'lattice_A': list(metals.values())}).sort_values('lattice_A')
sns.pointplot(data=metals_df, x='metal', y='lattice_A', color='tab:blue', ax=axes[1])
axes[1].set_title('fcc(111) metal set'); axes[1].set_xlabel('Metal'); axes[1].set_ylabel('Lattice constant (Å)')
metrics_df = pd.DataFrame({'metric':['Temperature (K)','Density (g/cm³)','OH bond (Å)','Sim time (ps)'], 'value':[water['temperature_K'], water['density_g_cm3'], water['oh_bond_A'], water['sim_time_ps']]})
sns.barplot(data=metrics_df, y='metric', x='value', hue='metric', legend=False, palette='magma', ax=axes[2])
axes[2].set_title('Water setup descriptors'); axes[2].set_xlabel('Value'); axes[2].set_ylabel('')
plt.tight_layout(); fig.savefig(IMG / 'benchmark_overview.png', dpi=200, bbox_inches='tight'); plt.close(fig)

bar_df = pd.DataFrame({'reaction': list(barriers.keys()), 'barrier_eV': list(barriers.values())})
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=bar_df, x='reaction', y='barrier_eV', hue='reaction', legend=False, palette='crest', ax=ax)
for i,v in enumerate(bar_df['barrier_eV']): ax.text(i, v+0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
ax.set_title('CRBH20 subset reference barriers'); ax.set_xlabel('Reaction'); ax.set_ylabel('DFT barrier (eV)'); ax.set_ylim(0, max(bar_df['barrier_eV'])+0.25)
plt.tight_layout(); fig.savefig(IMG / 'reaction_barriers.png', dpi=200, bbox_inches='tight'); plt.close(fig)

metals_df['relative_to_pt'] = metals_df['lattice_A'] - metals_df.loc[metals_df['metal']=='Pt','lattice_A'].iloc[0]
fig, ax = plt.subplots(figsize=(6.5,4))
sns.scatterplot(data=metals_df, x='lattice_A', y='relative_to_pt', hue='metal', s=90, palette='tab10', ax=ax)
for _,r in metals_df.iterrows(): ax.text(r['lattice_A']+0.005, r['relative_to_pt']+0.002, r['metal'], fontsize=9)
ax.axhline(0, ls='--', c='gray', lw=1)
ax.set_title('Surface-metal span in adsorption benchmark'); ax.set_xlabel('Lattice constant (Å)'); ax.set_ylabel('Relative to Pt (Å)')
plt.tight_layout(); fig.savefig(IMG / 'adsorption_metal_span.png', dpi=200, bbox_inches='tight'); plt.close(fig)

print('done')
