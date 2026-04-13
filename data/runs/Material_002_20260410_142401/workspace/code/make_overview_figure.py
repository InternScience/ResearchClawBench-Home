import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
IMG.mkdir(parents=True, exist_ok=True)

summary = json.loads((OUT / 'summary.json').read_text())
rxn = pd.read_csv(OUT / 'reaction_barriers.csv')
ads = pd.read_csv(OUT / 'adsorption_energies.csv')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
# panel 1
axes[0].bar(['Water MD', 'Adsorption', 'Reaction barriers'], [96, 12, 3], color=['tab:blue', 'tab:green', 'tab:orange'])
axes[0].set_ylabel('System/evaluation count')
axes[0].set_title('Benchmark coverage')
# panel 2
axes[1].bar(['Peak r', 'Peak g(r)', 'Mean T/100'], [summary['water']['first_peak_r_angstrom'], summary['water']['first_peak_height'], summary['water']['mean_temperature_K']/100.0], color='tab:blue')
axes[1].set_title('Water MD summary')
axes[1].set_ylabel('Value (mixed units)')
# panel 3
axes[2].bar(['Scaling $R^2$', 'Barrier MAE'], [summary['adsorption']['r2'], summary['reactions']['mae_eV']], color=['tab:green','tab:red'])
axes[2].set_title('Validation metrics')
axes[2].set_ylabel('Metric value')
for ax in axes:
    ax.tick_params(axis='x', rotation=20)
fig.suptitle('Overview of reproduced MACE-MP-0 evaluation tasks', y=1.05)
fig.tight_layout()
fig.savefig(IMG / 'data_overview.png', dpi=200, bbox_inches='tight')
print('wrote', IMG / 'data_overview.png')
