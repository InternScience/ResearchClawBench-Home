"""
Generate final candidates for inverse design.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Load calibrated data
vit_df = pd.read_csv('outputs/vitrimer_calibrated.csv')
print(f"Loaded {len(vit_df)} vitrimer systems")
print(f"Tg range: {vit_df['tg_calibrated'].min():.1f} - {vit_df['tg_calibrated'].max():.1f} K")
print(f"Uncertainty range: {vit_df['tg_cal_uncertainty'].min():.1f} - {vit_df['tg_cal_uncertainty'].max():.1f} K")

# Target Tg values for different applications
targets = {
    'Low Tg (flexible)': 320,
    'Medium Tg (general)': 380,
    'High Tg (rigid)': 430
}

candidates = []

for target_name, target_tg in targets.items():
    # Find molecules closest to target Tg
    vit_df['tg_distance'] = np.abs(vit_df['tg_calibrated'] - target_tg)
    
    # Get top candidates (highest confidence = lowest uncertainty)
    top_candidates = vit_df.nsmallest(5, 'tg_distance')
    
    print(f"\n{target_name} (target: {target_tg} K)")
    
    for _, row in top_candidates.iterrows():
        print(f"  Tg: {row['tg_calibrated']:.1f} ± {row['tg_cal_uncertainty']:.1f} K")
        candidates.append({
            'target_category': target_name,
            'target_tg': target_tg,
            'acid': row['acid'],
            'epoxide': row['epoxide'],
            'tg_raw': row['tg'],
            'tg_calibrated': row['tg_calibrated'],
            'uncertainty': row['tg_cal_uncertainty']
        })

cand_df = pd.DataFrame(candidates)
print(f"\nTotal candidates generated: {len(cand_df)}")

# Visualize candidates
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(vit_df['tg_calibrated'], vit_df['tg_cal_uncertainty'], 
         c='lightgray', alpha=0.3, s=5, label='All systems')

colors = {'Low Tg (flexible)': 'blue', 'Medium Tg (general)': 'green', 'High Tg (rigid)': 'red'}
for target_name in targets.keys():
    subset = cand_df[cand_df['target_category'] == target_name]
    if len(subset) > 0:
        ax.scatter(subset['tg_calibrated'], subset['uncertainty'], 
                  s=100, label=target_name, color=colors[target_name],
                  edgecolors='black', linewidth=1.5)

for target_name, target_tg in targets.items():
    ax.axvline(target_tg, color='red', linestyle='--', alpha=0.5)

ax.set_xlabel('Calibrated Tg (K)')
ax.set_ylabel('Uncertainty (K)')
ax.set_title('Candidate Selection in Property Space')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for target_name in targets.keys():
    subset = cand_df[cand_df['target_category'] == target_name]
    if len(subset) > 0:
        ax.scatter(range(len(subset)), subset['tg_calibrated'], 
                  s=100, label=target_name, color=colors[target_name],
                  edgecolors='black', linewidth=1.5)

for target_name, target_tg in targets.items():
    ax.axhline(target_tg, color='red', linestyle='--', alpha=0.3)

ax.set_xlabel('Candidate Index')
ax.set_ylabel('Calibrated Tg (K)')
ax.set_title('Selected Candidates vs Target')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/candidate_generation.png', dpi=150, bbox_inches='tight')
print("\nSaved: report/images/candidate_generation.png")
plt.close()

cand_df.to_csv('outputs/candidates.csv', index=False)
print("Saved: outputs/candidates.csv")

# Print candidate summary
print("\n" + "="*60)
print("CANDIDATE SUMMARY")
print("="*60)
for target_name in targets.keys():
    subset = cand_df[cand_df['target_category'] == target_name]
    print(f"\n{target_name}:")
    print(f"  Count: {len(subset)}")
    if len(subset) > 0:
        print(f"  Mean Tg: {subset['tg_calibrated'].mean():.1f} K")
        print(f"  Mean uncertainty: {subset['uncertainty'].mean():.1f} K")
