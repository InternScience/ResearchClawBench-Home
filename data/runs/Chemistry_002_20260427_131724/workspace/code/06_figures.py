"""All figures for the report. PNG outputs to report/images/."""
import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

os.makedirs('report/images', exist_ok=True)

D = np.load('outputs/structure.npz', allow_pickle=True)
chains = D['chains']; coords=D['coords'].astype(np.float64)
resi=D['resi']; res=D['res']; name=D['name']
iA = chains=='A'; iD = chains=='D'
cA=coords[iA]; cD=coords[iD]
ca_a = (name[iA]=='CA'); ca_d = (name[iD]=='CA')
cA_ca = cA[ca_a]; cD_ca = cD[ca_d]
resA = resi[iA][ca_a]; resD = resi[iD][ca_d]

AIRS = json.load(open('outputs/airs.json'))
active_A = set(AIRS['active_chainA_barnase'])
active_D = set(AIRS['active_chainD_barstar'])
maskA = np.array([r in active_A for r in resA])
maskD = np.array([r in active_D for r in resD])

# ------------ Figure 1: data overview --------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axes[0]
ax.scatter(cA_ca[:,0], cA_ca[:,1], s=22, c='steelblue', label=f'Barnase (chain A, {len(cA_ca)} res)', alpha=0.75)
ax.scatter(cD_ca[:,0], cD_ca[:,1], s=22, c='orangered', label=f'Barstar (chain D, {len(cD_ca)} res)', alpha=0.75)
ax.scatter(cA_ca[maskA,0], cA_ca[maskA,1], s=85, facecolors='none', edgecolors='navy', lw=1.7, label=f'Active barnase ({maskA.sum()})')
ax.scatter(cD_ca[maskD,0], cD_ca[maskD,1], s=85, facecolors='none', edgecolors='darkred', lw=1.7, label=f'Active barstar ({maskD.sum()})')
ax.set_title('A. 1BRS bound complex (Cα projection, x–y)')
ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)'); ax.set_aspect('equal'); ax.legend(fontsize=8)

ax = axes[1]
sk = pd.read_csv('outputs/skempi_1brs.csv')
ax.hist(sk['ddG'], bins=22, color='steelblue', edgecolor='black')
ax.axvline(0, color='k', lw=1, ls='--')
ax.set_xlabel('ΔΔG (kcal/mol)'); ax.set_ylabel('count')
ax.set_title(f'B. SKEMPI 2.0 ΔΔG distribution\nfor 1BRS_A_D (n={len(sk)} mutations,\n{(sk.n_muts==1).sum()} single mutants)')

ax = axes[2]
agg = pd.read_csv('outputs/skempi_1brs_perresidue.csv')
agg['label'] = agg['chain']+agg['wt']+agg['resi'].astype(str)
agg = agg.sort_values('mean_ddG', ascending=False)
colors = ['#d35400' if c=='D' else '#2980b9' for c in agg['chain']]
ax.barh(agg['label'], agg['mean_ddG'], color=colors)
ax.invert_yaxis()
ax.set_xlabel('Mean ΔΔG (kcal/mol)')
ax.set_title('C. Per-residue mean ΔΔG\n(blue=barnase, orange=barstar)')
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig('report/images/fig1_data_overview.png', dpi=150)
plt.close()
print('saved fig1')

# ------------ Figure 2: AIR / interface map --------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.2,1]))
ax = axes[0]
ax.scatter(cA_ca[:,0], cA_ca[:,2], s=22, c='steelblue', alpha=0.6, label='Barnase Cα')
ax.scatter(cD_ca[:,0], cD_ca[:,2], s=22, c='orangered', alpha=0.6, label='Barstar Cα')
ax.scatter(cA_ca[maskA,0], cA_ca[maskA,2], s=80, facecolors='none', edgecolors='navy', lw=1.7)
ax.scatter(cD_ca[maskD,0], cD_ca[maskD,2], s=80, facecolors='none', edgecolors='darkred', lw=1.7)
for i,r in enumerate(resA):
    if r not in active_A: continue
    j = np.argmin(np.linalg.norm(cD_ca[maskD] - cA_ca[i], axis=1))
    src = cA_ca[i]; dst = cD_ca[maskD][j]
    ax.plot([src[0],dst[0]],[src[2],dst[2]], color='gray', lw=0.6, alpha=0.7)
ax.set_aspect('equal'); ax.legend(fontsize=9)
ax.set_xlabel('x (Å)'); ax.set_ylabel('z (Å)')
ax.set_title('A. AIR network across the barnase–barstar interface\n(grey lines = AIR pseudo-bonds; closed circles = active residues)')

ax = axes[1]
ax.axis('off')
text = ['Active residues (used to drive docking):',
        '',
        'Barnase chain A ({}):'.format(len(active_A)),
        '  ' + ', '.join(str(r) for r in sorted(active_A)),
        '',
        'Barstar chain D ({}):'.format(len(active_D)),
        '  ' + ', '.join(str(r) for r in sorted(active_D)),
        '',
        f'Passive barnase ({len(AIRS["passive_chainA_barnase"])}): residues within',
        '  6.5 Å of an active barnase residue',
        f'Passive barstar ({len(AIRS["passive_chainD_barstar"])}): residues within',
        '  6.5 Å of an active barstar residue',
        '',
        'Definition: an active residue has any heavy atom',
        'within 5.0 Å of the partner chain in the bound',
        '1BRS structure. AIRs use the HADDOCK soft',
        'minimum-distance (effective distance) formalism.',
        ]
ax.text(0.0, 0.95, '\n'.join(text), va='top', ha='left', fontsize=10, family='monospace')
ax.set_title('B. AIR set composition')
plt.tight_layout()
plt.savefig('report/images/fig2_air_definition.png', dpi=150)
plt.close()
print('saved fig2')

# ------------ Figure 3: HADDOCK score / RMSD funnel ------
poses = pd.read_csv('outputs/poses.csv')
clusters = pd.read_csv('outputs/clusters.csv')
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axes[0]
sc = ax.scatter(poses['lrmsd'], poses['score'], s=22, c=poses['Eair'], cmap='viridis', edgecolor='black', lw=0.2)
cb = plt.colorbar(sc, ax=ax); cb.set_label('E_AIR (Å²)')
ax.set_xlabel('Ligand RMSD to bound (Å)')
ax.set_ylabel('HADDOCK score')
ax.set_title('A. Score vs ligand-RMSD funnel\n(N=120 rigid-body poses)')
ax.axvline(4.0, color='red', ls='--', lw=1, label='4 Å acceptable cutoff')
ax.legend()

ax = axes[1]
ax.scatter(poses['irmsd'], poses['score'], s=22, c='steelblue', edgecolor='black', lw=0.2)
ax.set_xlabel('Interface RMSD to bound (Å)')
ax.set_ylabel('HADDOCK score')
ax.set_title('B. Score vs interface-RMSD')
ax.axvline(2.0, color='red', ls='--', lw=1, label='CAPRI medium (2 Å)')
ax.axvline(4.0, color='orange', ls='--', lw=1, label='CAPRI acceptable (4 Å)')
ax.legend(fontsize=8)

ax = axes[2]
top10 = clusters.head(10)
xpos = np.arange(len(top10))
ax.bar(xpos, top10['best_score'], color='steelblue', label='best HADDOCK score')
ax.set_xticks(xpos); ax.set_xticklabels(['C'+str(c) for c in top10['cluster']], rotation=45)
ax.set_ylabel('Best HADDOCK score')
ax2 = ax.twinx()
ax2.plot(xpos, top10['best_irmsd'], 'ro-', label='best i-RMSD')
ax2.plot(xpos, top10['best_lrmsd'], 'gs-', label='best L-RMSD')
ax2.set_ylabel('RMSD (Å)')
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lab1+lab2, fontsize=8, loc='upper left')
ax.set_title(f'C. Top 10 clusters\n({len(clusters)} clusters total, single-linkage L-RMSD ≤ 7.5 Å)')
plt.tight_layout()
plt.savefig('report/images/fig3_score_funnel.png', dpi=150)
plt.close()
print('saved fig3')

# ------------ Figure 4: predicted vs reference ------
T = np.load('outputs/top1_pose.npz')
cD_top1 = T['cD']
top1_score = poses.iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
ax.scatter(cA_ca[:,0], cA_ca[:,1], s=22, c='steelblue', alpha=0.7, label='Barnase (fixed)')
ax.scatter(cD_ca[:,0], cD_ca[:,1], s=22, c='orangered', alpha=0.7, label='Barstar (bound reference)')
cD_top1_ca = cD_top1[name[iD]=='CA']
ax.scatter(cD_top1_ca[:,0], cD_top1_ca[:,1], s=22, c='green', alpha=0.7, label='Barstar (top1 prediction)')
# draw segments between corresponding Cα atoms
for i in range(len(cD_ca)):
    ax.plot([cD_ca[i,0], cD_top1_ca[i,0]], [cD_ca[i,1], cD_top1_ca[i,1]], color='gray', lw=0.3, alpha=0.5)
ax.set_aspect('equal'); ax.legend()
ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
ax.set_title(f'A. Top1 predicted complex (green) vs reference (orange)\nL-RMSD={top1_score["lrmsd"]:.2f} Å, i-RMSD={top1_score["irmsd"]:.2f} Å, HS={top1_score["score"]:.1f}')

ax = axes[1]
b = pd.read_csv('outputs/per_residue_descriptors_bound.csv')
p = pd.read_csv('outputs/per_residue_descriptors_top1.csv')
m = b.merge(p, on=['chain','resi','aa'], suffixes=('_bound','_pred'))
ax.scatter(m['HSres_bound'], m['HSres_pred'],
           c=['#d35400' if c=='D' else '#2980b9' for c in m['chain']], s=28, edgecolor='black', lw=0.3)
lim = [min(m['HSres_bound'].min(), m['HSres_pred'].min())-1, max(m['HSres_bound'].max(), m['HSres_pred'].max())+1]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Per-residue HS contribution (bound, kcal/mol)')
ax.set_ylabel('Per-residue HS contribution (predicted top1, kcal/mol)')
sR,_ = spearmanr(m['HSres_bound'], m['HSres_pred'])
pR,_ = pearsonr(m['HSres_bound'], m['HSres_pred'])
ax.set_title(f'B. Per-residue interface energy bound vs predicted\nSpearman={sR:.2f}, Pearson={pR:.2f}, n={len(m)}')
plt.tight_layout()
plt.savefig('report/images/fig4_predicted_vs_reference.png', dpi=150)
plt.close()
print('saved fig4')

# ------------ Figure 5: SKEMPI validation ------
stats = json.load(open('outputs/validation_stats.json'))
mb = pd.read_csv('outputs/skempi_vs_bound_descriptors.csv').dropna(subset=['HSres','mean_ddG'])
mp = pd.read_csv('outputs/skempi_vs_pred_descriptors.csv').dropna(subset=['HSres','mean_ddG'])
pmb = pd.read_csv('outputs/skempi_per_mutation_bound.csv').dropna(subset=['n_close_partner','ddG'])
pmp = pd.read_csv('outputs/skempi_per_mutation_pred.csv').dropna(subset=['n_close_partner','ddG'])

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# A) Per-residue n_close_partner vs mean ddG (bound)
ax = axes[0,0]
colors = ['#d35400' if c=='D' else '#2980b9' for c in mb['chain']]
ax.scatter(mb['n_close_partner'], mb['mean_ddG'], c=colors, s=70, edgecolor='black')
for _, row in mb.iterrows():
    ax.annotate(f"{row['chain']}{row['wt']}{row['resi']}", (row['n_close_partner'], row['mean_ddG']),
                fontsize=7, xytext=(3,3), textcoords='offset points')
sR = stats['per_residue_bound_meanDDG']['n_close_partner']['spearman']
pR = stats['per_residue_bound_meanDDG']['n_close_partner']['pearson']
ax.set_xlabel('# distinct partner residues within 5 Å (bound)')
ax.set_ylabel('Mean ΔΔG (kcal/mol)')
ax.set_title(f'A. Bound per-residue\nSpearman={sR:+.2f}, Pearson={pR:+.2f}')

# B) Same but predicted top1
ax = axes[0,1]
colors = ['#d35400' if c=='D' else '#2980b9' for c in mp['chain']]
ax.scatter(mp['n_close_partner'], mp['mean_ddG'], c=colors, s=70, edgecolor='black')
for _, row in mp.iterrows():
    ax.annotate(f"{row['chain']}{row['wt']}{row['resi']}", (row['n_close_partner'], row['mean_ddG']),
                fontsize=7, xytext=(3,3), textcoords='offset points')
sR = stats['per_residue_pred_meanDDG']['n_close_partner']['spearman']
pR = stats['per_residue_pred_meanDDG']['n_close_partner']['pearson']
ax.set_xlabel('# distinct partner residues within 5 Å (predicted top1)')
ax.set_ylabel('Mean ΔΔG (kcal/mol)')
ax.set_title(f'B. Predicted top1 per-residue\nSpearman={sR:+.2f}, Pearson={pR:+.2f}')

# C) Per-mutation n_close_partner vs ddG (predicted top1)
ax = axes[0,2]
colors = ['#d35400' if c=='D' else '#2980b9' for c in pmp['chain']]
ax.scatter(pmp['n_close_partner']+np.random.uniform(-0.15,0.15,len(pmp)), pmp['ddG'], c=colors, s=40, edgecolor='black', alpha=0.85)
sR = stats['per_mutation_pred']['n_close_partner']['spearman']
pR = stats['per_mutation_pred']['n_close_partner']['pearson']
ax.set_xlabel('# partner residues within 5 Å (predicted top1)')
ax.set_ylabel('ΔΔG per single mutation (kcal/mol)')
ax.set_title(f'C. Per-mutation predicted top1 (n={len(pmp)})\nSpearman={sR:+.2f}, Pearson={pR:+.2f}')

# D) Per-mutation E_elec vs ddG (predicted top1)
ax = axes[1,0]
colors = ['#d35400' if c=='D' else '#2980b9' for c in pmp['chain']]
ax.scatter(pmp['E_elec'], pmp['ddG'], c=colors, s=40, edgecolor='black')
sR = stats['per_mutation_pred']['E_elec']['spearman']
pR = stats['per_mutation_pred']['E_elec']['pearson']
ax.set_xlabel('Per-residue E_elec (predicted top1, kcal/mol)')
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title(f'D. Per-mutation E_elec\nSpearman={sR:+.2f}, Pearson={pR:+.2f}')

# E) Per-mutation HSres vs ddG (predicted top1)
ax = axes[1,1]
colors = ['#d35400' if c=='D' else '#2980b9' for c in pmp['chain']]
ax.scatter(pmp['HSres'], pmp['ddG'], c=colors, s=40, edgecolor='black')
sR = stats['per_mutation_pred']['HSres']['spearman']
pR = stats['per_mutation_pred']['HSres']['pearson']
ax.set_xlabel('Per-residue HADDOCK score contribution (predicted)')
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title(f'E. Per-mutation HS_res\nSpearman={sR:+.2f}, Pearson={pR:+.2f}')

# F) bar summary: predictor-by-predictor Spearman for predicted top1
ax = axes[1,2]
predictors = ['n_contacts','n_close_partner','E_vdw','E_elec','E_des','HSres']
bounds_pres   = [stats['per_residue_bound_meanDDG'][p]['spearman'] for p in predictors]
preds_pres    = [stats['per_residue_pred_meanDDG'][p]['spearman'] for p in predictors]
bounds_perm   = [stats['per_mutation_bound'][p]['spearman'] for p in predictors]
preds_perm    = [stats['per_mutation_pred'][p]['spearman'] for p in predictors]
x = np.arange(len(predictors)); w=0.2
ax.bar(x-1.5*w, bounds_pres, w, label='bound, per-residue', color='lightblue')
ax.bar(x-0.5*w, preds_pres,  w, label='predicted, per-residue', color='steelblue')
ax.bar(x+0.5*w, bounds_perm, w, label='bound, per-mutation', color='gold')
ax.bar(x+1.5*w, preds_perm,  w, label='predicted, per-mutation', color='darkorange')
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(predictors, rotation=45, fontsize=8)
ax.set_ylabel('Spearman ρ with ΔΔG')
ax.set_title('F. Predictor comparison')
ax.legend(fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig('report/images/fig5_skempi_validation.png', dpi=150)
plt.close()
print('saved fig5')
print('all figures saved')
