"""Generate publication-quality figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 130,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})
sns.set_style('whitegrid')

ROOT = Path('/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_000_20260427_143635')
DATA = ROOT/'data/glambie'
OUT = ROOT/'outputs'
IMG = ROOT/'report/images'
IMG.mkdir(parents=True, exist_ok=True)

reg = pd.read_csv(OUT/'regional_annual_reconciled.csv')
glo = pd.read_csv(OUT/'global_annual_reconciled.csv')
src = pd.read_csv(OUT/'source_inventory.csv')
cmp_global = pd.read_csv(OUT/'comparison_global.csv')
cmp_regions = pd.read_csv(OUT/'comparison_vs_glambie.csv')

REGIONS = sorted(reg.region.unique(), key=lambda r: cmp_regions.loc[cmp_regions.region==r, 'region_num'].iloc[0] if (cmp_regions.region==r).any() else 99)
RNAME = {r: r.replace('_', ' ').title() for r in REGIONS}

# ---- FIG 1: Source inventory overview ----
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# (a) sources per region per method
piv = src.pivot_table(index='region', columns='method', values='file', aggfunc='count', fill_value=0)
piv = piv.reindex(REGIONS)
piv.index = [RNAME[r] for r in piv.index]
piv = piv[['glaciological','demdiff','altimetry','gravimetry','combined']]
piv.plot(kind='barh', stacked=True, ax=axes[0],
         color=['#2ca02c','#1f77b4','#9467bd','#d62728','#ff7f0e'])
axes[0].set_xlabel('Number of input sources')
axes[0].set_title('(a) Input sources per region by observational method')
axes[0].legend(loc='lower right', fontsize=8, framealpha=0.85)
axes[0].invert_yaxis()

# (b) coverage histogram
cov = src.groupby('method')['n_years_covered'].apply(list)
methods = ['glaciological','demdiff','altimetry','gravimetry','combined']
data = [cov.get(m, []) for m in methods]
axes[1].boxplot(data, labels=methods, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='#cce5ff'))
axes[1].set_ylabel('Years covered (out of 2000-2023)')
axes[1].set_title('(b) Per-source temporal coverage by method')
axes[1].set_ylim(0, 25)

plt.tight_layout()
plt.savefig(IMG/'fig01_data_overview.png')
plt.close()
print('Saved fig01')

# ---- FIG 2: Global annual mass change time series ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) annual Gt
ax = axes[0]
ax.errorbar(glo.year, glo.consensus_gt, yerr=glo.consensus_gt_err,
            color='black', marker='o', label='This study (reconciled)', lw=2, ms=4)
ax.errorbar(cmp_global.year, cmp_global.glambie_gt, yerr=cmp_global.glambie_gt_err,
            color='tab:red', marker='s', alpha=0.8, label='GlaMBIE (2024) consensus', lw=1.5, ms=3)
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Annual mass change (Gt yr$^{-1}$)')
ax.set_title('(a) Global annual glacial mass change, 2000-2023')
ax.legend(loc='lower left', fontsize=9)

# (b) cumulative
ax = axes[1]
ax.fill_between(glo.year, glo.cum_gt - glo.cum_gt_err, glo.cum_gt + glo.cum_gt_err,
                color='black', alpha=0.18)
ax.plot(glo.year, glo.cum_gt, color='black', lw=2, marker='o', ms=4, label='This study (reconciled)')
g_off = cmp_global.copy()
g_off['cum_glambie_gt'] = g_off.glambie_gt.cumsum()
g_off['cum_glambie_err'] = np.sqrt((g_off.glambie_gt_err**2).cumsum())
ax.fill_between(g_off.year, g_off.cum_glambie_gt - g_off.cum_glambie_err,
                            g_off.cum_glambie_gt + g_off.cum_glambie_err,
                color='tab:red', alpha=0.15)
ax.plot(g_off.year, g_off.cum_glambie_gt, color='tab:red', lw=1.8, marker='s', ms=3, label='GlaMBIE (2024) consensus')
ax.axhline(0, color='gray', lw=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Cumulative mass change (Gt)')
ax.set_title('(b) Cumulative global glacial mass change')
ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig(IMG/'fig02_global_timeseries.png')
plt.close()
print('Saved fig02')

# ---- FIG 3: Per-region annual time series (small multiples) ----
fig, axes = plt.subplots(5, 4, figsize=(18, 18), sharex=True)
axes = axes.flatten()
for i, r in enumerate(REGIONS):
    ax = axes[i]
    sub = reg[reg.region==r].sort_values('year')
    ax.fill_between(sub.year, sub.consensus_mwe - sub.consensus_mwe_err,
                              sub.consensus_mwe + sub.consensus_mwe_err,
                    color='black', alpha=0.18)
    ax.plot(sub.year, sub.consensus_mwe, color='black', lw=1.6, label='Reconciled')
    # overlay official
    rdir_num = cmp_regions.loc[cmp_regions.region==r, 'region_num'].iloc[0]
    off_path = DATA/'results/calendar_years'/f'{rdir_num}_{r}.csv'
    if off_path.exists():
        of = pd.read_csv(off_path); of['year']=of.start_dates.astype(int)
        of = of[of.year<=2023]
        ax.plot(of.year, of.combined_mwe, color='tab:red', lw=1.4, ls='--', label='GlaMBIE')
    # group means
    for col, c in [('altimetry_mwe','tab:blue'),('gravimetry_mwe','tab:purple'),
                   ('demdiff_glaciological_mwe','tab:green')]:
        if col in sub:
            ax.plot(sub.year, sub[col], color=c, lw=0.9, alpha=0.6)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'R{rdir_num}. {RNAME[r]}', fontsize=10)
    if i % 4 == 0: ax.set_ylabel('m w.e. yr$^{-1}$')
    if i >= 16: ax.set_xlabel('Year')
for j in range(len(REGIONS), len(axes)):
    axes[j].axis('off')
# Shared legend
handles = [
    plt.Line2D([],[],color='black', lw=1.6, label='Reconciled consensus'),
    plt.Line2D([],[],color='tab:red', lw=1.4, ls='--', label='GlaMBIE consensus'),
    plt.Line2D([],[],color='tab:blue', lw=0.9, alpha=0.7, label='Altimetry group'),
    plt.Line2D([],[],color='tab:purple',lw=0.9, alpha=0.7, label='Gravimetry group'),
    plt.Line2D([],[],color='tab:green', lw=0.9, alpha=0.7, label='DEMdiff+Glaciol. group'),
]
fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5,-0.005), fontsize=10)
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(IMG/'fig03_regional_timeseries.png')
plt.close()
print('Saved fig03')

# ---- FIG 4: Per-region cumulative bar comparison ----
fig, ax = plt.subplots(1, 1, figsize=(11, 6))
df = cmp_regions.sort_values('cumulative_glambie_gt')
y = np.arange(len(df))
ax.barh(y - 0.2, df.cumulative_glambie_gt, height=0.4, color='tab:red', alpha=0.85, label='GlaMBIE (2024)')
ax.barh(y + 0.2, df.cumulative_reconciled_gt, height=0.4, color='steelblue', alpha=0.9, label='This study (reconciled)')
ax.set_yticks(y)
ax.set_yticklabels([RNAME[r] for r in df.region])
ax.set_xlabel('Cumulative mass change 2000-2023 (Gt)')
ax.set_title('Cumulative regional glacial mass change, 2000-2023')
ax.axvline(0, color='gray', lw=0.5)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(IMG/'fig04_regional_cumulative.png')
plt.close()
print('Saved fig04')

# ---- FIG 5: Method-group spread per region (mean annual rate, m w.e./yr) ----
# For each region compute mean rate per group, plot spread
fig, ax = plt.subplots(1, 1, figsize=(13, 7))
plot_df = []
for r in REGIONS:
    sub = reg[reg.region==r]
    for g, lbl in [('altimetry_mwe','Altimetry'),('gravimetry_mwe','Gravimetry'),
                   ('demdiff_glaciological_mwe','DEMdiff+Glaciol.')]:
        if g in sub:
            v = sub[g].mean()
            plot_df.append(dict(region=RNAME[r], group=lbl, rate=v))
    plot_df.append(dict(region=RNAME[r], group='Consensus', rate=sub.consensus_mwe.mean()))
pdf = pd.DataFrame(plot_df)
order = [RNAME[r] for r in REGIONS]
sns.barplot(data=pdf, x='region', y='rate', hue='group', ax=ax, order=order,
            palette={'Altimetry':'tab:blue','Gravimetry':'tab:purple','DEMdiff+Glaciol.':'tab:green','Consensus':'black'})
ax.axhline(0, color='gray', lw=0.6)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Mean specific mass change 2000-2023 (m w.e. yr$^{-1}$)')
ax.set_xlabel('')
ax.set_title('Method-group reconciliation per region: 2000-2023 mean specific mass change')
ax.legend(title='Group', loc='lower right')
plt.tight_layout()
plt.savefig(IMG/'fig05_method_group_per_region.png')
plt.close()
print('Saved fig05')

# ---- FIG 6: Validation scatter & error bars ----
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
m = cmp_global
ax.errorbar(m.glambie_gt, m.reconciled_gt,
            xerr=m.glambie_gt_err, yerr=m.reconciled_gt_err,
            fmt='o', color='steelblue', alpha=0.75, ecolor='gray', capsize=2)
lim = [-650, 100]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('GlaMBIE consensus annual mass change (Gt)')
ax.set_ylabel('Reconciled annual mass change (Gt)')
ax.set_title('(a) Global annual mass change: this study vs GlaMBIE')
# add R^2
import scipy.stats as st
sl, ic, rval, _, _ = st.linregress(m.glambie_gt, m.reconciled_gt)
ax.text(0.04, 0.94, f'$R^2$={rval**2:.3f}\nslope={sl:.2f}', transform=ax.transAxes, va='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax = axes[1]
df = cmp_regions.copy().sort_values('region_num')
ax.scatter(df.cumulative_glambie_gt, df.cumulative_reconciled_gt, c='tab:purple', s=70, alpha=0.85)
for _, r in df.iterrows():
    ax.annotate(f"R{int(r.region_num)}", (r.cumulative_glambie_gt, r.cumulative_reconciled_gt),
                fontsize=7, xytext=(3,3), textcoords='offset points')
lim = [-1700, 100]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('GlaMBIE cumulative 2000-2023 (Gt)')
ax.set_ylabel('Reconciled cumulative 2000-2023 (Gt)')
ax.set_title('(b) Per-region cumulative agreement')
sl, ic, rval, _, _ = st.linregress(df.cumulative_glambie_gt, df.cumulative_reconciled_gt)
ax.text(0.04, 0.94, f'$R^2$={rval**2:.3f}\nslope={sl:.2f}', transform=ax.transAxes, va='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.savefig(IMG/'fig06_validation.png')
plt.close()
print('Saved fig06')

# ---- FIG 7: Heatmap of annual specific mass change m w.e./yr per region/year ----
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
mat = reg.pivot(index='region', columns='year', values='consensus_mwe')
mat = mat.reindex(REGIONS)
mat.index = [RNAME[r] for r in mat.index]
sns.heatmap(mat, cmap='RdBu', center=0, vmin=-2, vmax=2, ax=ax,
            cbar_kws={'label':'Specific mass change (m w.e. yr$^{-1}$)'})
ax.set_title('Reconciled annual specific mass change by region')
ax.set_xlabel('Year')
ax.set_ylabel('Region')
plt.tight_layout()
plt.savefig(IMG/'fig07_regional_heatmap.png')
plt.close()
print('Saved fig07')

# ---- FIG 8: Sea level equivalent contribution ----
# Convert Gt to mm SLE: 1 mm SLE = 360 Gt
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
glo['sle_mm_yr'] = -glo.consensus_gt / 360.0
glo['cum_sle_mm'] = glo['sle_mm_yr'].cumsum()
ax.bar(glo.year, glo.sle_mm_yr, color='steelblue', alpha=0.7, label='Annual contribution (mm SLE yr$^{-1}$)')
ax2 = ax.twinx()
ax2.plot(glo.year, glo.cum_sle_mm, color='black', marker='o', ms=4, label='Cumulative')
ax.set_xlabel('Year')
ax.set_ylabel('Annual sea level equivalent (mm yr$^{-1}$)')
ax2.set_ylabel('Cumulative sea level equivalent (mm)')
ax.set_title('Reconciled global glacier contribution to sea level rise (2000-2023)')
ax.legend(loc='upper left'); ax2.legend(loc='lower right')
plt.tight_layout()
plt.savefig(IMG/'fig08_sea_level_equivalent.png')
plt.close()
print('Saved fig08')

# Save SLE summary
glo[['year','consensus_gt','consensus_gt_err','consensus_mwe','consensus_mwe_err',
     'cum_gt','cum_gt_err','cum_mwe','cum_mwe_err','sle_mm_yr','cum_sle_mm']].to_csv(OUT/'global_annual_reconciled_with_sle.csv', index=False)

print('All figures saved to', IMG)
