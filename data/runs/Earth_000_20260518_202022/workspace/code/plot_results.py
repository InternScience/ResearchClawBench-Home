"""
Generate figures for the glacial mass change report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

OUTPUT_DIR = 'report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load official global
global_df = pd.read_csv('data/glambie/results/calendar_years/0_global.csv')
global_df['year'] = global_df['start_dates'].astype(int)
global_df = global_df[(global_df['year'] >= 2000) & (global_df['year'] <= 2023)]

# Load official regional
regions = []
for fname in sorted(os.listdir('data/glambie/results/calendar_years')):
    if fname.endswith('.csv') and fname != '0_global.csv':
        df = pd.read_csv(os.path.join('data/glambie/results/calendar_years', fname))
        df['year'] = df['start_dates'].astype(int)
        df = df[(df['year'] >= 2000) & (df['year'] <= 2023)]
        regions.append(df)
regional = pd.concat(regions, ignore_index=True)

# Load our simplified reconciliation
rec_global = pd.read_csv('outputs/reconciled_global.csv')

# Figure 1: Global annual mass change (Gt/yr) with uncertainty
fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(global_df['year'], global_df['combined_gt'] - global_df['combined_gt_errors'],
                global_df['combined_gt'] + global_df['combined_gt_errors'], color='steelblue', alpha=0.3, label='Uncertainty')
ax.plot(global_df['year'], global_df['combined_gt'], color='steelblue', linewidth=1.5, label='GlaMBIE combined')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mass change (Gt yr$^{-1}$)')
ax.set_title('Global glacier mass change (2000–2023)')
ax.legend()
ax.set_xlim(2000, 2023)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_global_annual_gt.png'))
plt.close(fig)

# Figure 2: Cumulative global mass change (Gt)
global_df = global_df.sort_values('year')
global_df['cumulative_gt'] = global_df['combined_gt'].cumsum()
global_df['cumulative_gt_err'] = np.sqrt((global_df['combined_gt_errors']**2).cumsum())

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(global_df['year'], global_df['cumulative_gt'] - global_df['cumulative_gt_err'],
                global_df['cumulative_gt'] + global_df['cumulative_gt_err'], color='coral', alpha=0.3, label='Uncertainty')
ax.plot(global_df['year'], global_df['cumulative_gt'], color='coral', linewidth=1.5, label='Cumulative loss')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Cumulative mass change (Gt)')
ax.set_title('Cumulative global glacier mass loss (2000–2023)')
ax.legend()
ax.set_xlim(2000, 2023)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_global_cumulative_gt.png'))
plt.close(fig)

# Figure 3: Regional mean mass change rates (bar chart)
regional_mean = regional.groupby('region').agg(
    mean_gt=('combined_gt', 'mean'),
    mean_mwe=('combined_mwe', 'mean')
).reset_index()
regional_mean = regional_mean.sort_values('mean_gt')

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['darkred' if v < 0 else 'darkgreen' for v in regional_mean['mean_gt']]
ax.barh(regional_mean['region'], regional_mean['mean_gt'], color=colors, alpha=0.7)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Mean annual mass change (Gt yr$^{-1}$)')
ax.set_title('Mean glacier mass change by region (2000–2023)')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_regional_mean_gt.png'))
plt.close(fig)

# Figure 4: Specific mass change (m w.e.) time series for selected regions
selected = ['1_alaska', '17_southern_andes', '13_central_asia', '19_antarctic_and_subantarctic']
sub = regional[regional['region'].isin(selected)]
fig, ax = plt.subplots(figsize=(10, 5))
for reg, grp in sub.groupby('region'):
    ax.plot(grp['year'], grp['combined_mwe'], label=reg, linewidth=1.2)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Specific mass change (m w.e. yr$^{-1}$)')
ax.set_title('Specific mass change for selected regions')
ax.legend()
ax.set_xlim(2000, 2023)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_regional_specific_mwe.png'))
plt.close(fig)

# Figure 5: Validation scatter - our simplified reconciliation vs official
merged = pd.merge(global_df[['year', 'combined_gt']], rec_global[['year', 'global_gt']], on='year')
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(merged['combined_gt'], merged['global_gt'], c='steelblue', edgecolors='k', alpha=0.7)
# 1:1 line
lims = [merged[['combined_gt', 'global_gt']].min().min() - 20, merged[['combined_gt', 'global_gt']].max().max() + 20]
ax.plot(lims, lims, 'k--', linewidth=1)
ax.set_xlabel('Official GlaMBIE (Gt yr$^{-1}$)')
ax.set_ylabel('Simplified reconciliation (Gt yr$^{-1}$)')
ax.set_title('Validation: simplified vs official global estimates')
ax.set_xlim(lims)
ax.set_ylim(lims)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_validation_scatter.png'))
plt.close(fig)

# Figure 6: Time series of per-method contributions (hydrological years, convert to calendar approx)
# Use hydrological year results for a region to show method breakdown
hydro = pd.read_csv('data/glambie/results/hydrological_years/1_alaska.csv')
hydro = hydro[(hydro['start_dates'] >= 2000) & (hydro['start_dates'] <= 2023)]
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(hydro['start_dates'], 0, hydro['demdiff_and_glaciological_gt'].fillna(0), label='DEM diff + glaciological', alpha=0.7)
ax.fill_between(hydro['start_dates'], hydro['demdiff_and_glaciological_gt'].fillna(0),
                hydro['demdiff_and_glaciological_gt'].fillna(0) + hydro['altimetry_gt'].fillna(0), label='Altimetry', alpha=0.7)
ax.fill_between(hydro['start_dates'], hydro['demdiff_and_glaciological_gt'].fillna(0) + hydro['altimetry_gt'].fillna(0),
                hydro['demdiff_and_glaciological_gt'].fillna(0) + hydro['altimetry_gt'].fillna(0) + hydro['gravimetry_gt'].fillna(0), label='Gravimetry', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mass change (Gt yr$^{-1}$)')
ax.set_title('Method contributions to Alaska mass change (hydrological years)')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig6_method_contributions_alaska.png'))
plt.close(fig)

# Figure 7: Global specific mass change (m w.e.) with uncertainty
fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(global_df['year'], global_df['combined_mwe'] - global_df['combined_mwe_errors'],
                global_df['combined_mwe'] + global_df['combined_mwe_errors'], color='teal', alpha=0.3, label='Uncertainty')
ax.plot(global_df['year'], global_df['combined_mwe'], color='teal', linewidth=1.5, label='Specific mass change')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Specific mass change (m w.e. yr$^{-1}$)')
ax.set_title('Global specific glacier mass change (2000–2023)')
ax.legend()
ax.set_xlim(2000, 2023)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig7_global_specific_mwe.png'))
plt.close(fig)

print('All figures saved to', OUTPUT_DIR)
