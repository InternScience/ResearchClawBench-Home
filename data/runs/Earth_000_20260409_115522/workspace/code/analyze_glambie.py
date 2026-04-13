import os
import re
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

BASE = Path('.')
INPUT_DIR = BASE / 'data' / 'glambie' / 'input'
RESULTS_DIR = BASE / 'data' / 'glambie' / 'results' / 'calendar_years'
OUT_DIR = BASE / 'outputs'
FIG_DIR = BASE / 'report' / 'images'
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

METHODS = ['glaciological', 'demdiff', 'altimetry', 'gravimetry', 'combined']
METHOD_LABELS = {
    'glaciological': 'Glaciological',
    'demdiff': 'DEM differencing',
    'altimetry': 'Altimetry',
    'gravimetry': 'Gravimetry',
    'combined': 'Hybrid / combined',
}
REGION_LABELS = {
    'alaska':'Alaska',
    'western_canada_us':'Western Canada & US',
    'arctic_canada_north':'Arctic Canada North',
    'arctic_canada_south':'Arctic Canada South',
    'greenland_periphery':'Greenland Periphery',
    'iceland':'Iceland',
    'svalbard':'Svalbard',
    'scandinavia':'Scandinavia',
    'russian_arctic':'Russian Arctic',
    'north_asia':'North Asia',
    'central_europe':'Central Europe',
    'caucasus_middle_east':'Caucasus & Middle East',
    'central_asia':'Central Asia',
    'south_asia_west':'South Asia West',
    'south_asia_east':'South Asia East',
    'low_latitudes':'Low Latitudes',
    'southern_andes':'Southern Andes',
    'new_zealand':'New Zealand',
    'antarctic_and_subantarctic':'Antarctic & Subantarctic',
    'global':'Global',
}


def parse_input_metadata():
    rows = []
    for f in glob.glob(str(INPUT_DIR / '*' / '*.csv')):
        base = os.path.basename(f).replace('.csv', '')
        m = re.match(r'(.+?)_(altimetry|combined|demdiff|glaciological|gravimetry)_(.+)', base)
        if not m:
            continue
        region, method, dataset_name = m.groups()
        df = pd.read_csv(f)
        unit = str(df['unit'].iloc[0]).lower()
        if unit == 'gt':
            unit = 'Gt'
        rows.append({
            'file': f,
            'region': region,
            'method': method,
            'dataset_name': dataset_name,
            'author': str(df['author'].iloc[0]),
            'n_obs': len(df),
            'start_min': df['start_dates'].min(),
            'end_max': df['end_dates'].max(),
            'unit': unit,
            'mean_period_years': (df['end_dates'] - df['start_dates']).mean(),
        })
    meta = pd.DataFrame(rows)
    meta['region_label'] = meta['region'].map(REGION_LABELS).fillna(meta['region'])
    meta['method_label'] = meta['method'].map(METHOD_LABELS)
    return meta


def load_results():
    rows = []
    for f in sorted(glob.glob(str(RESULTS_DIR / '*.csv'))):
        df = pd.read_csv(f)
        region = df['region'].iloc[0]
        df['year'] = df['start_dates'].round().astype(int)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out['region_label'] = out['region'].map(REGION_LABELS).fillna(out['region'])
    return out


def compute_summaries(meta, results):
    coverage = meta.groupby(['region', 'method']).size().unstack(fill_value=0).reindex(columns=METHODS, fill_value=0)
    coverage['total'] = coverage.sum(axis=1)
    coverage = coverage.reset_index()
    coverage['region_label'] = coverage['region'].map(REGION_LABELS).fillna(coverage['region'])

    annual = results.copy()
    annual['abs_gt'] = annual['combined_gt'].abs()

    region_stats = annual[annual['region'] != 'global'].groupby(['region', 'region_label']).agg(
        mean_gt=('combined_gt', 'mean'),
        mean_mwe=('combined_mwe', 'mean'),
        total_gt=('combined_gt', 'sum'),
        total_gt_unc_ss=('combined_gt_errors', lambda x: np.sqrt(np.sum(np.square(x)))),
        total_mwe=('combined_mwe', 'sum'),
        years=('year', 'nunique'),
        mean_abs_gt=('abs_gt', 'mean'),
        area_2000=('glacier_area', 'first'),
        area_2023=('glacier_area', 'last'),
    ).reset_index()
    global_total_abs = abs(annual[annual['region'] == 'global']['combined_gt'].sum())
    region_stats['share_of_global_loss_pct'] = 100 * region_stats['total_gt'].abs() / global_total_abs
    region_stats = region_stats.sort_values('total_gt')

    g = annual[annual['region'] == 'global']
    global_stats = {
        'total_gt': float(g['combined_gt'].sum()),
        'total_gt_unc_ss': float(np.sqrt(np.sum(np.square(g['combined_gt_errors'])))),
        'mean_gt': float(g['combined_gt'].mean()),
        'total_mwe': float(g['combined_mwe'].sum()),
        'mean_mwe': float(g['combined_mwe'].mean()),
    }

    return coverage, region_stats, global_stats


def plot_data_overview(coverage):
    plot_df = coverage.melt(id_vars=['region','region_label','total'], value_vars=METHODS, var_name='method', value_name='n_datasets')
    plot_df['method_label'] = plot_df['method'].map(METHOD_LABELS)
    order = coverage.sort_values('total', ascending=False)['region_label']
    heat = plot_df.pivot(index='region_label', columns='method_label', values='n_datasets').loc[order]
    plt.figure(figsize=(10, 8))
    sns.heatmap(heat, annot=True, fmt='g', cmap='Blues', cbar_kws={'label': 'Number of submitted datasets'})
    plt.xlabel('Observation method')
    plt.ylabel('Glacier region')
    plt.title('Observational coverage across the 19 GlaMBIE regions')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_data_coverage_heatmap.png', dpi=200)
    plt.close()


def plot_global_timeseries(results):
    g = results[results['region'] == 'global'].copy()
    cum = g['combined_gt'].cumsum()
    cum_unc = np.sqrt(np.cumsum(np.square(g['combined_gt_errors'])))
    plt.figure(figsize=(10, 5.5))
    plt.plot(g['year'], cum, color='navy', lw=2.5, label='Cumulative mass change')
    plt.fill_between(g['year'], cum - cum_unc, cum + cum_unc, color='skyblue', alpha=0.35, label='Propagated uncertainty')
    plt.axhline(0, color='black', lw=0.8)
    plt.ylabel('Cumulative mass change (Gt)')
    plt.xlabel('Year')
    plt.title('Global glacier mass change, 2000–2023')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_global_cumulative_mass_change.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    colors = np.where(g['combined_gt'] < 0, '#c0392b', '#2980b9')
    plt.bar(g['year'], g['combined_gt'], color=colors, width=0.8)
    plt.errorbar(g['year'], g['combined_gt'], yerr=g['combined_gt_errors'], fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
    plt.axhline(0, color='black', lw=0.8)
    plt.ylabel('Annual mass change (Gt yr$^{-1}$)')
    plt.xlabel('Year')
    plt.title('Annual global glacier mass change with 1σ uncertainties')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_global_annual_mass_change.png', dpi=200)
    plt.close()


def plot_regional_contributions(region_stats):
    top = region_stats.sort_values('total_gt').head(10).copy()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x='total_gt', y='region_label', color='#4472c4')
    plt.axvline(0, color='black', lw=0.8)
    plt.xlabel('Cumulative mass change, 2000–2023 (Gt)')
    plt.ylabel('Region')
    plt.title('Largest regional contributors to cumulative glacier mass loss')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_regional_cumulative_contributions.png', dpi=200)
    plt.close()


def plot_specific_vs_total(region_stats):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=region_stats, x='total_gt', y='total_mwe', size='area_2000', hue='share_of_global_loss_pct', sizes=(40, 450), palette='viridis')
    for _, r in region_stats.iterrows():
        if r['share_of_global_loss_pct'] >= 4 or r['region'] in ['iceland','central_europe','south_asia_east']:
            plt.text(r['total_gt'], r['total_mwe'], ' ' + r['region_label'], fontsize=8)
    plt.axvline(0, color='black', lw=0.8)
    plt.axhline(0, color='black', lw=0.8)
    plt.xlabel('Cumulative total mass change (Gt, 2000–2023)')
    plt.ylabel('Cumulative specific mass change (m w.e., 2000–2023)')
    plt.title('Regional specific vs total glacier mass change')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_specific_vs_total_change.png', dpi=200)
    plt.close()


def plot_validation(results):
    regional = results[results['region'] != 'global'].copy()
    sums = regional.groupby('year', as_index=False)[['combined_gt']].sum().rename(columns={'combined_gt':'sum_regions_gt'})
    sums_err = regional.groupby('year', as_index=False).agg(sum_regions_unc=('combined_gt_errors', lambda x: np.sqrt(np.sum(np.square(x)))))
    global_df = results[results['region'] == 'global'][['year','combined_gt','combined_gt_errors']].rename(columns={'combined_gt':'global_gt','combined_gt_errors':'global_unc'})
    cmp = global_df.merge(sums, on='year').merge(sums_err, on='year')
    cmp['difference_gt'] = cmp['global_gt'] - cmp['sum_regions_gt']
    cmp['difference_norm_sigma'] = cmp['difference_gt'] / np.sqrt(cmp['global_unc']**2 + cmp['sum_regions_unc']**2)
    cmp.to_csv(OUT_DIR / 'validation_global_vs_sum_regions.csv', index=False)

    plt.figure(figsize=(6,6))
    lim1 = min(cmp[['global_gt','sum_regions_gt']].min()) - 20
    lim2 = max(cmp[['global_gt','sum_regions_gt']].max()) + 20
    plt.scatter(cmp['sum_regions_gt'], cmp['global_gt'], s=55, color='#2e86c1')
    plt.plot([lim1, lim2], [lim1, lim2], '--', color='black', lw=1)
    for _, r in cmp.iterrows():
        if r['year'] in [2001, 2005, 2010, 2015, 2020, 2023]:
            plt.text(r['sum_regions_gt'], r['global_gt'], str(int(r['year'])), fontsize=8)
    plt.xlabel('Sum of regional annual mass change (Gt)')
    plt.ylabel('Published global annual mass change (Gt)')
    plt.title('Validation of global aggregation against regional sum')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure_validation_global_vs_sum_regions.png', dpi=200)
    plt.close()

    return cmp


def save_tables(meta, coverage, region_stats, results, global_stats, validation):
    meta.to_csv(OUT_DIR / 'input_dataset_inventory.csv', index=False)
    coverage.to_csv(OUT_DIR / 'regional_dataset_counts.csv', index=False)
    region_stats.to_csv(OUT_DIR / 'regional_summary_statistics.csv', index=False)
    results.to_csv(OUT_DIR / 'calendar_year_results_all_regions.csv', index=False)
    pd.DataFrame([global_stats]).to_csv(OUT_DIR / 'global_summary_statistics.csv', index=False)
    validation.to_csv(OUT_DIR / 'validation_global_vs_sum_regions.csv', index=False)

    method_summary = meta.groupby('method_label').agg(
        n_datasets=('file','count'),
        n_regions=('region','nunique'),
        median_obs=('n_obs','median'),
        min_start=('start_min','min'),
        max_end=('end_max','max')
    ).reset_index().sort_values('n_datasets', ascending=False)
    method_summary.to_csv(OUT_DIR / 'method_summary.csv', index=False)


def main():
    meta = parse_input_metadata()
    results = load_results()
    coverage, region_stats, global_stats = compute_summaries(meta, results)
    plot_data_overview(coverage)
    plot_global_timeseries(results)
    plot_regional_contributions(region_stats)
    plot_specific_vs_total(region_stats)
    validation = plot_validation(results)
    save_tables(meta, coverage, region_stats, results, global_stats, validation)
    print('Analysis complete')
    print('Global total Gt:', round(global_stats['total_gt'], 3))
    print('Global total mwe:', round(global_stats['total_mwe'], 3))
    print('Validation mean abs diff Gt:', round(validation['difference_gt'].abs().mean(), 3))

if __name__ == '__main__':
    main()
