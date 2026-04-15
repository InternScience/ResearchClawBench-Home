from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'glambie'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def ensure_dirs():
    OUT.mkdir(exist_ok=True, parents=True)
    IMG.mkdir(exist_ok=True, parents=True)


def load_calendar_results():
    root = DATA / 'results' / 'calendar_years'
    frames = []
    for p in sorted(root.glob('*.csv')):
        df = pd.read_csv(p)
        df['source_file'] = p.name
        df['series_type'] = 'global' if p.name == '0_global.csv' else 'regional'
        df = df[(df['start_dates'] >= 2000) & (df['end_dates'] <= 2023)].copy()
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df['year'] = all_df['start_dates'].astype(int)
    return all_df


def load_hydrological_results():
    root = DATA / 'results' / 'hydrological_years'
    frames = []
    for p in sorted(root.glob('*.csv')):
        df = pd.read_csv(p)
        df['source_file'] = p.name
        df = df[(df['start_dates'] >= 1999.75) & (df['end_dates'] <= 2023.75)].copy()
        df['hydro_year'] = np.floor(df['start_dates'] + 0.25).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_input_coverage():
    root = DATA / 'input'
    rows = []
    for region_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        region_key = region_dir.name.split('_', 1)[1]
        for p in sorted(region_dir.glob('*.csv')):
            df = pd.read_csv(p)
            stem = p.stem
            rem = stem[len(region_key) + 1:] if stem.startswith(region_key + '_') else stem
            method = rem.split('_', 1)[0]
            rows.append({
                'region_dir': region_dir.name,
                'region': region_key,
                'file': p.name,
                'method': method,
                'n_rows': len(df),
                'min_start': float(df['start_dates'].min()),
                'max_end': float(df['end_dates'].max()),
                'unit': ','.join(sorted(df['unit'].dropna().astype(str).unique())) if 'unit' in df.columns else '',
                'author': '; '.join(sorted(df['author'].dropna().astype(str).unique())[:3]) if 'author' in df.columns else ''
            })
    return pd.DataFrame(rows)


def export_core_tables(cal_df, hyd_df, coverage_df):
    global_df = cal_df[cal_df['series_type'] == 'global'].copy().sort_values('year')
    regional_df = cal_df[cal_df['series_type'] == 'regional'].copy().sort_values(['region', 'year'])

    global_summary = global_df[['year', 'combined_gt', 'combined_gt_errors', 'combined_mwe', 'combined_mwe_errors', 'glacier_area']].copy()
    global_summary['cumulative_gt'] = global_summary['combined_gt'].cumsum()
    global_summary['cumulative_gt_error_rss'] = np.sqrt((global_summary['combined_gt_errors'] ** 2).cumsum())
    global_summary['cumulative_mwe'] = global_summary['combined_mwe'].cumsum()
    global_summary['cumulative_mwe_error_rss'] = np.sqrt((global_summary['combined_mwe_errors'] ** 2).cumsum())
    global_summary.to_csv(OUT / 'global_annual_summary.csv', index=False)
    global_summary[['year', 'cumulative_gt', 'cumulative_gt_error_rss', 'cumulative_mwe', 'cumulative_mwe_error_rss']].to_csv(OUT / 'global_cumulative_summary.csv', index=False)

    regional_summary = regional_df[['region', 'year', 'combined_gt', 'combined_gt_errors', 'combined_mwe', 'combined_mwe_errors', 'glacier_area']].copy()
    regional_summary.to_csv(OUT / 'regional_annual_summary.csv', index=False)

    reg_totals = regional_summary.groupby('region', as_index=False).agg(
        total_gt=('combined_gt', 'sum'),
        mean_annual_gt=('combined_gt', 'mean'),
        total_mwe=('combined_mwe', 'sum'),
        mean_annual_mwe=('combined_mwe', 'mean'),
        mean_area_km2=('glacier_area', 'mean'),
        years=('year', 'count')
    )
    reg_totals['rss_gt_error'] = regional_summary.groupby('region')['combined_gt_errors'].apply(lambda s: float(np.sqrt(np.sum(s ** 2)))).values
    reg_totals['rss_mwe_error'] = regional_summary.groupby('region')['combined_mwe_errors'].apply(lambda s: float(np.sqrt(np.sum(s ** 2)))).values
    reg_totals = reg_totals.sort_values('total_gt')
    reg_totals.to_csv(OUT / 'regional_2000_2023_totals.csv', index=False)

    coverage_summary = coverage_df.groupby(['region', 'method'], as_index=False).agg(
        n_series=('file', 'count'),
        total_rows=('n_rows', 'sum'),
        first_start=('min_start', 'min'),
        last_end=('max_end', 'max')
    )
    coverage_summary.to_csv(OUT / 'method_coverage_by_region.csv', index=False)

    method_cols = ['altimetry', 'gravimetry', 'demdiff_and_glaciological']
    rows = []
    for region, g in hyd_df.groupby('region'):
        for method in method_cols:
            col = f'{method}_gt'
            err = f'{method}_gt_errors'
            if col not in g.columns:
                continue
            valid = g[['hydro_year', 'combined_gt', 'combined_gt_errors', col, err]].dropna()
            if len(valid) == 0:
                continue
            diff = valid[col] - valid['combined_gt']
            rows.append({
                'region': region,
                'method': method,
                'n_years': int(len(valid)),
                'mean_method_gt': float(valid[col].mean()),
                'mean_consensus_gt': float(valid['combined_gt'].mean()),
                'mean_abs_diff_gt': float(np.mean(np.abs(diff))),
                'rmse_gt': float(np.sqrt(np.mean(diff ** 2))),
                'correlation_gt': float(valid[[col, 'combined_gt']].corr().iloc[0, 1]) if len(valid) > 1 else np.nan,
                'mean_method_error_gt': float(valid[err].mean()) if err in valid else np.nan,
            })
    method_agreement = pd.DataFrame(rows).sort_values(['region', 'method'])
    method_agreement.to_csv(OUT / 'regional_method_agreement_summary.csv', index=False)

    claim_rows = []
    worst = global_summary.loc[global_summary['combined_gt'].idxmin()]
    best = global_summary.loc[global_summary['combined_gt'].idxmax()]
    claim_rows.extend([
        {'claim_id': 'global_total_gt_2000_2022', 'claim_text': 'Total global glacier mass change over 2000-2022 from consensus calendar-year series.', 'artifact': 'outputs/global_annual_summary.csv', 'value': float(global_summary['combined_gt'].sum()), 'units': 'Gt'},
        {'claim_id': 'global_mean_gt_yr', 'claim_text': 'Mean annual global glacier mass change over 2000-2022.', 'artifact': 'outputs/global_annual_summary.csv', 'value': float(global_summary['combined_gt'].mean()), 'units': 'Gt/yr'},
        {'claim_id': 'worst_global_year', 'claim_text': 'Most negative global annual mass-change year in 2000-2022.', 'artifact': 'outputs/global_annual_summary.csv', 'value': int(worst['year']), 'units': 'year'},
        {'claim_id': 'worst_global_year_gt', 'claim_text': 'Mass change in the most negative global annual mass-change year.', 'artifact': 'outputs/global_annual_summary.csv', 'value': float(worst['combined_gt']), 'units': 'Gt'},
        {'claim_id': 'most_positive_global_year', 'claim_text': 'Most positive global annual mass-change year in 2000-2022.', 'artifact': 'outputs/global_annual_summary.csv', 'value': int(best['year']), 'units': 'year'},
        {'claim_id': 'most_negative_region_total', 'claim_text': 'Region with largest cumulative mass loss over 2000-2022.', 'artifact': 'outputs/regional_2000_2023_totals.csv', 'value': reg_totals.iloc[0]['region'], 'units': 'region'},
        {'claim_id': 'least_negative_region_total_gt', 'claim_text': 'Cumulative mass change for the region with smallest absolute loss / closest to balance.', 'artifact': 'outputs/regional_2000_2023_totals.csv', 'value': float(reg_totals.iloc[-1]['total_gt']), 'units': 'Gt'}
    ])
    pd.DataFrame(claim_rows).to_csv(OUT / 'claim_recovery_table.csv', index=False)

    meta = {
        'analysis_window': {'start_year': 2000, 'end_year_inclusive': 2022, 'note': 'Calendar-year intervals fully contained within 2000-2023; excludes 2023-2024 interval.'},
        'n_global_years': int(len(global_summary)),
        'n_regions': int(regional_summary['region'].nunique()),
        'n_input_series': int(len(coverage_df)),
        'n_method_agreement_rows': int(len(method_agreement))
    }
    (OUT / 'analysis_metadata.json').write_text(json.dumps(meta, indent=2))

    return global_summary, regional_summary, reg_totals, coverage_summary, method_agreement


def make_figures(global_summary, regional_summary, reg_totals, coverage_summary, method_agreement):
    # Figure 1
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(global_summary['year'], global_summary['combined_gt'], color=np.where(global_summary['combined_gt'] < 0, '#2b8cbe', '#74c476'))
    ax1.errorbar(global_summary['year'], global_summary['combined_gt'], yerr=global_summary['combined_gt_errors'], fmt='none', ecolor='black', alpha=0.6, capsize=2)
    ax1.set_ylabel('Annual mass change (Gt)')
    ax1.set_xlabel('Year')
    ax1.set_title('Global glacier annual mass change, consensus GlaMBIE series')
    ax2 = ax1.twinx()
    ax2.plot(global_summary['year'], global_summary['combined_mwe'], color='#d95f0e', marker='o', linewidth=2)
    ax2.set_ylabel('Specific mass change (m w.e.)')
    fig.savefig(IMG / 'global_annual_mass_change.png')
    plt.close(fig)

    # Figure 2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(global_summary['year'], global_summary['cumulative_gt'], color='#54278f', linewidth=3)
    ax.fill_between(global_summary['year'], global_summary['cumulative_gt'] - global_summary['cumulative_gt_error_rss'], global_summary['cumulative_gt'] + global_summary['cumulative_gt_error_rss'], color='#bcbddc', alpha=0.4)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title('Cumulative global glacier mass change since 2000')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative mass change (Gt)')
    fig.savefig(IMG / 'global_cumulative_mass_change.png')
    plt.close(fig)

    # Figure 3
    heat = regional_summary.pivot(index='region', columns='year', values='combined_mwe')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heat.loc[heat.mean(axis=1).sort_values().index], cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': 'Specific mass change (m w.e.)'})
    ax.set_title('Regional annual specific mass change (consensus GlaMBIE series)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Region')
    fig.savefig(IMG / 'regional_heatmap_specific_change.png')
    plt.close(fig)

    # Figure 4
    fig, ax = plt.subplots(figsize=(10, 8))
    reg_plot = reg_totals.sort_values('total_gt')
    ax.barh(reg_plot['region'], reg_plot['total_gt'], color='#3182bd')
    ax.errorbar(reg_plot['total_gt'], reg_plot['region'], xerr=reg_plot['rss_gt_error'], fmt='none', ecolor='black', alpha=0.7)
    ax.set_title('Regional cumulative mass change, 2000-2022')
    ax.set_xlabel('Cumulative mass change (Gt)')
    ax.set_ylabel('Region')
    fig.savefig(IMG / 'regional_total_mass_change_ranked.png')
    plt.close(fig)

    # Figure 5
    cov_plot = coverage_summary.copy()
    cov_pivot = cov_plot.pivot(index='region', columns='method', values='n_series').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cov_pivot.loc[sorted(cov_pivot.index)], annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label': 'Number of input series'}, ax=ax)
    ax.set_title('Input observational-method coverage by region')
    ax.set_xlabel('Method family')
    ax.set_ylabel('Region')
    fig.savefig(IMG / 'method_coverage_by_region.png')
    plt.close(fig)

    # Figure 6
    map_names = {'altimetry': 'Altimetry', 'gravimetry': 'Gravimetry', 'demdiff_and_glaciological': 'DEM diff. + glaciological'}
    plot_df = method_agreement.copy()
    plot_df['method_label'] = plot_df['method'].map(map_names).fillna(plot_df['method'])
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(data=plot_df, x='mean_consensus_gt', y='mean_method_gt', hue='method_label', size='n_years', sizes=(40, 220), ax=ax)
    all_vals = np.concatenate([plot_df['mean_consensus_gt'].values, plot_df['mean_method_gt'].values])
    lims = [float(np.nanmin(all_vals)) - 10, float(np.nanmax(all_vals)) + 10]
    ax.plot(lims, lims, linestyle='--', color='black', linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Mean consensus annual mass change (Gt)')
    ax.set_ylabel('Mean method annual mass change (Gt)')
    ax.set_title('Method-specific hydrological-year means versus consensus means')
    fig.savefig(IMG / 'method_vs_consensus_comparison.png')
    plt.close(fig)


def main():
    ensure_dirs()
    cal_df = load_calendar_results()
    hyd_df = load_hydrological_results()
    coverage_df = load_input_coverage()
    global_summary, regional_summary, reg_totals, coverage_summary, method_agreement = export_core_tables(cal_df, hyd_df, coverage_df)
    make_figures(global_summary, regional_summary, reg_totals, coverage_summary, method_agreement)
    print(json.dumps({
        'status': 'ok',
        'global_total_gt': float(global_summary['combined_gt'].sum()),
        'n_regions': int(regional_summary['region'].nunique()),
        'figures_created': 6
    }, indent=2))


if __name__ == '__main__':
    main()
