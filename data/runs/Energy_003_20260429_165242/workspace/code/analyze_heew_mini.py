#!/usr/bin/env python3
"""Reproducible analysis for the HEEW mini dataset.

Outputs: quality diagnostics, overview tables, hierarchy validation, correlations,
and PNG figures used by report/report.md.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'HEEW_Mini-Dataset'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

ENERGY_COLS = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
WEATHER_COLS = ['Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]', 'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]']
BUILDINGS = [f'BN{i:03d}' for i in range(1, 11)]
ALL_ENERGY_IDS = BUILDINGS + ['CN01', 'Total']

sns.set_theme(style='whitegrid', context='talk')

def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'datetime' in out.columns:
        out['datetime'] = pd.to_datetime(out['datetime'])
    else:
        out['datetime'] = pd.to_datetime(out[['year', 'month', 'day', 'hour']])
    return out

def load_energy(entity: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f'{entity}_energy.csv')
    df = add_datetime(df)
    df['entity'] = entity
    return df

def robust_outlier_counts(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    counts = {}
    for col in cols:
        x = pd.to_numeric(df[col], errors='coerce')
        med = x.median(skipna=True)
        mad = (x - med).abs().median(skipna=True)
        if pd.isna(mad) or mad == 0:
            counts[col] = 0
        else:
            z = 0.6745 * (x - med) / mad
            counts[col] = int((z.abs() > 6).sum())
    return counts

def temporal_diagnostics(df: pd.DataFrame) -> dict:
    dt = pd.to_datetime(df['datetime'])
    dup = int(dt.duplicated().sum())
    start, end = dt.min(), dt.max()
    expected = pd.date_range(start, end, freq='h')
    missing_hours = int(len(expected.difference(pd.DatetimeIndex(dt))))
    unexpected_hours = int(len(pd.DatetimeIndex(dt).difference(expected)))
    irregular_steps = int((dt.sort_values().diff().dropna() != pd.Timedelta(hours=1)).sum())
    return {
        'start': str(start), 'end': str(end), 'rows': int(len(df)),
        'expected_hours_in_span': int(len(expected)), 'duplicate_timestamps': dup,
        'missing_hours_in_span': missing_hours, 'unexpected_hours': unexpected_hours,
        'irregular_hour_steps': irregular_steps,
    }

def quality_row(name: str, df: pd.DataFrame, numeric_cols: list[str], kind: str) -> dict:
    row = {'file_entity': name, 'kind': kind, **temporal_diagnostics(df)}
    row['missing_values_total'] = int(df[numeric_cols].isna().sum().sum())
    row['negative_values_total'] = int((df[numeric_cols] < 0).sum().sum())
    row['zero_pv_hours'] = int((df['PV Power Generation [kW]'] == 0).sum()) if 'PV Power Generation [kW]' in df else np.nan
    row['nonzero_precip_hours'] = int((df['Precipitation [in]'] > 0).sum()) if 'Precipitation [in]' in df else np.nan
    for k, v in robust_outlier_counts(df, numeric_cols).items():
        row[f'robust_outliers__{k}'] = v
    return row

def main():
    energy = {entity: load_energy(entity) for entity in ALL_ENERGY_IDS}
    weather = add_datetime(pd.read_csv(DATA / 'Total_weather.csv'))
    weather['entity'] = 'Total_weather'

    # Overview and quality diagnostics
    overview = {
        'dataset': 'HEEW_Mini-Dataset',
        'entities': ALL_ENERGY_IDS,
        'independent_buildings': BUILDINGS,
        'energy_columns': ENERGY_COLS,
        'weather_columns': WEATHER_COLS,
        'energy_files': {},
        'weather_file': temporal_diagnostics(weather),
        'joined_total_energy_weather_rows': int(len(pd.merge(energy['Total'][['datetime'] + ENERGY_COLS], weather[['datetime'] + WEATHER_COLS], on='datetime', how='inner'))),
    }
    for entity, df in energy.items():
        overview['energy_files'][entity] = temporal_diagnostics(df)
    with open(OUT / 'data_overview.json', 'w') as f:
        json.dump(overview, f, indent=2)

    qrows = []
    for entity, df in energy.items():
        qrows.append(quality_row(entity, df, ENERGY_COLS, 'energy'))
    qrows.append(quality_row('Total_weather', weather, WEATHER_COLS, 'weather'))
    quality = pd.DataFrame(qrows)
    quality.to_csv(OUT / 'cleaning_diagnostics.csv', index=False)

    # Building annual totals and contributions.
    annual_rows = []
    for entity in BUILDINGS:
        df = energy[entity]
        rec = {'entity': entity, 'rows': len(df)}
        for col in ENERGY_COLS:
            rec[f'annual_sum__{col}'] = float(df[col].sum())
            rec[f'mean__{col}'] = float(df[col].mean())
            rec[f'max__{col}'] = float(df[col].max())
        annual_rows.append(rec)
    annual = pd.DataFrame(annual_rows)
    for col in ENERGY_COLS:
        total_col = f'annual_sum__{col}'
        annual[f'share_of_building_sum__{col}'] = annual[total_col] / annual[total_col].sum()
    annual.to_csv(OUT / 'building_annual_totals.csv', index=False)

    # Join Total and weather for correlations and profiles.
    total_join = pd.merge(energy['Total'][['datetime'] + ENERGY_COLS], weather[['datetime'] + WEATHER_COLS], on='datetime', how='inner')
    total_join['month'] = total_join['datetime'].dt.month
    total_join['hour'] = total_join['datetime'].dt.hour
    total_join.to_csv(OUT / 'total_energy_weather_joined.csv', index=False)
    corr_cols = ENERGY_COLS + WEATHER_COLS
    corr = total_join[corr_cols].corr(method='pearson')
    corr.to_csv(OUT / 'correlation_matrix.csv')

    # Hierarchical consistency: building sums versus CN01 and Total; CN01 vs Total.
    merged = None
    for entity in BUILDINGS:
        cols = ['datetime'] + ENERGY_COLS
        tmp = energy[entity][cols].copy()
        tmp = tmp.rename(columns={c: f'{entity}__{c}' for c in ENERGY_COLS})
        merged = tmp if merged is None else merged.merge(tmp, on='datetime', how='outer')
    bsum = pd.DataFrame({'datetime': merged['datetime']})
    for col in ENERGY_COLS:
        bsum[f'building_sum__{col}'] = merged[[f'{b}__{col}' for b in BUILDINGS]].sum(axis=1)
    hier_rows = []
    resid_hourly = bsum[['datetime']].copy()
    comparisons = [('sum_BN001_BN010', 'CN01'), ('sum_BN001_BN010', 'Total'), ('CN01', 'Total')]
    for left, right in comparisons:
        if left == 'sum_BN001_BN010':
            left_df = bsum.rename(columns={f'building_sum__{c}': c for c in ENERGY_COLS})[['datetime'] + ENERGY_COLS]
        else:
            left_df = energy[left][['datetime'] + ENERGY_COLS]
        right_df = energy[right][['datetime'] + ENERGY_COLS]
        comp = pd.merge(left_df, right_df, on='datetime', suffixes=('_left','_right'))
        for col in ENERGY_COLS:
            residual = comp[f'{col}_left'] - comp[f'{col}_right']
            denom = comp[f'{col}_right'].abs().replace(0, np.nan)
            resid_hourly[f'{left}_minus_{right}__{col}'] = residual.values
            hier_rows.append({
                'comparison': f'{left}_vs_{right}', 'variable': col,
                'n_hours': int(len(comp)), 'mean_abs_error': float(residual.abs().mean()),
                'max_abs_error': float(residual.abs().max()), 'rmse': float(np.sqrt((residual ** 2).mean())),
                'mean_abs_percentage_error': float((residual.abs()/denom).mean(skipna=True) * 100),
                'allclose_at_1e-6': bool(np.allclose(comp[f'{col}_left'], comp[f'{col}_right'], rtol=1e-9, atol=1e-6)),
            })
    hier = pd.DataFrame(hier_rows)
    hier.to_csv(OUT / 'hierarchical_consistency.csv', index=False)
    resid_hourly.to_csv(OUT / 'hierarchy_hourly_residuals.csv', index=False)

    # Cleaning algorithm pseudo steps/fidelity checklist.
    fidelity = {
        'named_protocol': 'HEEW mini data cleaning and validation workflow',
        'definition': 'Normalize timestamps; verify hourly completeness; detect duplicates/missing values/non-physical negatives; flag robust univariate outliers; merge weather and energy by datetime; validate hierarchy by additive consistency.',
        'assumptions': [
            'Hourly records should cover every hour between the first and last timestamp.',
            'Energy, generation, emissions, and weather magnitude columns are expected to be numeric and non-negative except no domain-specific upper bound is imposed.',
            'CN01 and Total aggregates should equal the sum of the 10 provided BN buildings in the mini dataset; CN01 and Total are expected identical in this compact version.'
        ],
        'invariants_checked': [
            'Unique hourly timestamps per file', 'No missing hours within 2014 span', 'No missing numeric cells', 'No negative numeric cells', 'Additive hierarchy residuals approximately zero'
        ],
        'implemented_artifacts': ['outputs/cleaning_diagnostics.csv', 'outputs/hierarchical_consistency.csv', 'outputs/hierarchy_hourly_residuals.csv']
    }
    with open(OUT / 'method_fidelity_checklist.json', 'w') as f:
        json.dump(fidelity, f, indent=2)

    # Figures.
    # 1 coverage heatmap: file x month completeness fraction.
    coverage_rows = []
    for entity, df in {**energy, 'Total_weather': weather}.items():
        dt = pd.to_datetime(df['datetime'])
        for m in range(1, 13):
            actual = int((dt.dt.month == m).sum())
            expected = len(pd.date_range(f'2014-{m:02d}-01', pd.Timestamp(2014, m, 1) + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23), freq='h'))
            coverage_rows.append({'entity': entity, 'month': m, 'coverage_fraction': actual/expected})
    cov = pd.DataFrame(coverage_rows).pivot(index='entity', columns='month', values='coverage_fraction').loc[ALL_ENERGY_IDS + ['Total_weather']]
    cov.to_csv(OUT / 'monthly_coverage_matrix.csv')
    plt.figure(figsize=(13, 6))
    sns.heatmap(cov, vmin=0, vmax=1, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label':'fraction of expected hourly records'})
    plt.title('Hourly coverage by hierarchy node and month (2014)')
    plt.xlabel('Month'); plt.ylabel('Entity')
    plt.tight_layout(); plt.savefig(IMG / 'data_coverage_heatmap.png', dpi=200); plt.close()

    # 2 monthly total energy and temperature/PV axis.
    monthly = total_join.set_index('datetime').resample('ME').agg({
        'Electricity [kW]':'sum','Heat [mmBTU]':'sum','Cooling Energy [Ton]':'sum','PV Power Generation [kW]':'sum','Greenhouse Gas Emission [Ton]':'sum','Temperature [°F]':'mean'
    })
    monthly.to_csv(OUT / 'monthly_total_summary.csv')
    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(monthly.index))
    ax1.plot(x, monthly['Electricity [kW]'], marker='o', label='Electricity sum [kWh-equivalent]')
    ax1.plot(x, monthly['Cooling Energy [Ton]'], marker='s', label='Cooling ton-hours')
    ax1.plot(x, monthly['PV Power Generation [kW]'], marker='^', label='PV generation [kWh-equivalent]')
    ax1.set_ylabel('Monthly totals')
    ax1.legend(loc='upper left', fontsize=10)
    ax2 = ax1.twinx()
    ax2.plot(x, monthly['Temperature [°F]'], color='crimson', marker='d', label='Mean temperature')
    ax2.set_ylabel('Mean temperature [°F]', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax1.set_xticks(x); ax1.set_xticklabels([d.strftime('%b') for d in monthly.index])
    ax1.set_title('Monthly aggregate energy, PV, and ambient temperature')
    fig.tight_layout(); fig.savefig(IMG / 'monthly_energy_weather.png', dpi=200); plt.close(fig)

    # 3 diurnal profiles.
    diurnal = total_join.groupby('hour')[ENERGY_COLS + ['Temperature [°F]']].mean()
    diurnal.to_csv(OUT / 'diurnal_total_profiles.csv')
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for col in ['Electricity [kW]', 'Cooling Energy [Ton]', 'PV Power Generation [kW]']:
        axes[0].plot(diurnal.index, diurnal[col], marker='o', label=col)
    axes[0].set_ylabel('Mean hourly value'); axes[0].legend(fontsize=10); axes[0].set_title('Mean diurnal profiles for Total energy node')
    axes[1].plot(diurnal.index, diurnal['Heat [mmBTU]'], marker='s', label='Heat [mmBTU]')
    axes[1].plot(diurnal.index, diurnal['Temperature [°F]'], marker='d', label='Temperature [°F]')
    axes[1].set_xlabel('Hour of day'); axes[1].set_ylabel('Mean hourly value'); axes[1].legend(fontsize=10)
    plt.tight_layout(); plt.savefig(IMG / 'diurnal_profiles.png', dpi=200); plt.close()

    # 4 correlation heatmap.
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True, cbar_kws={'label':'Pearson r'})
    plt.title('Correlation matrix: Total energy, GHG, and weather variables')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(IMG / 'correlation_heatmap.png', dpi=200); plt.close()

    # 5 hierarchy residuals: absolute residual summary log scale.
    hplot = hier.copy()
    hplot['variable_short'] = hplot['variable'].str.replace(' \[.*\]', '', regex=True)
    plt.figure(figsize=(13, 6))
    sns.barplot(data=hplot, x='variable_short', y='max_abs_error', hue='comparison')
    plt.yscale('symlog', linthresh=1e-12)
    plt.ylabel('Maximum absolute residual (symlog)')
    plt.xlabel('Variable')
    plt.title('Hierarchical aggregation residuals are numerically negligible')
    plt.xticks(rotation=20, ha='right')
    plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(IMG / 'hierarchy_residuals.png', dpi=200); plt.close()

    # 6 building totals stacked/grouped.
    plot_annual = annual[['entity','annual_sum__Electricity [kW]','annual_sum__Cooling Energy [Ton]','annual_sum__Greenhouse Gas Emission [Ton]']].copy()
    long = plot_annual.melt(id_vars='entity', var_name='variable', value_name='annual_total')
    long['variable'] = long['variable'].str.replace('annual_sum__','', regex=False).str.replace(' \[.*\]', '', regex=True)
    plt.figure(figsize=(13, 6))
    sns.barplot(data=long, x='entity', y='annual_total', hue='variable')
    plt.title('Annual totals by independent building')
    plt.ylabel('Annual total (native units summed over hours)')
    plt.xlabel('Building')
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(IMG / 'building_energy_totals.png', dpi=200); plt.close()

    # Direct claim recovery table.
    claim_rows = [
        {'claim': 'All provided energy and weather files cover 8760 hourly records in 2014 with no missing hours.', 'supporting_artifact': 'outputs/data_overview.json; outputs/cleaning_diagnostics.csv', 'status': 'verified' if (quality['rows'].eq(8760).all() and quality['missing_hours_in_span'].eq(0).all()) else 'not fully verified'},
        {'claim': 'The 10 building nodes sum exactly/numerically to CN01 and Total for all five energy/emission variables.', 'supporting_artifact': 'outputs/hierarchical_consistency.csv; report/images/hierarchy_residuals.png', 'status': 'verified' if hier['allclose_at_1e-6'].all() else 'not fully verified'},
        {'claim': 'Total weather can be joined one-to-one with Total energy at hourly resolution.', 'supporting_artifact': 'outputs/data_overview.json; outputs/total_energy_weather_joined.csv', 'status': 'verified' if len(total_join)==8760 else 'not fully verified'},
        {'claim': 'Temperature is strongly associated with cooling/electricity seasonality in the mini dataset.', 'supporting_artifact': 'outputs/correlation_matrix.csv; report/images/monthly_energy_weather.png; report/images/correlation_heatmap.png', 'status': 'quantified'},
        {'claim': 'The mini dataset is a compact 2014 subset, not the full 11,987,328-record 2014-2022 HEEW release.', 'supporting_artifact': 'INSTRUCTIONS.md; outputs/data_overview.json', 'status': 'limitation'}
    ]
    pd.DataFrame(claim_rows).to_csv(OUT / 'claim_recovery_table.csv', index=False)

    # Update target artifact inventory statuses.
    inv = {
      'primary_quantitative_outputs': [
        {'artifact':'outputs/data_overview.json','status':'satisfied'},
        {'artifact':'outputs/cleaning_diagnostics.csv','status':'satisfied'},
        {'artifact':'outputs/hierarchical_consistency.csv','status':'satisfied'},
        {'artifact':'outputs/correlation_matrix.csv','status':'satisfied'},
        {'artifact':'outputs/building_annual_totals.csv','status':'satisfied'}],
      'expected_figure_families': [{'artifact':f'report/images/{name}','status':'satisfied'} for name in ['data_coverage_heatmap.png','monthly_energy_weather.png','diurnal_profiles.png','correlation_heatmap.png','hierarchy_residuals.png','building_energy_totals.png']],
      'interpretability_or_validation_artifacts': [
        {'artifact':'outputs/claim_recovery_table.csv','status':'satisfied'},
        {'artifact':'outputs/method_fidelity_checklist.json','status':'satisfied'},
        {'artifact':'report/report.md','status':'planned'}]
    }
    with open(OUT / 'target_artifact_inventory.json','w') as f:
        json.dump(inv, f, indent=2)

    print(json.dumps({
        'rows_per_energy_file': {k:int(len(v)) for k,v in energy.items()},
        'weather_rows': int(len(weather)),
        'joined_rows': int(len(total_join)),
        'quality_rows': int(len(quality)),
        'hierarchy_allclose': bool(hier['allclose_at_1e-6'].all()),
        'figures': sorted([p.name for p in IMG.glob('*.png')])
    }, indent=2))

if __name__ == '__main__':
    main()
