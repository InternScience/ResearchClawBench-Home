import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data' / 'HEEW_Mini-Dataset'
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150

ENERGY_COLS = [
    'Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]',
    'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]'
]
WEATHER_COLS = [
    'Temperature [°F]', 'Dew Point [°F]', 'Humidity [%]',
    'Wind Speed [mph]', 'Wind Gust [mph]', 'Pressure [in]', 'Precipitation [in]'
]


def load_energy(name):
    df = pd.read_csv(DATA / f'{name}_energy.csv')
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['entity'] = name
    return df


def load_all():
    building_names = [f'BN{i:03d}' for i in range(1, 11)]
    buildings = pd.concat([load_energy(n) for n in building_names], ignore_index=True)
    cn = load_energy('CN01')
    total = load_energy('Total')
    weather = pd.read_csv(DATA / 'Total_weather.csv')
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    return buildings, cn, total, weather


def compute_qc(buildings, cn, total, weather):
    files = {}
    for name in list(buildings['entity'].unique()) + ['CN01', 'Total']:
        if name in {'CN01', 'Total'}:
            df = cn if name == 'CN01' else total
        else:
            df = buildings[buildings['entity'] == name]
        files[name] = {
            'rows': int(len(df)),
            'missing_values': int(df.isna().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'negative_numeric_values': int((df.select_dtypes(include=[np.number]) < 0).sum().sum()),
        }
    weather_summary = {
        'rows': int(len(weather)),
        'missing_values': int(weather.isna().sum().sum()),
        'duplicate_rows': int(weather.duplicated().sum()),
        'negative_precipitation': int((weather['Precipitation [in]'] < 0).sum()),
    }
    return {'energy_entities': files, 'weather': weather_summary}


def hierarchical_check(buildings, cn, total):
    agg = buildings.groupby('datetime')[ENERGY_COLS].sum().reset_index()
    merged_cn = agg.merge(cn[['datetime'] + ENERGY_COLS], on='datetime', suffixes=('_sum', '_cn'))
    merged_total = agg.merge(total[['datetime'] + ENERGY_COLS], on='datetime', suffixes=('_sum', '_total'))
    res = {}
    for label, merged, suffix in [('CN01', merged_cn, '_cn'), ('Total', merged_total, '_total')]:
        stats = {}
        for col in ENERGY_COLS:
            diff = merged[f'{col}_sum'] - merged[f'{col}{suffix}']
            stats[col] = {
                'max_abs_error': float(diff.abs().max()),
                'mean_abs_error': float(diff.abs().mean()),
                'allclose_1e-6': bool(np.allclose(merged[f'{col}_sum'], merged[f'{col}{suffix}'], atol=1e-6))
            }
        res[label] = stats
    return res, agg


def make_figures(buildings, total, weather, agg):
    total_weather = total[['datetime'] + ENERGY_COLS].merge(weather[['datetime'] + WEATHER_COLS], on='datetime')

    # Figure 1: mean monthly profiles for total
    monthly = total_weather.copy()
    monthly['month'] = monthly['datetime'].dt.month
    msum = monthly.groupby('month')[ENERGY_COLS].mean().reset_index()
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, col in enumerate(ENERGY_COLS):
        ax = axes[i]
        sns.lineplot(data=msum, x='month', y=col, marker='o', ax=ax, color='#1f77b4')
        ax.set_title(col)
        ax.set_xlabel('Month')
    axes[-1].axis('off')
    fig.suptitle('Monthly mean multi-energy profiles for Total (2014)', y=1.02)
    fig.tight_layout()
    fig.savefig(IMG / 'monthly_energy_profiles.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: weather-energy correlation heatmap
    corr = total_weather[ENERGY_COLS + WEATHER_COLS].corr().loc[ENERGY_COLS, WEATHER_COLS]
    plt.figure(figsize=(12, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation between energy variables and weather attributes (Total)')
    plt.tight_layout()
    plt.savefig(IMG / 'weather_energy_correlation.png', bbox_inches='tight')
    plt.close()

    # Figure 3: hierarchy validation scatter electricity and cooling
    compare = agg[['datetime', 'Electricity [kW]', 'Cooling Energy [Ton]']].merge(
        total[['datetime', 'Electricity [kW]', 'Cooling Energy [Ton]']], on='datetime', suffixes=('_sum', '_total'))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col in zip(axes, ['Electricity [kW]', 'Cooling Energy [Ton]']):
        sns.scatterplot(data=compare.sample(min(1500, len(compare)), random_state=0),
                        x=f'{col}_sum', y=f'{col}_total', s=18, alpha=0.6, ax=ax)
        lims = [compare[[f'{col}_sum', f'{col}_total']].min().min(), compare[[f'{col}_sum', f'{col}_total']].max().max()]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_title(col)
        ax.set_xlabel('Sum of 10 buildings')
        ax.set_ylabel('Reported Total')
    fig.suptitle('Hierarchical consistency validation')
    fig.tight_layout()
    fig.savefig(IMG / 'hierarchy_validation.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 4: building load diversity
    b = buildings.copy()
    b['month'] = b['datetime'].dt.month
    by_bldg = b.groupby('entity')[ENERGY_COLS].mean().reset_index().melt(id_vars='entity', var_name='variable', value_name='mean_value')
    plt.figure(figsize=(14, 7))
    sns.barplot(data=by_bldg[by_bldg['variable'].isin(['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]'])],
                x='entity', y='mean_value', hue='variable')
    plt.title('Average building-level energy intensity proxies in 2014')
    plt.xlabel('Building')
    plt.ylabel('Mean hourly value')
    plt.tight_layout()
    plt.savefig(IMG / 'building_energy_diversity.png', bbox_inches='tight')
    plt.close()

    # Figure 5: daily shape by season for total electricity and cooling
    season_map = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
    di = total.copy()
    di['season'] = di['datetime'].dt.month.map(season_map)
    di['hour'] = di['datetime'].dt.hour
    for var, fname in [('Electricity [kW]', 'diurnal_electricity_by_season.png'), ('Cooling Energy [Ton]', 'diurnal_cooling_by_season.png')]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=di.groupby(['season', 'hour'])[var].mean().reset_index(), x='hour', y=var, hue='season', marker='o')
        plt.title(f'Mean diurnal {var} by season (Total)')
        plt.tight_layout()
        plt.savefig(IMG / fname, bbox_inches='tight')
        plt.close()

    return corr


def main():
    buildings, cn, total, weather = load_all()
    qc = compute_qc(buildings, cn, total, weather)
    hierarchy, agg = hierarchical_check(buildings, cn, total)

    total_weather = total[['datetime'] + ENERGY_COLS].merge(weather[['datetime'] + WEATHER_COLS], on='datetime')
    corr = total_weather[ENERGY_COLS + WEATHER_COLS].corr()

    annual_stats = total[ENERGY_COLS].agg(['mean', 'std', 'min', 'max']).transpose()
    building_stats = buildings.groupby('entity')[ENERGY_COLS].mean().round(3)

    corr_energy_weather = make_figures(buildings, total, weather, agg)

    outputs = {
        'qc_summary': qc,
        'hierarchy_check': hierarchy,
        'total_annual_stats': annual_stats.round(4).to_dict(orient='index'),
        'building_mean_stats': building_stats.to_dict(orient='index'),
        'selected_energy_weather_correlation': corr_energy_weather.round(4).to_dict(),
    }
    with open(OUT / 'analysis_results.json', 'w') as f:
        json.dump(outputs, f, indent=2)
    annual_stats.to_csv(OUT / 'total_annual_stats.csv')
    building_stats.to_csv(OUT / 'building_mean_stats.csv')
    corr_energy_weather.to_csv(OUT / 'energy_weather_correlation.csv')
    print('Analysis complete.')

if __name__ == '__main__':
    main()
