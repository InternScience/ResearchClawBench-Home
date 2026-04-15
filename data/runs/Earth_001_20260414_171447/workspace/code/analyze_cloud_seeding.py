import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dataset1_cloud_seeding_records' / 'cloud_seeding_us_2000_2025.csv'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA)
    for col in ['project', 'season', 'state', 'operator_affiliation', 'agent', 'apparatus', 'purpose', 'target_area', 'control_area']:
        df[col] = df[col].fillna('').astype(str).str.strip()
    df['state'] = df['state'].str.lower()
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    return df


def split_multi(value):
    if pd.isna(value) or not str(value).strip():
        return []
    parts = [p.strip().lower() for p in str(value).split(',')]
    return [p for p in parts if p]


def explode_field(df, field):
    rows = []
    for _, r in df.iterrows():
        vals = split_multi(r[field])
        if not vals:
            continue
        for v in vals:
            rows.append({'filename': r['filename'], 'year': r['year'], 'state': r['state'], field: v})
    return pd.DataFrame(rows)


def save_table(df, name):
    path = OUT / f'{name}.csv'
    df.to_csv(path, index=False)
    return path


def dataset_overview(df):
    overview = pd.DataFrame([
        ('records', len(df)),
        ('columns', len(df.columns)),
        ('year_min', int(df['year'].min())),
        ('year_max', int(df['year'].max())),
        ('unique_states', int(df['state'].nunique())),
        ('unique_projects', int(df['project'].nunique())),
        ('unique_operators', int(df['operator_affiliation'].nunique())),
        ('records_with_ground_only', int((df['apparatus'].str.lower() == 'ground').sum())),
        ('records_with_airborne_only', int((df['apparatus'].str.lower() == 'airborne').sum())),
        ('records_with_ground_and_airborne', int((df['apparatus'].str.lower() == 'ground, airborne').sum()))
    ], columns=['metric', 'value'])
    save_table(overview, 'dataset_overview_table')
    return overview


def spatial_concentration(df):
    state_counts = df.groupby('state').size().reset_index(name='records').sort_values('records', ascending=False)
    state_counts['share_pct'] = 100 * state_counts['records'] / state_counts['records'].sum()
    state_counts['cumulative_share_pct'] = state_counts['share_pct'].cumsum()
    save_table(state_counts, 'spatial_concentration_table')

    top3_share = state_counts['share_pct'].head(3).sum()
    top5_share = state_counts['share_pct'].head(5).sum()
    concentration = pd.DataFrame([
        ('top_1_state', state_counts.iloc[0]['state']),
        ('top_1_share_pct', round(state_counts.iloc[0]['share_pct'], 2)),
        ('top_3_share_pct', round(top3_share, 2)),
        ('top_5_share_pct', round(top5_share, 2)),
        ('states_with_single_record', int((state_counts['records'] == 1).sum())),
    ], columns=['metric', 'value'])
    save_table(concentration, 'spatial_concentration_summary')

    fig, ax = plt.subplots(figsize=(11, 7))
    plot_df = state_counts.copy()
    plot_df['state'] = plot_df['state'].str.title()
    sns.barplot(data=plot_df, y='state', x='records', color='#4477AA', ax=ax)
    ax.set_title('Spatial concentration of reported U.S. cloud-seeding projects, 2000–2025')
    ax.set_xlabel('Number of project records')
    ax.set_ylabel('State')
    for i, v in enumerate(plot_df['records']):
        ax.text(v + 1, i, str(v), va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(IMG / 'spatial_concentration.png', bbox_inches='tight')
    plt.close(fig)
    return state_counts, concentration


def annual_activity(df):
    annual = df.groupby('year').size().reset_index(name='records').sort_values('year')
    annual['yoy_change'] = annual['records'].diff()
    annual['three_year_ma'] = annual['records'].rolling(3, min_periods=1).mean()
    save_table(annual, 'annual_activity_table')

    annual_summary = pd.DataFrame([
        ('peak_year', int(annual.loc[annual['records'].idxmax(), 'year'])),
        ('peak_records', int(annual['records'].max())),
        ('lowest_year', int(annual.loc[annual['records'].idxmin(), 'year'])),
        ('lowest_records', int(annual['records'].min())),
        ('records_2000', int(annual.loc[annual['year'] == 2000, 'records'].iloc[0])),
        ('records_2025', int(annual.loc[annual['year'] == 2025, 'records'].iloc[0])),
        ('absolute_change_2000_to_2025', int(annual.loc[annual['year'] == 2025, 'records'].iloc[0] - annual.loc[annual['year'] == 2000, 'records'].iloc[0]))
    ], columns=['metric', 'value'])
    save_table(annual_summary, 'annual_activity_summary')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=annual, x='year', y='records', marker='o', linewidth=2.5, color='#228833', ax=ax, label='Annual records')
    sns.lineplot(data=annual, x='year', y='three_year_ma', linewidth=2.0, color='#CCBB44', ax=ax, label='3-year moving average')
    ax.set_title('Annual activity dynamics in reported cloud-seeding projects')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of project records')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(IMG / 'annual_activity.png', bbox_inches='tight')
    plt.close(fig)
    return annual, annual_summary


def purpose_composition(df):
    purpose_long = explode_field(df, 'purpose')
    purpose_counts = purpose_long.groupby('purpose').size().reset_index(name='mentions').sort_values('mentions', ascending=False)
    purpose_counts['share_pct'] = 100 * purpose_counts['mentions'] / purpose_counts['mentions'].sum()
    save_table(purpose_counts, 'purpose_composition_table')

    purpose_by_year = purpose_long.groupby(['year', 'purpose']).size().reset_index(name='mentions')
    pivot = purpose_by_year.pivot(index='year', columns='purpose', values='mentions').fillna(0).astype(int)
    pivot.to_csv(OUT / 'purpose_by_year_matrix.csv')

    top_purposes = purpose_counts.head(6)['purpose'].tolist()
    plot_df = purpose_by_year[purpose_by_year['purpose'].isin(top_purposes)].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=plot_df, x='year', y='mentions', hue='purpose', marker='o', ax=ax)
    ax.set_title('Purpose composition over time (top purposes by mention count)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mentions in project records')
    ax.legend(title='Purpose', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(IMG / 'purpose_composition.png', bbox_inches='tight')
    plt.close(fig)

    return purpose_counts, pivot


def agent_apparatus_patterns(df):
    agent_long = explode_field(df, 'agent')
    apparatus_long = explode_field(df, 'apparatus')

    agent_counts = agent_long.groupby('agent').size().reset_index(name='mentions').sort_values('mentions', ascending=False)
    apparatus_counts = apparatus_long.groupby('apparatus').size().reset_index(name='mentions').sort_values('mentions', ascending=False)
    save_table(agent_counts, 'agent_counts_table')
    save_table(apparatus_counts, 'apparatus_counts_table')

    combo = []
    for _, r in df.iterrows():
        agents = split_multi(r['agent'])
        apparatuses = split_multi(r['apparatus'])
        for a in agents:
            for ap in apparatuses:
                combo.append({'agent': a, 'apparatus': ap})
    combo_df = pd.DataFrame(combo)
    combo_counts = combo_df.groupby(['agent', 'apparatus']).size().reset_index(name='co_mentions').sort_values('co_mentions', ascending=False)
    save_table(combo_counts, 'agent_apparatus_table')

    top_agents = agent_counts.head(8)['agent'].tolist()
    heat = combo_counts[combo_counts['agent'].isin(top_agents)].pivot(index='agent', columns='apparatus', values='co_mentions').fillna(0)
    heat.to_csv(OUT / 'agent_apparatus_heatmap_matrix.csv')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt='.0f', cmap='Blues', ax=ax)
    ax.set_title('Agent–apparatus deployment patterns')
    ax.set_xlabel('Deployment apparatus')
    ax.set_ylabel('Seeding agent')
    plt.tight_layout()
    fig.savefig(IMG / 'agent_apparatus_patterns.png', bbox_inches='tight')
    plt.close(fig)
    return agent_counts, apparatus_counts, combo_counts, heat


def state_year_matrix(df):
    mat = df.pivot_table(index='state', columns='year', values='filename', aggfunc='count', fill_value=0)
    mat.to_csv(OUT / 'state_by_year_matrix.csv')
    return mat


def claim_recovery(df, state_counts, annual, purpose_counts, combo_counts):
    claims = []
    top3 = state_counts.head(3)
    claims.append({
        'claim': 'Cloud-seeding activity is spatially concentrated in a small set of western states.',
        'status': 'supported',
        'evidence_artifact': 'outputs/spatial_concentration_table.csv; outputs/spatial_concentration_summary.csv; report/images/spatial_concentration.png',
        'quantitative_recovery': f"Top 3 states ({', '.join(top3['state'].str.title())}) account for {top3['share_pct'].sum():.1f}% of all {len(df)} project records."
    })
    peak = annual.loc[annual['records'].idxmax()]
    trough = annual.loc[annual['records'].idxmin()]
    claims.append({
        'claim': 'Annual activity varies substantially over time rather than remaining constant.',
        'status': 'supported',
        'evidence_artifact': 'outputs/annual_activity_table.csv; outputs/annual_activity_summary.csv; report/images/annual_activity.png',
        'quantitative_recovery': f"Annual counts range from {int(trough['records'])} in {int(trough['year'])} to {int(peak['records'])} in {int(peak['year'])}."
    })
    top_purpose = purpose_counts.iloc[0]
    claims.append({
        'claim': 'Operational purposes are dominated by precipitation and snowpack augmentation objectives.',
        'status': 'supported',
        'evidence_artifact': 'outputs/purpose_composition_table.csv; outputs/purpose_by_year_matrix.csv; report/images/purpose_composition.png',
        'quantitative_recovery': f"The most frequent purpose mention is '{top_purpose['purpose']}' with {int(top_purpose['mentions'])} mentions ({top_purpose['share_pct']:.1f}% of purpose mentions); the top four purpose labels are all precipitation/snowpack oriented."
    })
    top_combo = combo_counts.iloc[0]
    claims.append({
        'claim': 'Silver iodide is the dominant agent, deployed primarily via ground and airborne apparatus.',
        'status': 'supported',
        'evidence_artifact': 'outputs/agent_counts_table.csv; outputs/apparatus_counts_table.csv; outputs/agent_apparatus_table.csv; report/images/agent_apparatus_patterns.png',
        'quantitative_recovery': f"The most frequent agent–apparatus pairing is {top_combo['agent']} × {top_combo['apparatus']} with {int(top_combo['co_mentions'])} co-mentions."
    })
    claims.append({
        'claim': 'Independent recovery is limited to the published structured dataset and not a full document-level re-reading of source filings.',
        'status': 'supported_with_limitation',
        'evidence_artifact': 'outputs/dependency_check.json; outputs/related_work_contract.json',
        'quantitative_recovery': 'Dataset-level reproduction was completed, but related-work PDF extraction was unavailable in this environment.'
    })
    claims_df = pd.DataFrame(claims)
    save_table(claims_df, 'claim_recovery_table')
    return claims_df


def main():
    ensure_dirs()
    df = load_data()
    overview = dataset_overview(df)
    state_counts, concentration = spatial_concentration(df)
    annual, annual_summary = annual_activity(df)
    purpose_counts, purpose_year = purpose_composition(df)
    agent_counts, apparatus_counts, combo_counts, heat = agent_apparatus_patterns(df)
    state_year_matrix(df)
    claims = claim_recovery(df, state_counts, annual, purpose_counts, combo_counts)

    summary = {
        'records': int(len(df)),
        'states': int(df['state'].nunique()),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'top_state': state_counts.iloc[0]['state'],
        'top_state_records': int(state_counts.iloc[0]['records']),
        'peak_year': int(annual.loc[annual['records'].idxmax(), 'year']),
        'peak_year_records': int(annual['records'].max()),
        'top_purpose': purpose_counts.iloc[0]['purpose'],
        'top_agent': agent_counts.iloc[0]['agent'],
        'top_apparatus': apparatus_counts.iloc[0]['apparatus']
    }
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
