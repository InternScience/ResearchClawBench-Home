
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dataset1_cloud_seeding_records'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200

STATE_ABBR = {
    'alabama':'AL','alaska':'AK','arizona':'AZ','arkansas':'AR','california':'CA','colorado':'CO','connecticut':'CT',
    'delaware':'DE','florida':'FL','georgia':'GA','hawaii':'HI','idaho':'ID','illinois':'IL','indiana':'IN','iowa':'IA',
    'kansas':'KS','kentucky':'KY','louisiana':'LA','maine':'ME','maryland':'MD','massachusetts':'MA','michigan':'MI',
    'minnesota':'MN','mississippi':'MS','missouri':'MO','montana':'MT','nebraska':'NE','nevada':'NV','new hampshire':'NH',
    'new jersey':'NJ','new mexico':'NM','new york':'NY','north carolina':'NC','north dakota':'ND','ohio':'OH','oklahoma':'OK',
    'oregon':'OR','pennsylvania':'PA','rhode island':'RI','south carolina':'SC','south dakota':'SD','tennessee':'TN','texas':'TX',
    'utah':'UT','vermont':'VT','virginia':'VA','washington':'WA','west virginia':'WV','wisconsin':'WI','wyoming':'WY'
}

PURPOSE_MAP = {
    'snowpack': 'snowpack augmentation',
    'precipitation': 'precipitation enhancement',
    'runoff': 'runoff augmentation',
    'hail': 'hail suppression',
    'fog': 'fog suppression',
    'research': 'research'
}

AGENT_MAP = {
    'silver iodide': 'silver iodide',
    'sodium iodide': 'sodium iodide',
    'ammonium iodide': 'ammonium iodide',
    'calcium chloride': 'calcium chloride',
    'dry ice': 'dry ice',
    'carbon dioxide': 'dry ice/carbon dioxide',
    'hygroscopic': 'hygroscopic materials',
    'ionized air': 'ionized air',
    'acetone': 'acetone',
    'propane': 'propane',
    'water': 'water',
    'potassium chloride': 'potassium chloride',
    'sodium chloride': 'sodium chloride',
    'cesium iodide': 'cesium iodide',
    'ammonium perchlorate': 'ammonium perchlorate',
    'sodium perchlorate': 'sodium perchlorate'
}


def split_tokens(val):
    if pd.isna(val):
        return []
    parts = [p.strip().lower() for p in str(val).split(',')]
    return [p for p in parts if p]


def normalize_purpose(tokens):
    out = []
    for t in tokens:
        matched = None
        for key, label in PURPOSE_MAP.items():
            if key in t:
                matched = label
                break
        if matched is None:
            matched = t
        out.append(matched)
    return sorted(set(out))


def normalize_agent(tokens):
    out = []
    for t in tokens:
        matched = None
        for key, label in AGENT_MAP.items():
            if key in t:
                matched = label
                break
        if matched is None:
            matched = t
        out.append(matched)
    return sorted(set(out))


def apparatus_category(x):
    if pd.isna(x):
        return 'unspecified'
    x = str(x).strip().lower()
    if x == 'ground, airborne' or x == 'airborne, ground':
        return 'mixed ground-airborne'
    return x


def load_data():
    df = pd.read_csv(DATA / 'cloud_seeding_us_2000_2025.csv')
    df['state'] = df['state'].str.strip().str.lower()
    df['state_abbr'] = df['state'].map(STATE_ABBR)
    df['apparatus_category'] = df['apparatus'].apply(apparatus_category)
    df['duration_days'] = (pd.to_datetime(df['end_date'], errors='coerce') - pd.to_datetime(df['start_date'], errors='coerce')).dt.days
    df['purpose_tokens'] = df['purpose'].apply(split_tokens).apply(normalize_purpose)
    df['agent_tokens'] = df['agent'].apply(split_tokens).apply(normalize_agent)
    return df


def save_table(df, name):
    df.to_csv(OUT / f'{name}.csv', index=False)


def main():
    df = load_data()
    summary = {
        'n_records': int(len(df)),
        'year_min': int(df['year'].min()),
        'year_max': int(df['year'].max()),
        'n_states': int(df['state'].nunique()),
        'n_operators': int(df['operator_affiliation'].nunique()),
        'n_agents_raw': int(df['agent'].nunique()),
        'n_apparatus_raw': int(df['apparatus'].nunique(dropna=True)),
        'n_purposes_raw': int(df['purpose'].nunique())
    }

    annual = df.groupby('year').size().reset_index(name='projects')
    annual['yoy_change'] = annual['projects'].diff()
    save_table(annual, 'annual_activity')

    states = df.groupby(['state','state_abbr']).size().reset_index(name='projects').sort_values('projects', ascending=False)
    states['share_pct'] = 100*states['projects']/len(df)
    states['cum_share_pct'] = states['share_pct'].cumsum()
    save_table(states, 'state_counts')

    season_year = df.groupby(['year','season']).size().reset_index(name='projects')
    save_table(season_year, 'year_season_counts')

    purpose_rows = []
    for _, row in df.iterrows():
        for p in row['purpose_tokens']:
            purpose_rows.append({'year': row['year'], 'purpose_group': p})
    purpose_df = pd.DataFrame(purpose_rows)
    purpose_counts = purpose_df.groupby('purpose_group').size().reset_index(name='mentions').sort_values('mentions', ascending=False)
    purpose_counts['share_pct'] = 100*purpose_counts['mentions']/len(purpose_df)
    save_table(purpose_counts, 'purpose_counts')
    purpose_year = purpose_df.groupby(['year','purpose_group']).size().reset_index(name='mentions')
    save_table(purpose_year, 'purpose_year_counts')

    agent_rows = []
    for _, row in df.iterrows():
        for a in row['agent_tokens']:
            agent_rows.append({'year': row['year'], 'agent_group': a, 'apparatus_category': row['apparatus_category']})
    agent_df = pd.DataFrame(agent_rows)
    agent_counts = agent_df.groupby('agent_group').size().reset_index(name='mentions').sort_values('mentions', ascending=False)
    agent_counts['share_pct'] = 100*agent_counts['mentions']/len(agent_df)
    save_table(agent_counts, 'agent_counts')

    apparatus = df.groupby('apparatus_category').size().reset_index(name='projects').sort_values('projects', ascending=False)
    apparatus['share_pct'] = 100*apparatus['projects']/len(df)
    save_table(apparatus, 'apparatus_counts')

    agent_app = agent_df.groupby(['agent_group','apparatus_category']).size().reset_index(name='mentions')
    save_table(agent_app, 'agent_apparatus_counts')

    operator_state = df.groupby(['operator_affiliation','state']).size().reset_index(name='projects').sort_values('projects', ascending=False)
    save_table(operator_state, 'operator_state_counts')

    # Figure 1 annual activity
    fig, ax = plt.subplots(figsize=(11,5.5))
    sns.lineplot(data=annual, x='year', y='projects', marker='o', linewidth=2.5, ax=ax, color='#2166ac')
    ax.set_title('Annual reported cloud-seeding projects in the United States, 2000–2025')
    ax.set_xlabel('Year')
    ax.set_ylabel('Project records')
    ax.set_xticks(annual['year'][::2])
    fig.tight_layout()
    fig.savefig(IMG / 'annual_activity.png')
    plt.close(fig)

    # Figure 2 state bar
    fig, ax = plt.subplots(figsize=(10,7))
    top_states = states.head(10).sort_values('projects')
    ax.barh(top_states['state'].str.title(), top_states['projects'], color='#4393c3')
    ax.set_title('Top reporting states by number of project records')
    ax.set_xlabel('Project records')
    ax.set_ylabel('State')
    fig.tight_layout()
    fig.savefig(IMG / 'state_concentration_bar.png')
    plt.close(fig)

    # Figure 3 map
    gdf = gpd.read_file(DATA / 'us_states.geojson')
    name_col = 'NAME' if 'NAME' in gdf.columns else ('name' if 'name' in gdf.columns else gdf.columns[0])
    gdf['state'] = gdf[name_col].str.strip().str.lower()
    map_df = gdf.merge(states[['state','projects']], on='state', how='left')
    map_df['projects'] = map_df['projects'].fillna(0)
    fig, ax = plt.subplots(figsize=(14,8))
    map_df.plot(column='projects', cmap='Blues', linewidth=0.5, edgecolor='white', legend=True, ax=ax,
                legend_kwds={'label':'Project records'})
    ax.set_title('Spatial concentration of reported cloud-seeding projects by state')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(IMG / 'state_concentration_map.png')
    plt.close(fig)

    # Figure 4 purpose composition
    fig, ax = plt.subplots(figsize=(9,6))
    purpose_top = purpose_counts.sort_values('mentions', ascending=False)
    ax.pie(purpose_top['mentions'], labels=purpose_top['purpose_group'], autopct='%1.1f%%', startangle=90,
           wedgeprops={'linewidth':1,'edgecolor':'white'})
    ax.set_title('Purpose composition based on purpose mentions')
    fig.tight_layout()
    fig.savefig(IMG / 'purpose_composition.png')
    plt.close(fig)

    # Figure 5 purpose over time
    pvt = purpose_year.pivot(index='year', columns='purpose_group', values='mentions').fillna(0)
    fig, ax = plt.subplots(figsize=(11,6))
    pvt.plot(kind='area', stacked=True, ax=ax, colormap='tab20c')
    ax.set_title('Annual dynamics of stated operational purposes')
    ax.set_xlabel('Year')
    ax.set_ylabel('Purpose mentions')
    ax.legend(title='Purpose group', bbox_to_anchor=(1.02,1), loc='upper left')
    fig.tight_layout()
    fig.savefig(IMG / 'purpose_dynamics.png')
    plt.close(fig)

    # Figure 6 apparatus distribution
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=apparatus, x='apparatus_category', y='projects', palette='Set2', ax=ax)
    ax.set_title('Deployment apparatus across project records')
    ax.set_xlabel('Apparatus category')
    ax.set_ylabel('Project records')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(IMG / 'apparatus_distribution.png')
    plt.close(fig)

    # Figure 7 agent-apparatus heatmap
    heat = agent_app.pivot(index='agent_group', columns='apparatus_category', values='mentions').fillna(0)
    heat = heat.loc[agent_counts.head(10)['agent_group']]
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(heat, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax)
    ax.set_title('Agent–apparatus deployment pattern among top seeding agents')
    ax.set_xlabel('Apparatus category')
    ax.set_ylabel('Agent group')
    fig.tight_layout()
    fig.savefig(IMG / 'agent_apparatus_heatmap.png')
    plt.close(fig)

    # validation crosstab figure
    state_purpose = []
    for _, row in df.iterrows():
        for p in row['purpose_tokens']:
            state_purpose.append({'state': row['state'], 'purpose_group': p})
    sp = pd.DataFrame(state_purpose)
    sp_tab = sp.groupby(['state','purpose_group']).size().reset_index(name='mentions')
    save_table(sp_tab, 'state_purpose_counts')
    top_state_names = states.head(8)['state'].tolist()
    sp_heat = sp_tab[sp_tab['state'].isin(top_state_names)].pivot(index='state', columns='purpose_group', values='mentions').fillna(0)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(sp_heat, annot=True, fmt='.0f', cmap='OrRd', ax=ax)
    ax.set_title('Validation plot: purpose mix within highest-activity states')
    ax.set_xlabel('Purpose group')
    ax.set_ylabel('State')
    fig.tight_layout()
    fig.savefig(IMG / 'state_purpose_heatmap.png')
    plt.close(fig)

    summary['top_5_states_share_pct'] = round(states.head(5)['share_pct'].sum(), 2)
    summary['top_3_states_share_pct'] = round(states.head(3)['share_pct'].sum(), 2)
    summary['winter_share_pct'] = round(100*(df['season'].str.contains('winter', case=False, na=False).sum()/len(df)), 2)
    summary['ground_share_pct'] = round(float(apparatus.loc[apparatus['apparatus_category']=='ground','share_pct'].iloc[0]),2)
    summary['airborne_share_pct'] = round(float(apparatus.loc[apparatus['apparatus_category']=='airborne','share_pct'].iloc[0]),2)
    summary['mixed_share_pct'] = round(float(apparatus.loc[apparatus['apparatus_category']=='mixed ground-airborne','share_pct'].iloc[0]),2)
    summary['silver_iodide_mention_share_pct'] = round(float(agent_counts.loc[agent_counts['agent_group']=='silver iodide','share_pct'].iloc[0]),2)
    summary['top_operator'] = str(df['operator_affiliation'].value_counts().idxmax())
    summary['top_operator_projects'] = int(df['operator_affiliation'].value_counts().max())

    with open(OUT / 'summary_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
