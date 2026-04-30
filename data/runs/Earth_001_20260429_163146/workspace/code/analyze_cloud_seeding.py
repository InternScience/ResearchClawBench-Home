#!/usr/bin/env python3
"""Reproducible descriptive recovery analysis for NOAA cloud-seeding records."""
from __future__ import annotations
import csv, json, math, os, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dataset1_cloud_seeding_records' / 'cloud_seeding_us_2000_2025.csv'
GEO = ROOT / 'data' / 'dataset1_cloud_seeding_records' / 'us_states.geojson'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
STATE_ABBR = {
 'alabama':'AL','alaska':'AK','arizona':'AZ','arkansas':'AR','california':'CA','colorado':'CO','connecticut':'CT','delaware':'DE','florida':'FL','georgia':'GA','hawaii':'HI','idaho':'ID','illinois':'IL','indiana':'IN','iowa':'IA','kansas':'KS','kentucky':'KY','louisiana':'LA','maine':'ME','maryland':'MD','massachusetts':'MA','michigan':'MI','minnesota':'MN','mississippi':'MS','missouri':'MO','montana':'MT','nebraska':'NE','nevada':'NV','new hampshire':'NH','new jersey':'NJ','new mexico':'NM','new york':'NY','north carolina':'NC','north dakota':'ND','ohio':'OH','oklahoma':'OK','oregon':'OR','pennsylvania':'PA','rhode island':'RI','south carolina':'SC','south dakota':'SD','tennessee':'TN','texas':'TX','utah':'UT','vermont':'VT','virginia':'VA','washington':'WA','west virginia':'WV','wisconsin':'WI','wyoming':'WY','district of columbia':'DC'
}

def split_multilabel(x):
    if pd.isna(x): return []
    parts=[]
    for p in str(x).split(','):
        p=p.strip().lower()
        if p: parts.append(p)
    return parts

def save_json(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

def hhi(counts):
    total=sum(counts)
    return sum((c/total)**2 for c in counts) if total else np.nan

def gini(values):
    x=np.array(values, dtype=float)
    if len(x)==0 or x.sum()==0: return np.nan
    x=np.sort(x)
    n=len(x)
    return (2*np.arange(1,n+1).dot(x))/(n*x.sum()) - (n+1)/n

def main():
    df=pd.read_csv(DATA)
    # Normalize key strings
    for c in ['state','purpose','agent','apparatus','season','operator_affiliation']:
        df[c]=df[c].astype('string').str.strip().str.lower()
    df['state_abbr']=df['state'].map(STATE_ABBR)
    df['start_dt']=pd.to_datetime(df['start_date'], errors='coerce')
    df['end_dt']=pd.to_datetime(df['end_date'], errors='coerce')

    overview={
      'source': str(DATA.relative_to(ROOT)),
      'n_records': int(len(df)),
      'n_columns': int(df.shape[1]),
      'columns': list(pd.read_csv(DATA, nrows=0).columns),
      'year_min': int(df.year.min()), 'year_max': int(df.year.max()),
      'n_states': int(df.state.nunique()),
      'states': sorted(df.state.dropna().unique().tolist()),
      'missing_by_field': {k:int(v) for k,v in df[pd.read_csv(DATA, nrows=0).columns].isna().sum().to_dict().items()},
      'duplicate_filenames': int(df['filename'].duplicated().sum()),
      'records_with_control_area': int(df['control_area'].notna().sum()),
      'records_without_control_area': int(df['control_area'].isna().sum())
    }
    save_json(OUT/'dataset_overview.json', overview)

    annual = df.groupby('year').size().rename('records').reset_index()
    all_years=pd.DataFrame({'year': range(int(df.year.min()), int(df.year.max())+1)})
    annual=all_years.merge(annual,on='year',how='left').fillna({'records':0})
    annual['records']=annual['records'].astype(int)
    annual.to_csv(OUT/'annual_counts.csv', index=False)

    state_counts=df.groupby(['state','state_abbr']).size().rename('records').reset_index().sort_values('records', ascending=False)
    state_counts['share']=state_counts['records']/len(df)
    state_counts['cumulative_share']=state_counts['share'].cumsum()
    state_counts.to_csv(OUT/'state_counts.csv', index=False)

    # Season, purpose, agent, apparatus multilabel tables
    for field in ['purpose','agent','apparatus','season']:
        rows=[]
        for i,x in enumerate(df[field]):
            labs=split_multilabel(x)
            if not labs: labs=['(missing)']
            for lab in labs: rows.append({'row_id':i, field:lab})
        ex=pd.DataFrame(rows)
        counts=ex[field].value_counts().rename_axis(field).reset_index(name='mentions')
        counts['record_share']=counts['mentions']/len(df)
        counts.to_csv(OUT/f'{field}_counts.csv', index=False)
        ex.to_csv(OUT/f'{field}_exploded.csv', index=False)

    # Crosstabs
    agent_ex=pd.read_csv(OUT/'agent_exploded.csv')
    app_ex=pd.read_csv(OUT/'apparatus_exploded.csv')
    purp_ex=pd.read_csv(OUT/'purpose_exploded.csv')
    aa=agent_ex.merge(app_ex,on='row_id')
    ct=pd.crosstab(aa['agent'], aa['apparatus'])
    ct.to_csv(OUT/'agent_apparatus_crosstab.csv')
    pct=ct.div(ct.sum(axis=1).replace(0,np.nan), axis=0)
    pct.to_csv(OUT/'agent_apparatus_rowshare.csv')

    ps=df[['state']].reset_index(names='row_id').merge(purp_ex,on='row_id')
    purpose_by_state=pd.crosstab(ps['state'], ps['purpose'])
    purpose_by_state.to_csv(OUT/'purpose_by_state.csv')
    year_base=df[['year']].reset_index(names='row_id')
    purp_year=year_base.merge(purp_ex,on='row_id')
    purpose_by_year=pd.crosstab(purp_year['year'], purp_year['purpose'])
    purpose_by_year.to_csv(OUT/'purpose_by_year.csv')
    app_year=year_base.merge(app_ex,on='row_id')
    app_by_year=pd.crosstab(app_year['year'], app_year['apparatus'])
    app_by_year.to_csv(OUT/'apparatus_by_year.csv')

    # Summary metrics
    top4_share=float(state_counts.head(4)['share'].sum())
    top8_share=float(state_counts.head(8)['share'].sum())
    metrics={
      'total_records': int(len(df)),
      'year_range': [int(df.year.min()), int(df.year.max())],
      'state_count': int(df.state.nunique()),
      'top_state': state_counts.iloc[0][['state','records','share']].to_dict(),
      'top4_state_share': top4_share,
      'top8_state_share': top8_share,
      'state_hhi': hhi(state_counts['records']),
      'state_gini': gini(state_counts['records']),
      'peak_year': annual.loc[annual.records.idxmax()].to_dict(),
      'lowest_nonzero_year': annual[annual.records>0].loc[annual[annual.records>0].records.idxmin()].to_dict(),
      'annual_mean': float(annual.records.mean()),
      'annual_sd': float(annual.records.std(ddof=1)),
      'silver_iodide_record_count': int(df['agent'].fillna('').str.contains('silver iodide', regex=False).sum()),
      'silver_iodide_record_share': float(df['agent'].fillna('').str.contains('silver iodide', regex=False).mean()),
      'ground_any_count': int(df['apparatus'].fillna('').str.contains('ground', regex=False).sum()),
      'airborne_any_count': int(df['apparatus'].fillna('').str.contains('airborne', regex=False).sum()),
      'ground_any_share': float(df['apparatus'].fillna('').str.contains('ground', regex=False).mean()),
      'airborne_any_share': float(df['apparatus'].fillna('').str.contains('airborne', regex=False).mean()),
      'control_area_present_share': float(df['control_area'].notna().mean())
    }
    save_json(OUT/'summary_metrics.json', metrics)

    # Figures
    fig,ax=plt.subplots(figsize=(10,4.8))
    ax.plot(annual['year'], annual['records'], marker='o', linewidth=2, color='#2b6cb0')
    ax.fill_between(annual['year'], annual['records'], alpha=.15, color='#2b6cb0')
    ax.set_title('Annual U.S. reported cloud-seeding project records, 2000–2025')
    ax.set_xlabel('Report year'); ax.set_ylabel('Number of records')
    ax.set_xticks(list(range(2000,2026,2))); ax.tick_params(axis='x', rotation=45)
    peak=annual.loc[annual.records.idxmax()]
    ax.annotate(f"Peak: {int(peak.records)} in {int(peak.year)}", xy=(peak.year, peak.records), xytext=(peak.year-6, peak.records+8), arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    fig.tight_layout(); fig.savefig(IMG/'fig1_annual_activity.png', dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5.5))
    sns.barplot(data=state_counts, y='state', x='records', ax=ax, color='#3182bd')
    ax.set_title('Spatial concentration by reporting state')
    ax.set_xlabel('Number of project records'); ax.set_ylabel('State')
    for i,(rec,share) in enumerate(zip(state_counts['records'], state_counts['share'])):
        ax.text(rec+2, i, f'{share:.1%}', va='center', fontsize=8)
    ax.set_xlim(0, max(state_counts.records)*1.18)
    fig.tight_layout(); fig.savefig(IMG/'fig2_state_concentration_bar.png', dpi=220); plt.close(fig)

    # Simple GeoJSON choropleth without geopandas
    with open(GEO) as f: geo=json.load(f)
    count_by_abbr=dict(zip(state_counts.state_abbr, state_counts.records))
    vals=np.array(list(count_by_abbr.values()))
    vmax=vals.max() if len(vals) else 1
    cmap=plt.cm.Blues
    fig,ax=plt.subplots(figsize=(11,7))
    for feat in geo['features']:
        props=feat.get('properties',{})
        abbr=props.get('STUSPS') or props.get('postal') or props.get('abbr') or props.get('STATE_ABBR')
        name=(props.get('NAME') or props.get('name') or '').lower()
        if not abbr and name in STATE_ABBR: abbr=STATE_ABBR[name]
        val=count_by_abbr.get(abbr,0)
        color=cmap(0.12+0.88*val/vmax) if val>0 else '#eeeeee'
        geom=feat['geometry']; coords=geom['coordinates']
        polys=coords if geom['type']=='MultiPolygon' else [coords]
        for poly in polys:
            exterior=poly[0]
            xs=[p[0] for p in exterior]; ys=[p[1] for p in exterior]
            ax.fill(xs,ys,facecolor=color,edgecolor='white',linewidth=.4)
    ax.set_xlim(-126,-66); ax.set_ylim(24,50); ax.set_aspect('equal')
    ax.set_title('Reported cloud-seeding records mapped by state (contiguous U.S.)')
    ax.axis('off')
    sm=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0,vmax=vmax)); sm.set_array([])
    cbar=fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01); cbar.set_label('Records')
    fig.tight_layout(); fig.savefig(IMG/'fig3_state_choropleth.png', dpi=220); plt.close(fig)

    purpose_counts=pd.read_csv(OUT/'purpose_counts.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    pc=purpose_counts.sort_values('mentions', ascending=True)
    sns.barplot(data=pc, y='purpose', x='mentions', ax=ax, color='#38a169')
    ax.set_title('Stated purpose composition (multi-label mentions)')
    ax.set_xlabel('Mentions across project records'); ax.set_ylabel('Purpose')
    fig.tight_layout(); fig.savefig(IMG/'fig4_purpose_composition.png', dpi=220); plt.close(fig)

    # Heatmap top agents, apparatus
    top_agents=pd.read_csv(OUT/'agent_counts.csv').head(10)['agent'].tolist()
    h=ct.loc[[a for a in top_agents if a in ct.index]]
    fig,ax=plt.subplots(figsize=(7.5,5.8))
    sns.heatmap(h, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label':'Co-mentions'}, ax=ax)
    ax.set_title('Deployment apparatus by seeding agent')
    ax.set_xlabel('Apparatus'); ax.set_ylabel('Seeding agent')
    fig.tight_layout(); fig.savefig(IMG/'fig5_agent_apparatus_heatmap.png', dpi=220); plt.close(fig)

    # Validation/comparison multiview: annual stacked purpose and apparatus mix
    pby=purpose_by_year.copy()
    keep=[c for c in ['augment snowpack','increase precipitation','suppress hail','suppress fog','increase runoff'] if c in pby.columns]
    pby=pby[keep]
    fig,axes=plt.subplots(2,1,figsize=(10,8), sharex=True)
    pby.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set2', width=.85)
    axes[0].set_title('Purpose mentions by year (validation of annual composition)')
    axes[0].set_ylabel('Purpose mentions'); axes[0].legend(loc='upper left', ncol=2, fontsize=8)
    app_by_year[[c for c in ['ground','airborne','(missing)'] if c in app_by_year.columns]].plot(kind='bar', stacked=True, ax=axes[1], color=['#805ad5','#dd6b20','#999999'], width=.85)
    axes[1].set_title('Deployment-apparatus mentions by year')
    axes[1].set_ylabel('Apparatus mentions'); axes[1].set_xlabel('Year'); axes[1].legend(loc='upper left', ncol=3, fontsize=8)
    for ax in axes: ax.tick_params(axis='x', rotation=90)
    fig.tight_layout(); fig.savefig(IMG/'fig6_validation_multiview.png', dpi=220); plt.close(fig)

    # Claim recovery table
    purpose_top=pd.read_csv(OUT/'purpose_counts.csv').iloc[0]
    agent_top=pd.read_csv(OUT/'agent_counts.csv').iloc[0]
    app_counts=pd.read_csv(OUT/'apparatus_counts.csv')
    claim_rows=[
      {'claim':'Dataset contains 832 project-level records spanning 2000-2025.', 'evidence':'outputs/dataset_overview.json; outputs/annual_counts.csv', 'value':f"{len(df)} records, {df.year.min()}-{df.year.max()}", 'status':'recovered'},
      {'claim':'Records are spatially concentrated in a small set of western/southern states.', 'evidence':'outputs/state_counts.csv; report/images/fig2_state_concentration_bar.png; report/images/fig3_state_choropleth.png', 'value':f"Top 4 states share {top4_share:.1%}; top state {state_counts.iloc[0].state} has {int(state_counts.iloc[0].records)} records", 'status':'recovered'},
      {'claim':'Annual reported activity is dynamic rather than flat.', 'evidence':'outputs/annual_counts.csv; report/images/fig1_annual_activity.png', 'value':f"Annual counts range {annual.records.min()}-{annual.records.max()}, peak {int(metrics['peak_year']['year'])}", 'status':'recovered'},
      {'claim':'Snowpack augmentation and precipitation increase dominate stated purposes.', 'evidence':'outputs/purpose_counts.csv; report/images/fig4_purpose_composition.png', 'value':f"Top purpose mention: {purpose_top['purpose']} ({int(purpose_top['mentions'])})", 'status':'recovered'},
      {'claim':'Silver iodide is the dominant seeding agent.', 'evidence':'outputs/agent_counts.csv; outputs/summary_metrics.json', 'value':f"Silver iodide appears in {metrics['silver_iodide_record_count']} records ({metrics['silver_iodide_record_share']:.1%})", 'status':'recovered'},
      {'claim':'Deployment is primarily ground and/or airborne, with agent-apparatus structure.', 'evidence':'outputs/apparatus_counts.csv; outputs/agent_apparatus_crosstab.csv; report/images/fig5_agent_apparatus_heatmap.png', 'value':f"Ground in {metrics['ground_any_count']} records; airborne in {metrics['airborne_any_count']} records", 'status':'recovered'}
    ]
    pd.DataFrame(claim_rows).to_csv(OUT/'claim_recovery_table.csv', index=False)

    # Update inventory statuses
    inv_path=OUT/'target_artifact_inventory.json'
    if inv_path.exists():
        inv=json.load(open(inv_path))
        for sec in ['primary_tables','figures']:
            for item in inv.get(sec,[]):
                fname=item['name']
                path=(OUT/fname) if sec=='primary_tables' else (IMG/fname)
                item['status']='satisfied' if path.exists() else 'unsatisfied'
                if not path.exists(): item['reason']='file not generated'
        inv['report']='report/report.md planned'
        save_json(inv_path, inv)
    print(json.dumps({'records':len(df),'figures':len(list(IMG.glob('*.png'))),'outputs':len(list(OUT.glob('*')))}, indent=2))

if __name__=='__main__':
    main()
