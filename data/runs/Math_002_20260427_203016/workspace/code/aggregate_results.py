"""Aggregate results, build summary CSVs and tables."""
import json, os, csv
import numpy as np
import pandas as pd

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))

with open(os.path.join(OUT, 'results_per_instance.json')) as f:
    recs = json.load(f)

# Convert string booleans back if any
for r in recs:
    for k in ('pp_success', 'lnspp_success', 'hybrid_success'):
        if isinstance(r[k], str):
            r[k] = (r[k] == 'True')

df = pd.DataFrame(recs)
# Numeric coercions
for col in ['pp_time_s','pp_soc','pp_makespan','lnspp_time_s','lnspp_soc',
            'lnspp_makespan','lnspp_iters','hybrid_time_s','hybrid_soc',
            'hybrid_makespan','hybrid_iters','hybrid_train_s','wall_s',
            'cumulative_min','hybrid_qsize']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print('total instances', len(df))
df.to_csv(os.path.join(OUT, 'results_per_instance.csv'), index=False)

# ---- per (family, n_agents) summary ----
summary = (df
    .groupby(['family', 'n_agents'])
    .agg(
        n=('pp_success', 'count'),
        pp_succ=('pp_success', 'mean'),
        lns_succ=('lnspp_success', 'mean'),
        hyb_succ=('hybrid_success', 'mean'),
        pp_time=('pp_time_s', 'mean'),
        lns_time=('lnspp_time_s', 'mean'),
        hyb_time=('hybrid_time_s', 'mean'),
        pp_soc=('pp_soc', lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        lns_soc=('lnspp_soc', lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        hyb_soc=('hybrid_soc', lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        pp_make=('pp_makespan', lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        lns_make=('lnspp_makespan', lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        hyb_make=('hybrid_makespan', lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        lns_iters=('lnspp_iters', 'mean'),
        hyb_iters=('hybrid_iters', 'mean'),
    )
    .reset_index())
summary.to_csv(os.path.join(OUT, 'results_summary.csv'), index=False)
print(summary.to_string())

# Compact comparison tables
def pivot_metric(metric_pp, metric_lns, metric_hyb, name):
    tab = summary[['family','n_agents', metric_pp, metric_lns, metric_hyb]].copy()
    tab.columns = ['family', 'n_agents', 'PP', 'LNS-PP', 'LNS-Hybrid']
    tab.to_csv(os.path.join(OUT, f'table_{name}.csv'), index=False)

pivot_metric('pp_succ','lns_succ','hyb_succ','success_rate')
pivot_metric('pp_time','lns_time','hyb_time','runtime')
pivot_metric('pp_soc','lns_soc','hyb_soc','sum_of_costs')
pivot_metric('pp_make','lns_make','hyb_make','makespan')

# Aggregate per family
fam_summary = (df.groupby('family')
    .agg(
        n=('pp_success', 'count'),
        pp_succ=('pp_success', 'mean'),
        lns_succ=('lnspp_success', 'mean'),
        hyb_succ=('hybrid_success', 'mean'),
        pp_time=('pp_time_s', 'mean'),
        lns_time=('lnspp_time_s', 'mean'),
        hyb_time=('hybrid_time_s', 'mean'),
    )
    .reset_index())
fam_summary.to_csv(os.path.join(OUT, 'results_family_summary.csv'), index=False)
print('\n', fam_summary.to_string())

# Overall headline numbers
print('\n=== overall success rate ===')
print(f"  PP        : {df['pp_success'].mean():.3f}")
print(f"  LNS-PP    : {df['lnspp_success'].mean():.3f}")
print(f"  LNS-Hybrid: {df['hybrid_success'].mean():.3f}")
print(f'\noverall avg runtime (s):')
print(f"  PP        : {df['pp_time_s'].mean():.3f}")
print(f"  LNS-PP    : {df['lnspp_time_s'].mean():.3f}")
print(f"  LNS-Hybrid: {df['hybrid_time_s'].mean():.3f}")
