"""Build comparison vs official GlaMBIE consensus and write CSV."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_000_20260427_143635')
RES = ROOT/'data/glambie/results/calendar_years'
OUT = ROOT/'outputs'

reg = pd.read_csv(OUT/'regional_annual_reconciled.csv')
glo = pd.read_csv(OUT/'global_annual_reconciled.csv')

# Official global
g = pd.read_csv(RES/'0_global.csv')
g['year'] = g.start_dates.astype(int)
g = g[g.year<=2023]
g_match = g.set_index('year')[['combined_gt','combined_gt_errors','combined_mwe','combined_mwe_errors']]

cmp = glo.merge(g_match, left_on='year', right_index=True)
cmp = cmp.rename(columns={'consensus_gt':'reconciled_gt',
                          'consensus_gt_err':'reconciled_gt_err',
                          'consensus_mwe':'reconciled_mwe',
                          'consensus_mwe_err':'reconciled_mwe_err',
                          'combined_gt':'glambie_gt',
                          'combined_gt_errors':'glambie_gt_err',
                          'combined_mwe':'glambie_mwe',
                          'combined_mwe_errors':'glambie_mwe_err'})
cmp['gt_diff'] = cmp['reconciled_gt'] - cmp['glambie_gt']
cmp['mwe_diff'] = cmp['reconciled_mwe'] - cmp['glambie_mwe']
cmp.to_csv(OUT/'comparison_global.csv', index=False)

# Per-region comparison
rows=[]
for rdir in sorted([p.name for p in RES.glob('*.csv') if p.name!='0_global.csv'], key=lambda s: int(s.split('_')[0])):
    rnum, rname = rdir.replace('.csv','').split('_',1)
    rnum = int(rnum)
    of = pd.read_csv(RES/rdir)
    of['year'] = of.start_dates.astype(int)
    of = of[of.year<=2023]
    rec = reg[reg.region==rname]
    if len(rec)==0: 
        # try basename matching
        cand = reg.region.unique()
        for c in cand:
            if c.startswith(rname.split('_')[0]):
                rec = reg[reg.region==c]; break
    m = rec.merge(of[['year','combined_gt','combined_gt_errors','combined_mwe','combined_mwe_errors','glacier_area']], on='year', how='inner')
    if len(m)==0: continue
    rmse_gt = np.sqrt(np.nanmean((m.consensus_gt - m.combined_gt)**2))
    rmse_mwe = np.sqrt(np.nanmean((m.consensus_mwe - m.combined_mwe)**2))
    bias_gt = np.nanmean(m.consensus_gt - m.combined_gt)
    bias_mwe = np.nanmean(m.consensus_mwe - m.combined_mwe)
    cor = np.corrcoef(m.consensus_gt.fillna(0), m.combined_gt.fillna(0))[0,1]
    rows.append(dict(region_num=rnum, region=rname,
                     mean_glambie_gt=m.combined_gt.mean(),
                     mean_reconciled_gt=m.consensus_gt.mean(),
                     bias_gt=bias_gt, rmse_gt=rmse_gt,
                     bias_mwe=bias_mwe, rmse_mwe=rmse_mwe,
                     correlation=cor,
                     cumulative_glambie_gt=m.combined_gt.sum(),
                     cumulative_reconciled_gt=m.consensus_gt.sum()))
pd.DataFrame(rows).sort_values('region_num').to_csv(OUT/'comparison_vs_glambie.csv', index=False)
print(pd.DataFrame(rows).sort_values('region_num').to_string(index=False))
print('\nGLOBAL:')
print(cmp[['year','reconciled_gt','glambie_gt','gt_diff','reconciled_mwe','glambie_mwe','mwe_diff']].to_string(index=False))
