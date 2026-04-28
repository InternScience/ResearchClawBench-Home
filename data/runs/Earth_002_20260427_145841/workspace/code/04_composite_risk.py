"""
Step 4: Build composite risk index.

Components for each mangrove point and each scenario s:
  R_TC = standardized TC exposure (intense + 0.5*major + 0.25*total) -> normalized 0..1
        (TC regime-shift component is invariant to scenario in this dataset; we anchor on the
         historical baseline as a proxy for cyclone-regime sensitivity following Mo et al. 2023.)
  R_SLR(s) = SLR rate exceedance score combining the Saintilan thresholds:
        score = clamp01((rate - 4) / (10 - 4))   if 4 <= rate < 10
        score = 1.0                              if rate >= 10
        score = (rate / 4) * 0.5                 if 0 <= rate < 4   (linear from 0 to 0.5)
        score = 0                                if rate < 0
        At 7 mm/yr score = 0.5; at 10 mm/yr score = 1.0  (Saintilan likely/highly likely band).
  Composite: R(s) = 0.5*R_TC + 0.5*R_SLR(s)  (equal weighting)

We also compute a "risk class" per point per scenario:
  high   : R(s) >= 0.6 OR slr_rate >= 7 OR tc_intense_per_decade >= 1
  medium : 0.3 <= R(s) < 0.6
  low    : R(s) < 0.3
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
df = pd.read_csv(os.path.join(ROOT, 'outputs/mangrove_with_tc_slr.csv'))

# ----- TC component (scenario-invariant baseline) -----
tc_score_raw = (df['tc_intense_per_decade'].astype(float)
                + 0.5 * (df['tc_major_within_200km'] - df['tc_intense_within_200km']) * 10.0 / 165.0
                + 0.25 * (df['tc_pts_within_200km'] - df['tc_major_within_200km']) * 10.0 / 165.0)

# Robust normalisation: divide by 95th percentile then clip.
p95 = np.nanpercentile(tc_score_raw, 95)
print(f'TC raw score 95th pct: {p95:.4f}')
df['tc_risk'] = np.clip(tc_score_raw / max(p95, 1e-6), 0, 1)

def slr_score(rate):
    rate = np.asarray(rate, dtype=float)
    s = np.zeros_like(rate)
    s = np.where(rate < 0, 0, s)
    mask1 = (rate >= 0) & (rate < 4)
    s = np.where(mask1, (rate / 4.0) * 0.4, s)
    mask2 = (rate >= 4) & (rate < 10)
    s = np.where(mask2, 0.4 + (rate - 4.0) / (10.0 - 4.0) * 0.4, s)
    mask3 = (rate >= 10) & (rate < 15)
    s = np.where(mask3, 0.8 + (rate - 10.0) / (15.0 - 10.0) * 0.2, s)
    s = np.where(rate >= 15, 1.0, s)
    return s

scenarios = ['ssp245', 'ssp370', 'ssp585']
year = 2100
for sc in scenarios:
    rate_col = f'slr_rate_mm_yr_{sc}_{year}'
    df[f'slr_risk_{sc}'] = slr_score(df[rate_col].values)
    df[f'composite_risk_{sc}'] = 0.5 * df['tc_risk'].values + 0.5 * df[f'slr_risk_{sc}'].values

# Threshold tags for Saintilan limits per scenario
for sc in scenarios:
    rate_col = f'slr_rate_mm_yr_{sc}_{year}'
    df[f'slr_above4_{sc}'] = (df[rate_col] >= 4).astype(int)
    df[f'slr_above7_{sc}'] = (df[rate_col] >= 7).astype(int)
    # Risk class
    r = df[f'composite_risk_{sc}'].values
    cls = np.where(r >= 0.7, 'very_high',
            np.where(r >= 0.5, 'high',
              np.where(r >= 0.35, 'medium', 'low')))
    df[f'risk_class_{sc}'] = cls

# Quick global summary
summary = []
for sc in scenarios:
    summary.append({
        'scenario': sc,
        'n_points': len(df),
        'mean_composite_risk': float(df[f'composite_risk_{sc}'].mean()),
        'frac_very_high_risk': float((df[f'risk_class_{sc}'] == 'very_high').mean()),
        'frac_high_risk': float((df[f'risk_class_{sc}'] == 'high').mean()),
        'frac_medium_risk': float((df[f'risk_class_{sc}'] == 'medium').mean()),
        'frac_low_risk': float((df[f'risk_class_{sc}'] == 'low').mean()),
        'frac_above_4mm_yr': float(df[f'slr_above4_{sc}'].mean()),
        'frac_above_7mm_yr': float(df[f'slr_above7_{sc}'].mean()),
        'median_slr_rate_mm_yr_2100': float(df[f'slr_rate_mm_yr_{sc}_2100'].median()),
        'mean_slr_rate_mm_yr_2100': float(df[f'slr_rate_mm_yr_{sc}_2100'].mean()),
    })

# TC-only stats are scenario-invariant in this dataset
tc_summary = {
    'mean_tc_risk': float(df['tc_risk'].mean()),
    'frac_intense_storm_exposed': float((df['tc_intense_per_decade'] > 0).mean()),
    'frac_intense_storm_high': float((df['tc_intense_per_decade'] >= 1.0).mean()),
    'frac_no_storm_baseline': float((df['tc_pts_within_200km'] == 0).mean()),
}

s_df = pd.DataFrame(summary)
s_df.to_csv(os.path.join(ROOT, 'outputs/scenario_comparison_global.csv'), index=False)
print(s_df.to_string(index=False))
print('\nTC-only baseline stats:', tc_summary)
import json
with open(os.path.join(ROOT, 'outputs/tc_baseline_summary.json'),'w') as f:
    json.dump(tc_summary, f, indent=2)

df.to_csv(os.path.join(ROOT, 'outputs/mangrove_point_risk.csv'), index=False)
print('Saved outputs/mangrove_point_risk.csv')
