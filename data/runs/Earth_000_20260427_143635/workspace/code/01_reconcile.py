"""
GlaMBIE reconciliation pipeline.

Steps:
  1. For each input CSV (per region, per source), resample the change
     time-series onto a common monthly grid (Jan-2000 .. Dec-2023) using
     proportional allocation (total change between start_date and end_date
     is spread uniformly across the months it intersects).
  2. Convert all values to consistent specific mass change (m w.e.) using
     the GlaMBIE regional area time series and a glacier density of 850
     kg/m^3 implicitly (most files already in m w.e.; Gt files use the
     GlaMBIE area).
  3. Aggregate monthly increments to calendar-year sums per source, then
     form group-level estimates per method as inverse-variance weighted
     means (with empirical between-source std added to the formal error).
  4. Combine method groups into a single regional consensus following the
     GlaMBIE three-group structure (altimetry, gravimetry,
     demdiff_and_glaciological), again with inverse-variance weighting.
  5. Aggregate to global by summing regional Gt and area-weighting m w.e.
"""
from __future__ import annotations
import os, glob, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Earth_000_20260427_143635')
DATA = ROOT/'data/glambie'
INP  = DATA/'input'
RES  = DATA/'results'
OUT  = ROOT/'outputs'
OUT.mkdir(exist_ok=True)

REGION_DIRS = sorted([d.name for d in INP.iterdir() if d.is_dir()])

# Region number -> short name and result CSV name
def parse_region(dirname):
    m = re.match(r'(\d+)_(.+)', dirname)
    return int(m.group(1)), m.group(2)

# ---------- Common monthly grid ----------
# Use 12*24 = 288 months Jan-2000 to Dec-2023
N_MONTHS = 12*24
MONTH_STARTS = 2000.0 + np.arange(N_MONTHS)/12.0
MONTH_ENDS   = MONTH_STARTS + 1.0/12.0
YEARS = np.arange(2000, 2024)

def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def resample_to_monthly(df: pd.DataFrame):
    """Distribute each row's 'changes' over the monthly grid proportional to overlap.
    Returns array length N_MONTHS of monthly increments and squared-error array.
    Values outside 2000-2024 are dropped.
    """
    monthly = np.zeros(N_MONTHS)
    monthly_var = np.zeros(N_MONTHS)
    for _, r in df.iterrows():
        s, e = float(r.start_dates), float(r.end_dates)
        dur = e - s
        if dur <= 0: continue
        change = float(r.changes)
        err    = float(r.errors) if r.errors == r.errors else 0.0
        # Iterate only over months that could overlap
        i0 = max(0, int(np.floor((s - 2000.0)*12)))
        i1 = min(N_MONTHS, int(np.ceil((e - 2000.0)*12)))
        for i in range(i0, i1):
            ov = overlap(s, e, MONTH_STARTS[i], MONTH_ENDS[i])
            if ov <= 0: continue
            frac = ov/dur
            monthly[i] += change * frac
            # Treat error scaling: if input is over an interval of duration dur,
            # we assume errors are independent across non-overlapping intervals
            # of the same source -> variance scales by frac^2 (same source)
            monthly_var[i] += (err*frac)**2
    return monthly, monthly_var

def annual_from_monthly(monthly, monthly_var):
    """Aggregate 288 monthly values into 24 annual sums."""
    monthly = monthly.reshape(24, 12)
    monthly_var = monthly_var.reshape(24, 12)
    return monthly.sum(axis=1), monthly_var.sum(axis=1)

# ---------- Region area time series (annual, Gt<->mwe) ----------
def region_area_series(region_dir):
    """Get annual mean glacier area (km^2) for each calendar year 2000-2023.
    Uses GlaMBIE calendar_years result file as authoritative area record.
    """
    rnum, rname = parse_region(region_dir)
    f = RES/'calendar_years'/f'{rnum}_{rname}.csv'
    df = pd.read_csv(f)
    # Years correspond to start_dates 2000.0 -> 2023.0
    area = pd.Series(df.glacier_area.values, index=df.start_dates.values.astype(int))
    out = np.array([area.get(y, np.nan) for y in YEARS])
    return out  # km^2

# ---------- Source-level time series ----------
METHOD_TOKENS = ['altimetry','demdiff','glaciological','gravimetry','combined']

def classify_method(fname):
    base = os.path.basename(fname).lower()
    for m in METHOD_TOKENS:
        if f'_{m}_' in base:
            return m
    return 'unknown'

def load_source_annual(path, area_km2):
    df = pd.read_csv(path)
    if len(df)==0: return None, None
    unit = df.unit.iloc[0].lower()
    monthly, monthly_var = resample_to_monthly(df)
    ann_change, ann_var = annual_from_monthly(monthly, monthly_var)
    # Convert to mwe and Gt
    if unit in ('m', 'mwe'):
        mwe = ann_change
        mwe_err = np.sqrt(ann_var)
        # Convert to Gt: dM[Gt] = mwe * area[km^2] * 1e6 m^2/km^2 * 1000 kg/m^3 / 1e12 = mwe*area/1000
        gt = mwe * area_km2 / 1000.0
        gt_err = mwe_err * area_km2 / 1000.0
    elif unit in ('gt',):
        gt = ann_change
        gt_err = np.sqrt(ann_var)
        mwe = gt * 1000.0 / area_km2
        mwe_err = gt_err * 1000.0 / area_km2
    else:
        return None, None
    return (mwe, mwe_err), (gt, gt_err)

# ---------- Group reconciliation ----------
def inverse_variance_combine(values, errors):
    """For each year, combine N estimates with their errors via inverse-variance.
    values, errors: shape (N_sources, 24)  (NaNs allowed where source absent)
    Returns mean (24,), error (24,), n_used (24,)
    """
    vals = np.array(values)
    errs = np.array(errors)
    out = np.full(vals.shape[1], np.nan)
    out_err = np.full(vals.shape[1], np.nan)
    n = np.zeros(vals.shape[1], dtype=int)
    for t in range(vals.shape[1]):
        v = vals[:, t]; e = errs[:, t]
        mask = np.isfinite(v) & np.isfinite(e) & (e > 0)
        if mask.sum() == 0: continue
        vv = v[mask]; ee = e[mask]
        w = 1.0/ee**2
        m = (w*vv).sum()/w.sum()
        # Formal combined error
        s_form = np.sqrt(1.0/w.sum())
        # Empirical between-source spread (population std)
        if mask.sum() >= 2:
            s_emp = np.sqrt(((vv - m)**2 * w).sum()/w.sum())
        else:
            s_emp = 0.0
        # Total uncertainty: max of formal and empirical, plus sqrt-quad combination
        s_tot = np.sqrt(s_form**2 + s_emp**2)
        out[t] = m
        out_err[t] = s_tot
        n[t] = int(mask.sum())
    return out, out_err, n

def fill_with_default(arr, default):
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = default
    return arr

# ---------- Main per-region pipeline ----------
def process_region(region_dir):
    rnum, rname = parse_region(region_dir)
    area_km2 = region_area_series(region_dir)
    files = sorted(glob.glob(str(INP/region_dir/'*.csv')))
    src_records = []  # rows: dict per source
    method_groups = {m: {'mwe':[], 'mwe_err':[], 'gt':[], 'gt_err':[], 'authors':[]}
                     for m in METHOD_TOKENS}
    for f in files:
        method = classify_method(f)
        if method == 'unknown': continue
        try:
            (mwe, mwe_err), (gt, gt_err) = load_source_annual(f, area_km2)
        except Exception as e:
            print('WARN', f, e); continue
        if mwe is None: continue
        # Mask years with no coverage (zeros from resample where source absent)
        # Detect coverage via original date range
        df = pd.read_csv(f)
        s_min, s_max = df.start_dates.min(), df.end_dates.max()
        coverage = ((YEARS+1) > s_min) & (YEARS < s_max)
        # also require at least 6 months of overlap to count as coverage in that year
        # use monthly to check
        monthly, _ = resample_to_monthly(df)
        # year coverage = number of months with non-zero contribution
        nonzero_months = (monthly != 0).reshape(24,12).sum(axis=1)
        # Hugonnet etc. provide year-spread of -3.26 m over 5 years -> nonzero in all months
        coverage = nonzero_months >= 3
        mwe_m = np.where(coverage, mwe, np.nan)
        gt_m  = np.where(coverage, gt,  np.nan)
        # If errors are zero in the row (e.g. a few altimetry pairs) keep nan
        e_mwe = np.where(coverage & (mwe_err>0), mwe_err, np.nan)
        e_gt  = np.where(coverage & (gt_err >0), gt_err,  np.nan)
        method_groups[method]['mwe'].append(mwe_m)
        method_groups[method]['mwe_err'].append(e_mwe)
        method_groups[method]['gt'].append(gt_m)
        method_groups[method]['gt_err'].append(e_gt)
        method_groups[method]['authors'].append(os.path.basename(f))
        src_records.append({'region':rname, 'method':method, 'file':os.path.basename(f),
                            'first_year_covered': int(YEARS[coverage].min()) if coverage.any() else None,
                            'last_year_covered':  int(YEARS[coverage].max()) if coverage.any() else None,
                            'n_years_covered': int(coverage.sum())})
    # Per-method group
    method_summary = {}
    for m in METHOD_TOKENS:
        g = method_groups[m]
        if not g['mwe']:
            method_summary[m] = None; continue
        mwe_mean, mwe_err, n_mwe = inverse_variance_combine(g['mwe'], g['mwe_err'])
        gt_mean,  gt_err,  n_gt  = inverse_variance_combine(g['gt'],  g['gt_err'])
        method_summary[m] = dict(
            mwe=mwe_mean, mwe_err=mwe_err, gt=gt_mean, gt_err=gt_err,
            n_sources=n_mwe, n_files=len(g['mwe']),
            authors=g['authors'])
    # Build combined consensus following GlaMBIE 3-group structure
    # Group A = altimetry, Group B = gravimetry, Group C = demdiff+glaciological
    # 'combined' (hybrid) entries are treated as a fourth comparator for sanity
    groupA = method_summary.get('altimetry')
    groupB = method_summary.get('gravimetry')
    # Combine demdiff & glaciological into Group C
    src_C_mwe, src_C_err = [], []
    src_C_gt, src_C_gterr = [], []
    for m in ('demdiff', 'glaciological'):
        s = method_summary.get(m)
        if s is None: continue
        src_C_mwe.append(s['mwe']); src_C_err.append(s['mwe_err'])
        src_C_gt.append(s['gt']);   src_C_gterr.append(s['gt_err'])
    if src_C_mwe:
        Cm, Cme, _ = inverse_variance_combine(src_C_mwe, src_C_err)
        Cg, Cge, _ = inverse_variance_combine(src_C_gt,  src_C_gterr)
        groupC = dict(mwe=Cm, mwe_err=Cme, gt=Cg, gt_err=Cge)
    else:
        groupC = None

    # Combine groups A,B,C with inverse-variance weighting
    g_mwe, g_mwe_err = [], []
    g_gt,  g_gt_err  = [], []
    g_names = []
    for nm, gs in [('altimetry', groupA), ('gravimetry', groupB), ('demdiff_glaciological', groupC)]:
        if gs is None: continue
        g_mwe.append(gs['mwe']); g_mwe_err.append(gs['mwe_err'])
        g_gt .append(gs['gt']);  g_gt_err .append(gs['gt_err'])
        g_names.append(nm)
    if not g_mwe:
        return None, None, src_records, method_summary
    cons_mwe, cons_mwe_err, n_groups = inverse_variance_combine(g_mwe, g_mwe_err)
    cons_gt,  cons_gt_err,  _        = inverse_variance_combine(g_gt,  g_gt_err)
    # If we got data only at the per-method level, also fall back to combined-only
    consensus_table = pd.DataFrame({
        'year': YEARS, 'region': rname,
        'glacier_area_km2': area_km2,
        'consensus_mwe': cons_mwe, 'consensus_mwe_err': cons_mwe_err,
        'consensus_gt':  cons_gt,  'consensus_gt_err':  cons_gt_err,
        'n_groups': n_groups,
    })
    # add per-group columns
    for nm, gs in [('altimetry', groupA), ('gravimetry', groupB), ('demdiff_glaciological', groupC)]:
        if gs is None:
            consensus_table[f'{nm}_mwe'] = np.nan
            consensus_table[f'{nm}_mwe_err'] = np.nan
            consensus_table[f'{nm}_gt']  = np.nan
            consensus_table[f'{nm}_gt_err']  = np.nan
        else:
            consensus_table[f'{nm}_mwe'] = gs['mwe']
            consensus_table[f'{nm}_mwe_err'] = gs['mwe_err']
            consensus_table[f'{nm}_gt']  = gs['gt']
            consensus_table[f'{nm}_gt_err']  = gs['gt_err']
    # Per-method mean (for diagnostics)
    for m in METHOD_TOKENS:
        s = method_summary[m]
        if s is None: continue
        consensus_table[f'{m}_mean_mwe'] = s['mwe']
        consensus_table[f'{m}_mean_mwe_err'] = s['mwe_err']
        consensus_table[f'{m}_n_sources'] = s['n_sources']
    return consensus_table, method_summary, src_records, method_summary

# ---------- Run ----------
all_regions = []
all_sources = []
method_summaries_all = {}
for rdir in REGION_DIRS:
    print('Processing', rdir, flush=True)
    cons, _msum, recs, msum = process_region(rdir)
    if cons is None: continue
    all_regions.append(cons)
    all_sources.extend(recs)
    method_summaries_all[rdir] = msum

reg = pd.concat(all_regions, ignore_index=True)
reg.to_csv(OUT/'regional_annual_reconciled.csv', index=False)

src_df = pd.DataFrame(all_sources)
src_df.to_csv(OUT/'source_inventory.csv', index=False)

# ---------- Global aggregation ----------
# Sum Gt across regions (treat regions as independent)
glo = reg.groupby('year').agg(
    glacier_area_km2=('glacier_area_km2','sum'),
    consensus_gt=('consensus_gt','sum'),
).reset_index()
# Error: sqrt(sum sq)
err = reg.groupby('year').apply(lambda d: np.sqrt(np.nansum(d['consensus_gt_err']**2))).reset_index()
err.columns = ['year', 'consensus_gt_err']
glo = glo.merge(err, on='year')
# Global m w.e. = sum_gt * 1000 / area
glo['consensus_mwe'] = glo['consensus_gt']*1000.0/glo['glacier_area_km2']
glo['consensus_mwe_err'] = glo['consensus_gt_err']*1000.0/glo['glacier_area_km2']

# Cumulative
glo['cum_gt'] = glo['consensus_gt'].cumsum()
glo['cum_gt_err'] = np.sqrt((glo['consensus_gt_err']**2).cumsum())
glo['cum_mwe'] = glo['consensus_mwe'].cumsum()
glo['cum_mwe_err'] = np.sqrt((glo['consensus_mwe_err']**2).cumsum())

glo.to_csv(OUT/'global_annual_reconciled.csv', index=False)
print('Wrote outputs')
print(glo.to_string(index=False))
