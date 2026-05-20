"""
Process GlaMBIE input files: parse, convert units to m w.e. using time-varying area,
annualize, and save.
"""
import os
import pandas as pd
import numpy as np
import json

INPUT_DIR = 'data/glambie/input'
AREA_FILE = 'outputs/region_areas.json'
OFFICIAL_DIR = 'data/glambie/results/calendar_years'
OUTPUT_FILE = 'outputs/annualized_inputs_mwe.csv'

DENSITY_FACTOR = 0.85  # m w.e. per m ice equivalent
MWE_TO_GT_FACTOR = 0.000997

with open(AREA_FILE) as f:
    area_dict = json.load(f)

region_map = {}
for k in area_dict:
    parts = k.split('_', 1)
    if len(parts) == 2:
        region_map[parts[1]] = k
    else:
        region_map[k] = k

# Load official area time series per region
area_ts = {}
for fname in os.listdir(OFFICIAL_DIR):
    if fname.endswith('.csv'):
        region_key = fname.replace('.csv', '')
        dfa = pd.read_csv(os.path.join(OFFICIAL_DIR, fname))
        dfa = dfa[(dfa['start_dates'] >= 1990) & (dfa['start_dates'] <= 2024)]
        area_ts[region_key] = dict(zip(dfa['start_dates'].astype(int), dfa['glacier_area']))

def get_area(region_key, year):
    ts = area_ts.get(region_key, {})
    return ts.get(int(year), np.nan)

records = []

for root, dirs, files in os.walk(INPUT_DIR):
    for fname in files:
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(root, fname)
        method = None
        for m in ['glaciological', 'demdiff', 'altimetry', 'gravimetry', 'combined']:
            if m in fname:
                method = m
                break
        if method is None:
            continue
        pos = fname.find(f'_{method}_')
        if pos == -1:
            continue
        region_name = fname[:pos]
        region_key = region_map.get(region_name)
        if region_key is None:
            for k in area_dict:
                if k.endswith(region_name):
                    region_key = k
                    break
        if region_key is None:
            print('Warning: no area for', region_name, fname)
            continue

        df = pd.read_csv(fpath)
        if df.empty:
            continue
        unit = str(df['unit'].iloc[0]).strip().lower()
        author = str(df['author'].iloc[0]).strip()

        for _, row in df.iterrows():
            start = float(row['start_dates'])
            end = float(row['end_dates'])
            change = float(row['changes'])
            error = float(row['errors'])
            dt = end - start
            if dt <= 0:
                continue

            # Determine conversion to m w.e.
            if unit == 'mwe':
                change_mwe = change
                error_mwe = error
            elif unit == 'm':
                change_mwe = change * DENSITY_FACTOR
                error_mwe = error * DENSITY_FACTOR
            elif unit in ('gt', 'Gt'):
                # Use average area over the interval
                y_start = int(np.floor(start))
                y_end = int(np.floor(end))
                areas = [get_area(region_key, y) for y in range(y_start, y_end + 1)]
                areas = [a for a in areas if not np.isnan(a)]
                if len(areas) == 0:
                    continue
                avg_area = np.mean(areas)
                change_mwe = change / (avg_area * MWE_TO_GT_FACTOR)
                error_mwe = error / (avg_area * MWE_TO_GT_FACTOR)
            else:
                continue

            y_start = int(np.floor(start))
            y_end = int(np.floor(end))
            for y in range(y_start, y_end + 1):
                overlap_start = max(start, float(y))
                overlap_end = min(end, float(y + 1))
                overlap = overlap_end - overlap_start
                if overlap <= 0:
                    continue
                frac = overlap / dt
                records.append({
                    'region': region_key,
                    'method': method,
                    'author': author,
                    'year': y,
                    'change_mwe': change_mwe * frac,
                    'error_mwe': error_mwe * frac,
                    'overlap': overlap
                })

out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_FILE, index=False)
print('Saved', OUTPUT_FILE, 'with', len(out_df), 'rows')
