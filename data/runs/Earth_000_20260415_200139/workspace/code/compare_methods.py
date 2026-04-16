import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# We use hydrological years for this because they contain the method columns
hydro_files = glob.glob('data/glambie/results/hydrological_years/*.csv')

methods = ['altimetry', 'gravimetry', 'demdiff_and_glaciological']
method_totals = {m: [] for m in methods}
method_errors = {m: [] for m in methods}

# We want to compare methods globally over time, but not all methods are available for all regions at all times.
# We can sum up the available Gt per method across all regions for each year.
# Let's create a DataFrame with years as index.
years = np.arange(2000, 2024)
global_method_gt = pd.DataFrame(index=years, columns=methods).fillna(0.0)
global_method_err = pd.DataFrame(index=years, columns=methods).fillna(0.0)

for f in hydro_files:
    df = pd.read_csv(f)
    # We round start_dates to map to our years index
    df['year'] = df['start_dates'].round().astype(int)
    
    for m in methods:
        col = f'{m}_gt'
        err_col = f'{m}_gt_errors'
        
        if col in df.columns:
            # Drop NaNs
            valid = df.dropna(subset=[col])
            for _, row in valid.iterrows():
                y = int(round(row['year']))
                if y in global_method_gt.index:
                    global_method_gt.loc[y, m] += row[col]
                    # Errors in quadrature
                    global_method_err.loc[y, m] = np.sqrt(global_method_err.loc[y, m]**2 + row[err_col]**2)

fig, ax = plt.subplots(figsize=(12, 6))

colors = {'altimetry': 'blue', 'gravimetry': 'orange', 'demdiff_and_glaciological': 'green'}
labels = {'altimetry': 'Altimetry', 'gravimetry': 'Gravimetry', 'demdiff_and_glaciological': 'DEM Diff & Glaciological'}

for m in methods:
    # Only plot if we have data for that method
    if global_method_gt[m].abs().sum() > 0:
        ax.plot(global_method_gt.index, global_method_gt[m], marker='o', label=labels[m], color=colors[m])
        ax.fill_between(global_method_gt.index, 
                        global_method_gt[m] - global_method_err[m],
                        global_method_gt[m] + global_method_err[m],
                        alpha=0.2, color=colors[m])

# Also plot combined
df_global = pd.read_csv('data/glambie/results/calendar_years/0_global.csv')
ax.plot(df_global['start_dates'], df_global['combined_gt'], 'k--', linewidth=2, label='Combined (Consensus)')
ax.fill_between(df_global['start_dates'], 
                df_global['combined_gt'] - df_global['combined_gt_errors'],
                df_global['combined_gt'] + df_global['combined_gt_errors'],
                alpha=0.2, color='k')

ax.set_xlabel('Year')
ax.set_ylabel('Mass Change (Gt)')
ax.set_title('Comparison of Observational Methods (Global Aggregation)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('report/images/methods_comparison.png')
print("Saved report/images/methods_comparison.png")
