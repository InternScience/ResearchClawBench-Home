import pandas as pd
import numpy as np
import glob

# Global statistics
df_global = pd.read_csv('data/glambie/results/calendar_years/0_global.csv')

total_loss_gt = df_global['combined_gt'].sum()
total_loss_gt_err = np.sqrt((df_global['combined_gt_errors']**2).sum())

total_loss_mwe = df_global['combined_mwe'].sum()
total_loss_mwe_err = np.sqrt((df_global['combined_mwe_errors']**2).sum())

avg_annual_loss_gt = df_global['combined_gt'].mean()
avg_annual_loss_gt_err = total_loss_gt_err / len(df_global)

print("Global Statistics (2000-2023):")
print(f"Total Mass Loss: {total_loss_gt:.2f} ± {total_loss_gt_err:.2f} Gt")
print(f"Total Specific Mass Loss: {total_loss_mwe:.2f} ± {total_loss_mwe_err:.2f} m w.e.")
print(f"Average Annual Mass Loss: {avg_annual_loss_gt:.2f} ± {avg_annual_loss_gt_err:.2f} Gt/yr")

# Decadal trends (2000-2009 vs 2010-2019)
df_2000s = df_global[(df_global['start_dates'] >= 2000) & (df_global['start_dates'] < 2010)]
df_2010s = df_global[(df_global['start_dates'] >= 2010) & (df_global['start_dates'] < 2020)]

loss_2000s = df_2000s['combined_gt'].mean()
loss_2010s = df_2010s['combined_gt'].mean()

print("\nDecadal Trends:")
print(f"Average Annual Loss (2000-2009): {loss_2000s:.2f} Gt/yr")
print(f"Average Annual Loss (2010-2019): {loss_2010s:.2f} Gt/yr")

# Regional statistics
files = glob.glob('data/glambie/results/calendar_years/*.csv')
files = [f for f in files if '0_global.csv' not in f]

regional_stats = []
for f in files:
    df = pd.read_csv(f)
    region_name = f.split('/')[-1].replace('.csv', '').split('_', 1)[1]
    loss_gt = df['combined_gt'].sum()
    loss_gt_err = np.sqrt((df['combined_gt_errors']**2).sum())
    loss_mwe = df['combined_mwe'].sum()
    regional_stats.append({
        'region': region_name,
        'total_loss_gt': loss_gt,
        'total_loss_gt_err': loss_gt_err,
        'total_loss_mwe': loss_mwe
    })

df_regions = pd.DataFrame(regional_stats).sort_values('total_loss_gt')

print("\nTop 5 Regions by Total Mass Loss (Gt):")
for _, row in df_regions.head(5).iterrows():
    print(f"{row['region']}: {row['total_loss_gt']:.2f} ± {row['total_loss_gt_err']:.2f} Gt")

print("\nTop 5 Regions by Specific Mass Loss (m w.e.):")
df_regions_mwe = df_regions.sort_values('total_loss_mwe')
for _, row in df_regions_mwe.head(5).iterrows():
    print(f"{row['region']}: {row['total_loss_mwe']:.2f} m w.e.")

