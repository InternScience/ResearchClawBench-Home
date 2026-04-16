import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('report/images', exist_ok=True)

df_global = pd.read_csv('data/glambie/results/calendar_years/0_global.csv')

# Calculate cumulative sum
df_global['cumulative_gt'] = df_global['combined_gt'].cumsum()
df_global['cumulative_mwe'] = df_global['combined_mwe'].cumsum()

# For errors, assuming they are independent, we add them in quadrature
import numpy as np
df_global['cumulative_gt_errors'] = np.sqrt((df_global['combined_gt_errors']**2).cumsum())
df_global['cumulative_mwe_errors'] = np.sqrt((df_global['combined_mwe_errors']**2).cumsum())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot Cumulative Gt
ax1.plot(df_global['end_dates'], df_global['cumulative_gt'], 'b-', label='Cumulative Mass Change (Gt)')
ax1.fill_between(df_global['end_dates'], 
                 df_global['cumulative_gt'] - df_global['cumulative_gt_errors'],
                 df_global['cumulative_gt'] + df_global['cumulative_gt_errors'], 
                 color='b', alpha=0.2, label='Uncertainty')
ax1.set_ylabel('Cumulative Mass Change (Gt)')
ax1.set_title('Global Cumulative Glacial Mass Change (2000-2023)')
ax1.grid(True)
ax1.legend()

# Plot Cumulative m w.e.
ax2.plot(df_global['end_dates'], df_global['cumulative_mwe'], 'r-', label='Cumulative Specific Mass Change (m w.e.)')
ax2.fill_between(df_global['end_dates'], 
                 df_global['cumulative_mwe'] - df_global['cumulative_mwe_errors'],
                 df_global['cumulative_mwe'] + df_global['cumulative_mwe_errors'], 
                 color='r', alpha=0.2, label='Uncertainty')
ax2.set_xlabel('Year')
ax2.set_ylabel('Cumulative Specific Mass Change (m w.e.)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('report/images/cumulative_mass_change.png')
print("Saved report/images/cumulative_mass_change.png")

# Calculate total loss
total_gt = df_global['cumulative_gt'].iloc[-1]
total_gt_err = df_global['cumulative_gt_errors'].iloc[-1]
print(f"Total Mass Loss: {total_gt:.2f} +/- {total_gt_err:.2f} Gt")
