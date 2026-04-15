import pandas as pd
df = pd.read_excel('data/CS2_36/CS2_36_1_28_11.xlsx', sheet_name='Channel_1-009')
print(df.columns)
print(df.head())

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat

# Load CS2_36
xlsx_files = ['data/CS2_36/CS2_36_1_28_11.xlsx']  # pick one
for f in xlsx_files:
    print(f'Loading {f}')
    xl = pd.ExcelFile(f)
    print('Sheets:', xl.sheet_names)
    # Assume first sheet has data, or find discharge
    df = pd.read_excel(f, sheet_name=0)
    print(df.head())
    print(df.columns)
    # Plot V vs capacity or time
    if 'Voltage(V)' in df.columns and 'Capacity(Ah)' in df.columns:
        plt.figure()
        plt.plot(df['Capacity(Ah)'], df['Voltage(V)'])
        plt.savefig('outputs/cs2_vq.png')
        print('Saved cs2_vq.png')

# NASA B0005
mat = loadmat('data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/B0005.mat')
cycles = mat['cycle'][0]
discharges = []
for i, cyc in enumerate(cycles):
    if cyc['type'][0][0] == 'discharge':
        data = cyc['data'][0][0]
        time = data['Time'][0]
        volt = data['Voltage_measured'][0][0]
        cap = data['Capacity'][0][0][-1] if 'Capacity' in data.dtype.names else np.trapz(-data['Current_measured'][0][0], time)/3600
        discharges.append((i, volt, time, cap))
print(f'Found {len(discharges)} discharges in B0005')
# Plot first few
for j in range(min(3, len(discharges))):
    _, v, t, cap = discharges[j]
    plt.figure()
    plt.plot(t/3600, v)
    plt.title(f'B0005 discharge {j+1}, cap={cap:.3f}Ah')
    plt.savefig(f'outputs/nasa_disch{j}.png')

# Oxford Example
ox = loadmat('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat')
print('Oxford keys:', list(ox.keys()))
if 'ExampleDC_C1' in ox:
    dc = ox['ExampleDC_C1']
    print(dc.dtype.names)
