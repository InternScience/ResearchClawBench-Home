import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.integrate import cumtrapz

# CS2 extract discharge
df = pd.read_excel('data/CS2_36/CS2_36_1_28_11.xlsx', sheet_name='Channel_1-009')
time = df['Test_Time(s)'].values
volt = df['Voltage(V)'].values
curr = df['Current(A)'].values

# Find discharge segments (curr ~ -1)
disch_mask = (curr < -0.5)
q_disch = -cumtrapz(curr[disch_mask], time[disch_mask], initial=0)/3600  # Ah
plt.figure()
plt.plot(q_disch, volt[disch_mask])
plt.xlabel('Capacity (Ah)')
plt.ylabel('Voltage (V)')
plt.title('CS2_36 Discharge V-Q')
plt.savefig('outputs/cs2_vq.png')

print('CS2 cap:', q_disch[-1])

# NASA
mat = loadmat('data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/B0005.mat')['B0005']['cycle'][0][0]
disch_idx = np.where([c['type'][0][0] == 'discharge' for c in mat])[0][:3]
for i, idx in enumerate(disch_idx):
    data = mat[idx][0][0]['data'][0][0]
    t = data['Time'][0].flatten()
    v = data['Voltage_measured'][0].flatten()
    cap = data['Capacity'][0][0]
    plt.figure()
    plt.plot(t/3600, v)
    plt.title(f'NASA B0005 disch {i+1}, cap={cap:.3f}Ah')
    plt.xlabel('Time (h)')
    plt.ylabel('V')
    plt.savefig(f'outputs/nasa_disch{i+1}.png')

# Oxford
ox = loadmat('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat')
print('Oxford:', list(ox.keys()))
