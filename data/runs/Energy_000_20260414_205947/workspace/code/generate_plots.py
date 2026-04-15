import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.integrate import cumulative_trapezoid

plt.ioff()  # non-interactive

# CS2_36
df = pd.read_excel('data/CS2_36/CS2_36_1_28_11.xlsx', sheet_name='Channel_1-009')
time_cs = df['Test_Time(s)'].values
volt_cs = df['Voltage(V)'].values
curr_cs = df['Current(A)'].values
d_cap_cs = df['Discharge_Capacity(Ah)'].fillna(method='ffill').values

# First discharge segment (curr < -0.5)
mask = curr_cs < -0.5
t_seg = time_cs[mask] - time_cs[mask][0]
q_seg = cumulative_trapezoid(-curr_cs[mask], t_seg, initial=0)/3600
plt.figure(figsize=(8,6))
plt.plot(q_seg, volt_cs[mask])
plt.xlabel('Capacity (Ah)')
plt.ylabel('Voltage (V)')
plt.title('CS2_36 1C Discharge V-Q')
plt.grid(True)
plt.savefig('report/images/cs2_vq.png', dpi=150, bbox_inches='tight')
print('CS2 max cap:', q_seg[-1])

# NASA B0005
path_nasa = 'data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/B0005.mat'
mat_nasa = loadmat(path_nasa)
b0005_cycles = mat_nasa['B0005'][0,0]['cycle'][0,0]
disch_cycles = [c for c in b0005_cycles if isinstance(c, np.ndarray) and c['type'][0] == b' discharge']
print(f'NASA discharges: {len(disch_cycles)}')
for i in range(min(3, len(disch_cycles))):
    data = disch_cycles[i][0,0]['data'][0,0]
    t = data['Time'][0].flatten()/3600
    v = data['Voltage_measured'][0].flatten()
    cap = data['Capacity'][0][0][0,0]
    plt.figure(figsize=(8,6))
    plt.plot(t, v)
    plt.xlabel('Time (h)')
    plt.ylabel('Voltage (V)')
    plt.title(f'NASA B0005 Discharge {i+1}, Cap = {cap:.3f} Ah')
    plt.grid(True)
    plt.savefig(f'report/images/nasa_disch{i+1}.png', dpi=150, bbox_inches='tight')

# Oxford ExampleDC_C1 discharge
path_ox = 'data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat'
mat_ox = loadmat(path_ox)
if 'ExampleDC_C1' in mat_ox:
    dc = mat_ox['ExampleDC_C1'][0,0]
    if 'dc' in dc.dtype.names:
        dc_data = dc['dc'][0,0]
        if 'Time' in dc_data.dtype.names:
            t_ox = dc_data['Time'][0].flatten()
            v_ox = dc_data['Voltage_measured'][0].flatten() if 'Voltage_measured' in dc_data.dtype.names else dc_data['v'][0].flatten()
            plt.figure(figsize=(8,6))
            plt.plot(t_ox/3600, v_ox)
            plt.xlabel('Time (h)')
            plt.ylabel('Voltage (V)')
            plt.title('Oxford Dynamic Discharge Cycle 1')
            plt.grid(True)
            plt.savefig('report/images/oxford_dc.png', dpi=150, bbox_inches='tight')

print('Plots saved to report/images/')
