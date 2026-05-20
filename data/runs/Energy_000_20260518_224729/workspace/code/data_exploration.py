"""
Data exploration script for all three datasets.
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ==================== CS2_36 DATA ====================
print("Loading CS2_36 data...")
cs2_files = ['CS2_36_1_10_11.xlsx', 'CS2_36_1_18_11.xlsx', 'CS2_36_1_24_11.xlsx', 'CS2_36_1_28_11.xlsx']
cs2_discharge_curves = []

for f in cs2_files:
    df = pd.read_excel(f'data/CS2_36/{f}', sheet_name='Channel_1-009')
    # Extract discharge curves (Step 7)
    discharge = df[df['Step_Index'] == 7].copy()
    discharge = discharge.sort_values('Step_Time(s)')
    cs2_discharge_curves.append({
        'file': f,
        'time': discharge['Step_Time(s)'].values,
        'voltage': discharge['Voltage(V)'].values,
        'current': discharge['Current(A)'].values,
        'capacity': discharge['Discharge_Capacity(Ah)'].values
    })
    print(f"  {f}: {len(discharge)} discharge points, V={discharge['Voltage(V)'].min():.3f}-{discharge['Voltage(V)'].max():.3f}")

# Plot CS2_36 discharge curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, curve in enumerate(cs2_discharge_curves):
    ax = axes[i]
    ax.plot(curve['capacity'], curve['voltage'], 'b-', linewidth=1.5)
    ax.set_xlabel('Discharge Capacity (Ah)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'CS2_36 Discharge: {curve["file"]}')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/cs2_36_discharge_curves.png', dpi=150)
plt.close()

# ==================== NASA DATA ====================
print("\nLoading NASA data...")
nasa_batteries = {}
for f in ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']:
    mat = loadmat(f'data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/{f}')
    bname = f.split('.')[0]
    b = mat[bname]
    cycle = b[0,0]['cycle']
    
    discharge_curves = []
    for i in range(cycle.shape[1]):
        c = cycle[0,i]
        if c['type'][0] == 'discharge':
            data = c['data'][0,0]
            voltage = data['Voltage_measured'].flatten()
            current = data['Current_measured'].flatten()
            temp = data['Temperature_measured'].flatten()
            time = data['Time'].flatten()
            # Check if it's a constant current discharge
            if len(voltage) > 50:
                discharge_curves.append({
                    'time': time,
                    'voltage': voltage,
                    'current': current,
                    'temperature': temp,
                    'cycle_idx': i
                })
    
    # Pick first, middle, and last discharge cycles
    if len(discharge_curves) >= 3:
        selected = [discharge_curves[0], discharge_curves[len(discharge_curves)//2], discharge_curves[-1]]
    else:
        selected = discharge_curves
    nasa_batteries[bname] = selected
    print(f"  {bname}: {len(discharge_curves)} discharge cycles, selected {len(selected)}")

# Plot NASA discharge curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, (bname, curves) in enumerate(nasa_batteries.items()):
    ax = axes[idx]
    for ci, curve in enumerate(curves):
        label = f'Cycle {curve["cycle_idx"]}'
        ax.plot(curve['time'], curve['voltage'], label=label, linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'NASA {bname} Discharge Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/nasa_discharge_curves.png', dpi=150)
plt.close()

# ==================== OXFORD DATA ====================
print("\nLoading Oxford data...")
mat = loadmat('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat')
dc = mat['ExampleDC_C1'][0,0]['dc'][0,0]
ch = mat['ExampleDC_C1'][0,0]['ch'][0,0]

oxford_data = {
    'dc_time': dc['t'].flatten(),
    'dc_voltage': dc['v'].flatten(),
    'dc_current': dc['i'].flatten(),
    'dc_temp': dc['T'].flatten(),
    'ch_time': ch['t'].flatten(),
    'ch_voltage': ch['v'].flatten(),
}
print(f"  DC: {len(oxford_data['dc_time'])} points, current range: {oxford_data['dc_current'].min():.1f} to {oxford_data['dc_current'].max():.1f} mA")

# Plot Oxford data
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
axes[0].plot(oxford_data['dc_time'], oxford_data['dc_voltage'], 'b-', linewidth=1)
axes[0].set_ylabel('Voltage (V)')
axes[0].set_title('Oxford Battery: Dynamic Discharge (Artemis Urban Drive Cycle)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(oxford_data['dc_time'], oxford_data['dc_current'], 'r-', linewidth=1)
axes[1].set_ylabel('Current (mA)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(oxford_data['dc_time'], oxford_data['dc_temp'], 'g-', linewidth=1)
axes[2].set_ylabel('Temperature (°C)')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/oxford_dynamic_profile.png', dpi=150)
plt.close()

# Save processed data
np.savez('outputs/processed_data.npz',
         cs2_discharge_curves=cs2_discharge_curves,
         nasa_batteries=nasa_batteries,
         oxford_data=oxford_data)
print("\nData exploration complete. Figures saved to report/images/")
