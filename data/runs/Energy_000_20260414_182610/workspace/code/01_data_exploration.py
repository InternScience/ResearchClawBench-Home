"""
Step 1: Data Loading and Exploration
Load CS2_36, NASA PCoE, and Oxford datasets. Generate data overview figures.
"""
import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ============================================================
# 1. Load CS2_36 Data (Primary Reference)
# ============================================================
print("Loading CS2_36 data...")
cs2_files = {
    'Cycle 10': 'data/CS2_36/CS2_36_1_10_11.xlsx',
    'Cycle 18': 'data/CS2_36/CS2_36_1_18_11.xlsx',
    'Cycle 24': 'data/CS2_36/CS2_36_1_24_11.xlsx',
    'Cycle 28': 'data/CS2_36/CS2_36_1_28_11.xlsx',
}

cs2_data = {}
for label, fpath in cs2_files.items():
    df = pd.read_excel(fpath, sheet_name='Channel_1-009')
    cs2_data[label] = df
    print(f"  {label}: {len(df)} rows, columns: {list(df.columns[:8])}")

# Extract discharge curves from CS2_36
cs2_discharge_curves = {}
for label, df in cs2_data.items():
    # Find discharge steps (negative current)
    cycle_groups = df.groupby('Cycle_Index')
    for cyc_idx, cyc_df in cycle_groups:
        discharge_steps = cyc_df[cyc_df['Current(A)'] < -0.1]
        if len(discharge_steps) > 10:
            key = f"{label}_Cyc{cyc_idx}"
            cs2_discharge_curves[key] = {
                'time': discharge_steps['Test_Time(s)'].values,
                'voltage': discharge_steps['Voltage(V)'].values,
                'current': discharge_steps['Current(A)'].values,
                'capacity': discharge_steps['Discharge_Capacity(Ah)'].values,
                'temp': discharge_steps.get('AC_Impedance(Ohm)', pd.Series([0]*len(discharge_steps))).values,
            }

print(f"  Found {len(cs2_discharge_curves)} discharge curves in CS2_36")

# ============================================================
# 2. Load NASA PCoE Data (Validation)
# ============================================================
print("\nLoading NASA PCoE data...")
nasa_batteries = {}
for bat_name in ['B0005', 'B0006', 'B0007', 'B0018']:
    fpath = f'data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/{bat_name}.mat'
    mat = scipy.io.loadmat(fpath, squeeze_me=True, struct_as_record=False)
    bat = mat[bat_name]
    
    discharge_cycles = []
    for i, c in enumerate(bat.cycle):
        if c.type == 'discharge':
            d = c.data
            discharge_cycles.append({
                'index': i,
                'voltage': np.array(d.Voltage_measured).flatten(),
                'current': np.array(d.Current_measured).flatten(),
                'temperature': np.array(d.Temperature_measured).flatten(),
                'capacity': float(d.Capacity) if hasattr(d, 'Capacity') and np.isscalar(d.Capacity) else 0,
                'ambient_temp': c.ambient_temperature,
            })
    
    nasa_batteries[bat_name] = discharge_cycles
    capacities = [c['capacity'] for c in discharge_cycles if c['capacity'] > 0]
    print(f"  {bat_name}: {len(bat.cycle)} total cycles, {len(discharge_cycles)} discharge, "
          f"capacity range: {min(capacities):.3f} - {max(capacities):.3f} Ah" if capacities else
          f"  {bat_name}: {len(bat.cycle)} total cycles, {len(discharge_cycles)} discharge")

# ============================================================
# 3. Load Oxford Data (Dynamic Validation)
# ============================================================
print("\nLoading Oxford data...")
mat = scipy.io.loadmat('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat', squeeze_me=True, struct_as_record=False)
oxford = mat['ExampleDC_C1']
oxford_charge = {
    'time': np.array(oxford.ch.t).flatten(),
    'voltage': np.array(oxford.ch.v).flatten(),
    'current': np.array(oxford.ch.i).flatten(),
    'capacity': np.array(oxford.ch.q).flatten(),
    'temperature': np.array(oxford.ch.T).flatten(),
}
oxford_discharge = {
    'time': np.array(oxford.dc.t).flatten(),
    'voltage': np.array(oxford.dc.v).flatten(),
    'current': np.array(oxford.dc.i).flatten(),
    'capacity': np.array(oxford.dc.q).flatten(),
    'temperature': np.array(oxford.dc.T).flatten(),
}
print(f"  Charge: {len(oxford_charge['time'])} points")
print(f"  Discharge: {len(oxford_discharge['time'])} points")

# ============================================================
# 4. Generate Data Overview Figures
# ============================================================
print("\nGenerating data overview figures...")

# Figure 1: CS2_36 discharge voltage curves at different cycles
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot selected CS2 discharge curves
selected_keys = [k for k in cs2_discharge_curves.keys() if 'Cyc1' in k][:4]
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_keys)))
for key, color in zip(selected_keys, colors):
    d = cs2_discharge_curves[key]
    axes[0].plot(d['capacity'], d['voltage'], color=color, label=key.split('_')[0], linewidth=1.5)
axes[0].set_xlabel('Discharge Capacity (Ah)', fontsize=12)
axes[0].set_ylabel('Voltage (V)', fontsize=12)
axes[0].set_title('CS2_36: Discharge Curves', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# NASA battery capacity fade
for bat_name, cycles in nasa_batteries.items():
    caps = [(c['index'], c['capacity']) for c in cycles if c['capacity'] > 0]
    if caps:
        idxs, caps_vals = zip(*caps)
        axes[1].plot(idxs, caps_vals, 'o-', label=bat_name, markersize=3, linewidth=1.5)
axes[1].set_xlabel('Cycle Index', fontsize=12)
axes[1].set_ylabel('Capacity (Ah)', fontsize=12)
axes[1].set_title('NASA PCoE: Capacity Fade', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Oxford dynamic profile
axes[2].plot(oxford_discharge['time'], oxford_discharge['voltage'], 'b-', linewidth=1, label='Voltage')
ax2 = axes[2].twinx()
ax2.plot(oxford_discharge['time'], oxford_discharge['current'], 'r-', linewidth=0.5, alpha=0.7, label='Current')
axes[2].set_xlabel('Time (s)', fontsize=12)
axes[2].set_ylabel('Voltage (V)', fontsize=12, color='b')
ax2.set_ylabel('Current (A)', fontsize=12, color='r')
axes[2].set_title('Oxford: Dynamic Discharge Profile', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_data_overview.png")

# Figure 2: NASA detailed voltage/temperature profiles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, bat_name in enumerate(['B0005', 'B0006']):
    ax = axes[0, idx]
    cycles = nasa_batteries[bat_name]
    # Plot first, middle, last discharge
    n = len(cycles)
    for ci in [0, n//4, n//2, 3*n//4, n-1]:
        if ci < n:
            c = cycles[ci]
            t_norm = np.linspace(0, 1, len(c['voltage']))
            ax.plot(t_norm, c['voltage'], linewidth=1, label=f"Cycle {c['index']}")
    ax.set_xlabel('Normalized Time', fontsize=11)
    ax.set_ylabel('Voltage (V)', fontsize=11)
    ax.set_title(f'{bat_name}: Voltage Profiles', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for idx, bat_name in enumerate(['B0005', 'B0006']):
    ax = axes[1, idx]
    cycles = nasa_batteries[bat_name]
    n = len(cycles)
    for ci in [0, n//4, n//2, 3*n//4, n-1]:
        if ci < n:
            c = cycles[ci]
            t_norm = np.linspace(0, 1, len(c['temperature']))
            ax.plot(t_norm, c['temperature'], linewidth=1, label=f"Cycle {c['index']}")
    ax.set_xlabel('Normalized Time', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_title(f'{bat_name}: Temperature Profiles', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig2_nasa_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_nasa_profiles.png")

# Save summary statistics
summary = {
    'cs2_36': {
        'files': list(cs2_files.keys()),
        'total_discharge_curves': len(cs2_discharge_curves),
    },
    'nasa_pcoe': {
        'batteries': list(nasa_batteries.keys()),
        'discharge_cycles_per_battery': {k: len(v) for k, v in nasa_batteries.items()},
    },
    'oxford': {
        'charge_points': len(oxford_charge['time']),
        'discharge_points': len(oxford_discharge['time']),
    }
}

import json
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nData exploration complete. Summary saved to outputs/data_summary.json")
