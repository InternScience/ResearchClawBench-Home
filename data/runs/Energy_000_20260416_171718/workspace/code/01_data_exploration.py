#!/usr/bin/env python3
"""
Data Exploration Script for Battery Parameter Identification Task
Loads and explores all three datasets: NASA PCoE, CS2_36, and Oxford
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import json

# Set paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260416_171718"
DATA_DIR = os.path.join(WORKSPACE, "data")
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

print("=" * 60)
print("DATA EXPLORATION FOR BATTERY PARAMETER IDENTIFICATION")
print("=" * 60)

# ============================================================================
# 1. NASA PCoE Dataset
# ============================================================================
print("\n[1] Loading NASA PCoE Dataset...")
nasa_dir = os.path.join(DATA_DIR, "NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4")
nasa_files = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]

nasa_data = {}
for fname in nasa_files:
    fpath = os.path.join(nasa_dir, fname)
    mat_data = loadmat(fpath)
    battery_id = fname.replace(".mat", "")
    
    # Extract discharge cycles - structure is nested
    battery_array = mat_data[battery_id][0]
    cycles = battery_array[0]['cycle'][0]
    discharge_cycles = []
    
    for i in range(len(cycles)):
        cycle_type = cycles['type'][i][0][0]
        if cycle_type == 'd':  # 'd' for discharge
            data_struct = cycles['data'][i][0][0]
            discharge_cycles.append({
                'time': data_struct['Time'][0],
                'voltage': data_struct['Voltage_measured'][0],
                'current': data_struct['Current_measured'][0],
                'temperature': data_struct['Temperature_measured'][0],
                'capacity': float(data_struct['Capacity'][0][0]) if 'Capacity' in data_struct.dtype.names else 0.0
            })
    
    nasa_data[battery_id] = discharge_cycles
    print(f"  {battery_id}: {len(discharge_cycles)} discharge cycles")

# ============================================================================
# 2. CS2_36 Dataset (CALCE)
# ============================================================================
print("\n[2] Loading CS2_36 Dataset (CALCE)...")
cs2_dir = os.path.join(DATA_DIR, "CS2_36")
cs2_files = sorted([f for f in os.listdir(cs2_dir) if f.endswith('.xlsx')])

cs2_data = {}
for fname in cs2_files:
    fpath = os.path.join(cs2_dir, fname)
    cycle_id = fname.replace(".xlsx", "")
    
    # Read Excel file
    df = pd.read_excel(fpath, sheet_name=0)
    cs2_data[cycle_id] = df
    print(f"  {cycle_id}: {len(df)} rows, columns: {list(df.columns)[:8]}...")

# ============================================================================
# 3. Oxford Battery Degradation Dataset
# ============================================================================
print("\n[3] Loading Oxford Battery Degradation Dataset...")
oxford_dir = os.path.join(DATA_DIR, "Oxford Battery Degradation Dataset")
oxford_file = os.path.join(oxford_dir, "ExampleDC_C1.mat")

oxford_mat = loadmat(oxford_file)
print(f"  Oxford file keys: {list(oxford_mat.keys())}")

# Extract charge and discharge data
oxford_data = {}
if 'ch' in oxford_mat:
    ch_data = oxford_mat['ch'][0][0]
    oxford_data['charge'] = {
        'time': ch_data[0],
        'voltage': ch_data[1],
        'charge': ch_data[2],
        'temperature': ch_data[3]
    }
    print(f"  Charge: {len(ch_data[0])} samples")

if 'dc' in oxford_mat:
    dc_data = oxford_mat['dc'][0][0]
    oxford_data['discharge'] = {
        'time': dc_data[0],
        'voltage': dc_data[1],
        'current': dc_data[2],
        'charge': dc_data[3],
        'temperature': dc_data[4]
    }
    print(f"  Discharge: {len(dc_data[0])} samples")

# ============================================================================
# Generate Data Overview Plots
# ============================================================================
print("\n[4] Generating Data Overview Plots...")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Battery Dataset Overview', fontsize=14, fontweight='bold')

# Plot 1: NASA PCoE - Sample discharge curves
ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(nasa_files)))
for i, (battery_id, cycles) in enumerate(nasa_data.items()):
    if len(cycles) > 0:
        # Plot first few cycles
        for j, cycle in enumerate(cycles[:3]):
            alpha = 0.7 - 0.2 * j
            ax.plot(cycle['time']/60, cycle['voltage'], color=colors[i], 
                   alpha=alpha, label=f'{battery_id} Cycle {j+1}' if j==0 else None)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('NASA PCoE: Sample Discharge Curves')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: NASA PCoE - Temperature profiles
ax = axes[0, 1]
for i, (battery_id, cycles) in enumerate(nasa_data.items()):
    if len(cycles) > 0:
        cycle = cycles[0]
        ax.plot(cycle['time']/60, cycle['temperature'], color=colors[i], 
               label=battery_id)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('NASA PCoE: Temperature During Discharge')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: CS2_36 - Sample voltage curves
ax = axes[1, 0]
for i, (cycle_id, df) in enumerate(cs2_data.items()):
    # Find voltage and time columns
    time_col = [c for c in df.columns if 'time' in c.lower() or 'second' in c.lower()]
    volt_col = [c for c in df.columns if 'volt' in c.lower()]
    
    if time_col and volt_col:
        t = df[time_col[0]].values
        v = df[volt_col[0]].values
        # Downsample for clarity
        step = max(1, len(t)//500)
        ax.plot(t[::step]/60, v[::step], label=cycle_id, alpha=0.7)
        
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('CS2_36 (CALCE): Discharge Curves')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 4: CS2_36 - Capacity fade over cycles
ax = axes[1, 1]
cycle_numbers = []
capacities = []
for i, (cycle_id, df) in enumerate(sorted(cs2_data.items())):
    # Try to extract capacity from filename or data
    cap_col = [c for c in df.columns if 'capac' in c.lower() or 'ah' in c.lower()]
    if cap_col:
        cap = df[cap_col[0]].iloc[-1] if len(df[cap_col[0]]) > 0 else 0
    else:
        # Estimate from voltage curve end
        cap = 2.0  # Default estimate
    
    # Extract cycle number from filename
    try:
        cyc_num = int(''.join(filter(str.isdigit, cycle_id.split('_')[-1])))
    except:
        cyc_num = i
    
    cycle_numbers.append(cyc_num)
    capacities.append(cap)

ax.scatter(cycle_numbers, capacities, s=50, alpha=0.7)
ax.set_xlabel('Cycle Number')
ax.set_ylabel('Capacity (Ah)')
ax.set_title('CS2_36: Capacity vs Cycle Number')
ax.grid(True, alpha=0.3)

# Plot 5: Oxford - Dynamic discharge profile
ax = axes[2, 0]
if 'discharge' in oxford_data:
    od = oxford_data['discharge']
    ax.plot(od['time']/60, od['voltage'], 'b-', label='Voltage', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(od['time']/60, od['current']/1000, 'r-', label='Current', alpha=0.5)
    ax2.set_ylabel('Current (A)', color='red')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)', color='blue')
ax.set_title('Oxford: Dynamic Drive Cycle Discharge')
ax.grid(True, alpha=0.3)

# Plot 6: Oxford - Current profile detail
ax = axes[2, 1]
if 'discharge' in oxford_data:
    od = oxford_data['discharge']
    ax.plot(od['time']/60, od['current']/1000, 'r-', alpha=0.7)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Current (A)')
ax.set_title('Oxford: Current Profile (Urban Artemis)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'data_overview.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {IMAGES_DIR}/data_overview.png")
plt.close()

# ============================================================================
# Save Data Summary
# ============================================================================
data_summary = {
    "nasa_pcoe": {
        "batteries": list(nasa_data.keys()),
        "discharge_cycles_per_battery": {k: len(v) for k, v in nasa_data.items()}
    },
    "cs2_36": {
        "files": list(cs2_data.keys()),
        "rows_per_file": {k: len(v) for k, v in cs2_data.items()}
    },
    "oxford": {
        "has_charge": 'charge' in oxford_data,
        "has_discharge": 'discharge' in oxford_data,
        "charge_samples": len(oxford_data.get('charge', {}).get('time', [])),
        "discharge_samples": len(oxford_data.get('discharge', {}).get('time', []))
    }
}

with open(os.path.join(OUTPUTS_DIR, 'data_summary.json'), 'w') as f:
    json.dump(data_summary, f, indent=2)
print(f"\nSaved data summary to: {OUTPUTS_DIR}/data_summary.json")

# Save sample data for model training
print("\n[5] Preparing sample data for parameter identification...")

# Extract representative discharge curve from CS2_36 for parameter ID
cs2_sample = {}
for cycle_id, df in cs2_data.items():
    time_col = [c for c in df.columns if 'time' in c.lower() or 'second' in c.lower()]
    volt_col = [c for c in df.columns if 'volt' in c.lower()]
    curr_col = [c for c in df.columns if 'curr' in c.lower()]
    temp_col = [c for c in df.columns if 'temp' in c.lower()]
    
    if time_col and volt_col:
        cs2_sample[cycle_id] = {
            'time': df[time_col[0]].values.tolist(),
            'voltage': df[volt_col[0]].values.tolist(),
            'current': df[curr_col[0]].values.tolist() if curr_col else [1.0] * len(df),
            'temperature': df[temp_col[0]].values.tolist() if temp_col else [25.0] * len(df)
        }

with open(os.path.join(OUTPUTS_DIR, 'cs2_sample_data.json'), 'w') as f:
    json.dump(cs2_sample, f, indent=2)
print(f"  Saved CS2_36 sample data to: {OUTPUTS_DIR}/cs2_sample_data.json")

# Extract NASA sample data
nasa_sample = {}
for battery_id, cycles in nasa_data.items():
    if len(cycles) > 0:
        # Take first discharge cycle
        c = cycles[0]
        nasa_sample[battery_id] = {
            'time': c['time'].tolist(),
            'voltage': c['voltage'].tolist(),
            'current': c['current'].tolist(),
            'temperature': c['temperature'].tolist(),
            'capacity': c['capacity']
        }

with open(os.path.join(OUTPUTS_DIR, 'nasa_sample_data.json'), 'w') as f:
    json.dump(nasa_sample, f, indent=2)
print(f"  Saved NASA sample data to: {OUTPUTS_DIR}/nasa_sample_data.json")

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETE")
print("=" * 60)
