"""
Phase 1: Data Exploration and Preprocessing
Load all three datasets, extract discharge curves, and save processed data.
"""
import numpy as np
import pandas as pd
import scipy.io
import os
import json

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_000_20260415_130453"
OUTPUTS = os.path.join(WORKSPACE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

# ============================================================
# 1. NASA PCoE Dataset - Extract discharge curves
# ============================================================
print("=" * 60)
print("1. NASA PCoE Dataset Processing")
print("=" * 60)

nasa_dir = os.path.join(WORKSPACE, "data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4")
nasa_batteries = ["B0005", "B0006", "B0007", "B0018"]

nasa_discharge_data = {}
nasa_capacity_fade = {}

for batt_name in nasa_batteries:
    fp = os.path.join(nasa_dir, f"{batt_name}.mat")
    data = scipy.io.loadmat(fp)
    cycles = data[batt_name][0, 0]['cycle']
    
    discharge_curves = []
    capacities = []
    
    for i in range(cycles.size):
        c = cycles[0, i]
        ctype = c['type'][0][0]
        if ctype == 'd':  # discharge
            d = c['data'][0, 0]
            voltage = d['Voltage_measured'].flatten()
            current = d['Current_measured'].flatten()
            temp = d['Temperature_measured'].flatten()
            time_arr = d['Time'].flatten()
            capacity = float(d['Capacity'].flatten()[0])
            
            discharge_curves.append({
                'cycle': i,
                'time': time_arr.tolist(),
                'voltage': voltage.tolist(),
                'current': current.tolist(),
                'temperature': temp.tolist(),
                'capacity': capacity
            })
            capacities.append(capacity)
    
    nasa_discharge_data[batt_name] = discharge_curves
    nasa_capacity_fade[batt_name] = capacities
    
    print(f"  {batt_name}: {len(discharge_curves)} discharge cycles")
    if capacities:
        print(f"    Initial capacity: {capacities[0]:.4f} Ah")
        print(f"    Final capacity: {capacities[-1]:.4f} Ah")

# Save NASA discharge summary (first few cycles of B0005 for reference)
nasa_summary = {}
for batt in nasa_batteries:
    cycles_data = []
    for dc in nasa_discharge_data[batt][:5]:  # first 5 discharge cycles
        cycles_data.append({
            'cycle': dc['cycle'],
            'capacity': dc['capacity'],
            'n_points': len(dc['voltage']),
            'voltage_range': [min(dc['voltage']), max(dc['voltage'])]
        })
    nasa_summary[batt] = cycles_data

with open(os.path.join(OUTPUTS, "nasa_discharge_summary.json"), 'w') as f:
    json.dump(nasa_summary, f, indent=2)

# Save capacity fade curves for all batteries
capacity_fade_data = {}
for batt in nasa_batteries:
    discharge_cycles = [dc['cycle'] for dc in nasa_discharge_data[batt]]
    cap_values = [dc['capacity'] for dc in nasa_discharge_data[batt]]
    capacity_fade_data[batt] = {
        'cycles': discharge_cycles,
        'capacities': cap_values
    }

np.savez(os.path.join(OUTPUTS, "nasa_capacity_fade.npz"), **{
    batt: np.array(capacity_fade_data[batt]['capacities']) for batt in nasa_batteries
})
np.savez(os.path.join(OUTPUTS, "nasa_cycle_numbers.npz"), **{
    batt: np.array(capacity_fade_data[batt]['cycles']) for batt in nasa_batteries
})

# Save a representative discharge curve from B0005 (early cycle)
ref_batt = "B0005"
# Get a mid-life discharge curve for parameter identification
mid_idx = len(nasa_discharge_data[ref_batt]) // 2
ref_discharge = nasa_discharge_data[ref_batt][mid_idx]
np.savez(os.path.join(OUTPUTS, "nasa_reference_discharge.npz"),
         time=np.array(ref_discharge['time']),
         voltage=np.array(ref_discharge['voltage']),
         current=np.array(ref_discharge['current']),
         temperature=np.array(ref_discharge['temperature']),
         capacity=ref_discharge['capacity'],
         cycle=ref_discharge['cycle'])

print(f"\n  Reference discharge saved: {ref_batt} cycle {ref_discharge['cycle']}")
print(f"    Capacity: {ref_discharge['capacity']:.4f} Ah")
print(f"    Points: {len(ref_discharge['voltage'])}")

# ============================================================
# 2. CS2_36 Dataset - Extract discharge curves
# ============================================================
print("\n" + "=" * 60)
print("2. CS2_36 Dataset Processing")
print("=" * 60)

cs2_dir = os.path.join(WORKSPACE, "data/CS2_36")
cs2_files = sorted([f for f in os.listdir(cs2_dir) if f.endswith('.xlsx')])

cs2_all_data = {}

for fname in cs2_files:
    fp = os.path.join(cs2_dir, fname)
    xls = pd.ExcelFile(fp)
    
    # Find the channel data sheet
    channel_sheet = [s for s in xls.sheet_names if s.startswith('Channel_')]
    if not channel_sheet:
        continue
    
    df = pd.read_excel(fp, sheet_name=channel_sheet[0], header=None)
    # Row 0 is header
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)
    
    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    print(f"\n  {fname}: {len(df)} data points")
    print(f"    Columns: {list(df.columns)}")
    
    # Extract key signals
    if 'Test_Time(s)' in df.columns and 'Voltage(V)' in df.columns:
        cs2_all_data[fname] = {
            'test_time': df['Test_Time(s)'].values.astype(float),
            'voltage': df['Voltage(V)'].values.astype(float),
            'current': df['Current(A)'].values.astype(float),
            'cycle_index': df['Cycle_Index'].values.astype(int),
            'step_index': df['Step_Index'].values.astype(int),
            'discharge_capacity': df.get('Discharge_Capacity(Ah)', pd.Series(0, index=df.index)).values.astype(float),
            'charge_capacity': df.get('Charge_Capacity(Ah)', pd.Series(0, index=df.index)).values.astype(float),
        }

# Save CS2 data
for fname, data in cs2_all_data.items():
    safe_name = fname.replace('.xlsx', '').replace('.', '_')
    np.savez(os.path.join(OUTPUTS, f"cs2_{safe_name}.npz"), **data)

# Extract discharge segments from CS2 data (primary reference for parameter ID)
# Discharge = negative current
cs2_discharge_segments = {}
for fname, data in cs2_all_data.items():
    mask = data['current'] < -0.01  # discharge threshold
    discharge_times = data['test_time'][mask]
    discharge_voltages = data['voltage'][mask]
    discharge_currents = data['current'][mask]
    
    # Find continuous discharge segments
    diff = np.diff(np.concatenate([[0], mask.astype(int)]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) > 0 and len(ends) > 0:
        # Take the longest discharge segment as reference
        lengths = ends - starts
        best_idx = np.argmax(lengths)
        seg_start = starts[best_idx]
        seg_end = ends[best_idx]
        
        cs2_discharge_segments[fname] = {
            'time': discharge_times,
            'voltage': discharge_voltages,
            'current': discharge_currents,
            'longest_segment': {
                'start': int(seg_start),
                'end': int(seg_end),
                'time': data['test_time'][seg_start:seg_end].tolist(),
                'voltage': data['voltage'][seg_start:seg_end].tolist(),
                'current': data['current'][seg_start:seg_end].tolist(),
            }
        }
        print(f"  {fname}: {len(starts)} discharge segments, longest: {seg_end - seg_start} points")

# Save CS2 discharge reference (use first file as primary)
if cs2_files:
    ref_cs2 = cs2_discharge_segments.get(cs2_files[0], {})
    if 'longest_segment' in ref_cs2:
        seg = ref_cs2['longest_segment']
        np.savez(os.path.join(OUTPUTS, "cs2_reference_discharge.npz"),
                 time=np.array(seg['time']),
                 voltage=np.array(seg['voltage']),
                 current=np.array(seg['current']))
        print(f"\n  CS2 reference discharge saved: {cs2_files[0]}")
        print(f"    Points: {len(seg['time'])}")
        print(f"    Voltage range: [{min(seg['voltage']):.4f}, {max(seg['voltage']):.4f}] V")

# ============================================================
# 3. Oxford Dataset - Extract drive cycle data
# ============================================================
print("\n" + "=" * 60)
print("3. Oxford Battery Degradation Dataset Processing")
print("=" * 60)

oxford_dir = os.path.join(WORKSPACE, "data/Oxford Battery Degradation Dataset")
oxford_fp = os.path.join(oxford_dir, "ExampleDC_C1.mat")

data = scipy.io.loadmat(oxford_fp)
ex = data['ExampleDC_C1'][0, 0]

# Charge data
ch = ex['ch'][0, 0]
ch_fields = ch.dtype.names
print(f"\n  Charge fields: {ch_fields}")
ch_data = {}
for fld in ch_fields:
    arr = ch[fld].flatten()
    ch_data[fld] = arr
    print(f"    {fld}: shape={arr.shape}")

# Discharge data (drive cycle)
dc = ex['dc'][0, 0]
dc_fields = dc.dtype.names
print(f"\n  Discharge (drive cycle) fields: {dc_fields}")
dc_data = {}
for fld in dc_fields:
    arr = dc[fld].flatten()
    dc_data[fld] = arr
    print(f"    {fld}: shape={arr.shape}")

# Save Oxford data
np.savez(os.path.join(OUTPUTS, "oxford_drive_cycle.npz"),
         dc_time=dc_data.get('t', np.array([])),
         dc_voltage=dc_data.get('v', np.array([])),
         dc_current=dc_data.get('i', np.array([])),
         dc_charge=dc_data.get('q', np.array([])),
         dc_temperature=dc_data.get('T', np.array([])),
         ch_time=ch_data.get('t', np.array([])),
         ch_voltage=ch_data.get('v', np.array([])),
         ch_charge=ch_data.get('q', np.array([])),
         ch_temperature=ch_data.get('T', np.array([])))

print(f"\n  Oxford drive cycle saved:")
if 'v' in dc_data:
    print(f"    Time: {len(dc_data['t'])} points, range [{dc_data['t'][0]:.1f}, {dc_data['t'][-1]:.1f}] s")
    print(f"    Voltage: range [{min(dc_data['v']):.4f}, {max(dc_data['v']):.4f}] V")
if 'i' in dc_data:
    print(f"    Current: range [{min(dc_data['i']):.2f}, {max(dc_data['i']):.2f}] mA")

print("\n" + "=" * 60)
print("Data exploration complete. All outputs saved to outputs/")
print("=" * 60)
