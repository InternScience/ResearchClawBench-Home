"""
Data Loading Module - Load and preprocess all three battery datasets
"""
import numpy as np
import scipy.io as sio
import os
import sys

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_000_20260416_182016"

def load_nasa_data(battery_id='B0005'):
    """Load NASA PCoE battery aging data"""
    fpath = os.path.join(WORKSPACE, f"data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/{battery_id}.mat")
    mat = sio.loadmat(fpath)
    cycle_data = mat[battery_id]['cycle'][0,0]
    
    discharge_cycles = []
    for i in range(cycle_data.shape[1]):
        c = cycle_data[0, i]
        ctype = str(c['type'][0])
        if ctype == 'discharge':
            data = c['data'][0,0]
            voltage = data['Voltage_measured'].flatten()
            current = data['Current_measured'].flatten()
            temperature = data['Temperature_measured'].flatten()
            time = data['Time'].flatten()
            cap = data['Capacity'].flatten()[0] if 'Capacity' in data.dtype.names else None
            discharge_cycles.append({
                'cycle_index': i,
                'voltage': voltage,
                'current': current,
                'temperature': temperature,
                'time': time,
                'capacity': cap
            })
    return discharge_cycles

def load_cs2_36_data(file_idx=0):
    """Load CALCE CS2_36 data"""
    files = ['CS2_36_1_10_11.xlsx', 'CS2_36_1_18_11.xlsx', 'CS2_36_1_24_11.xlsx', 'CS2_36_1_28_11.xlsx']
    fpath = os.path.join(WORKSPACE, f"data/CS2_36/{files[file_idx]}")
    
    try:
        import openpyxl
        wb = openpyxl.load_workbook(fpath, read_only=True)
        ws = wb['Channel_1-009']
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        
        headers = rows[0]
        data_rows = rows[1:]
        
        time_col = list(headers).index('Test_Time(s)')
        current_col = list(headers).index('Current(A)')
        voltage_col = list(headers).index('Voltage(V)')
        discharge_cap_col = list(headers).index('Discharge_Capacity(Ah)')
        cycle_col = list(headers).index('Cycle_Index')
        step_col = list(headers).index('Step_Index')
        
        time_arr = np.array([r[time_col] for r in data_rows if r[time_col] is not None], dtype=float)
        current_arr = np.array([r[current_col] for r in data_rows if r[current_col] is not None], dtype=float)
        voltage_arr = np.array([r[voltage_col] for r in data_rows if r[voltage_col] is not None], dtype=float)
        discharge_cap_arr = np.array([r[discharge_cap_col] for r in data_rows if r[discharge_cap_col] is not None], dtype=float)
        cycle_arr = np.array([r[cycle_col] for r in data_rows if r[cycle_col] is not None], dtype=int)
        step_arr = np.array([r[step_col] for r in data_rows if r[step_col] is not None], dtype=int)
        
        # Extract discharge segments (negative current)
        discharge_mask = current_arr < -0.01
        
        return {
            'time': time_arr,
            'current': current_arr,
            'voltage': voltage_arr,
            'discharge_capacity': discharge_cap_arr,
            'cycle_index': cycle_arr,
            'step_index': step_arr,
            'discharge_mask': discharge_mask
        }
    except Exception as e:
        print(f"Error loading CS2_36: {e}")
        return None

def load_oxford_data():
    """Load Oxford battery degradation data"""
    fpath = os.path.join(WORKSPACE, "data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat")
    mat = sio.loadmat(fpath)
    
    dc = mat['ExampleDC_C1']['dc'][0,0]
    ch = mat['ExampleDC_C1']['ch'][0,0]
    
    discharge = {
        'time': dc['t'][0,0].flatten(),
        'voltage': dc['v'][0,0].flatten(),
        'charge': dc['q'][0,0].flatten(),
        'temperature': dc['T'][0,0].flatten(),
        'current': dc['i'][0,0].flatten()
    }
    
    charge = {
        'time': ch['t'][0,0].flatten(),
        'voltage': ch['v'][0,0].flatten(),
        'charge': ch['q'][0,0].flatten(),
        'temperature': ch['T'][0,0].flatten(),
        'current': ch['i'][0,0].flatten()
    }
    
    return {'discharge': discharge, 'charge': charge}

if __name__ == '__main__':
    print("Loading NASA data...")
    nasa = load_nasa_data('B0005')
    print(f"  {len(nasa)} discharge cycles loaded")
    print(f"  First cycle: {len(nasa[0]['voltage'])} points, V range: [{nasa[0]['voltage'].min():.3f}, {nasa[0]['voltage'].max():.3f}]")
    
    print("\nLoading CS2_36 data...")
    cs2 = load_cs2_36_data(0)
    if cs2:
        print(f"  {len(cs2['time'])} data points, {len(np.unique(cs2['cycle_index']))} cycles")
    
    print("\nLoading Oxford data...")
    oxford = load_oxford_data()
    print(f"  Discharge: {len(oxford['discharge']['voltage'])} points")
    print(f"  V range: [{oxford['discharge']['voltage'].min():.3f}, {oxford['discharge']['voltage'].max():.3f}]")
