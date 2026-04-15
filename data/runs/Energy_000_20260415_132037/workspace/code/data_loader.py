"""
Data Loader Module for Li-ion Battery Parameter Identification
Loads and preprocesses NASA, CALCE, and Oxford battery datasets
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
import os

class BatteryDataLoader:
    """Unified interface for loading battery aging datasets"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_nasa_data(self, battery_id='B0005'):
        """Load NASA PCoE battery aging dataset"""
        filepath = os.path.join(self.data_path, 'NASA PCoE Dataset Repository', 
                                '1. BatteryAgingARC-FY08Q4', f'{battery_id}.mat')
        mat_data = sio.loadmat(filepath)
        battery_data = mat_data[battery_id]
        cycles = battery_data[0,0]['cycle'][0]
        
        discharge_data = []
        for c in cycles:
            if c['type'][0] == 'discharge':
                data = c['data'][0,0]
                discharge_data.append({
                    'voltage': data['Voltage_measured'].flatten(),
                    'current': data['Current_measured'].flatten(),
                    'temperature': data['Temperature_measured'].flatten(),
                    'time': data['Time'].flatten(),
                    'capacity': data['Capacity'][0,0]
                })
        return discharge_data
    
    def load_oxford_data(self):
        """Load Oxford Battery Degradation Dataset"""
        filepath = os.path.join(self.data_path, 'Oxford Battery Degradation Dataset', 
                                'ExampleDC_C1.mat')
        mat_data = sio.loadmat(filepath)
        data = mat_data['ExampleDC_C1'][0,0]
        
        # Extract discharge data
        dc = data['dc'][0,0]
        discharge_data = {
            'time': dc['t'].flatten(),
            'voltage': dc['v'].flatten(),
            'current': dc['i'].flatten() / 1000,  # Convert mA to A
            'temperature': dc['T'].flatten(),
            'capacity': -dc['q'].flatten() / 1000  # Convert mAh to Ah
        }
        
        # Extract charge data
        ch = data['ch'][0,0]
        charge_data = {
            'time': ch['t'].flatten(),
            'voltage': ch['v'].flatten(),
            'current': ch['i'].flatten() / 1000,
            'temperature': ch['T'].flatten(),
            'capacity': ch['q'].flatten() / 1000
        }
        
        return {'charge': charge_data, 'discharge': discharge_data}
    
    def extract_discharge_features(self, discharge_cycles, soc_points=None):
        """Extract voltage features at specific SOC points"""
        if soc_points is None:
            soc_points = np.linspace(0, 1, 21)  # 5% intervals
        
        features = []
        capacities = []
        
        for cycle in discharge_cycles:
            v = cycle['voltage']
            t = cycle['time']
            cap = cycle['capacity']
            
            # Normalize time to SOC
            if len(t) > 1:
                soc = 1 - (t - t.min()) / (t.max() - t.min())
                
                # Interpolate voltage at SOC points
                try:
                    v_interp = interp1d(soc, v, kind='linear', 
                                       bounds_error=False, fill_value='extrapolate')
                    v_features = v_interp(soc_points)
                    features.append(v_features)
                    capacities.append(cap)
                except:
                    continue
        
        return np.array(features), np.array(capacities), soc_points
    
    def get_experimental_data_summary(self):
        """Get summary statistics of all datasets"""
        summary = {}
        
        # NASA data
        for bid in ['B0005', 'B0006', 'B0007', 'B0018']:
            try:
                data = self.load_nasa_data(bid)
                summary[f'NASA_{bid}'] = {
                    'discharge_cycles': len(data),
                    'avg_capacity': np.mean([d['capacity'] for d in data]),
                    'voltage_range': [np.min([d['voltage'].min() for d in data]),
                                     np.max([d['voltage'].max() for d in data])]
                }
            except Exception as e:
                summary[f'NASA_{bid}'] = {'error': str(e)}
        
        # Oxford data
        try:
            oxford = self.load_oxford_data()
            summary['Oxford'] = {
                'discharge_points': len(oxford['discharge']['voltage']),
                'voltage_range': [oxford['discharge']['voltage'].min(),
                                 oxford['discharge']['voltage'].max()],
                'current_type': 'dynamic (Artemis Urban)'
            }
        except Exception as e:
            summary['Oxford'] = {'error': str(e)}
        
        return summary
