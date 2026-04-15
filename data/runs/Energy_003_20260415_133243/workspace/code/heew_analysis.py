
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import json
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/outputs', exist_ok=True)
os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/report/images', exist_ok=True)

DATA_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/data/HEEW_Mini-Dataset'
OUTPUT_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/outputs'
REPORT_IMG_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_003_20260415_133243/report/images'

def load_energy_data(building_id):
    filepath = os.path.join(DATA_DIR, building_id + '_energy.csv')
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df

def load_weather_data():
    filepath = os.path.join(DATA_DIR, 'Total_weather.csv')
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def load_all_buildings():
    buildings = ['BN001', 'BN002', 'BN003', 'BN004', 'BN005', 
                 'BN006', 'BN007', 'BN008', 'BN009', 'BN010']
    data = {}
    for b in buildings:
        data[b] = load_energy_data(b)
    return data

def main():
    print('HEEW Dataset Analysis Starting...')
    buildings = load_all_buildings()
    print(f'Loaded {len(buildings)} buildings')

if __name__ == '__main__':
    main()
