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
    filepath = os.path.join(DATA_DIR, f'{building_id}_energy.csv')
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

def data_quality_analysis():
    print("="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    buildings = load_all_buildings()
    results = {}
    
    print("\n1. Missing Values Analysis:")
    for b, df in buildings.items():
        missing = int(df.isnull().sum().sum())
        print(f"   {b}: {missing} missing values")
        results[b] = {'missing': missing}
    
    print("\n2. Data Completeness (records expected: 8760):")
    for b, df in buildings.items():
        n_records = int(len(df))
        completeness = float((n_records / 8760) * 100)
        print(f"   {b}: {n_records} records ({completeness:.1f}%)")
        results[b]['completeness'] = completeness
    
    print("\n3. Anomaly Detection - Negative Values:")
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]']
    for b, df in buildings.items():
        has_negative = False
        for col in energy_cols:
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                has_negative = True
                print(f"   {b}: {col} has {neg_count} negative values")
        if not has_negative:
            print(f"   {b}: No negative values found")
    
    print("\n4. Outlier Detection (IQR method, >3*IQR):")
    for b, df in buildings.items():
        outliers = {}
        for col in energy_cols:
            Q1 = float(df[col].quantile(0.25))
            Q3 = float(df[col].quantile(0.75))
            IQR = Q3 - Q1
            outlier_count = int(((df[col] < (Q1 - 3*IQR)) | (df[col] > (Q3 + 3*IQR))).sum())
            outliers[col] = outlier_count
        print(f"   {b}: {outliers}")
        results[b]['outliers'] = outliers
    
    with open(os.path.join(OUTPUT_DIR, 'data_quality_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def hierarchical_consistency_check():
    print("\n" + "="*60)
    print("HIERARCHICAL CONSISTENCY CHECK")
    print("="*60)
    
    buildings = load_all_buildings()
    total = load_energy_data('Total')
    cn01 = load_energy_data('CN01')
    
    building_sum = sum([df.set_index('datetime')['Electricity [kW]'] for df in buildings.values()])
    building_sum = building_sum.sort_index()
    
    total_elec = total.set_index('datetime')['Electricity [kW]'].sort_index()
    
    corr_total = float(np.corrcoef(building_sum.values, total_elec.values)[0, 1])
    
    print(f"\nCorrelation between sum of buildings and Total: {corr_total:.6f}")
    
    mape = float(np.mean(np.abs((building_sum.values - total_elec.values) / (total_elec.values + 1e-10))) * 100)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    
    results = {
        'correlation': corr_total,
        'mape': mape
    }
    
    with open(os.path.join(OUTPUT_DIR, 'hierarchical_consistency.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_energy_profiles():
    print("\nGenerating energy load profile plots...")
    
    buildings = load_all_buildings()
    total = load_energy_data('Total')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Energy Load Profiles - HEEW Dataset (2014)', fontsize=14, fontweight='bold')
    
    ax1 = axes[0, 0]
    for b, df in buildings.items():
        df['hour_of_day'] = df['datetime'].dt.hour
        hourly_avg = df.groupby('hour_of_day')['Electricity [kW]'].mean()
        ax1.plot(hourly_avg.index, hourly_avg.values, label=b, alpha=0.7)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Electricity [kW]')
    ax1.set_title('Average Daily Electricity Profile by Building')
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    monthly_data = {}
    for b, df in buildings.items():
        monthly = df.groupby('month')['Electricity [kW]'].mean()
        monthly_data[b] = monthly
    monthly_df = pd.DataFrame(monthly_data)
    monthly_df.plot(kind='bar', ax=ax2, legend=False)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Avg Electricity [kW]')
    ax2.set_title('Monthly Average Electricity by Building')
    ax2.tick_params(axis='x', rotation=0)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    elec_data = [df['Electricity [kW]'].values for df in buildings.values()]
    ax3.boxplot(elec_data, labels=buildings.keys())
    ax3.set_xlabel('Building')
    ax3.set_ylabel('Electricity [kW]')
    ax3.set_title('Electricity Consumption Distribution')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    sample_week = total[(total['month'] == 7) & (total['day'] <= 7)]
    ax4.plot(sample_week['datetime'], sample_week['Electricity [kW]'], color='blue', linewidth=1)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Electricity [kW]')
    ax4.set_title('Total Electricity - Sample Week (July 1-7, 2014)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'energy_load_profiles.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'energy_load_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: energy_load_profiles.png")

def plot_weather_analysis():
    print("\nGenerating weather analysis plots...")
    
    weather = load_weather_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weather Data Analysis - HEEW Dataset (2014)', fontsize=14, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.hist(weather['Temperature [°F]'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax1.axvline(weather['Temperature [°F]'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {weather["Temperature [°F]"].mean():.1f}°F')
    ax1.set_xlabel('Temperature [°F]')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Temperature Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    sample = weather[(weather['datetime'].dt.month == 7)]
    ax2.plot(sample['datetime'], sample['Temperature [°F]'], color='orange', linewidth=1)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Temperature [°F]')
    ax2.set_title('Temperature - July 2014')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.scatter(weather['Temperature [°F]'], weather['Humidity [%]'], alpha=0.3, s=1)
    ax3.set_xlabel('Temperature [°F]')
    ax3.set_ylabel('Humidity [%]')
    ax3.set_title('Temperature vs Humidity')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    weather['month'] = weather['datetime'].dt.month
    monthly_temp = weather.groupby('month')['Temperature [°F]'].mean()
    monthly_hum = weather.groupby('month')['Humidity [%]'].mean()
    
    ax4_twin = ax4.twinx()
    ax4.bar(monthly_temp.index - 0.2, monthly_temp.values, 0.4, label='Temperature [°F]', color='coral')
    ax4_twin.bar(monthly_temp.index + 0.2, monthly_hum.values, 0.4, label='Humidity [%]', color='skyblue')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Temperature [°F]', color='coral')
    ax4_twin.set_ylabel('Humidity [%]', color='skyblue')
    ax4.set_title('Monthly Weather Statistics')
    ax4.set_xticks(range(1, 13))
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'weather_analysis.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'weather_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: weather_analysis.png")

def plot_correlation_heatmap():
    print("\nGenerating correlation heatmap...")
    
    total = load_energy_data('Total')
    weather = load_weather_data()
    merged = pd.merge(total, weather, on='datetime', how='inner')
    
    energy_cols = ['Electricity [kW]', 'Heat [mmBTU]', 'Cooling Energy [Ton]', 
                   'PV Power Generation [kW]', 'Greenhouse Gas Emission [Ton]']
    weather_cols = ['Temperature [°F]', 'Humidity [%]', 'Wind Speed [mph]', 
                    'Pressure [in]', 'Precipitation [in]']
    
    all_cols = energy_cols + weather_cols
    corr_matrix = merged[all_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix: Energy and Weather Variables', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: correlation_heatmap.png")

def plot_hierarchical_validation():
    print("\nGenerating hierarchical validation plot...")
    
    buildings = load_all_buildings()
    total = load_energy_data('Total')
    
    building_sum = sum([df.set_index('datetime')['Electricity [kW]'] for df in buildings.values()])
    building_sum = building_sum.sort_index()
    total_elec = total.set_index('datetime')['Electricity [kW]'].sort_index()
    
    sample_dates = total_elec.index[:24*7]  # First week
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Hierarchical Aggregation Validation', fontsize=14, fontweight='bold')
    
    ax1 = axes[0]
    ax1.plot(sample_dates, building_sum.loc[sample_dates].values, label='Sum of Buildings', linewidth=2, alpha=0.8)
    ax1.plot(sample_dates, total_elec.loc[sample_dates].values, label='Total (Ground Truth)', linewidth=2, alpha=0.8, linestyle='--')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Electricity [kW]')
    ax1.set_title('First Week of 2014: Sum of Buildings vs Total')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    np.random.seed(42)
    sample_indices = np.random.choice(len(building_sum), size=2000, replace=False)
    x_vals = building_sum.values[sample_indices]
    y_vals = total_elec.values[sample_indices]
    ax2.scatter(x_vals, y_vals, alpha=0.3, s=5)
    ax2.plot([x_vals.min(), x_vals.max()], [x_vals.min(), x_vals.max()], 'r--', linewidth=2, label='Perfect Correlation')
    ax2.set_xlabel('Sum of Buildings Electricity [kW]')
    ax2.set_ylabel('Total Electricity [kW]')
    ax2.set_title(f'Scatter Plot (Correlation: {np.corrcoef(x_vals, y_vals)[0,1]:.6f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'hierarchical_validation.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'hierarchical_validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: hierarchical_validation.png")

def generate_statistics():
    print("\n" + "="*60)
    print("GENERATING DESCRIPTIVE STATISTICS")
    print("="*60)
    
    buildings = load_all_buildings()
    total = load_energy_data('Total')
    weather = load_weather_data()
    
    stats = {}
    
    # Building statistics
    for b, df in buildings.items():
        stats[b] = {
            'electricity_mean': float(df['Electricity [kW]'].mean()),
            'electricity_std': float(df['Electricity [kW]'].std()),
            'heat_mean': float(df['Heat [mmBTU]'].mean()),
            'cooling_mean': float(df['Cooling Energy [Ton]'].mean()),
            'pv_mean': float(df['PV Power Generation [kW]'].mean()),
            'ghg_mean': float(df['Greenhouse Gas Emission [Ton]'].mean())
        }
    
    # Total statistics
    stats['Total'] = {
        'electricity_mean': float(total['Electricity [kW]'].mean()),
        'electricity_std': float(total['Electricity [kW]'].std()),
        'heat_mean': float(total['Heat [mmBTU]'].mean()),
        'cooling_mean': float(total['Cooling Energy [Ton]'].mean()),
        'pv_mean': float(total['PV Power Generation [kW]'].mean()),
        'ghg_mean': float(total['Greenhouse Gas Emission [Ton]'].mean())
    }
    
    # Weather statistics
    stats['Weather'] = {
        'temperature_mean': float(weather['Temperature [°F]'].mean()),
        'temperature_std': float(weather['Temperature [°F]'].std()),
        'humidity_mean': float(weather['Humidity [%]'].mean()),
        'wind_speed_mean': float(weather['Wind Speed [mph]'].mean()),
        'pressure_mean': float(weather['Pressure [in]'].mean()),
        'precipitation_sum': float(weather['Precipitation [in]'].sum())
    }
    
    with open(os.path.join(OUTPUT_DIR, 'descriptive_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Statistics saved to descriptive_statistics.json")
    return stats

def main():
    print("\n" + "="*60)
    print("HEEW DATASET ANALYSIS")
    print("="*60)
    print("Analyzing Hierarchical Energy and Weather (HEEW) Mini-Dataset")
    print("Data: 10 Buildings, 1 Community, 1 Total - Year 2014")
    print("="*60 + "\n")
    
    # Run all analyses
    data_quality_analysis()
    hierarchical_consistency_check()
    generate_statistics()
    
    # Generate plots
    plot_energy_profiles()
    plot_weather_analysis()
    plot_correlation_heatmap()
    plot_hierarchical_validation()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {REPORT_IMG_DIR}")

if __name__ == '__main__':
    main()
