import pandas as pd
import glob
import os

calendar_files = glob.glob('data/glambie/results/calendar_years/*.csv')
hydro_files = glob.glob('data/glambie/results/hydrological_years/*.csv')

print(f"Calendar files: {len(calendar_files)}")
print(f"Hydrological files: {len(hydro_files)}")

df_global = pd.read_csv('data/glambie/results/calendar_years/0_global.csv')
print("\nGlobal Calendar Year Data:")
print(df_global.head())
print(df_global.columns)
print(f"Global min year: {df_global['start_dates'].min()}, max year: {df_global['end_dates'].max()}")

df_alaska = pd.read_csv('data/glambie/results/hydrological_years/1_alaska.csv')
print("\nAlaska Hydrological Year Data:")
print(df_alaska.head())
print(df_alaska.columns)
print(f"Alaska min year: {df_alaska['start_dates'].min()}, max year: {df_alaska['end_dates'].max()}")
