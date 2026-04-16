import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs and images directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
buses = pd.read_csv('data/buses.csv', index_col=0)
links = pd.read_csv('data/links.csv')
demand = pd.read_csv('data/demand.csv')
generators = pd.read_csv('data/generators.csv')
storage = pd.read_csv('data/storage.csv')
wind_cf = pd.read_csv('data/wind_cf.csv')

# The demand seems to be way too high. Let's check the demand file again.
print(demand.head())
