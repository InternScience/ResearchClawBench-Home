import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

df = pd.read_csv('data/polarization_dependence_data.csv')
angles = np.radians(df['angle_degrees'])
intensities = df['intensity']

def fit_func(theta, A, B, theta0):
    return A + B * np.cos(theta - theta0)**2

popt, pcov = curve_fit(fit_func, angles, intensities, p0=[np.min(intensities), np.max(intensities)-np.min(intensities), 0])

print(f"Fit results:")
print(f"Background A: {popt[0]:.4f}")
print(f"Amplitude B: {popt[1]:.4f}")
print(f"Phase theta0: {np.degrees(popt[2]):.2f} deg")

# Write to outputs
with open('outputs/polarization_fit.txt', 'w') as f:
    f.write(f"Background A: {popt[0]:.4f}\n")
    f.write(f"Amplitude B: {popt[1]:.4f}\n")
    f.write(f"Phase theta0: {np.degrees(popt[2]):.2f} deg\n")

