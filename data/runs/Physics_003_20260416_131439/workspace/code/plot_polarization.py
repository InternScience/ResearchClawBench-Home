import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/polarization_dependence_data.csv')

plt.figure(figsize=(8, 6))
# Convert to polar
angles = np.radians(df['angle_degrees'])
intensities = df['intensity']

# Fit to A + B * cos^2(theta - theta0)
from scipy.optimize import curve_fit

def fit_func(theta, A, B, theta0):
    return A + B * np.cos(theta - theta0)**2

popt, pcov = curve_fit(fit_func, angles, intensities, p0=[np.min(intensities), np.max(intensities)-np.min(intensities), 0])

theta_fit = np.linspace(0, np.pi, 100)
I_fit = fit_func(theta_fit, *popt)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(angles, intensities, 'ro', label='Data')
ax.plot(angles + np.pi, intensities, 'ro') # Symmetry for polar plot
ax.plot(theta_fit, I_fit, 'b-', label='Fit: $A + B\cos^2(\\theta - \\theta_0)$')
ax.plot(theta_fit + np.pi, I_fit, 'b-')

ax.set_title('Polarization Dependence of Replica Band Intensity')
ax.legend()
plt.tight_layout()
plt.savefig('report/images/polarization_polar.png')
plt.close()

# Also save a standard plot
plt.figure(figsize=(6, 4))
plt.plot(df['angle_degrees'], intensities, 'ro', label='Data')
theta_deg_fit = np.linspace(0, 180, 100)
plt.plot(theta_deg_fit, fit_func(np.radians(theta_deg_fit), *popt), 'b-', label='Fit')
plt.xlabel('Polarization Angle (°)')
plt.ylabel('Intensity')
plt.title('Polarization Dependence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('report/images/polarization_cartesian.png')
plt.close()
