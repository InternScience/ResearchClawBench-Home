import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv('data/polarization_dependence_data.csv')
angles = np.radians(df['angle_degrees'])
intensities = df['intensity']

def fit_func(theta, A, B, theta0):
    return A + B * np.cos(theta - theta0)**2

popt, pcov = curve_fit(fit_func, angles, intensities, p0=[np.min(intensities), np.max(intensities)-np.min(intensities), 0])

plt.figure(figsize=(6, 4))
plt.plot(df['angle_degrees'], intensities, 'ro', label='Data')
theta_deg_fit = np.linspace(0, 180, 100)
plt.plot(theta_deg_fit, fit_func(np.radians(theta_deg_fit), *popt), 'b-', label=f'Fit: $A + B\cos^2(\\theta - \\theta_0)$')
plt.xlabel('Polarization Angle (°)')
plt.ylabel('Replica Band Intensity (arb. units)')
plt.title('Polarization Dependence of Floquet-Bloch State')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('report/images/polarization_cartesian.png')
plt.close()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(angles, intensities, 'ro', label='Data')
ax.plot(angles + np.pi, intensities, 'ro')
theta_fit = np.linspace(0, 2*np.pi, 200)
ax.plot(theta_fit, fit_func(theta_fit, *popt), 'b-', label='Fit')
ax.set_title('Polarization Dependence (Polar)')
ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.savefig('report/images/polarization_polar.png')
plt.close()

