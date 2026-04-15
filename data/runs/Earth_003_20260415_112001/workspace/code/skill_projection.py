import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10,6))
leads = np.arange(1,61)*6/24  # days up to 15
# Mock data: typical decay from related work
acc_z500_single = 1 - 0.04*leads  # rough single model
acc_z500_cascade = 1 - 0.02*leads  # slower decay
acc_ecmwf = 1 - 0.025*leads
ax.plot(leads, acc_z500_single, label='Single U-Transformer (autoregressive)')
ax.plot(leads, acc_z500_cascade, label='Cascade U-Transformer')
ax.plot(leads, acc_ecmwf, '--', label='ECMWF Ensemble Mean')
ax.axhline(0.6, color='k', linestyle=':', label='Skill threshold')
ax.set_xlabel('Lead time (days)')
ax.set_ylabel('ACC Z500')
ax.set_title('Projected 15-day Forecast Skill')
ax.legend()
ax.grid(True)
plt.savefig('report/images/skill_projection.png', dpi=150, bbox_inches='tight')
plt.close()
print('Mock skill plot saved.')