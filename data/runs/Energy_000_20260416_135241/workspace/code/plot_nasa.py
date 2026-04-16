import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/B0005.mat')
cycles = mat['B0005']['cycle'][0,0][0]

plt.figure(figsize=(8, 5))
for i, cycle in enumerate(cycles):
    if cycle['type'][0] == 'discharge':
        time = cycle['data'][0,0]['Time'][0]
        voltage = cycle['data'][0,0]['Voltage_measured'][0]
        plt.plot(time, voltage, label=f'Cycle {i}' if i < 10 else None, color='blue', alpha=0.3)
        if i > 100: break

plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('NASA B0005 Discharge Curves')
plt.grid()
plt.savefig('report/images/nasa_discharge.png')
