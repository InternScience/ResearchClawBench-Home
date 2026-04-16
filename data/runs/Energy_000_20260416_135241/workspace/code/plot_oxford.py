import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat')
dc = mat['ExampleDC_C1']['dc'][0,0]
time = dc['t'][0,0].flatten()
voltage = dc['v'][0,0].flatten()
current = dc['i'][0,0].flatten()

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, voltage)
plt.ylabel('Voltage (V)')
plt.title('Oxford Dynamic Profile')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, current)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.grid()

plt.tight_layout()
plt.savefig('report/images/oxford_profile.png')
