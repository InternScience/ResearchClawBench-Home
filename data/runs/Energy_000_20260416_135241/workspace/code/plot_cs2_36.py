import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('data/CS2_36/CS2_36_1_10_11.xlsx', sheet_name='Channel_1-009')
discharge = df[df['Step_Index'] == 7].copy()
discharge['Time_s'] = discharge['Test_Time(s)'] - discharge['Test_Time(s)'].min()

plt.figure(figsize=(8, 5))
plt.plot(discharge['Time_s'], discharge['Voltage(V)'], label='Discharge Voltage (1C)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('CS2_36 1C Discharge Curve')
plt.legend()
plt.grid()
plt.savefig('report/images/cs2_36_discharge.png')
