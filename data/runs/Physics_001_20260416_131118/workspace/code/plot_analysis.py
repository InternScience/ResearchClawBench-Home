import json
import numpy as np
import matplotlib.pyplot as plt
import os

with open("outputs/processed_data.json", "r") as f:
    data = json.load(f)

# Ensure output directory
os.makedirs("report/images", exist_ok=True)

# 1. Carrier Density Dependence
n_eff = np.array(data['Carrier Density Data (n_eff in m^-2):'])
Ds_conv = np.array(data['Conventional Superfluid Stiffness (D_s_conv):'])
Ds_geom = np.array(data['Quantum Geometric Superfluid Stiffness (D_s_geom):'])
Ds_exp_hole = np.array(data['Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole):'])
Ds_exp_electron = np.array(data['Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron):'])

plt.figure(figsize=(8, 6))
plt.plot(n_eff, Ds_conv, 'k--', label='Conventional (Fermi Liquid)')
plt.plot(n_eff, Ds_conv + Ds_geom, 'b-', label='Conventional + Quantum Geometric')
plt.scatter(n_eff, Ds_exp_hole, c='r', marker='o', label='Experimental (Hole-doped)', alpha=0.7)
plt.scatter(n_eff, Ds_exp_electron, c='g', marker='^', label='Experimental (Electron-doped)', alpha=0.7)
plt.xlabel('Carrier Density $n_{eff}$ ($m^{-2}$)')
plt.ylabel('Superfluid Stiffness $D_s$ (eV)')
plt.title('Superfluid Stiffness vs Carrier Density')
plt.legend()
plt.grid(True)
plt.savefig('report/images/carrier_density_dependence.png')
plt.close()

# 2. Temperature Dependence
T = np.array(data['Temperature Array (T in K):'])
Ds_bcs = np.array(data['BCS Model Data (D_s_bcs):'])
Ds_nodal = np.array(data['Nodal Superconductor Data (D_s_nodal):'])
Ds_power_n2 = np.array(data['Power Law n=2.0 Data (D_s_power_n2):'])
Ds_power_n2_5 = np.array(data['Power Law n=2.5 Data (D_s_power_n2_5):'])
Ds_power_n3 = np.array(data['Power Law n=3.0 Data (D_s_power_n3):'])
Ds_exp_T = np.array(data['Experimental Data with Noise (D_s_experimental):'])

plt.figure(figsize=(8, 6))
plt.plot(T[:len(Ds_bcs)], Ds_bcs[:len(T)], 'k--', label='BCS (s-wave)')
plt.plot(T[:len(Ds_nodal)], Ds_nodal[:len(T)], 'g-', label='Nodal (d-wave, n=1)')
plt.plot(T[:len(Ds_power_n2)], Ds_power_n2[:len(T)], 'c-', label='Power Law (n=2.0)')
plt.plot(T[:len(Ds_power_n2_5)], Ds_power_n2_5[:len(T)], 'm-', label='Power Law (n=2.5)')
plt.plot(T[:len(Ds_power_n3)], Ds_power_n3[:len(T)], 'y-', label='Power Law (n=3.0)')

min_len = min(len(T), len(Ds_exp_T))
plt.scatter(T[:min_len], Ds_exp_T[:min_len], c='r', marker='o', label='Experimental', alpha=0.5)
plt.xlabel('Temperature $T$ (K)')
plt.ylabel('Normalized Superfluid Stiffness $D_s(T)/D_s(0)$')
plt.title('Temperature Dependence of Superfluid Stiffness')
plt.legend()
plt.grid(True)
plt.savefig('report/images/temperature_dependence.png')
plt.close()

# 3. Current Dependence
I_dc = np.array(data['DC Current Array (I_dc in nA):'])
Ds_gl = np.array(data['Ginzburg-Landau Model (D_s_gl):'])
Ds_linear = np.array(data['Linear Meissner Model (D_s_linear):'])
Ds_dc_exp = np.array(data['Experimental DC Data (D_s_dc_exp):'])

plt.figure(figsize=(8, 6))
plt.plot(I_dc[:len(Ds_gl)], Ds_gl[:len(I_dc)], 'k--', label='Ginzburg-Landau')
plt.plot(I_dc[:len(Ds_linear)], Ds_linear[:len(I_dc)], 'b-', label='Linear Meissner')

min_len_dc = min(len(I_dc), len(Ds_dc_exp))
plt.scatter(I_dc[:min_len_dc], Ds_dc_exp[:min_len_dc], c='r', marker='o', label='Experimental', alpha=0.7)
plt.xlabel('DC Current $I_{dc}$ (nA)')
plt.ylabel('Normalized Superfluid Stiffness $D_s(I)/D_s(0)$')
plt.title('Current Dependence of Superfluid Stiffness')
plt.legend()
plt.grid(True)
plt.savefig('report/images/current_dependence.png')
plt.close()

# 4. Microwave Power Dependence
P_mw = np.array(data['Microwave Power Array (P_mw normalized):'])
I_mw = np.array(data['Microwave Current Amplitude (I_mw_amplitude in nA):'])
Ds_mw_exp = np.array(data['Experimental Microwave Data (D_s_mw_exp):'])

plt.figure(figsize=(8, 6))
plt.scatter(P_mw, Ds_mw_exp, c='r', marker='o', label='Experimental', alpha=0.7)
plt.xlabel('Microwave Power (Normalized)')
plt.ylabel('Normalized Superfluid Stiffness $D_s(P)/D_s(0)$')
plt.title('Microwave Power Dependence')
plt.legend()
plt.grid(True)
plt.savefig('report/images/microwave_power_dependence.png')
plt.close()

