"""Parse the MATBG Superfluid Stiffness Core Dataset into structured numpy arrays."""
import numpy as np
import json
import os
import re

# Read the raw data file
data_file = "data/MATBG Superfluid Stiffness Core Dataset.txt"
with open(data_file, 'r') as f:
    content = f.read()

def extract_array_simple(label_keyword, content):
    """Extract numpy array by finding the label keyword and then the bracket after it."""
    idx = content.find(label_keyword)
    if idx == -1:
        return None
    # Find the opening bracket after the label
    bracket_start = content.find('[', idx)
    bracket_end = content.find(']', bracket_start)
    if bracket_start == -1 or bracket_end == -1:
        return None
    values_str = content[bracket_start+1:bracket_end].strip()
    values = values_str.split()
    return np.array([float(v) for v in values])

# File 1: Carrier density data
n_eff = extract_array_simple("Carrier Density Data", content)
D_s_conv = extract_array_simple("Conventional Superfluid Stiffness", content)
D_s_geom = extract_array_simple("Quantum Geometric Superfluid Stiffness", content)
D_s_exp_hole = extract_array_simple("Experimental Superfluid Stiffness Hole-doped", content)
D_s_exp_electron = extract_array_simple("Experimental Superfluid Stiffness Electron-doped", content)

# File 2: Temperature dependence
T = extract_array_simple("Temperature Array", content)
D_s_bcs = extract_array_simple("BCS Model Data", content)
D_s_nodal = extract_array_simple("Nodal Superconductor Data", content)
D_s_power_n2 = extract_array_simple("Power Law n=2.0 Data", content)
D_s_power_n2_5 = extract_array_simple("Power Law n=2.5 Data", content)
D_s_power_n3 = extract_array_simple("Power Law n=3.0 Data", content)
D_s_experimental = extract_array_simple("Experimental Data with Noise", content)

# File 3: Current dependence
I_dc = extract_array_simple("DC Current Array", content)
D_s_gl = extract_array_simple("Ginzburg-Landau Model", content)
D_s_linear = extract_array_simple("Linear Meissner Model", content)
D_s_dc_exp = extract_array_simple("Experimental DC Data", content)
P_mw = extract_array_simple("Microwave Power Array", content)
I_mw_amplitude = extract_array_simple("Microwave Current Amplitude", content)
D_s_mw_exp = extract_array_simple("Experimental Microwave Data", content)

# Save parsed data
os.makedirs("outputs", exist_ok=True)

np.savez("outputs/carrier_density.npz", 
         n_eff=n_eff, D_s_conv=D_s_conv, D_s_geom=D_s_geom,
         D_s_exp_hole=D_s_exp_hole, D_s_exp_electron=D_s_exp_electron)

np.savez("outputs/temperature_dependence.npz",
         T=T, D_s_bcs=D_s_bcs, D_s_nodal=D_s_nodal,
         D_s_power_n2=D_s_power_n2, D_s_power_n2_5=D_s_power_n2_5,
         D_s_power_n3=D_s_power_n3, D_s_experimental=D_s_experimental)

np.savez("outputs/current_dependence.npz",
         I_dc=I_dc, D_s_gl=D_s_gl, D_s_linear=D_s_linear,
         D_s_dc_exp=D_s_dc_exp, P_mw=P_mw, I_mw_amplitude=I_mw_amplitude,
         D_s_mw_exp=D_s_mw_exp)

# Print summary
print("=== Data Summary ===")
for name, arr in [("n_eff", n_eff), ("D_s_conv", D_s_conv), ("D_s_geom", D_s_geom),
                  ("D_s_exp_hole", D_s_exp_hole), ("D_s_exp_electron", D_s_exp_electron),
                  ("T", T), ("D_s_bcs", D_s_bcs), ("D_s_nodal", D_s_nodal),
                  ("D_s_power_n2", D_s_power_n2), ("D_s_power_n2_5", D_s_power_n2_5),
                  ("D_s_power_n3", D_s_power_n3), ("D_s_experimental", D_s_experimental),
                  ("I_dc", I_dc), ("D_s_gl", D_s_gl), ("D_s_linear", D_s_linear),
                  ("D_s_dc_exp", D_s_dc_exp), ("P_mw", P_mw), ("I_mw_amplitude", I_mw_amplitude),
                  ("D_s_mw_exp", D_s_mw_exp)]:
    if arr is not None:
        print(f"{name}: {len(arr)} points, range [{arr[0]:.4e}, {arr[-1]:.4e}]")
    else:
        print(f"{name}: NOT FOUND!")

print("\nData saved successfully!")
