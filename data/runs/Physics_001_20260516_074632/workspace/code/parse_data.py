"""Parse MATBG Superfluid Stiffness Core Dataset into structured numpy arrays."""
import numpy as np
import json
import re
import os

DATA_PATH = "data/MATBG Superfluid Stiffness Core Dataset.txt"
OUTPUT_DIR = "outputs"

def parse_array(text, label):
    """Extract a numpy array from text like '**Label:**\n[1. 2. 3.]'"""
    # Find the label
    pattern = re.escape(f"**{label}:**") + r"\s*\n\s*\[(.*?)\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        arr_str = match.group(1).strip()
        arr_str = arr_str.replace('\n', ' ').replace('  ', ' ')
        return np.fromstring(arr_str, sep=' ')
    # Try alternate format
    pattern2 = re.escape(f"**{label}:**") + r"\s*\n(.*?)(?=\n\*\*|\Z)"
    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        arr_str = match2.group(1).strip()
        arr_str = arr_str.replace('[', '').replace(']', '')
        arr_str = arr_str.replace('\n', ' ').replace('  ', ' ')
        return np.fromstring(arr_str, sep=' ')
    return None

def parse_all():
    with open(DATA_PATH, 'r') as f:
        content = f.read()

    results = {}

    # --- File 1: Carrier Density Dependence ---
    f1_start = content.find("**File 1:")
    f1_end = content.find("**File 2:")
    f1_text = content[f1_start:f1_end] if f1_end > 0 else content[f1_start:]

    results['n_eff'] = parse_array(f1_text, "Carrier Density Data (n_eff in m^-2)")
    results['D_s_conv'] = parse_array(f1_text, "Conventional Superfluid Stiffness (D_s_conv)")
    results['D_s_geom'] = parse_array(f1_text, "Quantum Geometric Superfluid Stiffness (D_s_geom)")
    results['D_s_exp_hole'] = parse_array(f1_text, "Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole)")
    results['D_s_exp_electron'] = parse_array(f1_text, "Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron)")

    # --- File 2: Temperature Dependence ---
    f2_start = content.find("**File 2:")
    f2_end = content.find("**File 3:")
    f2_text = content[f2_start:f2_end] if f2_end > 0 else content[f2_start:]

    results['T_array'] = parse_array(f2_text, "Temperature Array (T in K)")
    results['D_s_bcs'] = parse_array(f2_text, "BCS Model Data (D_s_bcs)")
    results['D_s_nodal'] = parse_array(f2_text, "Nodal Superconductor Data (D_s_nodal)")
    results['D_s_power_n2'] = parse_array(f2_text, "Power Law n=2.0 Data (D_s_power_n2)")
    results['D_s_power_n2_5'] = parse_array(f2_text, "Power Law n=2.5 Data (D_s_power_n2_5)")
    results['D_s_power_n3'] = parse_array(f2_text, "Power Law n=3.0 Data (D_s_power_n3)")
    results['D_s_experimental'] = parse_array(f2_text, "Experimental Data with Noise (D_s_experimental)")

    # --- File 3: Current Dependence ---
    f3_start = content.find("**File 3:")
    f3_text = content[f3_start:]

    results['I_dc'] = parse_array(f3_text, "DC Current Array (I_dc in nA)")
    results['D_s_gl'] = parse_array(f3_text, "Ginzburg-Landau Model (D_s_gl)")
    results['D_s_linear'] = parse_array(f3_text, "Linear Meissner Model (D_s_linear)")
    results['D_s_dc_exp'] = parse_array(f3_text, "Experimental DC Data (D_s_dc_exp)")
    results['P_mw'] = parse_array(f3_text, "Microwave Power Array (P_mw normalized)")
    results['I_mw_amplitude'] = parse_array(f3_text, "Microwave Current Amplitude (I_mw_amplitude in nA)")
    results['D_s_mw_exp'] = parse_array(f3_text, "Experimental Microwave Data (D_s_mw_exp)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save as npz
    np.savez(os.path.join(OUTPUT_DIR, 'parsed_data.npz'), **results)

    # Save metadata
    meta = {k: (v.shape[0] if v is not None else None) for k, v in results.items()}
    with open(os.path.join(OUTPUT_DIR, 'data_shapes.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("Parsing complete. Data shapes:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    return results

if __name__ == "__main__":
    parse_all()
