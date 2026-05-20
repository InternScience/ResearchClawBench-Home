import numpy as np
import json
from pathlib import Path

workspace = Path(__file__).resolve().parent.parent
data_path = workspace / "data" / "MATBG Superfluid Stiffness Core Dataset.txt"
output_dir = workspace / "outputs"
output_dir.mkdir(exist_ok=True)

with open(data_path, "r") as f:
    lines = f.readlines()

def extract_array_after_label(lines, label):
    idx = 0
    while idx < len(lines):
        if label in lines[idx]:
            break
        idx += 1
    else:
        raise ValueError(f"Label not found: {label}")
    idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    block = []
    while idx < len(lines):
        line = lines[idx]
        block.append(line)
        if "]" in line:
            break
        idx += 1
    text = "".join(block)
    start = text.find('[')
    end = text.find(']', start)
    if start == -1 or end == -1:
        raise ValueError(f"Brackets not found for {label}")
    nums_str = text[start+1:end]
    nums_str = nums_str.replace('\n', ' ')
    nums = [float(x) for x in nums_str.split() if x]
    return np.array(nums)

arrays = {}
arrays["n_eff"] = extract_array_after_label(lines, "Carrier Density Data (n_eff in m^-2):")
arrays["D_s_conv"] = extract_array_after_label(lines, "Conventional Superfluid Stiffness (D_s_conv):")
arrays["D_s_geom"] = extract_array_after_label(lines, "Quantum Geometric Superfluid Stiffness (D_s_geom):")
arrays["D_s_exp_hole"] = extract_array_after_label(lines, "Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole):")
arrays["D_s_exp_electron"] = extract_array_after_label(lines, "Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron):")

arrays["T"] = extract_array_after_label(lines, "Temperature Array (T in K):")
arrays["D_s_bcs"] = extract_array_after_label(lines, "BCS Model Data (D_s_bcs):")
arrays["D_s_nodal"] = extract_array_after_label(lines, "Nodal Superconductor Data (D_s_nodal):")
arrays["D_s_power_n2"] = extract_array_after_label(lines, "Power Law n=2.0 Data (D_s_power_n2):")
arrays["D_s_power_n2_5"] = extract_array_after_label(lines, "Power Law n=2.5 Data (D_s_power_n2_5):")
arrays["D_s_power_n3"] = extract_array_after_label(lines, "Power Law n=3.0 Data (D_s_power_n3):")
arrays["D_s_experimental"] = extract_array_after_label(lines, "Experimental Data with Noise (D_s_experimental):")

arrays["I_dc"] = extract_array_after_label(lines, "DC Current Array (I_dc in nA):")
arrays["D_s_gl"] = extract_array_after_label(lines, "Ginzburg-Landau Model (D_s_gl):")
arrays["D_s_linear"] = extract_array_after_label(lines, "Linear Meissner Model (D_s_linear):")
arrays["D_s_dc_exp"] = extract_array_after_label(lines, "Experimental DC Data (D_s_dc_exp):")
arrays["P_mw"] = extract_array_after_label(lines, "Microwave Power Array (P_mw normalized):")
arrays["I_mw_amplitude"] = extract_array_after_label(lines, "Microwave Current Amplitude (I_mw_amplitude in nA):")
arrays["D_s_mw_exp"] = extract_array_after_label(lines, "Experimental Microwave Data (D_s_mw_exp):")

npz_path = output_dir / "parsed_data.npz"
np.savez_compressed(npz_path, **arrays)

info = {k: {"shape": list(v.shape), "min": float(v.min()), "max": float(v.max())} for k, v in arrays.items()}
with open(output_dir / "data_info.json", "w") as f:
    json.dump(info, f, indent=2)

print("Parsed and saved to", npz_path)
print("Keys:", list(arrays.keys()))
