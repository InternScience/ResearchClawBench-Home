import json
import numpy as array
import numpy as np

def parse_array(s):
    # remove brackets
    s = s.replace('[', '').replace(']', '')
    # split by space or newline
    tokens = s.split()
    return np.array([float(t) for t in tokens])

with open("outputs/parsed_data_raw.json", "r") as f:
    raw = json.load(f)

processed = {}
for k, v in raw.items():
    if 'Array' in k or 'Data' in k or 'Superfluid Stiffness' in k or 'Model' in k or 'Amplitude' in k:
        try:
            processed[k] = parse_array(v).tolist()
        except:
            pass

with open("outputs/processed_data.json", "w") as f:
    json.dump(processed, f, indent=4)

