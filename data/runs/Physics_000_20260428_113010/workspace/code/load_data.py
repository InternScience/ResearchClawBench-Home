"""
Multi-component Icosahedral Shell Theory: data parsing.
Loads the dataset shipped in data/Multi-component Icosahedral Reproduction Data.txt
into a Python dict by exec-ing the assignment statements within a sandbox.
"""
from pathlib import Path
import re
import json

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "Multi-component Icosahedral Reproduction Data.txt"

def load_data():
    text = DATA_FILE.read_text()
    # Keep only lines that look like 'name = ...'
    ns = {}
    # Execute line-by-line so commented lines are skipped
    cur = ''
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('//'):
            continue
        cur += line + '\n'
    exec(cur, ns)
    keys = ns.get('data_file_indices', [])
    out = {k: ns[k] for k in keys if k in ns}
    return out

if __name__ == '__main__':
    d = load_data()
    summary = {k: (type(v).__name__, len(v) if hasattr(v,'__len__') else None) for k,v in d.items()}
    print(json.dumps(summary, indent=2))
    out_path = Path(__file__).resolve().parents[1] / 'outputs' / 'data_summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print('Wrote', out_path)
