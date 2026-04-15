import os
import json
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_filename(fn):
    # N40_d10_r1_XEB_amplitudes.json -> N=40, d=10, r=1
    base = os.path.basename(fn).replace('.json', '')
    parts = re.match(r'N(\\d+)_d(\\d+)_r(\\d+)_XEB_(amplitudes|counts)', base)
    if parts:
        N, d, r = map(int, parts.groups()[:3])
        return N, int(d), int(r)
    return None

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def amps_to_probs(amps):
    probs = {}
    for bs, amp_str in amps.items():
        amp = complex(amp_str)
        probs[bs] = abs(amp)**2
    return probs

def compute_xeb(N, probs, counts):
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    sum_weighted = 0.0
    for bs, c in counts.items():
        if bs in probs:
            sum_weighted += c * probs[bs]
    f_lin = sum_weighted / total_shots
    f_xeb = (2**N) * f_lin  # often report this, -1 optional for full but here subset
    return f_xeb, total_shots

def process_dataset(base_dir, dataset_name):
    amps_files = glob.glob(os.path.join(base_dir, '**/*_amplitudes.json'), recursive=True)
    fids = defaultdict(list)
    shots = defaultdict(list)
    nd_pairs = defaultdict(list)  # (N,d): list r
    for af in amps_files:
        parsed = parse_filename(af)
        if parsed is None: continue
        N,d,r = parsed
        cf_path = af.replace('_amplitudes.json', '_counts.json')
        cf_path = af.replace('data/amplitudes', 'data/results')
        if not os.path.exists(cf_path): continue
        amps = load_json(af)
        counts = load_json(cf_path)
        probs = amps_to_probs(amps)
        f, ts = compute_xeb(N, probs, counts)
        fids[(N,d)].append(f)
        shots[(N,d)].append(ts)
        nd_pairs[(N,d)].append(r)
    # save per dataset
    summary = {}
    for (N,d), fs in fids.items():
        summary[(N,d)] = {'mean_F': np.mean(fs), 'std_F': np.std(fs), 'num_r': len(fs)}
    json.dump({'fids': {str(k):v for k,v in fids.items()}, 'summary': {str(k):v for k,v in summary.items()}}, open(f'outputs/{dataset_name}.json', 'w'), indent=2)
    return fids, shots, summary

# Main
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

print('Processing N40 verification (d scan)')
fids40, shots40, sum40 = process_dataset('data/amplitudes/N40_verification', 'fidelities_N40_dscan')

print('Processing N scan d12')
fids_d12, shots_d12, sum_d12 = process_dataset('data/amplitudes/N_scan_depth12', 'fidelities_Nscan_d12')

# Handle empty
if not sum40:
    print('No N40 data found')
if not sum_d12:
    print('No d12 data found')

# Plots
fig, axs = plt.subplots(1,3, figsize=(15,5))

# F vs d N=40
ds40 = sorted(set(d for N,d in sum40 if N==40))
means40 = [sum40[(40,d)]['mean_F'] for d in ds40]
stds40 = [sum40[(40,d)]['std_F'] for d in ds40]
axs[0].errorbar(ds40, means40, yerr=stds40, fmt='o-')
axs[0].set_xlabel('Depth d')
axs[0].set_ylabel('XEB Fidelity')
axs[0].set_title('N=40 F vs d')
axs[0].set_yscale('log')

# F vs N d=12
ns12 = sorted(set(N for N,d in sumN if d==12))
means12 = [sumN[(N,12)]['mean_F'] for N in ns12]
stds12 = [sumN[(N,12)]['std_F'] for N in ns12]
axs[1].errorbar(ns12, means12, yerr=stds12, fmt='o-')
axs[1].set_xlabel('N qubits')
axs[1].set_ylabel('XEB Fidelity')
axs[1].set_title('d=12 F vs N')
axs[1].set_yscale('log')

# shots overview
all_shots40 = np.concatenate([np.array(shots40[(40,d)]) for d in ds40])
axs[2].hist(np.log10(all_shots40), bins=20, alpha=0.7, label='N40')
all_shotsN = np.concatenate([np.array(shotsN[(N,12)]) for N in ns12])
axs[2].hist(np.log10(all_shotsN), bins=20, alpha=0.7, label='d12')
axs[2].set_xlabel('log10(shots)')
axs[2].set_ylabel('count')
axs[2].legend()

plt.tight_layout()
plt.savefig('report/images/main_results.png')
print('Plots saved')