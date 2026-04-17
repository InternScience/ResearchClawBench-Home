#!/usr/bin/env python3
"""
Part 2: SKEMPI v2 Analysis and HADDOCK-style Scoring
"""
import os
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Chemistry_002_20260416_175027'
SKEMPI_FILE = os.path.join(BASE, 'data/skempi_v2.csv')
OUTPUT_DIR = os.path.join(BASE, 'outputs')

# ============================================================
# 1. Parse SKEMPI v2 for 1BRS
# ============================================================
R = 1.987e-3  # kcal/(mol*K)
T = 298.15    # K (25C standard)

skempi_data = []
with open(SKEMPI_FILE) as f:
    header = f.readline().strip('#').strip().split(';')
    for line in f:
        fields = line.strip().split(';')
        if not fields[0].startswith('1BRS'):
            continue
        
        entry = dict(zip(header, fields))
        
        # Parse mutation info
        mutation = entry.get('Mutation(s)_PDB', '')
        cleaned = entry.get('Mutation(s)_cleaned', '')
        location = entry.get('iMutation_Location(s)', '')
        
        # Parse affinities
        try:
            kd_mut = float(entry.get('Affinity_mut_parsed', ''))
            kd_wt = float(entry.get('Affinity_wt_parsed', ''))
        except (ValueError, TypeError):
            continue
        
        if kd_mut <= 0 or kd_wt <= 0:
            continue
        
        # Compute ddG = RT * ln(Kd_mut/Kd_wt)
        ddG = R * T * math.log(kd_mut / kd_wt)
        
        # Parse mutation details
        # Format: XCnnY where X=wt_aa, C=chain, nn=resnum, Y=mut_aa
        mut_chain = mutation[1] if len(mutation) > 1 else ''
        mut_resnum = ''
        wt_aa = mutation[0] if mutation else ''
        mut_aa = mutation[-1] if mutation else ''
        
        # Extract residue number
        import re
        m = re.match(r'([A-Z])([A-Z])(\d+)([A-Z])', mutation)
        if m:
            wt_aa = m.group(1)
            mut_chain = m.group(2)
            mut_resnum = int(m.group(3))
            mut_aa = m.group(4)
        
        skempi_data.append({
            'mutation': mutation,
            'cleaned': cleaned,
            'location': location,
            'chain': mut_chain,
            'resnum': mut_resnum,
            'wt_aa': wt_aa,
            'mut_aa': mut_aa,
            'kd_mut': kd_mut,
            'kd_wt': kd_wt,
            'ddG': ddG,
            'method': entry.get('Method', ''),
            'temperature': entry.get('Temperature', ''),
        })

print(f"Parsed {len(skempi_data)} 1BRS mutations from SKEMPI v2")

# Filter single mutations only
single_muts = [s for s in skempi_data if ',' not in s['mutation']]
print(f"Single mutations: {len(single_muts)}")

# ============================================================
# 2. Analyze mutations by location
# ============================================================
location_stats = defaultdict(list)
for s in single_muts:
    location_stats[s['location']].append(s['ddG'])

print("\nddG by mutation location:")
for loc in sorted(location_stats):
    vals = location_stats[loc]
    print(f"  {loc}: n={len(vals)}, mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, "
          f"min={np.min(vals):.2f}, max={np.max(vals):.2f}")

# ============================================================
# 3. Analyze mutations by chain
# ============================================================
chain_stats = defaultdict(list)
for s in single_muts:
    chain_stats[s['chain']].append(s['ddG'])

print("\nddG by chain:")
for ch in sorted(chain_stats):
    vals = chain_stats[ch]
    print(f"  Chain {ch}: n={len(vals)}, mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")

# ============================================================
# 4. Analyze mutations by residue type
# ============================================================
# Amino acid properties
AA_PROPERTIES = {
    'G': 'nonpolar', 'A': 'nonpolar', 'V': 'nonpolar', 'L': 'nonpolar',
    'I': 'nonpolar', 'P': 'nonpolar', 'F': 'aromatic', 'W': 'aromatic',
    'M': 'nonpolar', 'S': 'polar', 'T': 'polar', 'C': 'polar',
    'Y': 'aromatic', 'N': 'polar', 'Q': 'polar', 'D': 'negative',
    'E': 'negative', 'K': 'positive', 'R': 'positive', 'H': 'positive'
}

wt_type_stats = defaultdict(list)
for s in single_muts:
    wt_type = AA_PROPERTIES.get(s['wt_aa'], 'unknown')
    wt_type_stats[wt_type].append(s['ddG'])

print("\nddG by wild-type residue type:")
for t in sorted(wt_type_stats):
    vals = wt_type_stats[t]
    print(f"  {t}: n={len(vals)}, mean={np.mean(vals):.2f}")

# ============================================================
# 5. Identify hotspot residues
# ============================================================
hotspots = []
for s in single_muts:
    if s['ddG'] > 2.0:  # Hotspot: ddG > 2 kcal/mol
        hotspots.append(s)

print(f"\nHotspot residues (ddG > 2 kcal/mol): {len(hotspots)}")
for h in sorted(hotspots, key=lambda x: x['ddG'], reverse=True):
    print(f"  {h['mutation']}: ddG={h['ddG']:.2f} kcal/mol, location={h['location']}")

# ============================================================
# 6. Save results
# ============================================================
skempi_results = {
    'n_total': len(skempi_data),
    'n_single': len(single_muts),
    'n_hotspots': len(hotspots),
    'location_stats': {loc: {'n': len(vals), 'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                       for loc, vals in location_stats.items()},
    'chain_stats': {ch: {'n': len(vals), 'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                    for ch, vals in chain_stats.items()},
    'hotspots': [{'mutation': h['mutation'], 'ddG': h['ddG'], 'location': h['location']}
                 for h in sorted(hotspots, key=lambda x: x['ddG'], reverse=True)],
    'all_single_mutations': [{'mutation': s['mutation'], 'ddG': s['ddG'], 'location': s['location'],
                               'chain': s['chain'], 'resnum': s['resnum']}
                              for s in single_muts]
}

with open(os.path.join(OUTPUT_DIR, 'skempi_analysis.json'), 'w') as f:
    json.dump(skempi_results, f, indent=2)

# Also save as CSV for plotting
df = pd.DataFrame(single_muts)
df.to_csv(os.path.join(OUTPUT_DIR, 'skempi_1brs_mutations.csv'), index=False)

print("\nSKEMPI analysis saved")
