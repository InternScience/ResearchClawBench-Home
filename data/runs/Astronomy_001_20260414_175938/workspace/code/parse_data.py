import ast
import json
import csv
from pathlib import Path

# Read the data file
with open('data/DESI_EDE_Repro_Data.txt', 'r') as f:
    content = f.read()

# Extract the parameter dicts using regex or manual slice
# Since it's simple, use ast.literal_eval on each block
lines = content.splitlines()

# Parse lcdm_params
start = lines.index("lcdm_params = {") 
end = content.find('}', start) +1
lcdm_str = content[start:end].strip()
lcdm_params = ast.literal_eval(lcdm_str)

# Similarly for others
start = content.find("ede_params = {")
end = content.find('}', start) +1
ede_str = content[start:end].strip()
ede_params = ast.literal_eval(ede_str)

start = content.find("w0wa_params = {")
end = content.find('}', start) +1
w0wa_str = content[start:end].strip()
w0wa_params = ast.literal_eval(w0wa_str)

# Points
dvrd_start = content.find('desi_dvrd_points = [')
dvrd_end = content.find(']', dvrd_start) +1
desi_dvrd = ast.literal_eval(content[dvrd_start:dvrd_end])

fap_start = content.find('desi_fap_points = [')
fap_end = content.find(']', fap_start) +1
desi_fap = ast.literal_eval(content[fap_start:fap_end])

sne_start = content.find('sne_mu_points = [')
sne_end = content.find(']', sne_start) +1
sne_mu = ast.literal_eval(content[sne_start:sne_end])

# Save params json
params = {
    'lcdm': dict(lcdm_params),
    'ede': dict(ede_params),
    'w0wa': dict(w0wa_params)
}
Path('outputs').mkdir(exist_ok=True)
with open('outputs/parameters.json', 'w') as f:
    json.dump(params, f, indent=2)

# Save csv
with open('outputs/bao_dvrd.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['z', 'delta_dv_rd', 'error'])
    writer.writerows(desi_dvrd)

with open('outputs/bao_fap.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['z', 'delta_fap', 'error'])
    writer.writerows(desi_fap)

with open('outputs/sne_mu.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['z', 'delta_mu', 'error'])
    writer.writerows(sne_mu)

print('Parsed data saved to outputs/')