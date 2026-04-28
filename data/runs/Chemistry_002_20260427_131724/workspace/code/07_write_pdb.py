"""Write predicted top1 pose as a PDB file (chain A unchanged, chain D rotated/translated)."""
import numpy as np

T = np.load('outputs/top1_pose.npz')
cD = T['cD']
out_lines=[]
i_D = 0
for ln in open('data/1brs_AD.pdb'):
    if not ln.startswith('ATOM'):
        out_lines.append(ln)
        continue
    chain = ln[21]
    if chain == 'A':
        out_lines.append(ln)
    elif chain == 'D':
        x,y,z = cD[i_D]
        new = ln[:30] + f'{x:8.3f}{y:8.3f}{z:8.3f}' + ln[54:]
        out_lines.append(new)
        i_D += 1
    else:
        out_lines.append(ln)
open('outputs/top1_predicted.pdb','w').writelines(out_lines)
print('wrote outputs/top1_predicted.pdb, atoms moved:', i_D)
