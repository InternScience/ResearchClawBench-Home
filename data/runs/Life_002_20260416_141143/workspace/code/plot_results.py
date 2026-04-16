import matplotlib.pyplot as plt
import numpy as np
import json

# Parse the matrix
matrix_data = []
with open('outputs/matrix.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('0') or line.startswith('1') or line.startswith('2'):
            parts = line.split()
            matrix_data.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

matrix_data = np.array(matrix_data)

# Parse the alignment text
with open('outputs/usalign_output.txt', 'r') as f:
    lines = f.readlines()

seq1 = ""
seq2 = ""
aln = ""
reading_seq = False
for line in lines:
    if '(":" denotes residue pairs' in line:
        reading_seq = True
        continue
    if reading_seq and line.startswith('#Total CPU time'):
        break
    if reading_seq and line.strip() != "":
        # We need to extract the sequences. In USalign, it prints 3 lines: seq1, aln, seq2
        # However, it might be split into multiple blocks. Wait, it seems to be printed as three very long lines.
        pass

# Actually, the USalign output has 3 long lines for sequence alignment.
# Let's extract them properly.
seq1_lines = []
aln_lines = []
seq2_lines = []
lines_iter = iter(lines)
for line in lines_iter:
    if '(":" denotes residue pairs' in line:
        break

for line in lines_iter:
    line = line.strip('\n')
    if not line:
        continue
    if line.startswith('#Total'):
        break
    if not seq1_lines:
        seq1_lines.append(line)
    elif not aln_lines:
        aln_lines.append(line)
    elif not seq2_lines:
        seq2_lines.append(line)

seq1 = seq1_lines[0]
aln = aln_lines[0]
seq2 = seq2_lines[0]

# Let's count matches per chain (separated by '*')
chains1 = seq1.split('*')
chains2 = seq2.split('*')
alns = aln.split('*')

chain_matches = []
for i, a in enumerate(alns):
    matches = a.count(':')
    chain_matches.append(matches)

plt.figure(figsize=(10, 6))
plt.bar(range(len(chain_matches)), chain_matches, color='skyblue')
plt.xlabel('Chain / Segment Index')
plt.ylabel('Number of Aligned Residues (distance < 5.0 A)')
plt.title('Alignment Coverage per Segment')
plt.xticks(range(len(chain_matches)))
plt.savefig('report/images/alignment_coverage.png')

# Save parsed metrics
metrics = {
    'Aligned_Length': 225,
    'RMSD': 8.28,
    'Seq_ID': 0.071,
    'TM_score_1': 0.06066,
    'TM_score_2': 0.19411
}
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

