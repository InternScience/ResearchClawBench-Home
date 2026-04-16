import re

def parse_usalign(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    tm1, tm2, rmsd, seqid, aln_len = None, None, None, None, None
    for line in lines:
        if line.startswith("Aligned length="):
            parts = line.split(',')
            aln_len = int(parts[0].split('=')[1].strip())
            rmsd = float(parts[1].split('=')[1].strip())
            seqid = float(parts[2].split('=')[2].strip())
        elif line.startswith("TM-score=") and "normalized by length of Structure_1" in line:
            tm1 = float(line.split('=')[1].split('(')[0].strip())
        elif line.startswith("TM-score=") and "normalized by length of Structure_2" in line:
            tm2 = float(line.split('=')[1].split('(')[0].strip())
            
    print(f"Aligned Length: {aln_len}")
    print(f"RMSD: {rmsd}")
    print(f"Seq ID: {seqid}")
    print(f"TM-score (normalized by struct 1): {tm1}")
    print(f"TM-score (normalized by struct 2): {tm2}")

parse_usalign('outputs/usalign_output.txt')
