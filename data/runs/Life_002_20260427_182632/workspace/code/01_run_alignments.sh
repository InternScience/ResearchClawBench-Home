#!/usr/bin/env bash
# Pipeline driver: split input complexes into chain PDBs and run all
# alignments (US-align in monomer + multimer modes, Foldseek easy-search).
set -euo pipefail
cd "$(dirname "$0")/.."

USALIGN=/tmp/USalign
FOLDSEEK=/tmp/foldseek/bin/foldseek

mkdir -p outputs/usalign outputs/foldseek outputs/chains

# 1) chain extraction (already done with python script 00_prep_chains.py)
python3 code/00_prep_chains.py

# 2) US-align complex (multimer) mode
$USALIGN data/7xg4.pdb data/6n40.pdb -mm 1 -ter 1 \
    -o outputs/usalign/7xg4_vs_6n40_mm1 \
    -m outputs/usalign/7xg4_vs_6n40_mm1.matrix \
    -outfmt 1 > outputs/usalign/usalign_mm1.txt 2>&1

# 3) US-align all chain pairs (proteins only)
{
  echo -e "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali"
  for c in A B C D E F G H L; do
    $USALIGN outputs/chains/7xg4_${c}.pdb outputs/chains/6n40_A.pdb \
        -mol prot -outfmt 2 2>/dev/null | tail -1
  done
} > outputs/usalign/chain_pairs.tsv

# 4) per-chain detailed alignments (for the best-scoring pair)
$USALIGN outputs/chains/7xg4_A.pdb outputs/chains/6n40_A.pdb -mol prot \
    -o outputs/usalign/best_chainpair \
    -m outputs/usalign/best_chainpair.matrix \
    > outputs/usalign/best_chainpair.txt 2>&1

# 5) Foldseek multimer-search with TMalign mode
mkdir -p /tmp/fs_run
rm -rf /tmp/fs_run/*
$FOLDSEEK easy-multimersearch outputs/7xg4_prot.pdb outputs/6n40_prot.pdb \
    /tmp/fs_run/multimer_result /tmp/fs_run/tmp \
    --format-output "query,target,qstart,qend,tstart,tend,evalue,bits,prob,alntmscore,qtmscore,ttmscore,rmsd,lddt,qcov,tcov,complexqtmscore,complexttmscore,qchains,tchains,interfacelddt,complexassignid" \
    -e 100000 --max-seqs 4000 --exhaustive-search 1 \
    --min-assigned-chains-ratio 0 --tmscore-threshold 0 -s 9.5 \
    --alignment-type 1 > outputs/foldseek/multimer_log.txt 2>&1
cp /tmp/fs_run/multimer_result outputs/foldseek/multimer_result.tsv 2>/dev/null || true
cp /tmp/fs_run/multimer_result_report outputs/foldseek/multimer_report.tsv 2>/dev/null || true

# 6) Foldseek chain-level easy-search with TMalign-mode alignment
rm -rf /tmp/fs_run/*
$FOLDSEEK easy-search outputs/7xg4_prot.pdb outputs/6n40_prot.pdb \
    /tmp/fs_run/aln.tsv /tmp/fs_run/tmp \
    --format-output "query,target,qstart,qend,tstart,tend,evalue,bits,prob,alntmscore,qtmscore,ttmscore,rmsd,lddt,qcov,tcov,u,t,qaln,taln" \
    -e 100000 --max-seqs 4000 --exhaustive-search 1 \
    --tmscore-threshold 0 -s 9.5 --alignment-type 1 > outputs/foldseek/easy_log.txt 2>&1
cp /tmp/fs_run/aln.tsv outputs/foldseek/easy_search.tsv

# 7) Foldseek chain-level easy-search with default 3Di alignment (the
#    fast-mode used for large database screens). Saves the speed/sensitivity
#    point typical of Foldseek searches.
rm -rf /tmp/fs_run/*
$FOLDSEEK easy-search outputs/7xg4_prot.pdb outputs/6n40_prot.pdb \
    /tmp/fs_run/aln_3di.tsv /tmp/fs_run/tmp \
    --format-output "query,target,qstart,qend,tstart,tend,evalue,bits,prob,alntmscore,qtmscore,ttmscore,rmsd,lddt,qcov,tcov" \
    -e 100000 --max-seqs 4000 --exhaustive-search 1 \
    --tmscore-threshold 0 -s 9.5 > outputs/foldseek/easy_log_3di.txt 2>&1
cp /tmp/fs_run/aln_3di.tsv outputs/foldseek/easy_search_3di.tsv

echo "Pipeline complete."
ls -la outputs/usalign/ outputs/foldseek/
