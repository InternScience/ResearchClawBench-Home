#!/usr/bin/env python3
"""Reproducible protein-complex alignment fallback for 7xg4 vs 6n40.

This script does not claim to reimplement Foldseek-Multimer. It provides a
local, deterministic structural-alignment analysis when Foldseek/USalign/TMalign
executables are absent. It parses protein CA atoms, performs sequence-aware
chain-pair dynamic programming on CA distance under iterative Kabsch
superposition, ranks chain assignments, computes a TM-score-like similarity, and
exports tables/figures for the report.
"""
from __future__ import annotations
import os, json, math, itertools, csv
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
DATA=os.path.join(ROOT,'data')
OUT=os.path.join(ROOT,'outputs')
IMG=os.path.join(ROOT,'report','images')
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)

AA3_TO_1={
 'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
 'SEC':'U','PYL':'O','ASX':'B','GLX':'Z','UNK':'X'
}

def parse_pdb_ca(path):
    chains=defaultdict(dict)
    metadata={'path':path,'header':None,'title':[],'resolution_A':None,'compnd':[]}
    with open(path) as f:
        for line in f:
            rec=line[:6].strip()
            if rec=='HEADER': metadata['header']=line.rstrip('\n')
            elif rec=='TITLE': metadata['title'].append(line[10:].strip())
            elif rec=='COMPND': metadata['compnd'].append(line[10:].strip())
            elif line.startswith('REMARK   2 RESOLUTION.'):
                parts=line.split()
                for x in parts:
                    try:
                        val=float(x)
                        # skip REMARK record number; first real resolution follows RESOLUTION.
                        if val > 2.5:
                            metadata['resolution_A']=val
                            break
                    except ValueError:
                        pass
            elif rec=='ATOM':
                atom=line[12:16].strip(); resn=line[17:20].strip(); ch=line[21].strip() or '_'
                if resn not in AA3_TO_1: continue
                resseq=line[22:26].strip(); icode=line[26].strip()
                key=(resseq,icode,resn)
                if atom=='CA':
                    try: coord=np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])], dtype=float)
                    except ValueError: continue
                    chains[ch][key]=coord
    out={}
    for ch,d in chains.items():
        # PDB order mostly by residue; sort by integer if possible then insertion code
        def sortkey(k):
            try: r=int(k[0])
            except: r=10**9
            return (r,k[1],k[0])
        keys=sorted(d, key=sortkey)
        out[ch]={'residue_ids':[f'{k[2]}{k[0]}{k[1]}' for k in keys],
                 'sequence':' '.join(AA3_TO_1[k[2]] for k in keys).replace(' ',''),
                 'coords':np.vstack([d[k] for k in keys]) if keys else np.zeros((0,3))}
    metadata['title']=' '.join(metadata['title'])
    return metadata,out

def kabsch(P,Q):
    """Return R,t so P @ R.T + t approximately equals Q."""
    P=np.asarray(P,float); Q=np.asarray(Q,float)
    Pc=P.mean(axis=0); Qc=Q.mean(axis=0)
    X=P-Pc; Y=Q-Qc
    C=X.T @ Y
    V,S,Wt=np.linalg.svd(C)
    d=np.sign(np.linalg.det(V @ Wt))
    D=np.diag([1,1,d])
    R=V @ D @ Wt
    t=Qc - Pc @ R.T
    return R,t

def tm_d0(L):
    L=max(int(L),1)
    if L <= 15: return 0.5
    return max(0.5, 1.24*((L-15)**(1/3))-1.8)

def tm_score(dist, Lnorm):
    d0=tm_d0(Lnorm)
    return float(np.sum(1.0/(1.0+(np.asarray(dist)/d0)**2))/max(Lnorm,1))

def nw_align(seq1, seq2, match=2.0, mismatch=-1.0, gap=-1.5):
    n,m=len(seq1),len(seq2)
    F=np.zeros((n+1,m+1),float); Ptr=np.zeros((n+1,m+1),np.int8)
    F[:,0]=np.arange(n+1)*gap; F[0,:]=np.arange(m+1)*gap
    Ptr[1:,0]=1; Ptr[0,1:]=2
    for i in range(1,n+1):
        a=seq1[i-1]
        for j in range(1,m+1):
            s=match if a==seq2[j-1] else mismatch
            vals=[F[i-1,j-1]+s, F[i-1,j]+gap, F[i,j-1]+gap]
            Ptr[i,j]=int(np.argmax(vals)); F[i,j]=vals[Ptr[i,j]]
    pairs=[]; i=n; j=m
    while i>0 or j>0:
        p=Ptr[i,j]
        if i>0 and j>0 and p==0:
            pairs.append((i-1,j-1)); i-=1; j-=1
        elif i>0 and (j==0 or p==1): i-=1
        else: j-=1
    pairs.reverse()
    return pairs, float(F[n,m])

def distance_dp(P,Q,R,t, gap=-0.6, sigma=4.0, max_sep=12.0):
    # local-ish Needleman-Wunsch on structural proximity; returns matched index pairs
    Pt=P @ R.T + t
    n,m=len(P),len(Q)
    F=np.zeros((n+1,m+1),float); Ptr=np.zeros((n+1,m+1),np.int8)
    F[:,0]=np.arange(n+1)*gap; F[0,:]=np.arange(m+1)*gap
    Ptr[1:,0]=1; Ptr[0,1:]=2
    # compute row-wise distances to save memory OK <= 726
    for i in range(1,n+1):
        dists=np.linalg.norm(Q - Pt[i-1], axis=1)
        for j in range(1,m+1):
            d=dists[j-1]
            s=1.0/(1+(d/sigma)**2) - 0.15
            if d>max_sep: s -= 0.35
            vals=[F[i-1,j-1]+s, F[i-1,j]+gap, F[i,j-1]+gap]
            Ptr[i,j]=int(np.argmax(vals)); F[i,j]=vals[Ptr[i,j]]
    pairs=[]; i=n; j=m
    while i>0 or j>0:
        p=Ptr[i,j]
        if i>0 and j>0 and p==0:
            pairs.append((i-1,j-1)); i-=1; j-=1
        elif i>0 and (j==0 or p==1): i-=1
        else: j-=1
    pairs.reverse()
    return pairs, float(F[n,m])

def align_chain(P,Q,seqP,seqQ, seed='sequence'):
    if len(P)<3 or len(Q)<3:
        return None
    if seed=='sequence': pairs,_=nw_align(seqP,seqQ)
    else:
        n=min(len(P),len(Q)); pairs=[(i,i) for i in range(n)]
    if len(pairs)<3: pairs=[(i,i) for i in range(min(len(P),len(Q)))]
    R=np.eye(3); t=np.zeros(3)
    best=None
    prev=None
    for it in range(8):
        Pm=np.vstack([P[i] for i,j in pairs]); Qm=np.vstack([Q[j] for i,j in pairs])
        R,t=kabsch(Pm,Qm)
        pairs,score=distance_dp(P,Q,R,t)
        if len(pairs)<3: break
        key=(len(pairs), round(score,5))
        if key==prev: break
        prev=key
    Pm=np.vstack([P[i] for i,j in pairs]); Qm=np.vstack([Q[j] for i,j in pairs])
    R,t=kabsch(Pm,Qm)
    dist=np.linalg.norm(Pm @ R.T + t - Qm, axis=1)
    Lnorm=max(len(P),len(Q))
    res={'n_query':len(P),'n_target':len(Q),'aligned_len':len(pairs),'coverage_query':len(pairs)/len(P),'coverage_target':len(pairs)/len(Q),
         'rmsd':float(np.sqrt(np.mean(dist**2))), 'mean_dist':float(np.mean(dist)), 'median_dist':float(np.median(dist)),
         'tm_norm_max':tm_score(dist,Lnorm),'tm_norm_query':tm_score(dist,len(P)),'tm_norm_target':tm_score(dist,len(Q)),
         'rotation':R.tolist(),'translation':t.tolist(),'pairs':pairs,'distances':dist.tolist()}
    return res

def write_pdb_superposed(qmeta,qchains,tmeta,tchains,assignment,R,t,matched_pairs):
    qpath=qmeta['path']; tpath=tmeta['path']; outp=os.path.join(OUT,'superposed_matched_ca.pdb')
    # output only matched CA pseudo complex for compact validation, transformed query and native target
    serial=1
    with open(outp,'w') as f:
        f.write('REMARK Superposed matched CA atoms: transformed 7xg4 query (chains original) and 6n40 target\n')
        for qch,tch,pairs in matched_pairs:
            Q=tchains[tch]['coords']; P=qchains[qch]['coords']
            for k,(i,j) in enumerate(pairs):
                x,y,z=(P[i] @ R.T + t)
                f.write(f'ATOM  {serial:5d}  CA  GLY {qch:1s}{serial%9999:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n'); serial+=1
        f.write('TER\n')
        for qch,tch,pairs in matched_pairs:
            Q=tchains[tch]['coords']
            for k,(i,j) in enumerate(pairs):
                x,y,z=Q[j]
                f.write(f'ATOM  {serial:5d}  CA  GLY {tch:1s}{serial%9999:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n'); serial+=1
        f.write('END\n')
    return outp

def main():
    qmeta,qchains=parse_pdb_ca(os.path.join(DATA,'7xg4.pdb'))
    tmeta,tchains=parse_pdb_ca(os.path.join(DATA,'6n40.pdb'))
    # overview
    overview={
        '7xg4': {'metadata':qmeta, 'chains':{ch:{'ca_count':len(d['coords']),'sequence_length':len(d['sequence'])} for ch,d in sorted(qchains.items())}},
        '6n40': {'metadata':tmeta, 'chains':{ch:{'ca_count':len(d['coords']),'sequence_length':len(d['sequence'])} for ch,d in sorted(tchains.items())}}
    }
    with open(os.path.join(OUT,'structure_overview.json'),'w') as f: json.dump(overview,f,indent=2)

    rows=[]; chain_results={}
    target_chains=sorted(tchains)
    for qch in sorted(qchains):
        for tch in target_chains:
            res=align_chain(qchains[qch]['coords'],tchains[tch]['coords'],qchains[qch]['sequence'],tchains[tch]['sequence'])
            chain_results[(qch,tch)]=res
            rows.append({'query_chain':qch,'target_chain':tch, **{k:v for k,v in res.items() if k not in ['rotation','translation','pairs','distances']}})
    df=pd.DataFrame(rows).sort_values(['tm_norm_max','aligned_len'], ascending=False)
    df.to_csv(os.path.join(OUT,'chain_pair_metrics.csv'), index=False)

    # chain assignment: target has one protein chain, so evaluate all single-chain assignments and best by TM.
    cand=df.copy(); cand['assignment']='7xg4:'+cand['query_chain']+' -> 6n40:'+cand['target_chain']
    cand[['assignment','query_chain','target_chain','tm_norm_max','tm_norm_query','tm_norm_target','aligned_len','rmsd','coverage_query','coverage_target']].to_csv(os.path.join(OUT,'assignment_candidates.csv'), index=False)
    bestrow=df.iloc[0]
    qch=str(bestrow.query_chain); tch=str(bestrow.target_chain)
    best=chain_results[(qch,tch)]

    # Complex-level final transform from best chain assignment. Because target complex has one protein chain, complex assignment is single chain.
    pairs=best['pairs']
    P=qchains[qch]['coords']; Q=tchains[tch]['coords']
    Pm=np.vstack([P[i] for i,j in pairs]); Qm=np.vstack([Q[j] for i,j in pairs])
    R,t=kabsch(Pm,Qm)
    dist=np.linalg.norm(Pm @ R.T + t - Qm, axis=1)
    result={
        'query':'7xg4','target':'6n40',
        'chain_correspondence': [{'query_chain':qch,'target_chain':tch,'aligned_residue_pairs':len(pairs), 'query_ca':len(P), 'target_ca':len(Q)}],
        'superimposition': {'definition':'transformed_query_coord = query_coord @ rotation.T + translation', 'rotation_3x3':R.tolist(), 'translation_A':t.tolist()},
        'metrics': {'aligned_len':len(pairs),'rmsd_A':float(np.sqrt(np.mean(dist**2))), 'mean_distance_A':float(np.mean(dist)), 'median_distance_A':float(np.median(dist)), 'tm_score_norm_max_chain_length':tm_score(dist,max(len(P),len(Q))), 'tm_score_norm_query_length':tm_score(dist,len(P)), 'tm_score_norm_target_length':tm_score(dist,len(Q)), 'd0_norm_max_A':tm_d0(max(len(P),len(Q)))},
        'method_limitations':['Foldseek, USalign, and TMalign executables were not available; results use a local Kabsch/dynamic-programming fallback rather than exact Foldseek-Multimer.', '6n40 contains one protein chain, so complex chain correspondence reduces to selecting the best 7xg4 protein chain for 6n40 chain A.', 'TM score is computed on CA residue pairs from the fallback alignment and should be interpreted as TM-score-like, not an official TM-align/Foldseek value.'],
        'matched_residue_pairs_csv':'outputs/matched_residue_pairs.csv',
        'superposed_matched_ca_pdb':'outputs/superposed_matched_ca.pdb'
    }
    with open(os.path.join(OUT,'alignment_result.json'),'w') as f: json.dump(result,f,indent=2)
    pair_rows=[]
    for rank,(i,j) in enumerate(pairs,1):
        pair_rows.append({'rank':rank,'query_chain':qch,'query_index0':i,'query_residue':qchains[qch]['residue_ids'][i], 'target_chain':tch,'target_index0':j,'target_residue':tchains[tch]['residue_ids'][j], 'distance_A':dist[rank-1]})
    pd.DataFrame(pair_rows).to_csv(os.path.join(OUT,'matched_residue_pairs.csv'), index=False)
    write_pdb_superposed(qmeta,qchains,tmeta,tchains,[(qch,tch)],R,t,[(qch,tch,pairs)])

    # figures
    sns.set_theme(style='whitegrid')
    # data overview
    fig,ax=plt.subplots(figsize=(9,4.8))
    ov=[]
    for sid,chains in [('7xg4',qchains),('6n40',tchains)]:
        for ch,d in sorted(chains.items()): ov.append({'structure':sid,'chain':ch,'CA residues':len(d['coords'])})
    odf=pd.DataFrame(ov)
    sns.barplot(data=odf, x='chain', y='CA residues', hue='structure', ax=ax)
    ax.set_title('Protein CA residue counts by chain')
    ax.set_xlabel('Chain'); ax.set_ylabel('CA residues')
    fig.tight_layout(); fig.savefig(os.path.join(IMG,'data_overview.png'), dpi=220); plt.close(fig)

    # heatmap chain pair TM
    mat=df.pivot(index='query_chain', columns='target_chain', values='tm_norm_max').sort_index()
    fig,ax=plt.subplots(figsize=(4.2,7.0))
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label':'TM-score-like (max-length norm)'})
    ax.set_title('All-vs-all chain structural similarity')
    fig.tight_layout(); fig.savefig(os.path.join(IMG,'chain_pair_heatmap.png'), dpi=220); plt.close(fig)

    # superposition distance scatter
    fig,ax=plt.subplots(figsize=(8,4.5))
    x=np.arange(1,len(dist)+1)
    ax.scatter(x, dist, s=10, alpha=0.7)
    ax.axhline(np.mean(dist), color='red', ls='--', label=f'mean={np.mean(dist):.2f} Å')
    ax.axhline(np.median(dist), color='orange', ls=':', label=f'median={np.median(dist):.2f} Å')
    ax.set_xlabel('Matched residue pair rank'); ax.set_ylabel('CA distance after superposition (Å)')
    ax.set_title(f'Best assignment {qch}→{tch}: residual distances')
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(IMG,'superposition_scatter.png'), dpi=220); plt.close(fig)

    # 3D trace superposition, downsample target for visibility
    Pt=P @ R.T + t
    fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(111, projection='3d')
    ax.plot(Pt[:,0],Pt[:,1],Pt[:,2], color='#1f77b4', lw=1.2, label=f'7xg4 chain {qch} transformed')
    ax.plot(Q[:,0],Q[:,1],Q[:,2], color='#ff7f0e', lw=1.2, label=f'6n40 chain {tch}')
    # matched subset faint lines every ~len/50
    step=max(1,len(pairs)//50)
    for (i,j) in pairs[::step]:
        ax.plot([Pt[i,0],Q[j,0]],[Pt[i,1],Q[j,1]],[Pt[i,2],Q[j,2]], color='gray', alpha=0.18, lw=0.5)
    ax.set_title('CA trace superposition for best chain correspondence')
    ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)'); ax.set_zlabel('Z (Å)')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(IMG,'alignment_3d.png'), dpi=220); plt.close(fig)

    validation={
        'verified_from_workspace': ['Parsed data/7xg4.pdb and data/6n40.pdb directly.', 'Protein CA counts and chain identities were computed from ATOM records.', 'All alignment metrics and figures were generated by code/complex_align.py.'],
        'dependency_status_file':'outputs/dependency_check.json',
        'related_work_extraction_status':'PDF parser tool and pdftotext unavailable; related-work notes are limited to task-specified Foldseek-Multimer context and local dependency checks.',
        'limitations': result['method_limitations']
    }
    with open(os.path.join(OUT,'validation_summary.json'),'w') as f: json.dump(validation,f,indent=2)
    claim_rows=[
        {'claim':'7xg4 contains multiple protein chains and nucleic-acid chains; 6n40 contains one protein chain in the provided PDB.', 'supporting_artifact':'outputs/structure_overview.json; report/images/data_overview.png'},
        {'claim':f'Best fallback chain correspondence is 7xg4 chain {qch} to 6n40 chain {tch}.', 'supporting_artifact':'outputs/alignment_result.json; outputs/assignment_candidates.csv; report/images/chain_pair_heatmap.png'},
        {'claim':'Reported rotation and translation define the query-to-target superposition.', 'supporting_artifact':'outputs/alignment_result.json; outputs/superposed_matched_ca.pdb'},
        {'claim':'Similarity is weak/moderate under the fallback TM-score-like metric, not an official Foldseek-Multimer score.', 'supporting_artifact':'outputs/alignment_result.json; outputs/dependency_check.json; report/images/superposition_scatter.png'}]
    pd.DataFrame(claim_rows).to_csv(os.path.join(OUT,'claim_recovery_table.csv'), index=False)
    # update artifact inventory statuses
    inv=json.load(open(os.path.join(OUT,'target_artifact_inventory.json')))
    for section in ['primary_outputs','required_figures']:
        for item in inv[section]:
            path=os.path.join(ROOT,item['artifact'])
            item['status']='satisfied' if os.path.exists(path) else 'unsatisfied'
    inv['report']['status']='planned'
    with open(os.path.join(OUT,'target_artifact_inventory.json'),'w') as f: json.dump(inv,f,indent=2)
    print(json.dumps({'best_assignment':f'{qch}->{tch}', 'tm_score_like':result['metrics']['tm_score_norm_max_chain_length'], 'rmsd_A':result['metrics']['rmsd_A'], 'aligned_len':len(pairs)}, indent=2))

if __name__=='__main__': main()
