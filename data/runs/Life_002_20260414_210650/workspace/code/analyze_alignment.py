from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','SEC':'U','PYL':'O','ASX':'B','GLX':'Z','UNK':'X'}


def parse_pdb(path):
    chains=defaultdict(list)
    atom_counts=defaultdict(int)
    residues_seen=set()
    with open(path) as f:
        for line in f:
            if not line.startswith(('ATOM','HETATM')):
                continue
            chain=line[21].strip() or '_'
            atom=line[12:16].strip()
            resname=line[17:20].strip()
            resseq=line[22:26].strip()
            icode=line[26].strip()
            x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
            atom_counts[chain]+=1
            if atom=='CA':
                key=(chain,resseq,icode)
                if key not in residues_seen:
                    residues_seen.add(key)
                    chains[chain].append({'aa':AA3.get(resname,'X'),'coord':np.array([x,y,z]),'resseq':int(resseq),'icode':icode,'resname':resname})
    return chains, atom_counts


def nw(seq1, seq2, match=2, mismatch=-1, gap=-2):
    n,m=len(seq1),len(seq2)
    dp=np.zeros((n+1,m+1),dtype=float)
    bt=np.zeros((n+1,m+1),dtype=np.int8)
    for i in range(1,n+1):
        dp[i,0]=dp[i-1,0]+gap; bt[i,0]=1
    for j in range(1,m+1):
        dp[0,j]=dp[0,j-1]+gap; bt[0,j]=2
    for i in range(1,n+1):
        for j in range(1,m+1):
            s=match if seq1[i-1]==seq2[j-1] else mismatch
            vals=(dp[i-1,j-1]+s, dp[i-1,j]+gap, dp[i,j-1]+gap)
            b=int(np.argmax(vals))
            bt[i,j]=b; dp[i,j]=vals[b]
    i,j=n,m
    ali=[]
    while i>0 or j>0:
        b=bt[i,j]
        if i>0 and j>0 and b==0:
            ali.append((i-1,j-1)); i-=1; j-=1
        elif i>0 and (j==0 or b==1):
            ali.append((i-1,None)); i-=1
        else:
            ali.append((None,j-1)); j-=1
    ali.reverse()
    return float(dp[n,m]), ali


def kabsch(P,Q):
    Pc=P.mean(axis=0); Qc=Q.mean(axis=0)
    P0=P-Pc; Q0=Q-Qc
    C=P0.T@Q0
    V,S,Wt=np.linalg.svd(C)
    d=np.sign(np.linalg.det(V@Wt))
    D=np.diag([1.0,1.0,d])
    R=V@D@Wt
    t=Qc-Pc@R
    return R,t,S.tolist()


def tm_d0(L):
    L=max(int(L), 16)
    return 1.24*((L-15)**(1/3)) - 1.8


def tm_score(dists, Lnorm):
    d0=max(tm_d0(Lnorm),0.5)
    vals=1.0/(1.0+(dists/d0)**2)
    return float(vals.sum()/Lnorm), float(d0)


def analyze():
    base=Path('.')
    outdir=base/'outputs'
    imgdir=base/'report'/'images'
    qchains,qatoms=parse_pdb(base/'data'/'7xg4.pdb')
    tchains,tatoms=parse_pdb(base/'data'/'6n40.pdb')
    target_chain='A'
    tres=tchains[target_chain]
    tseq=''.join(r['aa'] for r in tres)

    structure_rows=[]
    for name,chains,atoms,path in [('7xg4',qchains,qatoms,'data/7xg4.pdb'),('6n40',tchains,tatoms,'data/6n40.pdb')]:
        for c,res in sorted(chains.items()):
            coords=np.array([r['coord'] for r in res])
            centroid=coords.mean(axis=0)
            rg=np.sqrt(((coords-centroid)**2).sum(axis=1).mean())
            structure_rows.append({'structure':name,'path':path,'chain':c,'ca_residues':len(res),'atom_records':atoms[c],'centroid_x':float(centroid[0]),'centroid_y':float(centroid[1]),'centroid_z':float(centroid[2]),'radius_of_gyration_ca':float(rg)})
    overview=pd.DataFrame(structure_rows)
    overview.to_csv(outdir/'structure_overview.csv', index=False)

    chain_results=[]
    detailed={}
    for qc,qres in sorted(qchains.items()):
        qseq=''.join(r['aa'] for r in qres)
        score,ali=nw(qseq,tseq)
        pairs=[(i,j) for i,j in ali if i is not None and j is not None]
        P=np.array([qres[i]['coord'] for i,j in pairs])
        Q=np.array([tres[j]['coord'] for i,j in pairs])
        R,t,sv=kabsch(P,Q)
        Pfit=P@R+t
        d=np.sqrt(((Pfit-Q)**2).sum(axis=1))
        ident=sum(1 for i,j in pairs if qseq[i]==tseq[j])
        cov_q=len(pairs)/len(qres)
        cov_t=len(pairs)/len(tres)
        tm_q,d0q=tm_score(d, len(qres))
        tm_t,d0t=tm_score(d, len(tres))
        chain_results.append({
            'query_chain':qc,'target_chain':target_chain,'query_len':len(qres),'target_len':len(tres),
            'aligned_residues':len(pairs),'sequence_identity':ident/len(pairs),'coverage_query':cov_q,'coverage_target':cov_t,
            'nw_score':score,'rmsd':float(np.sqrt((d**2).mean())),'tm_score_query_norm':tm_q,'tm_score_target_norm':tm_t,'tm_score_avg':(tm_q+tm_t)/2,
            'translation_x':float(t[0]),'translation_y':float(t[1]),'translation_z':float(t[2])
        })
        detailed[qc]={
            'rotation_matrix':R.tolist(),'translation_vector':t.tolist(),'singular_values':sv,
            'distance_summary':{'min':float(d.min()),'median':float(np.median(d)),'mean':float(d.mean()),'max':float(d.max())},
            'tm_parameters':{'d0_query':d0q,'d0_target':d0t},
            'aligned_pairs':[{
                'query_index':int(i),'target_index':int(j),
                'query_resseq':int(qres[i]['resseq']),'target_resseq':int(tres[j]['resseq']),
                'query_resname':qres[i]['resname'],'target_resname':tres[j]['resname'],
                'query_aa':qseq[i],'target_aa':tseq[j],'distance_after_superposition':float(dist)
            } for (i,j),dist in zip(pairs,d)]
        }
    chain_df=pd.DataFrame(chain_results).sort_values(['tm_score_avg','sequence_identity'], ascending=False)
    chain_df.to_csv(outdir/'chain_correspondence_scores.csv', index=False)

    best_chain=str(chain_df.iloc[0]['query_chain'])
    qres=qchains[best_chain]
    qseq=''.join(r['aa'] for r in qres)
    _,ali=nw(qseq,tseq)
    pairs=[(i,j) for i,j in ali if i is not None and j is not None]
    P=np.array([qres[i]['coord'] for i,j in pairs])
    Q=np.array([tres[j]['coord'] for i,j in pairs])
    R,t,sv=kabsch(P,Q)
    Pfit=P@R+t
    d=np.sqrt(((Pfit-Q)**2).sum(axis=1))

    transform={
        'selected_mapping':{'query_structure':'7xg4','query_chain':best_chain,'target_structure':'6n40','target_chain':'A'},
        'rotation_matrix':R.tolist(),
        'translation_vector':t.tolist(),
        'aligned_residue_count':len(pairs),
        'rmsd':float(np.sqrt((d**2).mean())),
        'tm_score_query_norm':float(chain_df.iloc[0]['tm_score_query_norm']),
        'tm_score_target_norm':float(chain_df.iloc[0]['tm_score_target_norm']),
        'tm_score_avg':float(chain_df.iloc[0]['tm_score_avg'])
    }
    (outdir/'superposition_transform.json').write_text(json.dumps(transform, indent=2))
    (outdir/'selected_alignment_pairs.json').write_text(json.dumps(detailed[best_chain], indent=2))

    concat_q=[]
    q_concat_map=[]
    for qc,res in sorted(qchains.items()):
        for idx,r in enumerate(res):
            concat_q.append(r)
            q_concat_map.append((qc,idx))
    qseq_concat=''.join(r['aa'] for r in concat_q)
    score,ali=nw(qseq_concat,tseq)
    pairs=[(i,j) for i,j in ali if i is not None and j is not None]
    P=np.array([concat_q[i]['coord'] for i,j in pairs])
    Q=np.array([tres[j]['coord'] for i,j in pairs])
    R2,t2,sv2=kabsch(P,Q)
    Pfit=P@R2+t2
    d2=np.sqrt(((Pfit-Q)**2).sum(axis=1))
    tmq2,d0q2=tm_score(d2,len(concat_q))
    tmt2,d0t2=tm_score(d2,len(tres))
    concat_result={
        'query_structure':'7xg4_all_chains_concatenated','target_structure':'6n40_A','query_len':len(concat_q),'target_len':len(tres),
        'aligned_residues':len(pairs),'sequence_identity':float(sum(1 for i,j in pairs if qseq_concat[i]==tseq[j])/len(pairs)),
        'coverage_query':len(pairs)/len(concat_q),'coverage_target':len(pairs)/len(tres),
        'rmsd':float(np.sqrt((d2**2).mean())),'tm_score_query_norm':tmq2,'tm_score_target_norm':tmt2,'tm_score_avg':(tmq2+tmt2)/2,
        'rotation_matrix':R2.tolist(),'translation_vector':t2.tolist()
    }
    (outdir/'complex_concatenated_alignment.json').write_text(json.dumps(concat_result, indent=2))

    summary={'best_chain_mapping':transform,'concatenated_complex_alignment':concat_result}
    (outdir/'tm_score_results.json').write_text(json.dumps(summary, indent=2))

    # Figure 1: chain comparison bars
    sns.set_theme(style='whitegrid')
    fig,ax=plt.subplots(figsize=(10,5))
    plot_df=chain_df.copy()
    x=np.arange(len(plot_df))
    ax.bar(x-0.18, plot_df['tm_score_avg'], width=0.36, label='Avg TM-score')
    ax.bar(x+0.18, plot_df['sequence_identity'], width=0.36, label='Sequence identity')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['query_chain'])
    ax.set_ylim(0,1)
    ax.set_xlabel('7xg4 query chain mapped to 6n40:A')
    ax.set_ylabel('Score')
    ax.set_title('Chain-wise correspondence scores against 6n40 chain A')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(imgdir/'chain_correspondence_scores.png', dpi=200)
    plt.close(fig)

    # Figure 2: distance profile for best mapping
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(np.arange(1,len(d)+1), d, color='#3366cc', lw=1.5)
    ax.axhline(np.median(d), color='tomato', ls='--', label=f'Median = {np.median(d):.2f} Å')
    ax.axhline(np.mean(d), color='green', ls=':', label=f'Mean = {np.mean(d):.2f} Å')
    ax.set_xlabel('Aligned residue index')
    ax.set_ylabel('Cα distance after superposition (Å)')
    ax.set_title(f'Residue-wise structural deviation for best mapping 7xg4:{best_chain} vs 6n40:A')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(imgdir/'best_alignment_distance_profile.png', dpi=200)
    plt.close(fig)

    # Figure 3: centroid scatter projection
    rows=[]
    for qc,res in sorted(qchains.items()):
        coords=np.array([r['coord'] for r in res])
        centroid=coords.mean(axis=0)
        rows.append({'structure':'7xg4','chain':qc,'x':centroid[0],'y':centroid[1],'z':centroid[2],'ca_residues':len(res)})
    coords=np.array([r['coord'] for r in tres])
    centroid=coords.mean(axis=0)
    rows.append({'structure':'6n40','chain':'A','x':centroid[0],'y':centroid[1],'z':centroid[2],'ca_residues':len(tres)})
    cdf=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(6,6))
    for struct,sub in cdf.groupby('structure'):
        ax.scatter(sub['x'], sub['y'], s=sub['ca_residues']*0.8, alpha=0.8, label=struct)
        for _,r in sub.iterrows():
            ax.text(r['x'], r['y'], f"{r['structure']}:{r['chain']}", fontsize=8)
    ax.set_xlabel('Centroid X (Å)')
    ax.set_ylabel('Centroid Y (Å)')
    ax.set_title('Chain centroids in Cartesian projection')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(imgdir/'structure_centroid_projection.png', dpi=200)
    plt.close(fig)

    print(json.dumps({'best_chain':best_chain,'best_tm_avg':float(chain_df.iloc[0]['tm_score_avg']),'concat_tm_avg':concat_result['tm_score_avg']}, indent=2))

if __name__=='__main__':
    analyze()
