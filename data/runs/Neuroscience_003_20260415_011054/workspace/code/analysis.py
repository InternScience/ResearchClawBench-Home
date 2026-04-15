#!/usr/bin/env python3
"""
Feature selection for preserving continuous cellular trajectories
in single-cell protein imaging (iIF) data from RPE cells.
Optimized greedy diverse selection.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')
import json
import os

np.random.seed(42)
sc.settings.verbosity = 1
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("STEP 1: Loading data")
adata = sc.read_h5ad('data/adata_RPE.h5ad')
print(f"Shape: {adata.shape}")
adata.X = adata.layers['raw'].copy()
feature_names = list(adata.var_names)
n_features = len(feature_names)

print("STEP 2: Trajectory")
adata_proc = adata.copy()
sc.pp.scale(adata_proc, max_value=10)
sc.pp.neighbors(adata_proc, n_neighbors=30, method='umap', random_state=42)
sc.tl.diffmap(adata_proc, n_comps=15)
g0_idx = np.where(adata_proc.obs['phase'] == 'G0')[0]
dc = adata_proc.obsm['X_diffmap'][g0_idx, :3]
adata_proc.uns['iroot'] = g0_idx[np.argmin(np.linalg.norm(dc - dc.mean(0), axis=1))]
sc.tl.dpt(adata_proc, n_dcs=10)
adata.obs['dpt'] = adata_proc.obs['dpt_pseudotime']
adata.obsm['X_diffmap'] = adata_proc.obsm['X_diffmap']
pt = adata.obs['dpt'].values.copy()
pt[np.isinf(pt)] = np.nan
pt[np.isnan(pt)] = np.nanmax(pt[~np.isnan(pt)]) + 0.1
pseudotime = pt
print(f"Pseudotime: {pt.min():.3f} - {pt.max():.3f}")

print("STEP 3: Scoring features")
X_raw = adata.X
diffmap = adata.obsm['X_diffmap'][:, :10]

print("  Precomputing feature-feature correlations...")
corr_matrix = np.abs(np.corrcoef(X_raw.T))

spearman_abs = np.zeros(n_features)
for i in range(n_features):
    r, _ = spearmanr(X_raw[:, i], pseudotime)
    spearman_abs[i] = abs(r)

var_exp = np.zeros(n_features)
for i in range(n_features):
    reg = LinearRegression().fit(diffmap, X_raw[:, i])
    pred = reg.predict(diffmap)
    ss_res = np.sum((X_raw[:, i] - pred)**2)
    ss_tot = np.sum((X_raw[:, i] - X_raw[:, i].mean())**2)
    var_exp[i] = max(0, 1 - ss_res/ss_tot) if ss_tot > 0 else 0

n_bins = 20
pt_bin = pd.qcut(pseudotime, q=n_bins, labels=False, duplicates='drop')
mi = np.zeros(n_features)
for i in range(n_features):
    fb = pd.qcut(X_raw[:, i], q=n_bins, labels=False, duplicates='drop')
    mi[i] = mutual_info_score(pt_bin, fb)

def norm(x):
    lo, hi = x.min(), x.max()
    return (x-lo)/(hi-lo) if hi > lo else np.zeros_like(x)

composite = 0.40*norm(spearman_abs) + 0.35*norm(var_exp) + 0.25*norm(mi)

print("STEP 4: Greedy diverse selection")

def knn_pres(X_ref, X_sub, k=30):
    nn_r = NearestNeighbors(n_neighbors=k+1).fit(X_ref)
    nn_s = NearestNeighbors(n_neighbors=k+1).fit(X_sub)
    _, ir = nn_r.kneighbors(X_ref)
    _, is_ = nn_s.kneighbors(X_sub)
    return np.mean([len(set(ir[i,1:]) & set(is_[i,1:]))/k for i in range(X_ref.shape[0])])

X_scaled = adata_proc.X
target_sizes = [10, 20, 30, 50, 80, 120]

selected = []
remaining = set(range(n_features))
best0 = np.argmax(composite)
selected.append(best0)
remaining.remove(best0)

for step in range(1, max(target_sizes)):
    best_score, best_feat = -1, -1
    for f in remaining:
        max_corr = max(corr_matrix[f, s] for s in selected)
        score = composite[f] * (1 - 0.7 * max_corr**2)
        if score > best_score:
            best_score = score
            best_feat = f
    selected.append(best_feat)
    remaining.remove(best_feat)

print(f"\n{'Size':<8} {'Greedy':<12} {'Variance':<12} {'Random':<12}")
print("-"*44)
pg_list, pv_list, pr_list = [], [], []
var_rank = np.argsort(np.var(X_raw, axis=0))[::-1]
for sz in target_sizes:
    pg = knn_pres(X_scaled, X_scaled[:, selected[:sz]])
    pg_list.append(pg)
    pv = knn_pres(X_scaled, X_scaled[:, var_rank[:sz]])
    pv_list.append(pv)
    prs = [knn_pres(X_scaled, X_scaled[:, np.random.choice(n_features, sz, replace=False)]) for _ in range(3)]
    pr_list.append(np.mean(prs))
    print(f"{sz:<8} {pg:.4f}       {pv:.4f}       {np.mean(prs):.4f}")

target_size = 30
for i, sz in enumerate(target_sizes):
    if pg_list[i] >= 0.7:
        target_size = sz
        break

sel_idx = selected[:target_size]
sel_feats = [feature_names[i] for i in sel_idx]
print(f"\nSelected {target_size} features")
for r, idx in enumerate(sel_idx):
    print(f"  {r+1}. {feature_names[idx]} (comp={composite[idx]:.3f}, spr={spearman_abs[idx]:.3f})")

print("STEP 5: Validation")
adata_sel = sc.AnnData(X=X_raw[:, sel_idx].copy())
adata_sel.obs = adata.obs.copy()
sc.pp.scale(adata_sel, max_value=10)
sc.pp.neighbors(adata_sel, n_neighbors=30, method='umap', random_state=42)
sc.tl.diffmap(adata_sel, n_comps=15)
sg0 = np.where(adata_sel.obs['phase']=='G0')[0]
dcs = adata_sel.obsm['X_diffmap'][sg0,:3]
adata_sel.uns['iroot'] = sg0[np.argmin(np.linalg.norm(dcs - dcs.mean(0), axis=1))]
sc.tl.dpt(adata_sel, n_dcs=10)
pt_sel = adata_sel.obs['dpt_pseudotime'].values.copy()
pt_sel[np.isinf(pt_sel)] = np.nan
pt_sel = np.nan_to_num(pt_sel, nan=np.nanmax(pt_sel[~np.isnan(pt_sel)]))
adata.obs['dpt_sel'] = pt_sel
valid = np.isfinite(pseudotime) & np.isfinite(pt_sel)
corr_pt, pval_pt = spearmanr(pseudotime[valid], pt_sel[valid])
si = target_sizes.index(target_size)
print(f"Pseudotime corr: r={corr_pt:.4f}")
print(f"KNN greedy={pg_list[si]:.4f}, var={pv_list[si]:.4f}, rand={pr_list[si]:.4f}")

print("STEP 6: Figures")
dc_full = adata.obsm['X_diffmap']

fig, axes = plt.subplots(1,3,figsize=(18,5))
for s,c in {'cycling':'#e74c3c','arrested':'#3498db','nan':'#95a5a6'}.items():
    m = adata.obs['state']==s; axes[0].scatter(dc_full[m,0],dc_full[m,1],c=c,s=5,alpha=0.5,label=s)
axes[0].set_xlabel('DC1'); axes[0].set_ylabel('DC2'); axes[0].set_title('Cell State'); axes[0].legend(markerscale=3)
for p,c in {'G0':'#2ecc71','G1':'#3498db','S':'#e74c3c','G2':'#f39c12'}.items():
    m = adata.obs['phase']==p; axes[1].scatter(dc_full[m,0],dc_full[m,1],c=c,s=5,alpha=0.5,label=p)
axes[1].set_xlabel('DC1'); axes[1].set_ylabel('DC2'); axes[1].set_title('Phase'); axes[1].legend(markerscale=3)
sc2=axes[2].scatter(dc_full[:,0],dc_full[:,1],c=pseudotime,cmap='viridis',s=5,alpha=0.5)
plt.colorbar(sc2,ax=axes[2],label='Pseudotime'); axes[2].set_xlabel('DC1'); axes[2].set_ylabel('DC2'); axes[2].set_title('Pseudotime')
plt.tight_layout(); plt.savefig('report/images/fig1_data_overview.png',dpi=150,bbox_inches='tight'); plt.close()
print("  fig1")

fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].hist(spearman_abs,bins=50,color='#3498db',edgecolor='white'); axes[0].set_xlabel('|Spearman|'); axes[0].set_title('Pseudotime Correlations')
axes[1].hist(var_exp,bins=50,color='#e74c3c',edgecolor='white'); axes[1].set_xlabel('R2'); axes[1].set_title('Variance Explained by DCs')
axes[2].hist(composite,bins=50,color='#2ecc71',edgecolor='white'); axes[2].set_xlabel('Score'); axes[2].set_title('Composite Score')
plt.tight_layout(); plt.savefig('report/images/fig2_feature_scoring.png',dpi=150,bbox_inches='tight'); plt.close()
print("  fig2")

fig,ax=plt.subplots(figsize=(10,6))
ax.plot(target_sizes,pg_list,'o-',color='#e74c3c',lw=2,ms=8,label='Greedy Diverse')
ax.plot(target_sizes,pv_list,'s-',color='#3498db',lw=2,ms=8,label='Variance')
ax.plot(target_sizes,pr_list,'^-',color='#95a5a6',lw=2,ms=8,label='Random')
ax.axvline(x=target_size,color='#e74c3c',ls=':',alpha=0.5)
ax.set_xlabel('Number of Features'); ax.set_ylabel('KNN Preservation'); ax.set_title('Trajectory Preservation vs. Subset Size')
ax.legend(); ax.set_ylim(0,1.05); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig3_preservation_comparison.png',dpi=150,bbox_inches='tight'); plt.close()
print("  fig3")

tn=min(target_size,30)
si2=np.argsort(pseudotime)
Xs=X_raw[si2][:,:tn]
Xsm=uniform_filter1d(Xs,size=50,axis=0)
fig,ax=plt.subplots(figsize=(14,8))
im=ax.imshow(Xsm.T,aspect='auto',cmap='RdBu_r',extent=[0,1,0,tn])
ax.set_xlabel('Pseudotime'); ax.set_ylabel('Feature'); ax.set_title(f'Top {tn} Features Along Pseudotime')
ax.set_yticks(range(tn)); ax.set_yticklabels([feature_names[sel_idx[i]][:30] for i in range(tn)],fontsize=7)
plt.colorbar(im,ax=ax,label='Expression'); plt.tight_layout(); plt.savefig('report/images/fig4_heatmap_pseudotime.png',dpi=150,bbox_inches='tight'); plt.close()
print("  fig4")

fig,axes=plt.subplots(1,2,figsize=(14,6))
dc_s=adata_sel.obsm['X_diffmap']
s1=axes[0].scatter(dc_full[:,0],dc_full[:,1],c=pseudotime,cmap='viridis',s=5,alpha=0.5)
axes[0].set_title(f'Full ({n_features})'); axes[0].set_xlabel('DC1'); axes[0].set_ylabel('DC2'); plt.colorbar(s1,ax=axes[0],label='Pseudotime')
s2=axes[1].scatter(dc_s[:,0],dc_s[:,1],c=pt_sel,cmap='viridis',s=5,alpha=0.5)
axes[1].set_title(f'Selected ({target_size})'); axes[1].set_xlabel('DC1'); axes[1].set_ylabel('DC2'); plt.colorbar(s2,ax=axes[1],label='Pseudotime')
plt.suptitle('Trajectory: Full vs Selected',fontsize=14,y=1.02); plt.tight_layout(); plt.savefig('report/images/fig5_trajectory_comparison.png',dpi=150,bbox_inches='tight'); plt.close()
print("  fig5")

cats={'cell':[],'cyto':[],'nuc':[],'ring':[],'edge':[]}
ss=set(sel_idx)
for i,n in enumerate(feature_names):
    for c in cats:
        if f'_{c}' in n: cats[c].append(i); break
cn=list(cats.keys())
ct=[len(cats[c]) for c in cn]
cs=[len(set(cats[c])&ss) for c in cn]
cp=[s/t*100 if t>0 else 0 for s,t in zip(cs,ct)]
fig,axes=plt.subplots(1,2,figsize=(14,5))
x=np.arange(len(cn));w=0.35
axes[0].bar(x-w/2,ct,w,label='Total',color='#3498db'); axes[0].bar(x+w/2,cs,w,label='Selected',color='#e74c3c')
axes[0].set_xticks(x); axes[0].set_xticklabels(cn); axes[0].set_title('By Compartment'); axes[0].legend()
axes[1].bar(cn,cp,color='#2ecc71'); axes[1].set_ylabel('% Selected'); axes[1].set_title('Selection Rate'); axes[1].set_ylim(0,100)
plt.tight_layout(); plt.savefig('report/images/fig6_category_breakdown.png',dpi=150,bbox_inches='tight'); plt.close()
print("  fig6")

fig,ax=plt.subplots(figsize=(7,7))
ax.scatter(pseudotime[valid],pt_sel[valid],s=3,alpha=0.3,c='#3498db')
mx=max(pseudotime.max(),pt_sel.max()); ax.plot([0,mx],[0,mx],'r--',lw=1)
ax.set_xlabel('Pseudotime (Full)'); ax.set_ylabel(f'Pseudotime ({target_size})'); ax.set_title(f'Agreement (r={corr_pt:.3f})')
plt.tight_layout(); plt.savefig('report/images/fig7_pseudotime_scatter.png',dpi=150,bbox_inches='tight'); plt.close()
print("  fig7")

print("STEP 7: Saving")
pd.DataFrame({'rank':range(1,target_size+1),'feature':sel_feats,
    'composite':[float(composite[i]) for i in sel_idx],
    'spearman':[float(spearman_abs[i]) for i in sel_idx],
    'var_explained':[float(var_exp[i]) for i in sel_idx],
    'mi':[float(mi[i]) for i in sel_idx]}).to_csv('outputs/selected_features.csv',index=False)

pd.DataFrame({'feature':feature_names,'composite':composite,'spearman':spearman_abs,
    'var_explained':var_exp,'mi':mi}).to_csv('outputs/all_feature_scores.csv',index=False)

pd.DataFrame({'size':target_sizes,'greedy':pg_list,'variance':pv_list,'random':pr_list}).to_csv('outputs/preservation_metrics.csv',index=False)

json.dump({'n_cells':int(adata.shape[0]),'n_features_total':n_features,'n_features_selected':target_size,
    'selected_features':sel_feats,'pseudotime_corr':float(corr_pt),
    'knn_greedy':float(pg_list[si]),'knn_variance':float(pv_list[si]),'knn_random':float(pr_list[si]),
    'states':adata.obs['state'].value_counts().to_dict(),'phases':adata.obs['phase'].value_counts().to_dict()
},open('outputs/summary.json','w'),indent=2)

print("DONE")
