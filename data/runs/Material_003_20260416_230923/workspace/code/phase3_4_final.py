#!/usr/bin/env python3
"""Phase 3 & 4: Streamlined VAE-inspired framework + Inverse Design"""
import pandas as pd, numpy as np, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import cdist
import json, os, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size':12,'axes.labelsize':14,'axes.titlesize':16,'savefig.dpi':150,'savefig.bbox':'tight'})
BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_003_20260416_230923'
IMG = os.path.join(BASE,'report','images')
OUT = os.path.join(BASE,'outputs')

df_vit = pd.read_csv(os.path.join(OUT,'vitrimer_calibrated_tg.csv'))
data = np.load(os.path.join(OUT,'vitrimer_features.npz'), allow_pickle=True)
X = data['X_combined'].astype(np.float32)
vi = data['valid_indices']
tg_cal = df_vit.loc[vi,'tg_calibrated'].values
tg_md = df_vit.loc[vi,'tg'].values
print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

# Encoder: PCA
t0=time.time()
sc = StandardScaler(); Xs = sc.fit_transform(X)
LD = 32
pca_enc = PCA(LD); z_all = pca_enc.fit_transform(Xs)
ve = pca_enc.explained_variance_ratio_.sum()
print(f"PCA encoder: {ve*100:.1f}% var, {time.time()-t0:.1f}s")

# Property predictor
t0=time.time()
mlp = MLPRegressor((64,32), max_iter=300, random_state=42, early_stopping=True, 
                   validation_fraction=0.1, learning_rate_init=0.002)
mlp.fit(z_all, tg_cal)
pred_all = mlp.predict(z_all)
r2v = r2_score(tg_cal, pred_all); maev = mean_absolute_error(tg_cal, pred_all)
print(f"MLP: R²={r2v:.3f}, MAE={maev:.1f}K, {time.time()-t0:.1f}s")

# Decoder quality
Xr = sc.inverse_transform(pca_enc.inverse_transform(z_all))
rmse_recon = np.sqrt(np.mean((X - Xr)**2))
print(f"Recon RMSE: {rmse_recon:.4f}")

# ---- Fig 5: Training ----
fig,axes=plt.subplots(1,3,figsize=(18,5))
if hasattr(mlp,'loss_curve_'):
    axes[0].plot(mlp.loss_curve_,'b-',lw=1.5)
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Loss')
axes[0].set_title('(a) Property Predictor Training')
axes[1].bar(range(LD), pca_enc.explained_variance_ratio_*100, color='steelblue')
axes[1].set_xlabel('Component'); axes[1].set_ylabel('Var Explained (%)')
axes[1].set_title(f'(b) PCA Encoder ({ve*100:.1f}% total)')
rps = np.mean((X-Xr)**2, axis=1)
axes[2].hist(rps, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Reconstruction MSE'); axes[2].set_ylabel('Count')
axes[2].set_title(f'(c) Reconstruction Quality')
plt.tight_layout(); plt.savefig(os.path.join(IMG,'fig5_vae_training.png')); plt.close()
print("Saved fig5")

# ---- Fig 6: Latent space ----
pca2 = PCA(2); z2d = pca2.fit_transform(z_all)
fig,axes=plt.subplots(1,3,figsize=(20,6))
sc1=axes[0].scatter(z2d[:,0],z2d[:,1],c=tg_cal,cmap='RdYlBu_r',alpha=0.3,s=5)
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2'); axes[0].set_title('(a) Latent (Cal. Tg)')
plt.colorbar(sc1,ax=axes[0],label='Tg(K)')
sc2=axes[1].scatter(z2d[:,0],z2d[:,1],c=tg_md,cmap='RdYlBu_r',alpha=0.3,s=5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2'); axes[1].set_title('(b) Latent (MD Tg)')
plt.colorbar(sc2,ax=axes[1],label='MD Tg(K)')
axes[2].scatter(tg_cal,pred_all,c='steelblue',alpha=0.3,s=5)
lm=[min(tg_cal.min(),pred_all.min())-10,max(tg_cal.max(),pred_all.max())+10]
axes[2].plot(lm,lm,'r--',lw=2)
axes[2].set_xlabel('True Tg(K)'); axes[2].set_ylabel('Pred Tg(K)')
axes[2].set_title(f'(c) Property Pred\nR²={r2v:.3f}, MAE={maev:.1f}K')
plt.tight_layout(); plt.savefig(os.path.join(IMG,'fig6_latent_space.png')); plt.close()
print("Saved fig6")

# ============================================================
# PHASE 4: Inverse Design
# ============================================================
print("\n--- Phase 4: Inverse Design ---")
targets = {'High Tg (>480 K)':(480,600), 'Medium-High (420-480 K)':(420,480), 'Medium (360-420 K)':(360,420)}
all_gen = []

for tn,(tl,th) in targets.items():
    mask = (tg_cal>=tl)&(tg_cal<=th)
    n_in = mask.sum()
    if n_in < 5: continue
    zt = z_all[mask]
    cent = zt.mean(0); cov = np.cov(zt.T)+np.eye(LD)*0.01
    gz = []
    for _ in range(150):
        i1,i2=np.random.choice(len(zt),2,replace=True)
        gz.append(np.random.uniform(0.2,0.8)*zt[i1]+(1-np.random.uniform(0.2,0.8))*zt[i2])
    for _ in range(150):
        gz.append(zt[np.random.choice(len(zt))]+np.random.randn(LD)*0.3)
    for _ in range(150):
        gz.append(np.random.multivariate_normal(cent, cov*0.5))
    gz = np.array(gz)
    tg_g = mlp.predict(gz)
    ir = (tg_g>=tl)&(tg_g<=th)
    print(f"  {tn}: {ir.sum()}/{len(gz)} in range")
    for i in range(len(gz)):
        all_gen.append({'target':tn,'tg_predicted':float(tg_g[i]),'in_range':bool(ir[i]),'z':gz[i]})

# Top candidates with nearest neighbors
top = []
for tn,(tl,th) in targets.items():
    cs = [c for c in all_gen if c['target']==tn and c['in_range']]
    cs.sort(key=lambda c: abs(c['tg_predicted']-(tl+th)/2))
    top.extend(cs[:20])

if top:
    tz = np.array([c['z'] for c in top])
    d = cdist(tz, z_all)
    for i,c in enumerate(top):
        nn = np.argmin(d[i]); ri = vi[nn]
        c['nearest_acid']=df_vit.loc[ri,'acid']; c['nearest_epoxide']=df_vit.loc[ri,'epoxide']
        c['nearest_tg_cal']=float(df_vit.loc[ri,'tg_calibrated'])
        c['nearest_tg_md']=float(df_vit.loc[ri,'tg']); c['nn_dist']=float(d[i,nn])
    cdf = pd.DataFrame([{k:v for k,v in c.items() if k!='z'} for c in top])
    cdf.to_csv(os.path.join(OUT,'top_candidates.csv'),index=False)
    print(f"Saved {len(top)} top candidates")

# Novel combinations
tg_p90 = np.percentile(tg_cal, 90)
hi = df_vit.loc[vi[tg_cal>=tg_p90]]
at = hi['acid'].value_counts().head(8).index.tolist()
et = hi['epoxide'].value_counts().head(8).index.tolist()
ex = set(zip(df_vit['acid'],df_vit['epoxide']))
novel = [{'acid':a,'epoxide':e} for a in at for e in et if (a,e) not in ex]
print(f"Novel combos: {len(novel)}")

if novel:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    FP=128
    def fp(s):
        m=Chem.MolFromSmiles(s)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=FP)) if m else np.zeros(FP)
    def desc(s):
        m=Chem.MolFromSmiles(s)
        if not m: return [0]*6
        return [Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.NumHDonors(m),
                Descriptors.NumHAcceptors(m),Descriptors.TPSA(m),Descriptors.NumRotatableBonds(m)]
    nf = np.array([np.concatenate([fp(c['acid']),fp(c['epoxide']),desc(c['acid']),desc(c['epoxide'])]) for c in novel])
    nz = pca_enc.transform(sc.transform(nf))
    tn = mlp.predict(nz)
    for i,c in enumerate(novel): c['tg_predicted']=float(tn[i])
    ndf = pd.DataFrame(novel).sort_values('tg_predicted',ascending=False)
    ndf.to_csv(os.path.join(OUT,'novel_vitrimer_candidates.csv'),index=False)
    print(f"Novel Tg: {tn.min():.1f}-{tn.max():.1f} K")

# ---- Fig 7: Inverse Design ----
fig,axes=plt.subplots(2,2,figsize=(16,14))
gt=[c['tg_predicted'] for c in all_gen]
axes[0,0].hist(tg_cal,bins=50,color='coral',edgecolor='black',alpha=0.5,label='Training',density=True)
axes[0,0].hist(gt,bins=50,color='steelblue',edgecolor='black',alpha=0.5,label='Generated',density=True)
for _,(tl,th) in targets.items(): axes[0,0].axvspan(tl,th,alpha=0.1,color='green')
axes[0,0].set_xlabel('Tg(K)'); axes[0,0].set_ylabel('Density'); axes[0,0].set_title('(a) Generated vs Training'); axes[0,0].legend()

gza = np.array([c['z'] for c in all_gen])
gz2d = pca2.transform(gza)
axes[0,1].scatter(z2d[:,0],z2d[:,1],c='lightgray',alpha=0.2,s=3)
sc3=axes[0,1].scatter(gz2d[:,0],gz2d[:,1],c=[c['tg_predicted'] for c in all_gen],cmap='RdYlBu_r',alpha=0.5,s=10)
axes[0,1].set_xlabel('PC1'); axes[0,1].set_ylabel('PC2'); axes[0,1].set_title('(b) Generated in Latent Space')
plt.colorbar(sc3,ax=axes[0,1],label='Tg(K)')

sr={}
for tn,(tl,th) in targets.items():
    cs=[c for c in all_gen if c['target']==tn]
    sr[tn]=sum(c['in_range'] for c in cs)/len(cs)*100 if cs else 0
bars=axes[1,0].bar(range(len(sr)),list(sr.values()),color=['#e74c3c','#f39c12','#2ecc71'])
axes[1,0].set_xticks(range(len(sr))); axes[1,0].set_xticklabels([k.split('(')[0].strip() for k in sr],rotation=15)
axes[1,0].set_ylabel('Success (%)'); axes[1,0].set_title('(c) Target Success')
for b,v in zip(bars,sr.values()): axes[1,0].text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{v:.1f}%',ha='center')

if novel:
    ns=min(15,len(ndf))
    axes[1,1].barh(range(ns),ndf['tg_predicted'].head(ns).values,color='teal',edgecolor='black')
    axes[1,1].set_xlabel('Pred Tg(K)'); axes[1,1].set_title('(d) Top Novel Candidates'); axes[1,1].invert_yaxis()
plt.tight_layout(); plt.savefig(os.path.join(IMG,'fig7_inverse_design.png')); plt.close()
print("Saved fig7")

# ---- Fig 8: Chemical Diversity ----
ad = X[:,-12:-6]; ed = X[:,-6:]
fig,axes=plt.subplots(1,3,figsize=(18,6))
axes[0].hist(ad[:,0],bins=40,alpha=0.6,color='coral',label='Acid MW')
axes[0].hist(ed[:,0],bins=40,alpha=0.6,color='steelblue',label='Epoxide MW')
axes[0].set_xlabel('MW'); axes[0].set_title('(a) MW Distribution'); axes[0].legend()
axes[1].scatter((ad[:,0]+ed[:,0])/2,tg_cal,c='steelblue',alpha=0.2,s=5)
axes[1].set_xlabel('Avg MW'); axes[1].set_ylabel('Cal. Tg(K)'); axes[1].set_title('(b) Tg vs MW')
axes[2].hist(ad[:,1],bins=40,alpha=0.6,color='coral',label='Acid LogP')
axes[2].hist(ed[:,1],bins=40,alpha=0.6,color='steelblue',label='Epoxide LogP')
axes[2].set_xlabel('LogP'); axes[2].set_title('(c) LogP Distribution'); axes[2].legend()
plt.tight_layout(); plt.savefig(os.path.join(IMG,'fig8_chemical_diversity.png')); plt.close()
print("Saved fig8")

# ---- Fig 9: Heatmap ----
tn2=12
ta2=hi['acid'].value_counts().head(tn2).index.tolist()
te2=hi['epoxide'].value_counts().head(tn2).index.tolist()
hm=np.full((len(ta2),len(te2)),np.nan)
for i,a in enumerate(ta2):
    for j,e in enumerate(te2):
        m=df_vit[(df_vit['acid']==a)&(df_vit['epoxide']==e)]
        if len(m)>0: hm[i,j]=m['tg_calibrated'].values[0]
fig,ax=plt.subplots(figsize=(14,10))
sns.heatmap(hm,ax=ax,cmap='RdYlBu_r',annot=True,fmt='.0f',
            xticklabels=[s[:25]+'...' if len(s)>25 else s for s in te2],
            yticklabels=[s[:25]+'...' if len(s)>25 else s for s in ta2],
            mask=np.isnan(hm),cbar_kws={'label':'Cal. Tg(K)'})
ax.set_xlabel('Epoxide'); ax.set_ylabel('Acid'); ax.set_title('Tg Heatmap: Top Acid-Epoxide Combos')
plt.xticks(rotation=45,ha='right',fontsize=7); plt.yticks(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(IMG,'fig9_tg_heatmap.png')); plt.close()
print("Saved fig9")

# ---- Fig 10: Validation ----
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].scatter(tg_md,tg_cal,c='steelblue',alpha=0.2,s=5)
axes[0].plot([300,570],[300,570],'r--',lw=2)
axes[0].set_xlabel('MD Tg(K)'); axes[0].set_ylabel('Cal. Tg(K)'); axes[0].set_title('(a) MD vs Calibrated')
axes[1].scatter(tg_cal,pred_all,c='coral',alpha=0.2,s=5)
lm2=[min(tg_cal.min(),pred_all.min())-10,max(tg_cal.max(),pred_all.max())+10]
axes[1].plot(lm2,lm2,'r--',lw=2)
axes[1].set_xlabel('Cal. Tg(K)'); axes[1].set_ylabel('VAE Pred Tg(K)')
axes[1].set_title(f'(b) Cal. vs VAE Pred (R²={r2v:.3f})')
plt.tight_layout(); plt.savefig(os.path.join(IMG,'fig10_validation.png')); plt.close()
print("Saved fig10")

# Save results
res = {'n_vit':int(len(df_vit)),'n_valid':int(len(vi)),'n_gen':len(all_gen),
       'n_novel':len(novel),'success_rates':sr,'ld':LD,'pca_var':float(ve),
       'prop_r2':float(r2v),'prop_mae':float(maev),'recon_rmse':float(rmse_recon),
       'tg_stats':{'mean':float(tg_cal.mean()),'std':float(tg_cal.std()),
                   'min':float(tg_cal.min()),'max':float(tg_cal.max())}}
with open(os.path.join(OUT,'generation_results.json'),'w') as f: json.dump(res,f,indent=2)
print("\nDone! Phase 3 & 4 complete!")
