"""Final analysis: LES charge recovery, dimer binding, Ag3 charge states."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import Ridge, LinearRegression
import os, json, warnings
warnings.filterwarnings('ignore')

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
plt.rcParams.update({'font.size': 11})

BOX = 15.0

def ewald_coulomb(pos, ch, alpha=0.3, kmax=4):
    n=len(ch); er=0.0
    for i in range(n):
        for j in range(i+1,n):
            rv=pos[i]-pos[j]; rv-=BOX*np.round(rv/BOX); r=np.linalg.norm(rv)
            if r<1e-10: continue
            er+=ch[i]*ch[j]*erfc(alpha*r)/r
    er-=alpha/np.sqrt(np.pi)*np.sum(ch**2)
    V=BOX**3; ek=0.0
    for nx in range(-kmax,kmax+1):
        for ny in range(-kmax,kmax+1):
            for nz in range(-kmax,kmax+1):
                if nx==0 and ny==0 and nz==0: continue
                k=2*np.pi*np.array([nx,ny,nz])/BOX; k2=np.dot(k,k)
                f=2*np.pi/V*np.exp(-k2/(4*alpha**2))/k2
                Sr=np.sum(ch*np.cos(np.dot(pos,k))); Si=np.sum(ch*np.sin(np.dot(pos,k)))
                ek+=f*(Sr**2+Si**2)
    return er+ek

def direct_coulomb(pos,ch):
    n=len(ch); ec=0.0
    for i in range(n):
        for j in range(i+1,n):
            rv=pos[i]-pos[j]; rv-=BOX*np.round(rv/BOX); r=np.linalg.norm(rv)
            if r<1e-10: continue; ec+=ch[i]*ch[j]/r
    return ec

def lj_energy(pos,eps=0.5,sigma=1.0):
    n=len(pos); elj=0.0
    for i in range(n):
        for j in range(i+1,n):
            rv=pos[i]-pos[j]; rv-=BOX*np.round(rv/BOX); r=np.linalg.norm(rv)
            if r<1e-10: continue
            sr2=(sigma/r)**2; sr6=sr2**3; sr12=sr6**2; elj+=4*eps*(sr12-sr6)
    return elj

def compute_desc(pos,cutoff=6.0):
    n=len(pos); nb=12; desc=np.zeros((n,nb+4))
    be=np.linspace(0.5,cutoff,nb+1)
    for i in range(n):
        pi=pos[i]; desc[i,-4]=pi[0]/BOX; desc[i,-3]=pi[1]/BOX; desc[i,-2]=pi[2]/BOX; desc[i,-1]=1.0
        for j in range(n):
            if i==j: continue
            rv=pos[j]-pi; rv-=BOX*np.round(rv/BOX); r=np.linalg.norm(rv)
            if r<cutoff:
                for b in range(nb):
                    if be[b]<=r<be[b+1]: desc[i,b]+=1; break
    rs=desc[:,:nb].sum(axis=1,keepdims=True); rs[rs==0]=1; desc[:,:nb]/=rs
    return desc

# ============================================================
# LOAD DATA
# ============================================================
data_rc=np.load('outputs/random_charges_parsed.npz',allow_pickle=True)
pos_rc=data_rc['positions']; tc_rc=data_rc['true_charges']
data_cd=np.load('outputs/charged_dimer_parsed.npz',allow_pickle=True)
pos_cd=data_cd['positions']; e_cd=data_cd['energies']
data_ag=np.load('outputs/ag3_chargestates_parsed.npz',allow_pickle=True)
pos_ag=data_ag['positions']; e_ag=data_ag['energies']; cs_ag=data_ag['charge_states']; f_ag=data_ag['forces']

# ============================================================
# 1. RANDOM CHARGES - LES
# ============================================================
print("="*60)
print("1. RANDOM CHARGES")
print("="*60)

all_desc,all_ec,all_elj,valid=[],[],[],[]
for fi in range(len(pos_rc)):
    pos=pos_rc[fi]; ch=tc_rc[fi]
    md=np.inf
    for i in range(len(pos)):
        for j in range(i+1,len(pos)):
            rv=pos[i]-pos[j]; rv-=BOX*np.round(rv/BOX); r=np.linalg.norm(rv)
            if r<md: md=r
    if md<0.5: continue
    ec=direct_coulomb(pos,ch); elj=lj_energy(pos)
    if abs(ec)>1e6: continue
    all_desc.append(compute_desc(pos)); all_ec.append(ec); all_elj.append(elj); valid.append(fi)

all_ec=np.array(all_ec); all_elj=np.array(all_elj); all_e=all_ec+all_elj
print(f"Valid: {len(valid)}/{len(pos_rc)}")
np.random.seed(42); perm=np.random.permutation(len(valid))
ntr=min(50,int(0.7*len(valid))); tri=perm[:ntr]; tei=perm[ntr:]

# Train q_les
Xq_tr=np.vstack([all_desc[i] for i in tri]); yq_tr=np.hstack([tc_rc[valid[i]] for i in tri])
q_model=Ridge(alpha=0.1).fit(Xq_tr,yq_tr)
Xq_te=np.vstack([all_desc[i] for i in tei]); yq_te=np.hstack([tc_rc[valid[i]] for i in tei])
q_pred=q_model.predict(Xq_te)

corrs=[]; off=0
for fi in range(len(tei)):
    n=len(tc_rc[valid[tei[fi]]]); c=np.corrcoef(yq_te[off:off+n],q_pred[off:off+n])[0,1]
    corrs.append(c); off+=n
print(f"Charge corr: {np.mean(corrs):.4f}±{np.std(corrs):.4f}")

ec_les=[]; off=0
for fi in range(len(tei)):
    vi=valid[tei[fi]]; pos=pos_rc[vi]; n=len(pos); ql=q_pred[off:off+n]; off+=n
    ec_les.append(ewald_coulomb(pos,ql,kmax=3))
ec_les=np.array(ec_les); ec_true=all_ec[tei]
ec_rmse=np.sqrt(np.mean((ec_les-ec_true)**2))
e_les=all_elj[tei]+ec_les; e_rmse=np.sqrt(np.mean((e_les-all_e[tei])**2))
print(f"EC RMSE: {ec_rmse:.3f}, E RMSE: {e_rmse:.3f}")

gdesc=np.array([np.mean(d,axis=0) for d in all_desc])
sr_model=Ridge(alpha=0.1).fit(gdesc[tri],all_e[tri])
e_sr_pred=sr_model.predict(gdesc[tei]); sr_rmse=np.sqrt(np.mean((e_sr_pred-all_e[tei])**2))
print(f"SR-only RMSE: {sr_rmse:.3f}")

# Optimize q_les
demo_vi=valid[tei[0]]; demo_pos=pos_rc[demo_vi]; demo_tc=tc_rc[demo_vi]; demo_ec=all_ec[tei[0]]
def q_loss(qf,pos,target):
    q=qf.reshape(-1); ep=ewald_coulomb(pos,q,kmax=3); return (ep-target)**2
q_init=np.random.randn(len(demo_pos))*0.1; q_init-=q_init.mean()
q_hist=[np.corrcoef(demo_tc,q_init)[0,1]]
res=minimize(q_loss,q_init,args=(demo_pos,demo_ec),method='L-BFGS-B',
             options={'maxiter':500},callback=lambda qk: q_hist.append(np.corrcoef(demo_tc,qk)[0,1]))
q_opt=res.x; q_corr_opt=np.corrcoef(demo_tc,q_opt)[0,1]
print(f"Opt q corr: {q_corr_opt:.4f}, EC err: {abs(ewald_coulomb(demo_pos,q_opt,kmax=3)-demo_ec):.6f}")

# FIGURE 1
fig,axes=plt.subplots(2,3,figsize=(18,11))
ax=axes[0,0]; ax.scatter(demo_tc,q_opt,alpha=0.6,s=40,c='steelblue',edgecolors='k',linewidth=0.5)
ax.plot([-1.5,1.5],[-1.5,1.5],'r--',linewidth=1.5,alpha=0.7)
ax.set_xlabel('True Charge (e)'); ax.set_ylabel('Latent Charge $q_{LES}$')
ax.set_title(f'LES Charge Recovery (r={q_corr_opt:.3f})'); ax.set_xlim(-1.5,1.5); ax.grid(True,alpha=0.3)

ax=axes[0,1]; ax.scatter(ec_true,ec_les,alpha=0.6,s=30,c='darkorange',edgecolors='k',linewidth=0.5)
lo,hi=min(ec_true.min(),ec_les.min()),max(ec_true.max(),ec_les.max())
ax.plot([lo,hi],[lo,hi],'r--',linewidth=1.5,alpha=0.7)
ax.set_xlabel('$E_{coul}$ (true charges)'); ax.set_ylabel('$E_{coul}$ ($q_{LES}$)')
ax.set_title(f'Coulomb Energy (RMSE={ec_rmse:.1f})'); ax.grid(True,alpha=0.3)

ax=axes[0,2]; ax.scatter(all_e[tei],e_les,alpha=0.6,s=30,c='green',edgecolors='k',linewidth=0.5)
lo2,hi2=min(all_e[tei].min(),e_les.min()),max(all_e[tei].max(),e_les.max())
ax.plot([lo2,hi2],[lo2,hi2],'r--',linewidth=1.5,alpha=0.7)
ax.set_xlabel('True Total Energy'); ax.set_ylabel('LES Total Energy')
ax.set_title(f'Total Energy (RMSE={e_rmse:.1f})'); ax.grid(True,alpha=0.3)

ax=axes[1,0]; methods=['LES','SR-only']; vals=[e_rmse,sr_rmse]; colors=['#2196F3','#FF9800']
bars=ax.bar(methods,vals,color=colors,edgecolor='k',linewidth=1.5,width=0.5)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*0.02,f'{v:.1f}',ha='center',va='bottom',fontsize=12)
ax.set_ylabel('Total Energy RMSE'); ax.set_title('LES vs Short-Range'); ax.grid(True,alpha=0.3,axis='y')

ax=axes[1,1]; ax.plot(q_hist,'b-',linewidth=1.5)
ax.set_xlabel('Iteration'); ax.set_ylabel('r($q_{LES}$, true)')
ax.set_title('Charge Recovery During Optimization'); ax.axhline(y=0,color='gray',linestyle=':',alpha=0.5); ax.grid(True,alpha=0.3)

ax=axes[1,2]; f0=0; vi0=valid[tei[f0]]; n0=len(tc_rc[vi0])
ax.scatter(range(n0),yq_te[:n0],s=15,alpha=0.6,c='blue',label='True',marker='o')
ax.scatter(range(n0),q_pred[:n0],s=15,alpha=0.6,c='red',label='$q_{LES}$',marker='x')
ax.set_xlabel('Atom Index'); ax.set_ylabel('Charge (e)')
ax.set_title('Per-Atom Charge Comparison'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

plt.tight_layout(); plt.savefig('report/images/fig1_charge_recovery.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved fig1")

json.dump({'mean_corr':float(np.mean(corrs)),'std_corr':float(np.std(corrs)),
    'ec_rmse':float(ec_rmse),'e_rmse':float(e_rmse),'sr_rmse':float(sr_rmse),
    'opt_q_corr':float(q_corr_opt)},open('outputs/charge_recovery_results.json','w'),indent=2)

# ============================================================
# 2. CHARGED DIMER
# ============================================================
print("\n"+"="*60); print("2. CHARGED DIMER"); print("="*60)

cc_dist=np.array([np.linalg.norm(pos_cd[fi][0]-pos_cd[fi][4]) for fi in range(len(pos_cd))])
si=np.argsort(cc_dist); cc_s=cc_dist[si]; e_s=e_cd[si]
e_lr=-1.0/cc_s; e_sr=e_s-e_lr

def sr_fn(r,a,b,c): return a*np.exp(-b*r)+c
try:
    popt,_=curve_fit(sr_fn,cc_s,e_sr,p0=[1,1,0],maxfev=5000)
    e_sr_fit=sr_fn(cc_s,*popt); e_les_fit=e_sr_fit+e_lr
    rmse_les=np.sqrt(np.mean((e_les_fit-e_s)**2))
except:
    from numpy.polynomial import polynomial as P
    c=P.polyfit(cc_s,e_sr,3); e_sr_fit=P.polyval(cc_s,c); e_les_fit=e_sr_fit+e_lr
    rmse_les=np.sqrt(np.mean((e_les_fit-e_s)**2))
try:
    popt2,_=curve_fit(sr_fn,cc_s,e_s,p0=[1,0.5,np.mean(e_s)],maxfev=5000)
    e_sr_only=sr_fn(cc_s,*popt2); rmse_sr=np.sqrt(np.mean((e_sr_only-e_s)**2))
except:
    c2=P.polyfit(cc_s,e_s,5); e_sr_only=P.polyval(cc_s,c2); rmse_sr=np.sqrt(np.mean((e_sr_only-e_s)**2))

print(f"LES RMSE: {rmse_les:.4f}, SR RMSE: {rmse_sr:.4f}")

fig,axes=plt.subplots(1,3,figsize=(18,5.5))
ax=axes[0]; ax.scatter(cc_s,e_s,s=30,alpha=0.6,c='steelblue',edgecolors='k',linewidth=0.5,label='Reference')
ax.plot(cc_s,e_les_fit,'r-',linewidth=2,alpha=0.8,label=f'LES (RMSE={rmse_les:.4f})')
ax.set_xlabel('C-C Distance (Å)'); ax.set_ylabel('Total Energy')
ax.set_title('Charged Dimer Binding Curve'); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)

ax=axes[1]; ax.scatter(cc_s,e_s,s=20,alpha=0.5,c='gray',edgecolors='k',linewidth=0.3,label='Total',zorder=1)
ax.plot(cc_s,e_sr_fit,'b-',linewidth=1.5,alpha=0.8,label='$E_{sr}$',zorder=3)
ax.plot(cc_s,e_lr,'r--',linewidth=1.5,alpha=0.8,label='$E_{lr}=-1/r$',zorder=2)
ax.set_xlabel('C-C Distance (Å)'); ax.set_ylabel('Energy')
ax.set_title('LES Energy Decomposition'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

ax=axes[2]; methods=['LES','SR-only']; rmses=[rmse_les,rmse_sr]; colors=['#2196F3','#FF9800']
bars=ax.bar(methods,rmses,color=colors,edgecolor='k',linewidth=1.5,width=0.5)
for b,v in zip(bars,rmses): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.002,f'{v:.4f}',ha='center',va='bottom',fontsize=12)
ax.set_ylabel('RMSE'); ax.set_title('Model Comparison'); ax.grid(True,alpha=0.3,axis='y')

plt.tight_layout(); plt.savefig('report/images/fig2_charged_dimer.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved fig2")

json.dump({'n_frames':int(len(pos_cd)),'dist_range':[float(cc_dist.min()),float(cc_dist.max())],
    'e_range':[float(e_cd.min()),float(e_cd.max())],'rmse_les':float(rmse_les),'rmse_sr':float(rmse_sr)},
    open('outputs/dimer_results.json','w'),indent=2)

# ============================================================
# 3. Ag3
# ============================================================
print("\n"+"="*60); print("3. Ag3"); print("="*60)
pos_m=cs_ag==1; neg_m=cs_ag==-1
pos_e=e_ag[pos_m]; neg_e=e_ag[neg_m]
pos_p=pos_ag[pos_m]; neg_p=pos_ag[neg_m]

def ag_desc(pp):
    n=len(pp); d=np.zeros((n,3))
    for fi in range(n): p=pp[fi]; d[fi,0]=np.linalg.norm(p[0]-p[1]); d[fi,1]=np.linalg.norm(p[1]-p[2]); d[fi,2]=np.linalg.norm(p[0]-p[2])
    return d

dp=ag_desc(pos_p); dn=ag_desc(neg_p)
mbl_p=np.mean(dp); mbl_n=np.mean(dn)
print(f"+1: E={pos_e.mean():.3f}±{pos_e.std():.3f}, bond={mbl_p:.3f}Å")
print(f"-1: E={neg_e.mean():.3f}±{neg_e.std():.3f}, bond={mbl_n:.3f}Å")
print(f"Pos identical: {np.allclose(pos_p,neg_p)}, E identical: {np.allclose(pos_e,neg_e)}")

mbl_all=np.mean(np.vstack([dp,dn]),axis=1); e_all=np.hstack([pos_e,neg_e])
X_all=mbl_all.reshape(-1,1)
joint_m=LinearRegression().fit(X_all,e_all); joint_rmse=np.sqrt(np.mean((joint_m.predict(X_all)-e_all)**2))
sep_m_pos=LinearRegression().fit(X_all[:30],pos_e); sep_m_neg=LinearRegression().fit(X_all[30:],neg_e)
sep_rmse=np.sqrt(np.mean((np.hstack([sep_m_pos.predict(X_all[:30]),sep_m_neg.predict(X_all[30:])])-e_all)**2))
print(f"Joint RMSE: {joint_rmse:.4f}, Sep RMSE: {sep_rmse:.4f}")

fig,axes=plt.subplots(1,3,figsize=(18,5.5))
ax=axes[0]; ax.scatter(dp.mean(axis=1),pos_e,s=40,alpha=0.6,c='red',edgecolors='k',linewidth=0.5,label='+1',marker='o')
ax.scatter(dn.mean(axis=1),neg_e,s=40,alpha=0.6,c='blue',edgecolors='k',linewidth=0.5,label='-1',marker='s')
ax.plot(np.sort(mbl_all),joint_m.predict(np.sort(mbl_all).reshape(-1,1)),'k--',linewidth=1.5,alpha=0.7,label='Joint fit')
ax.set_xlabel('Mean Bond Length (Å)'); ax.set_ylabel('Energy')
ax.set_title('Ag$_3$: Energy vs Geometry'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)

ax=axes[1]; pd_vals=e_all[:30]-e_all[30:]
ax.bar(range(30),pd_vals,color=['green' if abs(d)<1e-6 else 'red' for d in pd_vals],edgecolor='k',linewidth=0.5)
ax.set_xlabel('Frame Pair'); ax.set_ylabel('ΔE (+1 minus -1)')
ax.set_title('Charge State Energy Differences'); ax.axhline(y=0,color='gray',linestyle=':',alpha=0.5); ax.grid(True,alpha=0.3)

ax=axes[2]; bins=np.linspace(min(e_ag),max(e_ag),15)
ax.hist(pos_e,bins=bins,alpha=0.6,color='red',label='+1',edgecolor='k')
ax.hist(neg_e,bins=bins,alpha=0.6,color='blue',label='-1',edgecolor='k')
ax.set_xlabel('Energy'); ax.set_ylabel('Count')
ax.set_title('Energy Distribution'); ax.legend(fontsize=10); ax.grid(True,alpha=0.3)

plt.tight_layout(); plt.savefig('report/images/fig3_ag3_chargestates.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved fig3")

json.dump({'n_frames':int(len(pos_ag)),'n_pos':int(np.sum(pos_m)),'n_neg':int(np.sum(neg_m)),
    'e_pos_mean':float(pos_e.mean()),'e_pos_std':float(pos_e.std()),
    'e_neg_mean':float(neg_e.mean()),'e_neg_std':float(neg_e.std()),
    'joint_rmse':float(joint_rmse),'sep_rmse':float(sep_rmse),
    'pos_identical':bool(np.allclose(pos_p,neg_p)),'e_identical':bool(np.allclose(pos_e,neg_e))},
    open('outputs/ag3_results.json','w'),indent=2)

# ============================================================
# 4. OVERVIEW
# ============================================================
print("\nGenerating overview...")
fig,axes=plt.subplots(1,3,figsize=(18,5))
rc0=pos_rc[0]; tc0=tc_rc[0]
ax=axes[0]; ax.scatter(rc0[tc0>0,0],rc0[tc0>0,1],s=15,c='red',alpha=0.6,label='+1e',edgecolors='k',linewidth=0.3)
ax.scatter(rc0[tc0<0,0],rc0[tc0<0,1],s=15,c='blue',alpha=0.6,label='-1e',edgecolors='k',linewidth=0.3)
ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)'); ax.set_title('Random Charges'); ax.legend(fontsize=9)
ax.set_xlim(0,15); ax.set_ylim(0,15); ax.set_aspect('equal'); ax.grid(True,alpha=0.3)

cd0=pos_cd[0]
ax=axes[1]; ax.scatter(cd0[:4,0],cd0[:4,1],s=50,c='red',alpha=0.7,label='Mol 1 (+1e)',edgecolors='k',linewidth=0.5)
ax.scatter(cd0[4:,0],cd0[4:,1],s=50,c='blue',alpha=0.7,label='Mol 2 (-1e)',edgecolors='k',linewidth=0.5)
ax.plot([cd0[0,0],cd0[4,0]],[cd0[0,1],cd0[4,1]],'k--',alpha=0.3)
ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)'); ax.set_title('Charged Dimer'); ax.legend(fontsize=9); ax.set_aspect('equal'); ax.grid(True,alpha=0.3)

ag0=pos_ag[0]
ax=axes[2]; ax.scatter(ag0[:,0],ag0[:,1],s=80,c='green',alpha=0.7,edgecolors='k',linewidth=1)
for i in range(3):
    for j in range(i+1,3): ax.plot([ag0[i,0],ag0[j,0]],[ag0[i,1],ag0[j,1]],'gray',alpha=0.5,linewidth=1)
ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)'); ax.set_title('Ag$_3$ Trimer'); ax.set_aspect('equal'); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig0_data_overview.png',dpi=150,bbox_inches='tight'); plt.close()

# Ewald
demo_p=pos_rc[valid[0]]; demo_c=tc_rc[valid[0]]
kvs=list(range(1,8)); ec_k=[ewald_coulomb(demo_p,demo_c,kmax=k) for k in kvs]
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
ax=axes[0]; ax.plot(kvs,ec_k,'k-o',linewidth=2,markersize=6)
ax.set_xlabel('k-max'); ax.set_ylabel('Ewald Energy'); ax.set_title('Ewald Convergence'); ax.grid(True,alpha=0.3)
ax=axes[1]; u,c=np.unique(demo_c,return_counts=True)
ax.bar(['+1','-1'],c,color=['red','blue'],edgecolor='k',linewidth=1.5,width=0.4)
for i,cc in enumerate(c): ax.text(i,cc+0.5,str(cc),ha='center',fontsize=11)
ax.set_xlabel('Charge (e)'); ax.set_ylabel('Count'); ax.set_title('Charge Distribution'); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig5_ewald_decomposition.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved overview + ewald figs")
print("\nALL DONE!")
