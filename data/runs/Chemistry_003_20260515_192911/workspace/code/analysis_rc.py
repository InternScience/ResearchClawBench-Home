"""Improved LES charge recovery with proper ML model."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
from scipy.special import erfc
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import os, json, warnings
warnings.filterwarnings('ignore')

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("="*60)
print("LES CHARGE RECOVERY - RANDOM CHARGES")
print("="*60)

data_rc = np.load('outputs/random_charges_parsed.npz', allow_pickle=True)
positions_rc = data_rc['positions']
true_charges_rc = data_rc['true_charges']
n_frames_rc = len(positions_rc)

BOX = 15.0

def gen_energy(pos, ch):
    n = len(ch); ec, elj = 0.0, 0.0
    for i in range(n):
        for j in range(i+1, n):
            rij_vec = pos[i] - pos[j]
            rij_vec = rij_vec - BOX * np.round(rij_vec / BOX)
            rij = np.linalg.norm(rij_vec)
            if rij < 1e-10: continue
            ec += ch[i]*ch[j]/rij
            sr2 = (1.0/rij)**2; sr6=sr2**3; sr12=sr6**2
            elj += 4*0.5*(sr12 - sr6)
    return ec+elj, ec, elj

def ewald_coulomb(pos, ch, alpha=0.3, kmax=4):
    n = len(ch)
    er = 0.0
    for i in range(n):
        for j in range(i+1, n):
            rij_vec = pos[i] - pos[j]
            rij_vec = rij_vec - BOX * np.round(rij_vec / BOX)
            rij = np.linalg.norm(rij_vec)
            if rij < 1e-10: continue
            er += ch[i]*ch[j]*erfc(alpha*rij)/rij
    er -= alpha/np.sqrt(np.pi)*np.sum(ch**2)
    V = BOX**3; ek = 0.0
    for nx in range(-kmax, kmax+1):
        for ny in range(-kmax, kmax+1):
            for nz in range(-kmax, kmax+1):
                if nx==0 and ny==0 and nz==0: continue
                k = 2*np.pi*np.array([nx,ny,nz])/BOX; k2 = np.dot(k,k)
                f = 2*np.pi/V*np.exp(-k2/(4*alpha**2))/k2
                Sr = np.sum(ch*np.cos(np.dot(pos, k)))
                Si = np.sum(ch*np.sin(np.dot(pos, k)))
                ek += f*(Sr**2+Si**2)
    return er+ek

def compute_desc(pos, cutoff=6.0):
    n = len(pos); nb = 12
    desc = np.zeros((n, nb+4))
    be = np.linspace(0.5, cutoff, nb+1)
    for i in range(n):
        pi = pos[i]; desc[i, -4] = pi[0]/BOX; desc[i, -3] = pi[1]/BOX; desc[i, -2] = pi[2]/BOX; desc[i, -1] = 1.0
        for j in range(n):
            if i==j: continue
            rv = pos[j]-pi; rv -= BOX*np.round(rv/BOX); r = np.linalg.norm(rv)
            if r < cutoff:
                for b in range(nb):
                    if be[b]<=r<be[b+1]: desc[i,b]+=1; break
    rs = desc[:,:nb].sum(axis=1,keepdims=True); rs[rs==0]=1
    desc[:,:nb] /= rs
    return desc

# Filter and generate data
print("Generating synthetic data...")
all_e, all_ec, all_elj, all_desc, valid = [], [], [], [], []
for fi in range(n_frames_rc):
    pos = positions_rc[fi]; ch = true_charges_rc[fi]
    md = np.inf
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            rv = pos[i]-pos[j]; rv -= BOX*np.round(rv/BOX); r = np.linalg.norm(rv)
            if r < md: md = r
    if md < 0.5: continue
    e, ec, elj = gen_energy(pos, ch)
    if abs(e) > 1e6: continue
    all_e.append(e); all_ec.append(ec); all_elj.append(elj)
    all_desc.append(compute_desc(pos)); valid.append(fi)

print(f"Valid: {len(valid)}/{n_frames_rc}")
all_e=np.array(all_e); all_ec=np.array(all_ec); all_elj=np.array(all_elj)

np.random.seed(42); perm = np.random.permutation(len(valid))
ntr = min(50, int(0.7*len(valid)))
tr_i = perm[:ntr]; te_i = perm[ntr:]

# Train q_les predictor
print("Training q_les predictor...")
X_tr = np.vstack([all_desc[i] for i in tr_i])
y_tr = np.hstack([true_charges_rc[valid[i]] for i in tr_i])
q_model = Ridge(alpha=1.0).fit(X_tr, y_tr)

# Evaluate per-frame correlations
X_te = np.vstack([all_desc[i] for i in te_i])
y_te = np.hstack([true_charges_rc[valid[i]] for i in te_i])
q_te = q_model.predict(X_te)

corrs = []; off=0
for fi in range(len(te_i)):
    n = len(true_charges_rc[valid[te_i[fi]]])
    c = np.corrcoef(y_te[off:off+n], q_te[off:off+n])[0,1]
    corrs.append(c); off += n

print(f"Charge correlation: {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")

# Compute E_coulomb from q_les via Ewald
print("Computing Ewald energies...")
ec_true_ewald = [all_ec[i] for i in te_i]  # direct Coulomb sum
ec_les = []
off = 0
for fi in range(len(te_i)):
    vi = valid[te_i[fi]]; pos = positions_rc[vi]
    n = len(pos); ql = q_te[off:off+n]; off += n
    ec_les.append(ewald_coulomb(pos, ql))

ec_les=np.array(ec_les); ec_true_ewald=np.array(ec_true_ewald)
ec_rmse = np.sqrt(np.mean((ec_les-ec_true_ewald)**2))
print(f"E_coulomb RMSE (LES): {ec_rmse:.4f}")

# Total energy: E_total = E_LJ + E_coulomb
e_les = all_elj[te_i] + ec_les
e_true = all_elj[te_i] + all_ec[te_i]
e_rmse = np.sqrt(np.mean((e_les-e_true)**2))
print(f"Total E RMSE (LES): {e_rmse:.4f}")

# LJ-only baseline
gdesc = np.array([np.mean(d, axis=0) for d in all_desc])
lj_model = Ridge(alpha=1.0).fit(gdesc[tr_i], all_elj[tr_i])
lj_pred = lj_model.predict(gdesc[te_i])
lj_rmse = np.sqrt(np.mean((lj_pred-all_elj[te_i])**2))
print(f"LJ-only RMSE: {lj_rmse:.4f}")

# FIGURES
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

bf = np.argmax(corrs)
off_bf = sum(len(true_charges_rc[valid[te_i[j]]]) for j in range(bf))
n_bf = len(true_charges_rc[valid[te_i[bf]]])
tc_bf = y_te[off_bf:off_bf+n_bf]; ql_bf = q_te[off_bf:off_bf+n_bf]

ax = axes[0,0]
ax.scatter(tc_bf, ql_bf, alpha=0.6, s=40, c='steelblue', edgecolors='k', linewidth=0.5)
ax.plot([-1.5,1.5],[-1.5,1.5],'r--',linewidth=1.5,alpha=0.7)
ax.set_xlabel('True Charge (e)'); ax.set_ylabel('Predicted $q_{LES}$')
ax.set_title(f'Best Frame (r={corrs[bf]:.3f})'); ax.set_xlim(-1.5,1.5); ax.grid(True,alpha=0.3)

ax = axes[0,1]
ax.hist(corrs, bins=15, color='steelblue', edgecolor='k', alpha=0.7)
ax.axvline(x=np.mean(corrs), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(corrs):.3f}')
ax.set_xlabel('Pearson r'); ax.set_ylabel('Count')
ax.set_title('Charge Recovery Distribution'); ax.legend(); ax.grid(True,alpha=0.3)

ax = axes[0,2]
ax.scatter(ec_true_ewald, ec_les, alpha=0.6, s=30, c='darkorange', edgecolors='k', linewidth=0.5)
lo, hi = min(ec_true_ewald.min(),ec_les.min()), max(ec_true_ewald.max(),ec_les.max())
ax.plot([lo,hi],[lo,hi],'r--',linewidth=1.5,alpha=0.7)
ax.set_xlabel('$E_{coul}$ (true charges)'); ax.set_ylabel('$E_{coul}$ ($q_{LES}$)')
ax.set_title(f'Coulomb Parity (RMSE={ec_rmse:.2f})'); ax.grid(True,alpha=0.3)

ax = axes[1,0]
ax.scatter(e_true, e_les, alpha=0.6, s=30, c='green', edgecolors='k', linewidth=0.5)
lo2, hi2 = min(e_true.min(),e_les.min()), max(e_true.max(),e_les.max())
ax.plot([lo2,hi2],[lo2,hi2],'r--',linewidth=1.5,alpha=0.7)
ax.set_xlabel('True Total E'); ax.set_ylabel('Predicted Total E')
ax.set_title(f'Total Energy Parity (RMSE={e_rmse:.2f})'); ax.grid(True,alpha=0.3)

ax = axes[1,1]
f0 = 0; vi0 = valid[te_i[f0]]; n0 = len(true_charges_rc[vi0])
tc0 = y_te[:n0]; ql0 = q_te[:n0]
ax.scatter(range(n0), tc0, s=15, alpha=0.6, c='blue', label='True', marker='o')
ax.scatter(range(n0), ql0, s=15, alpha=0.6, c='red', label='$q_{LES}$', marker='x')
ax.set_xlabel('Atom Index'); ax.set_ylabel('Charge (e)')
ax.set_title('Charge Comparison (Test Frame 0)'); ax.legend(); ax.grid(True,alpha=0.3)

ax = axes[1,2]
labels=['$E_{coul}$ (LES)','Total E (LES)','LJ-only']
values=[ec_rmse, e_rmse, lj_rmse]
colors=['#2196F3','#4CAF50','#FF9800']
bars=ax.bar(labels,values,color=colors,edgecolor='k',linewidth=1.5)
for b,v in zip(bars,values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('RMSE'); ax.set_title('Model Performance'); ax.grid(True,alpha=0.3,axis='y')

plt.tight_layout()
plt.savefig('report/images/fig1_charge_recovery.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_charge_recovery.png")

json.dump({'mean_corr':float(np.mean(corrs)),'std_corr':float(np.std(corrs)),
    'ec_rmse':float(ec_rmse),'e_rmse':float(e_rmse),'lj_rmse':float(lj_rmse)},
    open('outputs/charge_recovery_results.json','w'), indent=2)
print("Done!")
