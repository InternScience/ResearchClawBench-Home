#!/usr/bin/env python3
"""
MMGA Parameter Identification for Li-ion Battery ECAT Model.
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import os, json, time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ═══════ ECAT MODEL ═══════
PARAM_NAMES = ['Rp_neg','Rp_pos','k0_neg','k0_pos','Ds_neg','Ds_pos',
    'sigma_neg','sigma_pos','eps_s_neg','eps_s_pos','L_neg','L_pos',
    'h_conv','Cp_cell','R_ohmic','alpha_neg','alpha_pos']
N_PARAMS = len(PARAM_NAMES)
A_CELL=0.03; C_MAX_N=33133.0; C_MAX_P=37800.0; F_C=96485.3329; R_C=8.314462

BOUNDS_LOWER = np.array([2e-7,2e-7, 0.1,0.1, 1e-14,1e-14, 10,1, 0.40,0.40,
    50e-6,50e-6, 5.0,0.5, 0.005, 0.40,0.40])
BOUNDS_UPPER = np.array([3e-6,3e-6, 10,10, 5e-13,5e-13, 100,10, 0.65,0.65,
    120e-6,120e-6, 50.0,5.0, 0.15, 0.60,0.60])

def ocv_neg(th):
    th=np.clip(th,0.005,0.995)
    return np.clip(0.124+1.5*np.exp(-80*(th-0.13))+0.035*np.tanh((th-0.1)/0.01)
        -0.012*np.tanh((th-0.3)/0.01)+0.008*np.tanh((th-0.5)/0.05)
        -0.005*np.tanh((th-0.7)/0.05),0.005,0.40)
def ocv_pos(th):
    th=np.clip(th,0.005,0.995)
    return np.clip(3.0+1.5*(1-th)+0.1*np.log((1-th+0.01)/(th+0.01)),2.8,4.3)

def simulate_ecat(pv, I_app=1.1, dt=5.0):
    """Simulate CC discharge with SPM + thermal model."""
    Rp_n,Rp_p,k0n,k0p,Dsn,Dsp,sig_n,sig_p,eps_n,eps_p,L_n,L_p,h,Cp,Ro,al_n,al_p = pv
    Q_n=A_CELL*L_n*eps_n*C_MAX_N*F_C; Q_p=A_CELL*L_p*eps_p*C_MAX_P*F_C
    Q_total=min(Q_n*0.78,Q_p*0.72)
    ns=min(max(int(Q_total/I_app/dt),100),3000)
    cavg_n=0.85*C_MAX_N; cavg_p=0.25*C_MAX_P
    T_cell=298.15; Qd=0
    v_out=np.zeros(ns); t_out=np.zeros(ns); T_out=np.zeros(ns); c_out=np.zeros(ns)
    for s in range(ns):
        a_n=3*eps_n/Rp_n; a_p=3*eps_p/Rp_p
        jn=I_app/(A_CELL*L_n*a_n+1e-30); jp=I_app/(A_CELL*L_p*a_p+1e-30)
        cs_sn=np.clip(cavg_n+jn*Rp_n/(5*F_C*Dsn+1e-30),1,C_MAX_N-1)
        cs_sp=np.clip(cavg_p-jp*Rp_p/(5*F_C*Dsp+1e-30),1,C_MAX_P-1)
        Un=ocv_neg(cs_sn/C_MAX_N); Up=ocv_pos(cs_sp/C_MAX_P)
        th_sn=cs_sn/C_MAX_N; th_sp=cs_sp/C_MAX_P
        i0n=max(k0n*th_sn**al_n*(1-th_sn)**(1-al_n),1e-10)
        i0p=max(k0p*th_sp**al_p*(1-th_sp)**(1-al_p),1e-10)
        eta_n=R_C*T_cell/(al_n*F_C)*np.arcsinh(np.clip(jn/(2*i0n),-50,50))
        eta_p=R_C*T_cell/(al_p*F_C)*np.arcsinh(np.clip(jp/(2*i0p),-50,50))
        V=np.clip(Up+eta_p-Un+eta_n-I_app*Ro,2.5,4.3)
        v_out[s]=V; t_out[s]=s*dt; T_out[s]=T_cell; c_out[s]=Qd/3600
        if V<2.7:
            return {'time':t_out[:s+1],'voltage':v_out[:s+1],
                    'temperature':T_out[:s+1],'capacity':c_out[:s+1]}
        Q_gen=max(I_app*(Up-Un+I_app*Ro),0)
        T_cell+=dt*(Q_gen-h*A_CELL*(T_cell-298.15))/max(Cp,0.1)
        cavg_n-=I_app*dt/(A_CELL*L_n*eps_n*F_C)
        cavg_p+=I_app*dt/(A_CELL*L_p*eps_p*F_C)
        cavg_n=np.clip(cavg_n,1,C_MAX_N-1); cavg_p=np.clip(cavg_p,1,C_MAX_P-1)
        Qd+=I_app*dt
    return {'time':t_out,'voltage':v_out,'temperature':T_out,'capacity':c_out}

# ═══════ LHS ═══════
def lhs_sample(n):
    u=np.zeros((n,N_PARAMS))
    for j in range(N_PARAMS):
        perm=np.random.permutation(n)
        for i in range(n): u[i,j]=(perm[i]+np.random.uniform())/n
    return BOUNDS_LOWER+u*(BOUNDS_UPPER-BOUNDS_LOWER)

# ═══════ LOAD DATA ═══════
print("="*60)
print("MMGA Parameter Identification for Li-ion Battery ECAT Model")
print("="*60)

# CS2_36
df_ch=pd.read_excel('data/CS2_36/CS2_36_1_10_11.xlsx',sheet_name='Channel_1-009')
c1=df_ch[df_ch['Cycle_Index']==1]; s7=c1[c1['Step_Index']==7]
cs2_v=s7['Voltage(V)'].values; cs2_cap=s7['Discharge_Capacity(Ah)'].values
cs2_t=s7['Test_Time(s)'].values-s7['Test_Time(s)'].iloc[0]
cs2_aging=[]
for f in ['CS2_36_1_10_11','CS2_36_1_18_11','CS2_36_1_24_11','CS2_36_1_28_11']:
    dfs=pd.read_excel(f'data/CS2_36/{f}.xlsx',sheet_name='Statistics_1-009')
    cs2_aging.extend(dfs['Discharge_Capacity(Ah)'].values)
cs2_aging=np.array(cs2_aging)

# NASA
nasa_caps={}; nasa_first={}
for bat in ['B0005','B0006','B0007','B0018']:
    mat=sio.loadmat(f'data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/{bat}.mat')
    cycles=mat[bat][0,0]['cycle']; caps=[]; fd=None
    for i in range(cycles.shape[1]):
        c=cycles[0,i]
        if c['type'].item()=='discharge':
            d=c['data'][0,0]; caps.append(float(np.asarray(d['Capacity']).flatten()[-1]))
            if fd is None:
                fd={'v':np.asarray(d['Voltage_measured']).flatten(),
                    'I':np.abs(np.asarray(d['Current_measured']).flatten()),
                    'T':np.asarray(d['Temperature_measured']).flatten(),
                    't':np.asarray(d['Time']).flatten()}
    nasa_caps[bat]=np.array(caps); nasa_first[bat]=fd

# Oxford
ox_mat=sio.loadmat('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat')
ox_dc=ox_mat['ExampleDC_C1'][0,0]['dc'][0,0]
ox_v=np.asarray(ox_dc['v']).flatten()
ox_I=np.abs(np.asarray(ox_dc['i']).flatten())/1000
ox_t=np.asarray(ox_dc['t']).flatten()
ox_T=np.asarray(ox_dc['T']).flatten()

print(f"CS2_36: {len(cs2_v)} pts, {cs2_cap[-1]:.3f} Ah")
for b in nasa_caps: print(f"NASA {b}: {len(nasa_caps[b])} cycles")
print(f"Oxford: {len(ox_v)} pts")

# ═══════ GENERATE LHS TRAINING DATA ═══════
N_TRAIN=600
print(f"\n=== Generating {N_TRAIN} LHS samples ===")
samples=lhs_sample(N_TRAIN)
X_list=[]; Y_list=[]; t0=time.time()
for i in range(N_TRAIN):
    if (i+1)%100==0: print(f"  {i+1}/{N_TRAIN} ({time.time()-t0:.0f}s)")
    try:
        r=simulate_ecat(samples[i],dt=5.0)
        if len(r['voltage'])<30: continue
        f=interp1d(np.linspace(0,1,len(r['voltage'])),r['voltage'],kind='linear')
        v_res=f(np.linspace(0,1,100))
        X_list.append(samples[i]); Y_list.append(v_res)
    except: continue
X_train=np.array(X_list); Y_train=np.array(Y_list)
print(f"  Valid: {len(X_train)} samples ({time.time()-t0:.0f}s)")

# ═══════ TRAIN ANN ═══════
print("\n=== Training ANN Meta-Model ===")
sc_X=StandardScaler(); sc_Y=StandardScaler()
Xs=sc_X.fit_transform(X_train); Ys=sc_Y.fit_transform(Y_train)
Xtr,Xva,Ytr,Yva=train_test_split(Xs,Ys,test_size=0.2,random_state=42)
print(f"  Train: {Xtr.shape[0]}, Val: {Xva.shape[0]}")

ann=MLPRegressor(hidden_layer_sizes=(128,64,32),activation='relu',solver='adam',
    max_iter=1000,learning_rate_init=0.001,early_stopping=True,
    validation_fraction=0.15,random_state=42)
t0=time.time(); ann.fit(Xtr,Ytr)
tr_time=time.time()-t0
Ytr_p=sc_Y.inverse_transform(ann.predict(Xtr))
Yva_p=sc_Y.inverse_transform(ann.predict(Xva))
Ytr_o=sc_Y.inverse_transform(Ytr); Yva_o=sc_Y.inverse_transform(Yva)
tr_rmse=np.sqrt(mean_squared_error(Ytr_o,Ytr_p))
va_rmse=np.sqrt(mean_squared_error(Yva_o,Yva_p))
tr_r2=r2_score(Ytr_o,Ytr_p); va_r2=r2_score(Yva_o,Yva_p)
print(f"  Train RMSE: {tr_rmse:.6f} V, R²: {tr_r2:.6f}")
print(f"  Val RMSE:   {va_rmse:.6f} V, R²: {va_r2:.6f}")
print(f"  Training: {tr_time:.1f}s, {ann.n_iter_} iterations")
ann_metrics={'train_rmse':tr_rmse,'val_rmse':va_rmse,'train_r2':tr_r2,'val_r2':va_r2,'n_iter':int(ann.n_iter_)}

def ann_predict(pv):
    return sc_Y.inverse_transform(ann.predict(sc_X.transform(pv.reshape(1,-1))))[0]

# ═══════ MMGA OPTIMIZATION ═══════
print("\n=== MMGA Optimization ===")
ref_v=cs2_v.copy()
f_ref=interp1d(np.linspace(0,1,len(ref_v)),ref_v,kind='linear')
ref_norm=f_ref(np.linspace(0,1,100))

POP_M,GEN_M=100,80
def mmga_fit(x):
    try: return 1.0/(1.0+np.sqrt(np.mean((ann_predict(x)-ref_norm)**2)))
    except: return 0.0

pop=lhs_sample(POP_M)
fits=np.array([mmga_fit(pop[i]) for i in range(POP_M)])
bi=np.argmax(fits); best_x=pop[bi].copy(); best_f=fits[bi]
mmga_hist={'best_rmse':[np.sqrt(max(1/best_f-1,1e-12))],'avg_rmse':[]}
t0=time.time()
for g in range(1,GEN_M+1):
    sel=np.zeros_like(pop)
    for i in range(POP_M):
        c=np.random.choice(POP_M,5,replace=False)
        sel[i]=pop[c[np.argmax(fits[c])]]
    new_pop=np.zeros_like(pop); new_pop[0]=best_x.copy(); i=1
    while i<POP_M-1:
        p1,p2=sel[i-1],sel[min(i,POP_M-1)]
        c1,c2=p1.copy(),p2.copy()
        for j in range(N_PARAMS):
            if np.random.random()<0.5:
                u=np.random.random()
                beta=(2*u)**0.05 if u<=0.5 else (1/(2*(1-u)))**0.05
                c1[j]=0.5*((1+beta)*p1[j]+(1-beta)*p2[j])
                c2[j]=0.5*((1-beta)*p1[j]+(1+beta)*p2[j])
        for j in range(N_PARAMS):
            for idx in [i,i+1]:
                if idx<POP_M and np.random.random()<0.15:
                    d=BOUNDS_UPPER[j]-BOUNDS_LOWER[j]
                    arr=c1 if idx==i else c2
                    arr[j]=np.clip(arr[j]+np.random.normal(0,0.05)*d,BOUNDS_LOWER[j],BOUNDS_UPPER[j])
        new_pop[i]=np.clip(c1,BOUNDS_LOWER,BOUNDS_UPPER)
        new_pop[i+1]=np.clip(c2,BOUNDS_LOWER,BOUNDS_UPPER)
        i+=2
    if i<POP_M: new_pop[i]=sel[i]
    pop=new_pop
    fits=np.array([mmga_fit(pop[i]) for i in range(POP_M)])
    gi=np.argmax(fits)
    if fits[gi]>best_f: best_f=fits[gi]; best_x=pop[gi].copy()
    rmse_b=np.sqrt(max(1/best_f-1,1e-12))
    mmga_hist['best_rmse'].append(rmse_b)
    mmga_hist['avg_rmse'].append(np.sqrt(max(1/np.mean(fits)-1,1e-12)))
    if g%10==0 or g==GEN_M:
        print(f"  Gen {g}: Best RMSE={rmse_b:.6f} V ({time.time()-t0:.0f}s)")

mmga_params=dict(zip(PARAM_NAMES,best_x))
mmga_time=time.time()-t0
print(f"  MMGA done: RMSE={rmse_b:.6f} V, {mmga_time:.1f}s")

# ═══════ DIRECT GA BENCHMARK ═══════
print("\n=== Direct GA Benchmark ===")
POP_D,GEN_D=20,3
def dir_fit(x):
    try:
        r=simulate_ecat(x,dt=5.0)
        f=interp1d(np.linspace(0,1,len(r['voltage'])),r['voltage'],kind='linear')
        return 1.0/(1.0+np.sqrt(np.mean((f(np.linspace(0,1,100))-ref_norm)**2)))
    except: return 0.0

pop_d=lhs_sample(POP_D)
fits_d=np.array([dir_fit(pop_d[i]) for i in range(POP_D)])
bdi=np.argmax(fits_d); best_dx=pop_d[bdi].copy(); best_df=fits_d[bdi]
dir_hist={'best_rmse':[np.sqrt(max(1/best_df-1,1e-12))],'avg_rmse':[]}
t0=time.time()
for g in range(GEN_D):
    si=np.argsort(fits_d)[::-1]
    new_pop=np.zeros_like(pop_d); new_pop[0]=best_dx.copy()
    for i in range(1,POP_D,2):
        p1,p2=pop_d[si[i%len(si)]],pop_d[si[(i+1)%len(si)]]
        c1,c2=p1.copy(),p2.copy()
        for j in range(N_PARAMS):
            if np.random.random()<0.5:
                u=np.random.random(); beta=(2*u)**0.05 if u<=0.5 else (1/(2*(1-u)))**0.05
                c1[j]=0.5*((1+beta)*p1[j]+(1-beta)*p2[j])
                c2[j]=0.5*((1-beta)*p1[j]+(1+beta)*p2[j])
        for j in range(N_PARAMS):
            for arr,idx in [(c1,i),(c2,i+1)]:
                if idx<POP_D and np.random.random()<0.2:
                    d=BOUNDS_UPPER[j]-BOUNDS_LOWER[j]
                    arr[j]=np.clip(arr[j]+np.random.normal(0,0.05)*d,BOUNDS_LOWER[j],BOUNDS_UPPER[j])
        if i<POP_D: new_pop[i]=np.clip(c1,BOUNDS_LOWER,BOUNDS_UPPER)
        if i+1<POP_D: new_pop[i+1]=np.clip(c2,BOUNDS_LOWER,BOUNDS_UPPER)
    pop_d=new_pop
    fits_d=np.array([dir_fit(pop_d[i]) for i in range(POP_D)])
    gi=np.argmax(fits_d)
    if fits_d[gi]>best_df: best_df=fits_d[gi]; best_dx=pop_d[gi].copy()
    rmse_d=np.sqrt(max(1/best_df-1,1e-12))
    dir_hist['best_rmse'].append(rmse_d)
    dir_hist['avg_rmse'].append(np.sqrt(max(1/np.mean(fits_d)-1,1e-12)))
    print(f"  Gen {g}: RMSE={rmse_d:.6f} V ({time.time()-t0:.0f}s)")
dir_params=dict(zip(PARAM_NAMES,best_dx))
dir_time=time.time()-t0
print(f"  Direct GA done: RMSE={rmse_d:.6f} V, {dir_time:.1f}s")

# ═══════ VALIDATION ═══════
print("\n=== Validation ===")
def validate(pvec,exp_v,label=""):
    r=simulate_ecat(pvec,dt=5.0)
    fs=interp1d(np.linspace(0,1,len(r['voltage'])),r['voltage'],kind='linear')
    fe=interp1d(np.linspace(0,1,len(exp_v)),exp_v,kind='linear')
    x=np.linspace(0,1,100); si,ei=fs(x),fe(x)
    rmse=np.sqrt(np.mean((si-ei)**2)); mae=np.mean(np.abs(si-ei)); r2=r2_score(ei,si)
    print(f"  {label}: RMSE={rmse:.4f}V MAE={mae:.4f}V R²={r2:.4f}")
    return {'rmse':rmse,'mae':mae,'r2':r2,'sim_v':si,'exp_v':ei,
            'full_v':r['voltage'],'full_t':r['time'],'full_T':r['temperature']}

mmga_vec=np.array([mmga_params[n] for n in PARAM_NAMES])
dir_vec=np.array([dir_params[n] for n in PARAM_NAMES])
val_cs2=validate(mmga_vec,cs2_v,"MMGA vs CS2_36")
val_cs2d=validate(dir_vec,cs2_v,"Direct GA vs CS2_36")
val_nasa=validate(mmga_vec,nasa_first['B0005']['v'],"MMGA vs NASA B0005")

# ═══════ GENERATE FIGURES ═══════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':12,
    'figure.dpi':150,'lines.linewidth':1.5,'axes.grid':True,'grid.alpha':0.3})
fd='report/images'

# Fig 1: Data Overview
fig,axes=plt.subplots(2,3,figsize=(15,9))
axes[0,0].plot(cs2_cap,cs2_v,'b-',lw=2)
axes[0,0].set_xlabel('Capacity (Ah)'); axes[0,0].set_ylabel('Voltage (V)')
axes[0,0].set_title('(a) CS2_36 1C Discharge')
axes[0,1].plot(nasa_first['B0005']['v'],'r-',lw=2)
axes[0,1].set_xlabel('Data Point'); axes[0,1].set_ylabel('Voltage (V)')
axes[0,1].set_title('(b) NASA B0005 Discharge')
t_ox_h=(ox_t-ox_t[0])/3600
axes[0,2].plot(t_ox_h,ox_v,'g-',lw=1)
axes[0,2].set_xlabel('Time (h)'); axes[0,2].set_ylabel('Voltage (V)')
axes[0,2].set_title('(c) Oxford Artemis Drive Cycle')
for bat,c in zip(['B0005','B0006','B0007','B0018'],['b','r','g','m']):
    axes[1,0].plot(nasa_caps[bat],'-',color=c,label=bat,lw=1.5)
axes[1,0].set_xlabel('Cycle'); axes[1,0].set_ylabel('Capacity (Ah)')
axes[1,0].set_title('(d) NASA Capacity Fade'); axes[1,0].legend()
axes[1,1].plot(range(len(cs2_aging)),cs2_aging,'b-o',lw=1.5,ms=2)
axes[1,1].set_xlabel('Cycle'); axes[1,1].set_ylabel('Discharge Capacity (Ah)')
axes[1,1].set_title('(e) CS2_36 Capacity Fade')
axes[1,2].scatter(X_train[:,0]*1e6,X_train[:,1]*1e6,s=5,alpha=0.5,c='teal')
axes[1,2].set_xlabel('$R_{p,neg}$ (μm)'); axes[1,2].set_ylabel('$R_{p,pos}$ (μm)')
axes[1,2].set_title('(f) LHS Sampling (17D → 2D)')
plt.tight_layout(); plt.savefig(f'{fd}/fig1_data_overview.png',dpi=150,bbox_inches='tight'); plt.close()
print("  Saved fig1_data_overview.png")

# Fig 2: ANN Performance
fig,axes=plt.subplots(2,2,figsize=(12,10))
for i in range(min(40,Ytr_o.shape[0])):
    axes[0,0].plot(Ytr_o[i],'b-',alpha=0.15,lw=0.7)
    axes[0,0].plot(Ytr_p[i],'r--',alpha=0.15,lw=0.7)
axes[0,0].set_title('(a) Training: True (blue) vs Predicted (red)'); axes[0,0].set_xlim([0,99])
for i in range(min(40,Yva_o.shape[0])):
    axes[0,1].plot(Yva_o[i],'b-',alpha=0.15,lw=0.7)
    axes[0,1].plot(Yva_p[i],'r--',alpha=0.15,lw=0.7)
axes[0,1].set_title('(b) Validation: True (blue) vs Predicted (red)'); axes[0,1].set_xlim([0,99])
axes[1,0].scatter(Yva_o.flatten(),Yva_p.flatten(),s=3,alpha=0.3,c='teal')
lims=[min(Yva_o.min(),Yva_p.min()),max(Yva_o.max(),Yva_p.max())]
axes[1,0].plot(lims,lims,'k--',lw=1)
axes[1,0].set_xlabel('True (V)'); axes[1,0].set_ylabel('Predicted (V)')
axes[1,0].set_title(f'(c) Parity Plot (R²={va_r2:.4f})')
axes[1,1].plot(ann.loss_curve_,'b-',lw=2)
axes[1,1].set_xlabel('Iteration'); axes[1,1].set_ylabel('Loss')
axes[1,1].set_title('(d) ANN Training Loss'); axes[1,1].set_yscale('log')
plt.tight_layout(); plt.savefig(f'{fd}/fig2_ann_performance.png',dpi=150,bbox_inches='tight'); plt.close()
print("  Saved fig2_ann_performance.png")

# Fig 3: MMGA Convergence
fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].plot(mmga_hist['best_rmse'],'b-o',lw=2,ms=2,label='MMGA Best')
axes[0].plot(mmga_hist['avg_rmse'],'b--',lw=1,alpha=0.5,label='MMGA Avg')
n_m=len(mmga_hist['best_rmse']); n_d=len(dir_hist['best_rmse'])
scale=max(n_m//n_d,1)
dx=[i*scale for i in range(n_d)]
dx_avg=[i*scale for i in range(len(dir_hist['avg_rmse']))]
axes[0].plot(dx,dir_hist['best_rmse'],'r-s',lw=2,ms=5,label='Direct GA Best')
axes[0].plot(dx_avg,dir_hist['avg_rmse'],'r--',lw=1,alpha=0.5,label='Direct GA Avg')
axes[0].set_xlabel('Generation'); axes[0].set_ylabel('RMSE (V)')
axes[0].set_title('(a) Convergence'); axes[0].legend(); axes[0].set_yscale('log')
bars=axes[1].bar([0,1],[mmga_time,dir_time],0.5,color=['steelblue','coral'],alpha=0.8)
axes[1].set_ylabel('Time (s)'); axes[1].set_title('(b) Computation Speed')
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(['MMGA\n(ANN)','Direct GA\n(Physics)'])
for b,rv in zip(bars,[rmse_b,rmse_d]):
    axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+5,f'RMSE={rv:.4f}V',ha='center',va='bottom',fontsize=10)
plt.tight_layout(); plt.savefig(f'{fd}/fig3_mmga_convergence.png',dpi=150,bbox_inches='tight'); plt.close()
print("  Saved fig3_mmga_convergence.png")

# Fig 4: Parameter Comparison
fig,ax=plt.subplots(figsize=(14,6))
x=np.arange(N_PARAMS); w=0.35
mn=[(mmga_params[n]-BOUNDS_LOWER[i])/(BOUNDS_UPPER[i]-BOUNDS_LOWER[i]) for i,n in enumerate(PARAM_NAMES)]
dn=[(dir_params[n]-BOUNDS_LOWER[i])/(BOUNDS_UPPER[i]-BOUNDS_LOWER[i]) for i,n in enumerate(PARAM_NAMES)]
ax.bar(x-w/2,mn,w,label='MMGA',color='steelblue',alpha=0.8)
ax.bar(x+w/2,dn,w,label='Direct GA',color='coral',alpha=0.8)
short=['Rp_n','Rp_p','k0_n','k0_p','Ds_n','Ds_p','σ_n','σ_p','ε_n','ε_p',
       'L_n','L_p','h','Cp','Rf','α_n','α_p']
ax.set_xticks(x); ax.set_xticklabels(short,rotation=45,ha='right')
ax.set_ylabel('Normalized [0,1]'); ax.set_title('Identified Parameters'); ax.legend()
plt.tight_layout(); plt.savefig(f'{fd}/fig4_parameter_comparison.png',dpi=150,bbox_inches='tight'); plt.close()
print("  Saved fig4_parameter_comparison.png")

# Fig 5: Validation
fig,axes=plt.subplots(1,3,figsize=(15,5))
x100=np.linspace(0,1,100)
axes[0].plot(x100,val_cs2['sim_v'],'b-',lw=2,label='MMGA')
axes[0].plot(x100,val_cs2d['sim_v'],'r--',lw=2,label='Direct GA')
axes[0].plot(x100,val_cs2['exp_v'],'k:',lw=2,label='Experiment')
axes[0].set_xlabel('Normalized Capacity'); axes[0].set_ylabel('Voltage (V)')
axes[0].set_title('(a) CS2_36 Validation'); axes[0].legend()
axes[1].plot(x100,val_nasa['sim_v'],'b-',lw=2,label='MMGA')
axes[1].plot(x100,val_nasa['exp_v'],'k:',lw=2,label='Experiment')
axes[1].set_xlabel('Normalized Capacity'); axes[1].set_ylabel('Voltage (V)')
axes[1].set_title('(b) NASA B0005 Validation'); axes[1].legend()
r_mmga=simulate_ecat(mmga_vec,dt=5.0)
axes[2].plot(r_mmga['time']/60,r_mmga['temperature']-273.15,'b-',lw=2,label='MMGA Model')
axes[2].set_xlabel('Time (min)'); axes[2].set_ylabel('Temperature (°C)')
axes[2].set_title('(c) Thermal Response'); axes[2].legend()
plt.tight_layout(); plt.savefig(f'{fd}/fig5_validation.png',dpi=150,bbox_inches='tight'); plt.close()
print("  Saved fig5_validation.png")

# Fig 6: Sensitivity Analysis
fig,axes=plt.subplots(1,2,figsize=(12,5))
for bat,c in zip(['B0005','B0006','B0007','B0018'],['b','r','g','m']):
    axes[0].plot(nasa_caps[bat],'-',color=c,label=bat,lw=1.5)
axes[0].set_xlabel('Cycle'); axes[0].set_ylabel('Capacity (Ah)')
axes[0].set_title('(a) NASA Aging Data'); axes[0].legend()
ax=axes[1]
for pi,lab,col in zip([0,2,4,8,14],['Rp_neg','k0_neg','Ds_neg','ε_s_neg','R_ohmic'],
                       ['b','r','g','m','c']):
    rmses_s=[]
    for frac in np.linspace(0.1,0.9,15):
        pv=mmga_vec.copy()
        pv[pi]=BOUNDS_LOWER[pi]+frac*(BOUNDS_UPPER[pi]-BOUNDS_LOWER[pi])
        try:
            r=simulate_ecat(pv,dt=5.0)
            f=interp1d(np.linspace(0,1,len(r['voltage'])),r['voltage'],kind='linear')
            rmses_s.append(np.sqrt(np.mean((f(x100)-ref_norm)**2)))
        except: rmses_s.append(np.nan)
    ax.plot(np.linspace(0.1,0.9,15),rmses_s,'-o',color=col,label=lab,lw=1.5,ms=3)
ax.set_xlabel('Normalized Parameter'); ax.set_ylabel('RMSE (V)')
ax.set_title('(b) Parameter Sensitivity'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{fd}/fig6_sensitivity.png',dpi=150,bbox_inches='tight'); plt.close()
print("  Saved fig6_sensitivity.png")

# ═══════ SAVE RESULTS ═══════
results={
    'mmga_params':{k:float(v) for k,v in mmga_params.items()},
    'mmga_rmse':float(rmse_b),'mmga_time_s':float(mmga_time),
    'direct_params':{k:float(v) for k,v in dir_params.items()},
    'direct_rmse':float(rmse_d),'direct_time_s':float(dir_time),
    'ann_metrics':{k:float(v) if isinstance(v,(float,np.floating)) else v for k,v in ann_metrics.items()},
    'validation':{
        'cs2_36_mmga':{'rmse':float(val_cs2['rmse']),'mae':float(val_cs2['mae']),'r2':float(val_cs2['r2'])},
        'cs2_36_direct':{'rmse':float(val_cs2d['rmse']),'mae':float(val_cs2d['mae']),'r2':float(val_cs2d['r2'])},
        'nasa_b0005':{'rmse':float(val_nasa['rmse']),'mae':float(val_nasa['mae']),'r2':float(val_nasa['r2'])},
    }}
with open('outputs/results.json','w') as f: json.dump(results,f,indent=2)
np.savez('outputs/ann_validation.npz',Y_val_true=Yva_o,Y_val_pred=Yva_p,Y_train_true=Ytr_o,Y_train_pred=Ytr_p)
np.savez('outputs/training_data.npz',X_params=X_train,Y_voltage=Y_train)
print("\n=== All results saved ===")
print(json.dumps(results,indent=2))
print("\nDONE")
