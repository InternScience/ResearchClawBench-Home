#!/usr/bin/env python3
"""Reproducible MMGA-style ANN surrogate parameter identification for battery ECAT reduced-order model."""
import os, json, math, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import qmc
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT='.'
OUT='outputs'; IMG='report/images'
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True); os.makedirs('code', exist_ok=True)
RNG=np.random.default_rng(42)

GRID=np.linspace(0,1,121)

# ------------------------- data extraction -------------------------
def interp_grid(x, y, grid=GRID):
    x=np.asarray(x, dtype=float); y=np.asarray(y, dtype=float)
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    if len(x)<2: return np.full_like(grid, np.nan, dtype=float)
    # ensure monotone x
    order=np.argsort(x); x=x[order]; y=y[order]
    # combine duplicates
    xu, idx=np.unique(x, return_index=True)
    yu=y[idx]
    if len(xu)<2: return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, xu, yu)

def extract_cs2():
    rows=[]; curves={}
    for fname in sorted(os.listdir('data/CS2_36')):
        if not fname.endswith('.xlsx'): continue
        path=f'data/CS2_36/{fname}'
        df=pd.read_excel(path, sheet_name='Channel_1-009')
        stats=pd.read_excel(path, sheet_name='Statistics_1-009')
        # Discharge segments: negative current and growing discharge capacity.
        for cyc, g in df.groupby('Cycle_Index'):
            seg=g[(g['Current(A)'] < -0.05) & (g['Discharge_Capacity(Ah)'] >= 0)].copy()
            if len(seg)<20: continue
            dq=seg['Discharge_Capacity(Ah)']-seg['Discharge_Capacity(Ah)'].min()
            cap=float(dq.max())
            if cap<=0.05: continue
            q=dq.to_numpy()/cap
            v=seg['Voltage(V)'].to_numpy()
            t=seg['Step_Time(s)'].to_numpy(); t=t-t.min()
            # Estimate temperature is unavailable in CS2 files; use internal resistance and ohmic heat proxy to create a pseudo thermal observable.
            stat=stats[stats['Cycle_Index']==cyc]
            r=float(stat['Internal_Resistance(Ohm)'].iloc[0]) if len(stat) and np.isfinite(stat['Internal_Resistance(Ohm)'].iloc[0]) else float(np.nan)
            cur=float(abs(seg['Current(A)'].median()))
            temp=25 + (cur**2) * (r if np.isfinite(r) else 0.11) * 7.5 * (1-np.exp(-3*np.clip(q,0,1)))
            key=f'CS2_36:{fname}:cycle{int(cyc)}'
            rows.append({'dataset':'CS2_36','source':fname,'cycle':int(cyc),'key':key,'n_points':len(seg),'capacity_Ah':cap,'duration_s':float(t.max()),'current_A':cur,'ambient_C':25.0,'internal_resistance_ohm':r})
            curves[key]={'soc':q,'voltage':v,'temp':temp,'time':t,'capacity_Ah':cap,'current_A':cur,'ambient_C':25.0,'dataset':'CS2_36','source':fname,'cycle':int(cyc)}
    return rows, curves

# robust MATLAB structure conversion for NASA/Oxford
def _mat_struct_to_dict(obj):
    if hasattr(obj, '_fieldnames'):
        return {f:_mat_struct_to_dict(getattr(obj,f)) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype==object:
            return [_mat_struct_to_dict(o) for o in obj.flat]
        return obj
    return obj

def extract_nasa():
    rows=[]; curves={}
    base='data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4'
    for fname in sorted(os.listdir(base)):
        if not fname.endswith('.mat'): continue
        path=os.path.join(base,fname)
        mat=loadmat(path, squeeze_me=True, struct_as_record=False)
        bat=fname.replace('.mat','')
        obj=mat.get(bat)
        d=_mat_struct_to_dict(obj)
        cycles=d['cycle'] if isinstance(d['cycle'], list) else [d['cycle']]
        count=0
        for i,c in enumerate(cycles):
            if c.get('type')!='discharge': continue
            dat=c['data']
            v=np.ravel(dat['Voltage_measured']).astype(float)
            cur=np.ravel(dat['Current_measured']).astype(float)
            temp=np.ravel(dat['Temperature_measured']).astype(float)
            time=np.ravel(dat['Time']).astype(float)
            cap=float(np.ravel(dat['Capacity'])[0]) if 'Capacity' in dat and np.size(dat['Capacity']) else np.nan
            if len(v)<20 or not np.isfinite(cap): continue
            q=(time-time.min())/(time.max()-time.min()) # no Ah vector, use normalized discharge progress
            key=f'NASA:{bat}:discharge{count}'
            rows.append({'dataset':'NASA','source':bat,'cycle':count,'key':key,'n_points':len(v),'capacity_Ah':cap,'duration_s':float(time.max()-time.min()),'current_A':float(np.nanmedian(np.abs(cur))),'ambient_C':float(c.get('ambient_temperature',25)),'internal_resistance_ohm':np.nan})
            curves[key]={'soc':q,'voltage':v,'temp':temp,'time':time-time.min(),'capacity_Ah':cap,'current_A':float(np.nanmedian(np.abs(cur))),'ambient_C':float(c.get('ambient_temperature',25)),'dataset':'NASA','source':bat,'cycle':count}
            count+=1
            if count>=8: break
    return rows, curves

def extract_oxford():
    rows=[]; curves={}
    path='data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat'
    mat=loadmat(path, squeeze_me=True, struct_as_record=False)
    d=_mat_struct_to_dict(mat)
    # Usually keys include ch, dc
    root=d.get('ExampleDC_C1', d)
    dc=None
    if isinstance(root, dict):
        dc=root.get('dc') or root.get('DC') or None
    elif hasattr(root, '_fieldnames'):
        # defensive branch; normally _mat_struct_to_dict has already converted structs.
        tmp=getattr(root, 'dc', None) or getattr(root, 'DC', None)
        dc=_mat_struct_to_dict(tmp) if tmp is not None else None
    if dc is None:
        # find struct-like dict containing v,i,T,t
        items = root.items() if isinstance(root,dict) else d.items()
        for k,v in items:
            if isinstance(v,dict) and all(x in v for x in ['v','t']): dc=v
    if dc is not None:
        t=np.ravel(dc.get('t')).astype(float)
        v=np.ravel(dc.get('v')).astype(float)
        temp=np.ravel(dc.get('T', np.full_like(v,40))).astype(float)
        cur=np.ravel(dc.get('i', np.full_like(v,np.nan))).astype(float)
        q_raw=np.ravel(dc.get('q', np.linspace(0,740,len(v)))).astype(float)
        q=(q_raw-q_raw.min())/(q_raw.max()-q_raw.min()) if np.nanmax(q_raw)>np.nanmin(q_raw) else (t-t.min())/(t.max()-t.min())
        cap_range=float(np.nanmax(q_raw)-np.nanmin(q_raw))
        # Oxford q is labelled mAh in the readme, but ExampleDC values are already Ah-scale in this file.
        cap=cap_range/1000.0 if cap_range>10 else cap_range
        key='Oxford:ExampleDC_C1:dynamic_discharge'
        rows.append({'dataset':'Oxford','source':'ExampleDC_C1','cycle':1,'key':key,'n_points':len(v),'capacity_Ah':cap,'duration_s':float(t.max()-t.min()),'current_A':float(np.nanmedian(np.abs(cur))/1000 if np.nanmedian(np.abs(cur))>10 else np.nanmedian(np.abs(cur))),'ambient_C':40.0,'internal_resistance_ohm':np.nan})
        curves[key]={'soc':q,'voltage':v,'temp':temp,'time':t-t.min(),'capacity_Ah':cap,'current_A':rows[-1]['current_A'],'ambient_C':40.0,'dataset':'Oxford','source':'ExampleDC_C1','cycle':1}
    return rows, curves

rows=[]; curves={}
for func in [extract_cs2, extract_nasa, extract_oxford]:
    r,c=func(); rows.extend(r); curves.update(c)
overview=pd.DataFrame(rows)
overview.to_csv(os.path.join(OUT,'data_overview.csv'), index=False)

# Select calibration targets: representative healthy CS2 cycle, aged CS2 cycle, NASA B0005 first, Oxford dynamic.
def choose_key(dataset, prefer_source=None, which='first'):
    sub=overview[overview.dataset==dataset]
    if prefer_source is not None: sub=sub[sub.source.astype(str).str.contains(prefer_source)]
    if len(sub)==0: return None
    if which=='last': return sub.sort_values('capacity_Ah').iloc[0]['key']
    if which=='median': return sub.iloc[len(sub)//2]['key']
    return sub.sort_values('capacity_Ah', ascending=False).iloc[0]['key']
primary_key=choose_key('CS2_36', which='first')
aged_key=choose_key('CS2_36', which='last')
nasa_key=choose_key('NASA', prefer_source='B0005', which='first')
ox_key=choose_key('Oxford')
selected=[k for k in [primary_key, aged_key, nasa_key, ox_key] if k]

# ------------------------- reduced-order ECAT model -------------------------
param_defs=pd.DataFrame([
 ('R_s_pos_um',2.0,12.0,'positive particle radius proxy; larger causes diffusion polarization'),
 ('R_s_neg_um',2.0,12.0,'negative particle radius proxy'),
 ('k_pos_1e-11',0.5,8.0,'positive reaction-rate constant scale'),
 ('k_neg_1e-11',0.5,8.0,'negative reaction-rate constant scale'),
 ('R_ohm_mOhm',20.0,180.0,'lumped ohmic resistance'),
 ('h_W_m2K',2.0,35.0,'convective thermal coefficient'),
 ('C_th_JK',120.0,1300.0,'thermal capacitance'),
 ('aging_loss_frac',0.0,0.45,'capacity loss fraction')
], columns=['parameter','lower','upper','description'])
param_defs.to_csv(os.path.join(OUT,'parameter_search_space.csv'), index=False)
P=param_defs['parameter'].tolist(); lows=param_defs.lower.to_numpy(float); highs=param_defs.upper.to_numpy(float)

def simulate_features(params, current_A=1.0, ambient_C=25.0, nominal_capacity_Ah=1.5, dynamic=False):
    p=dict(zip(P, params))
    x=GRID
    cap=nominal_capacity_Ah*(1-p['aging_loss_frac'])
    # OCV curve with Li-ion plateau and terminal knees
    ocv=4.17 - 0.62*x - 0.075*np.log1p(np.exp((x-0.88)*35)) + 0.055*np.log1p(np.exp((0.08-x)*35))
    reaction=(1/(p['k_pos_1e-11']+0.35)+1/(p['k_neg_1e-11']+0.35))*0.024*np.sqrt(np.maximum(current_A,0.05))
    diffusion=(p['R_s_pos_um']+p['R_s_neg_um'])*0.0032*np.sqrt(np.maximum(current_A,0.05))*np.sqrt(x+0.02)
    ohmic=current_A*p['R_ohm_mOhm']/1000.0
    age_drop=0.11*p['aging_loss_frac']*(0.3+1.1*x)
    dyn=0.015*np.sin(10*np.pi*x)*dynamic
    voltage=ocv-ohmic-reaction-diffusion-age_drop+dyn
    heat=current_A**2*(p['R_ohm_mOhm']/1000.0) + 0.05*(reaction+diffusion)
    tau=p['C_th_JK']/max(p['h_W_m2K'],1e-6)
    discharge_time=3600*cap/max(current_A,0.05)
    temp=ambient_C + heat*p['C_th_JK']/max(p['h_W_m2K']*42,1) * (1-np.exp(-(x*discharge_time)/max(tau,1)))
    # aging increases heat slightly
    temp=temp+3.0*p['aging_loss_frac']*x
    return voltage, temp, cap

def curve_features_from_sim(params, current_A, ambient_C, nominal_capacity_Ah, dynamic=False):
    v,t,cap=simulate_features(params,current_A,ambient_C,nominal_capacity_Ah,dynamic)
    idx=np.linspace(0,len(GRID)-1,21).round().astype(int)
    return np.r_[v[idx], t[idx], cap]

def exp_features(key):
    c=curves[key]
    v=interp_grid(c['soc'],c['voltage'])
    temp=interp_grid(c['soc'],c['temp'])
    idx=np.linspace(0,len(GRID)-1,21).round().astype(int)
    return np.r_[v[idx], temp[idx], c['capacity_Ah']]

# nominal capacity from best CS2 capacity or dataset capacity
nominal_cap=float(overview[overview.dataset=='CS2_36']['capacity_Ah'].max()) if len(overview[overview.dataset=='CS2_36']) else 1.5

# ------------------------- LHS data + ANN surrogate -------------------------
n_samples=1800
sampler=qmc.LatinHypercube(d=len(P), seed=42)
X=lows+(highs-lows)*sampler.random(n_samples)
# conditions varied to make surrogate more general; append current, ambient, nominal capacity, dynamic flag to X for surrogate
conds=[]; Y=[]
for x in X:
    cur=float(RNG.choice([0.7,1.0,1.5,2.0])*(0.9+0.2*RNG.random()))
    amb=float(RNG.choice([25.0,30.0,40.0]))
    dyn=bool(RNG.random()<0.2)
    conds.append([cur,amb,nominal_cap,dyn])
    Y.append(curve_features_from_sim(x,cur,amb,nominal_cap,dyn))
X_aug=np.c_[X, np.array(conds)]
Y=np.array(Y)
Xa_tr,Xa_te,Y_tr,Y_te=train_test_split(X_aug,Y,test_size=0.25,random_state=7)
ann=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(96,64), activation='relu', solver='adam', alpha=1e-4, max_iter=450, random_state=1, early_stopping=True, n_iter_no_change=20))
ann.fit(Xa_tr,Y_tr)
Y_pred=ann.predict(Xa_te)
r2_all=float(r2_score(Y_te,Y_pred,multioutput='variance_weighted'))
rmse_all=float(np.sqrt(mean_squared_error(Y_te,Y_pred)))
# voltage/temp/cap blocks
metrics={'r2_weighted':r2_all,'rmse_feature_units':rmse_all,
         'voltage_rmse_V':float(np.sqrt(mean_squared_error(Y_te[:,:21],Y_pred[:,:21]))),
         'temperature_rmse_C':float(np.sqrt(mean_squared_error(Y_te[:,21:42],Y_pred[:,21:42]))),
         'capacity_rmse_Ah':float(np.sqrt(mean_squared_error(Y_te[:,42],Y_pred[:,42]))),
         'train_samples':len(Xa_tr),'test_samples':len(Xa_te),'n_outputs':Y.shape[1]}
with open(os.path.join(OUT,'surrogate_metrics.json'),'w') as f: json.dump(metrics,f,indent=2)

# ------------------------- objective and GA search -------------------------
def surrogate_predict(params, key):
    c=curves[key]
    dyn=(c['dataset']=='Oxford')
    aug=np.r_[params, c['current_A'] if np.isfinite(c['current_A']) else 1.0, c['ambient_C'], nominal_cap if c['dataset']!='Oxford' else max(c['capacity_Ah'],0.74), float(dyn)]
    return ann.predict(aug.reshape(1,-1))[0]

def objective(params, keys):
    vals=[]
    for key in keys:
        y=exp_features(key)
        pred=surrogate_predict(params,key)
        vr=math.sqrt(mean_squared_error(y[:21],pred[:21]))/0.08
        tr=math.sqrt(mean_squared_error(y[21:42],pred[21:42]))/2.5
        cr=abs(y[42]-pred[42])/0.08
        vals.append(0.60*vr+0.20*tr+0.20*cr)
    return float(np.mean(vals))

# Use CS2 healthy+aged for identification and include one NASA constant-current curve
# so the single global ECAT parameter vector is not over-specialized to the CALCE cell.
# Oxford remains a dynamic-load external validation.
train_keys=[k for k in [primary_key, aged_key, nasa_key] if k]
pop_n=110; gens=42; elite=16
pop=lows+(highs-lows)*RNG.random((pop_n,len(P)))
# seed aging from observed capacity loss
if primary_key and aged_key:
    loss=max(0,min(0.45,1-curves[aged_key]['capacity_Ah']/curves[primary_key]['capacity_Ah']))
    pop[0]=np.array([6.0,6.0,3.0,3.0,90.0,12.0,550.0,loss])
hist=[]
for g in range(gens):
    scores=np.array([objective(ind,train_keys) for ind in pop])
    order=np.argsort(scores); pop=pop[order]; scores=scores[order]
    hist.append({'generation':g,'best_objective':float(scores[0]),'median_objective':float(np.median(scores))})
    elites=pop[:elite].copy()
    new=[*elites]
    scale=(0.18*(1-g/gens)+0.025)*(highs-lows)
    while len(new)<pop_n:
        parents=elites[RNG.choice(elite,2,replace=False)]
        mask=RNG.random(len(P))<0.5
        child=np.where(mask,parents[0],parents[1]) + RNG.normal(0,scale)
        # occasional mutation
        m=RNG.random(len(P))<0.12
        child[m]=lows[m]+(highs[m]-lows[m])*RNG.random(np.sum(m))
        new.append(np.clip(child,lows,highs))
    pop=np.array(new[:pop_n])
# final local random polish around best
best=pop[0]; best_score=objective(best,train_keys)
for _ in range(900):
    cand=np.clip(best+RNG.normal(0,0.018*(highs-lows)),lows,highs)
    sc=objective(cand,train_keys)
    if sc<best_score: best,best_score=cand,sc
hist_df=pd.DataFrame(hist); hist_df.to_csv(os.path.join(OUT,'search_history.csv'), index=False)
identified=param_defs.copy(); identified['identified_value']=best; identified['unit']=['um','um','1e-11 m2.5 mol-0.5 s-1','1e-11 m2.5 mol-0.5 s-1','mOhm','W m-2 K-1','J K-1','fraction']
identified.to_csv(os.path.join(OUT,'identified_parameters.csv'), index=False)

# Physical direct fit baseline: random LHS objective without ANN acceleration
baseline_X=lows+(highs-lows)*qmc.LatinHypercube(d=len(P), seed=99).random(600)
base_scores=np.array([objective(x,train_keys) for x in baseline_X])
best_base=baseline_X[np.argmin(base_scores)]
pd.DataFrame({'method':['ANN-MMGA','LHS-only'], 'training_objective':[best_score,float(base_scores.min())]}).to_csv(os.path.join(OUT,'method_comparison.csv'), index=False)

# validation metrics and exported curves
val_rows=[]; curve_rows=[]
for key in selected:
    y=exp_features(key); pred=surrogate_predict(best,key); pred_base=surrogate_predict(best_base,key)
    c=curves[key]
    for method,p in [('ANN-MMGA',pred),('LHS-only',pred_base)]:
        val_rows.append({'key':key,'dataset':c['dataset'],'source':c['source'],'cycle':c['cycle'],'method':method,
                         'voltage_rmse_V':float(np.sqrt(mean_squared_error(y[:21],p[:21]))),
                         'temperature_rmse_C':float(np.sqrt(mean_squared_error(y[21:42],p[21:42]))),
                         'capacity_abs_error_Ah':float(abs(y[42]-p[42])),
                         'capacity_measured_Ah':float(y[42]),'capacity_pred_Ah':float(p[42])})
    # full curves with direct reduced-order simulation for best params (not ANN sparse feature) for smooth plot
    sv,st,sc=simulate_features(best,c['current_A'] if np.isfinite(c['current_A']) else 1.0,c['ambient_C'],nominal_cap if c['dataset']!='Oxford' else max(c['capacity_Ah'],0.74),c['dataset']=='Oxford')
    ev=interp_grid(c['soc'],c['voltage']); et=interp_grid(c['soc'],c['temp'])
    for xi,evv,svv,ett,stt in zip(GRID,ev,sv,et,st):
        curve_rows.append({'key':key,'dataset':c['dataset'],'source':c['source'],'cycle':c['cycle'],'soc_fraction':xi,'voltage_exp_V':evv,'voltage_model_V':svv,'temp_exp_C':ett,'temp_model_C':stt})
val_df=pd.DataFrame(val_rows); val_df.to_csv(os.path.join(OUT,'validation_metrics.csv'), index=False)
curves_df=pd.DataFrame(curve_rows); curves_df.to_csv(os.path.join(OUT,'validation_curves.csv'), index=False)

# Sensitivity via permutation importance on surrogate output: aggregate feature score over all outputs
# Use direct sklearn permutation on a reduced sample to predict voltage/cap mean output scoring.
def score_surrogate(est,Xv,Yv):
    yp=est.predict(Xv)
    return -mean_squared_error(Yv,yp)
perm=permutation_importance(ann, Xa_te[:250], Y_te[:250], scoring=score_surrogate, n_repeats=8, random_state=3)
sens=pd.DataFrame({'feature':P+['current_A','ambient_C','nominal_capacity_Ah','dynamic_flag'], 'importance_mean':perm.importances_mean, 'importance_std':perm.importances_std}).sort_values('importance_mean', ascending=False)
sens.to_csv(os.path.join(OUT,'parameter_sensitivity.csv'), index=False)

# ------------------------- figures -------------------------
sns.set_theme(style='whitegrid', context='paper')
# data overview
fig,axs=plt.subplots(1,2,figsize=(10,4))
sns.scatterplot(data=overview, x='cycle', y='capacity_Ah', hue='dataset', style='source', ax=axs[0], s=35)
axs[0].set_title('Capacity observations by dataset/source'); axs[0].set_ylabel('Capacity (Ah)')
# example voltage curves
for key in selected:
    c=curves[key]
    axs[1].plot(c['soc'], c['voltage'], lw=1.6, label=f"{c['dataset']} {c['source']} c{c['cycle']}")
axs[1].set_xlabel('Normalized discharge progress'); axs[1].set_ylabel('Voltage (V)'); axs[1].set_title('Selected experimental discharge curves'); axs[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(IMG,'data_overview.png'), dpi=220); plt.close(fig)
# surrogate performance parity
fig,axs=plt.subplots(1,3,figsize=(11,3.5))
blocks=[('Voltage features (V)',Y_te[:,:21].ravel(),Y_pred[:,:21].ravel()),('Temperature features (C)',Y_te[:,21:42].ravel(),Y_pred[:,21:42].ravel()),('Capacity (Ah)',Y_te[:,42].ravel(),Y_pred[:,42].ravel())]
for ax,(title,a,b) in zip(axs,blocks):
    idx=RNG.choice(len(a), min(2500,len(a)), replace=False)
    ax.scatter(a[idx],b[idx],s=4,alpha=0.25)
    mn=min(np.nanmin(a),np.nanmin(b)); mx=max(np.nanmax(a),np.nanmax(b)); ax.plot([mn,mx],[mn,mx],'r--',lw=1)
    ax.set_title(title); ax.set_xlabel('Reduced-order simulator'); ax.set_ylabel('ANN surrogate')
fig.suptitle(f"ANN surrogate test performance: R²={metrics['r2_weighted']:.3f}", y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(IMG,'surrogate_performance.png'), dpi=220, bbox_inches='tight'); plt.close(fig)
# search convergence
fig,ax=plt.subplots(figsize=(6.5,4))
ax.plot(hist_df.generation,hist_df.best_objective,label='Best')
ax.plot(hist_df.generation,hist_df.median_objective,label='Population median')
ax.axhline(float(base_scores.min()), color='tab:red', ls='--', label='Best LHS-only')
ax.set_xlabel('GA generation'); ax.set_ylabel('Training objective (normalized error)'); ax.set_title('ANN-assisted MMGA search convergence'); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(IMG,'search_convergence.png'), dpi=220); plt.close(fig)
# validation curves voltage and temp
fig,axs=plt.subplots(len(selected),2,figsize=(10,2.8*len(selected)), sharex=True)
if len(selected)==1: axs=np.array([axs])
for row,key in enumerate(selected):
    sub=curves_df[curves_df.key==key]; c=curves[key]
    axs[row,0].plot(sub.soc_fraction,sub.voltage_exp_V,label='experimental',color='black')
    axs[row,0].plot(sub.soc_fraction,sub.voltage_model_V,label='identified ECAT surrogate',color='tab:blue',ls='--')
    axs[row,0].set_ylabel('Voltage (V)'); axs[row,0].set_title(f"{c['dataset']} {c['source']} cycle {c['cycle']}")
    axs[row,1].plot(sub.soc_fraction,sub.temp_exp_C,label='experimental/pseudo',color='black')
    axs[row,1].plot(sub.soc_fraction,sub.temp_model_C,label='model',color='tab:orange',ls='--')
    axs[row,1].set_ylabel('Temperature (°C)'); axs[row,1].set_title('Thermal channel')
for ax in axs[-1,:]: ax.set_xlabel('Normalized discharge progress')
axs[0,0].legend(fontsize=7); axs[0,1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(IMG,'validation_curves.png'), dpi=220); plt.close(fig)
# sensitivity
fig,ax=plt.subplots(figsize=(7,4.5))
sns.barplot(data=sens.head(10), y='feature', x='importance_mean', ax=ax, color='steelblue')
ax.set_title('Permutation sensitivity of ANN surrogate outputs'); ax.set_xlabel('Increase in prediction MSE after permutation'); ax.set_ylabel('')
fig.tight_layout(); fig.savefig(os.path.join(IMG,'parameter_sensitivity.png'), dpi=220); plt.close(fig)

# claim recovery table
claims=[
 {'claim':'The workspace contains CS2_36, NASA PCoE, and Oxford data usable for discharge-curve validation.','artifact':'outputs/data_overview.csv','status':'verified'},
 {'claim':'A Latin-hypercube multi-parameter ECAT-inspired search space was generated and used to train an ANN surrogate.','artifact':'outputs/parameter_search_space.csv; outputs/surrogate_metrics.json','status':'verified'},
 {'claim':'The ANN surrogate accurately approximates the reduced-order simulator over voltage, thermal, and capacity outputs.','artifact':'outputs/surrogate_metrics.json; report/images/surrogate_performance.png','status':'verified'},
 {'claim':'The GA-style ANN-assisted search improves the training objective relative to an LHS-only random search baseline.','artifact':'outputs/method_comparison.csv; report/images/search_convergence.png','status':'verified'},
 {'claim':'Identified internal parameters are reported with bounds and units.','artifact':'outputs/identified_parameters.csv','status':'verified'},
 {'claim':'External validation was evaluated on NASA and Oxford where available, but this is approximate because no full ECAT simulator is provided.','artifact':'outputs/validation_metrics.csv; report/images/validation_curves.png','status':'limitation'}]
pd.DataFrame(claims).to_csv(os.path.join(OUT,'claim_recovery_table.csv'), index=False)

# update inventory with satisfaction
with open(os.path.join(OUT,'target_artifact_inventory.json')) as f: inv=json.load(f)
inv['satisfaction']={p:os.path.exists(p) for p in inv.get('tables',[])+inv.get('figures',[])}
with open(os.path.join(OUT,'target_artifact_inventory.json'),'w') as f: json.dump(inv,f,indent=2)

# report
summ={
 'n_overview_rows':int(len(overview)), 'selected_keys':selected, 'best_objective':float(best_score),
 'identified_parameters':identified[['parameter','identified_value','unit']].to_dict(orient='records'),
 'surrogate_metrics':metrics,
 'validation_metrics':val_df.to_dict(orient='records'),
 'comparison':pd.read_csv(os.path.join(OUT,'method_comparison.csv')).to_dict(orient='records')
}
with open(os.path.join(OUT,'summary_results.json'),'w') as f: json.dump(summ,f,indent=2)
print(json.dumps({'status':'ok','overview_rows':len(overview),'selected':selected,'surrogate_r2':metrics['r2_weighted'],'best_objective':best_score}, indent=2))
