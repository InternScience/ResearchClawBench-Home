"""03_polarization_analysis.py
Analyze pump-polarization-angle dependence of replica intensity.

Two complementary measurements:

  (i) Tabulated intensity I(θ_p) at a fixed (E,k) point near the replica
      band, taken from data/polarization_dependence_data.csv.

 (ii) Spatially integrated replica intensity computed from the seven
      pump_on_angle_*  E(kx) maps in raw_trARPES_data.h5, summed over a
      box around the n=+1 replica vertex (E ≈ +ℏω, |kx| ≤ 0.05 Å^-1).

The two data sets are fit to the harmonic models commonly used to
distinguish Floquet-Bloch (FB) from Volkov / LAPE channels:

  M0:     I(θ) = c                                (isotropic, pure FB)
  M2:     I(θ) = c + A2 cos(2(θ − φ2))            (cos² selection rule, Volkov)
  M4:     I(θ) = c + A4 cos(4(θ − φ4))            (FB↔Volkov interference)
  M2+M4:  I(θ) = c + A2 cos(2(θ−φ2)) + A4 cos(4(θ−φ4))

Selection between these models is reported via reduced χ² and AIC.
"""
import h5py, json, numpy as np, csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.optimize import curve_fit
mpl.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':120})

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; IMG=ROOT/'report'/'images'

# ---- Tabulated I(θ) at fixed E,k ----
rows=list(csv.DictReader(open(DATA/'polarization_dependence_data.csv')))
ang=np.array([float(r['angle_degrees']) for r in rows])
I_csv=np.array([float(r['intensity']) for r in rows])

# ---- Spatially integrated replica intensity per polarization angle ----
with h5py.File(DATA/'raw_trARPES_data.h5','r') as f:
    e=f['energy_axis'][:]; kx=f['kx_axis'][:]
    pump_eV=float(f.attrs['pump_energy_eV'])
    off=f['pump_off_spectrum'][:]
    pol_angles=list(f['polarization_angles'][:])
    on_maps={}
    for a in pol_angles:
        on_maps[int(a)]=f[f'pump_on_angle_{int(a)}'][:]

# replica box: ±50 meV around +ℏω, |kx|<0.05 Å^-1
e_lo,e_hi=pump_eV-0.06, pump_eV+0.06
kbox=0.05
emask=(e>=e_lo)&(e<=e_hi); kmask=np.abs(kx)<=kbox
def replica_integral(arr):
    return float(arr[np.ix_(emask,kmask)].sum())
diffs={a: on_maps[a]-off for a in pol_angles}
I_int=np.array([replica_integral(diffs[int(a)]) for a in ang])

# Normalize for comparison
I_int_n=I_int/I_int.mean()

# ---- Fit harmonic models ----
def m0(t,c): return c+0*t
def m2(t,c,A,phi): return c+A*np.cos(2*(t-phi))
def m4(t,c,A,phi): return c+A*np.cos(4*(t-phi))
def m24(t,c,A2,p2,A4,p4): return c+A2*np.cos(2*(t-p2))+A4*np.cos(4*(t-p4))

def aic(rss,n,k): return n*np.log(rss/n)+2*k

def fit_set(theta,y,label):
    t=np.deg2rad(theta)
    out={}
    for name,fn,k,p0 in [
        ('M0',m0,1,[y.mean()]),
        ('M2',m2,3,[y.mean(),(y.max()-y.min())/2,0]),
        ('M4',m4,3,[y.mean(),(y.max()-y.min())/2,0]),
        ('M2+M4',m24,5,[y.mean(),(y.max()-y.min())/4,0,(y.max()-y.min())/2,0]),
    ]:
        try:
            popt,pcov=curve_fit(fn,t,y,p0=p0,maxfev=20000)
            yhat=fn(t,*popt)
            rss=float(((y-yhat)**2).sum())
            out[name]={'params':popt.tolist(),'rss':rss,'n':len(y),'k':k,
                       'AIC':float(aic(max(rss,1e-30),len(y),k))}
        except Exception as ex:
            out[name]={'error':str(ex)}
    out['data_range']=[float(y.min()),float(y.max())]
    out['mean']=float(y.mean())
    print(f'-- {label} --')
    for k,v in out.items():
        print(' ',k,v)
    return out

fit_csv=fit_set(ang,I_csv,'Tabulated I(θ) at fixed (E,k)')
fit_int=fit_set(ang,I_int_n,'Integrated replica intensity ΔI_box(θ)')

# ---- Plot ----
fig,axes=plt.subplots(1,2,figsize=(13,5.4))
ths=np.linspace(0,180,361)
trad=np.deg2rad(ths)

ax=axes[0]
ax.plot(ang,I_csv,'ko',ms=8,label='data (CSV, fixed E,k)')
for nm,col in [('M2','C0'),('M4','C3'),('M2+M4','C2')]:
    p=fit_csv[nm]['params']
    if nm=='M2': y=m2(trad,*p)
    elif nm=='M4': y=m4(trad,*p)
    else: y=m24(trad,*p)
    aicv=fit_csv[nm]['AIC']
    ax.plot(ths,y,col,lw=1.4,label=f'{nm}  AIC={aicv:.1f}')
ax.set_xlabel('Pump polarization angle  θ$_p$ (deg)')
ax.set_ylabel('Replica intensity (a.u.)')
ax.set_title('(a)  Replica I(θ$_p$)  at fixed (E,k$_x$)')
ax.set_xticks(np.arange(0,181,30))
ax.legend(fontsize=8)
ax.grid(True,alpha=0.3)

ax=axes[1]
ax.plot(ang,I_int_n,'ks',ms=8,label='data (Δ-spectrum integral)')
for nm,col in [('M2','C0'),('M4','C3'),('M2+M4','C2')]:
    p=fit_int[nm]['params']
    if nm=='M2': y=m2(trad,*p)
    elif nm=='M4': y=m4(trad,*p)
    else: y=m24(trad,*p)
    aicv=fit_int[nm]['AIC']
    ax.plot(ths,y,col,lw=1.4,label=f'{nm}  AIC={aicv:.1f}')
ax.set_xlabel('Pump polarization angle  θ$_p$ (deg)')
ax.set_ylabel('Integrated ΔI in replica box (norm.)')
ax.set_title('(b)  ∫ΔI over n=+1 replica box  vs  θ$_p$')
ax.set_xticks(np.arange(0,181,30))
ax.legend(fontsize=8)
ax.grid(True,alpha=0.3)

fig.suptitle('Polarization-angle dependence of the Floquet-Bloch replica',y=1.02)
fig.tight_layout()
fig.savefig(IMG/'fig05_polarization.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Saved fig05_polarization.png')

# ---- save fit ----
out={'CSV_fixed_Ek':fit_csv,'box_integrated':fit_int,
     'I_integrated_per_angle':{int(a):float(v) for a,v in zip(ang,I_int)},
     'pump_eV':pump_eV,'replica_box_e_eV':[e_lo,e_hi],'replica_box_kx_inv_A':kbox}
json.dump(out,open(OUT/'polarization_fit.json','w'),indent=2,default=float)
print('Saved polarization_fit.json')
