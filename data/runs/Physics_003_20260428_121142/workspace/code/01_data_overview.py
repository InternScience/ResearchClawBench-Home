"""01_data_overview.py
Overview figures of the tr-ARPES data: pump-off, pump-on (θ_p=0°), difference;
overlay the equilibrium Dirac cone vertex (extracted from data) and the n=±1
replica Dirac vertices located near (kx≈0, E≈±ℏω).
"""
import h5py, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':120})

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; IMG=ROOT/'report'/'images'
OUT.mkdir(exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)

with h5py.File(DATA/'raw_trARPES_data.h5','r') as f:
    e=f['energy_axis'][:]; kx=f['kx_axis'][:]
    off=f['pump_off_spectrum'][:]; on0=f['pump_on_angle_0'][:]
    pump_eV=float(f.attrs['pump_energy_eV'])
    sample=str(f.attrs['sample'])

diff=on0-off

# ---- Locate Dirac vertex from pump-off data ----
# At each energy, MDC peak |kx| ; the Dirac vertex is where MDC peak ~ 0
def mdc_peak(E_idx, spec):
    line=spec[E_idx].copy()
    return kx[np.argmax(line)]
peak_kx=np.array([np.abs(mdc_peak(i,off)) for i in range(len(e))])
dirac_E_data = float(e[np.argmin(peak_kx)])  # ~0
# Fit v_F: |E - E_D| = ℏv_F * |k|
mask=np.abs(e-dirac_E_data)>0.05
slope,intercept=np.polyfit(np.abs(e[mask]-dirac_E_data), peak_kx[mask], 1)
# E = (1/slope) * |k| ; ℏv_F = 1/slope (eV·Å)
hvF = 1.0/slope  # eV·Å
vF = hvF/6.582119e-16 * 1e-10   # m/s   (1 Å·eV/ℏ = 1.519e15 m/s = ~1.519 e6 m/s when v_F=1 eV·Å·1e-10/6.582e-16)
# Recompute v_F in m/s:  v_F = ℏv_F[eV·Å] * (1 eV)/(ℏ) * 1e-10 m/Å
vF_ms = hvF * 1.602176634e-19 / 1.054571817e-34 * 1e-10
print(f'Dirac E (data) = {dirac_E_data:.3f} eV')
print(f'ℏv_F  = {hvF:.3f} eV·Å    →  v_F = {vF_ms:.3e} m/s')

# ---- Plot ----
fig,axes=plt.subplots(1,3,figsize=(14,4.4))
extent=[kx.min(),kx.max(),e.min(),e.max()]
v_off=np.percentile(off,99.5)
im0=axes[0].imshow(off,origin='lower',aspect='auto',extent=extent,
                  cmap='inferno',vmin=0,vmax=v_off)
axes[0].set_title('(a)  Pump-off  E(k$_x$)')
axes[0].set_xlabel('k$_x$ (Å$^{-1}$)'); axes[0].set_ylabel('E − E$_F$ (eV)')
plt.colorbar(im0,ax=axes[0],fraction=0.046,label='Intensity (a.u.)')

im1=axes[1].imshow(on0,origin='lower',aspect='auto',extent=extent,
                  cmap='inferno',vmin=0,vmax=np.percentile(on0,99.5))
axes[1].set_title('(b)  Pump-on  E(k$_x$),  θ$_p$ = 0°')
axes[1].set_xlabel('k$_x$ (Å$^{-1}$)')
plt.colorbar(im1,ax=axes[1],fraction=0.046,label='Intensity (a.u.)')

vlim=np.max(np.abs(diff))
im2=axes[2].imshow(diff,origin='lower',aspect='auto',extent=extent,
                  cmap='RdBu_r',vmin=-vlim,vmax=vlim)
axes[2].set_title('(c)  Δ = pump-on − pump-off')
axes[2].set_xlabel('k$_x$ (Å$^{-1}$)')
plt.colorbar(im2,ax=axes[2],fraction=0.046,label='ΔI (a.u.)')

# Overlay vertices
ks=np.linspace(kx.min(),kx.max(),200)
for ax,is_diff in zip(axes,[False,False,True]):
    # main Dirac cone
    ax.plot(ks, dirac_E_data+hvF*np.abs(ks),'c-',lw=0.8,alpha=0.7)
    ax.plot(ks, dirac_E_data-hvF*np.abs(ks),'c-',lw=0.8,alpha=0.7)
    ax.scatter([0],[dirac_E_data],marker='x',c='cyan',s=60,lw=2,label='n=0 Dirac vertex')
    # replica n=+1
    ax.plot(ks, dirac_E_data+pump_eV+hvF*np.abs(ks),'lime',ls='--',lw=0.8,alpha=0.8)
    ax.plot(ks, dirac_E_data+pump_eV-hvF*np.abs(ks),'lime',ls='--',lw=0.8,alpha=0.8)
    ax.scatter([0],[dirac_E_data+pump_eV],marker='o',facecolors='none',edgecolors='lime',s=80,lw=1.6,label='n=+1 replica')
    # replica n=-1
    ax.plot(ks, dirac_E_data-pump_eV+hvF*np.abs(ks),'magenta',ls='--',lw=0.8,alpha=0.8)
    ax.plot(ks, dirac_E_data-pump_eV-hvF*np.abs(ks),'magenta',ls='--',lw=0.8,alpha=0.8)
    ax.scatter([0],[dirac_E_data-pump_eV],marker='o',facecolors='none',edgecolors='magenta',s=80,lw=1.6,label='n=−1 replica')
    ax.set_xlim(kx.min(),kx.max()); ax.set_ylim(e.min(),e.max())
axes[0].legend(loc='upper right',fontsize=7,framealpha=0.85)

fig.suptitle(f'Floquet-Bloch states in {sample}  —  MIR pump 5 μm (ℏω = {pump_eV} eV)',y=1.02)
fig.tight_layout()
fig.savefig(IMG/'fig01_data_overview.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Saved fig01_data_overview.png')

# ---- save a corrected replica summary ----
pump_on_max=on0.max(); off_max=off.max()
results={
    'sample':sample,
    'pump_eV':pump_eV,
    'dirac_E_data_eV':dirac_E_data,
    'hvF_eV_Angstrom':hvF,
    'vF_m_per_s':vF_ms,
    'replica_vertices':[
        {'order':+1,'E_eV':dirac_E_data+pump_eV,'kx_inv_A':0.0},
        {'order':-1,'E_eV':dirac_E_data-pump_eV,'kx_inv_A':0.0},
    ],
}
json.dump(results,open(OUT/'dirac_and_replica_fit.json','w'),indent=2)
print(json.dumps(results,indent=2))
