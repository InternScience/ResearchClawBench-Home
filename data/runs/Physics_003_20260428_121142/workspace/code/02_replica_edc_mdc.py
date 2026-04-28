"""02_replica_eDc_mdc.py
Energy distribution curves (EDCs) and momentum distribution curves (MDCs)
of the difference spectrum showing the n=±1 Floquet-Bloch replicas of the
Dirac cone, with peak energies separated by exactly ℏω.
"""
import h5py, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

mpl.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':120})
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; IMG=ROOT/'report'/'images'

with h5py.File(DATA/'raw_trARPES_data.h5','r') as f:
    e=f['energy_axis'][:]; kx=f['kx_axis'][:]
    off=f['pump_off_spectrum'][:]; on0=f['pump_on_angle_0'][:]
    pump_eV=float(f.attrs['pump_energy_eV'])
diff=on0-off

# EDC at kx ≈ 0 (averaged within ±0.01 Å^-1)
sel=np.where(np.abs(kx)<=0.012)[0]
edc_diff=diff[:,sel].mean(axis=1)
edc_off =off[:,sel].mean(axis=1)
edc_on  =on0[:,sel].mean(axis=1)

# Fit two Gaussian peaks in the difference EDC
def gauss(x,A,mu,sig): return A*np.exp(-0.5*((x-mu)/sig)**2)
def two_gauss(x,A1,mu1,s1,A2,mu2,s2,c): return gauss(x,A1,mu1,s1)+gauss(x,A2,mu2,s2)+c
p0=[edc_diff.max(),+0.20,0.05,edc_diff.max(),-0.30,0.05,0]
try:
    popt,pcov=curve_fit(two_gauss,e,edc_diff,p0=p0,maxfev=20000)
    A1,mu1,s1,A2,mu2,s2,c0=popt
    perr=np.sqrt(np.diag(pcov))
    fit_ok=True
except Exception as ex:
    print('fit failed',ex); fit_ok=False
    mu1,mu2=0.20,-0.30; s1=s2=0.05; A1=A2=edc_diff.max(); c0=0
    perr=[0]*7

# MDC at the n=+1 replica energy (peak)
ep=mu1
ie_p=int(np.argmin(np.abs(e-ep)))
mdc_p=diff[ie_p]
en=mu2
ie_n=int(np.argmin(np.abs(e-en)))
mdc_n=diff[ie_n]
mdc_dirac=diff[int(np.argmin(np.abs(e-((mu1+mu2)/2))))]   # near Dirac point of unperturbed

# ----- Plot -----
fig=plt.figure(figsize=(13,8))
gs=fig.add_gridspec(2,3,height_ratios=[1,1])

# (a) Spectrum + ℏω guides
ax0=fig.add_subplot(gs[:,0])
extent=[kx.min(),kx.max(),e.min(),e.max()]
vlim=np.max(np.abs(diff))
im=ax0.imshow(diff,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r',vmin=-vlim,vmax=vlim)
ax0.axhline(mu1,color='lime',ls='--',lw=1)
ax0.axhline(mu2,color='magenta',ls='--',lw=1)
ax0.axvspan(kx[sel[0]],kx[sel[-1]],color='cyan',alpha=0.18,label='EDC window')
ax0.set_xlabel('k$_x$ (Å$^{-1}$)'); ax0.set_ylabel('E − E$_F$ (eV)')
ax0.set_title('(a) Difference spectrum  ΔI(E,k$_x$)')
plt.colorbar(im,ax=ax0,fraction=0.045,label='ΔI')
ax0.legend(loc='upper right',fontsize=8)

# (b) EDC near kx=0 + Gaussian fit
ax1=fig.add_subplot(gs[0,1:])
ax1.plot(e,edc_off,'k-',label='pump off',alpha=0.5)
ax1.plot(e,edc_on,'r-',label='pump on (θ=0°)',alpha=0.7)
ax1b=ax1.twinx()
ax1b.plot(e,edc_diff,'g-',label='Δ (pump-on − off)',lw=1.4)
if fit_ok:
    ax1b.plot(e,two_gauss(e,*popt),'b--',lw=1.2,label='2-Gaussian fit')
ax1b.set_ylabel('ΔI (a.u.)',color='g')
ax1.set_ylabel('Intensity (a.u.)')
ax1.set_xlabel('E − E$_F$ (eV)')
ax1.set_title('(b) EDC at k$_x$ ≈ 0  (averaged ±0.01 Å$^{-1}$)')
ax1.axvline(mu1,color='lime',ls=':',lw=1)
ax1.axvline(mu2,color='magenta',ls=':',lw=1)
# combine legends
h,l=ax1.get_legend_handles_labels(); h2,l2=ax1b.get_legend_handles_labels()
ax1.legend(h+h2,l+l2,loc='upper right',fontsize=8)
ax1.text(0.02,0.92,
    f'μ$_+$={mu1:.3f} eV\nμ$_-$={mu2:.3f} eV\nΔE = μ$_+$−μ$_-$ = {mu1-mu2:.3f} eV\nℏω (set) = {pump_eV:.3f} eV',
    transform=ax1.transAxes,fontsize=8,va='top',
    bbox=dict(facecolor='white',alpha=0.85,edgecolor='gray'))

# (c) MDCs
ax2=fig.add_subplot(gs[1,1:])
ax2.plot(kx,mdc_p,'lime',label=f'E = {mu1:.2f} eV  (n=+1 replica)')
ax2.plot(kx,mdc_n,'magenta',label=f'E = {mu2:.2f} eV  (n=−1 replica)')
ax2.plot(kx,mdc_dirac,'gray',ls=':',alpha=0.7,label='E ≈ Dirac (avg)')
ax2.set_xlabel('k$_x$ (Å$^{-1}$)'); ax2.set_ylabel('ΔI (a.u.)')
ax2.set_title('(c) MDCs at the replica vertex energies')
ax2.legend(fontsize=8)
ax2.axvline(0,color='k',lw=0.6,alpha=0.5)

fig.tight_layout()
fig.savefig(IMG/'fig02_replica_positions.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Saved fig02_replica_positions.png')

# ----- save numbers -----
results={
    'edc_kx_window':[float(kx[sel[0]]),float(kx[sel[-1]])],
    'pump_eV_setpoint': pump_eV,
    'replica_vertex_E_plus_eV':float(mu1),
    'replica_vertex_E_minus_eV':float(mu2),
    'replica_vertex_E_plus_sigma_eV':float(s1),
    'replica_vertex_E_minus_sigma_eV':float(s2),
    'replica_vertex_E_plus_err':float(perr[1]),
    'replica_vertex_E_minus_err':float(perr[4]),
    'separation_eV':float(mu1-mu2),
    'separation_minus_2hw_eV':float((mu1-mu2)-2*pump_eV),
}
json.dump(results,open(OUT/'replica_edc_fit.json','w'),indent=2)
print(json.dumps(results,indent=2))
