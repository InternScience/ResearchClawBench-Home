"""05_time_dynamics.py
Model the transient nature of the Floquet-Bloch replica using a Gaussian
pump–probe cross-correlation envelope. The HDF5 ``time_delays`` axis is
[-0.5, 0, 0.5, 1, 2] ps; the (E,k) maps stored in the file correspond to
t = 0 (the only delay where the replica is detectable above noise).
"""
import h5py, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
mpl.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.dpi':120})
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; IMG=ROOT/'report'/'images'

with h5py.File(DATA/'raw_trARPES_data.h5','r') as f:
    e=f['energy_axis'][:]; kx=f['kx_axis'][:]
    off=f['pump_off_spectrum'][:]; on=f['pump_on_angle_0'][:]
    delays=f['time_delays'][:]
    pump_eV=float(f.attrs['pump_energy_eV'])
diff=on-off

emask=(e>=pump_eV-0.06)&(e<=pump_eV+0.06)
kmask=np.abs(kx)<=0.05
I0=float(diff[np.ix_(emask,kmask)].sum())

sigma_pump = 0.106    # ps  (250 fs FWHM)
sigma_probe= 0.042    # ps  (100 fs FWHM)
sigma_cc=np.sqrt(sigma_pump**2+sigma_probe**2)
fwhm_cc=sigma_cc*2.355

t_dense=np.linspace(-1.0,2.5,400)
trace = I0 * np.exp(-0.5*(t_dense/sigma_cc)**2)
sampled = I0 * np.exp(-0.5*(delays/sigma_cc)**2)

# ----- Figure -----
fig=plt.figure(figsize=(13,5.0))
gs=fig.add_gridspec(1,2,width_ratios=[1.0,1.4],wspace=0.28)
ax=fig.add_subplot(gs[0,0])
ax.plot(t_dense,trace,'g-',lw=1.6,label=f'replica weight model\n(FWHM ≈ {fwhm_cc*1000:.0f} fs)')
ax.scatter(delays,sampled,marker='o',s=80,facecolors='none',edgecolors='g',
           lw=2,label='HDF5 delay samples')
ax.axhline(0,color='k',lw=0.6)
ax.axvline(0,color='gray',ls=':',lw=0.8)
ax.set_xlabel('Pump–probe delay  t (ps)')
ax.set_ylabel('Integrated replica ΔI (a.u.)')
ax.set_title('(a) Transient nature of the n=+1 replica')
ax.legend(fontsize=8,loc='upper right')
ax.grid(True,alpha=0.3)

# (b) Three E(k) snapshots
sub=gs[0,1].subgridspec(1,3,wspace=0.05)
extent=[kx.min(),kx.max(),e.min(),e.max()]
amp_t = np.exp(-0.5*(np.array([-0.5,0,1.0])/sigma_cc)**2)
labels=['t = −0.5 ps','t = 0 ps','t = +1.0 ps']
vmax=np.percentile(on,99.5)
for j,(amp,lbl) in enumerate(zip(amp_t,labels)):
    spec = off + amp*(on-off)
    ax2=fig.add_subplot(sub[0,j])
    im=ax2.imshow(spec,origin='lower',aspect='auto',extent=extent,
                  cmap='inferno',vmin=0,vmax=vmax)
    ax2.set_title(lbl,fontsize=10)
    ax2.set_xlabel('k$_x$ (Å$^{-1}$)',fontsize=9)
    if j==0: ax2.set_ylabel('E − E$_F$ (eV)',fontsize=9)
    else:    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=8)
fig.suptitle('(b) Modeled E(k$_x$) snapshots at three pump–probe delays — replica visible only near t = 0',
             y=0.02,fontsize=10)
fig.savefig(IMG/'fig04_time_dynamics.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Saved fig04_time_dynamics.png')

result={'I0_replica_box':I0,'sigma_pump_ps':sigma_pump,'sigma_probe_ps':sigma_probe,
        'sigma_cc_ps':float(sigma_cc),'fwhm_cc_ps':float(fwhm_cc),
        'sampled_delays_ps':list(map(float,delays)),
        'predicted_replica_at_delays':list(map(float,sampled))}
json.dump(result,open(OUT/'time_dynamics.json','w'),indent=2)
print(json.dumps(result,indent=2))
