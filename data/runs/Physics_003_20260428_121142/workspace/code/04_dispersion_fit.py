"""04_dispersion_fit.py
Linear fit of the equilibrium Dirac dispersion |E - E_D| = ℏv_F |k|
and verification that the n = ±1 replica branches follow the same
dispersion shifted vertically by ±ℏω.
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
    pump_eV=float(f.attrs['pump_energy_eV'])
diff=on-off

def two_branches(arr):
    pos=np.zeros(len(e)); neg=np.zeros(len(e))
    for i in range(len(e)):
        line=arr[i]; bg=np.median(line); s=line-bg
        l=np.where(kx<0)[0]; r=np.where(kx>0)[0]
        neg[i]=kx[l][np.argmax(s[l])]
        pos[i]=kx[r][np.argmax(s[r])]
    return pos,neg

pos_off,neg_off=two_branches(off)
pos_diff,neg_diff=two_branches(diff)

# ---- Equilibrium cone fit ----
mask_up=(e>0.05); mask_lo=(e<-0.05)
m_up_pos,b_up_pos=np.polyfit(e[mask_up], pos_off[mask_up],1)
m_up_neg,b_up_neg=np.polyfit(e[mask_up], -neg_off[mask_up],1)
m_lo_pos,b_lo_pos=np.polyfit(e[mask_lo], pos_off[mask_lo],1)
m_lo_neg,b_lo_neg=np.polyfit(e[mask_lo], -neg_off[mask_lo],1)
hvF_branches=[1.0/abs(m) for m in (m_up_pos,m_up_neg,m_lo_pos,m_lo_neg)]
hvF_avg=float(np.mean(hvF_branches))
E_D = float(np.mean([-b_up_pos/m_up_pos,-b_up_neg/m_up_neg,
                     -b_lo_pos/m_lo_pos,-b_lo_neg/m_lo_neg]))
vF_ms = hvF_avg*1.602176634e-19/1.054571817e-34*1e-10
print(f'avg ℏv_F = {hvF_avg:.3f} eV·Å,  v_F = {vF_ms:.3e} m/s,  E_D = {E_D:.4f} eV')

# ---- Replica branches (only the wings of each replica cone) ----
def fitw(mask,arr):
    if mask.sum()<4: return None,None
    m,b=np.polyfit(e[mask],arr[mask],1); return float(m),float(b)
mp_up=(e>pump_eV+0.10) & (e<0.50)
mp_lo=(e<pump_eV-0.10) & (e>0.08)
mn_up=(e>-pump_eV+0.10) & (e<-0.08)
mn_lo=(e<-pump_eV-0.10) & (e>-0.50)

m_p_up,b_p_up=fitw(mp_up,pos_diff)
m_p_lo,b_p_lo=fitw(mp_lo,pos_diff)
m_n_up,b_n_up=fitw(mn_up,pos_diff)
m_n_lo,b_n_lo=fitw(mn_lo,pos_diff)

hvF_rp=float(np.mean([1/abs(m_p_up),1/abs(m_p_lo)]))
hvF_rn=float(np.mean([1/abs(m_n_up),1/abs(m_n_lo)]))
E_D_rp=float(np.mean([-b_p_up/m_p_up,-b_p_lo/m_p_lo]))
E_D_rn=float(np.mean([-b_n_up/m_n_up,-b_n_lo/m_n_lo]))
print(f'n=+1 replica: ℏv_F={hvF_rp:.3f} eV·Å, vertex E={E_D_rp:.4f} eV (expect ≈ {E_D+pump_eV:.4f})')
print(f'n=-1 replica: ℏv_F={hvF_rn:.3f} eV·Å, vertex E={E_D_rn:.4f} eV (expect ≈ {E_D-pump_eV:.4f})')

# ----- Plot -----
fig,axes=plt.subplots(1,2,figsize=(13,5.4))
extent=[kx.min(),kx.max(),e.min(),e.max()]

ax=axes[0]
ax.imshow(off,origin='lower',aspect='auto',extent=extent,cmap='inferno',
          vmin=0,vmax=np.percentile(off,99.5))
kgrid=np.linspace(0,kx.max(),50)
ax.plot( kgrid, E_D + hvF_avg*kgrid,'c-',lw=1.5,label=f'cone fit, ℏv$_F$={hvF_avg:.2f} eV·Å')
ax.plot( kgrid, E_D - hvF_avg*kgrid,'c-',lw=1.5)
ax.plot(-kgrid, E_D + hvF_avg*kgrid,'c-',lw=1.5)
ax.plot(-kgrid, E_D - hvF_avg*kgrid,'c-',lw=1.5)
for sign,col in [(+1,'lime'),(-1,'magenta')]:
    Es=E_D+sign*pump_eV
    ax.plot( kgrid, Es+hvF_avg*kgrid,col,ls='--',lw=1.0)
    ax.plot( kgrid, Es-hvF_avg*kgrid,col,ls='--',lw=1.0)
    ax.plot(-kgrid, Es+hvF_avg*kgrid,col,ls='--',lw=1.0)
    ax.plot(-kgrid, Es-hvF_avg*kgrid,col,ls='--',lw=1.0)
ax.plot([],[],'lime',ls='--',label='n=+1 replica = cone+ℏω')
ax.plot([],[],'magenta',ls='--',label='n=−1 replica = cone−ℏω')
ax.scatter(pos_off,e,c='w',s=2,alpha=0.4)
ax.scatter(neg_off,e,c='w',s=2,alpha=0.4)
ax.set_xlim(kx.min(),kx.max()); ax.set_ylim(e.min(),e.max())
ax.set_xlabel('k$_x$ (Å$^{-1}$)'); ax.set_ylabel('E − E$_F$ (eV)')
ax.set_title(f'(a) Pump-off cone:  v$_F$={vF_ms/1e6:.2f}×10$^6$ m/s,  E$_D$={E_D*1000:+.1f} meV')
ax.legend(loc='upper right',fontsize=8)

ax=axes[1]
vlim=np.max(np.abs(diff))
ax.imshow(diff,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r',vmin=-vlim,vmax=vlim)
for sign,col,lbl in [(+1,'lime','n=+1 replica fit'),(-1,'magenta','n=−1 replica fit')]:
    hvF_=hvF_rp if sign==1 else hvF_rn
    Es =E_D_rp if sign==1 else E_D_rn
    ax.plot( kgrid, Es+hvF_*kgrid,col,ls='--',lw=1.6,label=lbl)
    ax.plot(-kgrid, Es+hvF_*kgrid,col,ls='--',lw=1.6)
    ax.plot( kgrid, Es-hvF_*kgrid,col,ls='--',lw=1.6)
    ax.plot(-kgrid, Es-hvF_*kgrid,col,ls='--',lw=1.6)
# wing data points used for the fit
for mask in (mp_up,mp_lo,mn_up,mn_lo):
    ax.scatter(pos_diff[mask],e[mask],c='k',s=8,alpha=0.7)
    ax.scatter(neg_diff[mask],e[mask],c='k',s=8,alpha=0.7)
ax.set_xlabel('k$_x$ (Å$^{-1}$)'); ax.set_ylabel('E − E$_F$ (eV)')
ax.set_xlim(kx.min(),kx.max()); ax.set_ylim(e.min(),e.max())
ax.set_title('(b) ΔI(E,k$_x$) overlaid with replica-cone fits')
ax.legend(loc='upper right',fontsize=8)
fig.tight_layout()
fig.savefig(IMG/'fig03_dispersion_fit.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print('Saved fig03_dispersion_fit.png')

result={'hvF_branches_eV_A':list(map(float,hvF_branches)),
        'hvF_avg_eV_A':float(hvF_avg),'vF_m_per_s':float(vF_ms),
        'E_D_eV':float(E_D),'pump_eV':pump_eV,
        'replica_plus_hvF_eV_A':float(hvF_rp),
        'replica_plus_vertex_eV':float(E_D_rp),
        'replica_plus_vertex_minus_hw_eV':float(E_D_rp-pump_eV),
        'replica_minus_hvF_eV_A':float(hvF_rn),
        'replica_minus_vertex_eV':float(E_D_rn),
        'replica_minus_vertex_plus_hw_eV':float(E_D_rn+pump_eV),
        'literature_vF_m_per_s_graphene':1.0e6}
json.dump(result,open(OUT/'dispersion_fit.json','w'),indent=2)
print(json.dumps(result,indent=2))
