import ast, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

ROOT=Path('.')
OUT=ROOT/'outputs'
IMG=ROOT/'report'/'images'
OUT.mkdir(exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)
text=(ROOT/'data'/'DESI_EDE_Repro_Data.txt').read_text()
# execute safe assignment-only file after stripping comments ok (contains literals only)
ns={}
exec(compile(text,'DESI_EDE_Repro_Data.txt','exec'),{},ns)
models={'ΛCDM':ns['lcdm_params'],'EDE':ns['ede_params'],'w0wa':ns['w0wa_params']}
rows=[]
for model,d in models.items():
    for p,(mean,sigma) in d.items():
        rows.append({'model':model,'parameter':p,'mean':mean,'sigma':sigma})
param=pd.DataFrame(rows)
param.to_csv(OUT/'parameter_constraints.csv',index=False)

# Direct distance residual tables
for name,key,obs in [('dvrd','desi_dvrd_points','DESI Δ(DV/rd)'),('fap','desi_fap_points','DESI ΔFAP'),('sne_mu','sne_mu_points','Union3 Δμ')]:
    df=pd.DataFrame(ns[key],columns=['z','value','error'])
    df['observable']=obs
    df.to_csv(OUT/f'distance_points_{name}.csv',index=False)

# shifts relative to LCDM for common params and late/early params separately
lcdm=models['ΛCDM']
shift_rows=[]
for m in ['EDE','w0wa']:
    for p,(mean,sig) in models[m].items():
        if p in lcdm:
            lmean,lsig=lcdm[p]
            delta=mean-lmean
            comb=math.sqrt(sig**2+lsig**2)
            shift_rows.append({'comparison':f'{m}-ΛCDM','parameter':p,'delta':delta,'combined_sigma':comb,'z_shift':delta/comb,'model_mean':mean,'lcdm_mean':lmean,'model_sigma':sig,'lcdm_sigma':lsig})
shift=pd.DataFrame(shift_rows)
shift.to_csv(OUT/'model_parameter_shifts.csv',index=False)

# Derived EDE summary, including a_c conversion and z_c; log10_ac is log10(a_c) in provided file
f_mean,f_sig=models['EDE']['f_EDE']; loga_mean,loga_sig=models['EDE']['log10_ac']
a_c=10**loga_mean
# propagate: sigma_a = ln(10)*a*sigma_log; zc=1/a-1, sigma_z = ln(10)*(1/a)*sigma_log approximately
ede_summary={
 'f_EDE_mean':f_mean,'f_EDE_sigma':f_sig,'f_EDE_detection_sigma_vs_zero':f_mean/f_sig,
 'log10_a_c_mean':loga_mean,'log10_a_c_sigma':loga_sig,
 'a_c_mean':a_c,'a_c_sigma_linearized':math.log(10)*a_c*loga_sig,
 'z_c_mean':1/a_c-1,'z_c_sigma_linearized':math.log(10)*(1/a_c)*loga_sig,
 'H0_EDE':models['EDE']['H0'][0],'H0_EDE_sigma':models['EDE']['H0'][1],
 'H0_LCDM':models['ΛCDM']['H0'][0],'H0_LCDM_sigma':models['ΛCDM']['H0'][1],
 'H0_w0wa':models['w0wa']['H0'][0],'H0_w0wa_sigma':models['w0wa']['H0'][1],
 'H0_shift_EDE_vs_LCDM_sigma':float(shift[(shift.comparison=='EDE-ΛCDM')&(shift.parameter=='H0')]['z_shift'].iloc[0]),
 'H0_shift_w0wa_vs_LCDM_sigma':float(shift[(shift.comparison=='w0wa-ΛCDM')&(shift.parameter=='H0')]['z_shift'].iloc[0])
}
(OUT/'ede_parameter_summary.json').write_text(json.dumps(ede_summary,indent=2))

# Context goodness-of-fit from related work/task; mark source clearly
fit_rows=[
 {'source':'Poulin et al. 2019 related_work/paper_000','dataset':'Planck+BAO+Pantheon+SH0ES','model':'ΛCDM','delta_chi2_vs_lcdm':0.0,'note':'Table I; original EDE Hubble-tension analysis, not DESI DR2'},
 {'source':'Poulin et al. 2019 related_work/paper_000','dataset':'Planck+BAO+Pantheon+SH0ES','model':'EDE n=2','delta_chi2_vs_lcdm':-9.5,'note':'Table I'},
 {'source':'Poulin et al. 2019 related_work/paper_000','dataset':'Planck+BAO+Pantheon+SH0ES','model':'EDE n=3','delta_chi2_vs_lcdm':-14.5,'note':'Table I'},
 {'source':'Poulin et al. 2019 related_work/paper_000','dataset':'Planck+BAO+Pantheon+SH0ES','model':'EDE n=∞','delta_chi2_vs_lcdm':-9.1,'note':'Table I'},
 {'source':'McDonough et al. 2023 related_work/paper_001','dataset':'Planck PR3 TTTEEE primary CMB','model':'EDE','delta_chi2_vs_lcdm':-4.1,'note':'Review Table 2; Planck-only Plik example'},
 {'source':'Poulin et al. 2025 related_work/paper_003','dataset':'P-ACT+DESI DR2+lensing+Pantheon+/SH0ES','model':'EDE with SH0ES','delta_chi2_vs_lcdm':-35.4,'note':'Abstract; includes SH0ES prior and profile-likelihood context'},
]
fit=pd.DataFrame(fit_rows); fit.to_csv(OUT/'goodness_of_fit_context.csv',index=False)

# Report-ready model summary table
wide=param.pivot(index='parameter',columns='model',values='mean')
wide_sig=param.pivot(index='parameter',columns='model',values='sigma')
report_rows=[]
for p in sorted(param.parameter.unique()):
    row={'parameter':p}
    for m in ['ΛCDM','EDE','w0wa']:
        sub=param[(param.model==m)&(param.parameter==p)]
        row[m]='' if sub.empty else f"{sub['mean'].iloc[0]:.5g} ± {sub['sigma'].iloc[0]:.2g}"
    report_rows.append(row)
pd.DataFrame(report_rows).to_csv(OUT/'report_parameter_table.csv',index=False)

sns.set_theme(style='whitegrid', context='paper')
# Figure 1: key cosmological params errorbars
key=['omega_m','H0','sigma8','ns','ombh2']
fig,axes=plt.subplots(1,len(key),figsize=(13,3.2),constrained_layout=True)
colors={'ΛCDM':'#4c78a8','EDE':'#f58518','w0wa':'#54a24b'}
for ax,p in zip(axes,key):
    sub=param[param.parameter==p]
    y=np.arange(len(sub))
    ax.errorbar(sub['mean'], y, xerr=sub['sigma'], fmt='o', capsize=3, color='black', ecolor='black')
    for yi,(_,r) in zip(y,sub.iterrows()): ax.scatter(r['mean'], yi, s=45, color=colors[r['model']], zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(sub['model'])
    ax.set_title(p)
    ax.axvline(param[(param.model=='ΛCDM')&(param.parameter==p)]['mean'].iloc[0], color=colors['ΛCDM'], ls='--', lw=1)
fig.suptitle('CMB+DESI summary constraints: model-dependent parameter shifts')
fig.savefig(IMG/'parameter_constraints.png',dpi=220)
plt.close(fig)

# Figure 2: EDE posterior approximation independent Gaussians
fig,axes=plt.subplots(1,2,figsize=(8,3.2),constrained_layout=True)
xs=np.linspace(max(0,f_mean-4*f_sig),f_mean+4*f_sig,400)
axes[0].plot(xs,norm.pdf(xs,f_mean,f_sig),color=colors['EDE'])
axes[0].axvline(0,color='k',lw=1,ls=':'); axes[0].axvline(f_mean,color=colors['EDE'],ls='--')
axes[0].fill_between(xs,norm.pdf(xs,f_mean,f_sig),where=(xs>=f_mean-f_sig)&(xs<=f_mean+f_sig),alpha=.25,color=colors['EDE'])
axes[0].set_xlabel('$f_{EDE}$'); axes[0].set_ylabel('Gaussian density')
xs=np.linspace(loga_mean-4*loga_sig,loga_mean+4*loga_sig,400)
axes[1].plot(xs,norm.pdf(xs,loga_mean,loga_sig),color=colors['EDE'])
axes[1].axvline(loga_mean,color=colors['EDE'],ls='--')
axes[1].fill_between(xs,norm.pdf(xs,loga_mean,loga_sig),where=(xs>=loga_mean-loga_sig)&(xs<=loga_mean+loga_sig),alpha=.25,color=colors['EDE'])
axes[1].set_xlabel('$\log_{10} a_c$')
fig.suptitle('EDE posterior summaries as Gaussian approximations from mean±1σ')
fig.savefig(IMG/'ede_posterior_approx.png',dpi=220)
plt.close(fig)

# Figure 3: distance residual panels
fig,axes=plt.subplots(3,1,figsize=(7,8),sharex=False,constrained_layout=True)
for ax,(fname,title,ylabel) in zip(axes,[('distance_points_dvrd.csv','DESI BAO residuals: $\Delta(D_V/r_d)$','$\Delta(D_V/r_d)$'),('distance_points_fap.csv','DESI BAO residuals: $\Delta F_{AP}$','$\Delta F_{AP}$'),('distance_points_sne_mu.csv','Union3 SNe residuals: $\Delta\mu$','$\Delta\mu$')]):
    df=pd.read_csv(OUT/fname)
    ax.errorbar(df.z,df.value,yerr=df.error,fmt='o-',capsize=3)
    ax.axhline(0,color='k',lw=1,ls='--')
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel('redshift z')
fig.savefig(IMG/'distance_residuals.png',dpi=220)
plt.close(fig)

# Figure 4: heatmap z shifts common params
heat=shift.pivot(index='parameter',columns='comparison',values='z_shift').loc[[p for p in ['omega_m','H0','sigma8','ns','ombh2','ln10As','tau'] if p in shift.parameter.unique()]]
fig,ax=plt.subplots(figsize=(5.5,4.2),constrained_layout=True)
sns.heatmap(heat,annot=True,fmt='.2f',center=0,cmap='vlag',ax=ax,cbar_kws={'label':'shift / combined 1σ'})
ax.set_title('Parameter shifts relative to ΛCDM')
fig.savefig(IMG/'parameter_shift_heatmap.png',dpi=220)
plt.close(fig)

# Figure 5: goodness context
fig,ax=plt.subplots(figsize=(8,3.6),constrained_layout=True)
plot=fit.copy(); plot['label']=plot['source'].str.extract(r'^(.*?) related')[0].fillna(plot.source)+'\n'+plot['model']
ax.barh(plot['label'],plot['delta_chi2_vs_lcdm'],color=['#888888' if x==0 else colors['EDE'] for x in plot['delta_chi2_vs_lcdm']])
ax.axvline(0,color='k',lw=1); ax.set_xlabel('$\Delta\chi^2$ relative to ΛCDM (negative favors listed model)')
ax.set_title('Goodness-of-fit context from related work (not recomputed likelihoods)')
fig.savefig(IMG/'goodness_context.png',dpi=220)
plt.close(fig)

# Claim recovery table
claims=[
 {'claim':'EDE raises H0 relative to ΛCDM in the provided CMB+DESI summary.','artifact':'outputs/model_parameter_shifts.csv; outputs/ede_parameter_summary.json','support':f"H0 shift EDE-ΛCDM = {models['EDE']['H0'][0]-models['ΛCDM']['H0'][0]:.2f} km/s/Mpc, z_shift={ede_summary['H0_shift_EDE_vs_LCDM_sigma']:.2f}."},
 {'claim':'w0wa shifts H0 downward while increasing Ωm relative to ΛCDM, unlike EDE.','artifact':'outputs/model_parameter_shifts.csv; report/images/parameter_shift_heatmap.png','support':'w0wa H0 and Ωm shifts have opposite signs compared with EDE for these parameters.'},
 {'claim':'The provided EDE posterior prefers nonzero f_EDE at about 3σ under a Gaussian approximation.','artifact':'outputs/ede_parameter_summary.json; report/images/ede_posterior_approx.png','support':f"f_EDE={f_mean:.3f}±{f_sig:.3f}; mean/sigma={f_mean/f_sig:.2f}."},
 {'claim':'DESI BAO residuals in the extracted points are small and redshift structured, with low-z DV/rd negative and high-z closer to zero/positive.','artifact':'outputs/distance_points_dvrd.csv; report/images/distance_residuals.png','support':'Figure/table preserve redshift-level residuals and errors.'},
 {'claim':'Full Δχ² for the requested DESI DR2 CMB+BAO fits is not directly reproducible from local raw likelihoods.','artifact':'outputs/dependency_check.json; outputs/goodness_of_fit_context.csv','support':'Only summary constraints and figure-extracted points are supplied; related-work Δχ² values are reported as contextual evidence.'}
]
pd.DataFrame(claims).to_csv(OUT/'claim_recovery_table.csv',index=False)

# Update inventory status
inv=json.loads((OUT/'target_artifact_inventory.json').read_text())
for section in ['primary_artifacts','figure_families']:
    for item in inv[section]:
        target=ROOT/item['target']
        if target.exists(): item['status']='satisfied'
        else: item['status']='unsatisfied: file not produced'
(OUT/'target_artifact_inventory.json').write_text(json.dumps(inv,indent=2,ensure_ascii=False))
print(json.dumps({'parameters_rows':len(param),'shift_rows':len(shift),'figures':[p.name for p in sorted(IMG.glob('*.png'))],'ede_summary':ede_summary},indent=2))
