import os, json, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

os.makedirs('report/images', exist_ok=True)

# dataset overview
with open('outputs/data_summary.json') as f:
    data = json.load(f)
rows=[]
for ds, info in data.items():
    if ds=='muv':
        pos=np.mean([info[c]['positive_rate'] for c in info['label_cols']])
    elif ds=='clintox':
        pos=np.mean([info[c]['positive_rate'] for c in info['label_cols']])
    else:
        pos=info[info['label_cols'][0]]['positive_rate']
    rows.append({'dataset':ds.upper(),'samples':info['rows'],'avg_positive_rate':pos,'tasks':len(info['label_cols'])})
df=pd.DataFrame(rows)
fig,ax=plt.subplots(1,2,figsize=(12,4))
sns.barplot(data=df,x='dataset',y='samples',ax=ax[0],color='#4C72B0')
ax[0].set_yscale('log'); ax[0].set_title('Dataset size (log scale)')
sns.barplot(data=df,x='dataset',y='avg_positive_rate',ax=ax[1],color='#55A868')
ax[1].set_title('Average positive label rate')
plt.tight_layout(); plt.savefig('report/images/data_overview.png',dpi=220); plt.close()

# results summary
rows=[]
for fp in glob.glob('outputs/*_results.json'):
    with open(fp) as f:
        res=json.load(f)
    for r in res:
        rows.append({'dataset':r['dataset'].upper(),'model':r['model'],'roc_auc':r['test_metrics']['roc_auc'],'pr_auc':r['test_metrics']['pr_auc']})
resdf=pd.DataFrame(rows)
resdf.to_csv('outputs/all_results_summary.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(12,4))
sns.barplot(data=resdf,x='dataset',y='roc_auc',hue='model',ax=ax[0])
ax[0].set_title('Test ROC-AUC across datasets')
ax[0].set_ylim(0,1)
sns.barplot(data=resdf,x='dataset',y='pr_auc',hue='model',ax=ax[1])
ax[1].set_title('Test PR-AUC across datasets')
ax[1].set_ylim(0,1)
for a in ax: a.legend_.remove()
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)
plt.tight_layout(rect=[0,0,1,0.92]); plt.savefig('report/images/model_comparison.png',dpi=220); plt.close()

# delta plot
pivot=resdf.pivot(index='dataset',columns='model',values='roc_auc').reset_index()
pivot['delta_ka_minus_mlp']=pivot['KA-GNN']-pivot['GINE-MLP']
fig,ax=plt.subplots(figsize=(7,4))
sns.barplot(data=pivot,x='dataset',y='delta_ka_minus_mlp',palette=['#C44E52' if x<0 else '#55A868' for x in pivot['delta_ka_minus_mlp']],ax=ax)
ax.axhline(0,color='black',lw=1)
ax.set_title('ROC-AUC gain of KA-GNN over GINE-MLP')
ax.set_ylabel('Δ ROC-AUC')
plt.tight_layout(); plt.savefig('report/images/roc_auc_delta.png',dpi=220); plt.close()

# learning curves for representative datasets
sel=['bace','bbbp','clintox']
fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
for ax,ds in zip(axes,sel):
    for model in ['GINE-MLP','KA-GNN']:
        fp=f'outputs/{ds}_{model}_history.csv'
        h=pd.read_csv(fp)
        ax.plot(h['epoch'],h['roc_auc'],marker='o',label=model)
    ax.set_title(ds.upper())
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation ROC-AUC')
    ax.set_ylim(0.3,0.9)
axes[-1].legend()
plt.tight_layout(); plt.savefig('report/images/learning_curves.png',dpi=220); plt.close()

print('figures generated')
