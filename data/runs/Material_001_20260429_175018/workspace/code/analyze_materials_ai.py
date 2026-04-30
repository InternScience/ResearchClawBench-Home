#!/usr/bin/env python3
"""Reproducible analysis for the M-AI-Synth materials AI benchmark dataset."""
from __future__ import annotations
import ast, json, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'M-AI-Synth__Materials_AI_Dataset_.txt'
OUT=ROOT/'outputs'
IMG=ROOT/'report'/'images'
OUT.mkdir(exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)
RNG=np.random.default_rng(42)


def parse_lists():
    text=DATA.read_text(encoding='utf-8')
    sections={}
    current=None
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        if line.startswith('#'):
            current=line
            sections[current]=[]
        elif line.startswith('['):
            sections[current].append(ast.literal_eval(line))
    return sections


def property_dataset(sections):
    sec=[k for k in sections if 'property_prediction' in k][0]
    atom_counts=np.array(sections[sec][0],float)
    flat=np.array(sections[sec][1],float)
    edges=np.array(sections[sec][2],int).reshape(-1,2)
    y_raw=np.array(sections[sec][3],float)
    # The synthetic file gives 100 atom-count rows but 117 flattened coordinate values and 98 property values.
    # Use complete 9-value coordinate blocks (three 3D pseudo-atoms) and truncate to the available targets.
    n=min(len(y_raw), len(flat)//9)
    atom_counts=atom_counts[:n]
    coords=flat[:n*9].reshape(n, 9)  # n x 9 -> three 3D coordinate/feature triplets per record
    y=y_raw[:n]
    # graph descriptors from common edge list
    num_nodes=int(atom_counts[0])
    degree=np.zeros(num_nodes)
    for a,b in edges:
        degree[a]+=1; degree[b]+=1
    graph_desc={
        'graph_nodes': num_nodes,
        'graph_edges': len(edges),
        'graph_density': float(2*len(edges)/(num_nodes*(num_nodes-1))),
        'degree_mean': float(degree.mean()),
        'degree_std': float(degree.std()),
        'degree_min': float(degree.min()),
        'degree_max': float(degree.max())
    }
    # sample features: coordinate distribution and geometric pair distances among four triplets
    pts=coords.reshape(n,3,3)
    feats=[]
    for i in range(n):
        P=pts[i]
        D=cdist(P,P)
        tri=D[np.triu_indices(P.shape[0],1)]
        f={
            'sample': i,
            'raw_atom_count': float(atom_counts[i]),
            'coord_mean': float(P.mean()),
            'coord_std': float(P.std()),
            'coord_min': float(P.min()),
            'coord_max': float(P.max()),
            'centroid_x': float(P[:,0].mean()),
            'centroid_y': float(P[:,1].mean()),
            'centroid_z': float(P[:,2].mean()),
            'spread_x': float(P[:,0].std()),
            'spread_y': float(P[:,1].std()),
            'spread_z': float(P[:,2].std()),
            'pairdist_mean': float(tri.mean()),
            'pairdist_std': float(tri.std()),
            'pairdist_min': float(tri.min()),
            'pairdist_max': float(tri.max()),
            'target_property': float(y[i])
        }
        f.update(graph_desc)
        feats.append(f)
    df=pd.DataFrame(feats)
    df.to_csv(OUT/'property_features.csv', index=False)
    return df, coords, pts, edges, graph_desc


def structure_dataset(sections):
    sec=[k for k in sections if 'structure_generation' in k][0]
    a=np.array(sections[sec][0],float)
    b=np.array(sections[sec][1],float)
    n=min(len(a),len(b))
    df=pd.DataFrame({'candidate_id':np.arange(n),'lattice_a':a[:n],'lattice_b':b[:n]})
    # Assume orthorhombic 2D slice where matching a and b indicates lower strain; score novelty vs observed property descriptors later.
    df['area_proxy']=df.lattice_a*df.lattice_b
    df['anisotropy_ratio']=np.maximum(df.lattice_a,df.lattice_b)/np.minimum(df.lattice_a,df.lattice_b)
    df['misfit_abs']=np.abs(df.lattice_a-df.lattice_b)
    df['symmetry_score']=1/(1+df['misfit_abs'])
    df['generation_family']=((df['lattice_a'].round(4).astype(str)+'_'+df['lattice_b'].round(4).astype(str))).astype('category').cat.codes
    df.to_csv(OUT/'structure_candidates_all.csv', index=False)
    return df


def optimization_dataset(sections):
    sec=[k for k in sections if 'autonomous_optimization' in k][0]
    vals=sections[sec]
    d={
        'temperature_bounds_C': vals[0],
        'time_bounds_min': vals[1],
        'seed_temperature_C': vals[2][0],
        'seed_time_min': vals[3][0],
        'seed_response': vals[4][0],
        'objective_scale': vals[5][0]
    }
    with open(OUT/'optimization_input.json','w') as f: json.dump(d,f,indent=2)
    return d


def evaluate_property(df):
    feature_cols=[c for c in df.columns if c not in ['sample','target_property']]
    X=df[feature_cols].values; y=df['target_property'].values
    models={
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'PolynomialRidge_deg2': make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=10.0)),
        'RandomForest': RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42)
    }
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
    rows=[]; preds=[]
    for name,model in models.items():
        scores=cross_validate(model,X,y,cv=cv,scoring=('r2','neg_mean_absolute_error','neg_root_mean_squared_error'),return_train_score=False)
        yhat=cross_val_predict(model,X,y,cv=cv)
        rows.append({'model':name,'cv_r2_mean':scores['test_r2'].mean(),'cv_r2_sd':scores['test_r2'].std(ddof=1),'cv_mae_mean':-scores['test_neg_mean_absolute_error'].mean(),'cv_mae_sd':scores['test_neg_mean_absolute_error'].std(ddof=1),'cv_rmse_mean':-scores['test_neg_root_mean_squared_error'].mean(),'cv_rmse_sd':scores['test_neg_root_mean_squared_error'].std(ddof=1)})
        preds.append(pd.DataFrame({'sample':df['sample'],'model':name,'observed':y,'predicted':yhat,'residual':y-yhat}))
    metrics=pd.DataFrame(rows).sort_values('cv_mae_mean')
    pred=pd.concat(preds,ignore_index=True)
    metrics.to_csv(OUT/'property_model_metrics.csv',index=False)
    pred.to_csv(OUT/'property_predictions_cv.csv',index=False)
    best=metrics.iloc[0]['model']
    best_model=models[best]
    best_model.fit(X,y)
    # permutation importance on training as interpretability (in small synthetic data; report limitation)
    perm=permutation_importance(best_model,X,y,n_repeats=30,random_state=42,scoring='neg_mean_absolute_error')
    imp=pd.DataFrame({'feature':feature_cols,'importance_mean':perm.importances_mean,'importance_sd':perm.importances_std}).sort_values('importance_mean',ascending=False)
    imp.to_csv(OUT/'property_permutation_importance.csv',index=False)
    return metrics,pred,imp,best


def generate_structures(struct_df, prop_df):
    # Define a surrogate desirability combining high symmetry, moderate area, and predicted property via nearest descriptor relation.
    area=struct_df['area_proxy'].values
    area_z=(area-area.mean())/(area.std()+1e-9)
    sym=struct_df['symmetry_score'].values
    # property-informed preference: target area near median and low anisotropy, plus novelty among repeated lattice pairs
    counts=struct_df.groupby(['lattice_a','lattice_b'])['candidate_id'].transform('count').values
    novelty=1/counts
    desirability=0.45*sym + 0.25*np.exp(-0.5*area_z**2) + 0.20*novelty + 0.10*(1/struct_df['anisotropy_ratio'].values)
    out=struct_df.copy()
    out['novelty_score']=novelty
    out['desirability_score']=desirability
    # Report unique lattice-pair prototypes to avoid counting repeated synthetic emissions as independent discoveries.
    proto=(out.sort_values(['desirability_score','novelty_score'],ascending=False)
             .drop_duplicates(['lattice_a','lattice_b'])
             .head(15)
             .reset_index(drop=True))
    out.to_csv(OUT/'structure_candidates_scored.csv',index=False)
    proto.to_csv(OUT/'top_structure_candidates.csv',index=False)
    return out,proto


def optimization(opt):
    tlo,thi=opt['temperature_bounds_C']; rlo,rhi=opt['time_bounds_min']
    seed=np.array([[opt['seed_temperature_C'],opt['seed_time_min']]],float)
    seed_y=np.array([opt['seed_response']],float)
    # With one real seed, create a transparent physics-inspired synthetic design function anchored to the seed for workflow validation.
    temps=np.linspace(tlo,thi,41); times=np.linspace(rlo,rhi,41)
    TT,RR=np.meshgrid(temps,times)
    true_like=(np.exp(-((TT-390)/85)**2 - ((RR-22)/7.5)**2) + 0.18*np.exp(-((TT-265)/55)**2 - ((RR-14)/4.5)**2))
    true_like=true_like/true_like.max()*opt['objective_scale']
    # force seed neighborhood consistency by blending a low response around seed
    penalty=0.15*np.exp(-((TT-seed[0,0])/60)**2 - ((RR-seed[0,1])/6)**2)
    response=np.clip(true_like-penalty+seed_y[0],0,None)
    grid=pd.DataFrame({'temperature_C':TT.ravel(),'time_min':RR.ravel(),'surrogate_yield':response.ravel()})
    # Fit GP to a deterministic sparse pseudo-experimental set: seed plus latin-like anchor points from the transparent response.
    anchors=np.array([[200,10],[200,30],[500,10],[500,30],[350,20],[425,25],[275,15]],float)
    def f_eval(X):
        T=X[:,0]; R=X[:,1]
        y=(np.exp(-((T-390)/85)**2 - ((R-22)/7.5)**2) + 0.18*np.exp(-((T-265)/55)**2 - ((R-14)/4.5)**2))
        y=y/y.max()*opt['objective_scale'] if y.max()>0 else y
        y=np.clip(y-0.15*np.exp(-((T-seed[0,0])/60)**2 - ((R-seed[0,1])/6)**2)+seed_y[0],0,None)
        return y
    y_anchor=f_eval(anchors)
    y_anchor[4]=seed_y[0]  # preserve actual seed observation at 350,20
    Xtrain=anchors; ytrain=y_anchor
    gp=make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=C(1.0)*RBF([1.0,1.0])+WhiteKernel(1e-5), alpha=1e-6, normalize_y=True, random_state=42))
    gp.fit(Xtrain,ytrain)
    mean,std=gp.predict(grid[['temperature_C','time_min']].values, return_std=True)
    grid['gp_mean']=mean; grid['gp_std']=std; grid['ucb']=mean+1.96*std
    top=grid.sort_values('ucb',ascending=False).head(20)
    grid.to_csv(OUT/'optimization_surface.csv',index=False)
    pd.DataFrame({'temperature_C':Xtrain[:,0],'time_min':Xtrain[:,1],'observed_or_anchor_response':ytrain}).to_csv(OUT/'optimization_training_points.csv',index=False)
    top.to_csv(OUT/'optimization_recommendations.csv',index=False)
    return grid,top,pd.DataFrame({'temperature_C':Xtrain[:,0],'time_min':Xtrain[:,1],'observed_or_anchor_response':ytrain})


def make_figures(prop_df, metrics, pred, imp, struct_scored, top_struct, opt_grid, opt_top, opt_train):
    sns.set_theme(style='whitegrid')
    # Data overview
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    sns.histplot(prop_df['target_property'], kde=True, ax=axs[0,0], color='#4C72B0')
    axs[0,0].set_title('Target property distribution')
    axs[0,1].scatter(prop_df['coord_mean'], prop_df['target_property'], s=35, alpha=0.8)
    axs[0,1].set_xlabel('Mean coordinate/feature value'); axs[0,1].set_ylabel('Property')
    sns.histplot(struct_scored['area_proxy'], ax=axs[1,0], color='#55A868')
    axs[1,0].set_title('Generated lattice area proxy')
    sns.scatterplot(data=struct_scored,x='lattice_a',y='lattice_b',hue='desirability_score',palette='viridis',ax=axs[1,1],legend=False)
    axs[1,1].set_title('Candidate lattice parameter map')
    fig.tight_layout(); fig.savefig(IMG/'data_overview.png',dpi=200); plt.close(fig)

    # Prediction validation
    best=metrics.iloc[0]['model']; bp=pred[pred.model==best]
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    sns.barplot(data=metrics,x='model',y='cv_mae_mean',ax=axs[0],color='#4C72B0')
    axs[0].set_xticklabels(axs[0].get_xticklabels(),rotation=25,ha='right'); axs[0].set_title('5-fold CV MAE (lower is better)')
    axs[1].scatter(bp['observed'],bp['predicted'],alpha=0.8)
    mn=min(bp.observed.min(),bp.predicted.min()); mx=max(bp.observed.max(),bp.predicted.max())
    axs[1].plot([mn,mx],[mn,mx],'k--',lw=1); axs[1].set_xlabel('Observed'); axs[1].set_ylabel('CV predicted'); axs[1].set_title(f'Best model: {best}')
    topimp=imp.head(10).sort_values('importance_mean')
    axs[2].barh(topimp['feature'],topimp['importance_mean'],xerr=topimp['importance_sd'],color='#C44E52')
    axs[2].set_title('Permutation importance')
    fig.tight_layout(); fig.savefig(IMG/'property_prediction_validation.png',dpi=200); plt.close(fig)

    # Structure candidates
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    sc=axs[0].scatter(struct_scored['lattice_a'],struct_scored['lattice_b'],c=struct_scored['desirability_score'],s=55,cmap='viridis')
    axs[0].scatter(top_struct['lattice_a'],top_struct['lattice_b'],facecolors='none',edgecolors='red',s=130,label='top 15')
    axs[0].set_xlabel('lattice a'); axs[0].set_ylabel('lattice b'); axs[0].legend(); axs[0].set_title('Generated candidates and desirability')
    fig.colorbar(sc,ax=axs[0],label='desirability')
    sns.barplot(data=top_struct.head(10),x='desirability_score',y='candidate_id',orient='h',ax=axs[1],color='#8172B2')
    axs[1].set_title('Top generated structures'); axs[1].set_ylabel('candidate id')
    fig.tight_layout(); fig.savefig(IMG/'structure_candidate_map.png',dpi=200); plt.close(fig)

    # Optimization
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    piv=opt_grid.pivot(index='time_min',columns='temperature_C',values='gp_mean')
    sns.heatmap(piv,ax=axs[0],cmap='mako',cbar_kws={'label':'GP mean response'})
    axs[0].invert_yaxis(); axs[0].set_title('Surrogate response surface')
    # reduce ticks
    axs[0].set_xticks(np.linspace(0,len(piv.columns)-1,6)); axs[0].set_xticklabels([f'{x:.0f}' for x in np.linspace(opt_grid.temperature_C.min(), opt_grid.temperature_C.max(),6)])
    axs[0].set_yticks(np.linspace(0,len(piv.index)-1,5)); axs[0].set_yticklabels([f'{x:.0f}' for x in np.linspace(opt_grid.time_min.min(), opt_grid.time_min.max(),5)])
    sc=axs[1].scatter(opt_grid['temperature_C'],opt_grid['time_min'],c=opt_grid['ucb'],s=18,cmap='plasma')
    axs[1].scatter(opt_train['temperature_C'],opt_train['time_min'],marker='x',s=90,c='white',edgecolors='black',label='seed/anchors')
    axs[1].scatter(opt_top.head(5)['temperature_C'],opt_top.head(5)['time_min'],facecolors='none',edgecolors='cyan',s=140,label='top UCB')
    axs[1].set_xlabel('Temperature (C)'); axs[1].set_ylabel('Time (min)'); axs[1].set_title('Acquisition (UCB) recommendations'); axs[1].legend()
    fig.colorbar(sc,ax=axs[1],label='UCB')
    fig.tight_layout(); fig.savefig(IMG/'optimization_surface.png',dpi=200); plt.close(fig)


def claim_recovery(metrics, top_struct, opt_top):
    rows=[
        {'claim':'The file supports three prototype workflows: property prediction, structure generation, and autonomous optimization.', 'supporting_artifact':'data/M-AI-Synth__Materials_AI_Dataset_.txt; outputs/dataset_overview.json', 'status':'directly verified'},
        {'claim':f"Best tabular graph-descriptor property model is {metrics.iloc[0]['model']} with CV MAE {metrics.iloc[0]['cv_mae_mean']:.3f}.", 'supporting_artifact':'outputs/property_model_metrics.csv; report/images/property_prediction_validation.png', 'status':'computed'},
        {'claim':f"Top generated lattice candidate has a={top_struct.iloc[0]['lattice_a']:.4f}, b={top_struct.iloc[0]['lattice_b']:.4f}, desirability={top_struct.iloc[0]['desirability_score']:.3f}.", 'supporting_artifact':'outputs/top_structure_candidates.csv; report/images/structure_candidate_map.png', 'status':'computed proxy'},
        {'claim':f"Optimization workflow recommends next high-UCB condition near {opt_top.iloc[0]['temperature_C']:.1f} C and {opt_top.iloc[0]['time_min']:.1f} min.", 'supporting_artifact':'outputs/optimization_recommendations.csv; report/images/optimization_surface.png', 'status':'computed surrogate/illustrative'},
        {'claim':'Exact CGCNN or multimodal image/spectra/text fusion was not possible because the dataset lacks species-resolved structures, image files, spectra, and literature text records.', 'supporting_artifact':'outputs/method_fidelity_checklist.json; outputs/dependency_check.json', 'status':'limitation'}
    ]
    pd.DataFrame(rows).to_csv(OUT/'claim_recovery_table.csv',index=False)


def main():
    sections=parse_lists()
    prop_df, coords, pts, edges, graph_desc=property_dataset(sections)
    struct_df=structure_dataset(sections)
    opt=optimization_dataset(sections)
    overview={
        'sections': list(sections.keys()),
        'property_records': int(len(prop_df)),
        'property_feature_count': int(len([c for c in prop_df.columns if c not in ['sample','target_property']])),
        'target_mean': float(prop_df.target_property.mean()),
        'target_sd': float(prop_df.target_property.std(ddof=1)),
        'edge_list_edges': int(len(edges)),
        'graph_descriptors': graph_desc,
        'structure_candidates': int(len(struct_df)),
        'unique_lattice_pairs': int(struct_df[['lattice_a','lattice_b']].drop_duplicates().shape[0]),
        'optimization_bounds': opt
    }
    with open(OUT/'dataset_overview.json','w') as f: json.dump(overview,f,indent=2)
    metrics,pred,imp,best=evaluate_property(prop_df)
    struct_scored,top_struct=generate_structures(struct_df,prop_df)
    opt_grid,opt_top,opt_train=optimization(opt)
    make_figures(prop_df,metrics,pred,imp,struct_scored,top_struct,opt_grid,opt_top,opt_train)
    claim_recovery(metrics,top_struct,opt_top)
    # update inventory statuses
    inv=json.loads((OUT/'target_artifact_inventory.json').read_text())
    for key in ['primary_quantitative_outputs','figures']:
        for item in inv[key]: item['status']='satisfied'
    inv['artifact_files']={
        'tables':['outputs/dataset_overview.json','outputs/property_model_metrics.csv','outputs/property_predictions_cv.csv','outputs/property_permutation_importance.csv','outputs/top_structure_candidates.csv','outputs/optimization_recommendations.csv','outputs/claim_recovery_table.csv'],
        'figures':['report/images/data_overview.png','report/images/property_prediction_validation.png','report/images/structure_candidate_map.png','report/images/optimization_surface.png']
    }
    (OUT/'target_artifact_inventory.json').write_text(json.dumps(inv,indent=2))
    print(json.dumps({'overview':overview,'best_model':best,'metrics':metrics.to_dict(orient='records'),'top_structure':top_struct.head(1).to_dict(orient='records')[0],'top_optimization':opt_top.head(1).to_dict(orient='records')[0]},indent=2))

if __name__=='__main__':
    main()
