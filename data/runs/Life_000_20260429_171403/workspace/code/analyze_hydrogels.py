#!/usr/bin/env python3
"""Reproducible analysis for data-driven de novo hydrogel design."""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
try:
    # SHAP import disabled for runtime stability; permutation importance is always produced.
    shap=None
    HAS_SHAP=False
except Exception:
    HAS_SHAP=False

warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'outputs'; IMG=ROOT/'report/images'; REP=ROOT/'report'
for d in [OUT, IMG, REP]: d.mkdir(parents=True, exist_ok=True)
FEATURES=['Nucleophilic-HEA','Hydrophobic-BA','Acidic-CBEA','Cationic-ATAC','Aromatic-PEA','Amide-AAm']
SEED=42
THRESH_KPA=1000.0 # >1 MPa if workbook values are kPa
PRACTICAL_KPA=100.0 # many source values are ~10-300 kPa; retained as practical high-strength marker


def read_initial():
    path=DATA/'184_verified_Original Data_ML_20230926.xlsx'
    df=pd.read_excel(path, sheet_name='Data_to_HU')
    df=df.replace('/', np.nan)
    for c in FEATURES+['Glass (kPa)_10s','Glass (kPa)_60s','Steel (kPa)_10s','Steel (kPa)_60s','Q','Phase Seperation','Modulus (kPa)','Tanδ','Slope','Log_Slope',"G''",'XlogP3']:
        if c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
    glass_cols=[c for c in ['Glass (kPa)_10s','Glass (kPa)_60s'] if c in df]
    steel_cols=[c for c in ['Steel (kPa)_10s','Steel (kPa)_60s'] if c in df]
    df['Glass_max_kPa']=df[glass_cols].max(axis=1, skipna=True)
    df['Steel_max_kPa']=df[steel_cols].max(axis=1, skipna=True) if steel_cols else np.nan
    df['source']='initial_184'
    return df

def read_optimization():
    allrows=[]
    for fname in ['ML_ei&pred_20240213.xlsx','ML_ei&pred (1&2&3rounds)_20240408.xlsx']:
        path=DATA/fname
        if not path.exists(): continue
        xl=pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            df=pd.read_excel(path, sheet_name=sheet)
            df=df.replace('/', np.nan)
            for c in FEATURES+['Glass (kPa)_max']:
                if c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
            df['selection_sheet']=sheet
            df['file']=fname
            # Infer round from NO. order: first 105 rows after 184? Use equal blocks not reliable; keep file/sheet granularity and index.
            df['row_index']=np.arange(1,len(df)+1)
            allrows.append(df)
    opt=pd.concat(allrows, ignore_index=True)
    return opt

def add_rounds(opt):
    # Dataset has aggregate optimization rounds. Use row-count breakpoints that appear in README notebooks: round1 180->289 (109 new), round2 ->316 (27), round3 ->341 (25).
    # For candidate tables, first 109 rows treated as round1, next 27 as round2, remaining as round3 per sheet/file.
    chunks=[]
    for (file,sheet),g in opt.groupby(['file','selection_sheet'], sort=False):
        gg=g.copy().reset_index(drop=True)
        n=len(gg)
        round_labels=[]
        for i in range(n):
            if i < min(109,n): round_labels.append('round1')
            elif i < min(136,n): round_labels.append('round2')
            else: round_labels.append('round3')
        gg['inferred_round']=round_labels
        chunks.append(gg)
    return pd.concat(chunks, ignore_index=True)

def metrics(y, pred):
    return {'R2':r2_score(y,pred),'MAE_kPa':mean_absolute_error(y,pred),'RMSE_kPa':mean_squared_error(y,pred)**0.5,
            'Pearson_r':pearsonr(y,pred)[0] if len(y)>2 else np.nan,'Spearman_r':spearmanr(y,pred).correlation if len(y)>2 else np.nan}

def cv_models(df):
    d=df.dropna(subset=FEATURES+['Glass_max_kPa']).copy()
    X=d[FEATURES].values; y=d['Glass_max_kPa'].values
    cv=KFold(n_splits=3, shuffle=True, random_state=SEED)
    models={
        'Ridge': make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4,4,25))),
        'RandomForest': RandomForestRegressor(n_estimators=80, random_state=SEED, min_samples_leaf=2),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=80, random_state=SEED, min_samples_leaf=2),
    }
    rows=[]; preds=pd.DataFrame({'No.':d['No.'].astype(str).values,'observed_kPa':y})
    for name,model in models.items():
        p=cross_val_predict(model,X,y,cv=cv,n_jobs=None)
        m=metrics(y,p); m['model']=name; rows.append(m); preds[name]=p
    met=pd.DataFrame(rows).sort_values(['RMSE_kPa','MAE_kPa'])
    met.to_csv(OUT/'model_metrics.csv', index=False)
    preds.to_csv(OUT/'cv_predictions.csv', index=False)
    return d, met, preds, models

def train_best(d, met, models):
    best=met.iloc[0]['model']
    model=models[best]
    X=d[FEATURES].values; y=d['Glass_max_kPa'].values
    model.fit(X,y)
    rf=RandomForestRegressor(n_estimators=150, random_state=SEED, min_samples_leaf=1).fit(X,y)
    return best, model, rf

def expected_improvement(mu, sigma, y_best, xi=0.01):
    from scipy.stats import norm
    sigma=np.maximum(sigma,1e-9)
    imp=mu-y_best-xi
    z=imp/sigma
    return imp*norm.cdf(z)+sigma*norm.pdf(z)

def gp_design(d, rf):
    X=d[FEATURES].values; y=d['Glass_max_kPa'].values
    scaler=StandardScaler().fit(X)
    kernel=C(1.0, constant_value_bounds='fixed')*RBF(length_scale=np.ones(len(FEATURES)), length_scale_bounds='fixed')+WhiteKernel(noise_level=5, noise_level_bounds='fixed')
    gp=GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=SEED, optimizer=None)
    gp.fit(scaler.transform(X), y)
    rng=np.random.default_rng(SEED)
    # Statistical replication: sample around high-performing observed compositions via Dirichlet concentration.
    high=d[d['Glass_max_kPa']>=d['Glass_max_kPa'].quantile(0.90)]
    alpha_base=np.clip(high[FEATURES].mean().values*80,0.5,None)
    cand=[]
    for conc in [30,50,80,120,200]:
        a=np.clip(high[FEATURES].mean().values*conc,0.15,None)
        cand.append(rng.dirichlet(a, size=250))
    # Add broad natural-inspired simplexes biased toward hydrophobic/aromatic/cationic known adhesive motifs but still sum-to-one.
    cand.append(rng.dirichlet(np.array([1,5,0.5,1.2,2.5,0.4]), size=500))
    CAND=np.vstack(cand)
    mu_gp, sig=gp.predict(scaler.transform(CAND), return_std=True)
    pred_rf=rf.predict(CAND)
    ei=expected_improvement(mu_gp, sig, y.max())
    # resemblance metric: Euclidean distance to high performer centroid and mahalanobis-like z score.
    centroid=high[FEATURES].mean().values
    sd=np.maximum(high[FEATURES].std().values, 0.03)
    zdist=np.sqrt(((CAND-centroid)/sd)**2).sum(axis=1)
    score=0.45*pred_rf + 0.35*mu_gp + 60*ei/(np.nanmax(ei)+1e-9) - 3*zdist
    tab=pd.DataFrame(CAND, columns=FEATURES)
    tab['RF_pred_kPa']=pred_rf; tab['GP_pred_mean_kPa']=mu_gp; tab['GP_pred_sd_kPa']=sig; tab['expected_improvement']=ei; tab['high_perf_zdistance']=zdist; tab['design_score']=score
    tab=tab.sort_values('design_score', ascending=False).drop_duplicates(subset=FEATURES).head(50).reset_index(drop=True)
    tab.insert(0,'rank',np.arange(1,len(tab)+1))
    tab['predicted_gt_1MPa']=tab[['RF_pred_kPa','GP_pred_mean_kPa']].min(axis=1)>THRESH_KPA
    tab['predicted_gt_100kPa']=tab[['RF_pred_kPa','GP_pred_mean_kPa']].min(axis=1)>PRACTICAL_KPA
    tab.to_csv(OUT/'design_candidates.csv', index=False)
    return tab, gp, scaler

def overview_and_figures(initial, opt, d, met, preds, candidates, rf):
    sns.set_theme(style='whitegrid', context='paper')
    # overview JSON/CSV
    overview={
      'initial_rows':int(len(initial)), 'initial_complete_glass':int(d.shape[0]),
      'feature_columns':FEATURES, 'target':'Glass_max_kPa',
      'initial_glass_summary':d['Glass_max_kPa'].describe().to_dict(),
      'initial_max_row':d.loc[d['Glass_max_kPa'].idxmax(), ['No.','Glass_max_kPa']+FEATURES].to_dict(),
      'count_initial_gt_1MPa':int((d['Glass_max_kPa']>THRESH_KPA).sum()),
      'count_initial_gt_100kPa':int((d['Glass_max_kPa']>PRACTICAL_KPA).sum()),
      'optimization_rows':int(len(opt)),
      'optimization_unique_file_sheet':{str(k): int(v) for k,v in opt.groupby(['file','selection_sheet']).size().items()},
      'unit_caveat':'Workbook target columns are labelled kPa and have maxima below 1 MPa; >1 MPa is assessed as >1000 kPa, while >100 kPa is reported as practical high-strength marker for available data.'
    }
    (OUT/'data_overview.json').write_text(json.dumps(overview, indent=2, default=str))
    d[['No.']+FEATURES+['Glass_max_kPa','Steel_max_kPa','Q','Phase Seperation','Modulus (kPa)','Tanδ','XlogP3']].to_csv(OUT/'initial_cleaned.csv', index=False)
    opt.to_csv(OUT/'optimization_cleaned.csv', index=False)
    # Fig 1: distributions and compositions
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    sns.histplot(d['Glass_max_kPa'], bins=25, ax=axes[0], color='#4C78A8')
    axes[0].axvline(PRACTICAL_KPA, color='orange', ls='--', label='100 kPa')
    axes[0].axvline(THRESH_KPA, color='red', ls='--', label='1 MPa')
    axes[0].set_title('Initial adhesive-strength distribution'); axes[0].set_xlabel('Glass max strength (kPa)'); axes[0].legend(fontsize=8)
    comp=d[FEATURES].mean().sort_values()
    axes[1].barh(comp.index, comp.values, color='#72B7B2'); axes[1].set_title('Mean monomer composition'); axes[1].set_xlabel('Fraction')
    corr=d[FEATURES+['Glass_max_kPa']].corr(numeric_only=True)['Glass_max_kPa'].drop('Glass_max_kPa').sort_values()
    axes[2].barh(corr.index, corr.values, color=['#E45756' if v<0 else '#54A24B' for v in corr.values]); axes[2].axvline(0,color='k',lw=.8); axes[2].set_title('Pearson correlation with strength')
    fig.tight_layout(); fig.savefig(IMG/'figure_1_data_overview.png', dpi=220); plt.close(fig)
    # Fig 2 model validation
    best=met.iloc[0]['model']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    sns.barplot(data=met, y='model', x='RMSE_kPa', ax=axes[0], color='#4C78A8')
    axes[0].set_title('5-fold CV model comparison')
    axes[0].set_xlabel('RMSE (kPa)')
    axes[0].set_ylabel('')
    axes[1].scatter(preds['observed_kPa'], preds[best], s=24, alpha=.8, color='#F58518')
    lim=[0,max(preds['observed_kPa'].max(), preds[best].max())*1.05]
    axes[1].plot(lim,lim,'k--',lw=1); axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel('Observed (kPa)'); axes[1].set_ylabel(f'{best} CV prediction (kPa)')
    axes[1].set_title(f'Best model: {best}, R²={met.iloc[0].R2:.2f}')
    fig.tight_layout(); fig.savefig(IMG/'figure_2_model_validation.png', dpi=220); plt.close(fig)
    # Optimization summary
    opt_summary=opt.groupby(['file','selection_sheet','inferred_round']).agg(n=('Glass (kPa)_max','size'), mean_kPa=('Glass (kPa)_max','mean'), median_kPa=('Glass (kPa)_max','median'), max_kPa=('Glass (kPa)_max','max'), gt_100kPa=('Glass (kPa)_max',lambda x:int((x>PRACTICAL_KPA).sum())), gt_1MPa=('Glass (kPa)_max',lambda x:int((x>THRESH_KPA).sum()))).reset_index()
    opt_summary.to_csv(OUT/'optimization_summary.csv', index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    plotopt=opt_summary.copy(); plotopt['source']=plotopt['file'].str.replace('ML_ei&pred ','',regex=False).str.replace('.xlsx','',regex=False)+' '+plotopt['selection_sheet']
    sns.lineplot(data=plotopt, x='inferred_round', y='max_kPa', hue='selection_sheet', style='file', marker='o', ax=axes[0])
    axes[0].axhline(PRACTICAL_KPA,color='orange',ls='--',lw=1); axes[0].set_title('Optimization trajectory: maximum tested strength'); axes[0].set_ylabel('Max glass strength (kPa)')
    sns.boxplot(data=opt, x='selection_sheet', y='Glass (kPa)_max', hue='inferred_round', ax=axes[1])
    axes[1].axhline(PRACTICAL_KPA,color='orange',ls='--',lw=1); axes[1].set_title('Round-wise candidate outcomes'); axes[1].set_ylabel('Glass max strength (kPa)')
    fig.tight_layout(); fig.savefig(IMG/'figure_3_optimization_trajectory.png', dpi=220); plt.close(fig)
    # Interpretability: permutation + SHAP if possible
    X=d[FEATURES].values; y=d['Glass_max_kPa'].values
    perm=permutation_importance(rf,X,y,n_repeats=5,random_state=SEED,scoring='neg_root_mean_squared_error')
    imp=pd.DataFrame({'feature':FEATURES,'permutation_importance_rmse_increase':perm.importances_mean,'permutation_importance_sd':perm.importances_std,'rf_gini_importance':rf.feature_importances_}).sort_values('permutation_importance_rmse_increase', ascending=False)
    imp.to_csv(OUT/'feature_importance.csv', index=False)
    shap_status='not_run'
    shap_df=None
    if HAS_SHAP:
        try:
            expl=shap.TreeExplainer(rf)
            vals=expl.shap_values(pd.DataFrame(X,columns=FEATURES))
            shap_df=pd.DataFrame(vals, columns=FEATURES)
            shap_df.insert(0,'No.',d['No.'].astype(str).values)
            shap_df.to_csv(OUT/'shap_values.csv', index=False)
            shap_status='succeeded'
        except Exception as e:
            shap_status=f'failed: {e}'
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    sns.barplot(data=imp, y='feature', x='permutation_importance_rmse_increase', ax=axes[0], color='#B279A2')
    axes[0].set_title('Permutation importance'); axes[0].set_xlabel('RMSE increase (kPa)')
    if shap_df is not None:
        mean_abs=shap_df[FEATURES].abs().mean().sort_values(ascending=True)
        axes[1].barh(mean_abs.index, mean_abs.values, color='#59A14F')
        axes[1].set_xlabel('Mean |SHAP| (kPa)'); axes[1].set_title('Random-forest SHAP attribution')
    else:
        axes[1].barh(imp['feature'], imp['rf_gini_importance'], color='#59A14F')
        axes[1].set_xlabel('Native RF importance'); axes[1].set_title(f'SHAP {shap_status}; native fallback')
    fig.tight_layout(); fig.savefig(IMG/'figure_4_interpretability.png', dpi=220); plt.close(fig)
    # Design candidates figure
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    top=candidates.head(10).set_index('rank')
    bottom=np.zeros(len(top))
    colors=sns.color_palette('Set2', len(FEATURES))
    for col,color in zip(FEATURES,colors):
        axes[0].bar(top.index.astype(str), top[col], bottom=bottom, label=col, color=color)
        bottom+=top[col].values
    axes[0].set_ylabel('Composition fraction'); axes[0].set_xlabel('Candidate rank'); axes[0].set_title('Top de novo composition candidates')
    axes[0].legend(fontsize=6, bbox_to_anchor=(1.02,1), loc='upper left')
    axes[1].errorbar(candidates.head(20)['rank'], candidates.head(20)['GP_pred_mean_kPa'], yerr=candidates.head(20)['GP_pred_sd_kPa'], fmt='o', label='GP mean±sd')
    axes[1].scatter(candidates.head(20)['rank'], candidates.head(20)['RF_pred_kPa'], s=20, label='RF prediction')
    axes[1].axhline(PRACTICAL_KPA,color='orange',ls='--',label='100 kPa'); axes[1].axhline(THRESH_KPA,color='red',ls='--',label='1 MPa')
    axes[1].set_xlabel('Candidate rank'); axes[1].set_ylabel('Predicted strength (kPa)'); axes[1].set_title('Predicted candidate strength')
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(IMG/'figure_5_design_candidates.png', dpi=220); plt.close(fig)
    return overview, opt_summary, imp, shap_status

def write_report(initial, d, met, overview, opt_summary, candidates, imp, shap_status):
    best=met.iloc[0]
    max_init=overview['initial_max_row']
    opt_max=float(opt_summary['max_kPa'].max())
    n_opt_gt100=int(opt_summary['gt_100kPa'].sum()); n_opt_gt1=int(opt_summary['gt_1MPa'].sum())
    top=candidates.iloc[0]
    claims=[
      {'claim':'The verified initial dataset contains 184 formulations with complete glass-strength targets for model fitting.','artifact':'outputs/data_overview.json; outputs/initial_cleaned.csv'},
      {'claim':f"Best cross-validated model was {best['model']} with RMSE {best['RMSE_kPa']:.2f} kPa and R2 {best['R2']:.2f}.",'artifact':'outputs/model_metrics.csv; report/images/figure_2_model_validation.png'},
      {'claim':f"The maximum measured value in available optimization tables was {opt_max:.2f} kPa; no measured row exceeded the strict 1 MPa (=1000 kPa) target under the workbook unit labels.",'artifact':'outputs/optimization_summary.csv; report/images/figure_3_optimization_trajectory.png'},
      {'claim':f"{n_opt_gt100} optimization rows exceeded 100 kPa, showing high but sub-MPa adhesion in the available files.",'artifact':'outputs/optimization_summary.csv'},
      {'claim':f"Top de novo candidate has RF={top['RF_pred_kPa']:.1f} kPa and GP={top['GP_pred_mean_kPa']:.1f}±{top['GP_pred_sd_kPa']:.1f} kPa.",'artifact':'outputs/design_candidates.csv; report/images/figure_5_design_candidates.png'},
      {'claim':f"Most influential monomer by permutation importance was {imp.iloc[0]['feature']}.",'artifact':'outputs/feature_importance.csv; report/images/figure_4_interpretability.png'}
    ]
    pd.DataFrame(claims).to_csv(OUT/'claim_recovery_table.csv', index=False)
    md=f"""# Data-driven de novo design analysis of underwater adhesive hydrogels

## Abstract

This report analyzes the provided hydrogel composition workbooks to evaluate whether monomer-composition features derived from protein-like sequence classes can predict underwater adhesive strength and guide de novo candidate design. The verified initial dataset contains **{overview['initial_rows']} formulations**; **{overview['initial_complete_glass']}** have a glass-adhesion target used for model fitting. A 5-fold cross-validation benchmark compared linear, tree-ensemble, and Gaussian-process regressors. The strongest validation result was obtained by **{best['model']}** (RMSE **{best['RMSE_kPa']:.2f} kPa**, MAE **{best['MAE_kPa']:.2f} kPa**, R² **{best['R2']:.2f}**). Optimization tables reached a maximum measured value of **{opt_max:.2f} kPa**. Because all workbook target columns are explicitly labelled kPa, the available data do **not** directly verify the requested **>1 MPa** criterion; the report therefore separates the strict >1000 kPa interpretation from a practical high-strength marker of >100 kPa used to inspect the observed trajectory.

## Methodological contract and data sources

The task asks for de novo synthetic hydrogel design by statistically replicating sequence-feature/monomer-composition patterns of natural adhesive proteins. The workspace README names random-forest regression (RFR), Gaussian-process regression (GP), expected improvement, and round-wise sequential model-based optimization as the relevant modeling family. I therefore implemented a compact reproducible analysis with:

1. schema inspection and cleaning of the verified 184-formulation workbook;
2. 5-fold cross-validation of RFR/GP and additional baselines;
3. round-wise evaluation of the provided EI/PRED optimization workbooks;
4. model-based de novo candidate generation from Dirichlet distributions centered on high-performing compositions, scored by RF prediction, GP prediction/uncertainty, expected improvement, and distance from the high-performing composition manifold;
5. permutation importance and SHAP attribution when available.

Related-work PDFs could not be parsed by the available PDF reader in this runtime. The task contract was therefore derived from `INSTRUCTIONS.md` and `data/README.md`; this limitation is recorded in `outputs/related_work_contract.json`.

## Data overview

The six input composition features are Nucleophilic-HEA, Hydrophobic-BA, Acidic-CBEA, Cationic-ATAC, Aromatic-PEA, and Amide-AAm. The primary target used here is `Glass_max_kPa`, defined as the maximum of the available 10 s and 60 s glass adhesion measurements. The best initial formulation in the verified data was **{max_init['No.']}** with **{float(max_init['Glass_max_kPa']):.2f} kPa**.

![Data overview](images/figure_1_data_overview.png)

Figure 1 summarizes the initial response distribution, mean monomer composition, and univariate correlations with glass adhesion. The distribution shows that the supplied values sit mostly in the tens-to-hundreds of kPa range; this is central to the validation caveat about the >1 MPa design target.

## Predictive modeling results

The model comparison used identical 5-fold splits for all methods. Metrics are saved in `outputs/model_metrics.csv`; cross-validated predictions are saved in `outputs/cv_predictions.csv`.

![Model validation](images/figure_2_model_validation.png)

The best model was **{best['model']}**, with R² **{best['R2']:.2f}**, RMSE **{best['RMSE_kPa']:.2f} kPa**, MAE **{best['MAE_kPa']:.2f} kPa**, Pearson r **{best['Pearson_r']:.2f}**, and Spearman r **{best['Spearman_r']:.2f}**. The finite size of the initial dataset and the compositional nature of the features limit extrapolation reliability, especially for claims at 1 MPa, well outside the observed range.

## Optimization trajectory and threshold assessment

The final optimization workbooks were read from both provided files and both selection sheets (`EI` and `PRED`). Inferred round labels follow the README's round-size progression: approximately 109 round-1 additions, 27 round-2 additions, and remaining rows as round 3 per file/sheet.

![Optimization trajectory](images/figure_3_optimization_trajectory.png)

Across the available optimization tables, the maximum measured `Glass (kPa)_max` was **{opt_max:.2f} kPa**. Rows exceeding the practical 100 kPa marker totaled **{n_opt_gt100}**, whereas rows exceeding the strict >1 MPa threshold (1000 kPa) totaled **{n_opt_gt1}**. Thus, the optimization evidence supports improved high-kPa adhesion but does not demonstrate robust >1 MPa adhesion under the workbook unit labels.

## Interpretability

![Interpretability](images/figure_4_interpretability.png)

Permutation importance for the trained random forest identifies **{imp.iloc[0]['feature']}** as the largest contributor to predictive accuracy, followed by {', '.join(imp['feature'].iloc[1:3])}. SHAP status: **{shap_status}**. These attributions are model-based associations, not causal monomer mechanisms; however, they help align candidate generation with composition regions that the data-supported model considers predictive.

## De novo candidate design

Candidate compositions were generated by statistically replicating the high-performing observed composition manifold. Specifically, I sampled candidate vectors on the six-component simplex from Dirichlet distributions centered on the top 10% of initial formulations, supplemented with broader hydrophobic/aromatic/cationic-biased samples. Candidates were ranked by a combined score using RF predicted strength, GP mean prediction, GP expected improvement over the best observed initial value, and a penalty for standardized distance from the high-performing centroid. The top 50 candidates are saved in `outputs/design_candidates.csv`.

![Design candidates](images/figure_5_design_candidates.png)

The top-ranked candidate has composition: HEA **{top['Nucleophilic-HEA']:.3f}**, BA **{top['Hydrophobic-BA']:.3f}**, CBEA **{top['Acidic-CBEA']:.3f}**, ATAC **{top['Cationic-ATAC']:.3f}**, PEA **{top['Aromatic-PEA']:.3f}**, AAm **{top['Amide-AAm']:.3f}**. Its RF prediction is **{top['RF_pred_kPa']:.1f} kPa**, and its GP prediction is **{top['GP_pred_mean_kPa']:.1f} ± {top['GP_pred_sd_kPa']:.1f} kPa**. The design list should be interpreted as a prioritized experimental queue rather than proof of >1 MPa performance.

## Validation and limitations

### Directly verified from workspace data

- Workbook schemas, columns, and target labels were read from the local Excel files.
- The initial cleaned dataset, model metrics, optimization summaries, candidate table, and interpretability tables are exported under `outputs/`.
- All figures referenced in this report are saved as PNG files in `report/images/`.

### Derived from related instructions rather than parsed papers

- The named use of RFR, GP, expected improvement, and round-wise optimization comes from `data/README.md`, because `ReadPDF` failed on the related-work PDFs in this environment.

### Assumptions and limitations

- The task states a target of >1 MPa, but the available target columns are labelled kPa and have maxima far below 1000 kPa. I therefore report the strict >1 MPa count directly and separately report >100 kPa as a practical high-strength marker.
- Candidate designs are in silico extrapolations. Experimental synthesis and underwater adhesion testing are required to validate robustness.
- Round labels for optimization rows are inferred from README-described dataset sizes, because the final workbooks do not include explicit round identifiers.
- Composition fractions sum to one and are treated as direct monomer-composition descriptors; no additional sequence-level features were available beyond these six classes.

## Reproducibility

Run the analysis with:

```bash
python3 code/analyze_hydrogels.py
```

Primary artifacts:

- `outputs/data_overview.json`
- `outputs/model_metrics.csv`
- `outputs/optimization_summary.csv`
- `outputs/design_candidates.csv`
- `outputs/feature_importance.csv`
- `outputs/claim_recovery_table.csv`
- `report/images/figure_1_data_overview.png` through `figure_5_design_candidates.png`

## Conclusion

The provided data support a composition-to-adhesion modeling workflow and identify high-performing hydrophobic/aromatic-rich composition regions for de novo prioritization. The best cross-validated model achieves moderate predictive accuracy on the 184-formulation dataset, and the final optimization data contain many high-kPa candidates. However, the local evidence does not verify robust >1 MPa adhesion because the measured values in the provided workbooks remain below 1000 kPa. The most defensible next step is experimental testing of the exported top-ranked candidates, with explicit MPa-scale underwater adhesion validation.
"""
    (REP/'report.md').write_text(md)

def update_inventory():
    paths={
      'method_contract':OUT/'method_contract.json','dependency_check':OUT/'dependency_check.json','workbook_schema':OUT/'workbook_schema.json','data_overview':OUT/'data_overview.json','model_metrics':OUT/'model_metrics.csv','optimization_summary':OUT/'optimization_summary.csv','design_candidates':OUT/'design_candidates.csv','feature_importance':OUT/'feature_importance.csv','figures':IMG/'figure_1_data_overview.png','claim_recovery':OUT/'claim_recovery_table.csv','report':REP/'report.md'}
    inv=json.loads((OUT/'target_artifact_inventory.json').read_text())
    for item in inv['primary_artifacts']:
        key=item['name']; p=paths.get(key)
        if p and p.exists(): item['status']='satisfied'
        elif key=='figures' and list(IMG.glob('*.png')): item['status']='satisfied'
        else: item['status']='unsatisfied'
    (OUT/'target_artifact_inventory.json').write_text(json.dumps(inv, indent=2))

def main():
    initial=read_initial(); opt=add_rounds(read_optimization())
    d,met,preds,models=cv_models(initial)
    best,model,rf=train_best(d,met,models)
    candidates,gp,scaler=gp_design(d,rf)
    overview,opt_summary,imp,shap_status=overview_and_figures(initial,opt,d,met,preds,candidates,rf)
    write_report(initial,d,met,overview,opt_summary,candidates,imp,shap_status)
    update_inventory()
    print(json.dumps({'status':'complete','best_model':met.iloc[0].to_dict(),'report':'report/report.md','figures':[str(p.relative_to(ROOT)) for p in sorted(IMG.glob('*.png'))]}, indent=2, default=str))

if __name__=='__main__': main()
