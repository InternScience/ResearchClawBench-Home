"""
02_models.py — Train and benchmark surrogate ML models.

Models compared (5-fold CV, repeated 5x with different seeds):
  Ridge, RandomForestRegressor, GradientBoostingRegressor,
  GaussianProcessRegressor (Matern + WhiteKernel).
Targets: Glass_kPa (primary).  Inputs: 6 monomer mole fractions.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)

MONOMERS = ["Nucleophilic-HEA","Hydrophobic-BA","Acidic-CBEA",
            "Cationic-ATAC","Aromatic-PEA","Amide-AAm"]

def load_data():
    df = pd.read_csv(os.path.join(OUT, "training_184_clean.csv"))
    return df

def get_models():
    return {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                               max_features="sqrt", random_state=0, n_jobs=-1),
        "GradientBoost": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                    learning_rate=0.05, random_state=0),
        "GP-Matern": Pipeline([("sc", StandardScaler()),
            ("gp", GaussianProcessRegressor(
                kernel=C(1.0,(1e-2,1e2))*Matern(length_scale=0.5,length_scale_bounds=(1e-2,1e2),nu=2.5)
                       + WhiteKernel(noise_level=1.0,noise_level_bounds=(1e-3,1e3)),
                normalize_y=True, n_restarts_optimizer=4, random_state=0))]),
    }

def cv_eval(model, X, y, seeds=(0,1,2,3,4)):
    res = []
    for s in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        for tr, te in kf.split(X):
            m = model
            try:
                m.fit(X.iloc[tr], y.iloc[tr])
            except Exception:
                m.fit(X.values[tr], y.values[tr])
            p = m.predict(X.iloc[te] if hasattr(X,"iloc") else X[te])
            res.append({
                "seed": s,
                "r2":   r2_score(y.iloc[te], p),
                "mae":  mean_absolute_error(y.iloc[te], p),
                "rmse": float(np.sqrt(mean_squared_error(y.iloc[te], p))),
            })
    return pd.DataFrame(res)

def main():
    df = load_data()
    X = df[MONOMERS].copy()
    y = df["Glass_kPa"].copy()

    rows = []
    detail = {}
    fold_preds = {}
    for name, mdl in get_models().items():
        # need a fresh model each fold
        from sklearn.base import clone
        # repeated 5-fold CV with fresh clones
        records = []
        oof_pred = np.zeros(len(y))
        for s in range(5):
            kf = KFold(n_splits=5, shuffle=True, random_state=s)
            for tr, te in kf.split(X):
                m = clone(mdl)
                m.fit(X.iloc[tr], y.iloc[tr])
                p = m.predict(X.iloc[te])
                if s == 0:
                    oof_pred[te] = p
                records.append({
                    "seed": s,
                    "r2":   r2_score(y.iloc[te], p),
                    "mae":  mean_absolute_error(y.iloc[te], p),
                    "rmse": float(np.sqrt(mean_squared_error(y.iloc[te], p))),
                })
        sub = pd.DataFrame(records)
        rows.append({"model": name,
                     "r2_mean": sub["r2"].mean(),  "r2_std": sub["r2"].std(),
                     "mae_mean":sub["mae"].mean(), "mae_std":sub["mae"].std(),
                     "rmse_mean":sub["rmse"].mean(),"rmse_std":sub["rmse"].std()})
        detail[name] = sub.to_dict(orient="list")
        fold_preds[name] = oof_pred
        print(name, rows[-1])

    res = pd.DataFrame(rows).sort_values("rmse_mean")
    res.to_csv(os.path.join(OUT, "model_cv_metrics.csv"), index=False)
    with open(os.path.join(OUT, "model_cv_detail.json"), "w") as fh:
        json.dump(detail, fh, indent=2)

    # === Fig: CV metrics bar chart ===
    sns.set_theme(context="talk", style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("r2_mean","r2_std","R²"),("mae_mean","mae_std","MAE (kPa)"),
               ("rmse_mean","rmse_std","RMSE (kPa)")]
    colors = sns.color_palette("Set2", len(res))
    for ax, (mu, sd, lab) in zip(axes, metrics):
        ax.bar(res["model"], res[mu], yerr=res[sd], color=colors, capsize=4)
        ax.set_ylabel(lab); ax.set_title(lab)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Repeated 5-fold CV (5 seeds) — Glass adhesion regression",
                 fontsize=16, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig02_model_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # === Fig: parity (OOF) for the best two models ===
    best = res["model"].iloc[:2].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12,5.5))
    for ax, name in zip(axes, best):
        p = fold_preds[name]
        ax.scatter(y, p, c="#3b7dd8", s=35, edgecolor="k", lw=.3, alpha=.8)
        lo, hi = 0, max(y.max(), p.max())*1.05
        ax.plot([lo,hi],[lo,hi],"--",color="grey")
        ax.set_xlabel("Measured Glass adhesion (kPa)")
        ax.set_ylabel("Predicted (out-of-fold)")
        ax.set_title(f"{name}  (R²={r2_score(y,p):.2f}, MAE={mean_absolute_error(y,p):.1f})")
    fig.suptitle("Out-of-fold parity plots", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig03_parity_oof.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # === Save best (RandomForest) and GP refits on all data for downstream BO ===
    from sklearn.base import clone
    full_models = {}
    for name, mdl in get_models().items():
        m = clone(mdl)
        m.fit(X, y)
        full_models[name] = m
    import pickle
    with open(os.path.join(OUT, "full_models.pkl"), "wb") as fh:
        pickle.dump(full_models, fh)

    # === RF feature importance fig ===
    rf = full_models["RandomForest"]
    imp = pd.Series(rf.feature_importances_, index=MONOMERS).sort_values()
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.barh([m.split("-",1)[1] for m in imp.index], imp.values, color="#4caf72")
    ax.set_xlabel("RandomForest impurity-based feature importance")
    ax.set_title("Drivers of Glass adhesion in 184-formulation data")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig04_rf_importance.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    imp.to_csv(os.path.join(OUT, "rf_feature_importance.csv"))

    print(res)

if __name__ == "__main__":
    main()
