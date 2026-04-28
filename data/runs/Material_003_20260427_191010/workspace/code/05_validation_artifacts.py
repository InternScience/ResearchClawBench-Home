"""
Phase 7: Validation artifacts.
  - render top candidate molecules per target (RDKit)
  - claim-recovery summary
  - method fidelity checklist
"""
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
top = pd.read_csv(os.path.join(ROOT, "outputs/designed_candidates_top.csv"))

# --- render one figure per target showing 3 best candidates -----------------
import matplotlib.image as mpimg
from io import BytesIO
from PIL import Image as PILImage

def smile_image(smi, size=300):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    AllChem.Compute2DCoords(m)
    return Draw.MolToImage(m, size=(size, size))

targets = sorted(top.target_Tg.unique())
n_per = 3
fig, axes = plt.subplots(len(targets), n_per * 2, figsize=(15, 3.4 * len(targets)))
if axes.ndim == 1:
    axes = axes[None, :]
for r, tgt in enumerate(targets):
    sub = top[top.target_Tg == tgt].head(n_per)
    for c, (_, row) in enumerate(sub.iterrows()):
        ax_a = axes[r, 2*c]; ax_e = axes[r, 2*c+1]
        ima = smile_image(row.acid_canonical)
        ime = smile_image(row.epoxide_canonical)
        if ima is not None: ax_a.imshow(ima)
        if ime is not None: ax_e.imshow(ime)
        ax_a.set_title(f"{tgt:.0f} K  acid",  fontsize=9)
        ax_e.set_title(f"pred {row.pred_tg_reencoded:.0f} K  epoxide", fontsize=9)
        for ax in (ax_a, ax_e):
            ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Top novel vitrimer candidates per target Tg (3 per target)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "report/images/fig_top_candidates.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# --- claim recovery table ---------------------------------------------------
def n(x): return int(x) if isinstance(x, (np.integer,)) else x
cm = pd.read_csv(os.path.join(ROOT, "outputs/calibration_metrics.csv"))
pp = json.load(open(os.path.join(ROOT, "outputs/pair_predictor_metrics.json")))
vit = pd.read_csv(os.path.join(ROOT, "outputs/vitrimer_calibrated_tg.csv"))
cand = pd.read_csv(os.path.join(ROOT, "outputs/designed_candidates.csv"))

claims = []
claims.append(dict(claim="MD over-predicts experimental Tg with positive bias",
                   value=f"+{cm.iloc[0]['bias']:.1f} K",
                   evidence="outputs/calibration_metrics.csv (raw MD baseline)"))
claims.append(dict(claim="GP calibration removes the MD bias and matches a linear baseline",
                   value=f"R²={cm.iloc[2]['R2']:.3f}, RMSE={cm.iloc[2]['RMSE']:.1f} K, "
                         f"bias={cm.iloc[2]['bias']:+.2f} K (LOOCV)",
                   evidence="outputs/calibration_metrics.csv"))
claims.append(dict(claim="Calibrated Tg distribution is shifted ~63 K below the MD distribution",
                   value=f"mean MD={vit['tg_md'].mean():.1f} K vs mean calibrated={vit['tg_calibrated'].mean():.1f} K",
                   evidence="outputs/vitrimer_calibrated_tg.csv"))
claims.append(dict(claim="Pair-level Tg predictor on graph-VAE latents generalises",
                   value=f"R²={pp['test_r2']:.3f}, MAE={pp['test_mae']:.1f} K",
                   evidence="outputs/pair_predictor_metrics.json"))
claims.append(dict(claim="Inverse design produces valid SMILES pairs at three Tg targets",
                   value="; ".join(
                       f"{t:.0f} K: {cand[cand.target_Tg==t].valid.mean()*100:.0f}% valid"
                       for t in [350,400,450]),
                   evidence="outputs/designed_candidates.csv"))
claims.append(dict(claim="All valid candidates are also novel (not in training set of pairs)",
                   value=f"valid={int(cand.valid.sum())}, novel pair={int(cand.novel_pair.sum())}",
                   evidence="outputs/designed_candidates.csv"))
claims.append(dict(claim="Top novel candidates re-encode to within ±1 K of target for 350/400 K",
                   value=top.assign(err=(top.pred_tg_reencoded-top.target_Tg).abs())
                            .groupby("target_Tg").err.median().to_dict(),
                   evidence="outputs/designed_candidates_top.csv"))
pd.DataFrame(claims).to_csv(os.path.join(ROOT, "outputs/claim_recovery.csv"), index=False)

# --- method fidelity checklist ---------------------------------------------
fidelity = {
  "molecular_dynamics_Tg": {
    "implemented": True,
    "notes": "MD-simulated Tg values are provided directly in tg_calibration.csv "
             "and tg_vitrimer_MD.csv. We did not re-run MD; we use the supplied "
             "values (column tg/tg_md) as the simulator output."
  },
  "gaussian_process_calibration": {
    "implemented": True,
    "details": {
      "kernel": "C * RBF + WhiteKernel",
      "heteroscedastic_noise": "per-point alpha = std_md^2",
      "training_set": "295 polymers in tg_calibration.csv",
      "loo_cv_R2": 0.6782,
      "loo_cv_RMSE_K": 54.13,
      "produces_uncertainty": True
    }
  },
  "graph_variational_autoencoder": {
    "implemented": True,
    "details": {
      "encoder": "3-layer GIN-style message passing on RDKit atom graphs "
                 "(atom features + bond features), sum readout",
      "decoder": "GRU SMILES decoder conditioned on z, teacher-forced training",
      "latent_dim": 64,
      "training_corpus_size": 8000,
      "validation_token_accuracy": 0.888
    },
    "deviations": [
      "Decoder is a SMILES GRU (not graph-output decoder) for tractability on CPU. "
      "The encoder remains graph-based, so the model is faithfully a graph-input "
      "VAE; this is the same encoder/decoder split used in many chemistry VAEs."
    ]
  },
  "inverse_design_in_latent_space": {
    "implemented": True,
    "details": {
      "predictor": "MLP from concatenated (z_acid, z_epoxide) -> calibrated Tg",
      "predictor_test_R2": 0.738,
      "predictor_test_MAE_K": 12.3,
      "optimization": "Adam in latent space minimising (pred-target)^2 + L2",
      "targets_K": [350, 400, 450],
      "candidates_per_target": 400,
      "validity_rate_top_K_targets": [0.51, 0.375, 0.29]
    }
  },
  "experimental_validation": {
    "implemented": False,
    "deviation": "Wet-lab synthesis and DSC measurement are impossible in this "
                 "automated workspace. We substitute *in silico* validation: "
                 "decoded candidates are re-encoded and re-scored by the same "
                 "Tg predictor, and SMILES validity / novelty are checked.",
  }
}
json.dump(fidelity, open(os.path.join(ROOT, "outputs/method_fidelity_checklist.json"), "w"), indent=2)
print("Validation artifacts written.")
