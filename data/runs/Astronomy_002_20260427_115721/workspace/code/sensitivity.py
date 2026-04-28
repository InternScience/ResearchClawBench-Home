"""Add a sensitivity variant that inflates errors so chi2/dof = 1.

For a GLS solution with chi2 > dof, the rough rule is to scale all
uncertainties by sqrt(chi2/dof). We re-run the baseline GLS with C scaled by
this factor and report the inflated H0 uncertainty.
"""
import json, math, os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gls_h0 import build_system, solve_gls, report_h0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")

# Baseline
A,y,C,names,meta = build_system()
theta, Cov, resid, chi2, dof = solve_gls(A, y, C)
H0, sH, aH, saH = report_h0(theta, Cov, names)

scale = math.sqrt(chi2/dof)
C_scaled = C * (scale**2)
theta2, Cov2, resid2, chi2_2, dof2 = solve_gls(A, y, C_scaled)
H0s, sHs, aH2, saH2 = report_h0(theta2, Cov2, names)

# A jackknife "robust" variant: drop the worst-residual primary host (NGC1309)
A,y,C,names,meta = build_system(drop_host="NGC1309")
theta3, Cov3, resid3, chi2_3, dof3 = solve_gls(A, y, C)
H0r, sHr, _, _ = report_h0(theta3, Cov3, names)

# Save sensitivity table
tbl = pd.DataFrame([
    dict(name="baseline", H0=H0, sigma_H0=sH, chi2=chi2, dof=dof, chi2_red=chi2/dof, scale_factor=1.0),
    dict(name="baseline_inflated", H0=H0s, sigma_H0=sHs, chi2=chi2_2, dof=dof2,
         chi2_red=chi2_2/dof2, scale_factor=scale),
    dict(name="drop_NGC1309 (robust)", H0=H0r, sigma_H0=sHr, chi2=chi2_3, dof=dof3,
         chi2_red=chi2_3/dof3, scale_factor=1.0),
])
tbl.to_csv(os.path.join(OUT, "sensitivity_table.csv"), index=False)
print(tbl.to_string(index=False))

with open(os.path.join(OUT, "h0_consensus.json"), "w") as f:
    json.dump({
        "baseline":            {"H0": H0,  "sigma_H0": sH},
        "baseline_inflated":   {"H0": H0s, "sigma_H0": sHs, "scale_factor": scale},
        "robust_drop_NGC1309": {"H0": H0r, "sigma_H0": sHr},
    }, f, indent=2)
