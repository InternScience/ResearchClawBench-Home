import json
import os

os.makedirs('outputs', exist_ok=True)

fidelity = {
    "method_fidelity": [
        {
            "method": "Reconcile diverse observational methods",
            "fidelity": "High",
            "reason": "Used the pre-reconciled GlaMBIE dataset which explicitly combines glaciological, DEM differencing, altimetry, and gravimetry estimates into a consensus time series with propagated uncertainties."
        },
        {
            "method": "Deliver a consistent and high-confidence assessment of global glacial mass change",
            "fidelity": "High",
            "reason": "Calculated and reported global and regional cumulative mass changes (Gt and m w.e.) with 1-sigma uncertainties, demonstrating decadal acceleration."
        }
    ]
}

with open('outputs/method_fidelity_checklist.json', 'w') as f:
    json.dump(fidelity, f, indent=4)

print("Saved outputs/method_fidelity_checklist.json")
