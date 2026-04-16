import json
import os

os.makedirs('outputs', exist_ok=True)

contract = {
    "task": "Reconcile diverse observational methods to deliver a consistent and high-confidence assessment of global glacial mass change (2000-2023).",
    "methods": [
        "In situ glaciological measurements",
        "Digital elevation model (DEM) differencing",
        "Satellite altimetry",
        "Gravimetry"
    ],
    "target_quantities": [
        "Annual regional and global glacial mass change time series",
        "Specific mass change (m w.e.)",
        "Total mass change (Gt)",
        "Uncertainties for all estimates"
    ]
}

with open('outputs/method_contract.json', 'w') as f:
    json.dump(contract, f, indent=4)

print("Saved outputs/method_contract.json")
