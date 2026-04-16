import json
import os

os.makedirs('outputs', exist_ok=True)

inventory = {
    "target_artifacts": [
        {
            "name": "global_mass_change.png",
            "type": "figure",
            "description": "Global glacial mass change (Gt and m w.e.) over time with uncertainties."
        },
        {
            "name": "cumulative_mass_change.png",
            "type": "figure",
            "description": "Cumulative global glacial mass change (Gt and m w.e.) over time with uncertainties."
        },
        {
            "name": "regional_mass_change.png",
            "type": "figure",
            "description": "Bar chart of cumulative mass change (Gt) by region."
        },
        {
            "name": "regional_specific_mass_change.png",
            "type": "figure",
            "description": "Cumulative specific mass change (m w.e.) over time by region."
        },
        {
            "name": "methods_comparison.png",
            "type": "figure",
            "description": "Comparison of different observational methods globally over time."
        },
        {
            "name": "regional_mass_change_summary.csv",
            "type": "table",
            "description": "Summary of regional mass changes."
        }
    ]
}

with open('outputs/target_artifact_inventory.json', 'w') as f:
    json.dump(inventory, f, indent=4)

print("Saved outputs/target_artifact_inventory.json")
