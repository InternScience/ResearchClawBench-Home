import json

method_contract = {
    "task": "Design and evaluate an optimal personalized neoantigen vaccine composition",
    "objective": "MinSum",
    "budget": 10,
    "metrics": [
        "per-cell immune response probability",
        "coverage ratio of tumor cells",
        "IoU of optimal vaccine compositions",
        "optimization runtime data"
    ]
}

target_artifact_inventory = {
    "fig1_mutations_per_cell.png": "Distribution of mutations per cell",
    "fig2_vaccine_composition.png": "Vaccine composition weights",
    "fig3_response_probability.png": "Distribution of per-cell immune response probability",
    "fig4_coverage_ratio.png": "Coverage ratio of tumor cells vs response threshold",
    "fig5_iou_heatmap.png": "IoU of optimal vaccine compositions across repetitions",
    "fig6_runtime.png": "Optimization runtime vs population size"
}

with open('outputs/method_contract.json', 'w') as f:
    json.dump(method_contract, f, indent=4)

with open('outputs/target_artifact_inventory.json', 'w') as f:
    json.dump(target_artifact_inventory, f, indent=4)
