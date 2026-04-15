"""Generate method contract and artifact inventory files."""
import json
import os

os.makedirs("outputs", exist_ok=True)

method_contract = {
    "task": "MATBG Superfluid Stiffness Measurement",
    "scientific_goals": [
        "Directly measure superfluid stiffness of MATBG",
        "Test whether it exceeds conventional Fermi liquid theory predictions",
        "Investigate power-law temperature dependence for anisotropic gap nature",
        "Verify quantum geometric effects in flat-band superconductivity"
    ],
    "named_methods": [
        "Ginzburg-Landau theory (quadratic current dependence)",
        "BCS mean-field theory (s-wave gap)",
        "Nodal superconductor model (linear T dependence)",
        "Power-law temperature dependence analysis",
        "Quantum geometric contribution via Fubini-Study metric",
        "Particle-hole asymmetry analysis"
    ],
    "target_quantities": [
        "Superfluid stiffness D_s as function of carrier density n_eff",
        "Superfluid stiffness D_s as function of temperature T",
        "Superfluid stiffness D_s as function of DC current I_dc",
        "Enhancement factor: D_s_exp / D_s_conv",
        "Power-law exponent from low-T fit",
        "Critical current I_c from GL quadratic fit"
    ],
    "comparison_axes": [
        "Hole-doped vs electron-doped",
        "Conventional (Fermi liquid) vs quantum geometric",
        "BCS vs nodal vs power-law temperature models",
        "Ginzburg-Landau vs linear Meissner current models"
    ]
}

with open("outputs/method_contract.json", 'w') as f:
    json.dump(method_contract, f, indent=2)

artifact_inventory = {
    "primary_quantitative_answers": {
        "enhancement_factor_mean_hole": "~55x over conventional",
        "enhancement_factor_mean_electron": "~52x over conventional",
        "power_law_exponent": "fitted from low-T data",
        "critical_current_GL": "~50 nA",
        "critical_current_exp": "fitted from experimental data"
    },
    "required_figures": [
        "fig01_carrier_density.png - Carrier density dependence with quantum geometry enhancement",
        "fig02_temperature_dependence.png - Temperature dependence with power-law fit",
        "fig03_current_dependence.png - Current dependence GL vs linear",
        "fig04_quadratic_current.png - Quadratic current relationship verification",
        "fig05_summary.png - Comprehensive summary panel",
        "fig06_asymmetry.png - Hole-electron doping asymmetry"
    ],
    "required_tables": [
        "enhancement_stats.json - Enhancement factor statistics",
        "temperature_fit.json - Power law fit results",
        "current_fit.json - Quadratic current fit results",
        "all_results.json - Combined results"
    ],
    "interpretability_artifacts": [
        "Log-log power law residual plot (Fig 5e)",
        "Enhancement factor vs carrier density (Fig 1b, Fig 5d)",
        "Quadratic current verification (Fig 4)"
    ]
}

with open("outputs/target_artifact_inventory.json", 'w') as f:
    json.dump(artifact_inventory, f, indent=2)

print("Method contract and artifact inventory saved.")
