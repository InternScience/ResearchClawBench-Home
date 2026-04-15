import pypsa
import pandas as pd
network = pypsa.Network(\"outputs/network_baseline.nc\")
network.lopf(solver_name=\"highs\")
print(f\"Status: {network.lopf.successful}\")
print(f\"Objective (total cost €): {network.objective:.0f}\")

wind_gens = network.generators.carrier == \"onshore_wind\"
curtail = ((network.generators_t.p_max_pu[wind_gens] * network.generators.loc[wind_gens, \"p_nom\"]) - network.generators_t.p[wind_gens]).sum().sum()
results = {
    \"objective\": float(network.objective),
    \"total_demand\": float(network.loads_t.p_set.sum().sum()),
    \"total_gen\": float(network.generators_t.p.sum().sum()),
    \"total_curtail_wind_mwh\": float(curtail),
    \"max_line_loading\": float((network.lines_t.p0.abs() / network.lines.s_nom).max().max())
}
pd.DataFrame([results]).to_json(\"outputs/costs_baseline.json\")

network.generators_t.p.to_feather(\"outputs/gen_dispatch_baseline.feather\")
network.storage_units_t.p.to_feather(\"outputs/sto_dispatch_baseline.feather\")
network.lines_t.p0.to_feather(\"outputs/line_p_baseline.feather\")
network.storage_units_t.state_of_charge.to_feather(\"outputs/sto_soc_baseline.feather\")
print(\"Baseline results saved\")","path">code/solve_baseline.py