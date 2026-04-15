import pypsa
import pandas as pd
from pathlib import Path

snapshots = pd.date_range("2024-01-01", periods=168, freq="h")
network = pypsa.Network()
network.set_snapshots(snapshots)

# Buses
buses_df = pd.read_csv("data/buses.csv", index_col="name")
network.import_components_from_dataframe(buses_df, "Bus")

# Lines
lines_df = pd.read_csv("data/links.csv")
lines_df["name"] = lines_df["bus0"].astype(str) + "-" + lines_df["bus1"].astype(str)
lines_df["s_nom"] = lines_df["p_nom"]
lines_df["r"] = lines_df["length"] * 0.0032  # small resistance ohm/km
lines_df["x"] = lines_df["length"] * 0.32  # reactance ohm/km 400kV
lines_df = lines_df[["name", "bus0", "bus1", "s_nom", "r", "x"]]
network.import_components_from_dataframe(lines_df.set_index("name"), "Line")

# Generators
gens_df = pd.read_csv("data/generators.csv")
gens_df["name"] = gens_df["bus"].astype(str) + "-" + gens_df["carrier"].str.replace(" ", "_")
gens_df = gens_df.set_index("name")[["bus", "p_nom", "marginal_cost", "carrier"]]
network.import_components_from_dataframe(gens_df, "Generator")

# Loads
bus_list = list(network.buses.index)
load_df = pd.DataFrame({"bus": bus_list}, index=[f"load-{b}" for b in bus_list])
network.import_components_from_dataframe(load_df, "Load")
demand_df = pd.read_csv("data/demand.csv")
network.loads_t.p_set = demand_df[bus_list]

# Wind capacity factors p_max_pu
wind_cf_df = pd.read_csv("data/wind_cf.csv")
wind_gens = network.generators.index[network.generators.carrier == "onshore_wind"]
p_max_pu = pd.DataFrame({g: wind_cf_df[g.split("-")[0]] for g in wind_gens}, index=snapshots)
network.generators_t.p_max_pu[wind_gens] = p_max_pu

# Storage
sto_df = pd.read_csv("data/storage.csv")
sto_df["name"] = sto_df["bus"].astype(str) + "-" + sto_df["carrier"]
sto_df = sto_df.set_index("name")
sto_df["max_p_hours"] = sto_df["e_nom"] / sto_df["p_nom"]
sto_df["efficiency_store"] = sto_df["efficiency"]
sto_df["efficiency_dispatch"] = sto_df["efficiency"]
sto_df["marginal_cost"] = 0.0
sto_df = sto_df[["bus", "p_nom", "max_p_hours", "efficiency_store", "efficiency_dispatch", "marginal_cost"]]
network.import_components_from_dataframe(sto_df, "StorageUnit")

network.consistency_check()
print(network)
network.export_to_netcdf("outputs/network_baseline.nc")
print("Baseline network exported")