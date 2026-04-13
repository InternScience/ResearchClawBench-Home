from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "HEEW_Mini-Dataset"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"

ENERGY_VARS = [
    "Electricity [kW]",
    "Heat [mmBTU]",
    "Cooling Energy [Ton]",
    "PV Power Generation [kW]",
    "Greenhouse Gas Emission [Ton]",
]

WEATHER_VARS = [
    "Temperature [°F]",
    "Dew Point [°F]",
    "Humidity [%]",
    "Wind Speed [mph]",
    "Wind Gust [mph]",
    "Pressure [in]",
    "Precipitation [in]",
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_energy_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(DATA_DIR.glob("*_energy.csv")):
        key = path.stem.replace("_energy", "")
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df["entity"] = key
        df["dayofyear"] = df["datetime"].dt.dayofyear
        df["month_name"] = df["datetime"].dt.month_name().str.slice(0, 3)
        frames[key] = df
    return frames


def load_weather() -> pd.DataFrame:
    weather = pd.read_csv(DATA_DIR / "Total_weather.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather["dayofyear"] = weather["datetime"].dt.dayofyear
    weather["month_name"] = weather["datetime"].dt.month_name().str.slice(0, 3)
    return weather


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_quality(frames: dict[str, pd.DataFrame], weather: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    records = []
    for name, df in frames.items():
        row = {"entity": name, "rows": len(df), "missing_values": int(df.isna().sum().sum())}
        for col in ENERGY_VARS:
            row[f"{col} min"] = float(df[col].min())
            row[f"{col} max"] = float(df[col].max())
            row[f"{col} mean"] = float(df[col].mean())
        records.append(row)
    quality_df = pd.DataFrame(records).sort_values("entity")
    quality_df.to_csv(OUTPUT_DIR / "quality_summary.csv", index=False)

    quality_payload = {
        "energy_entities": list(quality_df["entity"]),
        "weather_rows": int(len(weather)),
        "weather_missing_values": int(weather.isna().sum().sum()),
        "energy_all_complete": bool((quality_df["missing_values"] == 0).all()),
        "weather_all_complete": bool(weather.isna().sum().sum() == 0),
    }
    write_json(OUTPUT_DIR / "quality_summary.json", quality_payload)
    return quality_df, quality_payload


def validate_hierarchy(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    building_names = sorted([name for name in frames if name.startswith("BN")])
    building_sum = sum(frames[name][ENERGY_VARS] for name in building_names)
    cn01 = frames["CN01"][ENERGY_VARS]
    total = frames["Total"][ENERGY_VARS]

    results = []
    for label, left, right in [
        ("sum_buildings_vs_CN01", building_sum, cn01),
        ("CN01_vs_Total", cn01, total),
    ]:
        diff = left - right
        for col in ENERGY_VARS:
            results.append(
                {
                    "comparison": label,
                    "variable": col,
                    "max_abs_error": float(diff[col].abs().max()),
                    "mean_abs_error": float(diff[col].abs().mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(diff[col])))),
                }
            )
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_DIR / "hierarchy_validation.csv", index=False)
    return out


def make_energy_weather_frame(total_energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = total_energy[["datetime"] + ENERGY_VARS].merge(weather[["datetime"] + WEATHER_VARS], on="datetime", how="inner")
    merged.to_csv(OUTPUT_DIR / "total_energy_weather_join.csv", index=False)
    return merged


def build_feature_table(frames: dict[str, pd.DataFrame], weather: pd.DataFrame) -> pd.DataFrame:
    weather_small = weather[["datetime", "Temperature [°F]", "Humidity [%]", "Precipitation [in]"]]
    records = []
    for name in sorted([n for n in frames if n.startswith("BN")]):
        df = frames[name].merge(weather_small, on="datetime", how="left")
        summer = df[df["datetime"].dt.month.isin([6, 7, 8])]
        winter = df[df["datetime"].dt.month.isin([12, 1, 2])]
        pv_day = df[(df["hour"] >= 10) & (df["hour"] <= 15)]
        records.append(
            {
                "entity": name,
                "electricity_mean": df["Electricity [kW]"].mean(),
                "heat_mean": df["Heat [mmBTU]"].mean(),
                "cooling_mean": df["Cooling Energy [Ton]"].mean(),
                "pv_mean": df["PV Power Generation [kW]"].mean(),
                "ghg_mean": df["Greenhouse Gas Emission [Ton]"].mean(),
                "electricity_cv": df["Electricity [kW]"].std() / df["Electricity [kW]"].mean(),
                "daily_peak_hour_mean": df.groupby(df["datetime"].dt.date)["Electricity [kW]"].idxmax().pipe(
                    lambda idx: df.loc[idx, "hour"].mean()
                ),
                "summer_cooling_mean": summer["Cooling Energy [Ton]"].mean(),
                "winter_heat_mean": winter["Heat [mmBTU]"].mean(),
                "pv_daylight_mean": pv_day["PV Power Generation [kW]"].mean(),
                "corr_elec_temp": df["Electricity [kW]"].corr(df["Temperature [°F]"]),
                "corr_heat_temp": df["Heat [mmBTU]"].corr(df["Temperature [°F]"]),
                "corr_cooling_temp": df["Cooling Energy [Ton]"].corr(df["Temperature [°F]"]),
                "corr_elec_humidity": df["Electricity [kW]"].corr(df["Humidity [%]"]),
                "corr_pv_precip": df["PV Power Generation [kW]"].corr(df["Precipitation [in]"]),
            }
        )
    features = pd.DataFrame(records).set_index("entity")
    features.to_csv(OUTPUT_DIR / "building_feature_table.csv")
    return features


def cluster_buildings(features: pd.DataFrame) -> tuple[pd.DataFrame, dict, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.values)
    link = linkage(pdist(scaled, metric="euclidean"), method="ward")

    silhouette_by_k = {}
    best_k = None
    best_score = -1.0
    for k in range(2, min(5, len(features) - 1) + 1):
        labels = fcluster(link, k, criterion="maxclust")
        score = silhouette_score(scaled, labels)
        silhouette_by_k[k] = float(score)
        if score > best_score:
            best_k = k
            best_score = score

    labels = fcluster(link, best_k, criterion="maxclust")
    clustered = features.copy()
    clustered["cluster"] = labels
    clustered.to_csv(OUTPUT_DIR / "building_clusters.csv")
    summary = {
        "best_k": int(best_k),
        "best_silhouette": float(best_score),
        "silhouette_by_k": silhouette_by_k,
        "cluster_members": {
            str(cluster): clustered.index[clustered["cluster"] == cluster].tolist()
            for cluster in sorted(clustered["cluster"].unique())
        },
    }
    write_json(OUTPUT_DIR / "clustering_summary.json", summary)
    return clustered, summary, link


def create_figures(
    frames: dict[str, pd.DataFrame],
    weather: pd.DataFrame,
    merged_total: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    clustered: pd.DataFrame,
    link: np.ndarray,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    total = frames["Total"]
    monthly = (
        total.assign(month=total["datetime"].dt.month)
        .groupby("month")[ENERGY_VARS]
        .mean()
        .reset_index()
        .melt(id_vars="month", var_name="variable", value_name="value")
    )
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=monthly, x="month", y="value", hue="variable", marker="o")
    plt.title("Monthly Mean Energy and Emissions for the Aggregated Series")
    plt.xlabel("Month")
    plt.ylabel("Mean hourly value")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "monthly_energy_profiles.png", dpi=200)
    plt.close()

    corr = merged_total[ENERGY_VARS + WEATHER_VARS].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Structure Across Energy, Emissions, and Weather")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "energy_weather_correlation_heatmap.png", dpi=200)
    plt.close()

    hierarchy_pivot = hierarchy_df.pivot(index="variable", columns="comparison", values="max_abs_error")
    plt.figure(figsize=(9, 5))
    sns.heatmap(hierarchy_pivot, annot=True, fmt=".2e", cmap="mako")
    plt.title("Maximum Absolute Aggregation Error by Variable")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hierarchy_validation_heatmap.png", dpi=200)
    plt.close()

    weather_energy = total[["datetime", "Cooling Energy [Ton]", "Heat [mmBTU]"]].merge(
        weather[["datetime", "Temperature [°F]"]], on="datetime", how="inner"
    )
    sample = weather_energy.sample(n=min(2500, len(weather_energy)), random_state=7)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=sample,
        x="Temperature [°F]",
        y="Cooling Energy [Ton]",
        s=18,
        alpha=0.5,
        label="Cooling",
    )
    sns.scatterplot(
        data=sample,
        x="Temperature [°F]",
        y="Heat [mmBTU]",
        s=18,
        alpha=0.5,
        label="Heat",
    )
    plt.title("Temperature Response of Cooling and Heat Demand")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "temperature_vs_thermal_loads.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    dendrogram(link, labels=clustered.index.tolist(), leaf_rotation=0)
    plt.title("Ward Hierarchical Clustering of Building Profiles")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "building_dendrogram.png", dpi=200)
    plt.close()

    cluster_plot = clustered.reset_index()
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=cluster_plot,
        x="corr_cooling_temp",
        y="winter_heat_mean",
        hue="cluster",
        style="cluster",
        s=120,
    )
    for _, row in cluster_plot.iterrows():
        plt.text(row["corr_cooling_temp"] + 0.003, row["winter_heat_mean"] + 0.05, row["entity"], fontsize=10)
    plt.title("Building Clusters by Thermal Sensitivity and Winter Heat Use")
    plt.xlabel("Correlation: cooling vs. temperature")
    plt.ylabel("Winter mean heat")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "building_cluster_scatter.png", dpi=200)
    plt.close()


def create_report_tables(
    frames: dict[str, pd.DataFrame], merged_total: pd.DataFrame, clustered: pd.DataFrame, hierarchy_df: pd.DataFrame
) -> None:
    overview = pd.DataFrame(
        [
            {
                "num_buildings": len([n for n in frames if n.startswith("BN")]),
                "num_aggregates": len([n for n in frames if not n.startswith("BN")]),
                "hours_per_series": len(next(iter(frames.values()))),
                "energy_variables": len(ENERGY_VARS),
                "weather_variables": len(WEATHER_VARS),
                "merged_rows": len(merged_total),
            }
        ]
    )
    overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)

    annual_stats = merged_total[ENERGY_VARS + WEATHER_VARS].agg(["mean", "std", "min", "max"]).T
    annual_stats.to_csv(OUTPUT_DIR / "annual_descriptive_stats.csv")

    cluster_sizes = clustered["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="count")
    cluster_sizes.to_csv(OUTPUT_DIR / "cluster_sizes.csv", index=False)

    hierarchy_summary = hierarchy_df.groupby("comparison")["max_abs_error"].max().reset_index()
    hierarchy_summary.to_csv(OUTPUT_DIR / "hierarchy_summary.csv", index=False)


def main() -> None:
    ensure_dirs()
    frames = load_energy_frames()
    weather = load_weather()
    _, _ = summarize_quality(frames, weather)
    hierarchy_df = validate_hierarchy(frames)
    merged_total = make_energy_weather_frame(frames["Total"], weather)
    features = build_feature_table(frames, weather)
    clustered, _, link = cluster_buildings(features)
    create_report_tables(frames, merged_total, clustered, hierarchy_df)
    create_figures(frames, weather, merged_total, hierarchy_df, clustered, link)


if __name__ == "__main__":
    main()
