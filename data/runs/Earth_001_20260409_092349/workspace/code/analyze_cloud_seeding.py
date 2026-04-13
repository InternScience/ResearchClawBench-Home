from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "dataset1_cloud_seeding_records" / "cloud_seeding_us_2000_2025.csv"
STATE_GEOJSON = ROOT / "data" / "dataset1_cloud_seeding_records" / "us_states.geojson"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "report" / "images"


def normalize_token(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    for column in ["state", "season", "operator_affiliation", "agent", "apparatus", "purpose"]:
        df[column] = df[column].fillna("unknown").astype(str).str.strip()
    df["project"] = df["project"].fillna("unknown").astype(str).str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
    return df


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUTS / f"{name}.csv", index=False)


def split_multivalue_series(series: pd.Series) -> pd.Series:
    items: list[str] = []
    for value in series.dropna():
        for part in str(value).split(","):
            token = normalize_token(part)
            if token and token != "unknown":
                items.append(token)
    return pd.Series(items, name="value")


def top_share(series: pd.Series, top_n: int = 5) -> pd.DataFrame:
    counts = series.value_counts().rename_axis("category").reset_index(name="records")
    counts["share"] = counts["records"] / counts["records"].sum()
    return counts.head(top_n)


def make_state_map(df: pd.DataFrame) -> None:
    state_counts = df.groupby("state").size().reset_index(name="records")
    geo = gpd.read_file(STATE_GEOJSON)
    name_column = "name" if "name" in geo.columns else geo.columns[0]
    geo["state_key"] = geo[name_column].str.strip().str.lower()
    merged = geo.merge(state_counts.assign(state_key=state_counts["state"].str.lower()), on="state_key", how="left")
    merged["records"] = merged["records"].fillna(0)

    fig, ax = plt.subplots(figsize=(13, 8))
    merged.plot(
        column="records",
        cmap="YlOrRd",
        linewidth=0.4,
        edgecolor="0.6",
        legend=True,
        ax=ax,
        legend_kwds={"label": "Project records", "shrink": 0.7},
    )
    ax.set_title("Reported cloud-seeding records by U.S. state, 2000-2025")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(IMAGES / "state_concentration_map.png", dpi=220)
    plt.close(fig)


def make_year_plot(year_counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=year_counts, x="year", y="records", marker="o", linewidth=2.2, ax=ax)
    ax.set_title("Annual cloud-seeding activity records")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of project records")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "annual_activity.png", dpi=220)
    plt.close(fig)


def make_purpose_plot(purpose_counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=purpose_counts.head(10), x="records", y="purpose", color="#2a6f97", ax=ax)
    ax.set_title("Purpose composition of reported projects")
    ax.set_xlabel("Number of project records")
    ax.set_ylabel("Purpose")
    fig.tight_layout()
    fig.savefig(IMAGES / "purpose_composition.png", dpi=220)
    plt.close(fig)


def make_agent_apparatus_heatmap(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("Top seeding agents by deployment apparatus")
    ax.set_xlabel("Apparatus")
    ax.set_ylabel("Agent")
    fig.tight_layout()
    fig.savefig(IMAGES / "agent_apparatus_heatmap.png", dpi=220)
    plt.close(fig)


def write_summary(df: pd.DataFrame, year_counts: pd.DataFrame, state_counts: pd.DataFrame) -> None:
    purpose_tokens = split_multivalue_series(df["purpose"])
    apparatus_tokens = split_multivalue_series(df["apparatus"])
    agent_tokens = split_multivalue_series(df["agent"])

    summary = {
        "records": int(len(df)),
        "years_covered": [int(df["year"].min()), int(df["year"].max())],
        "states_covered": int(df["state"].nunique()),
        "top_state_share": float(state_counts.iloc[0]["share"]),
        "top_3_state_share": float(state_counts.head(3)["share"].sum()),
        "peak_year": int(year_counts.sort_values("records", ascending=False).iloc[0]["year"]),
        "peak_year_records": int(year_counts.sort_values("records", ascending=False).iloc[0]["records"]),
        "latest_year_records": int(year_counts.sort_values("year").iloc[-1]["records"]),
        "median_duration_days": float(df["duration_days"].dropna().median()),
        "top_purpose_tokens": purpose_tokens.value_counts().head(5).to_dict(),
        "top_apparatus_tokens": apparatus_tokens.value_counts().to_dict(),
        "top_agent_tokens": agent_tokens.value_counts().head(5).to_dict(),
    }
    (OUTPUTS / "summary_metrics.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True, parents=True)
    sns.set_theme(style="whitegrid")

    df = load_data()

    overview = pd.DataFrame(
        {
            "metric": [
                "records",
                "distinct_projects",
                "states",
                "years",
                "median_duration_days",
            ],
            "value": [
                len(df),
                df["project"].nunique(),
                df["state"].nunique(),
                f"{df['year'].min()}-{df['year'].max()}",
                round(df["duration_days"].dropna().median(), 1),
            ],
        }
    )
    save_table(overview, "dataset_overview")

    year_counts = df.groupby("year").size().reset_index(name="records").sort_values("year")
    year_counts["year_over_year_change"] = year_counts["records"].diff()
    save_table(year_counts, "annual_activity")

    state_counts = df.groupby("state").size().reset_index(name="records").sort_values("records", ascending=False)
    state_counts["share"] = state_counts["records"] / state_counts["records"].sum()
    save_table(state_counts, "state_counts")

    purpose_counts = df.groupby("purpose").size().reset_index(name="records").sort_values("records", ascending=False)
    purpose_counts["share"] = purpose_counts["records"] / purpose_counts["records"].sum()
    save_table(purpose_counts, "purpose_counts")

    operator_counts = df.groupby("operator_affiliation").size().reset_index(name="records").sort_values("records", ascending=False)
    operator_counts["share"] = operator_counts["records"] / operator_counts["records"].sum()
    save_table(operator_counts, "operator_counts")

    apparatus_tokens = split_multivalue_series(df["apparatus"]).value_counts().rename_axis("apparatus").reset_index(name="records")
    apparatus_tokens["share"] = apparatus_tokens["records"] / apparatus_tokens["records"].sum()
    save_table(apparatus_tokens, "apparatus_token_counts")

    agent_tokens = split_multivalue_series(df["agent"]).value_counts().rename_axis("agent").reset_index(name="records")
    agent_tokens["share"] = agent_tokens["records"] / agent_tokens["records"].sum()
    save_table(agent_tokens, "agent_token_counts")

    exploded_agents = (
        df.assign(agent=df["agent"].str.split(","), apparatus=df["apparatus"].str.split(","))
        .explode("agent")
        .explode("apparatus")
    )
    exploded_agents["agent"] = exploded_agents["agent"].map(normalize_token)
    exploded_agents["apparatus"] = exploded_agents["apparatus"].map(normalize_token)
    exploded_agents = exploded_agents[
        (exploded_agents["agent"] != "unknown")
        & (exploded_agents["apparatus"] != "unknown")
        & (exploded_agents["agent"] != "")
        & (exploded_agents["apparatus"] != "")
    ]
    pair_counts = (
        exploded_agents.groupby(["agent", "apparatus"])
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
    )
    save_table(pair_counts, "agent_apparatus_pairs")

    top_agents = agent_tokens.head(6)["agent"].tolist()
    top_apparatus = apparatus_tokens.head(3)["apparatus"].tolist()
    matrix = (
        pair_counts[pair_counts["agent"].isin(top_agents) & pair_counts["apparatus"].isin(top_apparatus)]
        .pivot(index="agent", columns="apparatus", values="records")
        .fillna(0)
    )
    matrix = matrix.reindex(index=top_agents, columns=top_apparatus).fillna(0)

    make_state_map(df)
    make_year_plot(year_counts)
    make_purpose_plot(purpose_counts)
    make_agent_apparatus_heatmap(matrix)
    write_summary(df, year_counts, state_counts)


if __name__ == "__main__":
    main()
