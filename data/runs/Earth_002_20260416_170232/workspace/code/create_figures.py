#!/usr/bin/env python3
"""
Create figures for mangrove composite risk index report.
Generates data overview, main results, and comparison plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = Path("report/images")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

def load_risk_data():
    """Load risk assessment results."""
    risk_data = {}
    for scenario in ["ssp245", "ssp370", "ssp585"]:
        risk_data[scenario] = pd.read_csv(OUTPUT_DIR / f"risk_assessment_{scenario}.csv")
    
    summary = pd.read_csv(OUTPUT_DIR / "risk_summary.csv")
    tc_freq = pd.read_csv(OUTPUT_DIR / "tc_frequency_grid.csv")
    
    return risk_data, summary, tc_freq


def plot_slr_distribution(risk_data, summary):
    """Figure 1: SLR rate distributions across scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scenarios = ["ssp245", "ssp370", "ssp585"]
    scenario_labels = ["SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    
    for ax, scenario, label, color in zip(axes, scenarios, scenario_labels, colors):
        data = risk_data[scenario]["slr_rate"]
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor="black")
        ax.axvline(4, color="red", linestyle="--", linewidth=2, label="4 mm/yr threshold")
        ax.axvline(7, color="darkred", linestyle="--", linewidth=2, label="7 mm/yr threshold")
        ax.set_xlabel("SLR Rate (mm/yr)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{label}\nMean: {data.mean():.1f} mm/yr", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_slr_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'fig1_slr_distribution.png'}")


def plot_tc_frequency_map(tc_freq):
    """Figure 2: Global TC frequency distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Create scatter plot with color by frequency
    scatter = ax.scatter(
        tc_freq["lon_grid"], 
        tc_freq["lat_grid"],
        c=np.log10(tc_freq["freq_total"] + 1),
        cmap="YlOrRd",
        s=20,
        alpha=0.6,
        edgecolors="none"
    )
    
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Global Tropical Cyclone Frequency (Historical, log scale)", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, label="log10(Frequency + 1)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_tc_frequency.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'fig2_tc_frequency.png'}")


def plot_risk_comparison(summary):
    """Figure 3: Risk component comparison across scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    scenarios = ["ssp245", "ssp370", "ssp585"]
    scenario_labels = ["SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    
    # Left: Mean risk components
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.25
    
    slr_risk = summary["mean_slr_risk"].values
    tc_risk = summary["mean_tc_risk"].values
    comp_risk = summary["mean_composite_risk"].values
    
    bars1 = ax.bar(x - width, slr_risk, width, label="SLR Risk", color=colors[0])
    bars2 = ax.bar(x, tc_risk, width, label="TC Risk", color=colors[1])
    bars3 = ax.bar(x + width, comp_risk, width, label="Composite Risk", color=colors[2])
    
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Mean Risk Score (0-1)", fontsize=12)
    ax.set_title("Risk Components by SSP Scenario", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Right: Risk class distribution
    ax = axes[1]
    
    risk_dist = []
    for scenario in scenarios:
        risk_data = pd.read_csv(OUTPUT_DIR / f"risk_assessment_{scenario}.csv")
        dist = risk_data["risk_class"].value_counts()
        low = dist.get("Low", 0) / len(risk_data) * 100
        med = dist.get("Medium", 0) / len(risk_data) * 100
        high = dist.get("High", 0) / len(risk_data) * 100
        risk_dist.append([low, med, high])
    
    risk_dist = np.array(risk_dist)
    
    x = np.arange(len(scenarios))
    ax.bar(x, risk_dist[:, 2], label="High Risk", color="#d62728")
    ax.bar(x, risk_dist[:, 1], bottom=risk_dist[:, 2], label="Medium Risk", color="#ff7f0e")
    ax.bar(x, risk_dist[:, 0], bottom=risk_dist[:, 2] + risk_dist[:, 1], label="Low Risk", color="#2ca02c")
    
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Percentage of Mangrove Cells (%)", fontsize=12)
    ax.set_title("Risk Class Distribution by Scenario", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_risk_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'fig3_risk_comparison.png'}")


def plot_risk_maps(risk_data):
    """Figure 4: Spatial distribution of composite risk."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scenarios = ["ssp245", "ssp370", "ssp585"]
    scenario_labels = ["SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
    
    for ax, scenario, label in zip(axes, scenarios, scenario_labels):
        df = risk_data[scenario]
        
        # Scatter plot colored by risk
        scatter = ax.scatter(
            df["lon_mean"], 
            df["lat_mean"],
            c=df["composite_risk"],
            cmap="RdYlGn_r",
            vmin=0, vmax=1,
            s=30,
            alpha=0.7,
            edgecolors="none"
        )
        
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"{label}\nHigh Risk: {(df['risk_class']=='High').sum()} cells", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 60)
        
        cbar = plt.colorbar(scatter, ax=ax, label="Composite Risk Index")
    
    plt.suptitle("Global Mangrove Composite Risk Index by SSP Scenario", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_risk_maps.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'fig4_risk_maps.png'}")


def plot_slr_vs_tc_scatter(risk_data):
    """Figure 5: Relationship between SLR and TC risk components."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scenarios = ["ssp245", "ssp370", "ssp585"]
    scenario_labels = ["SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    
    for ax, scenario, label, color in zip(axes, scenarios, scenario_labels, colors):
        df = risk_data[scenario]
        
        ax.scatter(df["slr_risk"], df["tc_risk"], 
                   c=df["composite_risk"], cmap="viridis",
                   alpha=0.5, s=20, edgecolors="none")
        
        ax.axhline(0.33, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0.33, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0.66, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0.66, color="gray", linestyle="--", alpha=0.5)
        
        ax.set_xlabel("SLR Risk Score", fontsize=12)
        ax.set_ylabel("TC Risk Score", fontsize=12)
        ax.set_title(f"{label}\nCorrelation: {df['slr_risk'].corr(df['tc_risk']):.3f}", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_slr_tc_relationship.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'fig5_slr_tc_relationship.png'}")


def plot_high_risk_hotspots(risk_data):
    """Figure 6: High-risk mangrove locations by scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scenarios = ["ssp245", "ssp370", "ssp585"]
    scenario_labels = ["SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
    
    for ax, scenario, label in zip(axes, scenarios, scenario_labels):
        df = risk_data[scenario]
        high_risk = df[df["risk_class"] == "High"]
        
        # Plot all mangroves in light gray
        ax.scatter(df["lon_mean"], df["lat_mean"], 
                   c="lightgray", s=20, alpha=0.3, label="All mangroves")
        
        # Plot high risk in red
        ax.scatter(high_risk["lon_mean"], high_risk["lat_mean"], 
                   c="#d62728", s=40, alpha=0.8, label="High risk", edgecolors="black", linewidth=0.5)
        
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"{label}\n{len(high_risk)} high-risk cells ({len(high_risk)/len(df)*100:.1f}%)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 60)
        ax.legend(loc="lower right", fontsize=9)
    
    plt.suptitle("High-Risk Mangrove Locations by SSP Scenario", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_high_risk_hotspots.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {FIGURES_DIR / 'fig6_high_risk_hotspots.png'}")


def main():
    print("=" * 60)
    print("Loading risk data...")
    print("=" * 60)
    
    risk_data, summary, tc_freq = load_risk_data()
    
    print(f"Loaded data for scenarios: {list(risk_data.keys())}")
    print(f"Summary stats: {len(summary)} rows")
    
    print("\n" + "=" * 60)
    print("Creating figures...")
    print("=" * 60)
    
    print("\nFigure 1: SLR distributions...")
    plot_slr_distribution(risk_data, summary)
    
    print("Figure 2: TC frequency map...")
    plot_tc_frequency_map(tc_freq)
    
    print("Figure 3: Risk comparison...")
    plot_risk_comparison(summary)
    
    print("Figure 4: Risk maps...")
    plot_risk_maps(risk_data)
    
    print("Figure 5: SLR-TC relationship...")
    plot_slr_vs_tc_scatter(risk_data)
    
    print("Figure 6: High-risk hotspots...")
    plot_high_risk_hotspots(risk_data)
    
    print("\n" + "=" * 60)
    print("All figures saved to:", FIGURES_DIR.absolute())
    print("=" * 60)


if __name__ == "__main__":
    main()
