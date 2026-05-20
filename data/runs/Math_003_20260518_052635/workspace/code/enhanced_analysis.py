#!/usr/bin/env python3
"""
Enhanced analysis of IMO geometry problems.
"""

import json
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_problem_difficulty():
    """Analyze problem difficulty based on various metrics."""
    
    # Load problems
    with open("outputs/imo_problems.json", "r") as f:
        problems = json.load(f)
    
    analysis = {
        "difficulty_metrics": [],
        "relation_chains": [],
        "construction_patterns": []
    }
    
    for problem in problems:
        metrics = {
            "name": problem["name"],
            "num_points": len(problem["points"]),
            "num_premises": len(problem["premises"]),
            "relation_types": Counter(),
            "has_triangle": False,
            "has_circle": False,
            "has_perpendicular": False,
            "has_parallel": False,
            "construction_complexity": 0
        }
        
        # Analyze premises
        for premise in problem["premises"]:
            pred = premise["predicate"]
            metrics["relation_types"][pred] += 1
            
            # Check for key geometric objects
            if pred == "triangle":
                metrics["has_triangle"] = True
            elif pred in ["circle", "on_circle"]:
                metrics["has_circle"] = True
            elif pred == "perp":
                metrics["has_perpendicular"] = True
            elif pred == "para":
                metrics["has_parallel"] = True
            
            # Estimate construction complexity
            if pred in ["foot", "orthocenter", "incenter", "excenter", 
                       "circumcenter", "midpoint", "reflect", "mirror"]:
                metrics["construction_complexity"] += 1
        
        # Analyze conclusion
        conclusion = problem.get("conclusion", {})
        conclusion_pred = conclusion.get("predicate", "")
        
        metrics["conclusion_type"] = conclusion_pred
        
        # Compute difficulty score
        difficulty_score = (
            metrics["num_points"] * 0.3 +
            metrics["num_premises"] * 0.2 +
            metrics["construction_complexity"] * 0.5
        )
        
        metrics["difficulty_score"] = difficulty_score
        
        analysis["difficulty_metrics"].append(metrics)
    
    # Sort by difficulty
    analysis["difficulty_metrics"].sort(key=lambda x: x["difficulty_score"], reverse=True)
    
    return analysis


def create_difficulty_heatmap(analysis):
    """Create a heatmap of problem features."""
    
    metrics = analysis["difficulty_metrics"]
    
    # Extract features for heatmap
    features = [
        "num_points", "num_premises", "construction_complexity"
    ]
    
    problem_names = [m["name"].replace("translated_imo_", "") for m in metrics[:15]]
    feature_data = []
    
    for feature in features:
        values = [m[feature] for m in metrics[:15]]
        # Normalize
        max_val = max(values) if values else 1
        normalized = [v / max_val for v in values]
        feature_data.append(normalized)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    heatmap = ax.imshow(feature_data, cmap='YlOrRd', aspect='auto')
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(['Points', 'Premises', 'Construction\nComplexity'], fontsize=11)
    ax.set_xticks(range(len(problem_names)))
    ax.set_xticklabels(problem_names, rotation=45, ha='right', fontsize=9)
    
    ax.set_title('Problem Feature Heatmap\n(Top 15 by Difficulty)', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Normalized Value', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('report/images/figure_7_difficulty_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 7 (difficulty heatmap) saved")


def analyze_conclusion_patterns():
    """Analyze patterns in conclusions."""
    
    with open("outputs/imo_problems.json", "r") as f:
        problems = json.load(f)
    
    patterns = {
        "conclusion_by_year": defaultdict(list),
        "conclusion_by_position": defaultdict(list),
        "conclusion_complexity": []
    }
    
    for problem in problems:
        name = problem["name"]
        parts = name.split("_")
        
        if len(parts) >= 4:
            year = parts[3]
            position = parts[4] if len(parts) > 4 else "unknown"
        else:
            year = "unknown"
            position = "unknown"
        
        conclusion = problem.get("conclusion", {})
        conclusion_pred = conclusion.get("predicate", "unknown")
        
        patterns["conclusion_by_year"][year].append(conclusion_pred)
        patterns["conclusion_by_position"][position].append(conclusion_pred)
        
        # Estimate conclusion complexity
        complexity = len(conclusion.get("args", []))
        patterns["conclusion_complexity"].append({
            "name": name,
            "conclusion": conclusion_pred,
            "complexity": complexity
        })
    
    return patterns


def create_conclusion_analysis_figures(patterns):
    """Create figures analyzing conclusion patterns."""
    
    # Figure 8: Conclusion types by year
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # By year
    years = sorted(patterns["conclusion_by_year"].keys())[:10]  # Top 10 years
    conclusion_types = ["cong", "coll", "cyclic", "perp", "para", "eqangle", "eqratio"]
    
    data_by_year = []
    for year in years:
        counts = Counter(patterns["conclusion_by_year"][year])
        row = [counts.get(ct, 0) for ct in conclusion_types]
        data_by_year.append(row)
    
    x = np.arange(len(years))
    width = 0.1
    
    for i, ct in enumerate(conclusion_types):
        values = [row[i] for row in data_by_year]
        axes[0].bar(x + i * width, values, width, label=ct)
    
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Conclusion Types by IMO Year', fontsize=14)
    axes[0].set_xticks(x + width * 3)
    axes[0].set_xticklabels(years, rotation=45)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    # By position
    positions = sorted(patterns["conclusion_by_position"].keys())
    data_by_position = []
    for pos in positions:
        counts = Counter(patterns["conclusion_by_position"][pos])
        row = [counts.get(ct, 0) for ct in conclusion_types]
        data_by_position.append(row)
    
    x = np.arange(len(positions))
    
    for i, ct in enumerate(conclusion_types):
        values = [row[i] for row in data_by_position]
        axes[1].bar(x + i * width, values, width, label=ct)
    
    axes[1].set_xlabel('Problem Position', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Conclusion Types by Problem Position', fontsize=14)
    axes[1].set_xticks(x + width * 3)
    axes[1].set_xticklabels(positions, rotation=45)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/figure_8_conclusion_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 8 (conclusion patterns) saved")
    
    # Figure 9: Conclusion complexity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    complexity_data = patterns["conclusion_complexity"]
    complexity_by_type = defaultdict(list)
    
    for item in complexity_data:
        complexity_by_type[item["conclusion"]].append(item["complexity"])
    
    # Box plot
    positions = range(len(complexity_by_type))
    data = [complexity_by_type[ct] for ct in complexity_by_type.keys()]
    
    bp = ax.boxplot(data, positions=positions, widths=0.6)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(list(complexity_by_type.keys()), rotation=45, ha='right')
    ax.set_ylabel('Number of Arguments', fontsize=12)
    ax.set_title('Conclusion Complexity by Type', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/figure_9_conclusion_complexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 9 (conclusion complexity) saved")


def create_annotated_problem_table():
    """Create an annotated table of problems with difficulty scores."""
    
    with open("outputs/benchmark_analysis.json", "r") as f:
        analysis = json.load(f)
    
    # Create a summary table
    table_data = []
    
    for prob in analysis["problems"]:
        table_data.append({
            "Problem": prob["name"].replace("translated_imo_", ""),
            "Points": prob["num_points"],
            "Statements": prob["num_statements"],
            "Conclusion": prob["conclusion_type"],
            "Year": prob["name"].split("_")[3] if len(prob["name"].split("_")) >= 4 else "N/A",
            "Position": prob["name"].split("_")[4] if len(prob["name"].split("_")) >= 5 else "N/A"
        })
    
    # Save as JSON
    with open("outputs/problem_table.json", "w") as f:
        json.dump(table_data, f, indent=2)
    
    print(f"Created problem table with {len(table_data)} problems")
    return table_data


if __name__ == "__main__":
    print("Running enhanced analysis...")
    
    # Analyze difficulty
    difficulty_analysis = analyze_problem_difficulty()
    
    # Create difficulty heatmap
    create_difficulty_heatmap(difficulty_analysis)
    
    # Analyze conclusion patterns
    conclusion_patterns = analyze_conclusion_patterns()
    
    # Create conclusion analysis figures
    create_conclusion_analysis_figures(conclusion_patterns)
    
    # Create annotated problem table
    create_annotated_problem_table()
    
    print("\nEnhanced analysis complete!")