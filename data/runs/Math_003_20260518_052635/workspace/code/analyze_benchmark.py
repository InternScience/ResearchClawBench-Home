#!/usr/bin/env python3
"""
Analyze the IMO geometry benchmark.
"""

import json
from collections import Counter
from parse_problems import parse_imo_file, RelationType


def analyze_benchmark():
    """Analyze the IMO geometry benchmark."""
    problems = parse_imo_file("data/imo_ag_30.txt")
    
    analysis = {
        "total_problems": len(problems),
        "problems": [],
        "statistics": {}
    }
    
    # Count different features
    conclusion_types = Counter()
    statement_types = Counter()
    num_points_list = []
    num_statements_list = []
    
    for problem in problems:
        # Analyze each problem
        prob_info = {
            "name": problem.name,
            "num_points": len(problem.points),
            "num_statements": len(problem.statements),
            "conclusion_type": problem.conclusion.relation.value if problem.conclusion else None,
            "statement_types": [s.relation.value for s in problem.statements]
        }
        analysis["problems"].append(prob_info)
        
        # Update counters
        num_points_list.append(len(problem.points))
        num_statements_list.append(len(problem.statements))
        
        if problem.conclusion:
            conclusion_types[problem.conclusion.relation.value] += 1
        
        for stmt in problem.statements:
            statement_types[stmt.relation.value] += 1
    
    # Compute statistics
    analysis["statistics"] = {
        "avg_points": sum(num_points_list) / len(num_points_list),
        "min_points": min(num_points_list),
        "max_points": max(num_points_list),
        "avg_statements": sum(num_statements_list) / len(num_statements_list),
        "conclusion_types": dict(conclusion_types),
        "statement_types": dict(statement_types),
        "year_distribution": {}
    }
    
    # Year distribution
    for prob in problems:
        parts = prob.name.split("_")
        if len(parts) >= 4:
            year = parts[3]
            analysis["statistics"]["year_distribution"][year] = \
                analysis["statistics"]["year_distribution"].get(year, 0) + 1
    
    return analysis


def generate_analysis_report(analysis):
    """Generate a human-readable analysis report."""
    report = []
    report.append("# IMO Geometry Benchmark Analysis\n")
    report.append(f"## Overview")
    report.append(f"- Total problems: {analysis['total_problems']}")
    report.append(f"- Average points per problem: {analysis['statistics']['avg_points']:.1f}")
    report.append(f"- Points range: {analysis['statistics']['min_points']}-{analysis['statistics']['max_points']}")
    report.append(f"- Average statements per problem: {analysis['statistics']['avg_statements']:.1f}")
    
    report.append("\n## Conclusion Types")
    for ctype, count in sorted(analysis['statistics']['conclusion_types'].items()):
        report.append(f"- {ctype}: {count}")
    
    report.append("\n## Most Common Statement Types")
    sorted_stmts = sorted(analysis['statistics']['statement_types'].items(), 
                          key=lambda x: x[1], reverse=True)[:15]
    for stype, count in sorted_stmts:
        report.append(f"- {stype}: {count}")
    
    report.append("\n## Year Distribution")
    for year, count in sorted(analysis['statistics']['year_distribution'].items()):
        report.append(f"- {year}: {count}")
    
    return '\n'.join(report)


if __name__ == "__main__":
    analysis = analyze_benchmark()
    
    # Save analysis
    with open("outputs/benchmark_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Generate report
    report = generate_analysis_report(analysis)
    with open("outputs/benchmark_analysis.md", "w") as f:
        f.write(report)
    
    print(report)