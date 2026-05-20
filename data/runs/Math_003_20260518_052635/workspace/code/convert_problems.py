#!/usr/bin/env python3
"""
Convert IMO problems to a format suitable for the solver.
"""

import json
import re
from typing import List, Dict, Tuple
from parse_problems import parse_imo_file


def convert_imo_to_solver_format(problem) -> dict:
    """Convert a parsed IMO problem to solver format."""
    premises = []
    
    for stmt in problem.statements:
        # Map relation types to predicates
        pred_map = {
            "on_line": "on_line",
            "on_circle": "on_circle",
            "cong": "cong",
            "perp": "perp",
            "para": "para",
            "eqangle": "eqangle",
            "cyclic": "cyclic",
            "coll": "coll",
            "midpoint": "midpoint",
            "triangle": "triangle",
            "foot": "foot",
            "orthocenter": "orthocenter",
            "incenter": "incenter",
            "circle": "circle",
            "on_tline": "perp",
            "on_pline": "para",
            "on_bline": "bline",
            "mirror": "mirror",
            "reflect": "reflect",
            "angle_bisector": "angle_bisector",
            "eqdistance": "eqdistance",
            "excenter": "excenter",
            "excenter2": "excenter2",
            "incenter2": "incenter2",
            "parallelogram": "parallelogram",
            "segment": "segment",
            "free": "free",
            "iso_triangle": "iso_triangle",
            "r_triangle": "r_triangle",
            "cc_tangent": "cc_tangent",
            "on_dia": "on_dia",
            "eqangle2": "eqangle2",
            "eqangle3": "eqangle3",
            "on_aline": "on_aline",
            "eqratio": "eqratio",
            "circumcenter": "circumcenter",
            "ninepoints": "ninepoints",
            "centroid": "centroid",
            "nsquare": "nsquare",
            "psquare": "psquare"
        }
        
        pred = pred_map.get(stmt.relation.value, stmt.relation.value)
        args = []
        
        # Parse arguments based on relation type
        for arg in stmt.arguments:
            # Split by space if needed
            parts = arg.split()
            args.extend(parts)
        
        premises.append({
            "predicate": pred,
            "args": args
        })
    
    # Convert conclusion
    conclusion = {}
    if problem.conclusion:
        pred_map = {
            "cong": "cong",
            "perp": "perp",
            "para": "para",
            "eqangle": "eqangle",
            "cyclic": "cyclic",
            "coll": "coll",
            "eqratio": "eqratio"
        }
        
        pred = pred_map.get(problem.conclusion.relation.value, problem.conclusion.relation.value)
        args = []
        for arg in problem.conclusion.arguments:
            parts = arg.split()
            args.extend(parts)
        
        conclusion = {
            "predicate": pred,
            "args": args
        }
    
    return {
        "name": problem.name,
        "premises": premises,
        "conclusion": conclusion,
        "points": problem.points
    }


def convert_all_problems():
    """Convert all IMO problems to solver format."""
    problems = parse_imo_file("data/imo_ag_30.txt")
    
    solver_problems = []
    for problem in problems:
        solver_prob = convert_imo_to_solver_format(problem)
        solver_problems.append(solver_prob)
    
    # Save to JSON
    with open("outputs/imo_problems.json", "w") as f:
        json.dump(solver_problems, f, indent=2)
    
    print(f"Converted {len(solver_problems)} problems")
    return solver_problems


if __name__ == "__main__":
    convert_all_problems()