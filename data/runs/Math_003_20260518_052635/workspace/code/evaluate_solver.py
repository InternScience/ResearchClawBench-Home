#!/usr/bin/env python3
"""
Evaluate the geometry solver on IMO problems.
"""

import json
import time
from typing import List, Dict
from geometry_solver import GeometrySolver


def load_problems(filepath: str) -> List[dict]:
    """Load problems from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def evaluate_solver(problems: List[dict]) -> Dict:
    """Evaluate the solver on all problems."""
    solver = GeometrySolver()
    
    results = {
        "total_problems": len(problems),
        "solved": 0,
        "unsolved": 0,
        "details": [],
        "statistics": {
            "avg_derived_facts": 0,
            "avg_time": 0,
            "total_derived_facts": 0,
            "total_time": 0
        }
    }
    
    for problem in problems:
        start_time = time.time()
        
        try:
            result = solver.solve_problem(problem)
            elapsed = time.time() - start_time
            
            problem_result = {
                "name": problem["name"],
                "solved": result["solved"],
                "derived_facts": result["derived_facts"],
                "new_facts": result["new_facts_derived"],
                "time": elapsed,
                "trace_length": len(result["trace"])
            }
            
            results["details"].append(problem_result)
            
            if result["solved"]:
                results["solved"] += 1
            else:
                results["unsolved"] += 1
            
            results["statistics"]["total_derived_facts"] += result["derived_facts"]
            results["statistics"]["total_time"] += elapsed
            
        except Exception as e:
            elapsed = time.time() - start_time
            results["details"].append({
                "name": problem["name"],
                "solved": False,
                "error": str(e),
                "time": elapsed
            })
            results["unsolved"] += 1
    
    # Compute averages
    if results["total_problems"] > 0:
        results["statistics"]["avg_derived_facts"] = (
            results["statistics"]["total_derived_facts"] / results["total_problems"]
        )
        results["statistics"]["avg_time"] = (
            results["statistics"]["total_time"] / results["total_problems"]
        )
    
    return results


def print_results(results: Dict):
    """Print evaluation results."""
    print(f"IMO Geometry Solver Evaluation Results")
    print(f"=" * 50)
    print(f"Total problems: {results['total_problems']}")
    print(f"Solved: {results['solved']} ({results['solved']/results['total_problems']*100:.1f}%)")
    print(f"Unsolved: {results['unsolved']}")
    print(f"Average derived facts: {results['statistics']['avg_derived_facts']:.1f}")
    print(f"Average time: {results['statistics']['avg_time']:.3f}s")
    print()
    
    print("Problem Details:")
    print("-" * 50)
    for detail in results["details"]:
        status = "SOLVED" if detail["solved"] else "FAILED"
        print(f"{detail['name']}: {status} ({detail.get('derived_facts', 0)} facts, {detail.get('time', 0):.3f}s)")


if __name__ == "__main__":
    problems = load_problems("outputs/imo_problems.json")
    results = evaluate_solver(problems)
    
    # Save results
    with open("outputs/solver_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print_results(results)