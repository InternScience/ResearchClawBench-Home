#!/usr/bin/env python3
"""
Simple forward-chaining geometry theorem prover for IMO benchmark.
Uses predicate facts and a small set of inference rules extracted from data/rules.txt.
"""

import re
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Load rules (simplified subset for demo)
RULES = [
    ("perp_implies_para", ["perp A B C D"], ["para A B C D"]),
    ("cyclic_angle", ["cyclic A B C D"], ["eqangle A B C D"]),
    ("cong_trans", ["cong A B C D", "cong C D E F"], ["cong A B E F"]),
    ("simtri_aa", ["eqangle A B C D", "eqangle B C D E"], ["simtri A B C D E"]),
    ("midp_cong", ["midp M A B"], ["cong M A M B"]),
    ("orthocenter_foot", ["orthocenter H A B C"], ["foot F H B C"]),
]

def parse_problem(line):
    """Parse a problem line into premises and goal."""
    parts = line.strip().split('?')
    premises = parts[0].strip()
    goal = parts[1].strip() if len(parts) > 1 else ""
    facts = [f.strip() for f in premises.split(';') if f.strip()]
    return facts, goal

def forward_chain(facts, goal, max_steps=50):
    """Very simple forward inference."""
    fact_set = set(facts)
    for step in range(max_steps):
        new_facts = set()
        for rule_name, antecedents, consequents in RULES:
            if all(any(re.match(a.replace(' ', '.*'), f) for f in fact_set) for a in antecedents):
                for c in consequents:
                    new_facts.add(c)
        if not new_facts:
            break
        fact_set.update(new_facts)
        if any(re.match(goal.replace(' ', '.*'), f) for f in fact_set):
            return step + 1, True
    return max_steps, any(re.match(goal.replace(' ', '.*'), f) for f in fact_set)

def main():
    data_path = Path("data/imo_ag_30.txt")
    problems = [parse_problem(line) for line in data_path.read_text().splitlines() if line.strip()]

    results = []
    for i, (facts, goal) in enumerate(problems):
        steps, solved = forward_chain(facts, goal)
        results.append({
            "problem": f"imo_{i+1}",
            "solved": solved,
            "steps": steps if solved else None
        })

    # Save results
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Metrics
    solved_count = sum(r["solved"] for r in results)
    success_rate = solved_count / len(results)
    steps_list = [r["steps"] for r in results if r["steps"] is not None]

    print(f"Solved {solved_count}/{len(results)} problems ({success_rate:.1%})")
    if steps_list:
        print(f"Average proof length: {np.mean(steps_list):.1f} steps")

    # Figure 1: Success rate
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Solved", "Unsolved"], y=[solved_count, len(results)-solved_count])
    plt.title("IMO Geometry Benchmark Results")
    plt.ylabel("Number of Problems")
    plt.savefig("report/images/figure1_success.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Proof length distribution
    if steps_list:
        plt.figure(figsize=(6, 4))
        sns.histplot(steps_list, bins=10, kde=True)
        plt.title("Distribution of Proof Lengths")
        plt.xlabel("Steps to Solution")
        plt.savefig("report/images/figure2_proof_length.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Figure 3: Per-problem success
    plt.figure(figsize=(10, 4))
    colors = ["green" if r["solved"] else "red" for r in results]
    plt.bar(range(len(results)), [1]*len(results), color=colors)
    plt.title("Per-Problem Success (Green=Solved)")
    plt.xlabel("Problem Index")
    plt.savefig("report/images/figure3_per_problem.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Figures saved to report/images/")

if __name__ == "__main__":
    main()