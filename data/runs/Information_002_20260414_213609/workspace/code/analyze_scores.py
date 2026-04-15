#!/usr/bin/env python3
"""
Analysis of Hartree-Fock calculation steps from paper 2111.01152
Evaluating LLM performance on research-level theoretical physics calculations.
"""

import json
import os
import yaml
import re

# Create outputs directory
os.makedirs("outputs", exist_ok=True)
os.makedirs("report/images", exist_ok=True)

# Parse the YAML file to extract scoring data
with open("data/2111.01152/2111.01152.yaml", "r") as f:
    content = f.read()

# The YAML is a list of task entries
# Let's parse it more carefully
tasks_data = []
current_task = None

lines = content.split('\n')
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith('- task:'):
        if current_task:
            tasks_data.append(current_task)
        task_name = line.replace('- task:', '').strip()
        current_task = {'task': task_name, 'scores': {}, 'placeholders': {}}
    elif current_task and 'score:' in line and 'Haining' not in line and 'Will' not in line and 'Yasaman' not in line:
        # This is a task-level score section
        pass
    i += 1

if current_task:
    tasks_data.append(current_task)

# Manual extraction of task scores from the YAML
task_scores = [
    {
        "task": "Construct Kinetic Hamiltonian (continuum version, single-particle)",
        "in_paper": 1, "prompt_quality": 1, "follow_instructions": 1,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 1,
        "reviewer_scores": {"Haining": [2,0,0,1,1,2,2], "Will": [2,0,0,2,2,2,1], "Yasaman": [2,2,0,1,1,2,2]}
    },
    {
        "task": "Define each term in Kinetic Hamiltonian (continuum version)",
        "in_paper": 2, "prompt_quality": 2, "follow_instructions": 1,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 1,
        "reviewer_scores": {"Haining": [2,0,2,1,1,2,2], "Will": [0,0,2,0,1,2,2], "Yasaman": [2,2,2,1,1,1,1]}
    },
    {
        "task": "Construct Potential Hamiltonian (continuum version)",
        "in_paper": 1, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,0,0,1,2,2,2], "Will": [2,0,1,1,2,2,2], "Yasaman": [2,0,2,2,2,2,1]}
    },
    {
        "task": "Define each term in Potential Hamiltonian (continuum version)",
        "in_paper": 2, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,1,1,2,2], "Will": [2,2,2,1,2,2,2], "Yasaman": [0,2,2,2,2,2,1]}
    },
    {
        "task": "Convert from single-particle to second-quantized form, return in matrix",
        "in_paper": 2, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,2,2,2,2], "Will": [2,2,1,1,2,2,2], "Yasaman": [2,2,2,2,2,2,1]}
    },
    {
        "task": "Convert from single-particle to second-quantized form, return in summation",
        "in_paper": 2, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 1, "final_answer_accuracy": 1,
        "reviewer_scores": {"Haining": [2,0,2,2,2,2,2], "Will": [0,0,0,2,0,2,2], "Yasaman": [2,2,2,2,2,2,2]}
    },
    {
        "task": "Convert noninteracting Hamiltonian in real space to momentum space",
        "in_paper": 2, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 1,
        "reviewer_scores": {"Haining": [2,0,2,2,2,2,2], "Will": [2,2,2,2,2,2,2], "Yasaman": [2,0,2,2,2,2,2]}
    },
    {
        "task": "Particle-hole transformation",
        "in_paper": 0, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,2,2,2,2], "Will": [2,2,2,2,2,2,2], "Yasaman": [2,2,2,2,2,2,1]}
    },
    {
        "task": "Simplify the Hamiltonian in the particle-hole basis",
        "in_paper": 2, "prompt_quality": 1, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,2,1,2,2], "Will": [2,2,2,1,2,2,2], "Yasaman": [2,2,2,2,2,2,2]}
    },
    {
        "task": "Construct interaction Hamiltonian (momentum space)",
        "in_paper": 2, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,2,2,2,2], "Will": [2,2,2,2,2,2,2], "Yasaman": [2,2,2,2,2,2,2]}
    },
    {
        "task": "Wick's theorem",
        "in_paper": 0, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,2,2,2,2], "Will": [2,2,2,2,2,2,2], "Yasaman": [2,2,2,2,2,2,2]}
    },
    {
        "task": "Extract quadratic term",
        "in_paper": 0, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,2,2,2,2], "Will": [2,2,2,2,2,2,2], "Yasaman": [2,2,2,2,2,2,2]}
    },
    {
        "task": "Swap the index to combine Hartree and Fock terms",
        "in_paper": 2, "prompt_quality": 1, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,1,1,2,2], "Will": [2,2,2,1,2,2,2], "Yasaman": [2,2,2,1,1,2,2]}
    },
    {
        "task": "Reduce momentum in Hartree term",
        "in_paper": 2, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 1, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,2,1,2,2], "Will": [2,2,2,2,2,2,2], "Yasaman": [2,2,2,2,2,2,2]}
    },
    {
        "task": "Reduce momentum in Fock term",
        "in_paper": 2, "prompt_quality": 2, "follow_instructions": 2,
        "physics_logic": 2, "math_derivation": 2, "final_answer_accuracy": 2,
        "reviewer_scores": {"Haining": [2,2,2,2,2,2,2], "Will": [2,2,2,2,2,2,2], "Yasaman": [2,2,2,2,2,2,2]}
    },
]

# Save structured data
with open("outputs/task_scores.json", "w") as f:
    json.dump(task_scores, f, indent=2)

# Compute aggregate statistics
score_categories = ["in_paper", "prompt_quality", "follow_instructions", 
                    "physics_logic", "math_derivation", "final_answer_accuracy"]

summary = {}
for cat in score_categories:
    values = [t[cat] for t in task_scores]
    summary[cat] = {
        "mean": round(sum(values)/len(values), 2),
        "min": min(values),
        "max": max(values),
        "perfect_rate": round(sum(1 for v in values if v == 2)/len(values)*100, 1)
    }

# Overall score
all_scores = []
for t in task_scores:
    total = sum(t[cat] for cat in score_categories)
    all_scores.append(total)
    
summary["overall"] = {
    "mean_total": round(sum(all_scores)/len(all_scores), 2),
    "max_possible": len(score_categories) * 2,
    "mean_percentage": round(sum(all_scores)/(len(all_scores)*len(score_categories)*2)*100, 1),
    "perfect_tasks": sum(1 for s in all_scores if s == len(score_categories)*2)
}

# Reviewer agreement analysis
reviewer_agreement = {}
for cat_idx, cat in enumerate(score_categories):
    reviewer_vals = {"Haining": [], "Will": [], "Yasaman": []}
    for t in task_scores:
        for reviewer in reviewer_vals:
            if reviewer in t.get("reviewer_scores", {}):
                reviewer_vals[reviewer].append(t["reviewer_scores"][reviewer][cat_idx])
    
    reviewer_means = {}
    for reviewer, vals in reviewer_vals.items():
        if vals:
            reviewer_means[reviewer] = round(sum(vals)/len(vals), 2)
    reviewer_agreement[cat] = reviewer_means

summary["reviewer_means"] = reviewer_agreement

with open("outputs/summary_statistics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("=== Task Score Summary ===")
for cat in score_categories:
    s = summary[cat]
    print(f"{cat}: mean={s['mean']}, perfect_rate={s['perfect_rate']}%")
print(f"\nOverall: {summary['overall']['mean_percentage']}% accuracy")
print(f"Perfect tasks: {summary['overall']['perfect_tasks']}/{len(task_scores)}")

# Extract paper metadata
paper_info = {
    "arxiv_id": "2111.01152",
    "title": "Topological Phases in AB-Stacked MoTe2/WSe2",
    "system": "AB-stacked MoTe2/WSe2 moiré heterobilayer",
    "method": "Self-consistent Hartree-Fock in plane-wave basis",
    "key_phases": ["Z2 topological insulator", "Chern insulator", "Topological charge density wave"],
    "filling_factors": [1, 2, "2/3"],
    "num_calculation_steps": len(task_scores)
}

with open("outputs/paper_info.json", "w") as f:
    json.dump(paper_info, f, indent=2)

print("\nPaper info saved.")
print(f"Total calculation steps analyzed: {len(task_scores)}")
