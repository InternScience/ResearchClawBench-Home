#!/usr/bin/env python3
"""
Analysis of Hartree-Fock LLM Evaluation for paper 2111.01152
AB-stacked MoTe2/WSe2 moiré system
"""

import yaml
import json
import os
import numpy as np

# Parse the YAML file
yaml_path = 'data/2111.01152/2111.01152.yaml'
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

print(f"Total entries in YAML: {len(data)}")
print()

# Separate branch markers from tasks
tasks = []
branches = []
for entry in data:
    if 'branch' in entry:
        branches.append(entry['branch'])
    elif 'task' in entry:
        tasks.append(entry)

print(f"Number of branches: {len(branches)}")
print(f"Number of tasks: {len(tasks)}")
print()

# List all task names
print("=== Task Names ===")
for i, task in enumerate(tasks):
    print(f"  {i+1}. {task['task']}")
print()

# Analyze scores per task
print("=== Task-Level Scores ===")
score_categories = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']

task_scores = []
for i, task in enumerate(tasks):
    if 'score' in task:
        scores = task['score']
        task_info = {
            'task_name': task['task'],
            'task_index': i+1,
            'scores': {}
        }
        for cat in score_categories:
            if cat in scores:
                task_info['scores'][cat] = scores[cat]
        task_scores.append(task_info)
        print(f"  Task {i+1}: {task['task']}")
        for cat, val in task_info['scores'].items():
            print(f"    {cat}: {val}")
        print()

# Analyze placeholder scores per task
print("\n=== Placeholder Extraction Scores ===")
placeholder_data = []
for i, task in enumerate(tasks):
    if 'placeholder' in task:
        for ph_name, ph_val in task['placeholder'].items():
            if isinstance(ph_val, dict) and 'score' in ph_val:
                scores = ph_val['score']
                llm_val = ph_val.get('LLM', None)
                human_val = ph_val.get('human', None)
                
                # Extract individual grader scores
                grader_scores = {}
                for grader in ['Haining', 'Will', 'Yasaman']:
                    if grader in scores:
                        val = scores[grader]
                        if isinstance(val, (int, float)):
                            grader_scores[grader] = val
                
                placeholder_data.append({
                    'task_index': i+1,
                    'task_name': task['task'],
                    'placeholder': ph_name,
                    'llm_value': llm_val,
                    'human_value': human_val,
                    'grader_scores': grader_scores,
                    'mean_score': np.mean(list(grader_scores.values())) if grader_scores else None
                })

print(f"Total placeholder entries with scores: {len(placeholder_data)}")

# Save structured data
output = {
    'num_tasks': len(tasks),
    'num_branches': len(branches),
    'branches': branches,
    'task_names': [t['task'] for t in tasks],
    'task_scores': task_scores,
    'num_placeholders': len(placeholder_data),
    'placeholder_summary': placeholder_data
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/parsed_data.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print("\nData saved to outputs/parsed_data.json")

# Compute summary statistics
print("\n=== Summary Statistics ===")

# Task-level score averages
for cat in score_categories:
    vals = [ts['scores'][cat] for ts in task_scores if cat in ts['scores']]
    if vals:
        print(f"  {cat}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, min={min(vals)}, max={max(vals)}")

# Placeholder-level score averages per grader
print("\n=== Placeholder Scores by Grader ===")
for grader in ['Haining', 'Will', 'Yasaman']:
    vals = [pd['grader_scores'][grader] for pd in placeholder_data if grader in pd['grader_scores']]
    if vals:
        print(f"  {grader}: n={len(vals)}, mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")
        # Distribution
        for score_val in [0, 1, 2]:
            count = vals.count(score_val)
            print(f"    Score {score_val}: {count} ({100*count/len(vals):.1f}%)")

# Score distribution overall
all_ph_scores = [pd['mean_score'] for pd in placeholder_data if pd['mean_score'] is not None]
print(f"\n  Overall placeholder mean scores: n={len(all_ph_scores)}, mean={np.mean(all_ph_scores):.2f}, std={np.std(all_ph_scores):.2f}")

# Perfect scores (all graders give 2)
perfect = sum(1 for pd in placeholder_data if pd['mean_score'] == 2.0)
print(f"  Perfect score (2.0 from all graders): {perfect}/{len(all_ph_scores)} ({100*perfect/len(all_ph_scores):.1f}%)")

# Zero scores (all graders give 0)
zero = sum(1 for pd in placeholder_data if pd['mean_score'] == 0.0)
print(f"  Zero score (0.0 from all graders): {zero}/{len(all_ph_scores)} ({100*zero/len(all_ph_scores):.1f}%)")
