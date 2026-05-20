"""
Analysis of LLM performance on Hartree-Fock multi-step calculations.
Data: 2111.01152.yaml with 16 tasks, 3 annotators, 6 scoring dimensions.
"""
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Load data
with open('data/2111.01152/2111.01152.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Filter out entries without tasks
tasks = [d for d in data if d.get('task')]
print(f"Number of tasks: {len(tasks)}")

# Extract scores
score_dimensions = ['in_paper', 'prompt_quality', 'follow_instructions', 
                    'physics_logic', 'math_derivation', 'final_answer_accuracy']
annotators = ['Haining', 'Will', 'Yasaman']

records = []
for task in tasks:
    task_name = task['task']
    score = task.get('score', {})
    record = {'task': task_name}
    for dim in score_dimensions:
        val = score.get(dim)
        if val is not None and not isinstance(val, str):
            record[dim] = val
        else:
            record[dim] = np.nan
    records.append(record)

df_task_scores = pd.DataFrame(records)
print("\nTask scores (overall, from YAML score block):")
print(df_task_scores.to_string())

# Extract placeholder-level scores from annotators
placeholder_records = []
for task in tasks:
    task_name = task['task']
    placeholders = task.get('placeholder', {})
    for ph_name, ph_data in placeholders.items():
        scores = ph_data.get('score', {})
        for annotator in annotators:
            val = scores.get(annotator)
            if val is not None and not isinstance(val, str):
                placeholder_records.append({
                    'task': task_name,
                    'placeholder': ph_name,
                    'annotator': annotator,
                    'score': val
                })

df_ph = pd.DataFrame(placeholder_records)
print(f"\nNumber of placeholder-level scored items: {len(df_ph)}")

# Aggregate placeholder scores per task per annotator
task_annotator_scores = df_ph.groupby(['task', 'annotator'])['score'].mean().reset_index()
task_annotator_pivot = task_annotator_scores.pivot(index='task', columns='annotator', values='score')
print("\nAverage placeholder scores per task per annotator:")
print(task_annotator_pivot.to_string())

# Overall statistics
print("\n--- Overall Task-Level Score Statistics ---")
print(df_task_scores[score_dimensions].describe())

print("\n--- Mean Score Per Dimension ---")
mean_dims = df_task_scores[score_dimensions].mean()
print(mean_dims)

print("\n--- Mean Score Per Task ---")
mean_tasks = df_task_scores[score_dimensions].mean(axis=1)
mean_tasks.index = df_task_scores['task']
print(mean_tasks)

# Save outputs
outputs = {
    'task_scores': df_task_scores.to_dict(orient='records'),
    'placeholder_scores': df_ph.to_dict(orient='records'),
    'mean_per_dimension': mean_dims.to_dict(),
    'mean_per_task': mean_tasks.to_dict(),
    'task_annotator_pivot': task_annotator_pivot.to_dict()
}

with open('outputs/results.json', 'w') as f:
    json.dump(outputs, f, indent=2, default=str)

df_task_scores.to_csv('outputs/task_scores.csv', index=False)
df_ph.to_csv('outputs/placeholder_scores.csv', index=False)
task_annotator_pivot.to_csv('outputs/task_annotator_scores.csv')

print("\nSaved outputs/results.json, outputs/task_scores.csv, outputs/placeholder_scores.csv")
