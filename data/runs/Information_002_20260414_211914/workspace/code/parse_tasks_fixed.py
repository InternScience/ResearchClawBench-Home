import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

yaml_path = '../data/2111.01152/2111.01152.yaml'
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

tasks = []
categories = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']

for item in data:
    if 'task' in item:
        task_name = item['task']
        task_scores = item.get('score', {})
        numeric_scores = []
        for cat in categories:
            score = task_scores.get(cat, 0)
            numeric_scores.append(float(score) if isinstance(score, (int, float)) else 0)
        avg_task_score = np.mean(numeric_scores)
        tasks.append({'task': task_name, 'avg_score': avg_task_score, **{cat: float(task_scores.get(cat, 0)) if isinstance(task_scores.get(cat, 0), (int, float)) else 0 for cat in categories}})

df = pd.DataFrame(tasks)
df.to_csv('../outputs/task_scores.csv', index=False)
stats = df[categories].agg(['mean', 'std']).to_dict()
json.dump({'stats': stats, 'num_tasks': len(tasks), 'avg_overall': df['avg_score'].mean(), 'task_names': df['task'].tolist()}, open('../outputs/task_stats.json', 'w'), indent=2)

# Plot
os.makedirs('../report/images', exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 6))
df_melt = df.melt(id_vars=['task', 'avg_score'], value_vars=categories, var_name='category', value_name='score')
sns.barplot(data=df_melt, x='category', y='score', ax=ax)
ax.set_title('Average LLM Scores per Scoring Category (from YAML)')
ax.set_ylim(0, 2.5)
plt.tight_layout()
plt.savefig('../report/images/score_bar.png', dpi=300)
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 6))
df.boxplot(column='avg_score', by='task', ax=ax2)
ax2.set_title('Per-Task Average Scores')
plt.savefig('../report/images/per_task_box.png', dpi=300)
plt.close()

print(f'Parsed {len(tasks)} tasks, overall avg score: {df[\"avg_score\"].mean():.2f}')