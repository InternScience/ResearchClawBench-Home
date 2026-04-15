import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

yaml_path = '../data/2111.01152/2111.01152.yaml'
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

tasks = []
scores = []
categories = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']

for item in data:
    if 'task' in item:
        task_name = item['task']
        task_scores = item.get('score', {})
        avg_task_score = np.mean([task_scores.get(cat, 0) for cat in categories])
        placeholder_scores = {}
        if 'placeholder' in item:
            for k, v in item['placeholder'].items():
                if isinstance(v, dict) and 'score' in v:
                    humans = [v['score'].get(h, 0) for h in ['Haining', 'Will', 'Yasaman'] if h in v['score']]
                    placeholder_scores[k] = np.mean(humans) if humans else 0
        tasks.append({'task': task_name, 'avg_score': avg_task_score, **task_scores})
        scores.append(task_scores)

df = pd.DataFrame(tasks)
df.to_csv('../outputs/task_scores.csv', index=False)
stats = df[categories].agg(['mean', 'std']).to_dict()
json.dump({'stats': stats, 'num_tasks': len(tasks), 'task_names': df['task'].tolist()}, open('../outputs/task_stats.json', 'w'))

# Plot
os.makedirs('../report/images', exist_ok=True)
plt.figure(figsize=(10,6))
df_melt = df.melt(id_vars=['task'], value_vars=categories, var_name='category', value_name='score')
sns.barplot(data=df_melt, x='category', y='score')
plt.title('Average Scores per Category')
plt.savefig('../report/images/score_bar.png')
plt.close()

print('Parsed', len(tasks), 'tasks')