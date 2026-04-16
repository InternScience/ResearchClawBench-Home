import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df_tasks = pd.read_csv('../outputs/task_scores.csv')
df_tasks = df_tasks.dropna(subset=['in_paper']) # Drop the first row which is a branch marker

# 1. Heatmap of Task Scores
plt.figure(figsize=(10, 8))
score_cols = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']
sns.heatmap(df_tasks.set_index('task_id')[score_cols], annot=True, cmap='YlGnBu', vmin=0, vmax=2)
plt.title('Scores by Task and Evaluation Aspect')
plt.ylabel('Task ID')
plt.xlabel('Evaluation Aspect')
plt.tight_layout()
plt.savefig('../report/images/task_scores_heatmap.png')
plt.close()

# 2. Average Scores per Aspect
plt.figure(figsize=(10, 6))
avg_scores = df_tasks[score_cols].mean()
sns.barplot(x=avg_scores.index, y=avg_scores.values, palette='viridis')
plt.title('Average Scores by Evaluation Aspect')
plt.ylabel('Average Score (0-2)')
plt.xlabel('Evaluation Aspect')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 2.1)
for i, v in enumerate(avg_scores.values):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('../report/images/avg_scores_aspect.png')
plt.close()

# 3. Placeholder Extraction Scores
df_placeholders = pd.read_csv('../outputs/placeholder_scores.csv')
# Clean score column to be numeric, replacing '(?)' with NaN
df_placeholders['score'] = pd.to_numeric(df_placeholders['score'], errors='coerce')

plt.figure(figsize=(10, 6))
sns.countplot(data=df_placeholders.dropna(subset=['score']), x='score', hue='evaluator', palette='Set2')
plt.title('Distribution of Placeholder Extraction Scores by Evaluator')
plt.xlabel('Score (0=Incorrect, 1=Partial, 2=Correct)')
plt.ylabel('Count')
plt.legend(title='Evaluator')
plt.tight_layout()
plt.savefig('../report/images/placeholder_scores_dist.png')
plt.close()

# 4. Placeholder Scores Average per Task
plt.figure(figsize=(12, 6))
avg_ph_task = df_placeholders.dropna(subset=['score']).groupby('task_id')['score'].mean().reset_index()
sns.barplot(data=avg_ph_task, x='task_id', y='score', color='skyblue')
plt.title('Average Placeholder Extraction Score per Task')
plt.xlabel('Task ID')
plt.ylabel('Average Score')
plt.ylim(0, 2.1)
plt.axhline(df_placeholders['score'].mean(), color='red', linestyle='--', label='Overall Average')
plt.legend()
plt.tight_layout()
plt.savefig('../report/images/avg_placeholder_scores_task.png')
plt.close()

