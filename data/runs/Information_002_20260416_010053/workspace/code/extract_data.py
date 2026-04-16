import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def extract_scores(data):
    task_scores = []
    placeholder_scores = []
    
    for i, task in enumerate(data):
        task_name = task.get('task', f'Task {i+1}')
        
        # Extract task-level scores
        scores = task.get('score', {})
        task_info = {
            'task_id': i + 1,
            'task_name': task_name,
            'in_paper': scores.get('in_paper', None),
            'prompt_quality': scores.get('prompt_quality', None),
            'follow_instructions': scores.get('follow_instructions', None),
            'physics_logic': scores.get('physics_logic', None),
            'math_derivation': scores.get('math_derivation', None),
            'final_answer_accuracy': scores.get('final_answer_accuracy', None)
        }
        task_scores.append(task_info)
        
        # Extract placeholder-level scores
        placeholders = task.get('placeholder', {})
        for ph_name, ph_data in placeholders.items():
            ph_scores = ph_data.get('score', {})
            for evaluator, score in ph_scores.items():
                ph_info = {
                    'task_id': i + 1,
                    'task_name': task_name,
                    'placeholder': ph_name,
                    'evaluator': evaluator,
                    'score': score
                }
                placeholder_scores.append(ph_info)
                
    return pd.DataFrame(task_scores), pd.DataFrame(placeholder_scores)

if __name__ == '__main__':
    data = load_data('../data/2111.01152/2111.01152.yaml')
    df_tasks, df_placeholders = extract_scores(data)
    
    df_tasks.to_csv('../outputs/task_scores.csv', index=False)
    df_placeholders.to_csv('../outputs/placeholder_scores.csv', index=False)
    
    print("Task Scores Shape:", df_tasks.shape)
    print("Placeholder Scores Shape:", df_placeholders.shape)
