#!/usr/bin/env python3
"""
Parse the YAML scoring data for paper 2111.01152 and extract all structured scores.
"""
import yaml
import json
import os
import sys
from collections import defaultdict

def parse_task_scores(yaml_path: str) -> list:
    """Parse all tasks from the YAML file and extract scores."""
    with open(yaml_path, 'r') as f:
        tasks = yaml.safe_load(f)
    
    parsed = []
    for idx, task in enumerate(tasks):
        entry = {
            'task_id': idx,
            'branch': task.get('branch', ''),
            'task_name': task.get('task', ''),
            'source': task.get('source', {}),
            'answer': task.get('answer', ''),
        }
        
        # Extract overall scores
        if 'score' in task:
            for key, value in task['score'].items():
                if isinstance(value, (int, float)):
                    entry[f'score_{key}'] = value
        
        # Extract placeholder-level scores (per annotator)
        if 'placeholder' in task:
            ph_scores = {}
            for ph_key, ph_val in task['placeholder'].items():
                if ph_val and 'score' in ph_val:
                    for annotator, score in ph_val['score'].items():
                        if score is not None and not isinstance(score, str):
                            if annotator not in ph_scores:
                                ph_scores[annotator] = []
                            ph_scores[annotator].append(score)
            
            # Compute per-annotator mean and count
            for annotator, scores in ph_scores.items():
                valid = [s for s in scores if isinstance(s, (int, float))]
                if valid:
                    entry[f'placeholder_{annotator}_mean'] = sum(valid) / len(valid)
                    entry[f'placeholder_{annotator}_count'] = len(valid)
                    entry[f'placeholder_{annotator}_total'] = sum(valid)
                    entry[f'placeholder_{annotator}_max'] = len(valid) * 3  # assuming max 3
        
        parsed.append(entry)
    
    return parsed


def extract_annotator_agreement(yaml_path: str) -> dict:
    """Extract per-placeholder per-annotator scores for agreement analysis."""
    with open(yaml_path, 'r') as f:
        tasks = yaml.safe_load(f)
    
    # Collect all placeholder-level scores
    agreements = []
    for task in tasks:
        task_name = task.get('task', '')
        if 'placeholder' in task:
            for ph_key, ph_val in task['placeholder'].items():
                if ph_val and 'score' in ph_val:
                    scores = ph_val['score']
                    # Check if we have multiple annotator scores
                    valid_scores = {}
                    for annotator, score in scores.items():
                        if isinstance(score, (int, float)):
                            valid_scores[annotator] = score
                    
                    if len(valid_scores) >= 2:
                        agreements.append({
                            'task': task_name,
                            'placeholder': ph_key,
                            'scores': valid_scores
                        })
    
    return agreements


def compute_task_level_scores(yaml_path: str) -> list:
    """Compute task-level overall scores from the YAML."""
    with open(yaml_path, 'r') as f:
        tasks = yaml.safe_load(f)
    
    task_scores = []
    for idx, task in enumerate(tasks):
        entry = {
            'task_id': idx,
            'task_name': task.get('task', ''),
            'branch': task.get('branch', ''),
        }
        if 'score' in task:
            for key, value in task['score'].items():
                if isinstance(value, (int, float)):
                    entry[key] = value
        task_scores.append(entry)
    
    return task_scores


if __name__ == '__main__':
    yaml_path = os.path.join(os.path.dirname(__file__), '..', 'data', '2111.01152', '2111.01152.yaml')
    
    parsed = parse_task_scores(yaml_path)
    agreements = extract_annotator_agreement(yaml_path)
    task_scores = compute_task_level_scores(yaml_path)
    
    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'parsed_tasks.json'), 'w') as f:
        json.dump(parsed, f, indent=2)
    
    with open(os.path.join(output_dir, 'annotator_agreements.json'), 'w') as f:
        json.dump(agreements, f, indent=2)
    
    with open(os.path.join(output_dir, 'task_scores.json'), 'w') as f:
        json.dump(task_scores, f, indent=2)
    
    print(f"Parsed {len(parsed)} tasks")
    print(f"Extracted {len(agreements)} placeholder-level agreement data points")
    print(f"Extracted {len(task_scores)} task-level score entries")
