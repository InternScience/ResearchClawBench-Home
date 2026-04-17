#!/usr/bin/env python3
"""
Detailed analysis: LLM vs Human answers, error categorization, and step-by-step comparison
"""

import yaml
import json
import os
import numpy as np
from collections import defaultdict

yaml_path = 'data/2111.01152/2111.01152.yaml'
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

tasks = [entry for entry in data if 'task' in entry]
os.makedirs('outputs', exist_ok=True)

# ============================================================
# 1. Detailed per-task analysis with LLM vs Human comparison
# ============================================================
detailed_analysis = []

for i, task in enumerate(tasks):
    task_info = {
        'step': i+1,
        'task_name': task['task'],
        'has_answer': 'answer' in task and task['answer'] is not None,
        'answer_preview': str(task.get('answer', ''))[:200] if task.get('answer') else None,
        'task_scores': task.get('score', {}),
        'placeholders': [],
        'error_analysis': []
    }
    
    if 'placeholder' in task:
        for ph_name, ph_val in task['placeholder'].items():
            if isinstance(ph_val, dict):
                llm = ph_val.get('LLM', None)
                human = ph_val.get('human', None)
                scores = {}
                if 'score' in ph_val:
                    for g in ['Haining', 'Will', 'Yasaman']:
                        if g in ph_val['score'] and isinstance(ph_val['score'][g], (int, float)):
                            scores[g] = ph_val['score'][g]
                
                mean_score = np.mean(list(scores.values())) if scores else None
                
                ph_info = {
                    'name': ph_name,
                    'llm': str(llm) if llm is not None else None,
                    'human': str(human) if human is not None else None,
                    'scores': scores,
                    'mean_score': float(mean_score) if mean_score is not None else None,
                    'has_human_correction': human is not None and str(human).strip() != '',
                    'llm_empty': llm is None or str(llm).strip() == '',
                }
                
                # Error categorization
                if mean_score is not None:
                    if mean_score == 0:
                        ph_info['error_type'] = 'completely_wrong'
                    elif mean_score < 1:
                        ph_info['error_type'] = 'mostly_wrong'
                    elif mean_score < 2:
                        ph_info['error_type'] = 'partial'
                    else:
                        ph_info['error_type'] = 'correct'
                
                task_info['placeholders'].append(ph_info)
    
    detailed_analysis.append(task_info)

# ============================================================
# 2. Key findings: Where LLM fails
# ============================================================
print("=== KEY FAILURE CASES ===")
failures = []
for task_info in detailed_analysis:
    for ph in task_info['placeholders']:
        if ph.get('mean_score') is not None and ph['mean_score'] < 1.0:
            failures.append({
                'step': task_info['step'],
                'task': task_info['task_name'],
                'placeholder': ph['name'],
                'llm': ph['llm'],
                'human': ph['human'],
                'mean_score': ph['mean_score'],
                'scores': ph['scores']
            })
            print(f"\nStep {task_info['step']}: {task_info['task_name']}")
            print(f"  Placeholder: {ph['name']}")
            print(f"  LLM: {ph['llm']}")
            print(f"  Human: {ph['human']}")
            print(f"  Scores: {ph['scores']} (mean={ph['mean_score']:.2f})")

print(f"\nTotal failure cases (mean < 1.0): {len(failures)}")

# ============================================================
# 3. Task-level summary table
# ============================================================
print("\n=== TASK SUMMARY TABLE ===")
print(f"{'Step':>4} | {'Task Name':<55} | {'#PH':>3} | {'Mean PH':>7} | {'Final':>5} | {'In Paper':>8}")
print("-" * 100)

task_summary = []
for task_info in detailed_analysis:
    ph_scores = [ph['mean_score'] for ph in task_info['placeholders'] if ph['mean_score'] is not None]
    mean_ph = np.mean(ph_scores) if ph_scores else None
    final = task_info['task_scores'].get('final_answer_accuracy', None)
    in_paper = task_info['task_scores'].get('in_paper', None)
    
    row = {
        'step': task_info['step'],
        'task': task_info['task_name'],
        'num_placeholders': len(task_info['placeholders']),
        'mean_placeholder_score': float(mean_ph) if mean_ph is not None else None,
        'final_answer_accuracy': final,
        'in_paper': in_paper
    }
    task_summary.append(row)
    
    print(f"{task_info['step']:>4} | {task_info['task_name']:<55} | {len(task_info['placeholders']):>3} | "
          f"{mean_ph:>7.2f}" if mean_ph is not None else f"{'N/A':>7}" + f" | {final if final is not None else 'N/A':>5} | {in_paper if in_paper is not None else 'N/A':>8}")

# ============================================================
# 4. Categorize common error types
# ============================================================
print("\n=== ERROR TYPE ANALYSIS ===")
error_categories = {
    'wrong_space': 0,  # real vs momentum
    'wrong_quantization': 0,  # single-particle vs second-quantized
    'wrong_particles': 0,  # electrons vs holes
    'incomplete_dof': 0,  # missing degrees of freedom
    'wrong_expression': 0,  # wrong mathematical expression
    'missing_info': 0,  # LLM didn't extract
    'other': 0
}

for f in failures:
    ph = f['placeholder']
    if 'real|momentum' in ph:
        error_categories['wrong_space'] += 1
    elif 'single-particle|second-quantized' in ph:
        error_categories['wrong_quantization'] += 1
    elif 'electrons|holes' in ph:
        error_categories['wrong_particles'] += 1
    elif f['llm'] is None or f['llm'] == '' or f['llm'] == 'None':
        error_categories['missing_info'] += 1
    else:
        error_categories['other'] += 1

for cat, count in error_categories.items():
    if count > 0:
        print(f"  {cat}: {count}")

# ============================================================
# 5. Save all outputs
# ============================================================
with open('outputs/detailed_analysis.json', 'w') as f:
    json.dump(detailed_analysis, f, indent=2, default=str)

with open('outputs/failure_cases.json', 'w') as f:
    json.dump(failures, f, indent=2, default=str)

with open('outputs/task_summary.json', 'w') as f:
    json.dump(task_summary, f, indent=2, default=str)

# ============================================================
# 6. Compute overall metrics
# ============================================================
overall_metrics = {
    'total_tasks': len(tasks),
    'total_placeholders': sum(len(t['placeholders']) for t in detailed_analysis),
    'total_failures': len(failures),
    'task_level': {
        cat: {
            'mean': float(np.mean([t['task_scores'][cat] for t in detailed_analysis if cat in t['task_scores']])),
            'values': [t['task_scores'].get(cat) for t in detailed_analysis]
        }
        for cat in ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']
    },
    'placeholder_level': {
        'overall_mean': float(np.mean([ph['mean_score'] for t in detailed_analysis for ph in t['placeholders'] if ph['mean_score'] is not None])),
        'perfect_rate': float(sum(1 for t in detailed_analysis for ph in t['placeholders'] if ph.get('mean_score') == 2.0) / 
                             max(1, sum(1 for t in detailed_analysis for ph in t['placeholders'] if ph['mean_score'] is not None))),
        'failure_rate': float(len(failures) / max(1, sum(1 for t in detailed_analysis for ph in t['placeholders'] if ph['mean_score'] is not None)))
    }
}

with open('outputs/overall_metrics.json', 'w') as f:
    json.dump(overall_metrics, f, indent=2, default=str)

print("\nAll outputs saved to outputs/")
print(f"\nOverall Metrics:")
print(f"  Total tasks: {overall_metrics['total_tasks']}")
print(f"  Total placeholders: {overall_metrics['total_placeholders']}")
print(f"  Placeholder mean score: {overall_metrics['placeholder_level']['overall_mean']:.2f}")
print(f"  Perfect rate: {overall_metrics['placeholder_level']['perfect_rate']:.1%}")
print(f"  Failure rate: {overall_metrics['placeholder_level']['failure_rate']:.1%}")
print(f"  Physics logic (always 2): {overall_metrics['task_level']['physics_logic']['mean']:.2f}")
print(f"  Final answer accuracy: {overall_metrics['task_level']['final_answer_accuracy']['mean']:.2f}")
