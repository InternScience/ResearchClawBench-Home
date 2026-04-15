#!/usr/bin/env python3
"""
Parse IMO geometry problems from imo_ag_30.txt into structured format.
Extracts problem names, construction sequences, and target conclusions.
"""

import json
import re
import os

def parse_problem_line(line):
    """Parse a single problem definition line."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    # Split into premises and conclusion
    parts = line.split(' ? ')
    if len(parts) != 2:
        return None
    
    premise_str, conclusion_str = parts
    
    # Parse individual construction steps separated by '; '
    constructions = []
    for step in premise_str.split('; '):
        step = step.strip()
        if not step:
            continue
        # Each step has form: "var1 var2 ... = predicate args..."
        eq_idx = step.find('=')
        if eq_idx == -1:
            continue
        vars_part = step[:eq_idx].strip()
        pred_part = step[eq_idx+1:].strip()
        
        # Split variables
        variables = [v.strip() for v in vars_part.split() if '@' not in v]
        # Handle coordinate annotations like x@4.96_-0.13
        annotated_vars = re.findall(r'(\w+)@[\d._-]+', vars_part)
        variables.extend(annotated_vars)
        
        # Parse predicate and arguments
        pred_parts = pred_part.split(', ')
        predicates = []
        for pp in pred_parts:
            pp = pp.strip()
            tokens = pp.split()
            if tokens:
                predicates.append({
                    'predicate': tokens[0],
                    'args': tokens[1:]
                })
        
        constructions.append({
            'variables': variables,
            'predicates': predicates,
            'raw': step
        })
    
    # Parse conclusion
    conclusion_tokens = conclusion_str.strip().split()
    conclusion = {
        'predicate': conclusion_tokens[0],
        'args': conclusion_tokens[1:]
    }
    
    return {
        'constructions': constructions,
        'conclusion': conclusion
    }


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'imo_ag_30.txt')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'parsed_problems.json')
    
    problems = {}
    current_name = None
    
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a problem name line
            if line.startswith('translated_imo_') and '=' not in line and '?' not in line:
                current_name = line
                problems[current_name] = {'raw_name': line}
            elif current_name and (' ? ' in line):
                parsed = parse_problem_line(line)
                if parsed:
                    problems[current_name].update(parsed)
    
    # Save parsed problems
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(problems, f, indent=2)
    
    print(f"Parsed {len(problems)} problems")
    print(f"Output saved to {output_path}")
    
    # Print summary statistics
    pred_counts = {}
    concl_counts = {}
    num_constructions = []
    
    for name, prob in problems.items():
        n_constr = len(prob.get('constructions', []))
        num_constructions.append(n_constr)
        
        for constr in prob.get('constructions', []):
            for pred in constr.get('predicates', []):
                p = pred['predicate']
                pred_counts[p] = pred_counts.get(p, 0) + 1
        
        concl = prob.get('conclusion', {})
        cp = concl.get('predicate', '')
        concl_counts[cp] = concl_counts.get(cp, 0) + 1
    
    print("\n=== Construction Predicate Distribution ===")
    for p, c in sorted(pred_counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c}")
    
    print("\n=== Conclusion Type Distribution ===")
    for p, c in sorted(concl_counts.items(), key=lambda x: -x[1]):
        print(f"  {p}: {c}")
    
    print(f"\n=== Construction Count Statistics ===")
    print(f"  Min: {min(num_constructions)}, Max: {max(num_constructions)}")
    print(f"  Mean: {sum(num_constructions)/len(num_constructions):.1f}")
    
    # Save stats
    stats = {
        'total_problems': len(problems),
        'predicate_distribution': pred_counts,
        'conclusion_distribution': concl_counts,
        'construction_count_stats': {
            'min': min(num_constructions),
            'max': max(num_constructions),
            'mean': sum(num_constructions)/len(num_constructions),
            'values': num_constructions
        },
        'problem_names': list(problems.keys())
    }
    
    stats_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'problem_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")


if __name__ == '__main__':
    main()
