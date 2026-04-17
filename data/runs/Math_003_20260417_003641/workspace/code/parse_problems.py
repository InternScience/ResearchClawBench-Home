"""
parse_problems.py - Parse and analyze the IMO AG 30 geometry problems
"""
import json
import re
from collections import Counter, defaultdict

def parse_problem(line):
    """Parse a single problem line into structured components."""
    parts = line.strip().split('?')
    if len(parts) != 2:
        return None
    
    construction_part = parts[0].strip()
    goal_part = parts[1].strip()
    
    # Parse constructions
    constructions = []
    steps = construction_part.split(';')
    for step in steps:
        step = step.strip()
        if not step:
            continue
        # Split into LHS = RHS
        if '=' in step:
            eq_parts = step.split('=', 1)
            lhs = eq_parts[0].strip()
            rhs = eq_parts[1].strip()
            
            # Extract construction names
            lhs_vars = lhs.split()
            
            # Extract construction types from RHS
            rhs_constructions = []
            rhs_parts = rhs.split(',')
            for rp in rhs_parts:
                rp = rp.strip()
                # Extract the construction function name
                tokens = rp.split()
                if tokens:
                    func_name = tokens[0]
                    args = tokens[1:]
                    rhs_constructions.append({
                        'function': func_name,
                        'args': args
                    })
            
            constructions.append({
                'variables': lhs_vars,
                'definitions': rhs_constructions
            })
        else:
            constructions.append({
                'variables': step.split(),
                'definitions': []
            })
    
    # Parse goal
    goal_tokens = goal_part.split()
    goal_predicate = goal_tokens[0] if goal_tokens else ''
    goal_args = goal_tokens[1:] if len(goal_tokens) > 1 else []
    
    return {
        'constructions': constructions,
        'goal': {
            'predicate': goal_predicate,
            'args': goal_args
        },
        'num_constructions': len(constructions),
        'raw_construction': construction_part,
        'raw_goal': goal_part
    }

def parse_definitions(filepath):
    """Parse the definitions file."""
    defs = {}
    current_name = None
    current_def = []
    
    with open(filepath) as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            if current_name and current_def:
                defs[current_name] = '\n'.join(current_def)
            current_name = None
            current_def = []
            i += 1
            continue
        
        if current_name is None:
            # First line of a definition block
            tokens = line.split()
            if tokens:
                current_name = tokens[0]
                current_def = [line]
        else:
            current_def.append(line)
        i += 1
    
    if current_name and current_def:
        defs[current_name] = '\n'.join(current_def)
    
    return defs

def parse_rules(filepath):
    """Parse the deduction rules."""
    rules = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '=>' in line:
                parts = line.split('=>')
                premises = [p.strip() for p in parts[0].split(',')]
                conclusions = [c.strip() for c in parts[1].split(',')]
                rules.append({
                    'premises': premises,
                    'conclusion': conclusions[0] if conclusions else '',
                    'raw': line
                })
    return rules

def extract_all_construction_functions(problems):
    """Extract all construction function names used across problems."""
    funcs = Counter()
    for name, prob in problems.items():
        for const in prob['constructions']:
            for defn in const['definitions']:
                funcs[defn['function']] += 1
    return funcs

def extract_all_predicates(rules):
    """Extract all predicates used in rules."""
    predicates = set()
    for rule in rules:
        for premise in rule['premises']:
            tokens = premise.split()
            if tokens:
                predicates.add(tokens[0])
        conclusion_tokens = rule['conclusion'].split()
        if conclusion_tokens:
            predicates.add(conclusion_tokens[0])
    return predicates

def count_points(problem):
    """Count the number of distinct points in a problem."""
    points = set()
    for const in problem['constructions']:
        for v in const['variables']:
            # Filter out coordinate annotations
            clean_v = v.split('@')[0]
            if clean_v and len(clean_v) <= 3:
                points.add(clean_v)
        for defn in const['definitions']:
            for arg in defn['args']:
                clean_arg = arg.split('@')[0]
                if clean_arg and len(clean_arg) <= 3:
                    points.add(clean_arg)
    return points

def main():
    # Parse problems
    problems = {}
    with open('data/imo_ag_30.txt') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            # This is a problem name
            name = line
            i += 1
            if i < len(lines):
                prob_line = lines[i].strip()
                parsed = parse_problem(prob_line)
                if parsed:
                    problems[name] = parsed
        i += 1
    
    # Parse definitions
    defs = parse_definitions('data/defs.txt')
    
    # Parse rules
    rules = parse_rules('data/rules.txt')
    
    print(f"Number of problems: {len(problems)}")
    print(f"Number of definitions: {len(defs)}")
    print(f"Number of rules: {len(rules)}")
    print()
    
    # Analyze problems
    goal_types = Counter()
    construction_counts = []
    point_counts = []
    construction_funcs = Counter()
    problem_details = []
    
    for name, prob in problems.items():
        goal_types[prob['goal']['predicate']] += 1
        construction_counts.append(prob['num_constructions'])
        
        points = count_points(prob)
        point_counts.append(len(points))
        
        for const in prob['constructions']:
            for defn in const['definitions']:
                construction_funcs[defn['function']] += 1
        
        problem_details.append({
            'name': name,
            'goal_type': prob['goal']['predicate'],
            'num_constructions': prob['num_constructions'],
            'num_points': len(points),
            'num_definitions': sum(len(c['definitions']) for c in prob['constructions']),
            'goal_args': prob['goal']['args']
        })
    
    print("Goal type distribution:")
    for gt, count in goal_types.most_common():
        print(f"  {gt}: {count}")
    
    print(f"\nConstruction counts: min={min(construction_counts)}, max={max(construction_counts)}, avg={sum(construction_counts)/len(construction_counts):.1f}")
    print(f"Point counts: min={min(point_counts)}, max={max(point_counts)}, avg={sum(point_counts)/len(point_counts):.1f}")
    
    print("\nTop 20 construction functions:")
    for func, count in construction_funcs.most_common(20):
        print(f"  {func}: {count}")
    
    # Rule analysis
    rule_predicates_premises = Counter()
    rule_predicates_conclusions = Counter()
    for rule in rules:
        for premise in rule['premises']:
            tokens = premise.split()
            if tokens:
                rule_predicates_premises[tokens[0]] += 1
        conclusion_tokens = rule['conclusion'].split()
        if conclusion_tokens:
            rule_predicates_conclusions[conclusion_tokens[0]] += 1
    
    print("\nRule premise predicates:")
    for pred, count in rule_predicates_premises.most_common():
        print(f"  {pred}: {count}")
    
    print("\nRule conclusion predicates:")
    for pred, count in rule_predicates_conclusions.most_common():
        print(f"  {pred}: {count}")
    
    # Save results
    results = {
        'num_problems': len(problems),
        'num_definitions': len(defs),
        'num_rules': len(rules),
        'goal_types': dict(goal_types),
        'construction_funcs': dict(construction_funcs.most_common(30)),
        'problem_details': problem_details,
        'rule_premise_predicates': dict(rule_predicates_premises),
        'rule_conclusion_predicates': dict(rule_predicates_conclusions),
        'construction_count_stats': {
            'min': min(construction_counts),
            'max': max(construction_counts),
            'mean': sum(construction_counts)/len(construction_counts),
            'values': construction_counts
        },
        'point_count_stats': {
            'min': min(point_counts),
            'max': max(point_counts),
            'mean': sum(point_counts)/len(point_counts),
            'values': point_counts
        }
    }
    
    with open('outputs/problem_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to outputs/problem_analysis.json")

if __name__ == '__main__':
    main()
