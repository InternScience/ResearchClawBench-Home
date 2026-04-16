#!/usr/bin/env python3
"""
Step 1 (fixed): Parse the IMO-AG-30 benchmark problems, definitions, and rules.
Each problem is on one or more lines: name line followed by construction+goal line(s).
"""

import json
import re
import os

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_180131"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")

def parse_problems(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    problems = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if this is a problem name (no = sign, no ?)
        if '=' not in line and '?' not in line:
            name = line
            # Collect all subsequent lines that belong to this problem
            body_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    break
                if '=' not in next_line and '?' not in next_line:
                    # This is the start of a new problem
                    break
                body_lines.append(next_line)
                i += 1
            
            # Join body lines and parse
            full_body = ' '.join(body_lines)
            
            # Split by ; to get construction statements
            statements = [s.strip() for s in full_body.split(';')]
            
            # Find the goal (contains ?)
            goal = None
            construction_stmts = []
            for stmt in statements:
                if '?' in stmt:
                    # Parse goal
                    goal_part = stmt.split('?')[1].strip()
                    m = re.match(r'(\w+)\s+(.*)', goal_part)
                    if m:
                        goal = {
                            'predicate': m.group(1),
                            'args': m.group(2).strip().split()
                        }
                elif stmt:
                    construction_stmts.append(stmt)
            
            # Parse each construction statement
            constructions = []
            all_points = set()
            for stmt in construction_stmts:
                # Format: "point1 point2 ... = constraint1, constraint2"
                eq_idx = stmt.find('=')
                if eq_idx >= 0:
                    lhs_str = stmt[:eq_idx].strip()
                    rhs_str = stmt[eq_idx+1:].strip()
                    
                    lhs_points = lhs_str.split()
                    all_points.update(lhs_points)
                    
                    # Parse constraints from rhs
                    constraints = []
                    # Split by , for multiple constraints
                    constraint_parts = [c.strip() for c in rhs_str.split(',')]
                    for cp in constraint_parts:
                        # Remove coordinate annotations like @4.96_-0.13
                        clean_cp = re.sub(r'@\S+', '', cp)
                        m = re.match(r'(\w+)\s+(.*)', clean_cp.strip())
                        if m:
                            pred = m.group(1)
                            args = m.group(2).strip().split()
                            constraints.append({'predicate': pred, 'args': args})
                            all_points.update(args)
                    
                    constructions.append({
                        'defined_points': lhs_points,
                        'raw_rhs': rhs_str,
                        'constraints': constraints
                    })
            
            if goal:
                all_points.update(goal['args'])
            
            problems.append({
                'name': name,
                'goal': goal,
                'constructions': constructions,
                'num_points': len(all_points),
                'num_construction_steps': len(constructions),
                'goal_predicate': goal['predicate'] if goal else None,
                'full_body': full_body
            })
        else:
            i += 1
    
    return problems


def parse_defs(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    
    definitions = {}
    lines = text.strip().split('\n')
    current_def = None
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # A definition header starts with a known predicate name followed by its signature
        m = re.match(r'^([a-z]\w+|on_\w+|foot|midpoint|mirror|reflect|angle_bisector|angle_mirror|circle|orthocenter|incenter|incenter2|excenter2|circumcenter|eqdistance|eqangle2|on_dia|cc_tangent|eqangle3|iso_triangle|r_triangle|parallelogram|nsquare|psquare|shift|s_angle|lc_tangent|tangent|on_circum|intersection_\w+|ieq_triangle|triangle12|3peq|trisect|trisegment|e5128|2l1c|pentagon|quadrangle|segment|triangle|free|isquare|square|rectangle|trapezoid|r_trapezoid|eq_quadrangle|eq_trapezoid|eqdia_quadrangle|centroid|ninepoints|excenter|on_opline|on_bline|on_line|on_pline|on_tline|on_circle|on_aline|on_aline2)\s+(.*)', stripped)
        if m:
            pred_name = m.group(1)
            sig = m.group(2).strip()
            current_def = pred_name
            definitions[current_def] = {
                'signature': sig,
                'conditions': [],
                'properties': [],
                'construction_method': None
            }
        elif current_def:
            if stripped.startswith('=') or 'ncoll' in stripped or 'diff' in stripped or 'npara' in stripped or 'nperp' in stripped or 'sameside' in stripped:
                definitions[current_def]['conditions'].append(stripped)
            elif ':' in stripped:
                definitions[current_def]['properties'].append(stripped)
            elif stripped and not stripped.startswith('='):
                definitions[current_def]['construction_method'] = stripped
    
    return definitions


def parse_rules(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    
    rules = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('=>')
        if len(parts) == 2:
            premises_str = parts[0].strip()
            conclusion_str = parts[1].strip()
            
            premises = []
            for p in premises_str.split(','):
                p = p.strip()
                m = re.match(r'(\w+)\s+(.*)', p)
                if m:
                    premises.append({
                        'predicate': m.group(1),
                        'args': m.group(2).strip().split()
                    })
                elif p:
                    premises.append({'predicate': p, 'args': []})
            
            conclusion_m = re.match(r'(\w+)\s+(.*)', conclusion_str)
            if conclusion_m:
                conclusion = {
                    'predicate': conclusion_m.group(1),
                    'args': conclusion_m.group(2).strip().split()
                }
            else:
                conclusion = {'predicate': conclusion_str, 'args': []}
            
            rules.append({
                'raw': line,
                'premises': premises,
                'conclusion': conclusion,
                'num_premises': len(premises)
            })
    
    return rules


if __name__ == '__main__':
    problems = parse_problems(os.path.join(WORKSPACE, 'data/imo_ag_30.txt'))
    definitions = parse_defs(os.path.join(WORKSPACE, 'data/defs.txt'))
    rules = parse_rules(os.path.join(WORKSPACE, 'data/rules.txt'))
    
    with open(os.path.join(OUTPUT_DIR, 'parsed_problems.json'), 'w') as f:
        json.dump(problems, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'parsed_definitions.json'), 'w') as f:
        json.dump(definitions, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'parsed_rules.json'), 'w') as f:
        json.dump(rules, f, indent=2)
    
    goal_types = {}
    for p in problems:
        gp = p['goal_predicate']
        if gp:
            goal_types[gp] = goal_types.get(gp, 0) + 1
    
    print(f"Parsed {len(problems)} problems")
    print(f"Parsed {len(definitions)} definitions")
    print(f"Parsed {len(rules)} inference rules")
    print(f"Goal type distribution: {goal_types}")
    print(f"Average num_points: {sum(p['num_points'] for p in problems)/len(problems):.1f}")
    print(f"Average num_construction_steps: {sum(p['num_construction_steps'] for p in problems)/len(problems):.1f}")
    
    # Print first 3 problems for verification
    for p in problems[:3]:
        print(f"\n--- {p['name']} ---")
        print(f"Goal: {p['goal']}")
        print(f"Points: {p['num_points']}, Steps: {p['num_construction_steps']}")