"""
IMO Geometry Problem Analyzer and Verifier
Parses formal geometry statements, classifies problems, and generates analysis.
"""
import re
import json
from collections import Counter, defaultdict

def parse_problem(problem_str):
    """Parse a formal geometry problem statement."""
    lines = problem_str.strip().split('\n')
    name = lines[0]
    
    # Split into premise and goal
    full_stmt = ' '.join(lines[1:])
    if '?' in full_stmt:
        premise_part, goal_part = full_stmt.split('?', 1)
        goal = goal_part.strip()
    else:
        premise_part = full_stmt
        goal = ""
    
    # Extract construction steps
    steps = [s.strip() for s in premise_part.split(';') if s.strip()]
    
    # Extract all predicates used
    predicates = set()
    for pred in re.findall(r'(\w+)\s*\(', premise_part + ' ' + goal):
        predicates.add(pred)
    # Also catch predicates without parens
    for pred in re.findall(r'\b(on_\w+|eq\w+|cong|coll|para|perp|cyclic|midp|circle|triangle|segment|foot|orthocenter|incenter|incenter2|excenter|excenter2|reflect|mirror|angle_bisector|angle_mirror|eqdistance|eqangle\d*|eqratio\d*|on_aline\d*|on_bline|on_circle|on_line|on_pline|on_tline|on_dia|on_circum|on_opline|cc_tangent\d*|iso_triangle|r_triangle|ieq_triangle|eq_triangle|parallelogram|square|isquare|trapezoid|eq_trapezoid|r_trapezoid|quadrangle|eq_quadrangle|eqdia_quadrangle|pentagon|nsquare|psquare|risos|shift|s_angle|free|tangent|lc_tangent|intersection_\w+|2l1c|e5128|3peq|trisect|trisegment)\b', premise_part + ' ' + goal):
        predicates.add(pred)
    
    return {
        'name': name,
        'premise': premise_part.strip(),
        'goal': goal,
        'steps': steps,
        'predicates': predicates
    }

def classify_problem(problem):
    """Classify a geometry problem by its goal type and construction complexity."""
    goal = problem['goal']
    
    # Goal classification
    if 'cong' in goal:
        goal_type = 'congruence'
    elif 'eqangle' in goal:
        goal_type = 'equal_angle'
    elif 'para' in goal:
        goal_type = 'parallel'
    elif 'perp' in goal:
        goal_type = 'perpendicular'
    elif 'cyclic' in goal:
        goal_type = 'cyclic'
    elif 'coll' in goal:
        goal_type = 'collinear'
    elif 'eqratio' in goal:
        goal_type = 'equal_ratio'
    else:
        goal_type = 'other'
    
    # Complexity metrics
    num_steps = len(problem['steps'])
    num_predicates = len(problem['predicates'])
    
    # Construction type analysis
    construction_types = set()
    for pred in problem['predicates']:
        if pred in ['on_circle', 'circle']:
            construction_types.add('circle')
        elif pred in ['on_line', 'on_pline', 'on_tline', 'on_bline', 'on_aline', 'on_dia', 'on_opline']:
            construction_types.add('line')
        elif pred in ['foot', 'midpoint', 'reflect', 'mirror']:
            construction_types.add('transformation')
        elif pred in ['orthocenter', 'incenter', 'incenter2', 'excenter', 'excenter2', 'centroid']:
            construction_types.add('center')
        elif pred in ['triangle', 'segment', 'iso_triangle', 'r_triangle', 'ieq_triangle', 'eq_triangle']:
            construction_types.add('primitive')
        elif pred in ['cyclic', 'cong', 'coll', 'para', 'perp', 'eqangle', 'eqratio']:
            construction_types.add('relation')
    
    return {
        'goal_type': goal_type,
        'num_steps': num_steps,
        'num_predicates': num_predicates,
        'construction_types': list(construction_types),
        'predicates': list(problem['predicates'])
    }

def analyze_predicate_usage(problems):
    """Analyze predicate usage across all problems."""
    predicate_counter = Counter()
    predicate_by_goal = defaultdict(lambda: Counter())
    
    for p in problems:
        cls = classify_problem(p)
        for pred in p['predicates']:
            predicate_counter[pred] += 1
            predicate_by_goal[cls['goal_type']][pred] += 1
    
    return predicate_counter, predicate_by_goal

def compute_difficulty_score(problem):
    """Compute a heuristic difficulty score for a problem."""
    cls = classify_problem(problem)
    score = 0
    score += cls['num_steps'] * 2
    score += cls['num_predicates']
    
    # Bonus for advanced constructions
    advanced = {'orthocenter', 'incenter', 'incenter2', 'excenter', 'excenter2', 
                'reflect', 'cc_tangent', 'cc_tangent0', 'on_aline', 'on_aline2',
                'eqangle3', 'angle_mirror', 'trisect'}
    for pred in cls['predicates']:
        if pred in advanced:
            score += 3
    
    # Bonus for harder goal types
    hard_goals = {'cyclic', 'collinear', 'equal_ratio'}
    if cls['goal_type'] in hard_goals:
        score += 5
    
    return score

def generate_proof_sketch(problem):
    """Generate a proof sketch based on problem structure."""
    cls = classify_problem(problem)
    goal = problem['goal']
    
    strategies = []
    
    if cls['goal_type'] == 'congruence':
        strategies.append("Strategy: Show segment congruence via triangle congruence or circle properties.")
        if 'circle' in cls['construction_types']:
            strategies.append("  - Use circle properties: radii of same circle are congruent.")
        if 'midpoint' in str(problem['predicates']):
            strategies.append("  - Use midpoint properties: midpoint divides segment equally.")
        if 'foot' in str(problem['predicates']):
            strategies.append("  - Consider right triangles formed by perpendicular feet.")
    
    elif cls['goal_type'] == 'cyclic':
        strategies.append("Strategy: Show four points are concyclic.")
        strategies.append("  - Approach 1: Show opposite angles sum to 180° (eqangle).")
        strategies.append("  - Approach 2: Show equal angles subtended by same chord.")
        strategies.append("  - Approach 3: Use power of a point.")
    
    elif cls['goal_type'] == 'collinear':
        strategies.append("Strategy: Show three points lie on same line.")
        strategies.append("  - Approach 1: Use Menelaus' theorem.")
        strategies.append("  - Approach 2: Show angle between segments is 0 or 180°.")
        strategies.append("  - Approach 3: Use radical axis theorem.")
    
    elif cls['goal_type'] == 'parallel':
        strategies.append("Strategy: Show lines are parallel.")
        strategies.append("  - Approach 1: Show corresponding angles are equal.")
        strategies.append("  - Approach 2: Use midpoint theorem.")
        strategies.append("  - Approach 3: Show both lines perpendicular to same line.")
    
    elif cls['goal_type'] == 'perpendicular':
        strategies.append("Strategy: Show lines are perpendicular.")
        strategies.append("  - Approach 1: Use circle with diameter.")
        strategies.append("  - Approach 2: Use Pythagorean theorem on distances.")
        strategies.append("  - Approach 3: Use orthocenter properties.")
    
    elif cls['goal_type'] == 'equal_angle':
        strategies.append("Strategy: Show angles are equal.")
        strategies.append("  - Approach 1: Use inscribed angle theorem (cyclic quadrilateral).")
        strategies.append("  - Approach 2: Use similar triangles.")
        strategies.append("  - Approach 3: Use angle bisector properties.")
    
    elif cls['goal_type'] == 'equal_ratio':
        strategies.append("Strategy: Show ratio of segments/lengths are equal.")
        strategies.append("  - Approach 1: Use similar triangles.")
        strategies.append("  - Approach 2: Use Menelaus/Ceva theorem.")
        strategies.append("  - Approach 3: Use intercept theorem.")
    
    return strategies

def analyze_rules():
    """Analyze the inference rules."""
    rules_text = open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/data/rules.txt').read()
    rules = [r.strip() for r in rules_text.strip().split('\n') if r.strip()]
    
    rule_categories = defaultdict(list)
    for rule in rules:
        if 'para' in rule and '=>' in rule:
            rule_categories['parallel'].append(rule)
        elif 'perp' in rule and '=>' in rule:
            rule_categories['perpendicular'].append(rule)
        elif 'cong' in rule and '=>' in rule:
            rule_categories['congruence'].append(rule)
        elif 'cyclic' in rule and '=>' in rule:
            rule_categories['cyclic'].append(rule)
        elif 'eqangle' in rule and '=>' in rule:
            rule_categories['angle'].append(rule)
        elif 'eqratio' in rule and '=>' in rule:
            rule_categories['ratio'].append(rule)
        elif 'midp' in rule and '=>' in rule:
            rule_categories['midpoint'].append(rule)
        elif 'simtri' in rule or 'contri' in rule:
            rule_categories['triangle_similarity'].append(rule)
        else:
            rule_categories['other'].append(rule)
    
    return rules, rule_categories

if __name__ == '__main__':
    # Load problems
    with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/data/imo_ag_30.txt') as f:
        content = f.read()
    
    # Parse problems
    problem_blocks = re.split(r'\n(?=translated_imo_)', content.strip())
    problems = []
    for block in problem_blocks:
        if block.strip():
            problems.append(parse_problem(block))
    
    print(f"Total problems parsed: {len(problems)}")
    
    # Classify all problems
    classifications = []
    for p in problems:
        cls = classify_problem(p)
        cls['name'] = p['name']
        cls['difficulty'] = compute_difficulty_score(p)
        cls['proof_strategies'] = generate_proof_sketch(p)
        classifications.append(cls)
    
    # Analyze predicates
    pred_counter, pred_by_goal = analyze_predicate_usage(problems)
    
    # Analyze rules
    rules, rule_cats = analyze_rules()
    
    # Goal type distribution
    goal_dist = Counter(c['goal_type'] for c in classifications)
    
    # Save results
    results = {
        'total_problems': len(problems),
        'classifications': classifications,
        'goal_distribution': dict(goal_dist),
        'predicate_frequency': dict(pred_counter.most_common(30)),
        'predicate_by_goal': {k: dict(v) for k, v in pred_by_goal.items()},
        'rule_categories': {k: len(v) for k, v in rule_cats.items()},
        'total_rules': len(rules),
        'problems': [{'name': p['name'], 'goal': p['goal'], 'num_steps': len(p['steps']), 
                       'predicates': list(p['predicates'])} for p in problems]
    }
    
    with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/outputs/analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Analysis complete. Results saved.")
    print(f"Goal distribution: {dict(goal_dist)}")
    print(f"Top predicates: {pred_counter.most_common(10)}")
