#!/usr/bin/env python3
"""
Step 2: Build a symbolic geometry proof engine.
Implements forward-chaining deduction using the 43 inference rules and
definition-derived premises from the IMO-AG-30 benchmark.

The engine extracts initial facts from problem constructions,
then repeatedly applies inference rules to derive new facts
until the goal is reached or a depth limit is hit.
"""

import json
import re
import os
import itertools
from collections import defaultdict

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_180131"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")

# ─── Fact representation ────────────────────────────────────────────
# A fact is a tuple: (predicate, args_tuple)
# e.g., ("cong", ("a", "b", "c", "d"))

def make_fact(predicate, args):
    """Create a canonical fact tuple."""
    return (predicate, tuple(args))

def fact_to_str(fact):
    """Convert fact to human-readable string."""
    pred, args = fact
    return f"{pred} {(' '.join(args))}"

# ─── Extract initial facts from problem constructions ────────────────
def extract_initial_facts(problem):
    """
    From each construction statement, extract the geometric facts
    that are immediately implied by the definition of each construction type.
    
    For example, "m = midpoint m a b" implies:
      coll m a b, cong m a m b
    
    And "h = orthocenter h a b c" implies:
      perp h a b c, perp h b c a
    """
    facts = set()
    point_defs = {}  # point -> how it was defined
    
    for constr in problem['constructions']:
        for constraint in constr['constraints']:
            pred = constraint['predicate']
            args = constraint['args']
            
            # Add the constraint itself as a fact
            facts.add(make_fact(pred, args))
            
            # Add derived facts based on construction definitions
            derived = get_derived_facts(pred, args, constr['defined_points'])
            facts.update(derived)
    
    # Also add basic facts from triangle/segment declarations
    for constr in problem['constructions']:
        defined = constr['defined_points']
        for constraint in constr['constraints']:
            pred = constraint['predicate']
            args = constraint['args']
            
            if pred == 'triangle':
                # triangle a b c => ncoll a b c (implicit)
                if len(args) >= 3:
                    facts.add(make_fact('ncoll', args[:3]))
            
            elif pred == 'segment':
                # segment a b => diff a b
                if len(args) >= 2:
                    facts.add(make_fact('diff', args[:2]))
            
            elif pred == 'iso_triangle':
                # iso_triangle s c p => eqangle c s c p p s p c, cong s c s p
                if len(args) >= 3:
                    s, c, p = args[0], args[1], args[2]
                    facts.add(make_fact('cong', [s, c, s, p]))
                    facts.add(make_fact('eqangle', [c, s, c, p, p, s, p, c]))
            
            elif pred == 'r_triangle':
                # r_triangle c a b => perp c a c b
                if len(args) >= 3:
                    c, a, b = args[0], args[1], args[2]
                    facts.add(make_fact('perp', [c, a, c, b]))
    
    return facts

def get_derived_facts(pred, args, defined_points):
    """
    Given a construction predicate and its arguments,
    return the set of facts that are directly implied.
    This encodes the geometric meaning of each construction type.
    """
    derived = set()
    
    # Midpoint: m = midpoint m a b => coll m a b, cong m a m b
    if pred == 'midpoint' and len(args) >= 3:
        m, a, b = args[0], args[1], args[2]
        derived.add(make_fact('coll', [m, a, b]))
        derived.add(make_fact('cong', [m, a, m, b]))
    
    # Foot: h1 = foot h1 a b c => perp h1 a b c, coll h1 b c
    elif pred == 'foot' and len(args) >= 4:
        h1, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('perp', [h1, a, b, c]))
        derived.add(make_fact('coll', [h1, b, c]))
    
    # Orthocenter: h = orthocenter h a b c => perp h a b c, perp h b c a, perp h c a b
    elif pred == 'orthocenter' and len(args) >= 4:
        h, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('perp', [h, a, b, c]))
        derived.add(make_fact('perp', [h, b, c, a]))
        derived.add(make_fact('perp', [h, c, a, b]))
    
    # Incenter: i = incenter i a b c => eqangle a b a i a i a c, etc.
    elif pred == 'incenter' and len(args) >= 4:
        i, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('eqangle', [a, b, a, i, a, i, a, c]))
        derived.add(make_fact('eqangle', [b, c, b, i, b, i, b, a]))
        derived.add(make_fact('eqangle', [c, a, c, i, c, i, c, b]))
    
    # Incenter2: t1 t2 t3 i = incenter2 t1 t2 t3 i a b c
    elif pred == 'incenter2' and len(args) >= 8:
        t1, t2, t3, i, a, b, c = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        # Incenter properties
        derived.add(make_fact('eqangle', [a, b, a, i, a, i, a, c]))
        derived.add(make_fact('eqangle', [b, c, b, i, b, i, b, a]))
        # Touch points on sides
        derived.add(make_fact('coll', [t1, b, c]))
        derived.add(make_fact('perp', [i, t1, b, c]))
        derived.add(make_fact('coll', [t2, c, a]))
        derived.add(make_fact('perp', [i, t2, c, a]))
        derived.add(make_fact('coll', [t3, a, b]))
        derived.add(make_fact('perp', [i, t3, a, b]))
        derived.add(make_fact('cong', [i, t1, i, t2]))
        derived.add(make_fact('cong', [i, t2, i, t3]))
    
    # Excenter2: similar to incenter2 but external
    elif pred == 'excenter2' and len(args) >= 8:
        t1, t2, t3, i, a, b, c = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        derived.add(make_fact('coll', [t1, b, c]))
        derived.add(make_fact('perp', [i, t1, b, c]))
        derived.add(make_fact('coll', [t2, c, a]))
        derived.add(make_fact('perp', [i, t2, c, a]))
        derived.add(make_fact('coll', [t3, a, b]))
        derived.add(make_fact('perp', [i, t3, a, b]))
        derived.add(make_fact('cong', [i, t1, i, t2]))
        derived.add(make_fact('cong', [i, t2, i, t3]))
    
    # Circle (circumcircle): o = circle o a b c => cong o a o b, cong o b o c, cong o a o c
    elif pred == 'circle' and len(args) >= 4:
        o, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('cong', [o, a, o, b]))
        derived.add(make_fact('cong', [o, b, o, c]))
        derived.add(make_fact('cong', [o, a, o, c]))
        derived.add(make_fact('ncoll', [a, b, c]))
    
    # on_circle: x = on_circle x o a => cong o x o a
    elif pred == 'on_circle' and len(args) >= 3:
        x, o, a = args[0], args[1], args[2]
        derived.add(make_fact('cong', [o, x, o, a]))
    
    # on_line: x = on_line x a b => coll x a b
    elif pred == 'on_line' and len(args) >= 3:
        x, a, b = args[0], args[1], args[2]
        derived.add(make_fact('coll', [x, a, b]))
    
    # on_bline: x = on_bline x a b => cong x a x b
    elif pred == 'on_bline' and len(args) >= 3:
        x, a, b = args[0], args[1], args[2]
        derived.add(make_fact('cong', [x, a, x, b]))
    
    # on_pline: x = on_pline x a b c => para x a b c
    elif pred == 'on_pline' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('para', [x, a, b, c]))
    
    # on_tline: x = on_tline x a b c => perp x a b c
    elif pred == 'on_tline' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('perp', [x, a, b, c]))
    
    # on_aline: x = on_aline x a b c d e => eqangle a x a b d c d e
    elif pred == 'on_aline' and len(args) >= 6:
        x, a, b, c, d, e = args[0], args[1], args[2], args[3], args[4], args[5]
        derived.add(make_fact('eqangle', [a, x, a, b, d, c, d, e]))
    
    # on_dia: x = on_dia x a b => perp x a x b
    elif pred == 'on_dia' and len(args) >= 3:
        x, a, b = args[0], args[1], args[2]
        derived.add(make_fact('perp', [x, a, x, b]))
    
    # Reflect: x = reflect x a b c => cong b a b x, cong c a c x, perp b c a x
    elif pred == 'reflect' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('cong', [b, a, b, x]))
        derived.add(make_fact('cong', [c, a, c, x]))
        derived.add(make_fact('perp', [b, c, a, x]))
    
    # Mirror: x = mirror x a b => coll x a b, cong b a b x
    elif pred == 'mirror' and len(args) >= 3:
        x, a, b = args[0], args[1], args[2]
        derived.add(make_fact('coll', [x, a, b]))
        derived.add(make_fact('cong', [b, a, b, x]))
    
    # Angle bisector: x = angle_bisector x a b c => eqangle b a b x b x b c
    elif pred == 'angle_bisector' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('eqangle', [b, a, b, x, b, x, b, c]))
    
    # Angle mirror: x = angle_mirror x a b c => eqangle b a b c b c b x
    elif pred == 'angle_mirror' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('eqangle', [b, a, b, c, b, c, b, x]))
    
    # Circumcenter: x = circumcenter x a b c => cong x a x b, cong x b x c
    elif pred == 'circumcenter' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('cong', [x, a, x, b]))
        derived.add(make_fact('cong', [x, b, x, c]))
    
    # Eqdistance: x = eqdistance x a b c => cong x a b c
    elif pred == 'eqdistance' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('cong', [x, a, b, c]))
    
    # Eqangle2: x = eqangle2 x a b c => eqangle a b a x c x c b
    elif pred == 'eqangle2' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('eqangle', [a, b, a, x, c, x, c, b]))
    
    # Parallelogram: x = parallelogram e a m x => para a b c x, para a x b c, etc.
    elif pred == 'parallelogram' and len(args) >= 4:
        # parallelogram a b c x => para a b c x, para a x b c
        a, b, c, x = args[0], args[1], args[2], args[3]
        derived.add(make_fact('para', [a, b, c, x]))
        derived.add(make_fact('para', [a, x, b, c]))
    
    # Intersection_lc: x = intersection_lc x a o b => coll x a b, cong o b o x
    elif pred == 'intersection_lc' and len(args) >= 4:
        x, a, o, b = args[0], args[1], args[2], args[3]
        derived.add(make_fact('coll', [x, a, b]))
        derived.add(make_fact('cong', [o, b, o, x]))
    
    # Intersection_ll: x = intersection_ll x a b c d => coll x a b, coll x c d
    elif pred == 'intersection_ll' and len(args) >= 5:
        x, a, b, c, d = args[0], args[1], args[2], args[3], args[4]
        derived.add(make_fact('coll', [x, a, b]))
        derived.add(make_fact('coll', [x, c, d]))
    
    # Intersection_cc: x = intersection_cc x o w a => cong o a o x, cong w a w x
    elif pred == 'intersection_cc' and len(args) >= 4:
        x, o, w, a = args[0], args[1], args[2], args[3]
        derived.add(make_fact('cong', [o, a, o, x]))
        derived.add(make_fact('cong', [w, a, w, x]))
    
    # cc_tangent: complex tangent construction
    elif pred == 'cc_tangent' and len(args) >= 4:
        q, t, p, s = args[0], args[1], args[2], args[3]
        # These are tangent points - we'd need more context
    
    # lc_tangent: x = lc_tangent x a o => perp a x a o
    elif pred == 'lc_tangent' and len(args) >= 3:
        x, a, o = args[0], args[1], args[2]
        derived.add(make_fact('perp', [a, x, a, o]))
    
    # Nsquare: x = nsquare x a b => cong x a a b, perp x a a b
    elif pred == 'nsquare' and len(args) >= 3:
        x, a, b = args[0], args[1], args[2]
        derived.add(make_fact('cong', [x, a, a, b]))
        derived.add(make_fact('perp', [x, a, a, b]))
    
    # Psquare: similar
    elif pred == 'psquare' and len(args) >= 3:
        x, a, b = args[0], args[1], args[2]
        derived.add(make_fact('cong', [x, a, a, b]))
        derived.add(make_fact('perp', [x, a, a, b]))
    
    # Shift: x = shift x b c d => cong x b c d, cong x c b d
    elif pred == 'shift' and len(args) >= 4:
        x, b, c, d = args[0], args[1], args[2], args[3]
        derived.add(make_fact('cong', [x, b, c, d]))
        derived.add(make_fact('cong', [x, c, b, d]))
    
    # on_opline: x = on_opline x a b => coll x a b
    elif pred == 'on_opline' and len(args) >= 3:
        x, a, b = args[0], args[1], args[2]
        derived.add(make_fact('coll', [x, a, b]))
    
    # eqangle3: x = eqangle3 x a b d e f => eqangle x a x b d e d f
    elif pred == 'eqangle3' and len(args) >= 6:
        x, a, b, d, e, f = args[0], args[1], args[2], args[3], args[4], args[5]
        derived.add(make_fact('eqangle', [x, a, x, b, d, e, d, f]))
    
    # tangent: tangent x y a o b
    elif pred == 'tangent' and len(args) >= 3:
        # x y : o a b => cong o x o b, perp a x o x
        pass  # Complex, handled specially
    
    # on_circum: x = on_circum x a b c => cyclic a b c x
    elif pred == 'on_circum' and len(args) >= 4:
        x, a, b, c = args[0], args[1], args[2], args[3]
        derived.add(make_fact('cyclic', [a, b, c, x]))
    
    return derived


# ─── Rule application ────────────────────────────────────────────────
def apply_rules(facts, rules, max_iterations=50):
    """
    Forward-chaining: repeatedly apply inference rules to derive new facts.
    Returns (all_facts, proof_trace, goal_reached).
    """
    all_facts = set(facts)
    new_facts = set(facts)
    proof_trace = []  # List of (rule_index, matched_premises, derived_conclusion)
    iteration = 0
    
    while new_facts and iteration < max_iterations:
        iteration += 1
        derived_this_round = set()
        
        for rule_idx, rule in enumerate(rules):
            premises = rule['premises']
            conclusion = rule['conclusion']
            
            if not premises:
                continue
            
            # Try to match all premises against known facts
            # For each premise, find all matching facts
            matches_per_premise = []
            for prem in premises:
                pred = prem['predicate']
                prem_args = prem['args']
                
                # Find facts with matching predicate
                matching = []
                for fact in all_facts:
                    if fact[0] == pred:
                        # Check if argument pattern matches
                        # The rule uses variable names; we need substitution
                        fact_args = fact[1]
                        if len(fact_args) == len(prem_args):
                            matching.append(fact)
                matches_per_premise.append(matching)
            
            if not all(matches_per_premise):
                continue
            
            # Try all combinations of matching facts
            for combo in itertools.product(*matches_per_premise):
                # Build substitution map from rule variables to actual points
                subst = {}
                consistent = True
                for prem, fact in zip(premises, combo):
                    for var, val in zip(prem['args'], fact[1]):
                        if var in subst:
                            if subst[var] != val:
                                consistent = False
                                break
                        else:
                            subst[var] = val
                    if not consistent:
                        break
                
                if not consistent:
                    continue
                
                # Apply substitution to conclusion
                conc_pred = conclusion['predicate']
                conc_args = []
                for arg in conclusion['args']:
                    if arg in subst:
                        conc_args.append(subst[arg])
                    else:
                        conc_args.append(arg)  # Free variable stays
                
                new_fact = make_fact(conc_pred, conc_args)
                
                if new_fact not in all_facts:
                    derived_this_round.add(new_fact)
                    proof_trace.append({
                        'rule_idx': rule_idx,
                        'rule_raw': rule['raw'],
                        'matched_facts': [fact_to_str(f) for f in combo],
                        'derived_fact': fact_to_str(new_fact),
                        'iteration': iteration
                    })
        
        all_facts.update(derived_this_round)
        new_facts = derived_this_round
        
        if not derived_this_round:
            break
    
    return all_facts, proof_trace, iteration


# ─── Check if goal is reached ────────────────────────────────────────
def check_goal(facts, goal):
    """Check if the goal fact is in the derived facts set."""
    if goal is None:
        return False
    
    goal_fact = make_fact(goal['predicate'], goal['args'])
    
    # Direct match
    if goal_fact in facts:
        return True
    
    # For cong goals, also check symmetric versions
    if goal['predicate'] == 'cong':
        args = goal['args']
        # cong a b c d is symmetric: cong a b c d = cong c d a b = cong b a d c = cong d c b a
        variants = [
            make_fact('cong', [args[2], args[3], args[0], args[1]]),
            make_fact('cong', [args[1], args[0], args[3], args[2]]),
            make_fact('cong', [args[3], args[2], args[1], args[0]]),
        ]
        for v in variants:
            if v in facts:
                return True
    
    # For coll goals, check permutations
    if goal['predicate'] == 'coll':
        args = list(goal['args'])
        for perm in itertools.permutations(args):
            if make_fact('coll', list(perm)) in facts:
                return True
    
    # For cyclic goals, check cyclic permutations
    if goal['predicate'] == 'cyclic':
        args = list(goal['args'])
        for i in range(len(args)):
            rotated = args[i:] + args[:i]
            if make_fact('cyclic', rotated) in facts:
                return True
    
    # For perp goals, check symmetric
    if goal['predicate'] == 'perp':
        args = goal['args']
        # perp A B C D = perp C D A B
        if make_fact('perp', [args[2], args[3], args[0], args[1]]) in facts:
            return True
    
    # For para goals, check symmetric
    if goal['predicate'] == 'para':
        args = goal['args']
        if make_fact('para', [args[2], args[3], args[0], args[1]]) in facts:
            return True
    
    return False


# ─── Main: Run proof engine on all problems ──────────────────────────
if __name__ == '__main__':
    # Load parsed data
    with open(os.path.join(OUTPUT_DIR, 'parsed_problems.json'), 'r') as f:
        problems = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'parsed_rules.json'), 'r') as f:
        rules = json.load(f)
    
    results = []
    total_solved = 0
    
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem['name']}")
        print(f"Goal: {problem['goal']['predicate']} {(' '.join(problem['goal']['args']))}")
        
        # Extract initial facts
        initial_facts = extract_initial_facts(problem)
        print(f"Initial facts: {len(initial_facts)}")
        
        # Run forward chaining
        all_facts, trace, iterations = apply_rules(initial_facts, rules, max_iterations=100)
        print(f"Derived facts after {iterations} iterations: {len(all_facts)}")
        print(f"Proof steps: {len(trace)}")
        
        # Check goal
        solved = check_goal(all_facts, problem['goal'])
        print(f"Goal reached: {solved}")
        
        if solved:
            total_solved += 1
        
        results.append({
            'name': problem['name'],
            'goal_predicate': problem['goal_predicate'],
            'num_points': problem['num_points'],
            'num_construction_steps': problem['num_construction_steps'],
            'initial_facts_count': len(initial_facts),
            'total_facts_count': len(all_facts),
            'proof_steps': len(trace),
            'iterations': iterations,
            'solved': solved
        })
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'proof_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"Total problems: {len(problems)}")
    print(f"Solved by pure forward chaining: {total_solved}")
    print(f"Solve rate: {total_solved/len(problems)*100:.1f}%")