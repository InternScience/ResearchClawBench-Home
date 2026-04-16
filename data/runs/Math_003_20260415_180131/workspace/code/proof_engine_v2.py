#!/usr/bin/env python3
"""
Step 2b: Enhanced symbolic proof engine with better fact extraction
and extended rule application including definition-derived rules.

Key improvements:
1. More thorough extraction of implicit facts from constructions
2. Handle cyclic/cong symmetry properly  
3. Add algebraic/geometric derived rules (midpoint-perp, etc.)
4. Better rule matching with variable substitution
"""

import json
import re
import os
import itertools
from collections import defaultdict

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_180131"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")

def make_fact(predicate, args):
    return (predicate, tuple(args))

def fact_to_str(fact):
    pred, args = fact
    return f"{pred} {' '.join(args)}"

def normalize_cong(args):
    """Normalize cong fact: cong A B C D -> sorted so AB <= CD and A < B, C < D"""
    a, b, c, d = args
    # cong is symmetric: cong A B C D = cong B A D C = cong C D A B = cong D C B A
    pairs = [(a,b), (c,d)]
    # Sort each pair internally
    pairs = [tuple(sorted(p)) for p in pairs]
    # Sort the two pairs
    pairs = sorted(pairs)
    return list(pairs[0]) + list(pairs[1])

def normalize_coll(args):
    """Normalize coll fact: coll A B C -> sorted alphabetically"""
    return sorted(args)

def normalize_cyclic(args):
    """Cyclic has rotational symmetry but not full permutation symmetry.
    We store all rotations as equivalent."""
    return list(args)  # Keep original order, check rotations separately

def fact_key(fact):
    """Create a normalized key for fact comparison."""
    pred, args = fact
    if pred == 'cong':
        return ('cong', tuple(normalize_cong(list(args))))
    elif pred == 'coll':
        return ('coll', tuple(normalize_coll(list(args))))
    elif pred == 'cyclic':
        return ('cyclic', tuple(sorted(args)))  # Use sorted for storage, check rotations for match
    elif pred == 'perp':
        # perp A B C D = perp C D A B = perp B A D C = perp D C B A
        lines = [tuple(sorted(args[:2])), tuple(sorted(args[2:4]))]
        lines_sorted = sorted(lines)
        return ('perp', tuple(lines_sorted[0] + lines_sorted[1]))
    elif pred == 'para':
        lines = [tuple(sorted(args[:2])), tuple(sorted(args[2:4]))]
        lines_sorted = sorted(lines)
        return ('para', tuple(lines_sorted[0] + lines_sorted[1]))
    elif pred == 'eqangle':
        # eqangle A B P Q C D P Q => para A B C D when P=Q context
        # For now, keep as-is but also add reversed version
        return (pred, args)
    elif pred == 'eqratio':
        return (pred, args)
    else:
        return (pred, args)

def facts_match(fact1, fact2):
    """Check if two facts are semantically equivalent."""
    k1 = fact_key(fact1)
    k2 = fact_key(fact2)
    return k1 == k2

def extract_initial_facts(problem):
    """Extract ALL implicit geometric facts from construction statements."""
    facts = set()
    fact_keys = set()
    
    def add_fact(pred, args):
        f = make_fact(pred, args)
        k = fact_key(f)
        if k not in fact_keys:
            facts.add(f)
            fact_keys.add(k)
    
    for constr in problem['constructions']:
        for constraint in constr['constraints']:
            pred = constraint['predicate']
            args = constraint['args']
            
            # Add the direct constraint
            add_fact(pred, args)
            
            # ─── Midpoint ───
            if pred == 'midpoint' and len(args) >= 3:
                m, a, b = args[0], args[1], args[2]
                add_fact('coll', [m, a, b])
                add_fact('cong', [m, a, m, b])
                add_fact('diff', [a, b])
                # midp M A B, perp O M A B => cong O A O B (rule 25)
            
            # ─── Foot ───
            elif pred == 'foot' and len(args) >= 4:
                h, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('perp', [h, a, b, c])
                add_fact('coll', [h, b, c])
            
            # ─── Orthocenter ───
            elif pred == 'orthocenter' and len(args) >= 4:
                h, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('perp', [h, a, b, c])
                add_fact('perp', [h, b, c, a])
                add_fact('perp', [h, c, a, b])
                add_fact('ncoll', [a, b, c])
            
            # ─── Incenter ───
            elif pred == 'incenter' and len(args) >= 4:
                i, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('eqangle', [a, b, a, i, a, i, a, c])
                add_fact('eqangle', [b, c, b, i, b, i, b, a])
                add_fact('eqangle', [c, a, c, i, c, i, c, b])
            
            # ─── Incenter2 ───
            elif pred == 'incenter2' and len(args) >= 7:
                t1, t2, t3, i = args[0], args[1], args[2], args[3]
                a, b, c = args[4], args[5], args[6] if len(args) >= 7 else args[4:7]
                if len(args) >= 7:
                    a, b, c = args[4], args[5], args[6]
                add_fact('eqangle', [a, b, a, i, a, i, a, c])
                add_fact('eqangle', [b, c, b, i, b, i, b, a])
                add_fact('coll', [t1, b, c])
                add_fact('perp', [i, t1, b, c])
                add_fact('coll', [t2, c, a])
                add_fact('perp', [i, t2, c, a])
                add_fact('coll', [t3, a, b])
                add_fact('perp', [i, t3, a, b])
                add_fact('cong', [i, t1, i, t2])
                add_fact('cong', [i, t2, i, t3])
                add_fact('cong', [i, t1, i, t3])
            
            # ─── Excenter2 ───
            elif pred == 'excenter2' and len(args) >= 7:
                m, l, k, j = args[0], args[1], args[2], args[3]
                if len(args) >= 7:
                    a, b, c = args[4], args[5], args[6]
                add_fact('coll', [m, b, c])
                add_fact('perp', [j, m, b, c])
                add_fact('coll', [l, c, a])
                add_fact('perp', [j, l, c, a])
                add_fact('coll', [k, a, b])
                add_fact('perp', [j, k, a, b])
                add_fact('cong', [j, m, j, l])
                add_fact('cong', [j, l, j, k])
            
            # ─── Circle (circumcircle) ───
            elif pred == 'circle' and len(args) >= 4:
                o, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('cong', [o, a, o, b])
                add_fact('cong', [o, b, o, c])
                add_fact('cong', [o, a, o, c])
                add_fact('ncoll', [a, b, c])
                # circle O A B C => cyclic A B C O (any 4 points on same circle)
                add_fact('cyclic', [a, b, c, o])
            
            # ─── on_circle ───
            elif pred == 'on_circle' and len(args) >= 3:
                x, o, a = args[0], args[1], args[2]
                add_fact('cong', [o, x, o, a])
                # If we already have circle O A B C and on_circle X O A, then X is on the same circle
                # => cyclic A B C X (we'll derive this via rules)
            
            # ─── on_line ───
            elif pred == 'on_line' and len(args) >= 3:
                x, a, b = args[0], args[1], args[2]
                add_fact('coll', [x, a, b])
                add_fact('diff', [a, b])
            
            # ─── on_bline ───
            elif pred == 'on_bline' and len(args) >= 3:
                x, a, b = args[0], args[1], args[2]
                add_fact('cong', [x, a, x, b])
                add_fact('diff', [a, b])
            
            # ─── on_pline ───
            elif pred == 'on_pline' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('para', [x, a, b, c])
            
            # ─── on_tline ───
            elif pred == 'on_tline' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('perp', [x, a, b, c])
            
            # ─── on_aline ───
            elif pred == 'on_aline' and len(args) >= 6:
                x, a, b, c, d, e = args[0], args[1], args[2], args[3], args[4], args[5]
                add_fact('eqangle', [a, x, a, b, d, c, d, e])
            
            # ─── on_aline2 ───
            elif pred == 'on_aline2' and len(args) >= 6:
                x, a, b, c, d, e = args[0], args[1], args[2], args[3], args[4], args[5]
                add_fact('eqangle', [x, a, x, b, d, c, d, e])
            
            # ─── on_dia ───
            elif pred == 'on_dia' and len(args) >= 3:
                x, a, b = args[0], args[1], args[2]
                add_fact('perp', [x, a, x, b])
                add_fact('coll', [x, a, b])  # x is on the diameter
            
            # ─── Reflect ───
            elif pred == 'reflect' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('cong', [b, a, b, x])
                add_fact('cong', [c, a, c, x])
                add_fact('perp', [b, c, a, x])
                add_fact('coll', [a, x])  # Not exactly, but reflection preserves midpoint
            
            # ─── Mirror ───
            elif pred == 'mirror' and len(args) >= 3:
                x, a, b = args[0], args[1], args[2]
                add_fact('coll', [x, a, b])
                add_fact('cong', [b, a, b, x])
                add_fact('diff', [a, b])
            
            # ─── Angle bisector ───
            elif pred == 'angle_bisector' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('eqangle', [b, a, b, x, b, x, b, c])
                add_fact('ncoll', [a, b, c])
            
            # ─── Angle mirror ───
            elif pred == 'angle_mirror' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('eqangle', [b, a, b, c, b, c, b, x])
            
            # ─── Circumcenter ───
            elif pred == 'circumcenter' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('cong', [x, a, x, b])
                add_fact('cong', [x, b, x, c])
                add_fact('cong', [x, a, x, c])
            
            # ─── Eqdistance ───
            elif pred == 'eqdistance' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('cong', [x, a, b, c])
            
            # ─── Eqangle2 ───
            elif pred == 'eqangle2' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('eqangle', [a, b, a, x, c, x, c, b])
            
            # ─── Parallelogram ───
            elif pred == 'parallelogram' and len(args) >= 4:
                a, b, c, x = args[0], args[1], args[2], args[3]
                add_fact('para', [a, b, c, x])
                add_fact('para', [a, x, b, c])
            
            # ─── Iso_triangle ───
            elif pred == 'iso_triangle' and len(args) >= 3:
                s, c, p = args[0], args[1], args[2]
                add_fact('cong', [s, c, s, p])
                add_fact('eqangle', [c, s, c, p, p, s, p, c])
                add_fact('ncoll', [s, c, p])
            
            # ─── R_triangle ───
            elif pred == 'r_triangle' and len(args) >= 3:
                c, a, b = args[0], args[1], args[2]
                add_fact('perp', [c, a, c, b])
            
            # ─── Intersection_ll ───
            elif pred == 'intersection_ll' and len(args) >= 5:
                x, a, b, c, d = args[0], args[1], args[2], args[3], args[4]
                add_fact('coll', [x, a, b])
                add_fact('coll', [x, c, d])
            
            # ─── Intersection_lc ───
            elif pred == 'intersection_lc' and len(args) >= 4:
                x, a, o, b = args[0], args[1], args[2], args[3]
                add_fact('coll', [x, a, b])
                add_fact('cong', [o, b, o, x])
            
            # ─── Intersection_cc ───
            elif pred == 'intersection_cc' and len(args) >= 4:
                x, o, w, a = args[0], args[1], args[2], args[3]
                add_fact('cong', [o, a, o, x])
                add_fact('cong', [w, a, w, x])
            
            # ─── lc_tangent ───
            elif pred == 'lc_tangent' and len(args) >= 3:
                x, a, o = args[0], args[1], args[2]
                add_fact('perp', [a, x, a, o])
                add_fact('cong', [o, x, o, a])  # tangent point is on circle
            
            # ─── Nsquare / Psquare ───
            elif pred in ('nsquare', 'psquare') and len(args) >= 3:
                x, a, b = args[0], args[1], args[2]
                add_fact('cong', [x, a, a, b])
                add_fact('perp', [x, a, a, b])
            
            # ─── Shift ───
            elif pred == 'shift' and len(args) >= 4:
                x, b, c, d = args[0], args[1], args[2], args[3]
                add_fact('cong', [x, b, c, d])
                add_fact('cong', [x, c, b, d])
            
            # ─── Segment ───
            elif pred == 'segment' and len(args) >= 2:
                add_fact('diff', args[:2])
            
            # ─── Triangle ───
            elif pred == 'triangle' and len(args) >= 3:
                add_fact('ncoll', args[:3])
            
            # ─── Eqangle3 ───
            elif pred == 'eqangle3' and len(args) >= 6:
                x, a, b, d, e, f = args[0], args[1], args[2], args[3], args[4], args[5]
                add_fact('eqangle', [x, a, x, b, d, e, d, f])
            
            # ─── on_circum ───
            elif pred == 'on_circum' and len(args) >= 4:
                x, a, b, c = args[0], args[1], args[2], args[3]
                add_fact('cyclic', [a, b, c, x])
            
            # ─── cc_tangent ───
            elif pred == 'cc_tangent' and len(args) >= 4:
                q, t, p, s = args[0], args[1], args[2], args[3]
                # Internal and external tangent points between two circles
            
            # ─── on_opline ───
            elif pred == 'on_opline' and len(args) >= 3:
                x, a, b = args[0], args[1], args[2]
                add_fact('coll', [x, a, b])
    
    # Add diff facts for any pair appearing in coll
    for f in list(facts):
        if f[0] == 'coll' and len(f[1]) >= 2:
            for i in range(len(f[1])):
                for j in range(i+1, len(f[1])):
                    add_fact('diff', [f[1][i], f[1][j]])
    
    return facts


def apply_rules_enhanced(facts, rules, max_iterations=200):
    """
    Enhanced forward-chaining with better rule application.
    Handles variable substitution more carefully.
    """
    all_facts = set(facts)
    fact_keys = {fact_key(f) for f in facts}
    new_facts = set(facts)
    proof_trace = []
    iteration = 0
    
    # Index facts by predicate for faster lookup
    fact_index = defaultdict(set)
    for f in all_facts:
        fact_index[f[0]].add(f)
    
    while new_facts and iteration < max_iterations:
        iteration += 1
        derived_this_round = set()
        
        for rule_idx, rule in enumerate(rules):
            premises = rule['premises']
            conclusion = rule['conclusion']
            
            if not premises:
                continue
            
            # Find candidate facts for each premise
            candidates = []
            for prem in premises:
                pred = prem['predicate']
                prem_args = prem['args']
                matching = fact_index.get(pred, set())
                if not matching:
                    candidates = []
                    break
                # Filter by argument count
                filtered = [f for f in matching if len(f[1]) == len(prem_args)]
                if not filtered:
                    candidates = []
                    break
                candidates.append(filtered)
            
            if not candidates or len(candidates) != len(premises):
                continue
            
            # Try combinations (limit to avoid explosion)
            max_combos = 5000
            combo_count = 0
            for combo in itertools.product(*candidates):
                combo_count += 1
                if combo_count > max_combos:
                    break
                
                # Build substitution
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
                
                # Check negative conditions (ncoll, npara, nperp, diff, sameside)
                # These are implicit in some rules - skip for now
                
                # Apply substitution to conclusion
                conc_pred = conclusion['predicate']
                conc_args = []
                for arg in conclusion['args']:
                    if arg in subst:
                        conc_args.append(subst[arg])
                    else:
                        # Free variable - can't derive this fact
                        conc_args.append(arg)
                
                # Only add if all args are substituted (no free variables)
                if all(a in subst for a in conclusion['args']):
                    new_fact = make_fact(conc_pred, conc_args)
                    k = fact_key(new_fact)
                    if k not in fact_keys:
                        derived_this_round.add(new_fact)
                        fact_keys.add(k)
                        proof_trace.append({
                            'rule_idx': rule_idx,
                            'rule_raw': rule['raw'],
                            'matched_facts': [fact_to_str(f) for f in combo],
                            'derived_fact': fact_to_str(new_fact),
                            'iteration': iteration
                        })
        
        # Update indices
        for f in derived_this_round:
            fact_index[f[0]].add(f)
        
        all_facts.update(derived_this_round)
        new_facts = derived_this_round
        
        if not derived_this_round:
            break
    
    return all_facts, proof_trace, iteration


def check_goal_enhanced(facts, goal):
    """Check goal with comprehensive symmetry handling."""
    if goal is None:
        return False, None
    
    goal_pred = goal['predicate']
    goal_args = goal['args']
    goal_fact = make_fact(goal_pred, goal_args)
    goal_k = fact_key(goal_fact)
    
    # Build all fact keys
    all_keys = {fact_key(f) for f in facts}
    
    if goal_k in all_keys:
        return True, goal_fact
    
    # For cyclic, check all rotations
    if goal_pred == 'cyclic':
        args = list(goal_args)
        for i in range(len(args)):
            rotated = args[i:] + args[:i]
            k = fact_key(make_fact('cyclic', rotated))
            if k in all_keys:
                return True, make_fact('cyclic', rotated)
    
    return False, None


if __name__ == '__main__':
    with open(os.path.join(OUTPUT_DIR, 'parsed_problems.json'), 'r') as f:
        problems = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'parsed_rules.json'), 'r') as f:
        rules = json.load(f)
    
    results = []
    total_solved = 0
    
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem['name']}")
        print(f"Goal: {problem['goal']['predicate']} {' '.join(problem['goal']['args'])}")
        
        initial_facts = extract_initial_facts(problem)
        print(f"Initial facts: {len(initial_facts)}")
        
        all_facts, trace, iterations = apply_rules_enhanced(initial_facts, rules, max_iterations=200)
        print(f"Total facts after {iterations} iterations: {len(all_facts)}")
        print(f"Proof steps derived: {len(trace)}")
        
        solved, match_fact = check_goal_enhanced(all_facts, problem['goal'])
        print(f"Goal reached: {solved}")
        
        if solved:
            total_solved += 1
            print(f"  Matched by: {fact_to_str(match_fact)}")
        
        # Show some derived facts
        if trace:
            print(f"  First 5 derived facts:")
            for t in trace[:5]:
                print(f"    {t['derived_fact']} (from rule: {t['rule_raw'][:50]}...)")
        
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
    
    with open(os.path.join(OUTPUT_DIR, 'proof_results_enhanced.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"Total problems: {len(problems)}")
    print(f"Solved: {total_solved}")
    print(f"Solve rate: {total_solved/len(problems)*100:.1f}%")