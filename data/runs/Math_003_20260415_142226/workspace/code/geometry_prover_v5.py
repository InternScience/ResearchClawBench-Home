#!/usr/bin/env python3
"""
Geometry Theorem Prover v5 - Correct definition parsing.

Key fix: Only extract lines with base geometric predicates (coll, cong, perp, 
para, eqangle, eqratio, cyclic) as derived facts. Ignore implementation hints.
"""

import json
import os
from itertools import permutations

# Base predicates that represent actual geometric facts
BASE_PREDS = {'coll', 'cong', 'perp', 'para', 'eqangle', 'eqratio', 'cyclic', 'midp'}


class Fact:
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = tuple(args)
    
    def __eq__(self, other):
        return isinstance(other, Fact) and self.predicate == other.predicate and self.args == other.args
    
    def __hash__(self):
        return hash((self.predicate, self.args))
    
    def __repr__(self):
        return f"{self.predicate} {' '.join(self.args)}"


def parse_fact_string(s):
    tokens = s.strip().split()
    if not tokens:
        return None
    return Fact(tokens[0], tokens[1:])


def load_rules(rules_path):
    rules = []
    with open(rules_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=>' not in line:
                continue
            parts = line.split('=>')
            premise_str = parts[0].strip()
            conclusion_str = parts[1].strip()
            premises = []
            negated = []
            for p in premise_str.split(','):
                p = p.strip()
                if not p:
                    continue
                pf = parse_fact_string(p)
                if pf:
                    if pf.predicate.startswith('n'):
                        negated.append(pf)
                    else:
                        premises.append(pf)
            conclusion = parse_fact_string(conclusion_str)
            if conclusion and premises:
                rules.append({'premises': premises, 'conclusion': conclusion, 'negated': negated, 'name': f"rule_{len(rules)}"})
    return rules


def load_definitions(defs_path):
    """Parse defs.txt carefully, extracting only base-predicate derived facts."""
    definitions = {}
    
    with open(defs_path, 'r') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        tokens = line.split()
        if len(tokens) < 2:
            i += 1
            continue
        
        def_name = tokens[0]
        def_args = tokens[1:]
        
        # Find the block for this definition
        j = i + 1
        derived_facts = []
        
        while j < len(lines):
            l = lines[j].strip()
            if not l:
                j += 1
                continue
            
            # Stop at implementation line (single word matching def_name or starting with it)
            impl_tokens = l.split()
            if impl_tokens and impl_tokens[0] == def_name and len(impl_tokens) <= len(def_args) + 1:
                break
            
            # Check for derived facts line: "vars : pred arg1 arg2, pred arg1 arg2 ..."
            if ':' in l and '=' not in l.split(':')[0]:
                var_part = l.split(':')[0].strip()
                fact_part = ':'.join(l.split(':')[1:]).strip()
                
                # Split by comma and semicolon
                fact_items = []
                for sep in [';', ',']:
                    new_items = []
                    for item in (fact_items if fact_items else [fact_part]):
                        new_items.extend(item.split(sep))
                    fact_items = new_items
                
                for fi in fact_items:
                    fi = fi.strip()
                    if not fi:
                        continue
                    ftokens = fi.split()
                    if len(ftokens) >= 2 and ftokens[0] in BASE_PREDS:
                        derived_facts.append((ftokens[0], tuple(ftokens[1:])))
            
            j += 1
        
        if derived_facts:
            definitions[def_name] = {'args': def_args, 'derived_facts': derived_facts}
        
        i += 1
    
    return definitions


def expand_construction(constr, defs):
    """Expand construction predicates into base facts using definitions."""
    facts = []
    for pred_info in constr.get('predicates', []):
        pred_name = pred_info['predicate']
        pred_args = pred_info['args']
        
        if pred_name in defs:
            d = defs[pred_name]
            formal_args = d['args']
            mapping = {}
            for idx, fa in enumerate(formal_args):
                if idx < len(pred_args):
                    mapping[fa] = pred_args[idx]
            
            for fact_pred, fact_args in d['derived_facts']:
                new_args = tuple(mapping.get(a, a) for a in fact_args)
                facts.append(Fact(fact_pred, new_args))
        
        # Keep original predicate too
        facts.append(Fact(pred_name, tuple(pred_args)))
    
    return facts


def _match(pattern, fact, existing_sub=None):
    if pattern.predicate != fact.predicate:
        return None
    if len(pattern.args) != len(fact.args):
        return None
    sub = dict(existing_sub) if existing_sub else {}
    for p_arg, f_arg in zip(pattern.args, fact.args):
        if p_arg in sub:
            if sub[p_arg] != f_arg:
                return None
        else:
            sub[p_arg] = f_arg
    return sub


def _apply_sub(pattern, sub):
    new_args = tuple(sub.get(a, a) for a in pattern.args)
    return Fact(pattern.predicate, new_args)


def _check_negated(negated, facts):
    for neg in negated:
        if neg in facts:
            return False
    return True


def check_symmetric(facts, target):
    pred = target.predicate
    args = target.args
    
    if pred == 'cong' and len(args) == 4:
        symms = [(args[2], args[3], args[0], args[1]), (args[1], args[0], args[3], args[2]), (args[3], args[2], args[1], args[0])]
        for s in symms:
            if Fact(pred, s) in facts:
                return True
    elif pred == 'coll':
        for perm in permutations(args):
            if Fact(pred, perm) in facts:
                return True
    elif pred == 'cyclic':
        for i in range(len(args)):
            rotated = args[i:] + args[:i]
            if Fact(pred, tuple(rotated)) in facts:
                return True
            if Fact(pred, tuple(reversed(rotated))) in facts:
                return True
    elif pred in ('para', 'perp') and len(args) == 4:
        if Fact(pred, (args[2], args[3], args[0], args[1])) in facts:
            return True
    elif pred == 'eqangle' and len(args) == 8:
        if Fact(pred, (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3])) in facts:
            return True
    elif pred == 'eqratio' and len(args) == 8:
        if Fact(pred, (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3])) in facts:
            return True
    return False


def solve_problem(problem_data, rules, defs, max_iterations=200, max_facts=30000):
    constructions = problem_data.get('constructions', [])
    conclusion_data = problem_data.get('conclusion', {})
    
    facts = set()
    proof_steps = []
    point_set = set()
    
    for constr in constructions:
        expanded = expand_construction(constr, defs)
        for fact in expanded:
            if fact not in facts:
                facts.add(fact)
                for arg in fact.args:
                    point_set.add(arg)
    
    initial_count = len(facts)
    total_derived = 0
    iteration = 0
    facts_list = list(facts)
    
    for iteration in range(1, max_iterations + 1):
        any_new = False
        new_this_iter = []
        
        for rule in rules:
            n_prem = len(rule['premises'])
            
            if n_prem == 1:
                pat = rule['premises'][0]
                for f1 in facts_list:
                    sub = _match(pat, f1)
                    if sub is not None:
                        if _check_negated(rule['negated'], facts):
                            nf = _apply_sub(rule['conclusion'], sub)
                            if nf not in facts:
                                facts.add(nf)
                                new_this_iter.append(nf)
                                total_derived += 1
                                proof_steps.append({
                                    'fact': str(nf), 'rule': rule['name'], 'supporting': [str(f1)]
                                })
                                if len(facts) > max_facts:
                                    break
            
            elif n_prem == 2:
                p1, p2 = rule['premises']
                for i_idx, f1 in enumerate(facts_list):
                    sub1 = _match(p1, f1)
                    if sub1 is None:
                        continue
                    for j_idx, f2 in enumerate(facts_list):
                        if i_idx == j_idx:
                            continue
                        sub = _match(p2, f2, sub1)
                        if sub is not None:
                            if _check_negated(rule['negated'], facts):
                                nf = _apply_sub(rule['conclusion'], sub)
                                if nf not in facts:
                                    facts.add(nf)
                                    new_this_iter.append(nf)
                                    total_derived += 1
                                    proof_steps.append({
                                        'fact': str(nf), 'rule': rule['name'], 'supporting': [str(f1), str(f2)]
                                    })
                                    if len(facts) > max_facts:
                                        break
                    if len(facts) > max_facts:
                        break
            
            elif n_prem == 3:
                p1, p2, p3 = rule['premises']
                for i_idx, f1 in enumerate(facts_list):
                    sub1 = _match(p1, f1)
                    if sub1 is None:
                        continue
                    for j_idx, f2 in enumerate(facts_list):
                        if j_idx == i_idx:
                            continue
                        sub12 = _match(p2, f2, sub1)
                        if sub12 is None:
                            continue
                        for k_idx, f3 in enumerate(facts_list):
                            if k_idx in (i_idx, j_idx):
                                continue
                            sub = _match(p3, f3, sub12)
                            if sub is not None:
                                if _check_negated(rule['negated'], facts):
                                    nf = _apply_sub(rule['conclusion'], sub)
                                    if nf not in facts:
                                        facts.add(nf)
                                        new_this_iter.append(nf)
                                        total_derived += 1
                                        proof_steps.append({
                                            'fact': str(nf), 'rule': rule['name'], 'supporting': [str(f1), str(f2), str(f3)]
                                        })
                                        if len(facts) > max_facts:
                                            break
                            if len(facts) > max_facts:
                                break
                        if len(facts) > max_facts:
                            break
                    if len(facts) > max_facts:
                        break
            
            if len(facts) > max_facts:
                break
        
        facts_list.extend(new_this_iter)
        if not new_this_iter:
            break
        if len(facts) > max_facts:
            break
    
    target = Fact(conclusion_data.get('predicate', ''), tuple(conclusion_data.get('args', [])))
    proven = target in facts
    if not proven:
        proven = check_symmetric(facts, target)
    
    return {
        'proven': proven,
        'target': str(target),
        'initial_facts': initial_count,
        'derived_facts': total_derived,
        'total_facts': len(facts),
        'iterations': iteration,
        'proof_steps': proof_steps[:100],
        'all_facts': sorted([str(f) for f in facts]),
        'point_set': list(point_set),
        'hit_max_facts': len(facts) > max_facts
    }


def format_fact(fact_str):
    tokens = fact_str.split()
    pred = tokens[0]
    args = tokens[1:]
    if pred == 'cong' and len(args) == 4:
        return f"Segment {args[0]}{args[1]} ≅ Segment {args[2]}{args[3]}"
    elif pred == 'coll':
        return f"Points {', '.join(args)} are collinear"
    elif pred == 'cyclic':
        return f"Points {', '.join(args)} are concyclic"
    elif pred == 'para' and len(args) == 4:
        return f"Line {args[0]}{args[1]} ∥ Line {args[2]}{args[3]}"
    elif pred == 'perp' and len(args) == 4:
        return f"Line {args[0]}{args[1]} ⊥ Line {args[2]}{args[3]}"
    elif pred == 'eqangle' and len(args) == 8:
        return f"∠({args[0]}{args[1]}, {args[2]}{args[3]}) = ∠({args[4]}{args[5]}, {args[6]}{args[7]})"
    elif pred == 'eqratio' and len(args) == 8:
        return f"|{args[0]}{args[1]}|/|{args[2]}{args[3]}| = |{args[4]}{args[5]}|/|{args[6]}{args[7]}|"
    else:
        return f"{pred}({' '.join(args)})"


def main():
    workspace = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(workspace, 'data')
    output_dir = os.path.join(workspace, 'outputs')
    
    rules = load_rules(os.path.join(data_dir, 'rules.txt'))
    defs = load_definitions(os.path.join(data_dir, 'defs.txt'))
    
    print(f"Loaded {len(rules)} rules, {len(defs)} definitions")
    for name, d in sorted(defs.items()):
        print(f"  {name}({d['args']}): {[f'{p} {a}' for p,a in d['derived_facts']]}")
    
    with open(os.path.join(output_dir, 'parsed_problems.json'), 'r') as f:
        problems = json.load(f)
    
    results = {}
    for name, prob_data in problems.items():
        print(f"\nSolving: {name}...", end=' ', flush=True)
        result = solve_problem(prob_data, rules, defs, max_iterations=200, max_facts=30000)
        results[name] = result
        status = "PROVEN" if result['proven'] else "Not proven"
        print(f"{status} ({result['total_facts']} facts, {result['derived_facts']} derived)")
    
    # Save results
    with open(os.path.join(output_dir, 'solver_results_v5.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate proofs
    proofs_dir = os.path.join(output_dir, 'proofs_v5')
    os.makedirs(proofs_dir, exist_ok=True)
    
    for name, result in results.items():
        lines = [f"# Proof for {name}", "", f"**Target:** {format_fact(result['target'])}", ""]
        if result['proven']:
            lines.append("**Status:** ✅ PROVEN")
        else:
            lines.append("**Status:** ❌ Not proven")
        lines.extend(["", "## Statistics",
            f"- Initial facts: {result['initial_facts']}",
            f"- Derived facts: {result['derived_facts']}",
            f"- Total facts: {result['total_facts']}",
            f"- Iterations: {result['iterations']}", ""])
        if result['proof_steps']:
            lines.append("## Proof Steps\n")
            for i, step in enumerate(result['proof_steps'][:30]):
                lines.append(f"{i+1}. **{format_fact(step['fact'])}**")
                lines.append(f"   - Rule: `{step['rule']}`")
                lines.append(f"   - From: {', '.join(format_fact(s) for s in step['supporting'])}")
                lines.append("")
        with open(os.path.join(proofs_dir, f"{name}_proof.md"), 'w') as f:
            f.write('\n'.join(lines))
    
    proven_count = sum(1 for r in results.values() if r['proven'])
    total_count = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {proven_count}/{total_count} proven ({100*proven_count/total_count:.1f}%)")
    print(f"{'='*60}")
    
    summary = {
        'total_problems': total_count,
        'proven_count': proven_count,
        'success_rate': proven_count / total_count,
        'per_problem': {name: {'proven': r['proven'], 'target': r['target'], 'total_facts': r['total_facts'], 'derived_facts': r['derived_facts'], 'iterations': r['iterations']} for name, r in results.items()}
    }
    with open(os.path.join(output_dir, 'summary_v5.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
