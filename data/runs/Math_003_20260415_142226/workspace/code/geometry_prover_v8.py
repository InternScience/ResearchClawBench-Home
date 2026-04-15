#!/usr/bin/env python3
"""
Geometry Theorem Prover v8 - Enhanced with extended rules and better matching.
Adds transitivity rules, symmetry rules, and cyclic predicate handling.
"""

import json
import os
from itertools import permutations, combinations

BASE_PREDS = {'coll', 'cong', 'perp', 'para', 'eqangle', 'eqratio', 'cyclic'}


class Fact:
    __slots__ = ('predicate', 'args')
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = tuple(args)
    def __eq__(self, other):
        return isinstance(other, Fact) and self.predicate == other.predicate and self.args == other.args
    def __hash__(self):
        return hash((self.predicate, self.args))
    def __repr__(self):
        return f"{self.predicate} {' '.join(self.args)}"


def parse_fact(s):
    t = s.strip().split()
    return Fact(t[0], t[1:]) if t else None


def load_rules(path):
    rules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=>' not in line:
                continue
            pre_str, con_str = [x.strip() for x in line.split('=>')]
            premises, negated = [], []
            for p in pre_str.split(','):
                pf = parse_fact(p)
                if pf:
                    (negated if pf.predicate.startswith('n') else premises).append(pf)
            conclusion = parse_fact(con_str)
            if conclusion and premises:
                rules.append({'premises': premises, 'conclusion': conclusion, 'negated': negated})
    return rules


def load_definitions(path):
    definitions = {}
    with open(path) as f:
        content = f.read()
    for block in content.split('\n\n'):
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) < 2:
            continue
        header = lines[0].split()
        def_name, def_args = header[0], header[1:]
        skip = {'pentagon', 'quadrangle', 'rectangle', 'square', 'isquare',
                'trapezoid', 'r_trapezoid', 'r_triangle', 'risos', 'ieq_triangle',
                'triangle12', 's_angle', 'lc_tangent', 'on_circum', 'tangent',
                'cc_tangent0', 'segment', 'free'}
        if def_name in skip:
            continue
        derived_facts = []
        for line in lines[1:]:
            if ':' not in line or '=' in line.split(':')[0]:
                continue
            fact_part = ':'.join(line.split(':')[1:]).strip()
            for item in fact_part.replace(';', ',').split(','):
                item = item.strip()
                if not item:
                    continue
                tokens = item.split()
                if len(tokens) >= 2 and tokens[0] in BASE_PREDS:
                    derived_facts.append((tokens[0], tuple(tokens[1:])))
        if derived_facts:
            definitions[def_name] = {'args': def_args, 'derived_facts': derived_facts}
    return definitions


def expand_construction(constr, defs):
    facts = []
    for pred_info in constr.get('predicates', []):
        pname = pred_info['predicate']
        pargs = pred_info['args']
        if pname in defs:
            d = defs[pname]
            mapping = {d['args'][i]: pargs[i] for i in range(min(len(d['args']), len(pargs)))}
            for fpred, fargs in d['derived_facts']:
                facts.append(Fact(fpred, tuple(mapping.get(a, a) for a in fargs)))
        facts.append(Fact(pname, tuple(pargs)))
    return facts


def _match(pattern, fact, sub=None):
    if pattern.predicate != fact.predicate or len(pattern.args) != len(fact.args):
        return None
    sub = dict(sub) if sub else {}
    for pa, fa in zip(pattern.args, fact.args):
        if pa in sub:
            if sub[pa] != fa:
                return None
        else:
            sub[pa] = fa
    return sub


def _apply(pattern, sub):
    return Fact(pattern.predicate, tuple(sub.get(a, a) for a in pattern.args))


def _check_neg(negated, facts):
    return all(n not in facts for n in negated)


def check_symmetric(facts, target):
    pred, args = target.predicate, target.args
    if pred == 'cong' and len(args) == 4:
        for s in [(args[2],args[3],args[0],args[1]),(args[1],args[0],args[3],args[2]),(args[3],args[2],args[1],args[0])]:
            if Fact(pred, s) in facts: return True
    elif pred == 'coll':
        for p in permutations(args):
            if Fact(pred, p) in facts: return True
    elif pred == 'cyclic':
        for i in range(len(args)):
            r = args[i:] + args[:i]
            if Fact(pred, tuple(r)) in facts or Fact(pred, tuple(reversed(r))) in facts: return True
    elif pred in ('para','perp') and len(args) == 4:
        if Fact(pred, (args[2],args[3],args[0],args[1])) in facts: return True
    elif pred == 'eqangle' and len(args) == 8:
        if Fact(pred, (args[4],args[5],args[6],args[7],args[0],args[1],args[2],args[3])) in facts: return True
    elif pred == 'eqratio' and len(args) == 8:
        if Fact(pred, (args[4],args[5],args[6],args[7],args[0],args[1],args[2],args[3])) in facts: return True
    return False


def solve(problem_data, rules, defs, max_iter=300, max_facts=50000):
    constructions = problem_data.get('constructions', [])
    conclusion = problem_data.get('conclusion', {})
    
    facts = set()
    proof_steps = []
    points = set()
    
    for constr in constructions:
        for f in expand_construction(constr, defs):
            if f not in facts:
                facts.add(f)
                points.update(f.args)
    
    initial_count = len(facts)
    total_derived = 0
    facts_list = list(facts)
    iteration = 0
    
    for iteration in range(1, max_iter + 1):
        new_this_iter = []
        for rule in rules:
            np = len(rule['premises'])
            if np == 1:
                pat = rule['premises'][0]
                for f1 in facts_list:
                    sub = _match(pat, f1)
                    if sub is not None and _check_neg(rule['negated'], facts):
                        nf = _apply(rule['conclusion'], sub)
                        if nf not in facts:
                            facts.add(nf); new_this_iter.append(nf); total_derived += 1
                            proof_steps.append({'fact': str(nf), 'rule': f"rule_{rules.index(rule)}", 'supporting': [str(f1)]})
                            if len(facts) > max_facts: break
            elif np == 2:
                p1, p2 = rule['premises']
                for i1, f1 in enumerate(facts_list):
                    sub1 = _match(p1, f1)
                    if sub1 is None: continue
                    for i2, f2 in enumerate(facts_list):
                        if i1 == i2: continue
                        sub = _match(p2, f2, sub1)
                        if sub is not None and _check_neg(rule['negated'], facts):
                            nf = _apply(rule['conclusion'], sub)
                            if nf not in facts:
                                facts.add(nf); new_this_iter.append(nf); total_derived += 1
                                proof_steps.append({'fact': str(nf), 'rule': f"rule_{rules.index(rule)}", 'supporting': [str(f1), str(f2)]})
                                if len(facts) > max_facts: break
                        if len(facts) > max_facts: break
                    if len(facts) > max_facts: break
            elif np == 3:
                p1, p2, p3 = rule['premises']
                for i1, f1 in enumerate(facts_list):
                    sub1 = _match(p1, f1)
                    if sub1 is None: continue
                    for i2, f2 in enumerate(facts_list):
                        if i2 == i1: continue
                        sub12 = _match(p2, f2, sub1)
                        if sub12 is None: continue
                        for i3, f3 in enumerate(facts_list):
                            if i3 in (i1, i2): continue
                            sub = _match(p3, f3, sub12)
                            if sub is not None and _check_neg(rule['negated'], facts):
                                nf = _apply(rule['conclusion'], sub)
                                if nf not in facts:
                                    facts.add(nf); new_this_iter.append(nf); total_derived += 1
                                    proof_steps.append({'fact': str(nf), 'rule': f"rule_{rules.index(rule)}", 'supporting': [str(f1), str(f2), str(f3)]})
                                    if len(facts) > max_facts: break
                            if len(facts) > max_facts: break
                        if len(facts) > max_facts: break
                    if len(facts) > max_facts: break
            if len(facts) > max_facts: break
        facts_list.extend(new_this_iter)
        if not new_this_iter: break
        if len(facts) > max_facts: break
    
    target = Fact(conclusion.get('predicate', ''), tuple(conclusion.get('args', [])))
    proven = target in facts or check_symmetric(facts, target)
    
    # Compute how close we got: check if any derived fact shares structure with target
    closeness = compute_closeness(facts, target)
    
    return {
        'proven': proven, 'target': str(target),
        'initial_facts': initial_count, 'derived_facts': total_derived,
        'total_facts': len(facts), 'iterations': iteration,
        'proof_steps': proof_steps[:100],
        'all_facts': sorted([str(f) for f in facts]),
        'point_set': list(points), 'hit_max': len(facts) > max_facts,
        'closeness': closeness
    }


def compute_closeness(facts, target):
    """Measure how close the KB came to proving the target."""
    pred, args = target.predicate, target.args
    
    # Check for partial matches
    same_pred = [f for f in facts if f.predicate == pred]
    
    # For cong targets, check if we have any cong involving the same points
    if pred == 'cong' and len(args) == 4:
        target_points = set(args)
        related = [f for f in same_pred if set(f.args) & target_points]
        return {'same_predicate': len(same_pred), 'related_to_target': len(related),
                'total_facts': len(facts)}
    
    return {'same_predicate': len(same_pred), 'total_facts': len(facts)}


def format_fact(s):
    t = s.split(); p = t[0]; a = t[1:]
    if p == 'cong' and len(a)==4: return f"Segment {a[0]}{a[1]} ≅ Segment {a[2]}{a[3]}"
    if p == 'coll': return f"Points {', '.join(a)} are collinear"
    if p == 'cyclic': return f"Points {', '.join(a)} are concyclic"
    if p == 'para' and len(a)==4: return f"Line {a[0]}{a[1]} ∥ Line {a[2]}{a[3]}"
    if p == 'perp' and len(a)==4: return f"Line {a[0]}{a[1]} ⊥ Line {a[2]}{a[3]}"
    if p == 'eqangle' and len(a)==8: return f"∠({a[0]}{a[1]}, {a[2]}{a[3]}) = ∠({a[4]}{a[5]}, {a[6]}{a[7]})"
    if p == 'eqratio' and len(a)==8: return f"|{a[0]}{a[1]}|/|{a[2]}{a[3]}| = |{a[4]}{a[5]}|/|{a[6]}{a[7]}|"
    return f"{p}({' '.join(a)})"


def main():
    ws = os.path.join(os.path.dirname(__file__), '..')
    rules = load_rules(os.path.join(ws, 'data', 'rules.txt'))
    defs = load_definitions(os.path.join(ws, 'data', 'defs.txt'))
    
    print(f"Loaded {len(rules)} rules, {len(defs)} definitions")
    
    with open(os.path.join(ws, 'outputs', 'parsed_problems.json')) as f:
        problems = json.load(f)
    
    results = {}
    for name, prob in problems.items():
        print(f"Solving: {name}...", end=' ', flush=True)
        r = solve(prob, rules, defs)
        results[name] = r
        print(f"{'PROVEN' if r['proven'] else 'Not proven'} ({r['total_facts']} facts, {r['derived_facts']} derived)")
    
    out = os.path.join(ws, 'outputs')
    with open(os.path.join(out, 'solver_results_v8.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    proofs_dir = os.path.join(out, 'proofs_v8')
    os.makedirs(proofs_dir, exist_ok=True)
    for name, r in results.items():
        lines = [f"# Proof for {name}", "", f"**Target:** {format_fact(r['target'])}", ""]
        lines.append("**Status:** ✅ PROVEN" if r['proven'] else "**Status:** ❌ Not proven")
        lines.extend(["", "## Statistics", f"- Initial facts: {r['initial_facts']}",
            f"- Derived facts: {r['derived_facts']}", f"- Total facts: {r['total_facts']}",
            f"- Iterations: {r['iterations']}", ""])
        if r['proof_steps']:
            lines.append("## Proof Steps\n")
            for i, step in enumerate(r['proof_steps'][:30]):
                lines.append(f"{i+1}. **{format_fact(step['fact'])}**")
                lines.append(f"   - Rule: `{step['rule']}`")
                lines.append(f"   - From: {', '.join(format_fact(s) for s in step['supporting'])}")
                lines.append("")
        with open(os.path.join(proofs_dir, f"{name}_proof.md"), 'w') as f:
            f.write('\n'.join(lines))
    
    pc = sum(1 for r in results.values() if r['proven'])
    tc = len(results)
    print(f"\nSUMMARY: {pc}/{tc} proven ({100*pc/tc:.1f}%)")
    
    summary = {
        'total_problems': tc, 'proven_count': pc, 'success_rate': pc/tc,
        'per_problem': {n: {'proven': r['proven'], 'target': r['target'],
            'total_facts': r['total_facts'], 'derived_facts': r['derived_facts'],
            'iterations': r['iterations'], 'closeness': r.get('closeness', {})} 
            for n, r in results.items()}
    }
    with open(os.path.join(out, 'summary_v8.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
