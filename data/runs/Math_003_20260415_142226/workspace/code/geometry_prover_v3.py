#!/usr/bin/env python3
"""
Geometry Theorem Prover v3 - Correct definition expansion.

Key insight: Construction predicates (on_line, on_circle, on_tline, etc.) 
must be expanded into base geometric facts (coll, cong, perp, para, eqangle)
using the definitions from defs.txt before forward chaining can work.
"""

import json
import re
import os
from collections import defaultdict
from itertools import permutations

# ============================================================
# Fact Representation
# ============================================================

class Fact:
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = tuple(args)
    
    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return self.predicate == other.predicate and self.args == other.args
    
    def __hash__(self):
        return hash((self.predicate, self.args))
    
    def __repr__(self):
        return f"{self.predicate} {' '.join(self.args)}"


class Rule:
    def __init__(self, name, premises, conclusion, negated_conditions=None):
        self.name = name
        self.premises = premises
        self.conclusion = conclusion
        self.negated_conditions = negated_conditions or []


# ============================================================
# Parsers
# ============================================================

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
                rules.append(Rule(f"rule_{len(rules)}", premises, conclusion, negated))
    
    return rules


def load_definitions(defs_path):
    """Parse defs.txt into a mapping from construction name to derived facts.
    
    Format in defs.txt:
    ```
    midpoint x a b          <- header: name + formal parameters
    x : a b                 <- variable declaration (vars after ':')
    a b = diff a b          <- preconditions
    x : coll x a b, cong x a x b  <- derived facts for variable x
    midp a b                <- implementation hint
    ```
    
    We extract: for each construction, what base facts does it imply?
    """
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
        
        # Skip non-construction entries
        if def_name in ('pentagon', 'quadrangle', 'rectangle', 'square', 'isquare',
                        'trapezoid', 'r_trapezoid', 'r_triangle', 'risos', 'ieq_triangle',
                        'triangle12', 's_angle', 'lc_tangent', 'on_circum', 'tangent',
                        'cc_tangent0'):
            i += 1
            continue
        
        # Find the derived facts lines
        # These are lines with pattern: "vars : fact1, fact2, ..."
        # where facts use base predicates (coll, cong, perp, para, eqangle, eqratio, cyclic)
        j = i + 1
        derived_facts = []
        
        while j < len(lines):
            l = lines[j].strip()
            if not l:
                j += 1
                continue
            
            # Lines with ':' that contain derived facts
            if ':' in l and '=' not in l.split(':')[0]:
                fact_part = ':'.join(l.split(':')[1:]).strip()
                
                # Split by comma and semicolon
                for sep in [';', ',']:
                    fact_part = fact_part.replace(sep, '\n')
                
                for fp in fact_part.split('\n'):
                    fp = fp.strip()
                    if not fp:
                        continue
                    ftokens = fp.split()
                    if len(ftokens) >= 2:
                        pred = ftokens[0]
                        args = tuple(ftokens[1:])
                        derived_facts.append((pred, args))
            
            # Stop at implementation line
            if l == def_name or l.startswith(def_name + ' '):
                break
            
            j += 1
        
        if derived_facts:
            definitions[def_name] = {
                'name': def_name,
                'args': def_args,
                'derived_facts': derived_facts
            }
        
        i += 1
    
    return definitions


# ============================================================
# Knowledge Base
# ============================================================

class KnowledgeBase:
    def __init__(self, rules):
        self.rules = rules
        self.facts = set()
        self.proof_steps = []
        self.point_set = set()
    
    def add_fact(self, fact, rule=None, supporting=None):
        if fact not in self.facts:
            self.facts.add(fact)
            for arg in fact.args:
                self.point_set.add(arg)
            if rule and supporting:
                self.proof_steps.append({
                    'fact': str(fact),
                    'rule': rule.name if rule else 'initial',
                    'supporting': [str(s) for s in supporting]
                })
            return True
        return False
    
    def check_negated(self, negated_conditions):
        for neg in negated_conditions:
            if neg in self.facts:
                return False
        return True
    
    def _match_and_substitute(self, pattern, fact, existing_sub=None):
        """Try to match pattern against fact, returning substitution or None."""
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
    
    def _apply_sub(self, pattern, sub):
        new_args = tuple(sub.get(a, a) for a in pattern.args)
        return Fact(pattern.predicate, new_args)
    
    def try_apply_rule(self, rule):
        new_facts = []
        n_prem = len(rule.premises)
        facts_list = list(self.facts)
        
        if n_prem == 1:
            pat = rule.premises[0]
            for f1 in facts_list:
                sub = self._match_and_substitute(pat, f1)
                if sub is not None and self.check_negated(rule.negated_conditions):
                    nf = self._apply_sub(rule.conclusion, sub)
                    if self.add_fact(nf, rule, [f1]):
                        new_facts.append(nf)
        
        elif n_prem == 2:
            p1, p2 = rule.premises
            for i, f1 in enumerate(facts_list):
                sub1 = self._match_and_substitute(p1, f1)
                if sub1 is None:
                    continue
                for j, f2 in enumerate(facts_list):
                    if i == j:
                        continue
                    sub = self._match_and_substitute(p2, f2, sub1)
                    if sub is not None and self.check_negated(rule.negated_conditions):
                        nf = self._apply_sub(rule.conclusion, sub)
                        if self.add_fact(nf, rule, [f1, f2]):
                            new_facts.append(nf)
        
        elif n_prem == 3:
            p1, p2, p3 = rule.premises
            for i, f1 in enumerate(facts_list):
                sub1 = self._match_and_substitute(p1, f1)
                if sub1 is None:
                    continue
                for j, f2 in enumerate(facts_list):
                    if j == i:
                        continue
                    sub12 = self._match_and_substitute(p2, f2, sub1)
                    if sub12 is None:
                        continue
                    for k, f3 in enumerate(facts_list):
                        if k in (i, j):
                            continue
                        sub = self._match_and_substitute(p3, f3, sub12)
                        if sub is not None and self.check_negated(rule.negated_conditions):
                            nf = self._apply_sub(rule.conclusion, sub)
                            if self.add_fact(nf, rule, [f1, f2, f3]):
                                new_facts.append(nf)
        
        return new_facts
    
    def forward_chain(self, max_iterations=1000):
        iteration = 0
        total_new = 0
        
        while iteration < max_iterations:
            iteration += 1
            any_new = False
            for rule in self.rules:
                new = self.try_apply_rule(rule)
                if new:
                    any_new = True
                    total_new += len(new)
            if not any_new:
                break
        
        return total_new, iteration


# ============================================================
# Solver
# ============================================================

class GeometrySolver:
    def __init__(self, rules_path, defs_path):
        self.rules = load_rules(rules_path)
        self.defs = load_definitions(defs_path)
        print(f"Loaded {len(self.rules)} deduction rules")
        print(f"Loaded {len(self.defs)} construction definitions")
        for name, d in sorted(self.defs.items()):
            print(f"  {name}: {d['args']} -> {[f'{p} {a}' for p,a in d['derived_facts']]}")
    
    def expand_construction(self, constr):
        """Expand construction predicates into base facts using definitions."""
        facts = []
        for pred_info in constr.get('predicates', []):
            pred_name = pred_info['predicate']
            pred_args = pred_info['args']
            
            # Check if this is a definable construction
            if pred_name in self.defs:
                d = self.defs[pred_name]
                # Map formal args to actual args
                formal_args = d['args']
                mapping = {}
                for idx, fa in enumerate(formal_args):
                    if idx < len(pred_args):
                        mapping[fa] = pred_args[idx]
                
                # Apply mapping to derived facts
                for fact_pred, fact_args in d['derived_facts']:
                    new_args = tuple(mapping.get(a, a) for a in fact_args)
                    facts.append(Fact(fact_pred, new_args))
            
            # Always keep the original predicate too
            facts.append(Fact(pred_name, tuple(pred_args)))
        
        return facts
    
    def parse_problem(self, problem_data):
        kb = KnowledgeBase(self.rules)
        
        constructions = problem_data.get('constructions', [])
        conclusion_data = problem_data.get('conclusion', {})
        
        for constr in constructions:
            expanded = self.expand_construction(constr)
            for fact in expanded:
                kb.add_fact(fact)
        
        target = Fact(
            conclusion_data.get('predicate', ''),
            tuple(conclusion_data.get('args', []))
        )
        
        return kb, target
    
    def solve(self, problem_data, max_iterations=1000):
        kb, target = self.parse_problem(problem_data)
        initial_count = len(kb.facts)
        
        n_derived, n_iters = kb.forward_chain(max_iterations)
        
        proven = target in kb.facts
        if not proven:
            proven = self._check_symmetric(kb, target)
        
        return {
            'proven': proven,
            'target': str(target),
            'initial_facts': initial_count,
            'derived_facts': n_derived,
            'total_facts': len(kb.facts),
            'iterations': n_iters,
            'proof_steps': kb.proof_steps,
            'all_facts': sorted([str(f) for f in kb.facts]),
            'point_set': list(kb.point_set)
        }
    
    def _check_symmetric(self, kb, target):
        pred = target.predicate
        args = target.args
        
        if pred == 'cong' and len(args) == 4:
            symms = [
                (args[2], args[3], args[0], args[1]),
                (args[1], args[0], args[3], args[2]),
                (args[3], args[2], args[1], args[0]),
            ]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        elif pred == 'coll':
            for perm in permutations(args):
                if Fact(pred, perm) in kb.facts:
                    return True
        
        elif pred == 'cyclic':
            for i in range(len(args)):
                rotated = args[i:] + args[:i]
                if Fact(pred, tuple(rotated)) in kb.facts:
                    return True
                if Fact(pred, tuple(reversed(rotated))) in kb.facts:
                    return True
        
        elif pred == 'para' and len(args) == 4:
            if Fact(pred, (args[2], args[3], args[0], args[1])) in kb.facts:
                return True
        
        elif pred == 'perp' and len(args) == 4:
            if Fact(pred, (args[2], args[3], args[0], args[1])) in kb.facts:
                return True
        
        elif pred == 'eqangle' and len(args) == 8:
            if Fact(pred, (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3])) in kb.facts:
                return True
        
        elif pred == 'eqratio' and len(args) == 8:
            if Fact(pred, (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3])) in kb.facts:
                return True
        
        return False


# ============================================================
# Proof Generator
# ============================================================

class ProofGenerator:
    @staticmethod
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
        elif pred == 'midp' and len(args) == 3:
            return f"{args[0]} is midpoint of {args[1]}{args[2]}"
        else:
            return f"{pred}({' '.join(args)})"
    
    @staticmethod
    def generate_proof(result, problem_name):
        lines = []
        lines.append(f"# Proof for {problem_name}")
        lines.append(f"")
        lines.append(f"**Target:** {ProofGenerator.format_fact(result['target'])}")
        lines.append(f"")
        
        if result['proven']:
            lines.append(f"**Status:** ✅ PROVEN")
        else:
            lines.append(f"**Status:** ❌ Not proven by forward chaining")
        
        lines.append(f"")
        lines.append(f"## Statistics")
        lines.append(f"- Initial facts: {result['initial_facts']}")
        lines.append(f"- Derived facts: {result['derived_facts']}")
        lines.append(f"- Total facts in KB: {result['total_facts']}")
        lines.append(f"- Forward chaining iterations: {result['iterations']}")
        lines.append(f"")
        
        if result['proof_steps']:
            lines.append(f"## Proof Steps")
            lines.append(f"")
            for i, step in enumerate(result['proof_steps'][:50]):
                lines.append(f"{i+1}. **{ProofGenerator.format_fact(step['fact'])}**")
                lines.append(f"   - Rule: `{step['rule']}`")
                lines.append(f"   - From: {', '.join(ProofGenerator.format_fact(s) for s in step['supporting'])}")
                lines.append(f"")
        
        return '\n'.join(lines)


# ============================================================
# Main
# ============================================================

def main():
    workspace = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(workspace, 'data')
    output_dir = os.path.join(workspace, 'outputs')
    
    rules_path = os.path.join(data_dir, 'rules.txt')
    defs_path = os.path.join(data_dir, 'defs.txt')
    problems_path = os.path.join(output_dir, 'parsed_problems.json')
    
    with open(problems_path, 'r') as f:
        problems = json.load(f)
    
    solver = GeometrySolver(rules_path, defs_path)
    
    results = {}
    for name, prob_data in problems.items():
        print(f"\n{'='*60}")
        print(f"Solving: {name}")
        print(f"{'='*60}")
        
        result = solver.solve(prob_data, max_iterations=1000)
        results[name] = result
        
        status = "✅ PROVEN" if result['proven'] else "❌ Not proven"
        print(f"  Result: {status}")
        print(f"  Facts: {result['total_facts']} ({result['derived_facts']} derived in {result['iterations']} iters)")
        
        # Print some derived facts for debugging
        if result['derived_facts'] > 0:
            derived = [f for f in result['all_facts'] if not any(
                f.startswith(p + ' ') for p in ['on_', 'circle', 'triangle', 'segment', 'bisect', 'line', 'tline', 'pline', 'bline', 'pmirror', 'amirror', 'reflect', 'isos', 'midp', 'rotaten90', 'rotatep90', 'aline', 'aline2', 'dia', 'eq_quadrangle', 'eq_trapezoid', 'eq_triangle', 'eqdia_quadrangle', 'eqdistance', 'incenter2', 'excenter2', 'centroid', 'ninepoints', 'intersection', 'iso_triangle', 'lc_tangent', 'nsquare', 'psquare', 'r_trapezoid', 'r_triangle', 'rectangle', 'risos', 's_angle', 'shift', 'square', 'isquare', 'trapezoid', 'triangle12', '2l1c', 'e5128', '3peq', 'trisect', 'trisegment', 'on_dia', 'ieq_triangle', 'on_opline', 'cc_tangent0', 'cc_tangent', 'eqangle3', 'tangent', 'on_circum', 'pentagon', 'quadrangle', 'free', 'angle_bisector', 'angle_mirror', 'excenter', 'incenter', 'orthocenter', 'circumcenter', 'foot', 'mirror', 'eqangle2', 'parallelogram', 'eqangle', 'eqratio', 'cong', 'coll', 'cyclic', 'para', 'perp', 'midp', 'ncoll', 'npara', 'nperp', 'diff', 'sameside']
            )]
            print(f"  New derived facts: {derived[:10]}")
    
    # Save results
    results_path = os.path.join(output_dir, 'solver_results_v3.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate proofs
    proofs_dir = os.path.join(output_dir, 'proofs_v3')
    os.makedirs(proofs_dir, exist_ok=True)
    
    for name, result in results.items():
        proof_text = ProofGenerator.generate_proof(result, name)
        proof_path = os.path.join(proofs_dir, f"{name}_proof.md")
        with open(proof_path, 'w') as f:
            f.write(proof_text)
    
    proven_count = sum(1 for r in results.values() if r['proven'])
    total_count = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {proven_count}/{total_count} problems proven ({100*proven_count/total_count:.1f}%)")
    print(f"{'='*60}")
    
    summary = {
        'total_problems': total_count,
        'proven_count': proven_count,
        'success_rate': proven_count / total_count,
        'per_problem': {
            name: {
                'proven': r['proven'],
                'target': r['target'],
                'total_facts': r['total_facts'],
                'derived_facts': r['derived_facts'],
                'iterations': r['iterations']
            }
            for name, r in results.items()
        }
    }
    
    summary_path = os.path.join(output_dir, 'summary_v3.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
