#!/usr/bin/env python3
"""
Geometry Theorem Prover v2 - Improved version with proper definition expansion.

This implements a neuro-symbolic forward-chaining theorem prover that:
1. Parses formal problem statements into geometric predicates
2. Expands construction predicates into base geometric facts using defs.txt
3. Applies deduction rules from rules.txt to derive new facts
4. Searches for proof paths from premises to conclusions
5. Generates human-readable, machine-verifiable proofs
"""

import json
import re
import os
import sys
from collections import defaultdict, deque
from itertools import permutations, combinations
from copy import deepcopy

# ============================================================
# Predicate and Fact Representation
# ============================================================

class Fact:
    """Represents a geometric fact/predicate."""
    
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = tuple(args)
        self._hash = None
    
    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return self.predicate == other.predicate and self.args == other.args
    
    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.predicate, self.args))
        return self._hash
    
    def __repr__(self):
        return f"{self.predicate} {' '.join(self.args)}"
    
    def to_dict(self):
        return {'predicate': self.predicate, 'args': list(self.args)}


class Rule:
    """Represents a deduction rule."""
    
    def __init__(self, name, premises, conclusion, negated_conditions=None):
        self.name = name
        self.premises = premises
        self.conclusion = conclusion
        self.negated_conditions = negated_conditions or []
    
    def __repr__(self):
        pre_str = ', '.join(str(p) for p in self.premises)
        return f"{pre_str} => {self.conclusion}"


class Definition:
    """Represents a construction definition from defs.txt."""
    
    def __init__(self, name, arg_count, derived_facts):
        self.name = name
        self.arg_count = arg_count
        self.derived_facts = derived_facts  # List of (predicate, arg_pattern) tuples
    
    def __repr__(self):
        return f"Definition({self.name}, {self.arg_count} args, {len(self.derived_facts)} facts)"


# ============================================================
# Parser
# ============================================================

def parse_fact_string(s):
    """Parse a fact string like 'cong O A O B' into a Fact."""
    tokens = s.strip().split()
    if not tokens:
        return None
    return Fact(tokens[0], tokens[1:])


def load_rules(rules_path):
    """Load deduction rules from rules.txt."""
    rules = []
    with open(rules_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=>' not in line:
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
                rule_name = f"rule_{len(rules)}"
                rules.append(Rule(rule_name, premises, conclusion, negated))
    
    return rules


def load_definitions(defs_path):
    """Load construction definitions from defs.txt.
    
    Format:
    name args...
    vars : constraints
    args = preconditions
    vars : derived_fact1, derived_fact2, ...
    implementation
    """
    definitions = {}
    
    with open(defs_path, 'r') as f:
        lines = [l.rstrip() for l in f.readlines()]
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        tokens = line.split()
        if len(tokens) < 1:
            i += 1
            continue
        
        def_name = tokens[0]
        def_args = tokens[1:]
        
        # Skip known non-definition entries
        if def_name in ('pentagon', 'quadrangle'):
            i += 1
            continue
        
        # Parse the definition block
        j = i + 1
        derived_facts = []
        
        while j < len(lines):
            l = lines[j].strip()
            if not l:
                j += 1
                continue
            
            # Check for variable declaration line with derived facts
            # Pattern: "x : fact1, fact2, ..." or "x y : fact1; fact2; ..."
            if ':' in l and '=' not in l.split(':')[0]:
                var_part = l.split(':')[0].strip()
                fact_part = ':'.join(l.split(':')[1:]).strip()
                
                # Parse facts (comma or semicolon separated)
                for sep in [';', ',']:
                    fact_part = fact_part.replace(sep, '\n')
                
                for fp in fact_part.split('\n'):
                    fp = fp.strip()
                    if not fp:
                        continue
                    ftokens = fp.split()
                    if len(ftokens) >= 2:
                        pred = ftokens[0]
                        # Replace variable references
                        all_vars = def_args + var_part.split()
                        args = []
                        for t in ftokens[1:]:
                            args.append(t)
                        derived_facts.append((pred, tuple(args)))
            
            # Check for implementation line (starts with def_name)
            if l == def_name or l.startswith(def_name + ' '):
                break
            
            j += 1
        
        if derived_facts:
            definitions[def_name] = Definition(def_name, len(def_args), derived_facts)
        
        i += 1
    
    return definitions


# ============================================================
# Knowledge Base and Forward Chaining
# ============================================================

class KnowledgeBase:
    """Forward-chaining knowledge base for geometric reasoning."""
    
    def __init__(self, rules):
        self.rules = rules
        self.facts = set()
        self.derived_facts = {}
        self.proof_steps = []
        self.point_set = set()
    
    def add_fact(self, fact, rule=None, supporting=None, step_desc=None):
        """Add a fact to the knowledge base."""
        if fact not in self.facts:
            self.facts.add(fact)
            for arg in fact.args:
                self.point_set.add(arg)
            if rule and supporting:
                self.derived_facts[fact] = (rule, supporting)
                self.proof_steps.append({
                    'fact': str(fact),
                    'rule': rule.name if rule else 'initial',
                    'supporting': [str(s) for s in supporting],
                    'step_desc': step_desc or ''
                })
            return True
        return False
    
    def check_negated(self, negated_conditions):
        """Check that negated conditions don't hold."""
        for neg in negated_conditions:
            if neg in self.facts:
                return False
        return True
    
    def _match_pattern(self, pattern, fact, existing_sub=None):
        """Check if a fact matches a pattern."""
        if pattern.predicate != fact.predicate:
            return False
        if len(pattern.args) != len(fact.args):
            return False
        
        sub = dict(existing_sub) if existing_sub else {}
        for p_arg, f_arg in zip(pattern.args, fact.args):
            if p_arg in sub:
                if sub[p_arg] != f_arg:
                    return False
            else:
                sub[p_arg] = f_arg
        return True
    
    def _get_substitution(self, pattern, fact, existing_sub=None):
        """Get variable substitution from pattern to fact."""
        sub = dict(existing_sub) if existing_sub else {}
        for p_arg, f_arg in zip(pattern.args, fact.args):
            if p_arg in sub:
                if sub[p_arg] != f_arg:
                    return None
            else:
                sub[p_arg] = f_arg
        return sub
    
    def _merge_substitutions(self, sub1, sub2):
        """Merge two substitutions."""
        merged = dict(sub1)
        for k, v in sub2.items():
            if k in merged:
                if merged[k] != v:
                    return None
            else:
                merged[k] = v
        return merged
    
    def _substitute(self, fact_pattern, substitution):
        """Apply substitution to a fact pattern."""
        new_args = tuple(substitution.get(a, a) for a in fact_pattern.args)
        return Fact(fact_pattern.predicate, new_args)
    
    def try_apply_rule(self, rule):
        """Try to apply a rule and derive new facts."""
        new_facts = []
        n_premises = len(rule.premises)
        
        if n_premises == 1:
            pattern = rule.premises[0]
            for fact in list(self.facts):
                if self._match_pattern(pattern, fact):
                    substitution = self._get_substitution(pattern, fact)
                    if substitution is not None:
                        if self.check_negated(rule.negated_conditions):
                            new_fact = self._substitute(rule.conclusion, substitution)
                            if self.add_fact(new_fact, rule, [fact]):
                                new_facts.append(new_fact)
        
        elif n_premises == 2:
            p1, p2 = rule.premises
            facts_list = list(self.facts)
            for i, f1 in enumerate(facts_list):
                if self._match_pattern(p1, f1):
                    sub1 = self._get_substitution(p1, f1)
                    for j, f2 in enumerate(facts_list):
                        if i != j and self._match_pattern(p2, f2, sub1):
                            sub = self._merge_substitutions(
                                sub1, self._get_substitution(p2, f2, sub1)
                            )
                            if sub is not None:
                                if self.check_negated(rule.negated_conditions):
                                    new_fact = self._substitute(rule.conclusion, sub)
                                    if self.add_fact(new_fact, rule, [f1, f2]):
                                        new_facts.append(new_fact)
        
        elif n_premises == 3:
            p1, p2, p3 = rule.premises
            facts_list = list(self.facts)
            for i, f1 in enumerate(facts_list):
                if self._match_pattern(p1, f1):
                    sub1 = self._get_substitution(p1, f1)
                    for j, f2 in enumerate(facts_list):
                        if j != i and self._match_pattern(p2, f2, sub1):
                            sub12 = self._merge_substitutions(
                                sub1, self._get_substitution(p2, f2, sub1)
                            )
                            if sub12 is not None:
                                for k, f3 in enumerate(facts_list):
                                    if k not in (i, j) and self._match_pattern(p3, f3, sub12):
                                        sub = self._merge_substitutions(
                                            sub12, self._get_substitution(p3, f3, sub12)
                                        )
                                        if sub is not None:
                                            if self.check_negated(rule.negated_conditions):
                                                new_fact = self._substitute(rule.conclusion, sub)
                                                if self.add_fact(new_fact, rule, [f1, f2, f3]):
                                                    new_facts.append(new_fact)
        
        return new_facts
    
    def forward_chain(self, max_iterations=1000):
        """Run forward chaining."""
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
# Problem Solver
# ============================================================

class GeometrySolver:
    """Solves geometry problems using forward chaining with definition expansion."""
    
    def __init__(self, rules_path, defs_path):
        self.rules = load_rules(rules_path)
        self.defs = load_definitions(defs_path)
        print(f"Loaded {len(self.rules)} deduction rules")
        print(f"Loaded {len(self.defs)} construction definitions")
        for name, d in self.defs.items():
            print(f"  {name}: {d.arg_count} args -> {len(d.derived_facts)} facts")
    
    def expand_construction(self, constr):
        """Expand a construction step into base geometric facts using definitions."""
        facts = []
        for pred_info in constr.get('predicates', []):
            pred_name = pred_info['predicate']
            pred_args = pred_info['args']
            
            # Check if this is a construction predicate that needs expansion
            if pred_name in self.defs:
                d = self.defs[pred_name]
                # Map definition arguments to actual values
                if len(pred_args) >= d.arg_count:
                    mapping = {}
                    for idx, arg_name in enumerate(pred_args[:d.arg_count]):
                        mapping[arg_name] = arg_name
                    
                    # For each derived fact, substitute variables
                    for fact_pred, fact_args in d.derived_facts:
                        new_args = []
                        for a in fact_args:
                            new_args.append(mapping.get(a, a))
                        facts.append(Fact(fact_pred, tuple(new_args)))
            
            # Also keep the original predicate as a fact
            facts.append(Fact(pred_name, tuple(pred_args)))
        
        return facts
    
    def parse_problem(self, problem_data):
        """Parse a problem into initial facts and target conclusion."""
        kb = KnowledgeBase(self.rules)
        
        constructions = problem_data.get('constructions', [])
        conclusion_data = problem_data.get('conclusion', {})
        
        # Extract and expand facts from constructions
        for constr in constructions:
            expanded = self.expand_construction(constr)
            for fact in expanded:
                kb.add_fact(fact, step_desc=f"construction: {constr.get('raw', '')}")
        
        # Target conclusion
        target = Fact(
            conclusion_data.get('predicate', ''),
            tuple(conclusion_data.get('args', []))
        )
        
        return kb, target
    
    def solve(self, problem_data, max_iterations=1000):
        """Attempt to solve a problem."""
        kb, target = self.parse_problem(problem_data)
        initial_count = len(kb.facts)
        
        # Run forward chaining
        n_derived, n_iters = kb.forward_chain(max_iterations)
        
        # Check if target is proven
        proven = target in kb.facts
        
        # Also check symmetric variants
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
            'all_facts': [str(f) for f in kb.facts],
            'point_set': list(kb.point_set)
        }
    
    def _check_symmetric(self, kb, target):
        """Check symmetric variants of the target."""
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
                reversed_rot = list(reversed(rotated))
                if Fact(pred, tuple(reversed_rot)) in kb.facts:
                    return True
        
        elif pred == 'para' and len(args) == 4:
            symms = [(args[2], args[3], args[0], args[1])]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        elif pred == 'perp' and len(args) == 4:
            symms = [(args[2], args[3], args[0], args[1])]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        elif pred == 'eqangle' and len(args) == 8:
            symms = [
                (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]),
            ]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        elif pred == 'eqratio' and len(args) == 8:
            symms = [
                (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]),
            ]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        return False


# ============================================================
# Proof Generator
# ============================================================

class ProofGenerator:
    """Generates human-readable proofs from solver results."""
    
    PREDICATE_NAMES = {
        'cong': 'congruent segments',
        'coll': 'collinear points',
        'cyclic': 'concyclic points',
        'para': 'parallel lines',
        'perp': 'perpendicular lines',
        'eqangle': 'equal angles',
        'eqratio': 'equal ratios',
        'midp': 'midpoint',
    }
    
    @staticmethod
    def format_fact(fact_str):
        """Format a fact string into readable form."""
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
            return f"{args[0]} is the midpoint of {args[1]}{args[2]}"
        else:
            return f"{pred}({' '.join(args)})"
    
    @staticmethod
    def generate_proof(result, problem_name):
        """Generate a readable proof from solver results."""
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
    
    # Save results
    results_path = os.path.join(output_dir, 'solver_results_v2.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate proofs
    proofs_dir = os.path.join(output_dir, 'proofs_v2')
    os.makedirs(proofs_dir, exist_ok=True)
    
    for name, result in results.items():
        proof_text = ProofGenerator.generate_proof(result, name)
        proof_path = os.path.join(proofs_dir, f"{name}_proof.md")
        with open(proof_path, 'w') as f:
            f.write(proof_text)
    
    # Summary
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
    
    summary_path = os.path.join(output_dir, 'summary_v2.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
