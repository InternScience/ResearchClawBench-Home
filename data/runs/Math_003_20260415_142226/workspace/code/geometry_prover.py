#!/usr/bin/env python3
"""
Geometry Theorem Prover for IMO-level Euclidean Geometry Problems.

This implements a neuro-symbolic forward-chaining theorem prover that:
1. Parses formal problem statements into geometric predicates
2. Applies deduction rules to derive new facts
3. Searches for proof paths from premises to conclusions
4. Generates human-readable, machine-verifiable proofs

The system uses the predicate language and deduction rules defined in
the data/ directory (defs.txt and rules.txt).
"""

import json
import re
import os
import sys
from collections import defaultdict, deque
from itertools import permutations, combinations
from copy import deepcopy
import hashlib

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
    
    def canonical(self):
        """Return a canonical string representation."""
        return f"{self.predicate} {' '.join(sorted(self.args))}"
    
    def to_dict(self):
        return {'predicate': self.predicate, 'args': list(self.args)}


class Rule:
    """Represents a deduction rule."""
    
    def __init__(self, name, premises, conclusion, negated_conditions=None):
        self.name = name
        self.premises = premises  # List of Fact patterns
        self.conclusion = conclusion  # Fact pattern
        self.negated_conditions = negated_conditions or []  # Facts that must NOT hold
    
    def __repr__(self):
        pre_str = ', '.join(str(p) for p in self.premises)
        return f"{pre_str} => {self.conclusion}"


# ============================================================
# Parser for defs.txt and rules.txt
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
            
            # Split on '=>'
            if '=>' not in line:
                continue
            
            parts = line.split('=>')
            premise_str = parts[0].strip()
            conclusion_str = parts[1].strip()
            
            # Parse premises (comma-separated)
            premises = []
            negated = []
            for p in premise_str.split(','):
                p = p.strip()
                if p.startswith('n'):
                    # Negated condition like ncoll, nperp, npara
                    negated.append(parse_fact_string(p))
                else:
                    pf = parse_fact_string(p)
                    if pf:
                        premises.append(pf)
            
            # Parse conclusion
            conclusion = parse_fact_string(conclusion_str)
            if conclusion and premises:
                rule_name = f"rule_{len(rules)}"
                rules.append(Rule(rule_name, premises, conclusion, negated))
    
    return rules


def load_definitions(defs_path):
    """Load construction definitions from defs.txt."""
    definitions = {}
    current_def = None
    
    with open(defs_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check for definition header: "name arg1 arg2 ..."
        # followed by a line with variable declarations
        tokens = line.split()
        
        # Look for the structure:
        # name args
        # vars : ...
        # constraints = ...
        # vars : conditions
        # implementation
        
        # Try to match definition pattern
        if i + 1 < len(lines) and ':' in lines[i+1]:
            def_name = tokens[0]
            def_args = tokens[1:]
            
            # Collect definition info
            j = i + 1
            var_decls = []
            constraints = []
            conditions = []
            impl_lines = []
            
            while j < len(lines):
                l = lines[j].strip()
                if not l:
                    j += 1
                    continue
                
                if ':' in l and '=' not in l.split(':')[0]:
                    # Variable declaration line
                    var_part = l.split(':')[0].strip()
                    cond_part = ':'.join(l.split(':')[1:]).strip()
                    var_decls.append((var_part, cond_part))
                elif l == '=' or (l.startswith('=') and len(l) <= 2):
                    # Constraint separator
                    j += 1
                    continue
                elif l.startswith(def_name + ' ') or l == def_name:
                    # Implementation line
                    impl_lines.append(l)
                    break
                else:
                    # Could be conditions after variable declaration
                    if var_decls and not l.startswith('='):
                        conditions.append(l)
                
                j += 1
            
            definitions[def_name] = {
                'name': def_name,
                'args': def_args,
                'var_decls': var_decls,
                'conditions': conditions,
                'implementation': impl_lines
            }
        
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
        self.derived_facts = {}  # fact -> (rule, supporting_facts)
        self.proof_steps = []
        self.point_set = set()
    
    def add_fact(self, fact, rule=None, supporting=None):
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
                    'supporting': [str(s) for s in supporting]
                })
            return True
        return False
    
    def check_negated(self, negated_conditions):
        """Check that negated conditions don't hold."""
        for neg in negated_conditions:
            if neg in self.facts:
                return False
        return True
    
    def try_apply_rule(self, rule):
        """Try to apply a rule and derive new facts. Returns list of new facts."""
        new_facts = []
        
        # For each rule, try all possible substitutions
        # This is a simplified unification approach
        if len(rule.premises) == 1:
            # Single-premise rules: try matching against all facts
            pattern = rule.premises[0]
            for fact in self.facts:
                if self._match_pattern(pattern, fact):
                    substitution = self._get_substitution(pattern, fact)
                    if substitution is not None:
                        if self.check_negated(rule.negated_conditions):
                            new_fact = self._substitute(rule.conclusion, substitution)
                            if self.add_fact(new_fact, rule, [fact]):
                                new_facts.append(new_fact)
        
        elif len(rule.premises) == 2:
            # Two-premise rules: find pairs of facts that match
            p1, p2 = rule.premises
            facts_list = list(self.facts)
            for i, f1 in enumerate(facts_list):
                if self._match_pattern(p1, f1):
                    sub1 = self._get_substitution(p1, f1)
                    for j, f2 in enumerate(facts_list):
                        if i != j and self._match_pattern(p2, f2, sub1):
                            sub2 = self._merge_substitutions(
                                sub1, self._get_substitution(p2, f2, sub1)
                            )
                            if sub2 is not None:
                                if self.check_negated(rule.negated_conditions):
                                    new_fact = self._substitute(rule.conclusion, sub2)
                                    if self.add_fact(new_fact, rule, [f1, f2]):
                                        new_facts.append(new_fact)
        
        elif len(rule.premises) == 3:
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
    
    def _match_pattern(self, pattern, fact, existing_sub=None):
        """Check if a fact matches a pattern (with optional existing substitution)."""
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
        """Merge two substitutions, returning None if incompatible."""
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
    
    def forward_chain(self, max_iterations=500):
        """Run forward chaining until no new facts are derived or max iterations reached."""
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
    """Solves geometry problems using forward chaining."""
    
    def __init__(self, rules_path, defs_path):
        self.rules = load_rules(rules_path)
        self.defs = load_definitions(defs_path)
        print(f"Loaded {len(self.rules)} deduction rules")
        print(f"Loaded {len(self.defs)} construction definitions")
    
    def parse_problem(self, problem_data):
        """Parse a problem into initial facts and target conclusion."""
        kb = KnowledgeBase(self.rules)
        
        constructions = problem_data.get('constructions', [])
        conclusion_data = problem_data.get('conclusion', {})
        
        # Extract initial facts from constructions
        for constr in constructions:
            for pred in constr.get('predicates', []):
                fact = Fact(pred['predicate'], tuple(pred['args']))
                kb.add_fact(fact)
        
        # Target conclusion
        target = Fact(
            conclusion_data.get('predicate', ''),
            tuple(conclusion_data.get('args', []))
        )
        
        return kb, target
    
    def solve(self, problem_data, max_iterations=500):
        """Attempt to solve a problem."""
        kb, target = self.parse_problem(problem_data)
        
        # Run forward chaining
        n_derived, n_iters = kb.forward_chain(max_iterations)
        
        # Check if target is proven
        proven = target in kb.facts
        
        # Also check symmetric variants for certain predicates
        if not proven:
            proven = self._check_symmetric(kb, target)
        
        return {
            'proven': proven,
            'target': str(target),
            'initial_facts': len(kb.facts) - n_derived,
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
        
        if pred == 'cong':
            # cong A B C D is same as cong C D A B, cong B A D C, etc.
            symms = [
                (args[2], args[3], args[0], args[1]),  # swap pairs
                (args[1], args[0], args[3], args[2]),  # reverse within pairs
                (args[3], args[2], args[1], args[0]),  # both
            ]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        elif pred == 'coll':
            # coll A B C == coll A C B == coll B A C == coll B C A == coll C A B == coll C B A
            for perm in permutations(args):
                if Fact(pred, perm) in kb.facts:
                    return True
        
        elif pred == 'cyclic':
            # cyclic A B C D has rotational symmetry
            for i in range(len(args)):
                rotated = args[i:] + args[:i]
                if Fact(pred, tuple(rotated)) in kb.facts:
                    return True
                reversed_rot = list(reversed(rotated))
                if Fact(pred, tuple(reversed_rot)) in kb.facts:
                    return True
        
        elif pred == 'para':
            symms = [
                (args[2], args[3], args[0], args[1]),
            ]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        elif pred == 'perp':
            symms = [
                (args[2], args[3], args[0], args[1]),
            ]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        elif pred == 'eqangle':
            # eqangle A B C D E F G H - swap the two angle pairs
            symms = [
                (args[4], args[5], args[6], args[7], args[0], args[1], args[2], args[3]),
            ]
            for s in symms:
                if Fact(pred, s) in kb.facts:
                    return True
        
        elif pred == 'eqratio':
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
        
        name = ProofGenerator.PREDICATE_NAMES.get(pred, pred)
        
        if pred == 'cong':
            return f"Segment {args[0]}{args[1]} ≅ Segment {args[2]}{args[3]}"
        elif pred == 'coll':
            return f"Points {', '.join(args)} are collinear"
        elif pred == 'cyclic':
            return f"Points {', '.join(args)} are concyclic"
        elif pred == 'para':
            return f"Line {args[0]}{args[1]} ∥ Line {args[2]}{args[3]}"
        elif pred == 'perp':
            return f"Line {args[0]}{args[1]} ⊥ Line {args[2]}{args[3]}"
        elif pred == 'eqangle':
            return f"∠({args[0]}{args[1]}, {args[2]}{args[3]}) = ∠({args[4]}{args[5]}, {args[6]}{args[7]})"
        elif pred == 'eqratio':
            return f"|{args[0]}{args[1]}|/|{args[2]}{args[3]}| = |{args[4]}{args[5]}|/|{args[6]}{args[7]}|"
        elif pred == 'midp':
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
            for i, step in enumerate(result['proof_steps'][:50]):  # Limit display
                lines.append(f"{i+1}. **{ProofGenerator.format_fact(step['fact'])}**")
                lines.append(f"   - Rule: `{step['rule']}`")
                lines.append(f"   - From: {', '.join(ProofGenerator.format_fact(s) for s in step['supporting'])}")
                lines.append(f"")
        
        return '\n'.join(lines)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    workspace = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(workspace, 'data')
    output_dir = os.path.join(workspace, 'outputs')
    
    rules_path = os.path.join(data_dir, 'rules.txt')
    defs_path = os.path.join(data_dir, 'defs.txt')
    problems_path = os.path.join(output_dir, 'parsed_problems.json')
    
    # Load parsed problems
    with open(problems_path, 'r') as f:
        problems = json.load(f)
    
    # Initialize solver
    solver = GeometrySolver(rules_path, defs_path)
    
    # Solve each problem
    results = {}
    for name, prob_data in problems.items():
        print(f"\n{'='*60}")
        print(f"Solving: {name}")
        print(f"{'='*60}")
        
        result = solver.solve(prob_data, max_iterations=500)
        results[name] = result
        
        status = "✅ PROVEN" if result['proven'] else "❌ Not proven"
        print(f"  Result: {status}")
        print(f"  Facts: {result['total_facts']} ({result['derived_facts']} derived)")
        print(f"  Iterations: {result['iterations']}")
    
    # Save results
    results_path = os.path.join(output_dir, 'solver_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate individual proofs
    proofs_dir = os.path.join(output_dir, 'proofs')
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
    
    # Save summary
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
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
