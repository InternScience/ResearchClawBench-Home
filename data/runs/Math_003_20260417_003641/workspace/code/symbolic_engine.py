"""
symbolic_engine.py - Symbolic deduction engine for geometry theorem proving
Implements forward-chaining deduction using the provided rules and definitions.
"""
import json
import re
import itertools
from collections import defaultdict, Counter
import time
import random

class GeometricFact:
    """Represents a geometric fact/predicate."""
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = tuple(args)
    
    def __repr__(self):
        return f"{self.predicate}({', '.join(self.args)})"
    
    def __eq__(self, other):
        return self.predicate == other.predicate and self.args == other.args
    
    def __hash__(self):
        return hash((self.predicate, self.args))

class DeductionRule:
    """Represents a deduction rule with premises and conclusion."""
    def __init__(self, premises, conclusion, raw):
        self.premises = premises  # list of (predicate, [var_names])
        self.conclusion = conclusion  # (predicate, [var_names])
        self.raw = raw
        self.variables = self._extract_variables()
    
    def _extract_variables(self):
        """Extract all unique variables from the rule."""
        vars_set = set()
        for pred, args in self.premises:
            for arg in args:
                vars_set.add(arg)
        pred, args = self.conclusion
        for arg in args:
            vars_set.add(arg)
        return vars_set
    
    def __repr__(self):
        premises_str = ', '.join(f"{p}({', '.join(a)})" for p, a in self.premises)
        conc_str = f"{self.conclusion[0]}({', '.join(self.conclusion[1])})"
        return f"{premises_str} => {conc_str}"

def parse_predicate(s):
    """Parse a predicate string like 'cong A B C D' into (predicate, [args])."""
    tokens = s.strip().split()
    if not tokens:
        return None
    return (tokens[0], tokens[1:])

def parse_rules(filepath):
    """Parse rules from file."""
    rules = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or '=>' not in line:
                continue
            parts = line.split('=>')
            premise_strs = [p.strip() for p in parts[0].split(',')]
            conclusion_str = parts[1].strip()
            
            premises = []
            for ps in premise_strs:
                parsed = parse_predicate(ps)
                if parsed:
                    premises.append(parsed)
            
            conclusion = parse_predicate(conclusion_str)
            if conclusion:
                rules.append(DeductionRule(premises, conclusion, line))
    
    return rules

class Problem:
    """Represents a geometry problem."""
    def __init__(self, name, constructions_raw, goal_raw):
        self.name = name
        self.constructions_raw = constructions_raw
        self.goal_raw = goal_raw
        self.points = set()
        self.initial_facts = set()
        self.goal = None
        self._parse()
    
    def _parse(self):
        """Parse the problem into points and initial facts."""
        # Parse goal
        goal_tokens = self.goal_raw.strip().split()
        self.goal = GeometricFact(goal_tokens[0], goal_tokens[1:])
        
        # Parse constructions to extract points and basic facts
        steps = self.constructions_raw.split(';')
        for step in steps:
            step = step.strip()
            if not step:
                continue
            if '=' in step:
                eq_parts = step.split('=', 1)
                lhs = eq_parts[0].strip()
                rhs = eq_parts[1].strip()
                
                # Extract point names from LHS
                for v in lhs.split():
                    clean_v = v.split('@')[0]
                    self.points.add(clean_v)
                
                # Parse RHS definitions
                defs = rhs.split(',')
                for d in defs:
                    d = d.strip()
                    tokens = d.split()
                    if not tokens:
                        continue
                    func = tokens[0]
                    args = [t.split('@')[0] for t in tokens[1:]]
                    
                    # Add points from args
                    for a in args:
                        if a and len(a) <= 3:
                            self.points.add(a)
                    
                    # Generate facts from construction definitions
                    self._add_construction_facts(func, args)
            else:
                for v in step.split():
                    clean_v = v.split('@')[0]
                    self.points.add(clean_v)
    
    def _add_construction_facts(self, func, args):
        """Add geometric facts implied by a construction."""
        if func == 'triangle':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('ncoll', (args[0], args[1], args[2])))
        elif func == 'on_line':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('coll', (args[0], args[1], args[2])))
        elif func == 'on_circle':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('cong', (args[1], args[0], args[1], args[2])))
        elif func == 'midpoint' or func == 'midp':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('midp', (args[0], args[1], args[2])))
                self.initial_facts.add(GeometricFact('coll', (args[0], args[1], args[2])))
                self.initial_facts.add(GeometricFact('cong', (args[0], args[1], args[0], args[2])))
        elif func == 'circle':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('cong', (args[0], args[1], args[0], args[2])))
                self.initial_facts.add(GeometricFact('cong', (args[0], args[2], args[0], args[3])))
                self.initial_facts.add(GeometricFact('cong', (args[0], args[1], args[0], args[3])))
                self.initial_facts.add(GeometricFact('cyclic', (args[1], args[2], args[3])))
        elif func == 'foot':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('perp', (args[0], args[1], args[2], args[3])))
                self.initial_facts.add(GeometricFact('coll', (args[0], args[2], args[3])))
        elif func == 'on_tline':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('perp', (args[0], args[1], args[2], args[3])))
        elif func == 'on_pline':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('para', (args[0], args[1], args[2], args[3])))
        elif func == 'on_bline':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('cong', (args[0], args[1], args[0], args[2])))
        elif func == 'orthocenter':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('perp', (args[0], args[1], args[2], args[3])))
                self.initial_facts.add(GeometricFact('perp', (args[0], args[2], args[3], args[1])))
                self.initial_facts.add(GeometricFact('perp', (args[0], args[3], args[1], args[2])))
        elif func == 'incenter' or func == 'incenter2':
            pass  # Complex - angle bisector facts
        elif func == 'reflect':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('perp', (args[2], args[3], args[1], args[0])))
                self.initial_facts.add(GeometricFact('cong', (args[2], args[1], args[2], args[0])))
                self.initial_facts.add(GeometricFact('cong', (args[3], args[1], args[3], args[0])))
        elif func == 'mirror':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('coll', (args[0], args[1], args[2])))
                self.initial_facts.add(GeometricFact('cong', (args[2], args[1], args[2], args[0])))
        elif func == 'parallelogram':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('para', (args[0], args[1], args[2], args[3])))
                self.initial_facts.add(GeometricFact('para', (args[0], args[3], args[1], args[2])))
                self.initial_facts.add(GeometricFact('cong', (args[0], args[1], args[2], args[3])))
                self.initial_facts.add(GeometricFact('cong', (args[0], args[3], args[1], args[2])))
        elif func == 'on_dia':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('perp', (args[0], args[1], args[0], args[2])))
        elif func == 'angle_bisector':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('eqangle', 
                    (args[1], args[0], args[1], args[3], args[1], args[3], args[1], args[2])))
        elif func == 'eqdistance':
            if len(args) >= 4:
                self.initial_facts.add(GeometricFact('cong', (args[0], args[1], args[2], args[3])))
        elif func == 'on_aline':
            if len(args) >= 6:
                self.initial_facts.add(GeometricFact('eqangle',
                    (args[1], args[0], args[1], args[2], args[3], args[4], args[3], args[5])))
        elif func == 'r_triangle':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('perp', (args[0], args[1], args[0], args[2])))
                self.initial_facts.add(GeometricFact('ncoll', (args[0], args[1], args[2])))
        elif func == 'iso_triangle':
            if len(args) >= 3:
                self.initial_facts.add(GeometricFact('cong', (args[0], args[1], args[0], args[2])))
                self.initial_facts.add(GeometricFact('ncoll', (args[0], args[1], args[2])))
        elif func == 'segment':
            if len(args) >= 2:
                self.initial_facts.add(GeometricFact('diff', (args[0], args[1])))

class SymbolicProver:
    """Forward-chaining symbolic deduction engine."""
    
    def __init__(self, rules):
        self.rules = rules
        self.stats = defaultdict(int)
    
    def prove(self, problem, max_iterations=50, max_facts=5000, timeout=30):
        """Attempt to prove the goal using forward chaining."""
        start_time = time.time()
        facts = set(problem.initial_facts)
        goal = problem.goal
        
        # Check if goal is already in initial facts
        if self._check_goal(facts, goal):
            return {
                'solved': True,
                'iterations': 0,
                'facts_derived': len(facts),
                'time': time.time() - start_time,
                'method': 'initial_facts'
            }
        
        iteration_facts = []
        for iteration in range(max_iterations):
            if time.time() - start_time > timeout:
                break
            
            new_facts = set()
            
            for rule in self.rules:
                # Try to apply this rule
                derived = self._apply_rule(rule, facts, problem.points)
                new_facts.update(derived)
                
                if len(new_facts) + len(facts) > max_facts:
                    break
            
            # Remove already known facts
            new_facts -= facts
            
            if not new_facts:
                break
            
            facts.update(new_facts)
            iteration_facts.append(len(new_facts))
            
            # Check goal
            if self._check_goal(facts, goal):
                return {
                    'solved': True,
                    'iterations': iteration + 1,
                    'facts_derived': len(facts),
                    'new_facts_per_iteration': iteration_facts,
                    'time': time.time() - start_time,
                    'method': 'forward_chaining'
                }
        
        return {
            'solved': False,
            'iterations': len(iteration_facts),
            'facts_derived': len(facts),
            'new_facts_per_iteration': iteration_facts,
            'time': time.time() - start_time,
            'method': 'forward_chaining'
        }
    
    def _check_goal(self, facts, goal):
        """Check if the goal is satisfied by current facts."""
        # Direct match
        if goal in facts:
            return True
        
        # Check with argument permutations for symmetric predicates
        if goal.predicate == 'cong':
            # cong A B C D means AB = CD
            # Also check cong C D A B
            if len(goal.args) == 4:
                a, b, c, d = goal.args
                variants = [
                    GeometricFact('cong', (a, b, c, d)),
                    GeometricFact('cong', (b, a, c, d)),
                    GeometricFact('cong', (a, b, d, c)),
                    GeometricFact('cong', (b, a, d, c)),
                    GeometricFact('cong', (c, d, a, b)),
                    GeometricFact('cong', (d, c, a, b)),
                    GeometricFact('cong', (c, d, b, a)),
                    GeometricFact('cong', (d, c, b, a)),
                ]
                for v in variants:
                    if v in facts:
                        return True
        
        elif goal.predicate == 'coll':
            # Collinearity is symmetric in all permutations
            if len(goal.args) >= 3:
                for perm in itertools.permutations(goal.args):
                    if GeometricFact('coll', perm) in facts:
                        return True
        
        elif goal.predicate == 'cyclic':
            # Cyclic is invariant under cyclic permutations and reversal
            if len(goal.args) >= 4:
                args = list(goal.args)
                for i in range(len(args)):
                    rotated = args[i:] + args[:i]
                    if GeometricFact('cyclic', tuple(rotated)) in facts:
                        return True
                    if GeometricFact('cyclic', tuple(reversed(rotated))) in facts:
                        return True
        
        elif goal.predicate == 'para':
            if len(goal.args) == 4:
                a, b, c, d = goal.args
                variants = [
                    GeometricFact('para', (a, b, c, d)),
                    GeometricFact('para', (b, a, c, d)),
                    GeometricFact('para', (a, b, d, c)),
                    GeometricFact('para', (b, a, d, c)),
                    GeometricFact('para', (c, d, a, b)),
                    GeometricFact('para', (d, c, a, b)),
                    GeometricFact('para', (c, d, b, a)),
                    GeometricFact('para', (d, c, b, a)),
                ]
                for v in variants:
                    if v in facts:
                        return True
        
        elif goal.predicate == 'perp':
            if len(goal.args) == 4:
                a, b, c, d = goal.args
                variants = [
                    GeometricFact('perp', (a, b, c, d)),
                    GeometricFact('perp', (b, a, c, d)),
                    GeometricFact('perp', (a, b, d, c)),
                    GeometricFact('perp', (b, a, d, c)),
                    GeometricFact('perp', (c, d, a, b)),
                    GeometricFact('perp', (d, c, a, b)),
                    GeometricFact('perp', (c, d, b, a)),
                    GeometricFact('perp', (d, c, b, a)),
                ]
                for v in variants:
                    if v in facts:
                        return True
        
        return False
    
    def _apply_rule(self, rule, facts, points):
        """Try to apply a rule to derive new facts."""
        new_facts = set()
        
        # For efficiency, only try rules with small premise sets
        if len(rule.premises) > 4:
            return new_facts
        
        # Get facts organized by predicate
        facts_by_pred = defaultdict(list)
        for f in facts:
            facts_by_pred[f.predicate].append(f)
        
        # Check if all premise predicates exist
        for pred, args in rule.premises:
            if pred not in facts_by_pred and pred not in ('ncoll', 'diff', 'npara', 'sameside'):
                return new_facts
        
        # Try to find matching substitutions
        # This is a simplified version - full unification would be more complex
        if len(rule.premises) <= 2:
            substitutions = self._find_substitutions(rule, facts_by_pred, points)
            for subst in substitutions[:100]:  # Limit to prevent explosion
                # Apply substitution to conclusion
                pred, args = rule.conclusion
                new_args = tuple(subst.get(a, a) for a in args)
                new_fact = GeometricFact(pred, new_args)
                new_facts.add(new_fact)
        
        return new_facts
    
    def _find_substitutions(self, rule, facts_by_pred, points, max_substs=100):
        """Find variable substitutions that satisfy all premises."""
        substitutions = []
        
        if not rule.premises:
            return substitutions
        
        # Start with first premise
        first_pred, first_args = rule.premises[0]
        
        # Handle negative predicates
        if first_pred in ('ncoll', 'diff', 'npara', 'sameside'):
            return substitutions
        
        if first_pred not in facts_by_pred:
            return substitutions
        
        for fact in facts_by_pred[first_pred]:
            if len(fact.args) != len(first_args):
                continue
            
            # Create initial substitution
            subst = {}
            valid = True
            for var, val in zip(first_args, fact.args):
                if var in subst:
                    if subst[var] != val:
                        valid = False
                        break
                else:
                    subst[var] = val
            
            if not valid:
                continue
            
            # Check remaining premises
            all_satisfied = True
            for pred, args in rule.premises[1:]:
                if pred in ('ncoll', 'diff', 'npara', 'sameside'):
                    continue  # Skip negative/auxiliary predicates
                
                # Substitute variables
                subst_args = tuple(subst.get(a, a) for a in args)
                
                # Check if any variable is unbound
                has_unbound = any(a.isupper() and len(a) == 1 for a in subst_args)
                
                if has_unbound:
                    # Try to find matching facts
                    found = False
                    if pred in facts_by_pred:
                        for f in facts_by_pred[pred]:
                            if len(f.args) != len(subst_args):
                                continue
                            match = True
                            temp_subst = dict(subst)
                            for sa, fa in zip(subst_args, f.args):
                                if sa.isupper() and len(sa) == 1:
                                    if sa in temp_subst:
                                        if temp_subst[sa] != fa:
                                            match = False
                                            break
                                    else:
                                        temp_subst[sa] = fa
                                elif sa != fa:
                                    match = False
                                    break
                            if match:
                                subst = temp_subst
                                found = True
                                break
                    if not found:
                        all_satisfied = False
                        break
                else:
                    target = GeometricFact(pred, subst_args)
                    if target not in set(facts_by_pred.get(pred, [])):
                        all_satisfied = False
                        break
            
            if all_satisfied:
                substitutions.append(subst)
                if len(substitutions) >= max_substs:
                    break
        
        return substitutions

def parse_problems(filepath):
    """Parse problems from the benchmark file."""
    problems = {}
    with open(filepath) as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            name = line
            i += 1
            if i < len(lines):
                prob_line = lines[i].strip()
                if '?' in prob_line:
                    parts = prob_line.split('?')
                    problems[name] = Problem(name, parts[0], parts[1])
        i += 1
    
    return problems

def main():
    # Parse rules and problems
    rules = parse_rules('data/rules.txt')
    problems = parse_problems('data/imo_ag_30.txt')
    
    print(f"Loaded {len(rules)} rules and {len(problems)} problems")
    print()
    
    # Create prover
    prover = SymbolicProver(rules)
    
    # Run on all problems
    results = {}
    solved_count = 0
    
    for name, problem in problems.items():
        print(f"Proving {name}...")
        print(f"  Points: {len(problem.points)}, Initial facts: {len(problem.initial_facts)}")
        print(f"  Goal: {problem.goal}")
        
        result = prover.prove(problem, max_iterations=30, timeout=10)
        results[name] = result
        
        if result['solved']:
            solved_count += 1
            print(f"  SOLVED in {result['iterations']} iterations, {result['facts_derived']} facts, {result['time']:.3f}s")
        else:
            print(f"  NOT SOLVED after {result['iterations']} iterations, {result['facts_derived']} facts, {result['time']:.3f}s")
        print()
    
    print(f"\n{'='*60}")
    print(f"Results: {solved_count}/{len(problems)} problems solved ({100*solved_count/len(problems):.1f}%)")
    
    # Save results
    with open('outputs/symbolic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to outputs/symbolic_results.json")

if __name__ == '__main__':
    main()
