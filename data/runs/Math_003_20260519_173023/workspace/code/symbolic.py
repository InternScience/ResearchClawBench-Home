"""
Symbolic forward-chaining theorem prover for Euclidean geometry.
"""
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import time

@dataclass(frozen=True)
class Fact:
    """A ground geometric fact."""
    predicate: str
    args: Tuple[str, ...]
    
    def __repr__(self):
        return f"{self.predicate}({','.join(self.args)})"

@dataclass
class ProofState:
    """Current state of a proof attempt."""
    facts: Set[Fact]
    goal: Fact
    depth: int = 0
    parent: Optional['ProofState'] = None
    applied_rule: Optional[str] = None


def match_pattern(pattern_pred: str, pattern_args: List[str],
                  fact: Fact,
                  substitution: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Try to match a pattern against a fact under current substitution."""
    if pattern_pred != fact.predicate:
        return None
    if len(pattern_args) != len(fact.args):
        return None
    
    new_sub = dict(substitution)
    for pa, fa in zip(pattern_args, fact.args):
        if pa.isupper():
            # Variable
            if pa in new_sub:
                if new_sub[pa] != fa:
                    return None
            else:
                new_sub[pa] = fa
        else:
            # Constant
            if pa != fa:
                return None
    return new_sub


def apply_substitution(pred: str, args: List[str], sub: Dict[str, str]) -> Optional[Fact]:
    """Apply substitution to a pattern to get a ground fact."""
    ground_args = []
    for a in args:
        if a.isupper():
            if a not in sub:
                return None
            ground_args.append(sub[a])
        else:
            ground_args.append(a)
    return Fact(pred, tuple(ground_args))


def try_apply_rule(rule, facts: Set[Fact]) -> List[Tuple[Dict[str, str], List[Fact]]]:
    """
    Try to apply a rule to a set of facts.
    Returns list of (substitution, new_facts) tuples.
    """
    from src.parser import Rule
    rule: Rule
    
    # Separate positive premises from negative/constraint premises
    pos_premises = []
    neg_premises = []
    
    for pred, args in rule.premises:
        if pred.startswith('n') and pred not in ('ncoll', 'nperp', 'npara', 'sameside'):
            # Not a negation predicate
            pos_premises.append((pred, args))
        elif pred in ('ncoll', 'nperp', 'npara', 'diff', 'sameside'):
            neg_premises.append((pred, args))
        else:
            pos_premises.append((pred, args))
    
    # Group facts by predicate for efficient lookup
    facts_by_pred = defaultdict(list)
    for f in facts:
        facts_by_pred[f.predicate].append(f)
    
    # Find all substitutions that satisfy positive premises
    def match_premises(premises, sub):
        if not premises:
            return [sub]
        pred, args = premises[0]
        rest = premises[1:]
        results = []
        for fact in facts_by_pred.get(pred, []):
            new_sub = match_pattern(pred, args, fact, sub)
            if new_sub is not None:
                results.extend(match_premises(rest, new_sub))
        return results
    
    substitutions = match_premises(pos_premises, {})
    
    # Filter by negative premises
    valid_results = []
    for sub in substitutions:
        valid = True
        for pred, args in neg_premises:
            ground = apply_substitution(pred, args, sub)
            if ground is None:
                valid = False
                break
            # Check negative conditions
            if pred == 'ncoll':
                a, b, c = ground.args[:3]
                if a == b or b == c or a == c:
                    valid = False
                    break
            elif pred == 'nperp':
                # Simplified: just check if perp doesn't exist
                perp_fact = Fact('perp', ground.args[:4])
                if perp_fact in facts:
                    valid = False
                    break
            elif pred == 'npara':
                para_fact = Fact('para', ground.args[:4])
                if para_fact in facts:
                    valid = False
                    break
            elif pred == 'diff':
                a, b = ground.args[:2]
                if a == b:
                    valid = False
                    break
            elif pred == 'sameside':
                # Simplified check
                pass
        
        if valid:
            new_facts = []
            for pred, args in rule.conclusions:
                gf = apply_substitution(pred, args, sub)
                if gf and gf not in facts:
                    new_facts.append(gf)
            if new_facts:
                valid_results.append((sub, new_facts))
    
    return valid_results


class ForwardProver:
    """Forward-chaining theorem prover."""
    
    def __init__(self, rules):
        self.rules = rules
        self.stats = {
            'nodes_expanded': 0,
            'facts_derived': 0,
            'time': 0
        }
    
    def prove(self, initial_facts: Set[Fact], goal: Fact,
              max_depth: int = 10, max_nodes: int = 10000) -> Optional[List[Fact]]:
        """
        Try to prove goal from initial facts using forward chaining.
        Returns proof trace (list of facts) or None.
        """
        start_time = time.time()
        self.stats = {
            'nodes_expanded': 0,
            'facts_derived': 0,
            'time': 0
        }
        
        # Normalize facts (canonical ordering for symmetric predicates)
        current_facts = set(self._normalize_fact(f) for f in initial_facts)
        goal = self._normalize_fact(goal)
        
        if goal in current_facts:
            return [goal]
        
        # BFS forward chaining
        visited = {frozenset(current_facts)}
        queue = [(current_facts, [])]
        nodes = 0
        
        while queue and nodes < max_nodes:
            facts, trace = queue.pop(0)
            nodes += 1
            self.stats['nodes_expanded'] = nodes
            
            # Try each rule
            for rule in self.rules:
                results = try_apply_rule(rule, facts)
                for sub, new_facts in results:
                    new_fact_set = set(facts)
                    new_trace = list(trace)
                    added = False
                    for nf in new_facts:
                        nf = self._normalize_fact(nf)
                        if nf not in new_fact_set:
                            new_fact_set.add(nf)
                            new_trace.append(nf)
                            self.stats['facts_derived'] += 1
                            added = True
                            
                            if nf == goal:
                                self.stats['time'] = time.time() - start_time
                                return new_trace
                    
                    if added:
                        fs = frozenset(new_fact_set)
                        if fs not in visited:
                            visited.add(fs)
                            queue.append((new_fact_set, new_trace))
        
        self.stats['time'] = time.time() - start_time
        return None
    
    def _normalize_fact(self, fact: Fact) -> Fact:
        """Normalize symmetric predicates."""
        sym_preds = {
            'cong': (4, [(1,0,3,2)]),
            'eqangle': (8, [(4,5,6,7,0,1,2,3)]),
            'eqratio': (8, [(4,5,6,7,0,1,2,3)]),
            'perp': (4, [(2,3,0,1)]),
            'para': (4, [(2,3,0,1)]),
            'cyclic': (4, list(itertools.permutations(range(4)))),
            'coll': (3, [(2,1,0)]),
        }
        
        if fact.predicate in sym_preds:
            n, perms = sym_preds[fact.predicate]
            args = fact.args[:n]
            best = min(tuple(args[i] for i in p) for p in perms)
            return Fact(fact.predicate, best)
        
        return fact


def problem_to_facts(problem) -> Set[Fact]:
    """Convert a problem's constructions to initial facts."""
    facts = set()
    
    # Track known points to generate diff facts
    known_points = set()
    
    for c in problem.constructions:
        if c.point:
            known_points.add(c.point)
        for a in c.args:
            known_points.add(a)
    
    # Add explicit constructions as facts
    for c in problem.constructions:
        pred = c.predicate
        args = c.args
        
        # Some constructions directly give facts
        if pred in ('on_line', 'on_tline', 'on_pline', 'on_bline', 'on_circle',
                    'on_aline', 'on_aline2', 'on_dia', 'on_opline', 'on_circum'):
            # These are constraints that can be translated
            if pred == 'on_line':
                facts.add(Fact('coll', tuple(args)))
            elif pred == 'on_tline':
                facts.add(Fact('perp', tuple(args[:2] + args[2:4])))
            elif pred == 'on_pline':
                facts.add(Fact('para', tuple(args[:2] + args[2:4])))
            elif pred == 'on_bline':
                facts.add(Fact('perp', tuple(args[:2] + args)))
            elif pred == 'on_circle':
                facts.add(Fact('cong', tuple(args[1:2] + args[:1] + args[1:3])))
            elif pred == 'on_dia':
                facts.add(Fact('perp', tuple(args[:2] + args[:2])))
        elif pred == 'midpoint':
            facts.add(Fact('midp', tuple(args)))
        elif pred == 'mirror':
            facts.add(Fact('pmirror', tuple(args)))
        elif pred == 'foot':
            facts.add(Fact('perp', tuple(args[:2] + args[1:4])))
            facts.add(Fact('coll', tuple(args[:1] + args[2:4])))
        elif pred == 'segment':
            pass  # Just declares two points
        elif pred == 'triangle':
            pass  # Just declares three points
        elif pred == 'free':
            pass
        elif pred in ('circle', 'orthocenter', 'incenter', 'centroid',
                       'circumcenter', 'ninepoints'):
            # These define properties
            if pred == 'circle' and len(args) == 4:
                facts.add(Fact('cong', (args[0], args[1], args[0], args[2])))
                facts.add(Fact('cong', (args[0], args[2], args[0], args[3])))
        else:
            # Generic predicate
            facts.add(Fact(pred, tuple(args)))
    
    return facts


def normalize_goal(goal_pred, goal_args):
    """Create a goal fact."""
    return Fact(goal_pred, tuple(goal_args))


if __name__ == '__main__':
    from src.parser import parse_problems, parse_rules
    
    problems = parse_problems('data/imo_ag_30.txt')
    rules = parse_rules('data/rules.txt')
    
    prover = ForwardProver(rules)
    
    # Test on first problem
    p = problems[0]
    facts = problem_to_facts(p)
    goal = normalize_goal(p.goal_predicate, p.goal_args)
    
    print(f"Problem: {p.name}")
    print(f"Initial facts: {len(facts)}")
    print(f"Goal: {goal}")
    
    result = prover.prove(facts, goal, max_depth=5, max_nodes=1000)
    print(f"Result: {result is not None}")
    print(f"Stats: {prover.stats}")
