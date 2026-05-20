"""
Advanced theorem prover with multiple search strategies.
"""
from typing import List, Tuple, Dict, Set, Optional, Callable
from collections import defaultdict
import itertools
import time
import random

from src.geometry_engine import Fact, GeometryState
from src.parser import Rule


def normalize_fact(fact: Fact) -> Fact:
    """Normalize symmetric predicates."""
    sym_perms = {
        'cong': [(0,1,2,3), (2,3,0,1)],
        'eqangle': [(0,1,2,3,4,5,6,7), (4,5,6,7,0,1,2,3)],
        'eqratio': [(0,1,2,3,4,5,6,7), (4,5,6,7,0,1,2,3)],
        'perp': [(0,1,2,3), (2,3,0,1)],
        'para': [(0,1,2,3), (2,3,0,1)],
        'coll': [(0,1,2), (2,1,0)],
    }
    
    if fact.predicate in sym_perms:
        perms = sym_perms[fact.predicate]
        args = list(fact.args)
        best = min(tuple(args[i] for i in p) for p in perms)
        return Fact(fact.predicate, best)
    
    if fact.predicate == 'cyclic':
        # Cyclic is invariant under rotation and reflection
        args = list(fact.args)
        n = len(args)
        best = None
        for start in range(n):
            for rev in [False, True]:
                if rev:
                    perm = [args[(start - i) % n] for i in range(n)]
                else:
                    perm = [args[(start + i) % n] for i in range(n)]
                t = tuple(perm)
                if best is None or t < best:
                    best = t
        return Fact(fact.predicate, best)
    
    return fact


class RuleMatcher:
    """Matches rules against facts."""
    
    def __init__(self, rules: List[Rule]):
        self.rules = rules
        self._index_rules()
    
    def _index_rules(self):
        """Index rules by their first positive premise predicate."""
        self.rule_index = defaultdict(list)
        for rule in self.rules:
            # Find first positive premise
            for pred, args in rule.premises:
                if pred not in ('ncoll', 'nperp', 'npara', 'diff', 'sameside'):
                    self.rule_index[pred].append(rule)
                    break
    
    def match_rule(self, rule: Rule, facts: Set[Fact]) -> List[Tuple[Dict[str, str], List[Fact]]]:
        """Find all ways to apply a rule."""
        # Separate premises
        pos_premises = []
        neg_premises = []
        
        for pred, args in rule.premises:
            if pred in ('ncoll', 'nperp', 'npara', 'diff', 'sameside'):
                neg_premises.append((pred, args))
            else:
                pos_premises.append((pred, args))
        
        facts_by_pred = defaultdict(list)
        for f in facts:
            facts_by_pred[f.predicate].append(f)
        
        # Recursive matching
        def match_list(premises, sub):
            if not premises:
                return [sub]
            pred, args = premises[0]
            results = []
            for fact in facts_by_pred.get(pred, []):
                new_sub = self._unify_pattern(pred, args, fact, sub)
                if new_sub is not None:
                    results.extend(match_list(premises[1:], new_sub))
            return results
        
        substitutions = match_list(pos_premises, {})
        
        valid = []
        for sub in substitutions:
            if self._check_neg_premises(neg_premises, sub, facts):
                new_facts = []
                for pred, args in rule.conclusions:
                    gf = self._apply_sub(pred, args, sub)
                    if gf and gf not in facts:
                        new_facts.append(gf)
                if new_facts:
                    valid.append((sub, new_facts))
        
        return valid
    
    def _unify_pattern(self, pred: str, args: List[str], fact: Fact, sub: Dict[str, str]) -> Optional[Dict[str, str]]:
        if pred != fact.predicate or len(args) != len(fact.args):
            return None
        new_sub = dict(sub)
        for pa, fa in zip(args, fact.args):
            if len(pa) == 1 and pa.isupper():
                if pa in new_sub:
                    if new_sub[pa] != fa:
                        return None
                else:
                    new_sub[pa] = fa
            elif pa != fa:
                return None
        return new_sub
    
    def _apply_sub(self, pred: str, args: List[str], sub: Dict[str, str]) -> Optional[Fact]:
        ground = []
        for a in args:
            if len(a) == 1 and a.isupper():
                if a not in sub:
                    return None
                ground.append(sub[a])
            else:
                ground.append(a)
        return normalize_fact(Fact(pred, tuple(ground)))
    
    def _check_neg_premises(self, neg_premises, sub, facts):
        for pred, args in neg_premises:
            ground_args = []
            for a in args:
                if len(a) == 1 and a.isupper():
                    if a not in sub:
                        return False
                    ground_args.append(sub[a])
                else:
                    ground_args.append(a)
            
            if pred == 'ncoll':
                a, b, c = ground_args[:3]
                if a == b or b == c or a == c:
                    return False
            elif pred == 'diff':
                a, b = ground_args[:2]
                if a == b:
                    return False
            elif pred == 'nperp':
                if Fact('perp', tuple(ground_args[:4])) in facts:
                    return False
            elif pred == 'npara':
                if Fact('para', tuple(ground_args[:4])) in facts:
                    return False
            elif pred == 'sameside':
                # Simplified
                pass
        return True
    
    def apply_all(self, facts: Set[Fact]) -> List[Tuple[Rule, Dict[str, str], List[Fact]]]:
        """Apply all rules to facts, return list of (rule, sub, new_facts)."""
        results = []
        seen_rules = set()
        for rule in self.rules:
            if id(rule) in seen_rules:
                continue
            seen_rules.add(id(rule))
            matches = self.match_rule(rule, facts)
            for sub, new_facts in matches:
                results.append((rule, sub, new_facts))
        return results


class SearchProver:
    """Prover with multiple search strategies."""
    
    def __init__(self, rules: List[Rule]):
        self.matcher = RuleMatcher(rules)
        self.stats = {}
    
    def prove_bfs(self, initial_facts: Set[Fact], goal: Fact,
                  max_depth: int = 15, max_nodes: int = 50000,
                  heuristic: Optional[Callable] = None) -> Optional[Dict]:
        """Breadth-first or best-first search."""
        start_time = time.time()
        goal = normalize_fact(goal)
        
        initial_facts = set(normalize_fact(f) for f in initial_facts)
        
        if goal in initial_facts:
            return {'success': True, 'proof': [goal], 'nodes': 1, 'depth': 0,
                    'time': time.time() - start_time, 'facts': len(initial_facts)}
        
        visited = set()
        
        if heuristic is None:
            # BFS
            queue = [(initial_facts, [])]
        else:
            # Best-first
            score = heuristic(initial_facts, goal)
            queue = [(score, initial_facts, [])]
        
        nodes = 0
        
        while queue and nodes < max_nodes:
            if heuristic is None:
                facts, trace = queue.pop(0)
            else:
                _, facts, trace = queue.pop(0)
            
            nodes += 1
            
            # Try all rules
            results = self.matcher.apply_all(facts)
            
            for rule, sub, new_facts in results:
                new_fact_set = set(facts)
                new_trace = list(trace)
                added = False
                
                for nf in new_facts:
                    if nf not in new_fact_set:
                        new_fact_set.add(nf)
                        new_trace.append(nf)
                        added = True
                        
                        if nf == goal:
                            elapsed = time.time() - start_time
                            return {
                                'success': True,
                                'proof': new_trace,
                                'nodes': nodes,
                                'depth': len(new_trace),
                                'time': elapsed,
                                'facts': len(new_fact_set)
                            }
                
                if added:
                    fs = frozenset(new_fact_set)
                    if fs not in visited:
                        visited.add(fs)
                        if len(new_trace) <= max_depth:
                            if heuristic is None:
                                queue.append((new_fact_set, new_trace))
                            else:
                                score = heuristic(new_fact_set, goal)
                                # Insert sorted
                                idx = 0
                                while idx < len(queue) and queue[idx][0] < score:
                                    idx += 1
                                queue.insert(idx, (score, new_fact_set, new_trace))
        
        elapsed = time.time() - start_time
        return {
            'success': False,
            'proof': None,
            'nodes': nodes,
            'depth': max_depth,
            'time': elapsed,
            'facts': len(initial_facts)
        }
    
    def prove_beam(self, initial_facts: Set[Fact], goal: Fact,
                   beam_width: int = 10, max_depth: int = 15) -> Optional[Dict]:
        """Beam search."""
        start_time = time.time()
        goal = normalize_fact(goal)
        initial_facts = set(normalize_fact(f) for f in initial_facts)
        
        if goal in initial_facts:
            return {'success': True, 'proof': [goal], 'nodes': 1, 'depth': 0,
                    'time': time.time() - start_time, 'facts': len(initial_facts)}
        
        beams = [(initial_facts, [])]
        nodes = 0
        
        for depth in range(max_depth):
            new_beams = []
            for facts, trace in beams:
                results = self.matcher.apply_all(facts)
                for rule, sub, new_facts in results:
                    new_fact_set = set(facts)
                    new_trace = list(trace)
                    added = False
                    
                    for nf in new_facts:
                        if nf not in new_fact_set:
                            new_fact_set.add(nf)
                            new_trace.append(nf)
                            added = True
                            
                            if nf == goal:
                                elapsed = time.time() - start_time
                                return {
                                    'success': True,
                                    'proof': new_trace,
                                    'nodes': nodes,
                                    'depth': len(new_trace),
                                    'time': elapsed,
                                    'facts': len(new_fact_set)
                                }
                    
                    if added:
                        nodes += 1
                        # Score by number of facts and heuristic proximity to goal
                        score = len(new_fact_set) + random.random()
                        new_beams.append((score, new_fact_set, new_trace))
            
            if not new_beams:
                break
            
            new_beams.sort(key=lambda x: x[0])
            beams = [(b[1], b[2]) for b in new_beams[:beam_width]]
        
        elapsed = time.time() - start_time
        return {
            'success': False,
            'proof': None,
            'nodes': nodes,
            'depth': max_depth,
            'time': elapsed,
            'facts': len(initial_facts)
        }


def simple_heuristic(facts: Set[Fact], goal: Fact) -> float:
    """Simple heuristic: prefer states with goal-related facts."""
    score = 0
    goal_points = set(goal.args)
    for f in facts:
        overlap = len(set(f.args) & goal_points)
        score -= overlap * 10  # More overlap = lower score (better)
    return score


def goal_distance_heuristic(facts: Set[Fact], goal: Fact) -> float:
    """Heuristic based on distance to goal."""
    score = 0
    goal_points = set(goal.args)
    goal_pred = goal.predicate
    
    for f in facts:
        # Same predicate is good
        if f.predicate == goal_pred:
            score -= 50
        # Shared points with goal
        overlap = len(set(f.args) & goal_points)
        score -= overlap * 5
    
    return score
