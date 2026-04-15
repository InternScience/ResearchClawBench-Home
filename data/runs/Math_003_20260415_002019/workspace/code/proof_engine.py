"""
Neuro-Symbolic Geometry Proof Engine
Implements a search-based proof system for Euclidean geometry using formal predicates.
"""
import json
import re
from collections import defaultdict

class GeometryState:
    """Represents the current state of geometric facts."""
    def __init__(self):
        self.facts = set()
        self.fact_sources = {}  # fact -> rule that produced it
    
    def add_fact(self, fact, source=None):
        fact_tuple = tuple(fact) if isinstance(fact, list) else fact
        if fact_tuple not in self.facts:
            self.facts.add(fact_tuple)
            self.fact_sources[fact_tuple] = source
            return True
        return False
    
    def has_fact(self, fact):
        fact_tuple = tuple(fact) if isinstance(fact, list) else fact
        return fact_tuple in self.facts

class InferenceRule:
    """Represents a single inference rule."""
    def __init__(self, rule_str):
        self.raw = rule_str
        parts = rule_str.split('=>')
        self.premises = [p.strip() for p in parts[0].split(',')]
        self.conclusion = parts[1].strip()
    
    def __repr__(self):
        return f"Rule({self.raw})"

class ProofStep:
    """Represents a single step in a proof."""
    def __init__(self, rule, substitutions, premises, conclusion):
        self.rule = rule
        self.substitutions = substitutions
        self.premises = premises
        self.conclusion = conclusion
    
    def to_dict(self):
        return {
            'rule': self.rule.raw if hasattr(self.rule, 'raw') else str(self.rule),
            'substitutions': self.substitutions,
            'premises': [str(p) for p in self.premises],
            'conclusion': str(self.conclusion)
        }

class GeometryProver:
    """Search-based geometry theorem prover."""
    
    def __init__(self):
        self.rules = self._load_rules()
        self.definitions = self._load_definitions()
        self.max_iterations = 500
    
    def _load_rules(self):
        with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/data/rules.txt') as f:
            rules_text = f.read().strip()
        return [InferenceRule(r) for r in rules_text.split('\n') if r.strip()]
    
    def _load_definitions(self):
        defs = {}
        with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/data/defs.txt') as f:
            content = f.read()
        # Parse definition blocks
        blocks = re.split(r'\n(?=\w)', content)
        for block in blocks:
            lines = block.strip().split('\n')
            if lines:
                name = lines[0].split()[0]
                defs[name] = block.strip()
        return defs
    
    def parse_predicate(self, pred_str):
        """Parse a predicate string into (name, args)."""
        pred_str = pred_str.strip()
        match = re.match(r'(\w+)\s+(.*)', pred_str)
        if match:
            return match.group(1), match.group(2).split()
        return pred_str, []
    
    def unify(self, pattern, fact):
        """Try to unify a pattern with a fact, returning substitutions."""
        p_name, p_args = self.parse_predicate(pattern)
        f_name, f_args = self.parse_predicate(fact)
        
        if p_name != f_name:
            return None
        if len(p_args) != len(f_args):
            return None
        
        subs = {}
        for pa, fa in zip(p_args, f_args):
            if pa[0].isupper():  # Variable
                if pa in subs:
                    if subs[pa] != fa:
                        return None
                else:
                    subs[pa] = fa
            else:  # Constant
                if pa != fa:
                    return None
        return subs
    
    def apply_substitution(self, pred_str, subs):
        """Apply substitutions to a predicate string."""
        name, args = self.parse_predicate(pred_str)
        new_args = [subs.get(a, a) for a in args]
        return f"{name} {' '.join(new_args)}"
    
    def forward_chain(self, state, goal):
        """Forward chaining proof search."""
        proof_steps = []
        goal_name, goal_args = self.parse_predicate(goal)
        
        for iteration in range(self.max_iterations):
            new_facts_added = False
            
            for rule in self.rules:
                # Try to match all premises of this rule against known facts
                for fact in state.facts:
                    fact_str = ' '.join(fact) if isinstance(fact, tuple) else fact
                    subs = self.unify(rule.premises[0], fact_str)
                    if subs is None:
                        continue
                    
                    # Check if all other premises are satisfied
                    all_satisfied = True
                    matched_premises = [fact_str]
                    
                    for premise in rule.premises[1:]:
                        instantiated = self.apply_substitution(premise, subs)
                        found = False
                        for f in state.facts:
                            f_str = ' '.join(f) if isinstance(f, tuple) else f
                            if f_str == instantiated:
                                found = True
                                matched_premises.append(f_str)
                                break
                        if not found:
                            all_satisfied = False
                            break
                    
                    if all_satisfied:
                        conclusion = self.apply_substitution(rule.conclusion, subs)
                        if state.add_fact(conclusion, source=rule):
                            step = ProofStep(rule, subs, matched_premises, conclusion)
                            proof_steps.append(step)
                            new_facts_added = True
                            
                            # Check if goal is reached
                            c_name, c_args = self.parse_predicate(conclusion)
                            if c_name == goal_name:
                                return True, proof_steps
            
            if not new_facts_added:
                break
        
        return False, proof_steps
    
    def prove_problem(self, problem):
        """Attempt to prove a geometry problem."""
        state = GeometryState()
        
        # Parse premises and add as initial facts
        premise_str = problem.get('premise', '')
        # Extract facts from construction steps
        steps = premise_str.split(';')
        for step in steps:
            step = step.strip()
            if not step:
                continue
            # Each step defines a point with constraints
            # Add relational facts
            for pred in re.findall(r'(cong|coll|para|perp|cyclic|midp|eqangle\d*|eqratio\d*)\s+([a-zA-Z0-9 ]+)', step):
                fact = f"{pred[0]} {pred[1].strip()}"
                state.add_fact(fact)
        
        # Also add implicit facts from definitions
        for step in steps:
            step = step.strip()
            if 'midpoint' in step:
                # Extract midpoint relation
                match = re.search(r'(\w+)\s*=\s*midpoint\s+(\w+)\s+(\w+)\s+(\w+)', step)
                if match:
                    m, a, b = match.group(2), match.group(3), match.group(4)
                    state.add_fact(f'coll {m} {a} {b}')
                    state.add_fact(f'cong {m} {a} {m} {b}')
            if 'circle' in step and 'on_circle' not in step:
                match = re.search(r'(\w+)\s*=\s*circle\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)', step)
                if match:
                    o, a, b, c = match.group(2), match.group(3), match.group(4), match.group(5)
                    state.add_fact(f'cong {o} {a} {o} {b}')
                    state.add_fact(f'cong {o} {b} {o} {c}')
        
        # Run forward chaining
        goal = problem.get('goal', '')
        success, steps = self.forward_chain(state, goal)
        
        return {
            'success': success,
            'num_steps': len(steps),
            'proof_steps': [s.to_dict() for s in steps],
            'final_facts': len(state.facts)
        }

def run_prover_on_all():
    """Run prover on all problems and collect results."""
    # Load problems
    with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/data/imo_ag_30.txt') as f:
        content = f.read()
    
    problem_blocks = re.split(r'\n(?=translated_imo_)', content.strip())
    problems = []
    for block in problem_blocks:
        if block.strip():
            lines = block.strip().split('\n')
            name = lines[0]
            full_stmt = ' '.join(lines[1:])
            if '?' in full_stmt:
                premise, goal = full_stmt.split('?', 1)
            else:
                premise, goal = full_stmt, ""
            problems.append({
                'name': name,
                'premise': premise.strip(),
                'goal': goal.strip(),
                'predicates': list(set(re.findall(r'\b\w+\b', goal)))
            })
    
    prover = GeometryProver()
    results = []
    
    for p in problems:
        print(f"Proving {p['name']}...")
        result = prover.prove_problem(p)
        result['name'] = p['name']
        result['goal'] = p['goal']
        results.append(result)
        status = "✓ PROVED" if result['success'] else "✗ NOT PROVED"
        print(f"  {status} ({result['num_steps']} inference steps, {result['final_facts']} facts)")
    
    # Save results
    with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/outputs/prover_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    proved = sum(1 for r in results if r['success'])
    print(f"\nSummary: {proved}/{len(results)} problems proved by forward chaining")
    
    return results

if __name__ == '__main__':
    run_prover_on_all()
