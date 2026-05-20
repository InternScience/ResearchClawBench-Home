"""
Parser for formal geometry problems, definitions, and rules.
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set

@dataclass
class Construction:
    """A geometric construction: point = predicate(args)"""
    point: str
    predicate: str
    args: List[str]

@dataclass
class GeometryProblem:
    """A geometry theorem proving problem."""
    name: str
    given_points: List[str]           # Points declared before constructions
    constructions: List[Construction] # Premise constructions
    goal_predicate: str               # Goal to prove
    goal_args: List[str]              # Goal arguments
    raw: str

@dataclass
class Definition:
    """A geometric definition."""
    name: str
    points: List[str]
    constraints: List[Tuple[str, List[str]]]
    constructions: List[Tuple[str, List[str]]]

@dataclass
class Rule:
    """An inference rule: premises => conclusion"""
    premises: List[Tuple[str, List[str]]]
    conclusions: List[Tuple[str, List[str]]]
    raw: str


def parse_problems(filepath: str) -> List[GeometryProblem]:
    """Parse imo_ag_30.txt into structured problems."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    problems = []
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    
    i = 0
    while i < len(lines):
        name = lines[i]
        i += 1
        if i >= len(lines):
            break
        stmt = lines[i]
        i += 1
        
        # Split into premises and goal
        if '?' not in stmt:
            continue
        premises_str, goal_str = stmt.split('?')
        premises_str = premises_str.strip()
        goal_str = goal_str.strip()
        
        # Parse goal
        goal_parts = goal_str.split()
        goal_predicate = goal_parts[0]
        goal_args = goal_parts[1:]
        
        # Parse premises
        given_points = []
        constructions = []
        declared_points = set()
        
        # Split by semicolon
        clauses = [c.strip() for c in premises_str.split(';') if c.strip()]
        
        for clause in clauses:
            if '=' in clause:
                # Construction: [points] = predicate args
                lhs, rhs = clause.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Handle multiple constructions for same point separated by comma
                parts = [p.strip() for p in rhs.split(',')]
                for part in parts:
                    tokens = part.split()
                    if not tokens:
                        continue
                    pred = tokens[0]
                    args = tokens[1:]
                    # lhs can be single point or multiple points
                    lhs_points = lhs.split()
                    for lp in lhs_points:
                        declared_points.add(lp)
                    # If single point, it's the constructed point
                    # If multiple (like "a b = segment a b"), no single constructed point
                    point = lhs if len(lhs_points) == 1 else None
                    constructions.append(Construction(point=point, predicate=pred, args=args))
            else:
                # Direct predicate assertion or point declaration
                tokens = clause.split()
                if len(tokens) >= 1:
                    # Could be point declarations like "a b c = triangle a b c"
                    # or standalone predicates
                    if len(tokens) >= 3 and tokens[1] == '=':
                        # point decls = predicate
                        decls = tokens[0].split()
                        for d in decls:
                            declared_points.add(d)
                        pred = tokens[2]
                        args = tokens[3:]
                        given_points.extend(decls)
                        constructions.append(Construction(point=None, predicate=pred, args=args))
                    else:
                        # Standalone predicate
                        pred = tokens[0]
                        args = tokens[1:]
                        constructions.append(Construction(point=None, predicate=pred, args=args))
        
        problems.append(GeometryProblem(
            name=name,
            given_points=given_points,
            constructions=constructions,
            goal_predicate=goal_predicate,
            goal_args=goal_args,
            raw=stmt
        ))
    
    return problems


def parse_rules(filepath: str) -> List[Rule]:
    """Parse rules.txt into inference rules."""
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    rules = []
    for line in lines:
        if '=>' not in line:
            continue
        premises_str, conclusions_str = line.split('=>', 1)
        
        premises = []
        for p in premises_str.split(','):
            p = p.strip()
            if not p:
                continue
            tokens = p.split()
            if len(tokens) < 2:
                continue
            # Handle conditions like ncoll, diff, etc.
            pred = tokens[0]
            args = tokens[1:]
            premises.append((pred, args))
        
        conclusions = []
        for c in conclusions_str.split(','):
            c = c.strip()
            if not c:
                continue
            tokens = c.split()
            if not tokens:
                continue
            pred = tokens[0]
            args = tokens[1:]
            conclusions.append((pred, args))
        
        rules.append(Rule(premises=premises, conclusions=conclusions, raw=line))
    
    return rules


def parse_definitions(filepath: str) -> List[Definition]:
    """Parse defs.txt into geometric definitions."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    defs = []
    blocks = content.split('\n\n')
    
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        
        name = lines[0].split()[0]
        points = []
        constraints = []
        constructions = []
        
        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == name:
                points = tokens[1:]
            elif tokens[0] == '=':
                continue
            elif tokens[0] in ('a:', 'b:', 'c:', 'd:', 'e:', 'x:', 'y:', 'z:', 'i:', 'o:', 'w:', 'm:', 'n:', 'p:', 'q:', 'r:', 't:'):
                # Variable constraint specification
                # e.g. "x : coll x b c, perp i x b c"
                rest = ' '.join(tokens[1:])
                for part in rest.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    ptokens = part.split()
                    if ptokens:
                        constraints.append((ptokens[0], ptokens[1:]))
            elif tokens[0] in ('free', 'segment', 'triangle', 'pentagon', 'quadrangle'):
                constructions.append((tokens[0], tokens[1:]))
            else:
                # Could be construction or constraint
                constructions.append((tokens[0], tokens[1:]))
        
        defs.append(Definition(name=name, points=points, constraints=constraints, constructions=constructions))
    
    return defs


if __name__ == '__main__':
    problems = parse_problems('data/imo_ag_30.txt')
    rules = parse_rules('data/rules.txt')
    defs = parse_definitions('data/defs.txt')
    
    print(f"Parsed {len(problems)} problems")
    print(f"Parsed {len(rules)} rules")
    print(f"Parsed {len(defs)} definitions")
    
    for p in problems[:3]:
        print(f"\n{p.name}:")
        print(f"  Constructions: {len(p.constructions)}")
        print(f"  Goal: {p.goal_predicate} {p.goal_args}")
