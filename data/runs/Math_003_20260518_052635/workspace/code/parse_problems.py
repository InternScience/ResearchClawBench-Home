#!/usr/bin/env python3
"""
Parser for IMO geometry problems in the formal language format.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum


class RelationType(Enum):
    """Types of geometric relations."""
    ON_LINE = "on_line"
    ON_CIRCLE = "on_circle"
    ON_TLINE = "on_tline"  # perpendicular line
    ON_PLINE = "on_pline"  # parallel line
    ON_ALINE = "on_aline"  # angle line
    ON_BLIONE = "on_bline"  # angle bisector line
    ON_DIA = "on_dia"  # on diameter
    ON_OPLINE = "on_opline"
    CONG = "cong"
    PERP = "perp"
    PARA = "para"
    EQANGLE = "eqangle"
    EQANGLE2 = "eqangle2"
    EQANGLE3 = "eqangle3"
    EQRATIO = "eqratio"
    CYCLIC = "cyclic"
    COLLINEAR = "coll"
    MIDPOINT = "midpoint"
    FOOT = "foot"
    MIRROR = "mirror"
    REFLECT = "reflect"
    ANGLE_BISECTOR = "angle_bisector"
    ANGLE_MIRROR = "angle_mirror"
    ORTHOCENTER = "orthocenter"
    INCENTER = "incenter"
    INCENTER2 = "incenter2"
    EXCENTER = "excenter"
    EXCENTER2 = "excenter2"
    CIRCLE = "circle"
    CIRCUMCENTER = "circumcenter"
    TRIANGLE = "triangle"
    SEGMENT = "segment"
    FREE = "free"
    EQDISTANCE = "eqdistance"
    PARALLELOGRAM = "parallelogram"
    CC_TANGENT = "cc_tangent"
    EQANGLE6 = "eqangle6"
    SIMTRI = "simtri"
    CONTRI = "contri"
    CONTRI2 = "contri2"
    SIMTRI2 = "simtri2"
    SIMTRI_STAR = "simtri*"
    CONTRI_STAR = "contri*"
    CC_TANGENT0 = "cc_tangent0"
    TANGENT = "tangent"
    LC_TANGENT = "lc_tangent"
    INTERSECTION_CC = "intersection_cc"
    INTERSECTION_LC = "intersection_lc"
    INTERSECTION_LL = "intersection_ll"
    INTERSECTION_LP = "intersection_lp"
    INTERSECTION_LT = "intersection_lt"
    INTERSECTION_PP = "intersection_pp"
    INTERSECTION_TT = "intersection_tt"
    NSQUARE = "nsquare"
    PSQUARE = "psquare"
    R_TRIANGLE = "r_triangle"
    ISO_TRIANGLE = "iso_triangle"
    I_EQUILATERAL = "ieq_triangle"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    I_SQUARE = "isquare"
    TRAPEZOID = "trapezoid"
    R_TRAPEZOID = "r_trapezoid"
    PENTAGON = "pentagon"
    QUADRANGLE = "quadrangle"
    EQ_QUADRANGLE = "eq_quadrangle"
    EQ_TRAPEZOID = "eq_trapezoid"
    EQDIA_QUADRANGLE = "eqdia_quadrangle"
    SHIFT = "shift"
    TRISEGMENT = "trisegment"
    TRISECT = "trisect"
    S_ANGLE = "s_angle"
    NCOLL = "ncoll"
    NPARA = "npara"
    NPERP = "nperp"
    DIFF = "diff"
    SAMESIDE = "sameside"
    R_CONST = "rconst"
    TRIANGLE12 = "triangle12"
    INCIRCLE2 = "2l1c"
    E5128 = "e5128"
    THREE_PEQ = "3peq"
    ON_CIRCUM = "on_circum"
    CIRCUMCENTER2 = "ninepoints"
    CENTROID = "centroid"


@dataclass
class Point:
    """Represents a point."""
    name: str
    x: Optional[float] = None
    y: Optional[float] = None
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class Statement:
    """Represents a geometric statement."""
    variables: List[str]
    relation: RelationType
    arguments: List[str]


@dataclass
class Problem:
    """Represents a geometry problem."""
    name: str
    statements: List[Statement]
    conclusion: Statement
    points: List[str] = None
    
    def __post_init__(self):
        if self.points is None:
            self.points = []


def parse_relation_args(relation_str: str) -> Tuple[RelationType, List[str]]:
    """Parse a relation string into relation type and arguments."""
    # Remove extra spaces
    relation_str = relation_str.strip()
    
    # Check for known relation types
    for rel_type in RelationType:
        if relation_str.startswith(rel_type.value):
            args_str = relation_str[len(rel_type.value):].strip()
            # Split by comma, but respect nested parentheses
            args = split_args(args_str)
            return rel_type, args
    
    # Default case
    return None, [relation_str]


def split_args(s: str) -> List[str]:
    """Split string by comma, respecting nested parentheses."""
    args = []
    current = ""
    depth = 0
    
    for char in s:
        if char == '(' or char == '[':
            depth += 1
            current += char
        elif char == ')' or char == ']':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            if current.strip():
                args.append(current.strip())
            current = ""
        else:
            current += char
    
    if current.strip():
        args.append(current.strip())
    
    return args


def parse_imo_problem(problem_text: str) -> Problem:
    """Parse an IMO geometry problem from text."""
    lines = problem_text.strip().split('\n')
    
    if not lines:
        return None
    
    problem_name = lines[0].strip()
    problem_body = ' '.join(lines[1:])
    
    # Split on '?' to separate premises from conclusion
    if '?' in problem_body:
        premises_str, conclusion_str = problem_body.split('?', 1)
    else:
        premises_str = problem_body
        conclusion_str = ""
    
    statements = []
    all_points = set()
    
    # Parse premises
    premises = premises_str.split(';')
    for premise in premises:
        premise = premise.strip()
        if not premise:
            continue
        
        # Check if it's a variable definition or a relation
        if ' = ' in premise:
            # Variable definition with relation
            var_part, relation_part = premise.split(' = ', 1)
            variables = var_part.strip().split()
            all_points.update(variables)
            
            # Parse multiple relations if present
            relations = relation_part.split(', ')
            for rel in relations:
                rel = rel.strip()
                if rel:
                    rel_type, args = parse_relation_args(rel)
                    if rel_type:
                        statements.append(Statement(
                            variables=variables,
                            relation=rel_type,
                            arguments=args
                        ))
        else:
            # Direct relation
            rel_type, args = parse_relation_args(premise)
            if rel_type:
                statements.append(Statement(
                    variables=[],
                    relation=rel_type,
                    arguments=args
                ))
    
    # Parse conclusion
    conclusion = None
    if conclusion_str:
        conclusion_str = conclusion_str.strip()
        rel_type, args = parse_relation_args(conclusion_str)
        if rel_type:
            conclusion = Statement(
                variables=[],
                relation=rel_type,
                arguments=args
            )
    
    return Problem(
        name=problem_name,
        statements=statements,
        conclusion=conclusion,
        points=sorted(list(all_points))
    )


def parse_imo_file(filepath: str) -> List[Problem]:
    """Parse the IMO geometry problems file."""
    problems = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by problem names
    problem_blocks = re.split(r'(translated_imo_\d+_p\w+)', content)
    
    i = 1
    while i < len(problem_blocks):
        name = problem_blocks[i].strip()
        body = problem_blocks[i + 1].strip() if i + 1 < len(problem_blocks) else ""
        
        problem_text = f"{name}\n{body}"
        problem = parse_imo_problem(problem_text)
        if problem:
            problems.append(problem)
        
        i += 2
    
    return problems


if __name__ == "__main__":
    # Test parser
    problems = parse_imo_file("data/imo_ag_30.txt")
    
    print(f"Found {len(problems)} problems\n")
    
    for p in problems[:3]:
        print(f"Problem: {p.name}")
        print(f"  Points: {p.points}")
        print(f"  Statements: {len(p.statements)}")
        print(f"  Conclusion: {p.conclusion}")
        print()