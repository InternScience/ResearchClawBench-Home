"""
Improved geometry symbolic engine with better fact extraction.
"""
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import time

@dataclass(frozen=True)
class Fact:
    predicate: str
    args: Tuple[str, ...]
    
    def __repr__(self):
        return f"{self.predicate}({','.join(self.args)})"
    
    def __hash__(self):
        return hash((self.predicate, self.args))


class GeometryState:
    """Represents the current state of geometric knowledge."""
    
    def __init__(self):
        self.facts: Set[Fact] = set()
        self.points: Set[str] = set()
        self.constructions: List[Tuple[str, str, List[str]]] = []
    
    def add_fact(self, pred: str, args: Tuple[str, ...]):
        self.facts.add(Fact(pred, args))
        for a in args:
            self.points.add(a)
    
    def add_construction(self, point: Optional[str], pred: str, args: List[str]):
        self.constructions.append((point, pred, args))
        if point:
            self.points.add(point)
        for a in args:
            self.points.add(a)
    
    def clone(self):
        s = GeometryState()
        s.facts = set(self.facts)
        s.points = set(self.points)
        s.constructions = list(self.constructions)
        return s
    
    def __repr__(self):
        return f"State({len(self.facts)} facts, {len(self.points)} points)"


def translate_construction(state: GeometryState, point: Optional[str], pred: str, args: List[str]):
    """Translate a construction into geometric facts."""
    
    # Guard: skip empty-arg constructions for predicates that need args
    if not args and pred not in ('free', 'segment', 'triangle', 'pentagon', 'quadrangle', 'isos'):
        return
    
    if pred == 'segment':
        # segment a b: just declares two distinct points
        state.add_fact('diff', tuple(args[:2]))
    
    elif pred == 'triangle':
        # triangle a b c: three non-collinear points
        if len(args) < 3:
            return
        a, b, c = args[:3]
        state.add_fact('diff', (a, b))
        state.add_fact('diff', (b, c))
        state.add_fact('diff', (a, c))
        state.add_fact('ncoll', (a, b, c))
    
    elif pred == 'r_triangle':
        # right triangle with right angle at c
        if len(args) < 3:
            return
        a, b, c = args[:3]
        state.add_fact('perp', (a, c, a, b))
        state.add_fact('diff', (a, b))
        state.add_fact('diff', (b, c))
        state.add_fact('diff', (a, c))
    
    elif pred == 'iso_triangle':
        # isosceles triangle
        if len(args) < 3:
            return
        a, b, c = args[:3]
        state.add_fact('eqangle', (b, a, b, c, c, b, c, a))
        state.add_fact('cong', (a, b, a, c))
    
    elif pred == 'free':
        state.add_fact('free', tuple(args[:1]))
    
    elif pred == 'on_line':
        # x on line a b => coll x a b
        x, a, b = args[:3]
        state.add_fact('coll', (x, a, b))
        state.add_fact('diff', (a, b))
    
    elif pred == 'on_opline':
        # x on opposite line a b => coll x a b
        x, a, b = args[:3]
        state.add_fact('coll', (x, a, b))
        state.add_fact('diff', (a, b))
    
    elif pred == 'on_tline':
        # x on tline a b c => perp x a b c
        x, a, b, c = args[:4]
        state.add_fact('perp', (x, a, b, c))
    
    elif pred == 'on_pline':
        # x on pline a b c => para x a b c
        x, a, b, c = args[:4]
        state.add_fact('para', (x, a, b, c))
    
    elif pred == 'on_bline':
        # x on bline a b => perp x a a b AND cong x a x b
        x, a, b = args[:3]
        state.add_fact('perp', (x, a, a, b))
        state.add_fact('cong', (x, a, x, b))
        state.add_fact('diff', (a, b))
    
    elif pred == 'on_circle':
        # x on_circle o a => cong o x o a
        x, o, a = args[:3]
        state.add_fact('cong', (o, x, o, a))
        state.add_fact('diff', (o, a))
    
    elif pred == 'on_dia':
        # x on_dia a b => perp x a x b
        x, a, b = args[:3]
        state.add_fact('perp', (x, a, x, b))
        state.add_fact('diff', (a, b))
    
    elif pred == 'on_circum':
        # x on_circum a b c => cyclic a b c x
        x, a, b, c = args[:4]
        state.add_fact('cyclic', (a, b, c, x))
    
    elif pred == 'midpoint':
        # midpoint x a b => coll x a b, cong x a x b
        x, a, b = args[:3]
        state.add_fact('midp', (x, a, b))
        state.add_fact('coll', (x, a, b))
        state.add_fact('cong', (x, a, x, b))
        state.add_fact('diff', (a, b))
    
    elif pred == 'mirror':
        # mirror x a b => coll x a b, cong b a b x
        x, a, b = args[:3]
        state.add_fact('pmirror', (x, a, b))
        state.add_fact('coll', (x, a, b))
        state.add_fact('cong', (b, a, b, x))
        state.add_fact('diff', (a, b))
    
    elif pred == 'foot':
        # foot x a b c => perp x a b c, coll x b c
        x, a, b, c = args[:4]
        state.add_fact('perp', (x, a, b, c))
        state.add_fact('coll', (x, b, c))
    
    elif pred == 'circle':
        # circle o a b c => cong o a o b, cong o b o c
        o, a, b, c = args[:4]
        state.add_fact('cong', (o, a, o, b))
        state.add_fact('cong', (o, b, o, c))
        state.add_fact('ncoll', (a, b, c))
    
    elif pred == 'orthocenter':
        # orthocenter h a b c => perp h a b c, perp h b c a
        h, a, b, c = args[:4]
        state.add_fact('perp', (h, a, b, c))
        state.add_fact('perp', (h, b, c, a))
        state.add_fact('perp', (h, c, a, b))
    
    elif pred == 'incenter':
        # incenter i a b c => eqangle a b a i a i a c, etc.
        i, a, b, c = args[:4]
        state.add_fact('eqangle', (a, b, a, i, a, i, a, c))
        state.add_fact('eqangle', (b, c, b, i, b, i, b, a))
        state.add_fact('eqangle', (c, a, c, i, c, i, c, b))
    
    elif pred == 'incenter2':
        # incenter2 x y z i a b c
        x, y, z, i, a, b, c = args[:7]
        state.add_fact('eqangle', (a, b, a, i, a, i, a, c))
        state.add_fact('eqangle', (b, c, b, i, b, i, b, a))
        state.add_fact('eqangle', (c, a, c, i, c, i, c, b))
        state.add_fact('coll', (x, b, c))
        state.add_fact('perp', (i, x, b, c))
        state.add_fact('coll', (y, c, a))
        state.add_fact('perp', (i, y, c, a))
        state.add_fact('coll', (z, a, b))
        state.add_fact('perp', (i, z, a, b))
        state.add_fact('cong', (i, x, i, y))
        state.add_fact('cong', (i, y, i, z))
    
    elif pred == 'excenter2':
        # Similar to incenter2 but with excenter properties
        x, y, z, i, a, b, c = args[:7]
        state.add_fact('eqangle', (a, b, a, i, a, i, a, c))
        state.add_fact('eqangle', (b, c, b, i, b, i, b, a))
        state.add_fact('eqangle', (c, a, c, i, c, i, c, b))
        state.add_fact('coll', (x, b, c))
        state.add_fact('perp', (i, x, b, c))
        state.add_fact('coll', (y, c, a))
        state.add_fact('perp', (i, y, c, a))
        state.add_fact('coll', (z, a, b))
        state.add_fact('perp', (i, z, a, b))
        state.add_fact('cong', (i, x, i, y))
        state.add_fact('cong', (i, y, i, z))
    
    elif pred == 'centroid':
        x, y, z, i, a, b, c = args[:7]
        state.add_fact('midp', (x, b, c))
        state.add_fact('midp', (y, c, a))
        state.add_fact('midp', (z, a, b))
        state.add_fact('coll', (a, x, i))
        state.add_fact('coll', (b, y, i))
        state.add_fact('coll', (c, z, i))
    
    elif pred == 'ninepoints':
        x, y, z, i, a, b, c = args[:7]
        state.add_fact('midp', (x, b, c))
        state.add_fact('midp', (y, c, a))
        state.add_fact('midp', (z, a, b))
        state.add_fact('cong', (i, x, i, y))
        state.add_fact('cong', (i, y, i, z))
    
    elif pred == 'angle_bisector':
        # x = angle_bisector x a b c => eqangle b a b x b x b c
        x, a, b, c = args[:4]
        state.add_fact('eqangle', (b, a, b, x, b, x, b, c))
        state.add_fact('bisect', (a, b, c))
    
    elif pred == 'angle_mirror':
        x, a, b, c = args[:4]
        state.add_fact('eqangle', (b, a, b, c, b, c, b, x))
        state.add_fact('amirror', (a, b, c))
    
    elif pred == 'reflect':
        x, a, b, c = args[:4]
        state.add_fact('cong', (b, a, b, x))
        state.add_fact('cong', (c, a, c, x))
        state.add_fact('perp', (b, c, a, x))
    
    elif pred == 'parallelogram':
        a, b, c, x = args[:4]
        state.add_fact('para', (a, b, c, x))
        state.add_fact('para', (a, x, b, c))
        state.add_fact('cong', (a, b, c, x))
        state.add_fact('cong', (a, x, b, c))
    
    elif pred == 'eqdistance':
        x, a, b, c = args[:4]
        state.add_fact('cong', (x, a, b, c))
    
    elif pred == 'eqangle2':
        x, a, b, c = args[:4]
        state.add_fact('eqangle', (a, b, a, x, c, x, c, b))
    
    elif pred == 'eqangle3':
        x, a, b, d, e, f = args[:6]
        state.add_fact('eqangle', (x, a, x, b, d, e, d, f))
    
    elif pred == 'on_aline':
        x, a, b, c, d, e = args[:6]
        state.add_fact('eqangle', (a, x, a, b, d, c, d, e))
    
    elif pred == 'on_aline2':
        x, a, b, c, d, e = args[:6]
        state.add_fact('eqangle', (x, a, x, b, d, c, d, e))
    
    elif pred == 'shift':
        x, b, c, d = args[:4]
        state.add_fact('cong', (x, b, c, d))
        state.add_fact('cong', (x, c, b, d))
    
    elif pred == 's_angle':
        x, a, b, y = args[:4]
        state.add_fact('s_angle', (a, b, x, y))
    
    elif pred == 'nsquare':
        x, a, b = args[:3]
        state.add_fact('cong', (x, a, a, b))
        state.add_fact('perp', (x, a, a, b))
    
    elif pred == 'psquare':
        x, a, b = args[:3]
        state.add_fact('cong', (x, a, a, b))
        state.add_fact('perp', (x, a, a, b))
    
    elif pred == 'square':
        a, b, x, y = args[:4]
        state.add_fact('perp', (a, b, b, x))
        state.add_fact('cong', (a, b, b, x))
        state.add_fact('para', (a, b, x, y))
        state.add_fact('para', (a, y, b, x))
        state.add_fact('perp', (a, y, y, x))
        state.add_fact('cong', (b, x, x, y))
        state.add_fact('cong', (x, y, y, a))
        state.add_fact('perp', (a, x, b, y))
        state.add_fact('cong', (a, x, b, y))
    
    elif pred == 'isquare':
        a, b, c, d = args[:4]
        state.add_fact('perp', (a, b, b, c))
        state.add_fact('cong', (a, b, b, c))
        state.add_fact('para', (a, b, c, d))
        state.add_fact('para', (a, d, b, c))
        state.add_fact('perp', (a, d, d, c))
        state.add_fact('cong', (b, c, c, d))
        state.add_fact('cong', (c, d, d, a))
        state.add_fact('perp', (a, c, b, d))
        state.add_fact('cong', (a, c, b, d))
    
    elif pred == 'rectangle':
        a, b, c, d = args[:4]
        state.add_fact('perp', (a, b, b, c))
        state.add_fact('para', (a, b, c, d))
        state.add_fact('para', (a, d, b, c))
        state.add_fact('perp', (a, b, a, d))
        state.add_fact('cong', (a, b, c, d))
        state.add_fact('cong', (a, d, b, c))
        state.add_fact('cong', (a, c, b, d))
    
    elif pred == 'trapezoid':
        a, b, c, d = args[:4]
        state.add_fact('para', (a, b, c, d))
    
    elif pred == 'eq_trapezoid':
        a, b, c, d = args[:4]
        state.add_fact('para', (d, c, a, b))
        state.add_fact('cong', (d, a, b, c))
    
    elif pred == 'r_trapezoid':
        a, b, c, d = args[:4]
        state.add_fact('para', (a, b, c, d))
        state.add_fact('perp', (a, b, a, d))
    
    elif pred == 'eq_quadrangle':
        a, b, c, d = args[:4]
        state.add_fact('cong', (d, a, b, c))
    
    elif pred == 'eqdia_quadrangle':
        a, b, c, d = args[:4]
        state.add_fact('cong', (d, b, a, c))
    
    elif pred == 'intersection_ll':
        x, a, b, c, d = args[:5]
        state.add_fact('coll', (x, a, b))
        state.add_fact('coll', (x, c, d))
    
    elif pred == 'intersection_lc':
        x, a, o, b = args[:4]
        state.add_fact('coll', (x, a, b))
        state.add_fact('cong', (o, b, o, x))
    
    elif pred == 'intersection_cc':
        x, o, w, a = args[:4]
        state.add_fact('cong', (o, a, o, x))
        state.add_fact('cong', (w, a, w, x))
    
    elif pred == 'intersection_lp':
        x, a, b, c, m, n = args[:6]
        state.add_fact('coll', (x, a, b))
        state.add_fact('para', (c, x, m, n))
    
    elif pred == 'intersection_lt':
        x, a, b, c, d, e = args[:6]
        state.add_fact('coll', (x, a, b))
        state.add_fact('perp', (x, c, d, e))
    
    elif pred == 'intersection_pp':
        x, a, b, c, d, e, f = args[:7]
        state.add_fact('para', (x, a, b, c))
        state.add_fact('para', (x, d, e, f))
    
    elif pred == 'intersection_tt':
        x, a, b, c, d, e, f = args[:7]
        state.add_fact('perp', (x, a, b, c))
        state.add_fact('perp', (x, d, e, f))
    
    elif pred == 'tangent':
        x, y, a, o, b = args[:5]
        state.add_fact('cong', (o, x, o, b))
        state.add_fact('perp', (a, x, o, x))
        state.add_fact('cong', (o, y, o, b))
        state.add_fact('perp', (a, y, o, y))
    
    elif pred == 'lc_tangent':
        x, a, o = args[:3]
        state.add_fact('perp', (a, x, a, o))
    
    elif pred == 'cc_tangent0':
        x, y, o, a, w, b = args[:6]
        state.add_fact('cong', (o, x, o, a))
        state.add_fact('cong', (w, y, w, b))
        state.add_fact('perp', (x, o, x, y))
        state.add_fact('perp', (y, w, y, x))
    
    elif pred == 'cc_tangent':
        x, y, z, i, o, a, w, b = args[:8]
        state.add_fact('cong', (o, x, o, a))
        state.add_fact('cong', (w, y, w, b))
        state.add_fact('perp', (x, o, x, y))
        state.add_fact('perp', (y, w, y, x))
        state.add_fact('cong', (o, z, o, a))
        state.add_fact('cong', (w, i, w, b))
        state.add_fact('perp', (z, o, z, i))
        state.add_fact('perp', (i, w, i, z))
    
    elif pred == 'e5128':
        x, y, a, b, c, d = args[:6]
        state.add_fact('cong', (c, b, c, x))
        state.add_fact('coll', (y, a, b))
        state.add_fact('coll', (x, y, d))
        state.add_fact('eqangle', (a, b, a, d, x, a, x, y))
    
    elif pred == '3peq':
        z, x, y, a, b, c = args[:6]
        state.add_fact('coll', (z, b, c))
        state.add_fact('coll', (x, a, b))
        state.add_fact('coll', (y, a, c))
        state.add_fact('coll', (x, y, z))
        state.add_fact('cong', (z, x, z, y))
    
    elif pred == 'trisect':
        x, y, a, b, c = args[:5]
        state.add_fact('coll', (x, a, c))
        state.add_fact('coll', (y, a, c))
        state.add_fact('eqangle', (b, a, b, x, b, x, b, y))
        state.add_fact('eqangle', (b, x, b, y, b, y, b, c))
    
    elif pred == 'trisegment':
        x, y, a, b = args[:4]
        state.add_fact('coll', (x, a, b))
        state.add_fact('coll', (y, a, b))
        state.add_fact('cong', (x, a, x, y))
        state.add_fact('cong', (y, x, y, b))
    
    elif pred == '2l1c':
        x, y, z, i, a, b, c, o = args[:8]
        state.add_fact('coll', (x, a, c))
        state.add_fact('coll', (y, b, c))
        state.add_fact('cong', (o, a, o, z))
        state.add_fact('coll', (i, o, z))
        state.add_fact('cong', (i, x, i, y))
        state.add_fact('cong', (i, y, i, z))
        state.add_fact('perp', (i, x, a, c))
        state.add_fact('perp', (i, y, b, c))
    
    elif pred == 'eq_triangle':
        x, b, c = args[:3]
        state.add_fact('cong', (x, b, b, c))
        state.add_fact('cong', (b, c, c, x))
    
    elif pred == 'ieq_triangle':
        if len(args) < 3:
            return
        a, b, c = args[:3]
        state.add_fact('cong', (a, b, b, c))
        state.add_fact('cong', (b, c, c, a))
    
    elif pred == 'triangle12':
        if len(args) < 3:
            return
        a, b, c = args[:3]
        state.add_fact('rconst', (a, b, a, c, 1, 2))
    
    elif pred == 'risos':
        if len(args) < 3:
            return
        a, b, c = args[:3]
        state.add_fact('perp', (a, b, a, c))
        state.add_fact('cong', (a, b, a, c))
    
    elif pred in ('eqangle6', 'eqratio6', 'cong', 'perp', 'para', 
                   'coll', 'cyclic', 'midp', 'pmirror', 'cong', 'contri',
                   'simtri', 'contri2', 'simtri2', 'contri*', 'simtri*',
                   'eqratio3', 'rconst', 'amirror', 'bisect', 'free',
                   'pentagon', 'quadrangle', 'isos', 'trapezoid'):
        # Direct facts
        state.add_fact(pred, tuple(args))
    
    else:
        # Unknown predicate, just add it
        state.add_fact(pred, tuple(args))


def problem_to_state(problem) -> GeometryState:
    """Convert a parsed problem to an initial geometry state."""
    state = GeometryState()
    
    for c in problem.constructions:
        args = list(c.args)
        # Some constructions use format: point = predicate arg1 arg2
        # where point is only on LHS, not in args
        if c.point and c.point not in args:
            # Prepend the constructed point to args
            args = [c.point] + args
        translate_construction(state, c.point, c.predicate, args)
        state.add_construction(c.point, c.predicate, args)
    
    return state
