#!/usr/bin/env python3
"""
Complete geometry problem solver for IMO problems.
"""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import itertools
import re


@dataclass
class GeometryFact:
    """Represents a geometric fact."""
    predicate: str
    args: tuple
    
    def __hash__(self):
        return hash((self.predicate, self.args))
    
    def __eq__(self, other):
        return self.predicate == other.predicate and self.args == other.args
    
    def __repr__(self):
        return f"{self.predicate}({', '.join(self.args)})"


class GeometrySolver:
    """Complete geometry problem solver."""
    
    def __init__(self):
        self.facts = set()
        self.variables = set()
        self.rules = self._define_inference_rules()
        self.derivation_trace = []
    
    def _define_inference_rules(self) -> List[dict]:
        """Define comprehensive inference rules for geometry."""
        rules = []
        
        # Transitivity rules
        rules.append({
            "name": "eqangle_trans",
            "premises": [
                ("eqangle", ("a", "b", "c", "d", "m", "n", "p", "q")),
                ("eqangle", ("c", "d", "e", "f", "p", "q", "r", "u"))
            ],
            "conclusion": ("eqangle", ("a", "b", "e", "f", "m", "n", "r", "u"))
        })
        
        rules.append({
            "name": "cong_trans",
            "premises": [
                ("cong", ("a", "b", "c", "d")),
                ("cong", ("c", "d", "e", "f"))
            ],
            "conclusion": ("cong", ("a", "b", "e", "f"))
        })
        
        # Perpendicular rules
        rules.append({
            "name": "perp_perp_para",
            "premises": [
                ("perp", ("a", "b", "c", "d")),
                ("perp", ("c", "d", "e", "f"))
            ],
            "conclusion": ("para", ("a", "b", "e", "f"))
        })
        
        rules.append({
            "name": "perp_symmetric",
            "premises": [
                ("perp", ("a", "b", "c", "d"))
            ],
            "conclusion": ("perp", ("c", "d", "a", "b"))
        })
        
        # Parallel rules
        rules.append({
            "name": "para_symmetric",
            "premises": [
                ("para", ("a", "b", "c", "d"))
            ],
            "conclusion": ("para", ("c", "d", "a", "b"))
        })
        
        rules.append({
            "name": "para_trans",
            "premises": [
                ("para", ("a", "b", "c", "d")),
                ("para", ("c", "d", "e", "f"))
            ],
            "conclusion": ("para", ("a", "b", "e", "f"))
        })
        
        # Congruence rules
        rules.append({
            "name": "cong_symmetric",
            "premises": [
                ("cong", ("a", "b", "c", "d"))
            ],
            "conclusion": ("cong", ("c", "d", "a", "b"))
        })
        
        # Cyclic quadrilateral rules
        rules.append({
            "name": "concyclic_4points",
            "premises": [
                ("on_circle", ("o", "a")),
                ("on_circle", ("o", "b")),
                ("on_circle", ("o", "c")),
                ("on_circle", ("o", "d"))
            ],
            "conclusion": ("cyclic", ("a", "b", "c", "d"))
        })
        
        rules.append({
            "name": "cyclic_eqangle",
            "premises": [
                ("cyclic", ("a", "b", "c", "d"))
            ],
            "conclusion": ("eqangle", ("a", "c", "a", "d", "b", "c", "b", "d"))
        })
        
        # Midpoint rules
        rules.append({
            "name": "midpoint_cong",
            "premises": [
                ("midpoint", ("m", "a", "b"))
            ],
            "conclusion": ("cong", ("m", "a", "m", "b"))
        })
        
        rules.append({
            "name": "midpoint_coll",
            "premises": [
                ("midpoint", ("m", "a", "b"))
            ],
            "conclusion": ("coll", ("m", "a", "b"))
        })
        
        rules.append({
            "name": "midpoints_para",
            "premises": [
                ("midpoint", ("m", "a", "b")),
                ("midpoint", ("n", "a", "c"))
            ],
            "conclusion": ("para", ("m", "n", "b", "c"))
        })
        
        # Perpendicular from midpoint
        rules.append({
            "name": "midp_perp_cong",
            "premises": [
                ("midpoint", ("m", "a", "b")),
                ("perp", ("o", "m", "a", "b"))
            ],
            "conclusion": ("cong", ("o", "a", "o", "b"))
        })
        
        # Collinearity rules
        rules.append({
            "name": "coll_trans",
            "premises": [
                ("coll", ("a", "b", "c")),
                ("coll", ("b", "c", "d"))
            ],
            "conclusion": ("coll", ("a", "b", "d"))
        })
        
        return rules
    
    def add_fact(self, predicate: str, args: tuple) -> bool:
        """Add a fact to the knowledge base."""
        fact = GeometryFact(predicate, args)
        if fact not in self.facts:
            self.facts.add(fact)
            return True
        return False
    
    def add_facts_from_parsed(self, statements: list):
        """Add facts from parsed problem statements."""
        for stmt in statements:
            pred = stmt.get("predicate", "")
            args = tuple(stmt.get("args", []))
            self.add_fact(pred, args)
    
    def parse_on_line(self, args: List[str]) -> List[Tuple]:
        """Parse on_line relation."""
        if len(args) >= 3:
            return [("on_line", tuple(args[:3]))]
        return []
    
    def parse_on_circle(self, args: List[str]) -> List[Tuple]:
        """Parse on_circle relation."""
        if len(args) >= 3:
            return [("on_circle", tuple(args[:3]))]
        return []
    
    def parse_cong(self, args: List[str]) -> List[Tuple]:
        """Parse congruence relation."""
        if len(args) >= 4:
            return [("cong", tuple(args[:4]))]
        return []
    
    def parse_perp(self, args: List[str]) -> List[Tuple]:
        """Parse perpendicularity relation."""
        if len(args) >= 4:
            return [("perp", tuple(args[:4]))]
        return []
    
    def parse_para(self, args: List[str]) -> List[Tuple]:
        """Parse parallel relation."""
        if len(args) >= 4:
            return [("para", tuple(args[:4]))]
        return []
    
    def parse_eqangle(self, args: List[str]) -> List[Tuple]:
        """Parse equal angles relation."""
        if len(args) >= 8:
            return [("eqangle", tuple(args[:8]))]
        return []
    
    def parse_cyclic(self, args: List[str]) -> List[Tuple]:
        """Parse cyclic quadrilateral relation."""
        if len(args) >= 4:
            return [("cyclic", tuple(args[:4]))]
        return []
    
    def parse_coll(self, args: List[str]) -> List[Tuple]:
        """Parse collinearity relation."""
        if len(args) >= 3:
            return [("coll", tuple(args[:3]))]
        return []
    
    def parse_midpoint(self, args: List[str]) -> List[Tuple]:
        """Parse midpoint relation."""
        if len(args) >= 3:
            return [("midpoint", tuple(args[:3]))]
        return []
    
    def parse_triangle(self, args: List[str]) -> List[Tuple]:
        """Parse triangle relation."""
        if len(args) >= 3:
            return [("triangle", tuple(args[:3]))]
        return []
    
    def parse_foot(self, args: List[str]) -> List[Tuple]:
        """Parse foot of perpendicular relation."""
        if len(args) >= 4:
            return [
                ("foot", tuple(args[:4])),
                ("perp", (args[0], args[1], args[2], args[3])),
                ("coll", (args[0], args[2], args[3]))
            ]
        return []
    
    def parse_orthocenter(self, args: List[str]) -> List[Tuple]:
        """Parse orthocenter relation."""
        if len(args) >= 4:
            return [
                ("orthocenter", tuple(args[:4])),
                ("perp", (args[0], args[1], args[2], args[3])),
                ("perp", (args[0], args[2], args[1], args[3])),
                ("perp", (args[0], args[3], args[1], args[2]))
            ]
        return []
    
    def parse_incenter(self, args: List[str]) -> List[Tuple]:
        """Parse incenter relation."""
        if len(args) >= 4:
            return [
                ("incenter", tuple(args[:4])),
                ("cong", (args[0], args[1], args[0], args[2])),
                ("cong", (args[0], args[2], args[0], args[3]))
            ]
        return []
    
    def parse_circle(self, args: List[str]) -> List[Tuple]:
        """Parse circle relation."""
        if len(args) >= 4:
            return [
                ("circle", tuple(args[:4])),
                ("cong", (args[0], args[1], args[0], args[2])),
                ("cong", (args[0], args[2], args[0], args[3]))
            ]
        return []
    
    def forward_chain(self, max_iterations: int = 100) -> int:
        """Perform forward chaining to derive new facts."""
        initial_count = len(self.facts)
        
        for iteration in range(max_iterations):
            new_facts = []
            
            for rule in self.rules:
                # Try to match premises
                if self._match_rule(rule):
                    # Add conclusion
                    conclusion = rule["conclusion"]
                    if self.add_fact(conclusion[0], conclusion[1]):
                        new_facts.append(conclusion)
                        self.derivation_trace.append({
                            "rule": rule["name"],
                            "conclusion": conclusion
                        })
            
            if not new_facts:
                break
        
        return len(self.facts) - initial_count
    
    def _match_rule(self, rule: dict) -> bool:
        """Check if a rule's premises match known facts."""
        for pred, args in rule["premises"]:
            if not any(f.predicate == pred and f.args == args for f in self.facts):
                return False
        return True
    
    def check_conclusion(self, conclusion_pred: str, conclusion_args: tuple) -> bool:
        """Check if a conclusion can be derived."""
        return GeometryFact(conclusion_pred, conclusion_args) in self.facts
    
    def solve_problem(self, problem: dict) -> dict:
        """Solve a geometry problem."""
        # Reset for new problem
        self.facts = set()
        self.derivation_trace = []
        
        # Parse and add facts from problem
        premises = problem.get("premises", [])
        for premise in premises:
            pred = premise.get("predicate", "")
            args = tuple(premise.get("args", []))
            
            # Parse special relations
            if pred == "on_line":
                for fact in self.parse_on_line(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "on_circle":
                for fact in self.parse_on_circle(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "cong":
                for fact in self.parse_cong(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "perp":
                for fact in self.parse_perp(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "para":
                for fact in self.parse_para(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "eqangle":
                for fact in self.parse_eqangle(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "cyclic":
                for fact in self.parse_cyclic(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "coll":
                for fact in self.parse_coll(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "midpoint":
                for fact in self.parse_midpoint(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "triangle":
                for fact in self.parse_triangle(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "foot":
                for fact in self.parse_foot(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "orthocenter":
                for fact in self.parse_orthocenter(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "incenter":
                for fact in self.parse_incenter(args):
                    self.add_fact(fact[0], fact[1])
            elif pred == "circle":
                for fact in self.parse_circle(args):
                    self.add_fact(fact[0], fact[1])
            else:
                self.add_fact(pred, args)
        
        # Forward chain
        derived_count = self.forward_chain()
        
        # Check conclusion
        conclusion = problem.get("conclusion", {})
        conclusion_pred = conclusion.get("predicate", "")
        conclusion_args = tuple(conclusion.get("args", []))
        
        solved = self.check_conclusion(conclusion_pred, conclusion_args)
        
        return {
            "solved": solved,
            "derived_facts": len(self.facts),
            "new_facts_derived": derived_count,
            "trace": self.derivation_trace
        }


def create_sample_problem():
    """Create a sample problem for testing."""
    return {
        "name": "sample",
        "premises": [
            {"predicate": "perp", "args": ["A", "B", "C", "D"]},
            {"predicate": "perp", "args": ["C", "D", "E", "F"]},
            {"predicate": "ncoll", "args": ["A", "B", "E"]}
        ],
        "conclusion": {"predicate": "para", "args": ["A", "B", "E", "F"]}
    }


if __name__ == "__main__":
    solver = GeometrySolver()
    
    # Test on sample problem
    problem = create_sample_problem()
    result = solver.solve_problem(problem)
    
    print("Problem: sample")
    print(f"Solved: {result['solved']}")
    print(f"Derived facts: {result['derived_facts']}")
    print(f"New facts: {result['new_facts_derived']}")
    print(f"Trace: {result['trace']}")