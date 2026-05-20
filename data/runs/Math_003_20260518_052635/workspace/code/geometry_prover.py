#!/usr/bin/env python3
"""
Simple forward-chaining theorem prover for Euclidean geometry.
Based on the rules defined in rules.txt.
"""

from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
from collections import defaultdict
import itertools


@dataclass
class Fact:
    """Represents a geometric fact."""
    predicate: str
    args: tuple
    
    def __hash__(self):
        return hash((self.predicate, self.args))
    
    def __eq__(self, other):
        return self.predicate == other.predicate and self.args == other.args
    
    def __repr__(self):
        return f"{self.predicate}({', '.join(self.args)})"


@dataclass
class Rule:
    """Represents an inference rule."""
    premises: List[Fact]
    conclusion: Fact
    name: str


class GeometryProver:
    """Forward-chaining theorem prover for geometry."""
    
    def __init__(self):
        self.facts = set()
        self.rules = self._define_rules()
        self.inference_history = []
    
    def _define_rules(self) -> List[Rule]:
        """Define geometric inference rules from rules.txt."""
        rules = []
        
        # Rule 1: perpendicular transitivity -> parallel
        # perp A B C D, perp C D E F, ncoll A B E => para A B E F
        rules.append(Rule(
            premises=[
                Fact("perp", ("A", "B", "C", "D")),
                Fact("perp", ("C", "D", "E", "F")),
                Fact("ncoll", ("A", "B", "E"))
            ],
            conclusion=Fact("para", ("A", "B", "E", "F")),
            name="perp_perp_para"
        ))
        
        # Rule 2: concyclic from circle
        # cong O A O B, cong O B O C, cong O C O D => cyclic A B C D
        rules.append(Rule(
            premises=[
                Fact("cong", ("O", "A", "O", "B")),
                Fact("cong", ("O", "B", "O", "C")),
                Fact("cong", ("O", "C", "O", "D"))
            ],
            conclusion=Fact("cyclic", ("A", "B", "C", "D")),
            name="circle_cyclic"
        ))
        
        # Rule 3: parallel from equal angles
        # eqangle A B P Q C D P Q => para A B C D
        rules.append(Rule(
            premises=[
                Fact("eqangle", ("A", "B", "P", "Q", "C", "D", "P", "Q"))
            ],
            conclusion=Fact("para", ("A", "B", "C", "D")),
            name="eqangle_para"
        ))
        
        # Rule 4: cyclic implies equal angles
        # cyclic A B P Q => eqangle P A P B Q A Q B
        rules.append(Rule(
            premises=[
                Fact("cyclic", ("A", "B", "P", "Q"))
            ],
            conclusion=Fact("eqangle", ("P", "A", "P", "B", "Q", "A", "Q", "B")),
            name="cyclic_eqangle"
        ))
        
        # Rule 5: midpoint and parallel
        # midp E A B, midp F A C => para E F B C
        rules.append(Rule(
            premises=[
                Fact("midp", ("E", "A", "B")),
                Fact("midp", ("F", "A", "C"))
            ],
            conclusion=Fact("para", ("E", "F", "B", "C")),
            name="midpoints_para"
        ))
        
        # Rule 6: perpendicular and parallel -> perpendicular
        # perp A B C D, perp E F G H, npara A B E F => eqangle A B E F C D G H
        rules.append(Rule(
            premises=[
                Fact("perp", ("A", "B", "C", "D")),
                Fact("perp", ("E", "F", "G", "H")),
                Fact("npara", ("A", "B", "E", "F"))
            ],
            conclusion=Fact("eqangle", ("A", "B", "E", "F", "C", "D", "G", "H")),
            name="perp_perp_eqangle"
        ))
        
        # Rule 7: equal angles transitivity
        # eqangle a b c d m n p q, eqangle c d e f p q r u => eqangle a b e f m n r u
        rules.append(Rule(
            premises=[
                Fact("eqangle", ("a", "b", "c", "d", "m", "n", "p", "q")),
                Fact("eqangle", ("c", "d", "e", "f", "p", "q", "r", "u"))
            ],
            conclusion=Fact("eqangle", ("a", "b", "e", "f", "m", "n", "r", "u")),
            name="eqangle_trans"
        ))
        
        # Rule 8: congruence from circle and perpendicular
        # perp A B B C, midp M A C => cong A M B M
        rules.append(Rule(
            premises=[
                Fact("perp", ("A", "B", "B", "C")),
                Fact("midp", ("M", "A", "C"))
            ],
            conclusion=Fact("cong", ("A", "M", "B", "M")),
            name="perp_midp_cong"
        ))
        
        # Rule 9: concyclic from angles (inverse)
        # eqangle6 P A P B Q A Q B, ncoll P Q A B => cyclic A B P Q
        rules.append(Rule(
            premises=[
                Fact("eqangle6", ("P", "A", "P", "B", "Q", "A", "Q", "B")),
                Fact("ncoll", ("P", "Q", "A"))
            ],
            conclusion=Fact("cyclic", ("A", "B", "P", "Q")),
            name="eqangle_cyclic"
        ))
        
        # Rule 10: collinearity from parallel
        # para A B A C => coll A B C
        rules.append(Rule(
            premises=[
                Fact("para", ("A", "B", "A", "C"))
            ],
            conclusion=Fact("coll", ("A", "B", "C")),
            name="para_coll"
        ))
        
        # Rule 11: perpendicular from equal distances and concyclic
        # cong A P B P, cong A Q B Q, cyclic A B P Q => perp P A A Q
        rules.append(Rule(
            premises=[
                Fact("cong", ("A", "P", "B", "P")),
                Fact("cong", ("A", "Q", "B", "Q")),
                Fact("cyclic", ("A", "B", "P", "Q"))
            ],
            conclusion=Fact("perp", ("P", "A", "A", "Q")),
            name="cong_cyclic_perp"
        ))
        
        # Rule 12: midpoint and perpendicular -> congruence
        # midp M A B, perp O M A B => cong O A O B
        rules.append(Rule(
            premises=[
                Fact("midp", ("M", "A", "B")),
                Fact("perp", ("O", "M", "A", "B"))
            ],
            conclusion=Fact("cong", ("O", "A", "O", "B")),
            name="midp_perp_cong"
        ))
        
        # Rule 13: perpendicular from equal angles and perpendicular
        # eqangle A B P Q C D U V, perp P Q U V => perp A B C D
        rules.append(Rule(
            premises=[
                Fact("eqangle", ("A", "B", "P", "Q", "C", "D", "U", "V")),
                Fact("perp", ("P", "Q", "U", "V"))
            ],
            conclusion=Fact("perp", ("A", "B", "C", "D")),
            name="eqangle_perp_perp"
        ))
        
        # Rule 14: concyclic from parallel and equal angles
        # cyclic A B C D, para A B C D => eqangle A D C D C D C B
        rules.append(Rule(
            premises=[
                Fact("cyclic", ("A", "B", "C", "D")),
                Fact("para", ("A", "B", "C", "D"))
            ],
            conclusion=Fact("eqangle", ("A", "D", "C", "D", "C", "D", "C", "B")),
            name="cyclic_para_eqangle"
        ))
        
        # Rule 15: congruence from perpendicular and midpoint
        # perp A B B C, midp M A C => cong A M B M
        rules.append(Rule(
            premises=[
                Fact("perp", ("A", "B", "B", "C")),
                Fact("midp", ("M", "A", "C"))
            ],
            conclusion=Fact("cong", ("A", "M", "B", "M")),
            name="perp_midp_cong2"
        ))
        
        return rules
    
    def add_fact(self, fact: Fact):
        """Add a fact to the knowledge base."""
        if fact not in self.facts:
            self.facts.add(fact)
            return True
        return False
    
    def add_facts_from_statement(self, predicate: str, args: tuple):
        """Add facts from a parsed statement."""
        fact = Fact(predicate, args)
        return self.add_fact(fact)
    
    def unify_fact(self, pattern: Fact, fact: Fact, bindings: dict) -> Optional[dict]:
        """Try to unify a pattern with a fact, returning bindings."""
        if pattern.predicate != fact.predicate:
            return None
        
        if len(pattern.args) != len(fact.args):
            return None
        
        new_bindings = bindings.copy()
        
        for p_arg, f_arg in zip(pattern.args, fact.args):
            if p_arg.isupper() and len(p_arg) == 1:  # Variable
                if p_arg in new_bindings:
                    if new_bindings[p_arg] != f_arg:
                        return None
                else:
                    new_bindings[p_arg] = f_arg
            elif p_arg != f_arg:  # Constants must match
                return None
        
        return new_bindings
    
    def apply_rule(self, rule: Rule) -> List[Tuple[Fact, dict]]:
        """Try to apply a rule, returning new facts and their derivations."""
        # Generate all possible substitutions
        # For simplicity, we'll try to match each premise against known facts
        
        possible_matches = []
        for premise in rule.premises:
            matches = []
            for fact in self.facts:
                bindings = self.unify_fact(premise, fact, {})
                if bindings is not None:
                    matches.append((fact, bindings))
            possible_matches.append(matches)
        
        # Try all combinations
        results = []
        if all(matches for matches in possible_matches):
            for combination in itertools.product(*possible_matches):
                # Merge all bindings
                merged_bindings = {}
                valid = True
                
                for fact, bindings in combination:
                    for key, value in bindings.items():
                        if key in merged_bindings:
                            if merged_bindings[key] != value:
                                valid = False
                                break
                        else:
                            merged_bindings[key] = value
                
                if valid:
                    # Apply bindings to conclusion
                    conclusion_args = tuple(
                        merged_bindings.get(arg, arg) for arg in rule.conclusion.args
                    )
                    new_fact = Fact(rule.conclusion.predicate, conclusion_args)
                    
                    if new_fact not in self.facts:
                        results.append((new_fact, merged_bindings))
        
        return results
    
    def forward_chain(self, max_iterations: int = 100) -> List[Fact]:
        """Perform forward chaining to derive new facts."""
        new_facts = []
        
        for iteration in range(max_iterations):
            derived_this_iteration = []
            
            for rule in self.rules:
                results = self.apply_rule(rule)
                for new_fact, bindings in results:
                    if self.add_fact(new_fact):
                        derived_this_iteration.append(new_fact)
                        new_facts.append(new_fact)
            
            if not derived_this_iteration:
                break
        
        return new_facts
    
    def check_conclusion(self, conclusion: Fact) -> bool:
        """Check if the conclusion can be derived."""
        return conclusion in self.facts
    
    def get_proof_trace(self) -> List[str]:
        """Get a trace of the proof."""
        return self.inference_history


def test_prover():
    """Test the prover on a simple problem."""
    prover = GeometryProver()
    
    # Add some basic facts
    prover.add_fact(Fact("ncoll", ("A", "B", "C")))
    prover.add_fact(Fact("perp", ("A", "B", "C", "D")))
    prover.add_fact(Fact("perp", ("C", "D", "E", "F")))
    prover.add_fact(Fact("ncoll", ("A", "B", "E")))
    
    # Forward chain
    new_facts = prover.forward_chain()
    
    print("Derived facts:")
    for fact in new_facts:
        print(f"  {fact}")
    
    # Check if we can derive para A B E F
    target = Fact("para", ("A", "B", "E", "F"))
    print(f"\nTarget {target} derived: {prover.check_conclusion(target)}")


if __name__ == "__main__":
    test_prover()