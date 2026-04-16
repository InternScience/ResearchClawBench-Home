import re
from itertools import permutations

def parse_fact(fact_str):
    parts = fact_str.split()
    return parts[0], parts[1:]

def match_rule(antecedents, facts):
    # This is a complex unification problem.
    pass

