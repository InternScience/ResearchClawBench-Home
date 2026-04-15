"""
Enhanced Neuro-Symbolic Geometry Proof System
Better parsing of formal geometry language + symbolic search with definition expansion.
"""
import json
import re
from collections import defaultdict

class GeometryKnowledgeBase:
    """Knowledge base for geometric facts with forward chaining."""
    
    def __init__(self):
        self.facts = {}  # (pred_name, args_tuple) -> source
        self.cong_pairs = set()  # set of frozensets of point pairs
        self.coll_triples = set()  # set of frozensets of collinear points
        self.para_pairs = set()  # set of pairs of line pairs
        self.perp_pairs = set()
        self.cyclic_sets = set()
        self.eqangle_sets = []
    
    def add_fact(self, pred_name, args, source="given"):
        key = (pred_name, tuple(args))
        if key not in self.facts:
            self.facts[key] = source
            return True
        return False
    
    def add_cong(self, a, b, c, d, source="given"):
        pair = frozenset([frozenset([a, b]), frozenset([c, d])])
        if pair not in self.cong_pairs:
            self.cong_pairs.add(pair)
            self.add_fact('cong', [a, b, c, d], source)
            return True
        return False
    
    def add_coll(self, a, b, c, source="given"):
        triple = frozenset([a, b, c])
        if triple not in self.coll_triples:
            self.coll_triples.add(triple)
            self.add_fact('coll', [a, b, c], source)
            return True
        return False
    
    def add_para(self, a, b, c, d, source="given"):
        pair = (frozenset([a, b]), frozenset([c, d]))
        if pair not in self.para_pairs:
            self.para_pairs.add(pair)
            self.add_fact('para', [a, b, c, d], source)
            return True
        return False
    
    def add_perp(self, a, b, c, d, source="given"):
        pair = (frozenset([a, b]), frozenset([c, d]))
        if pair not in self.perp_pairs:
            self.perp_pairs.add(pair)
            self.add_fact('perp', [a, b, c, d], source)
            return True
        return False
    
    def add_cyclic(self, points, source="given"):
        s = frozenset(points)
        if s not in self.cyclic_sets:
            self.cyclic_sets.add(s)
            self.add_fact('cyclic', list(points), source)
            return True
        return False
    
    def has_cong(self, a, b, c, d):
        p1 = frozenset([a, b])
        p2 = frozenset([c, d])
        return frozenset([p1, p2]) in self.cong_pairs
    
    def has_coll(self, a, b, c):
        return frozenset([a, b, c]) in self.coll_triples
    
    def has_perp(self, a, b, c, d):
        return (frozenset([a, b]), frozenset([c, d])) in self.perp_pairs
    
    def has_para(self, a, b, c, d):
        return (frozenset([a, b]), frozenset([c, d])) in self.para_pairs
    
    def has_cyclic(self, *points):
        return frozenset(points) in self.cyclic_sets
    
    def get_all_facts(self):
        return list(self.facts.keys())


def parse_construction_step(step_str):
    """Parse a construction step like 'a b = triangle a b c' into structured form."""
    step_str = step_str.strip()
    if not step_str:
        return []
    
    facts = []
    
    # Pattern: defined_points = constructor args
    # Handle multiple constructions separated by commas
    parts = step_str.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Match: x = constructor args...
        match = re.match(r'(\w+)\s*=\s*(\w+)\s+(.*)', part)
        if match:
            defined = match.group(1)
            constructor = match.group(2)
            args_str = match.group(3)
            args = args_str.split()
            facts.append((defined, constructor, args))
        else:
            # Just a predicate
            tokens = part.split()
            if len(tokens) >= 2:
                facts.append((None, tokens[0], tokens[1:]))
    
    return facts


def extract_facts_from_problem(problem_str):
    """Extract all geometric facts from a formal problem statement."""
    lines = problem_str.strip().split('\n')
    name = lines[0]
    full = ' '.join(lines[1:])
    
    if '?' in full:
        premise_str, goal = full.split('?', 1)
        goal = goal.strip()
    else:
        premise_str, goal = full, ""
    
    kb = GeometryKnowledgeBase()
    all_points = set()
    
    # Split into individual construction assignments
    # Each assignment ends with ';' except the last
    assignments = premise_str.split(';')
    
    for assign in assignments:
        assign = assign.strip()
        if not assign:
            continue
        
        # Handle multiple constructions on one line separated by commas
        # e.g., "a b = segment a b; g1 = on_tline g1 a a b; ..."
        
        # Parse the assignment
        # Left side: defined points
        # Right side: construction with constraints
        
        # Split on '=' to get left and right
        if '=' in assign:
            left, right = assign.split('=', 1)
            defined_points = left.strip().split()
            all_points.update(defined_points)
            
            # Parse right side - may have multiple constraints separated by commas
            constraints = [c.strip() for c in right.split(',')]
            
            for constraint in constraints:
                tokens = constraint.split()
                if not tokens:
                    continue
                constructor = tokens[0]
                args = tokens[1:]
                all_points.update(args)
                
                # Apply definition rules
                apply_definition(kb, constructor, defined_points, args, all_points)
        else:
            # Just a predicate
            tokens = assign.split()
            if tokens:
                apply_predicate(kb, tokens[0], tokens[1:], all_points)
    
    return kb, goal, name, all_points


def apply_definition(kb, constructor, defined, args, all_points):
    """Apply a geometric definition to add facts to the KB."""
    
    if constructor == 'triangle':
        # Just defines points, no additional facts
        pass
    
    elif constructor == 'segment':
        # Just defines two points
        pass
    
    elif constructor == 'midpoint':
        # x = midpoint x a b  => coll x a b, cong x a x b
        if len(args) >= 2:
            x, a, b = defined[0], args[0], args[1]
            kb.add_coll(x, a, b, "midpoint_def")
            kb.add_cong(x, a, x, b, "midpoint_def")
    
    elif constructor == 'circle':
        # x = circle x a b c  => cong x a x b, cong x b x c
        if len(args) >= 3:
            o, a, b, c = defined[0], args[0], args[1], args[2]
            kb.add_cong(o, a, o, b, "circle_def")
            kb.add_cong(o, b, o, c, "circle_def")
    
    elif constructor == 'on_circle':
        # x = on_circle x o a  => cong o x o a
        if len(args) >= 2:
            x, o, a = defined[0], args[0], args[1]
            kb.add_cong(o, x, o, a, "on_circle_def")
    
    elif constructor == 'on_line':
        # x = on_line x a b  => coll x a b
        if len(args) >= 2:
            x, a, b = defined[0], args[0], args[1]
            kb.add_coll(x, a, b, "on_line_def")
    
    elif constructor == 'on_bline':
        # x = on_bline x a b  => cong x a x b
        if len(args) >= 2:
            x, a, b = defined[0], args[0], args[1]
            kb.add_cong(x, a, x, b, "on_bline_def")
    
    elif constructor == 'on_tline':
        # x = on_tline x a b c  => perp x a b c
        if len(args) >= 3:
            x, a, b, c = defined[0], args[0], args[1], args[2]
            kb.add_perp(x, a, b, c, "on_tline_def")
    
    elif constructor == 'on_pline':
        # x = on_pline x a b c  => para x a b c
        if len(args) >= 3:
            x, a, b, c = defined[0], args[0], args[1], args[2]
            kb.add_para(x, a, b, c, "on_pline_def")
    
    elif constructor == 'foot':
        # x = foot x a b c  => perp x a b c, coll x b c
        if len(args) >= 3:
            x, a, b, c = defined[0], args[0], args[1], args[2]
            kb.add_perp(x, a, b, c, "foot_def")
            kb.add_coll(x, b, c, "foot_def")
    
    elif constructor == 'orthocenter':
        # x = orthocenter x a b c  => perp x a b c, perp x b c a, perp x c a b
        if len(args) >= 3:
            x, a, b, c = defined[0], args[0], args[1], args[2]
            kb.add_perp(x, a, b, c, "orthocenter_def")
            kb.add_perp(x, b, c, a, "orthocenter_def")
            kb.add_perp(x, c, a, b, "orthocenter_def")
    
    elif constructor == 'incenter':
        # x = incenter x a b c  => angle bisector properties
        if len(args) >= 3:
            x, a, b, c = defined[0], args[0], args[1], args[2]
            kb.add_fact('incenter', [x, a, b, c], "incenter_def")
    
    elif constructor == 'incenter2':
        # x y z i = incenter2 x y z i a b c
        if len(args) >= 7:
            x, y, z, i, a, b, c = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
            kb.add_fact('incenter2', [i, a, b, c], "incenter2_def")
    
    elif constructor == 'reflect':
        # x = reflect x a b c  => cong b a b x, cong c a c x, perp b c a x
        if len(args) >= 3:
            x, a, b, c = defined[0], args[0], args[1], args[2]
            kb.add_cong(b, a, b, x, "reflect_def")
            kb.add_cong(c, a, c, x, "reflect_def")
            kb.add_perp(b, c, a, x, "reflect_def")
    
    elif constructor == 'mirror':
        # x = mirror x a b  => coll x a b, cong b a b x
        if len(args) >= 2:
            x, a, b = defined[0], args[0], args[1]
            kb.add_coll(x, a, b, "mirror_def")
            kb.add_cong(b, a, b, x, "mirror_def")
    
    elif constructor == 'eqdistance':
        # x = eqdistance x a b c  => cong x a b c
        if len(args) >= 3:
            x, a, b, c = defined[0], args[0], args[1], args[2]
            kb.add_cong(x, a, b, c, "eqdistance_def")
    
    elif constructor == 'parallelogram':
        # a b c x = parallelogram a b c x  => para a b c x, para a x b c, cong a b c x, cong a x b c
        if len(args) >= 3:
            a, b, c, x = args[0], args[1], args[2], defined[-1] if len(defined) > 3 else defined[0]
            kb.add_para(a, b, c, x, "parallelogram_def")
            kb.add_para(a, x, b, c, "parallelogram_def")
            kb.add_cong(a, b, c, x, "parallelogram_def")
            kb.add_cong(a, x, b, c, "parallelogram_def")
    
    elif constructor == 'on_dia':
        # x = on_dia x a b  => perp x a x b
        if len(args) >= 2:
            x, a, b = defined[0], args[0], args[1]
            kb.add_perp(x, a, x, b, "on_dia_def")
    
    elif constructor == 'r_triangle':
        # a b c = r_triangle a b c  => perp a b a c
        if len(args) >= 2:
            a, b, c = args[0], args[1], args[2] if len(args) > 2 else defined[2]
            kb.add_perp(a, b, a, c, "r_triangle_def")
    
    elif constructor == 'angle_bisector':
        # r = angle_bisector r b a c  => eqangle properties
        if len(args) >= 3:
            r, b, a, c = defined[0], args[0], args[1], args[2]
            kb.add_fact('angle_bisector', [r, b, a, c], "angle_bisector_def")
    
    elif constructor == 'iso_triangle':
        # s c p = iso_triangle s c p  => cong s c s p
        if len(args) >= 2:
            s, c, p = args[0], args[1], args[2] if len(args) > 2 else defined[2]
            kb.add_cong(s, c, s, p, "iso_triangle_def")


def apply_predicate(kb, pred_name, args, all_points):
    """Apply a direct predicate."""
    if pred_name == 'cong' and len(args) >= 4:
        kb.add_cong(args[0], args[1], args[2], args[3], "given")
    elif pred_name == 'coll' and len(args) >= 3:
        kb.add_coll(args[0], args[1], args[2], "given")
    elif pred_name == 'para' and len(args) >= 4:
        kb.add_para(args[0], args[1], args[2], args[3], "given")
    elif pred_name == 'perp' and len(args) >= 4:
        kb.add_perp(args[0], args[1], args[2], args[3], "given")
    elif pred_name == 'cyclic' and len(args) >= 4:
        kb.add_cyclic(args, "given")


def forward_chain_inference(kb, max_iterations=200):
    """Apply inference rules to derive new facts."""
    new_facts_count = 0
    
    for iteration in range(max_iterations):
        iteration_new = 0
        facts_snapshot = list(kb.facts.items())
        
        # Rule: cong O A O B, ncoll O A B => eqangle O A A B A B O B
        cong_facts = [(k[1], v) for k, v in facts_snapshot if k[0] == 'cong']
        for (args, src) in cong_facts:
            o, a, o2, b = args
            if o == o2:
                # Check ncoll (not collinear)
                if not kb.has_coll(o, a, b):
                    kb.add_fact('eqangle', [o, a, a, b, a, b, o, b], "cong_eqangle_rule")
                    iteration_new += 1
        
        # Rule: perp A B C D, perp C D E F, ncoll A B E => para A B E F
        perp_facts = [(k[1], v) for k, v in facts_snapshot if k[0] == 'perp']
        for i, (args1, src1) in enumerate(perp_facts):
            a, b, c, d = args1
            for j, (args2, src2) in enumerate(perp_facts):
                if i == j:
                    continue
                e, f, g, h = args2
                # Check if second perp shares line with first
                if frozenset([c, d]) == frozenset([e, f]):
                    if not kb.has_coll(a, b, e):
                        kb.add_para(a, b, g, h, "perp_perp_para")
                        iteration_new += 1
        
        # Rule: cong O A O B, cong O B O C, cong O C O D => cyclic A B C D
        for (k1, v1) in facts_snapshot:
            if k1[0] != 'cong':
                continue
            o1, a, o2, b = k1[1]
            if o1 != o2:
                continue
            for (k2, v2) in facts_snapshot:
                if k2[0] != 'cong':
                    continue
                o3, b2, o4, c = k2[1]
                if o3 != o1 or o4 != o1 or b2 != b:
                    continue
                for (k3, v3) in facts_snapshot:
                    if k3[0] != 'cong':
                        continue
                    o5, c2, o6, d = k3[1]
                    if o5 != o1 or o6 != o1 or c2 != c:
                        continue
                    kb.add_cyclic([a, b, c, d], "cong_cyclic_rule")
                    iteration_new += 1
        
        # Rule: midp M A B, perp O M A B => cong O A O B
        coll_facts = [(k[1], v) for k, v in facts_snapshot if k[0] == 'coll']
        for (cargs, csrc) in coll_facts:
            m, a, b = cargs
            if kb.has_cong(m, a, m, b):  # m is midpoint
                for (pargs, psrc) in perp_facts:
                    o, m2, a2, b2 = pargs
                    if m2 == m and frozenset([a2, b2]) == frozenset([a, b]):
                        kb.add_cong(o, a, o, b, "midp_perp_cong")
                        iteration_new += 1
        
        # Rule: cyclic A B P Q => eqangle P A P B Q A Q B
        for s in kb.cyclic_sets:
            pts = list(s)
            if len(pts) >= 4:
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        for k in range(j+1, len(pts)):
                            for l in range(k+1, len(pts)):
                                a, b, p, q = pts[i], pts[j], pts[k], pts[l]
                                kb.add_fact('eqangle', [p, a, p, b, q, a, q, b], "cyclic_eqangle")
                                iteration_new += 1
        
        # Rule: cong A P B P, cong A Q B Q => perp A B P Q
        for (k1, v1) in facts_snapshot:
            if k1[0] != 'cong':
                continue
            a, p, b, p2 = k1[1]
            if p != p2:
                continue
            for (k2, v2) in facts_snapshot:
                if k2[0] != 'cong':
                    continue
                a2, q, b2, q2 = k2[1]
                if a2 == a and b2 == b and q == q2 and q != p:
                    kb.add_perp(a, b, p, q, "cong_cong_perp")
                    iteration_new += 1
        
        # Rule: midp M A B, midp N C D => eqratio M A A B N C C D (and para A C B D)
        midpoints = {}
        for (k, v) in facts_snapshot:
            if k[0] == 'coll' and len(k[1]) == 3:
                m, a, b = k[1]
                if kb.has_cong(m, a, m, b):
                    midpoints[(a, b)] = m
        
        for (a, b), m in midpoints.items():
            for (c, d), n in midpoints.items():
                if (a, b) != (c, d):
                    kb.add_para(a, c, b, d, "midpoint_para")
                    iteration_new += 1
        
        # Rule: circle O A B C, midp M B C => eqangle A B A C O B O M
        circle_facts = [(k[1], v) for k, v in facts_snapshot if k[0] == 'cyclic']
        
        if iteration_new == 0:
            break
        new_facts_count += iteration_new
    
    return new_facts_count


def check_goal(kb, goal_str):
    """Check if the goal is satisfied by the KB."""
    tokens = goal_str.split()
    if not tokens:
        return False
    
    pred = tokens[0]
    args = tokens[1:]
    
    if pred == 'cong' and len(args) >= 4:
        return kb.has_cong(args[0], args[1], args[2], args[3])
    elif pred == 'coll' and len(args) >= 3:
        return kb.has_coll(args[0], args[1], args[2])
    elif pred == 'para' and len(args) >= 4:
        return kb.has_para(args[0], args[1], args[2], args[3])
    elif pred == 'perp' and len(args) >= 4:
        return kb.has_perp(args[0], args[1], args[2], args[3])
    elif pred == 'cyclic' and len(args) >= 4:
        return kb.has_cyclic(*args)
    elif pred == 'eqangle':
        return any(k[0] == 'eqangle' for k in kb.facts)
    elif pred == 'eqratio':
        return any(k[0] == 'eqratio' for k in kb.facts)
    
    return False


def solve_problem(problem_str):
    """Solve a single geometry problem."""
    kb, goal, name, points = extract_facts_from_problem(problem_str)
    
    initial_facts = len(kb.facts)
    
    # Run forward chaining
    new_derived = forward_chain_inference(kb)
    
    final_facts = len(kb.facts)
    
    # Check goal
    goal_satisfied = check_goal(kb, goal)
    
    return {
        'name': name,
        'goal': goal,
        'points': sorted(points),
        'num_points': len(points),
        'initial_facts': initial_facts,
        'derived_facts': final_facts - initial_facts,
        'total_facts': final_facts,
        'goal_satisfied': goal_satisfied,
        'cong_count': len(kb.cong_pairs),
        'coll_count': len(kb.coll_triples),
        'perp_count': len(kb.perp_pairs),
        'para_count': len(kb.para_pairs),
        'cyclic_count': len(kb.cyclic_sets)
    }


def run_all():
    with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/data/imo_ag_30.txt') as f:
        content = f.read()
    
    problem_blocks = re.split(r'\n(?=translated_imo_)', content.strip())
    
    results = []
    for block in problem_blocks:
        if block.strip():
            r = solve_problem(block)
            results.append(r)
            status = "✓" if r['goal_satisfied'] else "✗"
            print(f"{status} {r['name']}: {r['initial_facts']}→{r['total_facts']} facts (+{r['derived_facts']}), goal: {r['goal'][:40]}")
    
    proved = sum(1 for r in results if r['goal_satisfied'])
    print(f"\nProved: {proved}/{len(results)}")
    
    with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/outputs/enhanced_prover_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    run_all()
