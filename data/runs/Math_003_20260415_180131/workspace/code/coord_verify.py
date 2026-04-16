#!/usr/bin/env python3
"""
Step 4b: Improved coordinate geometry verification engine.
Uses a more robust approach to assign coordinates by processing
construction statements sequentially with proper geometric computations.
"""

import json
import os
import math
import random
from collections import defaultdict

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_180131"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")

cos = math.cos
sin = math.sin
sqrt = math.sqrt
PI = math.pi

def dist(p1, p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def midpoint(p1, p2):
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def cross(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]

def normalize(v):
    d = sqrt(v[0]**2 + v[1]**2)
    if d < 1e-12:
        return (0, 0)
    return (v[0]/d, v[1]/d)

def line_intersect(a, b, c, d):
    """Find intersection of line AB and line CD."""
    ab = (b[0]-a[0], b[1]-a[1])
    cd = (d[0]-c[0], d[1]-c[1])
    denom = cross(ab, cd)
    if abs(denom) < 1e-10:
        return None  # Parallel or coincident
    ac = (c[0]-a[0], c[1]-a[1])
    t = cross(ac, cd) / denom
    return (a[0] + t*ab[0], a[1] + t*ab[1])

def project_point_to_line(p, a, b):
    """Project point P onto line AB."""
    ab = (b[0]-a[0], b[1]-a[1])
    ap = (p[0]-a[0], p[1]-a[1])
    t = dot(ap, ab) / dot(ab, ab)
    return (a[0] + t*ab[0], a[1] + t*ab[1])

def reflect_point_over_line(p, a, b):
    """Reflect point P over line AB."""
    proj = project_point_to_line(p, a, b)
    return (2*proj[0] - p[0], 2*proj[1] - p[1])

def circumcenter(a, b, c):
    """Compute circumcenter of triangle ABC."""
    ax, ay = a; bx, by = b; cx, cy = c
    D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-10:
        return None
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    return (ux, uy)

def incenter(a, b, c):
    """Compute incenter of triangle ABC."""
    ab_len = dist(a, b)
    bc_len = dist(b, c)
    ca_len = dist(c, a)
    s = ab_len + bc_len + ca_len
    if s < 1e-10:
        return None
    return ((bc_len*a[0]+ca_len*b[0]+ab_len*c[0])/s,
            (bc_len*a[1]+ca_len*b[1]+ab_len*c[1])/s)

def orthocenter_compute(a, b, c):
    """Compute orthocenter of triangle ABC."""
    # Altitude from A perpendicular to BC
    # Altitude from B perpendicular to AC
    bc = (c[0]-b[0], c[1]-b[1])
    ac = (c[0]-a[0], c[1]-a[1])
    alt_a_dir = (-bc[1], bc[0])  # perp to BC
    alt_b_dir = (-ac[1], ac[0])  # perp to AC
    # Intersection of altitude from A and altitude from B
    result = line_intersect(a, (a[0]+alt_a_dir[0], a[1]+alt_a_dir[1]),
                            b, (b[0]+alt_b_dir[0], b[1]+alt_b_dir[1]))
    return result


def assign_coords_for_problem(problem, seed=0):
    """
    Assign coordinates to all points in a problem by processing
    construction statements sequentially.
    Returns dict of {point_name: (x, y)} or None if failed.
    """
    random.seed(seed)
    coords = {}
    
    constructions = problem['constructions']
    
    for constr in constructions:
        defined_points = constr['defined_points']
        constraints = constr['constraints']
        
        for constraint in constraints:
            pred = constraint['predicate']
            args = constraint['args']
            
            try:
                # ─── Free/segment/triangle: assign random coordinates ───
                if pred == 'segment':
                    a, b = args[0], args[1]
                    if a not in coords:
                        coords[a] = (random.uniform(-5, 5), random.uniform(-5, 5))
                    if b not in coords:
                        dx, dy = random.uniform(2, 6), random.uniform(-3, 3)
                        coords[b] = (coords[a][0]+dx, coords[a][1]+dy)
                
                elif pred == 'triangle':
                    for p in args[:3]:
                        if p not in coords:
                            coords[p] = (random.uniform(-5, 5), random.uniform(-5, 5))
                
                elif pred == 'free':
                    if args[0] not in coords:
                        coords[args[0]] = (random.uniform(-5, 5), random.uniform(-5, 5))
                
                elif pred == 'iso_triangle':
                    s, c, p = args[0], args[1], args[2]
                    if s not in coords:
                        coords[s] = (random.uniform(-5, 5), random.uniform(-5, 5))
                    if c not in coords:
                        coords[c] = (coords[s][0]+random.uniform(2, 5), coords[s][1]+random.uniform(-2, 2))
                    sc_len = dist(coords[s], coords[c])
                    angle = random.uniform(0.3, 2.8)
                    coords[p] = (coords[s][0]+sc_len*cos(angle), coords[s][1]+sc_len*sin(angle))
                
                elif pred == 'r_triangle':
                    c, a, b = args[0], args[1], args[2]
                    if c not in coords:
                        coords[c] = (random.uniform(-5, 5), random.uniform(-5, 5))
                    if a not in coords:
                        coords[a] = (coords[c][0]+random.uniform(2, 5), coords[c][1])
                    ca = (coords[a][0]-coords[c][0], coords[a][1]-coords[c][1])
                    t = random.uniform(1, 4)
                    coords[b] = (coords[c][0]-t*ca[1], coords[c][1]+t*ca[0])
                
                # ─── Derived point constructions ───
                elif pred == 'midpoint':
                    m, a, b = args[0], args[1], args[2]
                    if a in coords and b in coords:
                        coords[m] = midpoint(coords[a], coords[b])
                
                elif pred == 'orthocenter':
                    h, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        result = orthocenter_compute(coords[a], coords[b], coords[c])
                        if result:
                            coords[h] = result
                        else:
                            return None
                
                elif pred == 'foot':
                    h1, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        coords[h1] = project_point_to_line(coords[a], coords[b], coords[c])
                
                elif pred == 'incenter':
                    i, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        result = incenter(coords[a], coords[b], coords[c])
                        if result:
                            coords[i] = result
                
                elif pred == 'incenter2':
                    if len(args) >= 7:
                        t1, t2, t3, i, a, b, c = args[:7]
                        if a in coords and b in coords and c in coords:
                            ic = incenter(coords[a], coords[b], coords[c])
                            if ic:
                                coords[i] = ic
                                # Touch points
                                coords[t1] = project_point_to_line(ic, coords[b], coords[c])
                                coords[t2] = project_point_to_line(ic, coords[c], coords[a])
                                coords[t3] = project_point_to_line(ic, coords[a], coords[b])
                
                elif pred == 'excenter2':
                    if len(args) >= 7:
                        m, l, k, j, a, b, c = args[:7]
                        # Excenter opposite A: intersection of external bisectors at B and C
                        if a in coords and b in coords and c in coords:
                            # Compute using formula: I_A = (-a*A + b*B + c*C) / (-a+b+c)
                            ab_len = dist(coords[a], coords[b])
                            bc_len = dist(coords[b], coords[c])
                            ca_len = dist(coords[c], coords[a])
                            s = -ab_len + bc_len + ca_len
                            if abs(s) > 1e-10:
                                coords[j] = ((-ab_len*coords[a][0]+bc_len*coords[b][0]+ca_len*coords[c][0])/s,
                                             (-ab_len*coords[a][1]+bc_len*coords[b][1]+ca_len*coords[c][1])/s)
                                coords[m] = project_point_to_line(coords[j], coords[b], coords[c])
                                coords[l] = project_point_to_line(coords[j], coords[c], coords[a])
                                coords[k] = project_point_to_line(coords[j], coords[a], coords[b])
                
                elif pred == 'circumcenter' or pred == 'circle':
                    if len(args) >= 4:
                        o, a, b, c = args[0], args[1], args[2], args[3]
                        if a in coords and b in coords and c in coords:
                            result = circumcenter(coords[a], coords[b], coords[c])
                            if result:
                                coords[o] = result
                            else:
                                return None
                
                elif pred == 'on_circle':
                    x, o, a = args[0], args[1], args[2]
                    if o in coords and a in coords:
                        r = dist(coords[o], coords[a])
                        angle = random.uniform(0, 2*PI)
                        coords[x] = (coords[o][0]+r*cos(angle), coords[o][1]+r*sin(angle))
                
                elif pred == 'on_line':
                    x, a, b = args[0], args[1], args[2]
                    if a in coords and b in coords:
                        t = random.uniform(-3, 4)
                        coords[x] = (coords[a][0]+t*(coords[b][0]-coords[a][0]),
                                     coords[a][1]+t*(coords[b][1]-coords[a][1]))
                
                elif pred == 'on_bline':
                    x, a, b = args[0], args[1], args[2]
                    if a in coords and b in coords:
                        mid = midpoint(coords[a], coords[b])
                        ab_perp = (-(coords[b][1]-coords[a][1]), coords[b][0]-coords[a][0])
                        t = random.uniform(-3, 3)
                        coords[x] = (mid[0]+t*ab_perp[0], mid[1]+t*ab_perp[1])
                
                elif pred == 'on_pline':
                    x, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        bc = (coords[c][0]-coords[b][0], coords[c][1]-coords[b][1])
                        t = random.uniform(-3, 4)
                        coords[x] = (coords[a][0]+t*bc[0], coords[a][1]+t*bc[1])
                
                elif pred == 'on_tline':
                    x, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        bc = (coords[c][0]-coords[b][0], coords[c][1]-coords[b][1])
                        bc_perp = (-bc[1], bc[0])
                        t = random.uniform(-3, 4)
                        coords[x] = (coords[a][0]+t*bc_perp[0], coords[a][1]+t*bc_perp[1])
                
                elif pred == 'on_dia':
                    x, a, b = args[0], args[1], args[2]
                    if a in coords and b in coords:
                        mid = midpoint(coords[a], coords[b])
                        r = dist(coords[a], coords[b])/2
                        angle = random.uniform(0, 2*PI)
                        coords[x] = (mid[0]+r*cos(angle), mid[1]+r*sin(angle))
                
                elif pred == 'mirror':
                    x, a, b = args[0], args[1], args[2]
                    if a in coords and b in coords:
                        coords[x] = (2*coords[b][0]-coords[a][0], 2*coords[b][1]-coords[a][1])
                
                elif pred == 'reflect':
                    x, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        coords[x] = reflect_point_over_line(coords[a], coords[b], coords[c])
                
                elif pred == 'angle_bisector':
                    x, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        ba = normalize((coords[a][0]-coords[b][0], coords[a][1]-coords[b][1]))
                        bc = normalize((coords[c][0]-coords[b][0], coords[c][1]-coords[b][1]))
                        bis = (ba[0]+bc[0], ba[1]+bc[1])
                        t = random.uniform(0.5, 3)
                        coords[x] = (coords[b][0]+t*bis[0], coords[b][1]+t*bis[1])
                
                elif pred == 'angle_mirror':
                    x, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        # Mirror of angle: extend angle on other side
                        ba = normalize((coords[a][0]-coords[b][0], coords[a][1]-coords[b][1]))
                        bc = normalize((coords[c][0]-coords[b][0], coords[c][1]-coords[b][1]))
                        # Reflection of BA over BC direction
                        bis = (bc[0]-ba[0], bc[1]-ba[1])
                        t = random.uniform(0.5, 3)
                        coords[x] = (coords[b][0]+t*bis[0], coords[b][1]+t*bis[1])
                
                elif pred == 'eqdistance':
                    x, a, b, c = args[0], args[1], args[2], args[3]
                    if a in coords and b in coords and c in coords:
                        bc_len = dist(coords[b], coords[c])
                        angle = random.uniform(0, 2*PI)
                        coords[x] = (coords[a][0]+bc_len*cos(angle), coords[a][1]+bc_len*sin(angle))
                
                elif pred == 'eqangle2':
                    x, a, b, c = args[0], args[1], args[2], args[3]
                    # Point X such that angle(A,B,A,X) = angle(C,X,C,B)
                    # This is complex - approximate
                    if a in coords and b in coords and c in coords:
                        # Place X somewhere reasonable
                        angle = random.uniform(0, 2*PI)
                        r = random.uniform(1, 5)
                        coords[x] = (coords[b][0]+r*cos(angle), coords[b][1]+r*sin(angle))
                
                elif pred == 'intersection_ll':
                    if len(args) >= 5:
                        x, a, b, c, d = args[0], args[1], args[2], args[3], args[4]
                        if a in coords and b in coords and c in coords and d in coords:
                            result = line_intersect(coords[a], coords[b], coords[c], coords[d])
                            if result:
                                coords[x] = result
                            else:
                                return None
                
                elif pred == 'intersection_lc':
                    if len(args) >= 4:
                        x, a, o, b = args[0], args[1], args[2], args[3]
                        if a in coords and o in coords and b in coords:
                            r = dist(coords[o], coords[b])
                            # Line through A: need another reference point
                            # Use the fact that X is on line through A and some other known point
                            # This requires context from other constraints
                            pass
                
                elif pred == 'intersection_cc':
                    if len(args) >= 4:
                        x, o, w, a = args[0], args[1], args[2], args[3]
                        if o in coords and w in coords and a in coords:
                            r1 = dist(coords[o], coords[a])
                            r2 = dist(coords[w], coords[a])
                            # Find intersection of two circles
                            d_ow = dist(coords[o], coords[w])
                            if d_ow > r1 + r2 or d_ow < abs(r1-r2):
                                return None
                            a_val = (r1**2 - r2**2 + d_ow**2) / (2*d_ow)
                            h_val = sqrt(max(0, r1**2 - a_val**2))
                            ow = normalize((coords[w][0]-coords[o][0], coords[w][1]-coords[o][1]))
                            px = coords[o][0] + a_val*ow[0]
                            py = coords[o][1] + a_val*ow[1]
                            ow_perp = (-ow[1], ow[0])
                            # Two intersection points; pick one randomly
                            sign = random.choice([-1, 1])
                            coords[x] = (px + sign*h_val*ow_perp[0], py + sign*h_val*ow_perp[1])
                
                elif pred == 'parallelogram':
                    if len(args) >= 4:
                        a, b, c, x = args[0], args[1], args[2], args[3]
                        if a in coords and b in coords and c in coords:
                            coords[x] = (coords[a][0]+coords[c][0]-coords[b][0],
                                         coords[a][1]+coords[c][1]-coords[b][1])
                
                elif pred == 'lc_tangent':
                    pass
                
                elif pred == 'cc_tangent':
                    pass
                
                elif pred == 'on_aline' or pred == 'on_aline2':
                    # Complex angle alignment
                    pass
                
                elif pred == 'eqangle3':
                    pass
                
                elif pred == 'on_circum':
                    if len(args) >= 4:
                        x, a, b, c = args[0], args[1], args[2], args[3]
                        if a in coords and b in coords and c in coords:
                            cc = circumcenter(coords[a], coords[b], coords[c])
                            if cc:
                                r = dist(cc, coords[a])
                                angle = random.uniform(0, 2*PI)
                                coords[x] = (cc[0]+r*cos(angle), cc[1]+r*sin(angle))
                
                elif pred == 'on_opline':
                    x, a, b = args[0], args[1], args[2]
                    if a in coords and b in coords:
                        t = random.uniform(-3, 4)
                        coords[x] = (coords[a][0]+t*(coords[b][0]-coords[a][0]),
                                     coords[a][1]+t*(coords[b][1]-coords[a][1]))
                
                elif pred == 'nsquare' or pred == 'psquare':
                    x, a, b = args[0], args[1], args[2]
                    if a in coords and b in coords:
                        ab = (coords[b][0]-coords[a][0], coords[b][1]-coords[a][1])
                        ab_len = dist(coords[a], coords[b])
                        ab_perp = normalize((-ab[1], ab[0]))
                        coords[x] = (coords[a][0]+ab_len*ab_perp[0], coords[a][1]+ab_len*ab_perp[1])
                
                elif pred == 'shift':
                    x, b, c, d = args[0], args[1], args[2], args[3]
                    # Shift: X such that XB = CD and XC = BD
                    pass
                
                elif pred == 'quadrangle' or pred == 'pentagon':
                    for p in args:
                        if p not in coords:
                            coords[p] = (random.uniform(-5, 5), random.uniform(-5, 5))
            
            except Exception as e:
                continue
    
    return coords if len(coords) > 0 else None


def check_goal_numerically(coords, goal, tol=1e-4):
    """Check if goal holds numerically with given tolerance."""
    if not goal or not coords:
        return False
    
    pred = goal['predicate']
    args = goal['args']
    
    # All args must have coordinates
    for a in args:
        if a not in coords:
            return False
    
    if pred == 'cong' and len(args) >= 4:
        a, b, c, d = args[0], args[1], args[2], args[3]
        ab = dist(coords[a], coords[b])
        cd = dist(coords[c], coords[d])
        return abs(ab - cd) < tol * max(ab, cd, 0.01)
    
    elif pred == 'coll' and len(args) >= 3:
        a, b, c = args[0], args[1], args[2]
        area = abs((coords[b][0]-coords[a][0])*(coords[c][1]-coords[a][1]) -
                   (coords[c][0]-coords[a][0])*(coords[b][1]-coords[a][1]))
        base = max(dist(coords[a], coords[b]), dist(coords[a], coords[c]), 0.01)
        return area < tol * base
    
    elif pred == 'cyclic' and len(args) >= 4:
        a, b, c, d = args[0], args[1], args[2], args[3]
        cc = circumcenter(coords[a], coords[b], coords[c])
        if cc is None:
            return False
        r_abc = dist(cc, coords[a])
        r_d = dist(cc, coords[d])
        return abs(r_abc - r_d) < tol * max(r_abc, 0.01)
    
    elif pred == 'perp' and len(args) >= 4:
        a, b, c, d = args[0], args[1], args[2], args[3]
        ab = (coords[b][0]-coords[a][0], coords[b][1]-coords[a][1])
        cd = (coords[d][0]-coords[c][0], coords[d][1]-coords[c][1])
        d_product = dot(ab, cd)
        mag = sqrt(dot(ab,ab)) * sqrt(dot(cd,cd))
        return abs(d_product) < tol * max(mag, 0.01)
    
    elif pred == 'para' and len(args) >= 4:
        a, b, c, d = args[0], args[1], args[2], args[3]
        ab = (coords[b][0]-coords[a][0], coords[b][1]-coords[a][1])
        cd = (coords[d][0]-coords[c][0], coords[d][1]-coords[c][1])
        c_product = cross(ab, cd)
        mag = sqrt(dot(ab,ab)) * sqrt(dot(cd,cd))
        return abs(c_product) < tol * max(mag, 0.01)
    
    elif pred == 'eqangle' and len(args) >= 8:
        # eqangle A B C D E F G H: angle(AB,CD) = angle(EF,GH)
        # Simplified: check using dot/cross products
        a, b, c, d, e, f, g, h = args[:8]
        ab = (coords[b][0]-coords[a][0], coords[b][1]-coords[a][1])
        cd = (coords[d][0]-coords[c][0], coords[d][1]-coords[c][1])
        ef = (coords[f][0]-coords[e][0], coords[f][1]-coords[e][1])
        gh = (coords[h][0]-coords[g][0], coords[h][1]-coords[g][1])
        # Compare angles using atan2
        angle1 = math.atan2(cross(ab, cd), dot(ab, cd))
        angle2 = math.atan2(cross(ef, gh), dot(ef, gh))
        return abs(angle1 - angle2) < tol or abs(abs(angle1 - angle2) - PI) < tol
    
    elif pred == 'eqratio' and len(args) >= 8:
        a, b, c, d, e, f, g, h = args[:8]
        ab = dist(coords[a], coords[b])
        cd = dist(coords[c], coords[d])
        ef = dist(coords[e], coords[f])
        gh = dist(coords[g], coords[h])
        if cd * gh < 1e-10:
            return False
        return abs(ab/cd - ef/gh) < tol
    
    return False


def verify_problem(problem, num_trials=200):
    """Run multiple coordinate trials to verify a problem's goal."""
    successes = 0
    valid_trials = 0
    
    for trial in range(num_trials):
        coords = assign_coords_for_problem(problem, seed=trial)
        if coords is None:
            continue
        
        # Check that enough points have coordinates
        all_points = set()
        for c in problem['constructions']:
            for con in c['constraints']:
                all_points.update(con['args'])
            all_points.update(c['defined_points'])
        if problem['goal']:
            all_points.update(problem['goal']['args'])
        
        covered = sum(1 for p in all_points if p in coords)
        if covered < len(all_points) * 0.8:
            continue
        
        valid_trials += 1
        if check_goal_numerically(coords, problem['goal']):
            successes += 1
    
    if valid_trials == 0:
        return False, 0.0, 0
    
    confidence = successes / valid_trials
    return confidence > 0.85, confidence, valid_trials


if __name__ == '__main__':
    with open(os.path.join(OUTPUT_DIR, 'parsed_problems.json'), 'r') as f:
        problems = json.load(f)
    
    results = []
    
    for problem in problems:
        name = problem['name']
        goal_pred = problem['goal_predicate']
        print(f"Verifying {name} (goal: {goal_pred})...")
        
        verified, confidence, trials = verify_problem(problem, num_trials=100)
        
        results.append({
            'name': name,
            'goal_predicate': goal_pred,
            'verified': verified,
            'confidence': confidence,
            'valid_trials': trials,
            'num_points': problem['num_points']
        })
        
        status = "VERIFIED" if verified else "NOT VERIFIED"
        print(f"  {status} (confidence={confidence:.2f}, trials={trials})")
    
    with open(os.path.join(OUTPUT_DIR, 'coord_verification_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    total_verified = sum(1 for r in results if r['verified'])
    print(f"\nTotal verified by coordinates: {total_verified}/{len(problems)}")