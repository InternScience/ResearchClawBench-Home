#!/usr/bin/env python3
"""
Analysis code for IMO geometry problems.
Parses formal geometry statements and generates statistics and visualizations.
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path

# Output directories
OUTPUTS_DIR = Path("outputs")
REPORT_IMAGES_DIR = Path("report/images")
DATA_DIR = Path("data")

def parse_problem_line(line):
    """Parse a single problem line from imo_ag_30.txt"""
    # Format: problem_id\nconstruction ? conclusion
    parts = line.strip().split(" ? ")
    if len(parts) != 2:
        return None
    
    # Split into problem_id line and construction line
    lines = line.strip().split('\n')
    if len(lines) < 2:
        return None
    
    problem_id_line = lines[0].strip()
    construction = parts[0]
    conclusion = parts[1]
    
    # Extract year and problem number from problem_id_line
    # e.g., "translated_imo_2000_p1" -> year=2000, problem=1
    match = re.search(r'(?:translated_)?imo_(\d+)_p(\d+\w*)', problem_id_line)
    year = int(match.group(1)) if match else None
    prob_num = match.group(2) if match else None
    
    # Use the full problem_id_line as the problem_id
    problem_id = problem_id_line
    
    return {
        'problem_id': problem_id,
        'year': year,
        'problem_num': prob_num,
        'construction': construction,
        'conclusion': conclusion,
        'full_line': line.strip()
    }

def extract_geometric_objects(construction):
    """Extract geometric objects (points, lines, circles) from construction"""
    objects = []
    
    # Pattern to match point definitions like "a b c = triangle a b c"
    # or "o = circle o a b c"
    patterns = [
        r'(\w+(?:\s+\w+)*)\s*=\s*(\w+)\s+(\w+(?:\s+\w+)*)',  # x = type args
        r'^(\w+)\s+(\w+)\s*=',  # a b = segment ...
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, construction)
        for match in matches:
            if len(match) >= 2:
                objects.append({
                    'points': match[0].split(),
                    'type': match[1],
                    'args': match[2].split() if len(match) > 2 else []
                })
    
    return objects

def extract_construction_primitives(construction):
    """Extract construction primitives (triangle, circle, midpoint, etc.)"""
    primitives = []
    
    # Common geometric constructions
    primitive_patterns = [
        r'\btriangle\b', r'\bcircle\b', r'\bmidpoint\b', r'\borthocenter\b',
        r'\bincenter\b', r'\bexcenter\b', r'\bcentroid\b', r'\bfoot\b',
        r'\breflect\b', r'\bmirror\b', r'\bangle_bisector\b', r'\bon_line\b',
        r'\bon_circle\b', r'\bon_bline\b', r'\bon_tline\b', r'\bon_pline\b',
        r'\bon_aline\b', r'\bon_dia\b', r'\bpara\b', r'\bperp\b',
        r'\bcong\b', r'\beqangle\b', r'\bcyclic\b', r'\bcoll\b',
        r'\biso_triangle\b', r'\br_triangle\b', r'\bsquare\b',
        r'\bquadrangle\b', r'\bpentagon\b', r'\btrapezoid\b',
        r'\brectangle\b', r'\bsegment\b', r'\bfree\b',
        r'\bintersection_ll\b', r'\bintersection_cc\b', r'\bintersection_lc\b',
        r'\bcc_tangent\b', r'\btangent\b', r'\bninepoints\b'
    ]
    
    for prim in primitive_patterns:
        if re.search(prim, construction, re.IGNORECASE):
            primitives.append(prim.strip('\\b'))
    
    return primitives

def extract_conclusion_type(conclusion):
    """Extract the type of conclusion being proved"""
    conclusion_types = {
        'cong': 'congruence',
        'eqangle': 'angle_equality',
        'para': 'parallel',
        'perp': 'perpendicular',
        'coll': 'collinearity',
        'cyclic': 'concyclic',
        'eqratio': 'ratio_equality',
        'eqdistance': 'distance_equality'
    }
    
    for key, value in conclusion_types.items():
        if key in conclusion.lower():
            return value
    return 'other'

def load_defs(filepath):
    """Load and parse geometric definitions"""
    defs = {}
    current_def = None
    current_content = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new definition (no leading whitespace and contains space)
            if not line.startswith(' ') and ' ' in line and '=' not in line[:20]:
                if current_def:
                    defs[current_def] = current_content
                current_def = line.split()[0]
                current_content = [line]
            else:
                current_content.append(line)
        
        if current_def:
            defs[current_def] = current_content
    
    return defs

def load_rules(filepath):
    """Load inference rules"""
    rules = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '=>' in line:
                parts = line.split('=>')
                if len(parts) == 2:
                    rules.append({
                        'premises': parts[0].strip(),
                        'conclusion': parts[1].strip()
                    })
    return rules

def analyze_problems(problems):
    """Generate comprehensive analysis of problems"""
    analysis = {
        'total_problems': len(problems),
        'years': [],
        'problem_nums': [],
        'conclusion_types': Counter(),
        'primitive_counts': Counter(),
        'objects_per_problem': [],
        'primitives_per_problem': [],
        'problems_by_year': defaultdict(list)
    }
    
    for prob in problems:
        if prob['year']:
            analysis['years'].append(prob['year'])
            analysis['problems_by_year'][prob['year']].append(prob['problem_num'])
        if prob['problem_num']:
            analysis['problem_nums'].append(prob['problem_num'])
        
        # Conclusion type
        conc_type = extract_conclusion_type(prob['conclusion'])
        analysis['conclusion_types'][conc_type] += 1
        
        # Primitives
        primitives = extract_construction_primitives(prob['construction'])
        analysis['primitive_counts'].update(primitives)
        analysis['primitives_per_problem'].append(len(primitives))
        
        # Objects
        objects = extract_geometric_objects(prob['construction'])
        analysis['objects_per_problem'].append(len(objects))
    
    return analysis

def generate_visualizations(analysis, problems):
    """Generate visualization data and save figures"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    
    figures = {}
    
    # Figure 1: Problems by Year
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    years_sorted = sorted(analysis['problems_by_year'].keys())
    counts_by_year = [len(analysis['problems_by_year'][y]) for y in years_sorted]
    ax1.bar(years_sorted, counts_by_year, color='steelblue', edgecolor='navy')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Problems', fontsize=12)
    ax1.set_title('Distribution of IMO Geometry Problems by Year', fontsize=14)
    ax1.set_xticks(years_sorted)
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fig1_path = REPORT_IMAGES_DIR / 'problems_by_year.png'
    fig1.savefig(fig1_path, dpi=150)
    figures['problems_by_year'] = str(fig1_path)
    plt.close()
    
    # Figure 2: Conclusion Types Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    conc_types = list(analysis['conclusion_types'].keys())
    conc_counts = list(analysis['conclusion_types'].values())
    colors = plt.cm.Set3(range(len(conc_types)))
    ax2.bar(conc_types, conc_counts, color=colors, edgecolor='black')
    ax2.set_xlabel('Conclusion Type', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Conclusion Types in IMO Problems', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fig2_path = REPORT_IMAGES_DIR / 'conclusion_types.png'
    fig2.savefig(fig2_path, dpi=150)
    figures['conclusion_types'] = str(fig2_path)
    plt.close()
    
    # Figure 3: Top Geometric Primitives
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    top_primitives = analysis['primitive_counts'].most_common(15)
    prim_names = [p[0] for p in top_primitives]
    prim_counts = [p[1] for p in top_primitives]
    colors = plt.cm.viridis([i/len(prim_names) for i in range(len(prim_names))])
    ax3.barh(prim_names, prim_counts, color=colors, edgecolor='black')
    ax3.set_xlabel('Frequency', fontsize=12)
    ax3.set_ylabel('Geometric Primitive', fontsize=12)
    ax3.set_title('Top 15 Geometric Primitives in IMO Problems', fontsize=14)
    ax3.invert_yaxis()
    plt.tight_layout()
    fig3_path = REPORT_IMAGES_DIR / 'top_primitives.png'
    fig3.savefig(fig3_path, dpi=150)
    figures['top_primitives'] = str(fig3_path)
    plt.close()
    
    # Figure 4: Complexity Metrics
    fig4, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Objects per problem
    axes[0].hist(analysis['objects_per_problem'], bins=range(min(analysis['objects_per_problem']), 
                                                              max(analysis['objects_per_problem'])+2),
                 color='coral', edgecolor='darkred', alpha=0.7)
    axes[0].set_xlabel('Number of Geometric Objects', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Objects per Problem', fontsize=14)
    
    # Primitives per problem
    axes[1].hist(analysis['primitives_per_problem'], bins=range(min(analysis['primitives_per_problem']), 
                                                                 max(analysis['primitives_per_problem'])+2),
                 color='teal', edgecolor='darkgreen', alpha=0.7)
    axes[1].set_xlabel('Number of Construction Primitives', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Primitives per Problem', fontsize=14)
    
    plt.tight_layout()
    fig4_path = REPORT_IMAGES_DIR / 'complexity_metrics.png'
    fig4.savefig(fig4_path, dpi=150)
    figures['complexity_metrics'] = str(fig4_path)
    plt.close()
    
    # Figure 5: Problem Timeline
    fig5, ax5 = plt.subplots(figsize=(14, 8))
    for year in sorted(analysis['problems_by_year'].keys()):
        probs = analysis['problems_by_year'][year]
        y_pos = list(analysis['problems_by_year'].keys()).index(year)
        for i, p in enumerate(probs):
            ax5.scatter(year, y_pos, s=200, c=plt.cm.tab10(i % 10), 
                       edgecolors='black', linewidth=1.5, label=f'P{p}' if y_pos == 0 else "")
    
    ax5.set_xlabel('Year', fontsize=12)
    ax5.set_ylabel('Problem Index', fontsize=12)
    ax5.set_title('IMO Geometry Problems Timeline (2000-2022)', fontsize=14)
    ax5.set_yticks(range(len(analysis['problems_by_year'])))
    ax5.set_yticklabels([f'{i+1}' for i in range(len(analysis['problems_by_year']))])
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    fig5_path = REPORT_IMAGES_DIR / 'problem_timeline.png'
    fig5.savefig(fig5_path, dpi=150)
    figures['problem_timeline'] = str(fig5_path)
    plt.close()
    
    return figures

def main():
    """Main analysis function"""
    print("=" * 60)
    print("IMO Geometry Problems Analysis")
    print("=" * 60)
    
    # Load problems
    print("\n[1] Loading problems from imo_ag_30.txt...")
    problems = []
    with open(DATA_DIR / 'imo_ag_30.txt', 'r') as f:
        content = f.read()
        lines = content.strip().split('\n')
        # Format: problem_id on one line, construction ? conclusion on next
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('translated_imo_') or line.startswith('imo_'):
                problem_id_line = line
                if i + 1 < len(lines):
                    construction_line = lines[i + 1]
                    full_entry = problem_id_line + '\n' + construction_line
                    prob = parse_problem_line(full_entry)
                    if prob:
                        problems.append(prob)
                i += 2
            else:
                i += 1
    
    print(f"    Loaded {len(problems)} problems")
    
    # Load definitions
    print("\n[2] Loading geometric definitions...")
    defs = load_defs(DATA_DIR / 'defs.txt')
    print(f"    Loaded {len(defs)} definitions")
    
    # Load rules
    print("\n[3] Loading inference rules...")
    rules = load_rules(DATA_DIR / 'rules.txt')
    print(f"    Loaded {len(rules)} inference rules")
    
    # Analyze problems
    print("\n[4] Analyzing problem structure...")
    analysis = analyze_problems(problems)
    
    print(f"\n    Summary Statistics:")
    print(f"    - Total problems: {analysis['total_problems']}")
    if analysis['years']:
        print(f"    - Years covered: {min(analysis['years'])} to {max(analysis['years'])}")
    else:
        print(f"    - Years covered: N/A")
    print(f"    - Unique conclusion types: {len(analysis['conclusion_types'])}")
    print(f"    - Unique primitives: {len(analysis['primitive_counts'])}")
    if analysis['objects_per_problem']:
        print(f"    - Avg objects/problem: {sum(analysis['objects_per_problem'])/len(analysis['objects_per_problem']):.2f}")
    print(f"    - Avg primitives/problem: {sum(analysis['primitives_per_problem'])/len(analysis['primitives_per_problem']):.2f}")
    
    # Generate visualizations
    print("\n[5] Generating visualizations...")
    figures = generate_visualizations(analysis, problems)
    print(f"    Generated {len(figures)} figures")
    
    # Save analysis results
    print("\n[6] Saving analysis results...")
    
    # Save detailed analysis JSON
    analysis_json = {
        'total_problems': analysis['total_problems'],
        'year_range': [min(analysis['years']), max(analysis['years'])] if analysis['years'] else [],
        'conclusion_types': dict(analysis['conclusion_types']),
        'top_primitives': analysis['primitive_counts'].most_common(20),
        'avg_objects_per_problem': sum(analysis['objects_per_problem'])/len(analysis['objects_per_problem']) if analysis['objects_per_problem'] else 0,
        'avg_primitives_per_problem': sum(analysis['primitives_per_problem'])/len(analysis['primitives_per_problem']),
        'problems_by_year': {str(k): v for k, v in analysis['problems_by_year'].items()},
        'figures': figures
    }
    
    with open(OUTPUTS_DIR / 'analysis_results.json', 'w') as f:
        json.dump(analysis_json, f, indent=2)
    
    # Save problem details
    problem_details = []
    for prob in problems:
        detail = {
            'problem_id': prob['problem_id'],
            'year': prob['year'],
            'problem_num': prob['problem_num'],
            'conclusion_type': extract_conclusion_type(prob['conclusion']),
            'primitives': extract_construction_primitives(prob['construction']),
            'num_objects': len(extract_geometric_objects(prob['construction']))
        }
        problem_details.append(detail)
    
    with open(OUTPUTS_DIR / 'problem_details.json', 'w') as f:
        json.dump(problem_details, f, indent=2)
    
    print("    Saved analysis_results.json")
    print("    Saved problem_details.json")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return analysis, figures

if __name__ == '__main__':
    main()
