#!/usr/bin/env python3
"""
Analysis of inference rules and their applicability to IMO geometry problems.
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
REPORT_IMAGES_DIR = Path("report/images")
DATA_DIR = Path("data")

def parse_rule(rule_line):
    """Parse an inference rule line"""
    if '=>' not in rule_line:
        return None
    
    parts = rule_line.split('=>')
    premises_str = parts[0].strip()
    conclusion_str = parts[1].strip()
    
    # Parse premises (comma or space separated conditions)
    premises = []
    # Split by comma but handle nested structures
    prem_parts = re.split(r',\s*(?![^()]*\))', premises_str)
    for p in prem_parts:
        p = p.strip()
        if p:
            premises.append(p)
    
    return {
        'premises': premises,
        'conclusion': conclusion_str,
        'raw': rule_line.strip()
    }

def categorize_rule(rule):
    """Categorize a rule by its conclusion type"""
    conclusion = rule['conclusion'].lower()
    
    categories = {
        'parallel': ['para'],
        'perpendicular': ['perp'],
        'congruence': ['cong'],
        'angle_equality': ['eqangle'],
        'collinearity': ['coll'],
        'cyclic': ['cyclic'],
        'ratio_equality': ['eqratio'],
        'triangle_similarity': ['simtri', 'contri'],
        'midpoint': ['midp']
    }
    
    for cat, keywords in categories.items():
        for kw in keywords:
            if kw in conclusion:
                return cat
    return 'other'

def extract_predicates(formula):
    """Extract predicates from a formula"""
    predicates = []
    # Match patterns like "para A B C D", "cong O A O B", etc.
    matches = re.findall(r'(\w+)\s+([A-Z]+\s+)+', formula)
    for match in matches:
        pred_name = match[0]
        predicates.append(pred_name)
    return predicates

def analyze_rules(rules):
    """Analyze the inference rules"""
    analysis = {
        'total_rules': len(rules),
        'by_category': Counter(),
        'premise_counts': [],
        'common_premise_predicates': Counter(),
        'common_conclusion_predicates': Counter(),
        'rules_by_category': defaultdict(list)
    }
    
    for i, rule in enumerate(rules):
        parsed = parse_rule(rule)
        if not parsed:
            continue
        
        category = categorize_rule(parsed)
        analysis['by_category'][category] += 1
        analysis['rules_by_category'][category].append({
            'index': i,
            'rule': parsed
        })
        
        # Count premises
        analysis['premise_counts'].append(len(parsed['premises']))
        
        # Extract predicates
        for prem in parsed['premises']:
            preds = extract_predicates(prem)
            analysis['common_premise_predicates'].update(preds)
        
        conc_preds = extract_predicates(parsed['conclusion'])
        analysis['common_conclusion_predicates'].update(conc_preds)
    
    return analysis

def generate_rule_visualizations(analysis):
    """Generate visualizations for rule analysis"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    figures = {}
    
    # Figure 1: Rules by Category
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    categories = list(analysis['by_category'].keys())
    counts = list(analysis['by_category'].values())
    colors = plt.cm.Set2(range(len(categories)))
    ax1.bar(categories, counts, color=colors, edgecolor='black')
    ax1.set_xlabel('Rule Category', fontsize=12)
    ax1.set_ylabel('Number of Rules', fontsize=12)
    ax1.set_title('Distribution of Inference Rules by Category', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fig1_path = REPORT_IMAGES_DIR / 'rules_by_category.png'
    fig1.savefig(fig1_path, dpi=150)
    figures['rules_by_category'] = str(fig1_path)
    plt.close()
    
    # Figure 2: Premise Count Distribution
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    premise_counts = analysis['premise_counts']
    ax2.hist(premise_counts, bins=range(min(premise_counts), max(premise_counts)+2),
             color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax2.set_xlabel('Number of Premises', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Premises per Inference Rule', fontsize=14)
    plt.tight_layout()
    fig2_path = REPORT_IMAGES_DIR / 'premise_distribution.png'
    fig2.savefig(fig2_path, dpi=150)
    figures['premise_distribution'] = str(fig2_path)
    plt.close()
    
    # Figure 3: Top Premise Predicates
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    top_predicates = analysis['common_premise_predicates'].most_common(15)
    pred_names = [p[0] for p in top_predicates]
    pred_counts = [p[1] for p in top_predicates]
    colors = plt.cm.plasma([i/len(pred_names) for i in range(len(pred_names))])
    ax3.barh(pred_names, pred_counts, color=colors, edgecolor='black')
    ax3.set_xlabel('Frequency', fontsize=12)
    ax3.set_ylabel('Predicate', fontsize=12)
    ax3.set_title('Top 15 Premise Predicates in Inference Rules', fontsize=14)
    ax3.invert_yaxis()
    plt.tight_layout()
    fig3_path = REPORT_IMAGES_DIR / 'top_premise_predicates.png'
    fig3.savefig(fig3_path, dpi=150)
    figures['top_premise_predicates'] = str(fig3_path)
    plt.close()
    
    return figures

def main():
    """Main function"""
    print("=" * 60)
    print("Inference Rule Analysis")
    print("=" * 60)
    
    # Load rules
    print("\n[1] Loading inference rules...")
    with open(DATA_DIR / 'rules.txt', 'r') as f:
        rules = [line.strip() for line in f if line.strip()]
    
    print(f"    Loaded {len(rules)} rules")
    
    # Analyze rules
    print("\n[2] Analyzing rule structure...")
    analysis = analyze_rules(rules)
    
    print(f"\n    Summary Statistics:")
    print(f"    - Total rules: {analysis['total_rules']}")
    print(f"    - Categories: {dict(analysis['by_category'])}")
    print(f"    - Avg premises/rule: {sum(analysis['premise_counts'])/len(analysis['premise_counts']):.2f}")
    
    # Generate visualizations
    print("\n[3] Generating visualizations...")
    figures = generate_rule_visualizations(analysis)
    print(f"    Generated {len(figures)} figures")
    
    # Save results
    print("\n[4] Saving analysis results...")
    
    # Convert defaultdict to regular dict for JSON serialization
    rules_by_cat_serializable = {}
    for cat, rules_list in analysis['rules_by_category'].items():
        rules_by_cat_serializable[cat] = [
            {'index': r['index'], 'conclusion': r['rule']['conclusion']}
            for r in rules_list
        ]
    
    rule_analysis_json = {
        'total_rules': analysis['total_rules'],
        'by_category': dict(analysis['by_category']),
        'avg_premises_per_rule': sum(analysis['premise_counts'])/len(analysis['premise_counts']),
        'top_premise_predicates': analysis['common_premise_predicates'].most_common(15),
        'top_conclusion_predicates': analysis['common_conclusion_predicates'].most_common(10),
        'rules_by_category': rules_by_cat_serializable,
        'figures': figures
    }
    
    with open(OUTPUTS_DIR / 'rule_analysis_results.json', 'w') as f:
        json.dump(rule_analysis_json, f, indent=2)
    
    print("    Saved rule_analysis_results.json")
    
    print("\n" + "=" * 60)
    print("Rule Analysis Complete!")
    print("=" * 60)
    
    return analysis, figures

if __name__ == '__main__':
    main()
