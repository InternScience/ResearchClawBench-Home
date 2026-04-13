import json, re
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'imo_ag_30.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')

KEYWORDS = [
    'on_line','on_circle','circle','on_bline','on_tline','midpoint','on_aline','foot','reflect',
    'angle_bisector','mirror','on_pline','orthocenter','eqdistance','incenter2','on_dia','cc_tangent',
    'excenter2','angle_mirror','parallelogram','triangle','r_triangle','iso_triangle','segment','free'
]
GOAL_TYPES = ['cong','coll','cyclic','eqangle','eqratio','perp','para']


def load_examples(path: Path):
    raw = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    examples = []
    i = 0
    while i < len(raw):
        if raw[i].startswith('translated_imo_'):
            pid = raw[i]
            statement = raw[i+1] if i+1 < len(raw) else ''
            examples.append((pid, statement))
            i += 2
        else:
            i += 1
    return examples


def parse_example(pid, line):
    premise, goal = (line.split('?', 1) + [''])[:2] if '?' in line else (line, '')
    goal = goal.strip()
    goal_type = goal.split()[0] if goal else 'unknown'
    constructs = []
    for seg in premise.split(';'):
        seg = seg.strip()
        if not seg:
            continue
        kws = re.findall(r'\b(' + '|'.join(map(re.escape, KEYWORDS)) + r')\b', seg)
        constructs.extend(kws)
    years = re.findall(r'(20\d\d)', pid)
    year = int(years[0]) if years else None
    return {
        'id': pid,
        'year': year,
        'text': line,
        'goal_type': goal_type,
        'n_chars': len(line),
        'n_semicolons': line.count(';'),
        'n_constructs': len(constructs),
        'constructs': constructs,
        'goal_tokens': len(goal.split()),
    }


def main():
    examples = load_examples(DATA)
    rows = [parse_example(pid, stmt) for pid, stmt in examples]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'benchmark_problem_table.csv', index=False)

    summary = {
        'n_problems': int(len(df)),
        'n_unique_ids': int(df['id'].nunique()),
        'goal_type_counts': df['goal_type'].value_counts().to_dict(),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'avg_chars': float(df['n_chars'].mean()),
        'avg_constructs': float(df['n_constructs'].mean()),
        'median_constructs': float(df['n_constructs'].median()),
    }
    (OUT / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2))

    construct_counter = Counter()
    for lst in df['constructs']:
        construct_counter.update(lst)
    pd.DataFrame(construct_counter.most_common(), columns=['construct','count']).to_csv(OUT / 'construct_counts.csv', index=False)

    plt.figure(figsize=(8,4.5))
    order = df['goal_type'].value_counts().index
    sns.countplot(data=df, x='goal_type', order=order, color='#4C78A8')
    plt.title('Distribution of theorem goal types in the IMO geometry benchmark')
    plt.xlabel('Goal type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(IMG / 'goal_type_distribution.png', dpi=200)
    plt.close()

    cc_df = pd.DataFrame(construct_counter.most_common(12), columns=['construct','count'])
    plt.figure(figsize=(9,5))
    sns.barplot(data=cc_df, y='construct', x='count', color='#F58518')
    plt.title('Most frequent construction primitives')
    plt.xlabel('Frequency across problem statements')
    plt.ylabel('Primitive')
    plt.tight_layout()
    plt.savefig(IMG / 'construction_primitive_frequency.png', dpi=200)
    plt.close()

    yearly = df.groupby('year', as_index=False).agg(mean_constructs=('n_constructs','mean'), mean_chars=('n_chars','mean'), n=('id','count'))
    yearly.to_csv(OUT / 'yearly_complexity.csv', index=False)
    fig, ax1 = plt.subplots(figsize=(9,5))
    sns.lineplot(data=yearly, x='year', y='mean_constructs', marker='o', ax=ax1, color='#54A24B')
    ax1.set_ylabel('Mean number of construction primitives', color='#54A24B')
    ax1.set_xlabel('Year')
    ax2 = ax1.twinx()
    sns.lineplot(data=yearly, x='year', y='mean_chars', marker='s', ax=ax2, color='#E45756')
    ax2.set_ylabel('Mean statement length (characters)', color='#E45756')
    plt.title('Benchmark complexity trends over time')
    fig.tight_layout()
    plt.savefig(IMG / 'complexity_by_year.png', dpi=200)
    plt.close()

    top_constructs = [c for c,_ in construct_counter.most_common(10)]
    mat = []
    index = []
    for g in [x for x in GOAL_TYPES if x in set(df['goal_type'])]:
        sub = df[df['goal_type']==g]
        counts = {c:0 for c in top_constructs}
        for lst in sub['constructs']:
            for c in set(lst):
                if c in counts:
                    counts[c]+=1
        total = max(len(sub),1)
        mat.append([counts[c]/total for c in top_constructs])
        index.append(g)
    heat = pd.DataFrame(mat, index=index, columns=top_constructs)
    heat.to_csv(OUT / 'construct_goal_heatmap.csv')
    plt.figure(figsize=(10,4.8))
    sns.heatmap(heat, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label':'Fraction of problems'})
    plt.title('Association between goal types and common construction primitives')
    plt.xlabel('Construction primitive')
    plt.ylabel('Goal type')
    plt.tight_layout()
    plt.savefig(IMG / 'construct_goal_heatmap.png', dpi=200)
    plt.close()

    weights = {'on_line':1,'on_circle':1.5,'circle':1,'on_bline':1.5,'on_tline':2,'midpoint':1,'on_aline':2.5,'foot':2,'reflect':2.5,
               'angle_bisector':2,'mirror':1.5,'on_pline':2,'orthocenter':3,'eqdistance':2,'incenter2':3,'on_dia':1.5,'cc_tangent':4,
               'excenter2':3,'angle_mirror':2.5,'parallelogram':2,'triangle':0.5,'r_triangle':1,'iso_triangle':1,'segment':0.5,'free':0.2}
    goal_bonus = {'cong':1,'coll':1,'cyclic':2,'eqangle':2.5,'eqratio':2.5,'perp':1.5,'para':1.5}
    df['heuristic_difficulty'] = df.apply(lambda r: sum(weights.get(c,1) for c in r['constructs']) + goal_bonus.get(r['goal_type'],0), axis=1)
    df.sort_values('heuristic_difficulty', ascending=False).to_csv(OUT / 'problem_difficulty_ranked.csv', index=False)

    plt.figure(figsize=(8,5))
    sns.histplot(df['heuristic_difficulty'], bins=10, color='#72B7B2', edgecolor='white')
    plt.title('Heuristic difficulty distribution of benchmark problems')
    plt.xlabel('Heuristic difficulty score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(IMG / 'difficulty_distribution.png', dpi=200)
    plt.close()

    baseline = {
        'approach': 'symbolic-first neuro-symbolic pipeline proposal',
        'retrieval_stage': 'retrieve candidate lemmas/rules by primitive and goal type',
        'search_stage': 'best-first expansion over geometric invariants and auxiliary constructions',
        'verification_stage': 'proof checked by formal Euclidean rule engine',
        'curriculum_signal': 'heuristic_difficulty from problem text only',
        'hardest_problem_ids': df.sort_values('heuristic_difficulty', ascending=False)['id'].head(10).tolist()
    }
    (OUT / 'proposed_baseline.json').write_text(json.dumps(baseline, indent=2))
    print(json.dumps(summary, indent=2))
    print('hardest:', baseline['hardest_problem_ids'])

if __name__ == '__main__':
    main()
