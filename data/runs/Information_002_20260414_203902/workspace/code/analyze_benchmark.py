from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / '2111.01152'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'


def main():
    OUT.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load((DATA / '2111.01152.yaml').read_text())
    rows, summary = [], []
    for i, item in enumerate(data, 1):
        task = item.get('task')
        score = item.get('score', {}) or {}
        summary.append({'task_index': i, 'task': task, **score})
        for field, details in (item.get('placeholder', {}) or {}).items():
            if isinstance(details, dict) and 'score' in details:
                s = details.get('score', {}) or {}
                rows.append({
                    'task_index': i,
                    'task': task,
                    'field': field,
                    'Haining': s.get('Haining'),
                    'Will': s.get('Will'),
                    'Yasaman': s.get('Yasaman'),
                    'LLM': details.get('LLM'),
                    'human': details.get('human'),
                })
    rows_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary)
    rows_df.to_csv(OUT / 'step_scores_long.csv', index=False)
    summary_df.to_csv(OUT / 'task_scores_summary.csv', index=False)
    rev = []
    for reviewer in ['Haining', 'Will', 'Yasaman']:
        vals = pd.to_numeric(rows_df[reviewer], errors='coerce').dropna()
        rev.append({'reviewer': reviewer, 'mean_step_score': vals.mean(), 'n_scored_fields': int(vals.shape[0])})
    pd.DataFrame(rev).to_csv(OUT / 'reviewer_score_summary.csv', index=False)
    mat = summary_df[[c for c in summary_df.columns if c != 'task_index']].set_index('task')
    mat.to_csv(OUT / 'category_heatmap_matrix.csv')
    sns.set_theme(style='whitegrid')
    cat_cols = [c for c in summary_df.columns if c not in ['task_index', 'task']]
    means = summary_df[cat_cols].apply(pd.to_numeric, errors='coerce').mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(x=means.index, y=means.values, color='#4C78A8')
    ax.set_ylabel('Mean score'); ax.set_xlabel('Rubric category'); ax.set_ylim(0, 2.1)
    plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.savefig(IMG / 'task_scores.png', dpi=200); plt.close()
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(mat.apply(pd.to_numeric, errors='coerce'), annot=True, fmt='.0f', cmap='YlGnBu', vmin=0, vmax=2, cbar_kws={'label': 'Task score'})
    ax.set_xlabel('Rubric category'); ax.set_ylabel('Task')
    plt.tight_layout(); plt.savefig(IMG / 'category_heatmap.png', dpi=200); plt.close()
    rev_df = pd.DataFrame(rev)
    plt.figure(figsize=(5.5, 4))
    ax = sns.barplot(data=rev_df, x='reviewer', y='mean_step_score', hue='reviewer', dodge=False, legend=False)
    ax.set_ylabel('Mean placeholder score'); ax.set_xlabel('Reviewer'); ax.set_ylim(0, 2.1)
    for i, v in enumerate(rev_df['mean_step_score']):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(IMG / 'reviewer_scores.png', dpi=200); plt.close()
    print(json.dumps({'tasks': len(summary_df), 'placeholders': len(rows_df)}, indent=2))


if __name__ == '__main__':
    main()
