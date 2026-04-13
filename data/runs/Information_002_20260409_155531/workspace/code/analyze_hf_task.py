import yaml, json, re, math, statistics
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path('.')
DATA = ROOT / 'data/2111.01152/2111.01152.yaml'
TEX = ROOT / 'data/2111.01152/2111.01152.tex'
SM = ROOT / 'data/2111.01152/2111.01152_SM.tex'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

raw = yaml.safe_load(DATA.read_text())
entries = [x for x in raw if isinstance(x, dict) and 'task' in x]

paper_text = TEX.read_text()
sm_text = SM.read_text()

paper_facts = {
    'system': 'AB-stacked MoTe2/WSe2 moire heterobilayer described by a two-layer continuum model',
    'single_particle_hamiltonian': r'H_\tau(r)=\begin{pmatrix}-\hbar^2 k^2/(2m_b)+\Delta_b(r) & \Delta_{T,\tau}(r) \\ \Delta_{T,\tau}^\dagger(r) & -\hbar^2 (k-\tau\kappa)^2/(2m_t)+\Delta_t(r)+V_{zt}\end{pmatrix}',
    'hole_basis_noninteracting': r'\hat{\mathcal H}_0=\sum_\tau \mathrm{Tr}\, h^{(\tau)}-\sum_{k_\alpha,k_\beta,l_\alpha,l_\beta,\tau}[h^{(\tau)}]^\intercal_{k_\alpha l_\alpha,k_\beta l_\beta} b^\dagger_{k_\alpha l_\alpha\tau} b_{k_\beta l_\beta\tau}',
    'interaction': r'\hat{\mathcal H}_{int}=\frac{1}{2A}\sum V(k_\alpha-k_\delta) b^\dagger_\alpha b^\dagger_\beta b_\gamma b_\delta \,\delta_{k_\alpha+k_\beta,k_\delta+k_\gamma}',
    'hf_interaction': r'\hat{\mathcal H}^{HF}_{int}=\frac{1}{A}\sum V(k_\alpha-k_\delta)\left(\langle b^\dagger_\alpha b_\delta\rangle b^\dagger_\beta b_\gamma-\langle b^\dagger_\alpha b_\gamma\rangle b^\dagger_\beta b_\delta\right)\delta_{k_\alpha+k_\beta,k_\delta+k_\gamma}',
    'coulomb': r'V(q)=2\pi e^2\tanh(qd)/(\epsilon q)',
}

records = []
for e in entries:
    score = e.get('score', {})
    total = sum(v for v in score.values() if isinstance(v, (int, float)))
    max_total = 2 * len([k for k,v in score.items() if isinstance(v, (int,float))])
    norm = total / max_total if max_total else None
    source_spans = []
    for fname, spans in (e.get('source') or {}).items():
        for span in spans:
            source_spans.append({'file': fname, 'start': span[0], 'end': span[1]})
    records.append({
        'task': e['task'],
        'answer': e.get('answer'),
        'score': score,
        'total_score': total,
        'max_score': max_total,
        'normalized_score': norm,
        'source_spans': source_spans,
        'n_placeholders': len(e.get('placeholder', {})),
    })

summary = {
    'n_tasks': len(records),
    'mean_total_score': statistics.mean(r['total_score'] for r in records),
    'mean_normalized_score': statistics.mean(r['normalized_score'] for r in records),
    'median_normalized_score': statistics.median(r['normalized_score'] for r in records),
    'tasks': records,
}

# Aggregate category scores
cat_vals = defaultdict(list)
for r in records:
    for k,v in r['score'].items():
        if isinstance(v,(int,float)):
            cat_vals[k].append(v)
summary['category_means'] = {k: statistics.mean(v) for k,v in cat_vals.items()}

# Derived observations
summary['top_tasks'] = sorted(records, key=lambda x: x['normalized_score'], reverse=True)[:5]
summary['bottom_tasks'] = sorted(records, key=lambda x: x['normalized_score'])[:5]

(OUT / 'hf_task_analysis.json').write_text(json.dumps(summary, indent=2))

# compact csv
lines = ['task,total_score,max_score,normalized_score']
for r in records:
    lines.append(f'"{r["task"]}",{r["total_score"]},{r["max_score"]},{r["normalized_score"]:.4f}')
(OUT / 'hf_task_scores.csv').write_text('\n'.join(lines))

print('wrote', OUT / 'hf_task_analysis.json')
print('wrote', OUT / 'hf_task_scores.csv')
print('mean normalized', summary['mean_normalized_score'])
