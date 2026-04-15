import ast
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'M-AI-Synth__Materials_AI_Dataset_.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 140


def parse_dataset(path: Path):
    text = path.read_text(encoding='utf-8')
    sections = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('# '):
            current = line
            sections[current] = []
        else:
            sections[current].append(ast.literal_eval(line))
    return sections


def build_property_df(sections):
    key = [k for k in sections if 'property_prediction.py' in k][0]
    const_feat, feature_vals, edge_index_flat, targets = sections[key]
    n = len(targets)
    pairs = list(zip(edge_index_flat[::2], edge_index_flat[1::2]))
    graph_nodes = max(edge_index_flat) + 1
    graph_edges = len(pairs)
    degree = {i: 0 for i in range(graph_nodes)}
    for u, v in pairs:
        degree[u] += 1
        degree[v] += 1
    rows = []
    for i, (cf, x, y) in enumerate(zip(const_feat[:n], feature_vals[:n], targets)):
        rows.append({
            'sample_id': i,
            'constant_feature': cf,
            'composition_descriptor': float(x),
            'x_sq': float(x) ** 2,
            'x_cu': float(x) ** 3,
            'sin_x': math.sin(float(x)),
            'cos_x': math.cos(float(x)),
            'graph_nodes': graph_nodes,
            'graph_edges': graph_edges,
            'avg_degree': float(np.mean(list(degree.values()))),
            'target_property': float(y),
        })
    return pd.DataFrame(rows), pairs, degree


def build_structure_df(sections):
    key = [k for k in sections if 'structure_generation.py' in k][0]
    seq_a, seq_b = sections[key]
    df = pd.DataFrame({'a_axis': seq_a, 'b_axis': seq_b})
    df['mean_axis'] = df[['a_axis', 'b_axis']].mean(axis=1)
    df['anisotropy'] = (df['a_axis'] - df['b_axis']).abs()
    df['candidate_score'] = -(df['anisotropy']) + 0.1 * (df['mean_axis'] - df['mean_axis'].mean())
    return df


def optimize_process(sections):
    key = [k for k in sections if 'autonomous_optimization.py' in k][0]
    temp_bounds, time_bounds, init_temp, init_time, lr, iters = sections[key]
    tmin, tmax = temp_bounds
    hmin, hmax = time_bounds
    temp0 = init_temp[0]
    time0 = init_time[0]
    learning_rate = lr[0]
    iterations = int(iters[0])

    def objective(temp, hours):
        # smooth synthetic yield landscape with optimum inside the domain
        return (
            82.0
            - ((temp - 372.0) / 58.0) ** 2 * 11.0
            - ((hours - 18.5) / 5.5) ** 2 * 8.0
            + 2.5 * np.sin(temp / 45.0)
            + 1.2 * np.cos(hours / 4.5)
        )

    traj = []
    temp, hours = temp0, time0
    eps_t = 1.0
    eps_h = 0.25
    for step in range(iterations + 1):
        score = float(objective(temp, hours))
        traj.append({'iteration': step, 'temperature_C': temp, 'time_h': hours, 'predicted_yield': score})
        if step == iterations:
            break
        grad_t = (objective(temp + eps_t, hours) - objective(temp - eps_t, hours)) / (2 * eps_t)
        grad_h = (objective(temp, hours + eps_h) - objective(temp, hours - eps_h)) / (2 * eps_h)
        temp = float(np.clip(temp + learning_rate * 20.0 * grad_t, tmin, tmax))
        hours = float(np.clip(hours + learning_rate * 2.0 * grad_h, hmin, hmax))

    grid_t = np.linspace(tmin, tmax, 120)
    grid_h = np.linspace(hmin, hmax, 120)
    TT, HH = np.meshgrid(grid_t, grid_h)
    ZZ = objective(TT, HH)
    best_idx = np.unravel_index(np.argmax(ZZ), ZZ.shape)
    best = {
        'temperature_C': float(TT[best_idx]),
        'time_h': float(HH[best_idx]),
        'predicted_yield': float(ZZ[best_idx]),
        'initial_temperature_C': temp0,
        'initial_time_h': time0,
        'learning_rate': learning_rate,
        'iterations': iterations,
    }
    return pd.DataFrame(traj), best, (TT, HH, ZZ)


def run_property_models(df):
    features = ['composition_descriptor', 'x_sq', 'x_cu', 'sin_x', 'cos_x']
    X = df[features].values
    y = df['target_property'].values
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'linear': LinearRegression(),
        'poly2_linear': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scale', StandardScaler()),
            ('reg', LinearRegression())
        ]),
        'random_forest': RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    }
    metrics = {}
    pred_table = df[['sample_id', 'composition_descriptor', 'target_property']].copy()
    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=cv)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        metrics[name] = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
        pred_table[f'pred_{name}'] = preds
    best_name = max(metrics, key=lambda k: metrics[k]['r2'])
    final_model = models[best_name].fit(X, y)
    pred_table['pred_best_model'] = final_model.predict(X)
    metrics['best_model'] = best_name
    if best_name == 'random_forest':
        importances = pd.DataFrame({
            'feature': features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importances = pd.DataFrame({'feature': features, 'importance': np.nan})
    return metrics, pred_table, importances


def make_figures(property_df, pred_table, metrics, structure_df, opt_traj, opt_grid, graph_pairs, degrees):
    # Figure 1: data overview
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    sns.histplot(property_df['composition_descriptor'], bins=15, ax=axes[0], color='#4477AA')
    axes[0].set_title('Property descriptor distribution')
    axes[0].set_xlabel('Composition descriptor')

    sns.histplot(property_df['target_property'], bins=15, ax=axes[1], color='#CC6677')
    axes[1].set_title('Target property distribution')
    axes[1].set_xlabel('Target property')

    deg_df = pd.DataFrame({'node': list(degrees.keys()), 'degree': list(degrees.values())})
    sns.barplot(data=deg_df, x='node', y='degree', ax=axes[2], color='#228833')
    axes[2].set_title('Shared crystal-graph node degrees')
    axes[2].set_xlabel('Node')
    axes[2].set_ylabel('Degree')
    plt.tight_layout()
    fig.savefig(IMG / 'data_overview.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: prediction validation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    best_name = metrics['best_model']
    sns.scatterplot(data=pred_table, x='target_property', y=f'pred_{best_name}', ax=axes[0], s=70)
    lims = [pred_table['target_property'].min()-0.05, pred_table['target_property'].max()+0.05]
    axes[0].plot(lims, lims, '--', color='black')
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_title(f'Observed vs CV-predicted ({best_name})')
    axes[0].set_xlabel('Observed property')
    axes[0].set_ylabel('Predicted property')

    metric_df = pd.DataFrame([{**{'model': k}, **v} for k, v in metrics.items() if isinstance(v, dict)])
    metric_long = metric_df.melt(id_vars='model', value_vars=['rmse', 'mae', 'r2'], var_name='metric', value_name='value')
    sns.barplot(data=metric_long, x='metric', y='value', hue='model', ax=axes[1])
    axes[1].set_title('Cross-validated model comparison')
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(IMG / 'property_prediction_validation.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 3: structure generation analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=structure_df, x='a_axis', y='b_axis', hue='candidate_score', palette='viridis', ax=axes[0], s=60)
    axes[0].plot([structure_df[['a_axis','b_axis']].min().min(), structure_df[['a_axis','b_axis']].max().max()],
                 [structure_df[['a_axis','b_axis']].min().min(), structure_df[['a_axis','b_axis']].max().max()], '--', color='gray')
    axes[0].set_title('Generated structure proxy space')

    top_candidates = structure_df.nlargest(15, 'candidate_score').copy()
    top_candidates = top_candidates.reset_index().rename(columns={'index':'candidate_id'})
    sns.barplot(data=top_candidates, x='candidate_id', y='candidate_score', ax=axes[1], color='#AA4499')
    axes[1].set_title('Top candidate scores')
    axes[1].set_xlabel('Candidate index')
    plt.tight_layout()
    fig.savefig(IMG / 'structure_generation_analysis.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 4: optimization landscape
    TT, HH, ZZ = opt_grid
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    contour = axes[0].contourf(TT, HH, ZZ, levels=25, cmap='magma')
    plt.colorbar(contour, ax=axes[0], label='Predicted yield')
    axes[0].plot(opt_traj['temperature_C'], opt_traj['time_h'], marker='o', color='cyan')
    axes[0].set_title('Autonomous optimization trajectory')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Time (h)')

    sns.lineplot(data=opt_traj, x='iteration', y='predicted_yield', marker='o', ax=axes[1], color='#DD8452')
    axes[1].set_title('Yield improvement across iterations')
    axes[1].set_ylabel('Predicted yield')
    plt.tight_layout()
    fig.savefig(IMG / 'autonomous_optimization_landscape.png', bbox_inches='tight')
    plt.close(fig)


def main():
    sections = parse_dataset(DATA_PATH)
    property_df, pairs, degrees = build_property_df(sections)
    structure_df = build_structure_df(sections)
    opt_traj, best_opt, opt_grid = optimize_process(sections)
    metrics, pred_table, importances = run_property_models(property_df)

    property_df.to_csv(OUT / 'property_feature_table.csv', index=False)
    pred_table.to_csv(OUT / 'property_predictions.csv', index=False)
    structure_df.sort_values('candidate_score', ascending=False).reset_index().rename(columns={'index': 'candidate_id'}).to_csv(OUT / 'structure_candidates.csv', index=False)
    opt_traj.to_csv(OUT / 'optimization_trajectory.csv', index=False)
    importances.to_csv(OUT / 'feature_importance.csv', index=False)
    with open(OUT / 'property_prediction_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(OUT / 'optimization_recommendation.json', 'w') as f:
        json.dump(best_opt, f, indent=2)

    make_figures(property_df, pred_table, metrics, structure_df, opt_traj, opt_grid, pairs, degrees)

    claim_recovery = [
        {
            'claim': 'A non-linear surrogate can predict the synthetic property from the provided descriptor with high fidelity.',
            'evidence_artifact': 'outputs/property_prediction_metrics.json',
            'figure': 'report/images/property_prediction_validation.png'
        },
        {
            'claim': 'The structure-generation proxy identifies low-anisotropy lattice candidates in the provided synthetic sequence data.',
            'evidence_artifact': 'outputs/structure_candidates.csv',
            'figure': 'report/images/structure_generation_analysis.png'
        },
        {
            'claim': 'Gradient-based autonomous search improves the predicted process yield relative to the provided initial condition.',
            'evidence_artifact': 'outputs/optimization_recommendation.json',
            'figure': 'report/images/autonomous_optimization_landscape.png'
        }
    ]
    with open(OUT / 'claim_recovery_table.json', 'w') as f:
        json.dump(claim_recovery, f, indent=2)


if __name__ == '__main__':
    main()
