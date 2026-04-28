"""Generate publication-quality PNG figures for the HF-Bench analysis."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "outputs")
IMGDIR = os.path.join(ROOT, "report", "images")
os.makedirs(IMGDIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GRADERS = ["Haining", "Will", "Yasaman"]
AXES = [
    "in_paper",
    "prompt_quality",
    "follow_instructions",
    "physics_logic",
    "math_derivation",
    "final_answer_accuracy",
]


def short_name(name, n=42):
    return name if len(name) <= n else name[: n - 1] + "…"


def load():
    df_ph = pd.read_csv(os.path.join(OUTDIR, "placeholder_scores.csv"))
    df_ans = pd.read_csv(os.path.join(OUTDIR, "answer_scores.csv"))
    per_task = pd.read_csv(os.path.join(OUTDIR, "per_task_summary.csv"))
    with open(os.path.join(OUTDIR, "summary.json")) as f:
        summary = json.load(f)
    return df_ph, df_ans, per_task, summary


def fig_placeholder_score_distribution(df_ph):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = [-0.5, 0.5, 1.5, 2.5]
    width = 0.27
    xs = np.array([0, 1, 2])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, g in enumerate(GRADERS):
        sub = df_ph.loc[df_ph["grader"] == g, "score"].dropna()
        counts, _ = np.histogram(sub, bins=bins)
        frac = counts / counts.sum()
        ax.bar(xs + (i - 1) * width, frac, width=width,
               label=g, color=colors[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0  (wrong)", "1  (partial)", "2  (correct)"])
    ax.set_ylabel("Fraction of placeholder judgements")
    ax.set_title("Placeholder-extraction score distribution per grader")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Grader", frameon=False)
    fig.tight_layout()
    out = os.path.join(IMGDIR, "fig1_placeholder_score_distribution.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_inter_rater_agreement(df_ph):
    # build wide table grader x (task, placeholder)
    wide = (df_ph.pivot_table(index=["task_index", "placeholder"],
                              columns="grader", values="score")
            .reset_index())
    M = np.zeros((len(GRADERS), len(GRADERS)))
    for i, gi in enumerate(GRADERS):
        for j, gj in enumerate(GRADERS):
            if i == j:
                M[i, j] = 1.0
                continue
            both = wide[[gi, gj]].dropna()
            if len(both) == 0:
                M[i, j] = np.nan
            else:
                M[i, j] = float((both[gi].values == both[gj].values).mean())

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(GRADERS)))
    ax.set_yticks(range(len(GRADERS)))
    ax.set_xticklabels(GRADERS)
    ax.set_yticklabels(GRADERS)
    for i in range(len(GRADERS)):
        for j in range(len(GRADERS)):
            ax.text(j, i, f"{M[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if M[i, j] < 0.65 else "black")
    ax.set_title("Pair-wise grader exact-agreement rate\n(placeholder scores)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(IMGDIR, "fig2_inter_rater_agreement.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_axis_means(df_ans):
    means = df_ans[AXES].mean()
    stds = df_ans[AXES].std()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(AXES))
    ax.bar(xs, means.values, yerr=stds.values,
           color="#4c72b0", edgecolor="black", capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([a.replace("_", "\n") for a in AXES], fontsize=9)
    ax.set_ylabel("Mean score across 16 tasks (0–2)")
    ax.set_ylim(0, 2.2)
    ax.set_title("Final-answer quality axes (mean ± SD across HF derivation steps)")
    for x, m in zip(xs, means.values):
        ax.text(x, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = os.path.join(IMGDIR, "fig3_answer_axis_means.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_per_task_score(per_task):
    pt = per_task.sort_values("task_index")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    xs = pt["task_index"].values
    ax.plot(xs, pt["placeholder_mean_score"], "o-",
            label="Placeholder mean (extraction)", color="#1f77b4")
    ax.plot(xs, pt["answer_mean_score"], "s--",
            label="Answer mean (derivation, 6 axes)", color="#d62728")
    ax.set_ylim(0, 2.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}\n{short_name(t,28)}"
                        for i, t in zip(xs, pt["task"])],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean score (0–2)")
    ax.set_title("Quality vs. position in the 16-step Hartree-Fock pipeline")
    ax.axhline(2.0, ls=":", color="gray", lw=0.8)
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    out = os.path.join(IMGDIR, "fig4_per_task_scores.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_axis_task_heatmap(df_ans):
    df = df_ans.sort_values("task_index").copy()
    M = df[AXES].values.astype(float)
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    im = ax.imshow(M, vmin=0, vmax=2, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels([a.replace("_", "\n") for a in AXES], fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r.task_index}. {short_name(r.task, 38)}"
                        for r in df.itertuples()], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color="black", fontsize=8)
    ax.set_title("Final-answer quality across tasks and axes")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Score (0–2)")
    fig.tight_layout()
    out = os.path.join(IMGDIR, "fig5_axis_task_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_score_breakdown(df_ph):
    # task-level placeholder full-credit rate vs. zero rate
    grp = (df_ph.dropna(subset=["score"])
           .groupby(["task_index", "task"])["score"]
           .agg(full=lambda s: float((s == 2).mean()),
                partial=lambda s: float((s == 1).mean()),
                wrong=lambda s: float((s == 0).mean()))
           .reset_index()
           .sort_values("task_index"))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    xs = grp["task_index"].values
    ax.bar(xs, grp["full"], color="#2ca02c", edgecolor="black", label="Correct (2)")
    ax.bar(xs, grp["partial"], bottom=grp["full"],
           color="#ffbb33", edgecolor="black", label="Partial (1)")
    ax.bar(xs, grp["wrong"], bottom=grp["full"] + grp["partial"],
           color="#d62728", edgecolor="black", label="Wrong (0)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}\n{short_name(t,28)}"
                        for i, t in zip(xs, grp["task"])],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of placeholder judgements")
    ax.set_ylim(0, 1.05)
    ax.set_title("Placeholder-extraction outcome composition per HF step")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(IMGDIR, "fig6_placeholder_breakdown.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_grader_means(summary):
    g = list(summary["per_grader_mean"].keys())
    v = list(summary["per_grader_mean"].values())
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(g, v, color="#7f7fbf", edgecolor="black")
    for x, y in zip(g, v):
        ax.text(x, y + 0.03, f"{y:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 2.1)
    ax.set_ylabel("Mean placeholder score")
    ax.set_title("Mean placeholder score per grader")
    fig.tight_layout()
    out = os.path.join(IMGDIR, "fig7_grader_means.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    df_ph, df_ans, per_task, summary = load()
    paths = []
    paths.append(fig_placeholder_score_distribution(df_ph))
    paths.append(fig_inter_rater_agreement(df_ph))
    paths.append(fig_axis_means(df_ans))
    paths.append(fig_per_task_score(per_task))
    paths.append(fig_axis_task_heatmap(df_ans))
    paths.append(fig_score_breakdown(df_ph))
    paths.append(fig_grader_means(summary))
    with open(os.path.join(OUTDIR, "figure_paths.json"), "w") as f:
        json.dump(paths, f, indent=2)
    print("Wrote figures:")
    for p in paths:
        print(" -", p)


if __name__ == "__main__":
    main()
