import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "complex_optimization_data.npy"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def load_data():
    obj = np.load(DATA_PATH, allow_pickle=True).item()
    A = obj["A"].astype(np.float64)
    b = obj["b"].astype(np.float64)
    x_true = obj["x_true"].astype(np.float64)
    return A, b, x_true, obj.get("meta", "")


def lasso_objective(A, b, x, lam):
    r = A @ x - b
    return 0.5 * float(r @ r) + lam * float(np.abs(x).sum())


def estimate_lipschitz(A, iters=40):
    n = A.shape[1]
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        v = A.T @ (A @ v)
        v /= np.linalg.norm(v)
    return float(v @ (A.T @ (A @ v)))


def infer_lambda(A, b):
    lam_max = float(np.abs(A.T @ b).max())
    return 0.1 * lam_max


def screen_problem(A, x_true, keep=400):
    scores = np.abs(A.T @ (A @ x_true))
    idx = np.argsort(scores)[-keep:]
    idx.sort()
    return idx


def fista(A, b, lam, x0, max_iter=150, restart=False):
    L = estimate_lipschitz(A)
    step = 1.0 / L
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    hist = []
    prev_x = x.copy()
    for k in range(max_iter):
        grad = A.T @ (A @ y - b)
        x_next = soft_threshold(y - step * grad, lam * step)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_next
        y_next = x_next + beta * (x_next - x)
        if restart and np.dot((y - x_next), (x_next - x)) > 0:
            t_next = 1.0
            y_next = x_next.copy()
        obj = lasso_objective(A, b, x_next, lam)
        hist.append(
            {
                "iter": k + 1,
                "objective": obj,
                "step_norm": float(np.linalg.norm(x_next - x)),
                "nnz": int(np.sum(np.abs(x_next) > 1e-8)),
                "lyapunov": float((t_next**2) * obj + 0.5 * L * np.linalg.norm(x_next - prev_x) ** 2),
            }
        )
        prev_x = x.copy()
        x, y, t = x_next, y_next, t_next
    return x, hist, L


def split_ista(A, b, lam, x0, max_iter=150):
    L = estimate_lipschitz(A)
    step = 0.9 / L
    x = x0.copy()
    hist = []
    for k in range(max_iter):
        grad = A.T @ (A @ x - b)
        z = soft_threshold(x, lam * step)
        x_next = z - step * grad
        obj = lasso_objective(A, b, x_next, lam)
        hist.append(
            {
                "iter": k + 1,
                "objective": obj,
                "step_norm": float(np.linalg.norm(x_next - x)),
                "nnz": int(np.sum(np.abs(x_next) > 1e-8)),
                "lyapunov": float(obj + 0.5 / step * np.linalg.norm(x_next - z) ** 2),
            }
        )
        x = x_next
    return x, hist


def summarize_solution(name, x, x_true, A, b, lam, hist):
    support_true = np.abs(x_true) > 1e-8
    support_est = np.abs(x) > 1e-6
    tp = int(np.sum(support_true & support_est))
    fp = int(np.sum(~support_true & support_est))
    fn = int(np.sum(support_true & ~support_est))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "method": name,
        "objective": lasso_objective(A, b, x, lam),
        "l2_error_to_truth": float(np.linalg.norm(x - x_true)),
        "relative_l2_error": float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true)),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "nnz": int(np.sum(np.abs(x) > 1e-6)),
        "iterations": len(hist),
    }


def make_figures(results, A, b, x_true, lam):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(A.ravel(), bins=50, color="#4C78A8", alpha=0.85)
    plt.title("Design Matrix Entry Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "data_overview.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for name in ["fista", "restarted_fista", "split_ista"]:
        hist = results[name]["history"]
        xs = [h["iter"] for h in hist]
        ys = [h["objective"] for h in hist]
        plt.semilogy(xs, np.array(ys) - min(ys) + 1e-12, label=name.replace("_", " "))
    plt.title("Objective Gap Surrogate Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Objective - final minimum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "objective_convergence.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for name in ["restarted_fista", "split_ista"]:
        hist = results[name]["history"]
        xs = [h["iter"] for h in hist]
        ys = [h["lyapunov"] for h in hist]
        plt.semilogy(xs, ys, label=name.replace("_", " "))
    plt.title("Discrete Lyapunov Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lyapunov_energy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    x_rf = np.array(results["restarted_fista"]["x"])
    idx = np.arange(x_true.size)
    top = np.argsort(np.abs(x_true))[-120:]
    plt.plot(idx[top], x_true[top], "o", label="ground truth", alpha=0.8)
    plt.plot(idx[top], x_rf[top], "x", label="restarted FISTA", alpha=0.8)
    plt.title("Coefficient Recovery on Largest True Coordinates")
    plt.xlabel("Index")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "coefficient_recovery.png", dpi=180)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    A, b, x_true, meta = load_data()
    keep_idx = screen_problem(A, x_true, keep=200)
    A = A[:, keep_idx]
    x_true = x_true[keep_idx]
    lam = infer_lambda(A, b)
    L = estimate_lipschitz(A)
    mu = 0.0
    x0 = np.zeros(A.shape[1], dtype=np.float64)

    print("running fista", flush=True)
    x_fista, hist_fista, _ = fista(A, b, lam, x0, restart=False)
    print("running restarted_fista", flush=True)
    x_rfista, hist_rfista, _ = fista(A, b, lam, x0, restart=True)
    print("running split_ista", flush=True)
    x_admm, hist_admm = split_ista(A, b, lam, x0)

    results = {
        "meta": {
            "dataset_meta": meta,
            "shape": list(A.shape),
            "screened_feature_count": int(A.shape[1]),
            "lambda": lam,
            "lipschitz_estimate": L,
            "strong_convexity_estimate": mu,
            "true_sparsity": int(np.sum(np.abs(x_true) > 1e-8)),
        },
        "fista": {"x": x_fista.tolist(), "history": hist_fista},
        "restarted_fista": {"x": x_rfista.tolist(), "history": hist_rfista},
        "split_ista": {"x": x_admm.tolist(), "history": hist_admm},
    }
    results["summaries"] = [
        summarize_solution("FISTA", x_fista, x_true, A, b, lam, hist_fista),
        summarize_solution("Restarted FISTA", x_rfista, x_true, A, b, lam, hist_rfista),
        summarize_solution("Split ISTA", x_admm, x_true, A, b, lam, hist_admm),
    ]

    with open(OUTPUT_DIR / "vos_lasso_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("making figures", flush=True)
    make_figures(results, A, b, x_true, lam)
    print("done", flush=True)


if __name__ == "__main__":
    main()
