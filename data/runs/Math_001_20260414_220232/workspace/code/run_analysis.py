
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def objective(A, b, x, lam):
    r = A @ x - b
    return 0.5 / A.shape[0] * float(r @ r) + lam * float(np.sum(np.abs(x)))


def grad_smooth(A, b, x):
    return (A.T @ (A @ x - b)) / A.shape[0]


def estimate_L(A):
    _, s, _ = svds(A, k=1, which='LM')
    return float(s[-1] ** 2 / A.shape[0])


def fista(A, b, lam, x0, max_iter=300, L=None, x_star=None, F_star=None):
    n = x0.size
    if L is None:
        L = estimate_L(A)
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    hist = []
    for k in range(max_iter):
        g = grad_smooth(A, b, y)
        x_next = soft_threshold(y - g / L, lam / L)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = x_next + ((t - 1) / t_next) * (x_next - x)
        x = x_next
        t = t_next
        F = objective(A, b, x, lam)
        err = float(np.linalg.norm(x - x_star)) if x_star is not None else None
        gap = float(F - F_star) if F_star is not None else None
        lyap = (k + 1)**2 * gap + 2 * L * err**2 if (gap is not None and err is not None) else None
        hist.append({'iter': k + 1, 'obj': F, 'err_norm': err, 'obj_gap': gap, 'lyapunov': lyap})
    return x, hist, {'L': L}


def pgd(A, b, lam, x0, max_iter=300, L=None, x_star=None, F_star=None):
    if L is None:
        L = estimate_L(A)
    x = x0.copy()
    hist = []
    for k in range(max_iter):
        g = grad_smooth(A, b, x)
        x = soft_threshold(x - g / L, lam / L)
        F = objective(A, b, x, lam)
        err = float(np.linalg.norm(x - x_star)) if x_star is not None else None
        gap = float(F - F_star) if F_star is not None else None
        lyap = gap + 0.5 * L * err**2 if (gap is not None and err is not None) else None
        hist.append({'iter': k + 1, 'obj': F, 'err_norm': err, 'obj_gap': gap, 'lyapunov': lyap})
    return x, hist, {'L': L}


def admm_lasso(A, b, lam, x0, rho=1.0, max_iter=300, x_star=None, F_star=None):
    m, n = A.shape
    AtA = A.T @ A / m
    Atb = A.T @ b / m
    M = AtA + rho * np.eye(n)
    Minv = np.linalg.inv(M)
    x = x0.copy()
    z = x0.copy()
    u = np.zeros_like(x0)
    hist = []
    for k in range(max_iter):
        x = Minv @ (Atb + rho * (z - u))
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
        F = objective(A, b, z, lam)
        err = float(np.linalg.norm(z - x_star)) if x_star is not None else None
        gap = float(F - F_star) if F_star is not None else None
        primal = float(np.linalg.norm(x - z))
        dual = float(rho * np.linalg.norm(z - z_old))
        lyap = gap + rho * primal**2 + (1.0 / max(rho,1e-12)) * dual**2 if gap is not None else None
        hist.append({'iter': k + 1, 'obj': F, 'err_norm': err, 'obj_gap': gap, 'primal_res': primal, 'dual_res': dual, 'lyapunov': lyap})
    return z, hist, {'rho': rho}


def choose_lambda(A, b):
    lmax = float(np.max(np.abs(A.T @ b)) / A.shape[0])
    return 0.1 * lmax


def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    D = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
    A, b, x_true = D['A'], D['b'], D['x_true']
    x0 = np.zeros(A.shape[1])
    lam = choose_lambda(A, b)
    L = estimate_L(A)

    x_ref, ref_hist, _ = fista(A, b, lam, x0, max_iter=1500, L=L)
    F_star = objective(A, b, x_ref, lam)

    x_pgd, hist_pgd, meta_pgd = pgd(A, b, lam, x0, max_iter=300, L=L, x_star=x_ref, F_star=F_star)
    x_fista, hist_fista, meta_fista = fista(A, b, lam, x0, max_iter=300, L=L, x_star=x_ref, F_star=F_star)
    x_admm, hist_admm, meta_admm = admm_lasso(A, b, lam, x0, rho=1.0, max_iter=300, x_star=x_ref, F_star=F_star)

    results = {
        'lambda': lam,
        'L': L,
        'F_star_ref': F_star,
        'reference_solution_stats': {
            'nnz': int(np.count_nonzero(np.abs(x_ref) > 1e-8)),
            'l1_norm': float(np.linalg.norm(x_ref, 1)),
            'l2_norm': float(np.linalg.norm(x_ref)),
            'dist_to_x_true': float(np.linalg.norm(x_ref - x_true)),
        },
        'methods': {
            'pgd': {'final_obj': hist_pgd[-1]['obj'], 'final_gap': hist_pgd[-1]['obj_gap'], 'final_err': hist_pgd[-1]['err_norm']},
            'fista': {'final_obj': hist_fista[-1]['obj'], 'final_gap': hist_fista[-1]['obj_gap'], 'final_err': hist_fista[-1]['err_norm']},
            'admm': {'final_obj': hist_admm[-1]['obj'], 'final_gap': hist_admm[-1]['obj_gap'], 'final_err': hist_admm[-1]['err_norm'], 'final_primal_res': hist_admm[-1]['primal_res'], 'final_dual_res': hist_admm[-1]['dual_res']}
        }
    }
    with open('outputs/main_results.json','w') as f:
        json.dump(results, f, indent=2)
    with open('outputs/convergence_histories.json','w') as f:
        json.dump({'pgd': hist_pgd, 'fista': hist_fista, 'admm': hist_admm}, f)

    # table csv
    import csv
    with open('outputs/convergence_table.csv','w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['method','final_obj','final_gap','final_err','extra1','extra2'])
        w.writerow(['pgd',hist_pgd[-1]['obj'],hist_pgd[-1]['obj_gap'],hist_pgd[-1]['err_norm'],'',''])
        w.writerow(['fista',hist_fista[-1]['obj'],hist_fista[-1]['obj_gap'],hist_fista[-1]['err_norm'],'',''])
        w.writerow(['admm',hist_admm[-1]['obj'],hist_admm[-1]['obj_gap'],hist_admm[-1]['err_norm'],hist_admm[-1]['primal_res'],hist_admm[-1]['dual_res']])

    # figures
    it_p = [d['iter'] for d in hist_pgd]
    it_f = [d['iter'] for d in hist_fista]
    it_a = [d['iter'] for d in hist_admm]

    plt.figure(figsize=(7,4.5))
    plt.semilogy(it_p, [max(d['obj_gap'],1e-16) for d in hist_pgd], label='PGD')
    plt.semilogy(it_f, [max(d['obj_gap'],1e-16) for d in hist_fista], label='FISTA / Nesterov')
    plt.semilogy(it_a, [max(d['obj_gap'],1e-16) for d in hist_admm], label='ADMM')
    plt.xlabel('Iteration')
    plt.ylabel('Objective gap to reference')
    plt.title('Convergence comparison on ill-conditioned Lasso')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/convergence_gap.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7,4.5))
    plt.semilogy(it_p, [max(d['err_norm'],1e-16) for d in hist_pgd], label='PGD error')
    plt.semilogy(it_f, [max(d['err_norm'],1e-16) for d in hist_fista], label='FISTA error')
    plt.semilogy(it_a, [max(d['err_norm'],1e-16) for d in hist_admm], label='ADMM error')
    plt.xlabel('Iteration')
    plt.ylabel(r'$||x_k-x_*||_2$')
    plt.title('Distance to reference solution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/solution_error.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7,4.5))
    plt.semilogy(it_f, [max(d['lyapunov'],1e-16) for d in hist_fista], label='Accelerated Lyapunov surrogate')
    plt.semilogy(it_a, [max(d['lyapunov'],1e-16) for d in hist_admm], label='ADMM residual Lyapunov surrogate')
    plt.xlabel('Iteration')
    plt.ylabel('Lyapunov surrogate')
    plt.title('Strong Lyapunov-style diagnostics')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/lyapunov_diagnostics.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7,4.5))
    idx = np.arange(min(150, x_true.size))
    plt.plot(idx, x_true[:len(idx)], label='ground truth', lw=1.5)
    plt.plot(idx, x_ref[:len(idx)], label='estimated x*', lw=1.2)
    plt.xlabel('Coefficient index')
    plt.ylabel('Value')
    plt.title('First 150 coefficients: truth vs recovered solution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/coefficient_recovery.png', dpi=180)
    plt.close()

if __name__ == '__main__':
    import os
    main()
