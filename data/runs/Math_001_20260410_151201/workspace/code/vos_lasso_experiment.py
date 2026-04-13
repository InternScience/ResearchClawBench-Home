import json, os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

DATA_PATH = 'data/complex_optimization_data.npy'
OUT_JSON = 'outputs/results.json'
OUT_NPZ = 'outputs/trajectories.npz'
IMG_DIR = 'report/images'
os.makedirs('outputs', exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def obj(A, b, x, lam, mu):
    r = A @ x - b
    return 0.5 * np.dot(r, r) + 0.5 * mu * np.dot(x, x) + lam * np.abs(x).sum()


def smooth_obj(A, b, x, mu):
    r = A @ x - b
    return 0.5 * np.dot(r, r) + 0.5 * mu * np.dot(x, x)


def grad(A, b, x, mu):
    return A.T @ (A @ x - b) + mu * x


def estimate_lipschitz(A, mu):
    smax = np.linalg.svd(A, compute_uv=False)[0]
    return smax**2 + mu


def power_norm(M, n_iter=200):
    v = np.random.randn(M.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        v = M @ v
        n = np.linalg.norm(v)
        v /= n
    return np.linalg.norm(M @ v)


def ista(A, b, lam, mu, x0, steps, L, x_star=None, f_star=None):
    x = x0.copy()
    traj = {'obj': [], 'dist': [], 'step': []}
    t = 1.0 / L
    for k in range(steps):
        x = soft_threshold(x - t * grad(A, b, x, mu), lam * t)
        fx = obj(A, b, x, lam, mu)
        traj['obj'].append(fx)
        traj['step'].append(k + 1)
        if x_star is not None:
            traj['dist'].append(float(np.linalg.norm(x - x_star)))
    return x, traj


def fista(A, b, lam, mu, x0, steps, L, x_star=None):
    x = x0.copy(); y = x0.copy(); tk = 1.0
    traj = {'obj': [], 'dist': [], 'step': [], 'lyap': []}
    t = 1.0 / L
    for k in range(steps):
        x_next = soft_threshold(y - t * grad(A, b, y, mu), lam * t)
        tk_next = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
        y = x_next + ((tk - 1) / tk_next) * (x_next - x)
        x, tk = x_next, tk_next
        fx = obj(A, b, x, lam, mu)
        traj['obj'].append(fx)
        traj['step'].append(k + 1)
        if x_star is not None:
            dist = float(np.linalg.norm(x - x_star))
            traj['dist'].append(dist)
            traj['lyap'].append(float((k+1)**2 * max(fx - obj(A,b,x_star,lam,mu),0) + 0.5*L*dist**2))
    return x, traj


def restarted_fista(A, b, lam, mu, x0, steps, L, x_star=None):
    x = x0.copy(); y = x0.copy(); tk = 1.0
    traj = {'obj': [], 'dist': [], 'step': [], 'restarts': []}
    t = 1.0 / L
    for k in range(steps):
        x_next = soft_threshold(y - t * grad(A, b, y, mu), lam * t)
        if np.dot(y - x_next, x_next - x) > 0:
            y = x.copy()
            tk = 1.0
            x_next = soft_threshold(y - t * grad(A, b, y, mu), lam * t)
            traj['restarts'].append(k + 1)
        tk_next = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
        y = x_next + ((tk - 1) / tk_next) * (x_next - x)
        x, tk = x_next, tk_next
        fx = obj(A, b, x, lam, mu)
        traj['obj'].append(fx)
        traj['step'].append(k + 1)
        if x_star is not None:
            traj['dist'].append(float(np.linalg.norm(x - x_star)))
    return x, traj


def admm_lasso(A, b, lam, mu, x0, steps, rho, x_star=None):
    n = A.shape[1]
    x = x0.copy(); z = x0.copy(); u = np.zeros_like(x0)
    M = A.T @ A + (mu + rho) * np.eye(n)
    Minv = np.linalg.inv(M)
    Atb = A.T @ b
    traj = {'obj': [], 'dist': [], 'primal': [], 'dual': [], 'step': [], 'lyap': []}
    z_prev = z.copy()
    f_star = obj(A,b,x_star,lam,mu) if x_star is not None else None
    for k in range(steps):
        x = Minv @ (Atb + rho * (z - u))
        z_prev = z.copy()
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
        fx = obj(A, b, z, lam, mu)
        r = np.linalg.norm(x - z)
        s = rho * np.linalg.norm(z - z_prev)
        traj['obj'].append(fx)
        traj['primal'].append(float(r))
        traj['dual'].append(float(s))
        traj['step'].append(k + 1)
        if x_star is not None:
            dist = float(np.linalg.norm(z - x_star))
            traj['dist'].append(dist)
            traj['lyap'].append(float(max(fx - f_star,0) + 0.5*rho*r*r + 0.5*mu*dist*dist))
    return z, traj


def main():
    data = np.load(DATA_PATH, allow_pickle=True).item()
    A, b, x_true = data['A'], data['b'], data['x_true']
    m, n = A.shape
    lam = 0.05 * np.max(np.abs(A.T @ b))
    mu = 1e-3
    x0 = np.zeros(n)
    L = estimate_lipschitz(A, mu)
    steps = 150

    x_ref, ref_traj = restarted_fista(A, b, lam, mu, x0, 4000, L)
    f_star = obj(A, b, x_ref, lam, mu)

    xi, ti = ista(A, b, lam, mu, x0, steps, L, x_ref, f_star)
    xf, tf = fista(A, b, lam, mu, x0, steps, L, x_ref)
    xr, tr = restarted_fista(A, b, lam, mu, x0, steps, L, x_ref)
    xa, ta = admm_lasso(A, b, lam, mu, x0, steps, rho=1.0, x_star=x_ref)

    results = {
        'lambda': lam,
        'mu': mu,
        'L': L,
        'f_star': f_star,
        'nnz_true': int((np.abs(x_true) > 1e-12).sum()),
        'nnz_ref': int((np.abs(x_ref) > 1e-8).sum()),
        'recovery_corr': float(np.corrcoef(x_ref, x_true)[0,1]),
        'final_obj': {
            'ISTA': ti['obj'][-1],
            'FISTA': tf['obj'][-1],
            'Restarted FISTA': tr['obj'][-1],
            'ADMM': ta['obj'][-1],
            'Reference': f_star,
        },
        'final_dist_to_ref': {
            'ISTA': ti['dist'][-1],
            'FISTA': tf['dist'][-1],
            'Restarted FISTA': tr['dist'][-1],
            'ADMM': ta['dist'][-1],
        },
        'restarts': tr['restarts'],
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(OUT_NPZ,
             ista_obj=np.array(ti['obj']), fista_obj=np.array(tf['obj']), rfista_obj=np.array(tr['obj']), admm_obj=np.array(ta['obj']),
             ista_dist=np.array(ti['dist']), fista_dist=np.array(tf['dist']), rfista_dist=np.array(tr['dist']), admm_dist=np.array(ta['dist']),
             fista_lyap=np.array(tf['lyap']), admm_lyap=np.array(ta['lyap']),
             admm_primal=np.array(ta['primal']), admm_dual=np.array(ta['dual']), x_true=x_true, x_ref=x_ref)

    # Figure 1: objective gap
    plt.figure(figsize=(7,5))
    for name, arr in [('ISTA', ti['obj']), ('FISTA', tf['obj']), ('Restarted FISTA', tr['obj']), ('ADMM', ta['obj'])]:
        plt.semilogy(np.maximum(np.array(arr) - f_star, 1e-16), label=name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective gap')
    plt.title('Convergence on strongly convex Lasso')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/objective_gap.png', dpi=200)
    plt.close()

    # Figure 2: distance to reference optimum
    plt.figure(figsize=(7,5))
    for name, arr in [('ISTA', ti['dist']), ('FISTA', tf['dist']), ('Restarted FISTA', tr['dist']), ('ADMM', ta['dist'])]:
        plt.semilogy(np.maximum(np.array(arr), 1e-16), label=name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel(r'$||x_k-x^*||_2$')
    plt.title('State convergence under VOS-inspired discretizations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/distance_to_optimum.png', dpi=200)
    plt.close()

    # Figure 3: Lyapunov surrogates
    plt.figure(figsize=(7,5))
    plt.semilogy(np.maximum(np.array(tf['lyap']),1e-16), label='FISTA energy surrogate', linewidth=2)
    plt.semilogy(np.maximum(np.array(ta['lyap']),1e-16), label='ADMM Lyapunov surrogate', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy / Lyapunov value')
    plt.title('Strong Lyapunov functions decrease along trajectories')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/lyapunov_decay.png', dpi=200)
    plt.close()

    # Figure 4: ADMM residuals
    plt.figure(figsize=(7,5))
    plt.semilogy(np.maximum(np.array(ta['primal']),1e-16), label='Primal residual', linewidth=2)
    plt.semilogy(np.maximum(np.array(ta['dual']),1e-16), label='Dual residual', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Residual norm')
    plt.title('ADMM residual convergence')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/admm_residuals.png', dpi=200)
    plt.close()

    # Figure 5: coefficient recovery
    idx = np.argsort(-np.abs(x_true))[:30]
    width = 0.4
    plt.figure(figsize=(10,5))
    xs = np.arange(len(idx))
    plt.bar(xs - width/2, x_true[idx], width, label='Ground truth')
    plt.bar(xs + width/2, x_ref[idx], width, label='Recovered')
    plt.xlabel('Top support index rank')
    plt.ylabel('Coefficient value')
    plt.title('Recovery of dominant sparse coefficients')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/coefficient_recovery.png', dpi=200)
    plt.close()

if __name__ == '__main__':
    main()
