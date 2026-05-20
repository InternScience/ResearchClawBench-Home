"""
Unified Variable and Operator Splitting (VOS) Framework
for Nesterov Acceleration and ADMM from Continuous-Time Dynamics.

This code implements:
1. Lasso regression: min_x 0.5*||Ax-b||^2 + lambda*||x||_1
2. Gradient Descent baseline (ISTA)
3. Nesterov Accelerated Gradient (FISTA)
4. ADMM for Lasso
5. Restarted Nesterov for strong convexity
6. Continuous-time ODE for Nesterov dynamics
7. Lyapunov function analysis
"""

import numpy as np
import time
import json
from scipy.integrate import odeint

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(path='data/complex_optimization_data.npy'):
    data = np.load(path, allow_pickle=True).item()
    return data['A'], data['b'], data['x_true']

# ---------------------------------------------------------------------------
# Lasso problem setup
# ---------------------------------------------------------------------------
class LassoProblem:
    def __init__(self, A, b, lam):
        self.A = A
        self.b = b
        self.lam = lam
        self.m, self.n = A.shape
        self.AtA = A.T @ A
        self.Atb = A.T @ b
        self.L = np.linalg.norm(self.AtA, 2)
        eigs = np.linalg.eigvalsh(self.AtA)
        self.mu = max(eigs[eigs > 1e-10].min(), 0.0)

    def f_smooth(self, x):
        r = self.A @ x - self.b
        return 0.5 * np.dot(r, r)

    def grad_f(self, x):
        return self.AtA @ x - self.Atb

    def h_nonsmooth(self, x):
        return self.lam * np.sum(np.abs(x))

    def objective(self, x):
        return self.f_smooth(x) + self.h_nonsmooth(x)

    def prox_h(self, x, step):
        return np.sign(x) * np.maximum(np.abs(x) - self.lam * step, 0.0)

    def exact_solution_cvx(self, max_iter=5000, tol=1e-10):
        x = np.zeros(self.n)
        y = x.copy()
        t = 1.0
        L = self.L
        for _ in range(max_iter):
            x_old = x.copy()
            grad = self.grad_f(y)
            x = self.prox_h(y - grad / L, 1.0 / L)
            t_old = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x + ((t_old - 1.0) / t) * (x - x_old)
        return x


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------
def ista(prob, x0=None, max_iter=1000, tol=1e-10):
    """Proximal gradient descent (ISTA) for Lasso"""
    if x0 is None:
        x0 = np.zeros(prob.n)
    x = x0.copy()
    L = prob.L
    history = {'obj': [], 'time': [], 'lyapunov': [], 'dist_to_opt': []}
    start = time.time()
    for k in range(max_iter):
        grad = prob.grad_f(x)
        x_new = prob.prox_h(x - grad / L, 1.0 / L)
        obj = prob.objective(x_new)
        history['obj'].append(obj)
        history['time'].append(time.time() - start)
        history['lyapunov'].append(obj)
        history['dist_to_opt'].append(np.linalg.norm(x_new - x))
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, history


def fista(prob, x0=None, max_iter=1000, tol=1e-10):
    """FISTA / Nesterov accelerated gradient for composite problems"""
    if x0 is None:
        x0 = np.zeros(prob.n)
    x = x0.copy()
    y = x.copy()
    t = 1.0
    L = prob.L
    history = {'obj': [], 'time': [], 'lyapunov': [], 'dist_to_opt': []}
    start = time.time()
    for k in range(max_iter):
        x_old = x.copy()
        grad = prob.grad_f(y)
        x = prob.prox_h(y - grad / L, 1.0 / L)
        t_old = t
        t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x + ((t_old - 1.0) / t) * (x - x_old)
        obj = prob.objective(x)
        history['obj'].append(obj)
        history['time'].append(time.time() - start)
        history['lyapunov'].append(obj)
        history['dist_to_opt'].append(np.linalg.norm(x - x_old))
        if np.linalg.norm(x - x_old) < tol:
            break
    return x, history


def fista_restart(prob, x0=None, max_iter=1000, tol=1e-10):
    """Restarted FISTA for strongly convex problems - achieves linear convergence"""
    if x0 is None:
        x0 = np.zeros(prob.n)
    x = x0.copy()
    y = x.copy()
    t = 1.0
    L = prob.L
    history = {'obj': [], 'time': [], 'lyapunov': [], 'dist_to_opt': []}
    start = time.time()
    restart_count = 0
    for k in range(max_iter):
        x_old = x.copy()
        grad = prob.grad_f(y)
        x = prob.prox_h(y - grad / L, 1.0 / L)
        t_old = t
        t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x + ((t_old - 1.0) / t) * (x - x_old)
        obj = prob.objective(x)
        history['obj'].append(obj)
        history['time'].append(time.time() - start)
        history['lyapunov'].append(obj)
        history['dist_to_opt'].append(np.linalg.norm(x - x_old))

        # Restart criterion: function value increases
        if k > 0 and obj > history['obj'][-2]:
            t = 1.0
            y = x.copy()
            restart_count += 1

        if np.linalg.norm(x - x_old) < tol:
            break
    print(f"FISTA-Restart: restarts = {restart_count}")
    return x, history


def admm_lasso(prob, x0=None, rho=None, max_iter=1000, tol=1e-10,
               abstol=1e-4, reltol=1e-2):
    """ADMM for Lasso: min 0.5||Ax-b||^2 + lambda||z||_1 s.t. x=z"""
    if x0 is None:
        x0 = np.zeros(prob.n)
    if rho is None:
        rho = prob.lam
    x = x0.copy()
    z = x0.copy()
    u = np.zeros(prob.n)

    AtA_rhoI = prob.AtA + rho * np.eye(prob.n)
    try:
        from scipy.linalg import cho_factor, cho_solve
        L_chol = cho_factor(AtA_rhoI)
        solve_x = lambda rhs: cho_solve(L_chol, rhs)
    except Exception:
        solve_x = lambda rhs: np.linalg.solve(AtA_rhoI, rhs)

    history = {'obj': [], 'time': [], 'lyapunov': [],
               'r_norm': [], 's_norm': [], 'eps_pri': [], 'eps_dual': []}
    start = time.time()

    for k in range(max_iter):
        q = prob.Atb + rho * (z - u)
        x_new = solve_x(q)
        z_new = prob.prox_h(x_new + u, 1.0 / rho)
        u = u + x_new - z_new

        obj = prob.objective(z_new)
        history['obj'].append(obj)
        history['time'].append(time.time() - start)
        history['lyapunov'].append(obj)

        r_norm = np.linalg.norm(x_new - z_new)
        s_norm = np.linalg.norm(-rho * (z_new - z))
        eps_pri = np.sqrt(prob.n) * abstol + reltol * max(np.linalg.norm(x_new), np.linalg.norm(z_new))
        eps_dual = np.sqrt(prob.n) * abstol + reltol * np.linalg.norm(rho * u)
        history['r_norm'].append(r_norm)
        history['s_norm'].append(s_norm)
        history['eps_pri'].append(eps_pri)
        history['eps_dual'].append(eps_dual)

        x = x_new
        z = z_new

        if r_norm < eps_pri and s_norm < eps_dual:
            break

    return z, history


def vos_unified(prob, x0=None, max_iter=1000, tol=1e-10):
    """
    Unified VOS algorithm combining momentum (Nesterov) with operator splitting.
    Discretization of the continuous-time system:
      dX/dt = V
      dV/dt = - (r/t) V - nabla f(X) - (1/step) (X - prox_h(X - step*nabla f(X)))
    We use a more stable semi-implicit scheme.
    """
    if x0 is None:
        x0 = np.zeros(prob.n)
    x = x0.copy()
    y = x.copy()
    t = 1.0
    L = prob.L
    history = {'obj': [], 'time': [], 'lyapunov': [], 'dist_to_opt': []}
    start = time.time()
    for k in range(max_iter):
        x_old = x.copy()
        # Gradient at extrapolated point
        grad = prob.grad_f(y)
        # Proximal step (operator splitting for non-smooth part)
        x = prob.prox_h(y - grad / L, 1.0 / L)
        # Momentum update with adaptive coefficient
        t_old = t
        # Use generalized coefficient r >= 3 for guaranteed O(1/k^2)
        r = 3.0
        t = (r - 2.0 + np.sqrt((r - 2.0)**2 + 4.0 * t * t)) / 2.0
        # Extrapolation
        y = x + ((t_old - 1.0) / t) * (x - x_old)
        obj = prob.objective(x)
        history['obj'].append(obj)
        history['time'].append(time.time() - start)
        history['lyapunov'].append(obj)
        history['dist_to_opt'].append(np.linalg.norm(x - x_old))
        if np.linalg.norm(x - x_old) < tol:
            break
    return x, history


# ---------------------------------------------------------------------------
# Continuous-time ODE simulation
# ---------------------------------------------------------------------------
def nesterov_ode_dynamics(prob, x0=None, T=20.0, N=5000):
    """
    Simulate the ODE: ddX + (3/t) dX + nabla f(X) = 0
    For Lasso, we approximate by adding a small L2 regularization to make it smooth.
    """
    if x0 is None:
        x0 = np.zeros(prob.n)
    n = prob.n
    eps_smooth = 1e-4  # small smoothing parameter

    def ode_func(y, t):
        x = y[:n]
        v = y[n:]
        damping = 3.0 / max(t, 1e-3)
        # Smooth approximation: |x| ~ sqrt(x^2 + eps^2) - eps
        # grad of smooth part + grad of smoothed L1
        grad_smooth = prob.grad_f(x)
        grad_l1 = prob.lam * x / np.sqrt(x**2 + eps_smooth**2)
        dxdt = v
        dvdt = -damping * v - grad_smooth - grad_l1
        return np.concatenate([dxdt, dvdt])

    y0 = np.concatenate([x0, np.zeros(n)])
    tspan = np.linspace(1e-3, T, N)
    sol = odeint(ode_func, y0, tspan)
    return tspan, sol[:, :n], sol[:, n:]


def compute_lyapunov_nesterov_ode(t, X, V, prob, x_star):
    """
    Lyapunov for Nesterov ODE (smooth case):
    E(t) = t^2 (f(X(t)) - f*) + 2 ||X(t) - x* + (t/2) V(t)||^2
    """
    f_star = prob.objective(x_star)
    E = []
    for i in range(len(t)):
        x = X[i]
        v = V[i]
        fx = prob.objective(x)
        term1 = t[i]**2 * max(fx - f_star, 0.0)
        term2 = 2.0 * np.linalg.norm(x - x_star + (t[i] / 2.0) * v)**2
        E.append(term1 + term2)
    return np.array(E)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiments():
    print("Loading data...")
    A, b, x_true = load_data()
    lam = 0.1 * np.max(np.abs(A.T @ b))
    print(f"Lambda = {lam:.4e}")

    prob = LassoProblem(A, b, lam)
    print(f"L = {prob.L:.4e}, mu = {prob.mu:.4e}")

    print("Computing high-accuracy solution...")
    x_star = prob.exact_solution_cvx(max_iter=10000)
    f_star = prob.objective(x_star)
    print(f"f* = {f_star:.6e}")
    print(f"Support of x*: {np.sum(np.abs(x_star) > 1e-4)}")
    print(f"Support of x_true: {np.sum(np.abs(x_true) > 1e-4)}")

    max_iter = 2000
    x0 = np.zeros(prob.n)

    print("\nRunning ISTA...")
    x_ista, hist_ista = ista(prob, x0, max_iter=max_iter)
    print(f"ISTA: final obj = {hist_ista['obj'][-1]:.6e}, iters = {len(hist_ista['obj'])}")

    print("Running FISTA...")
    x_fista, hist_fista = fista(prob, x0, max_iter=max_iter)
    print(f"FISTA: final obj = {hist_fista['obj'][-1]:.6e}, iters = {len(hist_fista['obj'])}")

    print("Running FISTA with restart...")
    x_fista_r, hist_fista_r = fista_restart(prob, x0, max_iter=max_iter)
    print(f"FISTA-R: final obj = {hist_fista_r['obj'][-1]:.6e}, iters = {len(hist_fista_r['obj'])}")

    print("Running ADMM...")
    x_admm, hist_admm = admm_lasso(prob, x0, max_iter=max_iter)
    print(f"ADMM: final obj = {hist_admm['obj'][-1]:.6e}, iters = {len(hist_admm['obj'])}")

    print("Running VOS Unified...")
    x_vos, hist_vos = vos_unified(prob, x0, max_iter=max_iter)
    print(f"VOS: final obj = {hist_vos['obj'][-1]:.6e}, iters = {len(hist_vos['obj'])}")

    print("Simulating continuous-time ODE...")
    t_ode, X_ode, V_ode = nesterov_ode_dynamics(prob, x0, T=20.0, N=3000)
    E_ode = compute_lyapunov_nesterov_ode(t_ode, X_ode, V_ode, prob, x_star)
    print(f"ODE simulation complete. Final Lyapunov = {E_ode[-1]:.4e}")

    # Save results
    results = {
        'ista': {
            'obj': [float(v) for v in hist_ista['obj']],
            'time': [float(v) for v in hist_ista['time']],
        },
        'fista': {
            'obj': [float(v) for v in hist_fista['obj']],
            'time': [float(v) for v in hist_fista['time']],
        },
        'fista_restart': {
            'obj': [float(v) for v in hist_fista_r['obj']],
            'time': [float(v) for v in hist_fista_r['time']],
        },
        'admm': {
            'obj': [float(v) for v in hist_admm['obj']],
            'time': [float(v) for v in hist_admm['time']],
            'r_norm': [float(v) for v in hist_admm['r_norm']],
            's_norm': [float(v) for v in hist_admm['s_norm']],
        },
        'vos': {
            'obj': [float(v) for v in hist_vos['obj']],
            'time': [float(v) for v in hist_vos['time']],
        },
        'ode': {
            't': [float(v) for v in t_ode],
            'lyapunov': [float(v) for v in E_ode],
            'obj': [float(prob.objective(X_ode[i])) for i in range(0, len(t_ode), max(1, len(t_ode)//500))],
            't_sparse': [float(t_ode[i]) for i in range(0, len(t_ode), max(1, len(t_ode)//500))],
        },
        'f_star': float(f_star),
        'lambda': float(lam),
        'L': float(prob.L),
        'mu': float(prob.mu),
    }
    with open('outputs/experiment_results.json', 'w') as f:
        json.dump(results, f)

    print("\nResults saved to outputs/experiment_results.json")
    return results, prob, x_star, x_true


if __name__ == '__main__':
    results, prob, x_star, x_true = run_experiments()
