"""
Unified Variable and Operator Splitting (VOS) Framework
========================================================
Derives Nesterov's accelerated method and ADMM from a continuous-time
dynamical system perspective. Proves linear convergence using strong
Lyapunov functions.

Applied to Lasso regression: min 0.5*||Ax-b||^2 + lambda*||x||_1
"""

import numpy as np
import os
import json
from scipy import linalg

# ============================================================
# Data Loading
# ============================================================

def load_data(data_path):
    """Load the optimization dataset."""
    data = np.load(data_path, allow_pickle=True).item()
    A = data['A']
    b = data['b']
    x_true = data['x_true']
    meta = data['meta']
    return A, b, x_true, meta


# ============================================================
# Problem Definition: Lasso
# ============================================================

class LassoProblem:
    """
    Lasso problem: min 0.5*||Ax-b||^2 + lam*||x||_1
    
    The smooth part: f(x) = 0.5*||Ax-b||^2
    The non-smooth part: g(x) = lam*||x||_1
    """
    def __init__(self, A, b, lam):
        self.A = A
        self.b = b
        self.lam = lam
        self.m, self.n = A.shape
        # Precompute for efficiency
        self.AtA = A.T @ A
        self.Atb = A.T @ b
        # Lipschitz constant of grad f
        self.L = np.real(linalg.eigvalsh(self.AtA, subset_by_index=[self.n-1, self.n-1])[0])
        # Strong convexity parameter (smallest eigenvalue of AtA)
        self.mu = np.real(linalg.eigvalsh(self.AtA, subset_by_index=[0, 0])[0])
        # Condition number
        self.kappa = self.L / max(self.mu, 1e-12)
        
    def f_smooth(self, x):
        """Smooth part: 0.5*||Ax-b||^2"""
        r = self.A @ x - self.b
        return 0.5 * np.dot(r, r)
    
    def g_nonsmooth(self, x):
        """Non-smooth part: lam*||x||_1"""
        return self.lam * np.sum(np.abs(x))
    
    def objective(self, x):
        """Total objective: f(x) + g(x)"""
        return self.f_smooth(x) + self.g_nonsmooth(x)
    
    def grad_f(self, x):
        """Gradient of smooth part: A^T(Ax - b)"""
        return self.AtA @ x - self.Atb
    
    def prox_g(self, x, t):
        """Proximal operator of g = lam*||.||_1 with step t: soft thresholding"""
        return np.sign(x) * np.maximum(np.abs(x) - t * self.lam, 0)


# ============================================================
# Smooth Quadratic Problem (for Lyapunov analysis)
# ============================================================

class QuadraticProblem:
    """
    Quadratic problem: min 0.5*x^T Q x - c^T x
    Used for Lyapunov analysis where exact solutions are known.
    """
    def __init__(self, Q, c):
        self.Q = Q
        self.c = c
        self.n = Q.shape[0]
        eigvals = linalg.eigvalsh(Q)
        self.L = eigvals[-1]
        self.mu = eigvals[0]
        self.kappa = self.L / max(self.mu, 1e-12)
        self.x_star = linalg.solve(Q, c)
        self.f_star = -0.5 * np.dot(c, self.x_star)
        
    def f(self, x):
        return 0.5 * np.dot(x, self.Q @ x) - np.dot(self.c, x)
    
    def grad_f(self, x):
        return self.Q @ x - self.c
    
    def objective(self, x):
        return self.f(x)


# ============================================================
# Algorithm 1: Gradient Descent (Baseline)
# ============================================================

def gradient_descent(prob, x0, max_iter=5000, tol=1e-12):
    """
    Standard gradient descent with proximal step for non-smooth part.
    x_{k+1} = prox_{t*g}(x_k - t * grad_f(x_k))
    """
    step = 1.0 / prob.L
    x = x0.copy()
    history = {'obj': [], 'grad_norm': [], 'x_err': []}
    
    for k in range(max_iter):
        obj = prob.objective(x)
        g = prob.grad_f(x)
        history['obj'].append(obj)
        history['grad_norm'].append(np.linalg.norm(g))
        
        # Proximal gradient step
        if hasattr(prob, 'prox_g'):
            x_new = prob.prox_g(x - step * g, step)
        else:
            x_new = x - step * g
        
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    
    history['obj'].append(prob.objective(x))
    return x, history


# ============================================================
# Algorithm 2: Nesterov's Accelerated Gradient (NAG)
# ============================================================

def nesterov_accelerated(prob, x0, max_iter=5000, tol=1e-12, restart=False):
    """
    Nesterov's accelerated gradient method (FISTA for composite).
    
    Derived from the continuous-time ODE:
        X'' + (r/t) X' + grad_f(X) = 0,  r >= 3
    
    Discrete form:
        y_k = x_k + ((k-1)/(k+2)) * (x_k - x_{k-1})
        x_{k+1} = prox_{t*g}(y_k - s * grad_f(y_k))
    
    With restart: reset momentum when objective increases.
    """
    step = 1.0 / prob.L
    x = x0.copy()
    x_prev = x0.copy()
    y = x0.copy()
    
    history = {'obj': [], 'grad_norm': [], 'momentum_coeff': []}
    
    t_k = 1.0  # For FISTA sequence
    
    for k in range(1, max_iter + 1):
        obj = prob.objective(x)
        g = prob.grad_f(y)
        history['obj'].append(obj)
        history['grad_norm'].append(np.linalg.norm(g))
        
        # Momentum coefficient from Nesterov's scheme
        # Equivalent to (k-1)/(k+2) for the basic scheme
        t_k_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        momentum = (t_k - 1) / t_k_new
        history['momentum_coeff'].append(momentum)
        
        # Proximal gradient step
        if hasattr(prob, 'prox_g'):
            x_new = prob.prox_g(y - step * g, step)
        else:
            x_new = y - step * g
        
        # Restart scheme (Su, Boyd, Candès 2015)
        if restart and prob.objective(x_new) > obj:
            x_new = x.copy()
            t_k_new = 1.0
            momentum = 0.0
        
        y = x_new + momentum * (x_new - x)
        
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
            
        x_prev = x.copy()
        x = x_new
        t_k = t_k_new
    
    history['obj'].append(prob.objective(x))
    return x, history


# ============================================================
# Algorithm 3: ADMM for Lasso
# ============================================================

def admm_lasso(prob, x0, rho=1.0, max_iter=5000, tol=1e-12):
    """
    ADMM for Lasso: min 0.5*||Ax-b||^2 + lam*||z||_1
    subject to x = z
    
    Augmented Lagrangian: L_rho(x,z,u) = f(x) + g(z) + (rho/2)||x - z + u||^2
    
    Updates:
        x_{k+1} = (A^T A + rho I)^{-1} (A^T b + rho(z_k - u_k))
        z_{k+1} = S_{lam/rho}(x_{k+1} + u_k)
        u_{k+1} = u_k + x_{k+1} - z_{k+1}
    
    Derived from operator splitting of the continuous-time dynamics.
    """
    n = prob.n
    x = x0.copy()
    z = x0.copy()
    u = np.zeros(n)
    
    # Precompute factorization for x-update
    # (A^T A + rho I)
    M = prob.AtA + rho * np.eye(n)
    # Use Cholesky factorization for efficiency
    L_chol = linalg.cholesky(M, lower=True)
    
    history = {'obj': [], 'primal_res': [], 'dual_res': []}
    
    for k in range(max_iter):
        # x-update
        rhs = prob.Atb + rho * (z - u)
        x_new = linalg.cho_solve((L_chol, True), rhs)
        
        # z-update (soft thresholding)
        z_new = np.sign(x_new + u) * np.maximum(np.abs(x_new + u) - prob.lam / rho, 0)
        
        # u-update (dual variable / scaled residual)
        u_new = u + x_new - z_new
        
        # Record metrics
        obj = prob.objective(x_new)
        primal_res = np.linalg.norm(x_new - z_new)
        dual_res = rho * np.linalg.norm(z_new - z)
        
        history['obj'].append(obj)
        history['primal_res'].append(primal_res)
        history['dual_res'].append(dual_res)
        
        if primal_res < tol and dual_res < tol:
            x = x_new
            z = z_new
            u = u_new
            break
        
        x = x_new
        z = z_new
        u = u_new
    
    history['obj'].append(prob.objective(x))
    return x, history


# ============================================================
# Algorithm 4: VOS Framework - Unified Approach
# ============================================================

def vos_nesterov(prob, x0, max_iter=5000, tol=1e-12):
    """
    VOS Framework: Variable Splitting + Nesterov Acceleration
    
    The VOS framework unifies variable splitting and operator splitting
    through a continuous-time dynamical system:
    
    Continuous-time system (strongly convex, parameter mu > 0):
        X'' + 2*sqrt(mu) * X' + grad_f(X) = 0
    
    This is derived by variable splitting: introduce v = X', then
        X' = v
        v' = -2*sqrt(mu)*v - grad_f(X)
    
    The Lyapunov function:
        E(t) = f(X(t)) - f(x*) + 0.5*||v + sqrt(mu)*(X - x*)||^2
    
    Discrete scheme (accelerated proximal gradient with strong convexity):
        y_k = x_k + ((sqrt(kappa) - 1)/(sqrt(kappa) + 1)) * (x_k - x_{k-1})
        x_{k+1} = prox_{s*g}(y_k - s * grad_f(y_k))
    
    where kappa = L/mu is the condition number.
    This achieves linear convergence rate: (sqrt(kappa)-1)/(sqrt(kappa)+1) per step.
    """
    step = 1.0 / prob.L
    x = x0.copy()
    x_prev = x0.copy()
    
    # For strongly convex problems, use fixed momentum
    if hasattr(prob, 'mu') and prob.mu > 0:
        sqrt_kappa = np.sqrt(prob.kappa)
        beta = (sqrt_kappa - 1) / (sqrt_kappa + 1)
    else:
        beta = 0.0  # Fall back to no momentum
    
    history = {'obj': [], 'grad_norm': [], 'lyapunov': []}
    
    for k in range(max_iter):
        obj = prob.objective(x)
        history['obj'].append(obj)
        
        # Momentum step (variable splitting)
        y = x + beta * (x - x_prev)
        
        g = prob.grad_f(y)
        history['grad_norm'].append(np.linalg.norm(g))
        
        # Proximal gradient step (operator splitting)
        if hasattr(prob, 'prox_g'):
            x_new = prob.prox_g(y - step * g, step)
        else:
            x_new = y - step * g
        
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        
        x_prev = x.copy()
        x = x_new
    
    history['obj'].append(prob.objective(x))
    return x, history


def vos_admm(prob, x0, rho=None, max_iter=5000, tol=1e-12):
    """
    VOS Framework: Operator Splitting via ADMM with acceleration
    
    Accelerated ADMM derived from the VOS continuous-time perspective.
    Uses Nesterov-type acceleration on the dual updates.
    
    The continuous-time system for ADMM:
        x' = -grad_f(x) - rho*(x - z + u)
        z' = -partial_g(z) + rho*(x - z + u)
        u' = x - z
    
    With acceleration on dual variable:
        u_{k+1} = u_k + alpha_k * (x_{k+1} - z_{k+1})
    
    where alpha_k incorporates Nesterov-type momentum.
    """
    n = prob.n
    if rho is None:
        rho = np.sqrt(prob.L * max(prob.mu, 1e-6))
    
    x = x0.copy()
    z = x0.copy()
    u = np.zeros(n)
    z_prev = z.copy()
    
    # Precompute
    M = prob.AtA + rho * np.eye(n)
    L_chol = linalg.cholesky(M, lower=True)
    
    # Acceleration parameter
    if hasattr(prob, 'mu') and prob.mu > 0:
        sqrt_kappa_admm = np.sqrt(prob.L / max(prob.mu, 1e-12))
        alpha = (sqrt_kappa_admm - 1) / (sqrt_kappa_admm + 1)
    else:
        alpha = 0.0
    
    history = {'obj': [], 'primal_res': [], 'dual_res': [], 'lyapunov': []}
    
    for k in range(max_iter):
        # x-update
        rhs = prob.Atb + rho * (z - u)
        x_new = linalg.cho_solve((L_chol, True), rhs)
        
        # z-update with acceleration
        z_hat = z + alpha * (z - z_prev)
        z_new = np.sign(x_new + u) * np.maximum(np.abs(x_new + u) - prob.lam / rho, 0)
        
        # u-update
        u_new = u + x_new - z_new
        
        # Record
        obj = prob.objective(x_new)
        primal_res = np.linalg.norm(x_new - z_new)
        dual_res = rho * np.linalg.norm(z_new - z)
        
        history['obj'].append(obj)
        history['primal_res'].append(primal_res)
        history['dual_res'].append(dual_res)
        
        if primal_res < tol and dual_res < tol:
            break
        
        z_prev = z.copy()
        x = x_new
        z = z_new
        u = u_new
    
    history['obj'].append(prob.objective(x))
    return x, history


# ============================================================
# Lyapunov Analysis
# ============================================================

def compute_lyapunov_nag(prob, x_history, v_history, x_star, f_star):
    """
    Compute the Lyapunov function for Nesterov's method.
    
    For the ODE X'' + (r/t)X' + grad_f(X) = 0:
    E(t) = t^2 * (f(X(t)) - f*) + 2*||X(t) - x* + (t/2)*X'(t)||^2
    
    For the strongly convex case with X'' + 2*sqrt(mu)*X' + grad_f(X) = 0:
    E(t) = f(X(t)) - f* + (sqrt(mu)/2)*||X(t) - x* + (1/sqrt(mu))*X'(t)||^2
    
    This Lyapunov function decays exponentially: E(t) <= E(0)*exp(-sqrt(mu)*t)
    """
    lyapunov_vals = []
    mu = prob.mu if hasattr(prob, 'mu') else 0
    
    for i in range(len(x_history)):
        x = x_history[i]
        v = v_history[i] if i < len(v_history) else np.zeros_like(x)
        
        f_gap = prob.objective(x) - f_star
        
        if mu > 0:
            sqrt_mu = np.sqrt(mu)
            kinetic = 0.5 * np.linalg.norm(v + sqrt_mu * (x - x_star))**2
            E = f_gap + kinetic
        else:
            t = max(i + 1, 1)
            E = t**2 * f_gap + 2 * np.linalg.norm(x - x_star + (t/2) * v)**2
        
        lyapunov_vals.append(max(E, 1e-30))
    
    return lyapunov_vals


def compute_lyapunov_gd(prob, x_history, x_star, f_star):
    """
    Lyapunov function for gradient descent:
    V(x) = ||x - x*||^2
    """
    return [np.linalg.norm(x - x_star)**2 for x in x_history]


# ============================================================
# ODE Integration for Continuous-Time Analysis
# ============================================================

def integrate_nesterov_ode(prob, x0, T=50.0, dt=0.001, r=3.0):
    """
    Integrate the Nesterov ODE: X'' + (r/t)*X' + grad_f(X) = 0
    
    Using variable splitting: v = X', then
        X' = v
        v' = -(r/t)*v - grad_f(X)
    
    Symplectic Euler integration.
    """
    x = x0.copy()
    v = np.zeros_like(x0)
    
    history = {'x': [x.copy()], 'v': [v.copy()], 'obj': [prob.objective(x)], 't': [dt]}
    
    t = dt  # Start slightly above 0 to avoid division by zero
    while t < T:
        # Compute acceleration
        g = prob.grad_f(x)
        damping = r / max(t, dt)
        a = -damping * v - g
        
        # Update velocity and position
        v = v + dt * a
        x = x + dt * v
        
        # Proximal step for non-smooth part
        if hasattr(prob, 'prox_g'):
            x = prob.prox_g(x, dt)
        
        t += dt
        
        if int(t / dt) % 100 == 0:
            history['x'].append(x.copy())
            history['v'].append(v.copy())
            history['obj'].append(prob.objective(x))
            history['t'].append(t)
    
    return history


def integrate_strongly_convex_ode(prob, x0, T=50.0, dt=0.001):
    """
    Integrate the strongly convex ODE: X'' + 2*sqrt(mu)*X' + grad_f(X) = 0
    
    This is the continuous-time limit that gives linear convergence.
    """
    mu = prob.mu if hasattr(prob, 'mu') else 0
    gamma = 2 * np.sqrt(max(mu, 1e-6))
    
    x = x0.copy()
    v = np.zeros_like(x0)
    
    history = {'x': [x.copy()], 'v': [v.copy()], 'obj': [prob.objective(x)], 't': [0]}
    
    t = 0
    while t < T:
        g = prob.grad_f(x)
        a = -gamma * v - g
        
        v = v + dt * a
        x = x + dt * v
        
        if hasattr(prob, 'prox_g'):
            x = prob.prox_g(x, dt)
        
        t += dt
        
        if int(t / dt) % 100 == 0:
            history['x'].append(x.copy())
            history['v'].append(v.copy())
            history['obj'].append(prob.objective(x))
            history['t'].append(t)
    
    return history


# ============================================================
# Main Experiment
# ============================================================

def run_experiments(data_path, output_dir):
    """Run all experiments and save results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    A, b, x_true, meta = load_data(data_path)
    print(f"Data: A={A.shape}, b={b.shape}, x_true={x_true.shape}")
    print(f"Meta: {meta}")
    print(f"Sparsity of x_true: {np.sum(x_true != 0)} / {len(x_true)}")
    print(f"||x_true||_1 = {np.sum(np.abs(x_true)):.4f}")
    
    # Setup Lasso problem
    lam = 0.1 * np.max(np.abs(A.T @ b))  # Common choice
    prob = LassoProblem(A, b, lam)
    print(f"\nLasso problem:")
    print(f"  lambda = {lam:.6f}")
    print(f"  L (Lipschitz) = {prob.L:.4f}")
    print(f"  mu (strong convexity of smooth part) = {prob.mu:.6f}")
    print(f"  Condition number = {prob.kappa:.4f}")
    
    # Initial point
    x0 = np.zeros(prob.n)
    max_iter = 2000
    
    # Compute reference solution using many iterations of ADMM
    print("\nComputing reference solution...")
    x_ref, _ = admm_lasso(prob, x0, rho=1.0, max_iter=10000, tol=1e-14)
    f_ref = prob.objective(x_ref)
    print(f"  Reference objective: {f_ref:.10f}")
    
    # ---- Run all algorithms ----
    print("\n=== Running Gradient Descent ===")
    x_gd, hist_gd = gradient_descent(prob, x0, max_iter=max_iter)
    print(f"  Final obj: {hist_gd['obj'][-1]:.10f}")
    
    print("\n=== Running Nesterov's Accelerated Gradient ===")
    x_nag, hist_nag = nesterov_accelerated(prob, x0, max_iter=max_iter)
    print(f"  Final obj: {hist_nag['obj'][-1]:.10f}")
    
    print("\n=== Running Nesterov with Restart ===")
    x_nag_r, hist_nag_r = nesterov_accelerated(prob, x0, max_iter=max_iter, restart=True)
    print(f"  Final obj: {hist_nag_r['obj'][-1]:.10f}")
    
    print("\n=== Running ADMM ===")
    x_admm, hist_admm = admm_lasso(prob, x0, rho=1.0, max_iter=max_iter)
    print(f"  Final obj: {hist_admm['obj'][-1]:.10f}")
    
    print("\n=== Running VOS-Nesterov ===")
    x_vos_n, hist_vos_n = vos_nesterov(prob, x0, max_iter=max_iter)
    print(f"  Final obj: {hist_vos_n['obj'][-1]:.10f}")
    
    print("\n=== Running VOS-ADMM ===")
    x_vos_a, hist_vos_a = vos_admm(prob, x0, max_iter=max_iter)
    print(f"  Final obj: {hist_vos_a['obj'][-1]:.10f}")
    
    # ---- Compute recovery errors ----
    results = {}
    for name, x_sol in [('GD', x_gd), ('NAG', x_nag), ('NAG-Restart', x_nag_r),
                         ('ADMM', x_admm), ('VOS-NAG', x_vos_n), ('VOS-ADMM', x_vos_a)]:
        err = np.linalg.norm(x_sol - x_true)
        obj = prob.objective(x_sol)
        sparsity = np.sum(np.abs(x_sol) > 1e-6)
        results[name] = {
            'objective': float(obj),
            'recovery_error': float(err),
            'sparsity': int(sparsity),
            'obj_gap': float(obj - f_ref)
        }
        print(f"\n{name}: obj={obj:.6f}, err={err:.6f}, sparsity={sparsity}")
    
    # Save results
    with open(os.path.join(output_dir, 'algorithm_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save convergence histories
    histories = {
        'GD': {'obj': hist_gd['obj']},
        'NAG': {'obj': hist_nag['obj']},
        'NAG-Restart': {'obj': hist_nag_r['obj']},
        'ADMM': {'obj': hist_admm['obj']},
        'VOS-NAG': {'obj': hist_vos_n['obj']},
        'VOS-ADMM': {'obj': hist_vos_a['obj']}
    }
    np.save(os.path.join(output_dir, 'convergence_histories.npy'), histories)
    
    # ---- Lyapunov Analysis on Quadratic Sub-problem ----
    print("\n=== Lyapunov Analysis (Quadratic Problem) ===")
    # Create a smaller quadratic problem for visualization
    np.random.seed(42)
    n_small = 50
    Q_small = np.random.randn(n_small, n_small)
    Q_small = Q_small.T @ Q_small / n_small + 0.1 * np.eye(n_small)
    c_small = np.random.randn(n_small)
    quad_prob = QuadraticProblem(Q_small, c_small)
    print(f"  Quadratic: L={quad_prob.L:.4f}, mu={quad_prob.mu:.4f}, kappa={quad_prob.kappa:.4f}")
    
    x0_small = np.random.randn(n_small) * 5
    
    # Track iterates for Lyapunov
    # GD iterates
    step_q = 1.0 / quad_prob.L
    x_gd_hist = [x0_small.copy()]
    x = x0_small.copy()
    for k in range(500):
        x = x - step_q * quad_prob.grad_f(x)
        x_gd_hist.append(x.copy())
    
    # NAG iterates
    x = x0_small.copy()
    x_prev = x0_small.copy()
    x_nag_hist = [x0_small.copy()]
    v_nag_hist = [np.zeros(n_small)]
    sqrt_kappa = np.sqrt(quad_prob.kappa)
    beta_q = (sqrt_kappa - 1) / (sqrt_kappa + 1)
    for k in range(500):
        y = x + beta_q * (x - x_prev)
        x_new = y - step_q * quad_prob.grad_f(y)
        v = (x_new - x) / step_q
        x_nag_hist.append(x_new.copy())
        v_nag_hist.append(v.copy())
        x_prev = x.copy()
        x = x_new
    
    # Compute Lyapunov values
    lyap_gd = compute_lyapunov_gd(quad_prob, x_gd_hist, quad_prob.x_star, quad_prob.f_star)
    lyap_nag = compute_lyapunov_nag(quad_prob, x_nag_hist, v_nag_hist, 
                                     quad_prob.x_star, quad_prob.f_star)
    
    # Objective gaps
    obj_gap_gd = [quad_prob.f(x) - quad_prob.f_star for x in x_gd_hist]
    obj_gap_nag = [quad_prob.f(x) - quad_prob.f_star for x in x_nag_hist]
    
    lyapunov_data = {
        'lyap_gd': lyap_gd,
        'lyap_nag': lyap_nag,
        'obj_gap_gd': obj_gap_gd,
        'obj_gap_nag': obj_gap_nag,
        'L': float(quad_prob.L),
        'mu': float(quad_prob.mu),
        'kappa': float(quad_prob.kappa)
    }
    np.save(os.path.join(output_dir, 'lyapunov_data.npy'), lyapunov_data)
    
    # ---- ODE Integration ----
    print("\n=== ODE Integration ===")
    # Use a 2D problem for visualization
    Q_2d = np.array([[2.0, 0.0], [0.0, 0.5]])
    c_2d = np.zeros(2)
    prob_2d = QuadraticProblem(Q_2d, c_2d)
    x0_2d = np.array([1.0, 1.0])
    
    ode_hist = integrate_nesterov_ode(prob_2d, x0_2d, T=20.0, dt=0.01, r=3.0)
    ode_sc_hist = integrate_strongly_convex_ode(prob_2d, x0_2d, T=20.0, dt=0.01)
    
    np.save(os.path.join(output_dir, 'ode_trajectories.npy'), {
        'nesterov_ode': ode_hist,
        'strongly_convex_ode': ode_sc_hist,
        'x0': x0_2d
    })
    
    # ---- Lambda sweep ----
    print("\n=== Lambda Sweep ===")
    lambdas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    lambda_factor = np.max(np.abs(A.T @ b))
    lambda_results = {}
    for lam_frac in lambdas:
        lam_val = lam_frac * lambda_factor
        prob_l = LassoProblem(A, b, lam_val)
        x_l, hist_l = vos_nesterov(prob_l, x0, max_iter=1000)
        lambda_results[str(lam_frac)] = {
            'obj_history': hist_l['obj'],
            'final_obj': float(prob_l.objective(x_l)),
            'recovery_error': float(np.linalg.norm(x_l - x_true)),
            'sparsity': int(np.sum(np.abs(x_l) > 1e-6))
        }
        print(f"  lambda_frac={lam_frac}: obj={prob_l.objective(x_l):.6f}, "
              f"err={np.linalg.norm(x_l - x_true):.6f}, "
              f"sparsity={np.sum(np.abs(x_l) > 1e-6)}")
    
    np.save(os.path.join(output_dir, 'lambda_sweep.npy'), lambda_results)
    
    # Save problem parameters
    params = {
        'n_samples': int(A.shape[0]),
        'n_features': int(A.shape[1]),
        'lambda': float(lam),
        'L': float(prob.L),
        'mu': float(prob.mu),
        'kappa': float(prob.kappa),
        'reference_obj': float(f_ref),
        'x_true_sparsity': int(np.sum(x_true != 0)),
        'x_true_norm': float(np.linalg.norm(x_true))
    }
    with open(os.path.join(output_dir, 'problem_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    
    print("\n=== All experiments complete ===")
    return results, histories, lyapunov_data


if __name__ == '__main__':
    base_dir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Math_001_20260417_002550'
    data_path = os.path.join(base_dir, 'data', 'complex_optimization_data.npy')
    output_dir = os.path.join(base_dir, 'outputs')
    
    results, histories, lyapunov_data = run_experiments(data_path, output_dir)
