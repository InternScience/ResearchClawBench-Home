"""
Variable and Operator Splitting (VOS) Framework
================================================
Unified derivation of Nesterov's accelerated method and ADMM
from a continuous-time dynamical system perspective.

Implements:
1. Nesterov's accelerated proximal gradient (FISTA) for Lasso
2. ADMM for Lasso
3. Continuous-time ODE analogue and Lyapunov analysis
4. Comparison and validation on high-dimensional data
"""

import numpy as np
from numpy.linalg import norm
import time
import json
import os

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
    A = data['A']
    b = data['b']
    x_true = data['x_true']
    return A, b, x_true

# ============================================================
# OBJECTIVE FUNCTION FOR LASSO
# ============================================================
def lasso_obj(A, b, x, lam):
    """Lasso objective: 0.5 * ||Ax - b||^2 + lam * ||x||_1"""
    residual = A @ x - b
    return 0.5 * norm(residual)**2 + lam * norm(x, 1)

def lasso_grad(A, b, x):
    """Gradient of the smooth part: A^T(Ax - b)"""
    return A.T @ (A @ x - b)

def soft_threshold(x, alpha):
    """Proximal operator for L1 norm (soft thresholding)"""
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def prox_l1(x, lam, step):
    """Proximal operator for lam * ||x||_1 with step size"""
    return soft_threshold(x, lam * step)

# ============================================================
# LIPSCHITZ CONSTANT ESTIMATION
# ============================================================
def estimate_lipschitz(A):
    """Estimate Lipschitz constant L = ||A^T A||_2 = sigma_max(A)^2"""
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    return s[0]**2

# ============================================================
# METHOD 1: ISTA (Proximal Gradient Descent) - Baseline
# ============================================================
def ista(A, b, lam, x0, L, max_iter=5000, tol=1e-8):
    """Iterative Shrinkage-Thresholding Algorithm (standard proximal gradient)"""
    step = 1.0 / L
    x = x0.copy()
    n = len(x0)
    
    history = {
        'obj': [], 'grad_norm': [], 'time': [], 'iter': []
    }
    
    t_start = time.time()
    for k in range(max_iter):
        grad = lasso_grad(A, b, x)
        x_new = soft_threshold(x - step * grad, lam * step)
        
        if k % 50 == 0:
            obj = lasso_obj(A, b, x_new, lam)
            history['obj'].append(obj)
            history['grad_norm'].append(norm(grad))
            history['time'].append(time.time() - t_start)
            history['iter'].append(k)
            
            # Check convergence
            if k > 0 and abs(history['obj'][-1] - history['obj'][-2]) < tol * abs(history['obj'][-1]):
                break
        
        x = x_new
    
    return x, history

# ============================================================
# METHOD 2: FISTA (Nesterov's Accelerated Proximal Gradient)
# ============================================================
def fista(A, b, lam, x0, L, max_iter=5000, tol=1e-8, restart=True):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm
    Nesterov's accelerated proximal gradient method with adaptive restart.
    """
    step = 1.0 / L
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    
    history = {
        'obj': [], 'grad_norm': [], 'time': [], 'iter': [],
        't_values': [], 'restart_points': []
    }
    
    t_start = time.time()
    for k in range(max_iter):
        x_old = x.copy()
        grad_y = lasso_grad(A, b, y)
        x = soft_threshold(y - step * grad_y, lam * step)
        
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        
        # Adaptive restart (O'Donoghue & Candès 2015)
        if restart:
            momentum_term = (x_old - x)
            grad_x = lasso_grad(A, b, x)
            if np.dot(grad_y.flatten(), momentum_term.flatten()) > 0:
                t = 1.0
                y = x.copy()
                history['restart_points'].append(k)
        
        if k % 50 == 0:
            obj = lasso_obj(A, b, x, lam)
            history['obj'].append(obj)
            history['grad_norm'].append(norm(lasso_grad(A, b, x)))
            history['time'].append(time.time() - t_start)
            history['iter'].append(k)
            history['t_values'].append(t)
            
            if k > 0 and abs(history['obj'][-1] - history['obj'][-2]) < tol * abs(history['obj'][-1]):
                break
        
    return x, history

# ============================================================
# METHOD 3: ADMM for Lasso
# ============================================================
def admm_lasso(A, b, lam, x0, rho=None, max_iter=5000, tol=1e-8):
    """
    ADMM for Lasso: minimize 0.5||Ax - b||^2 + lam||z||_1 s.t. x = z
    
    Scaled form updates:
    x^{k+1} = argmin_x 0.5||Ax-b||^2 + (rho/2)||x - z^k + u^k||^2
    z^{k+1} = S_{lam/rho}(x^{k+1} + u^k)
    u^{k+1} = u^k + x^{k+1} - z^{k+1}
    """
    n_features = A.shape[1]
    
    if rho is None:
        # Use a reasonable rho based on the data scale
        rho = 1.0
    
    x = x0.copy()
    z = x0.copy()
    u = np.zeros(n_features)
    
    # Precompute for x-update efficiency
    AtA = A.T @ A
    Atb = A.T @ b
    # x-update: (AtA + rho*I)x = Atb + rho*(z - u)
    # Solve using Cholesky or CG
    M = AtA + rho * np.eye(n_features)
    
    history = {
        'obj': [], 'primal_res': [], 'dual_res': [], 'time': [], 'iter': []
    }
    
    t_start = time.time()
    
    # Use conjugate gradient for large problems
    for k in range(max_iter):
        # x-update
        rhs = Atb + rho * (z - u)
        x, _ = cg_solve(M, rhs, x0=x, max_iter=100, tol=1e-12)
        
        # z-update (soft thresholding)
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        
        # u-update (dual variable)
        u = u + x - z
        
        if k % 50 == 0:
            obj = lasso_obj(A, b, x, lam)
            primal_res = norm(x - z)
            dual_res = norm(-rho * (z - z_old))
            history['obj'].append(obj)
            history['primal_res'].append(primal_res)
            history['dual_res'].append(dual_res)
            history['time'].append(time.time() - t_start)
            history['iter'].append(k)
            
            if primal_res < tol and dual_res < tol:
                break
    
    return x, history

def cg_solve(A, b_vec, x0=None, max_iter=200, tol=1e-8):
    """Conjugate gradient solver for Ax = b"""
    n = len(b_vec)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    r = b_vec - A @ x
    p = r.copy()
    rsold = np.dot(r, r)
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x, np.sqrt(rsnew)

# ============================================================
# CONTINUOUS-TIME DYNAMICAL SYSTEM (VOS Framework)
# ============================================================
def vos_continuous_time_simulation(A, b, lam, x0, L, T=100, n_steps=5000):
    """
    Simulate the continuous-time ODE from the VOS framework:
    
    ẍ + (3/t)ẋ + ∇f(x) + ∂g(x) ∋ 0
    
    where f(x) = 0.5||Ax-b||^2 (smooth) and g(x) = lam||x||_1 (non-smooth).
    
    For numerical simulation with non-smooth term, we use a splitting approach:
    - Smooth dynamics: ẍ + (3/t)ẋ + ∇f(x) = 0
    - Proximal correction: periodic soft-thresholding
    
    This is the unified VOS perspective: the continuous-time dynamics
    naturally encompass both Nesterov acceleration (through the 3/t damping)
    and operator splitting (through the proximal steps).
    """
    dt = T / n_steps
    n_features = len(x0)
    
    x = x0.copy()
    v = np.zeros(n_features)  # velocity ẋ
    
    history = {
        't': [], 'obj': [], 'lyapunov': [], 'x_norm': []
    }
    
    # For numerical stability near t=0, we start from a small positive t
    eps = 1e-6
    t = eps
    
    # Lyapunov function: E(t) = t^2(f(x) - f*) + 2||x + (t/2)v - x*||^2
    # Since we don't know x*, we use a reference value
    # We'll compute a reference by running FISTA
    x_ref, _ = fista(A, b, lam, x0, L, max_iter=5000, tol=1e-10)
    f_ref = 0.5 * norm(A @ x_ref - b)**2 + lam * norm(x_ref, 1)
    
    for i in range(n_steps):
        t = eps + i * dt
        
        # Gradient of smooth part
        grad = lasso_grad(A, b, x)
        
        # Acceleration from ODE: ẍ = - (3/t) v - grad
        if t > eps:
            a = -(3.0 / t) * v - grad
        else:
            a = -grad
        
        # Semi-implicit Euler for velocity
        v = v + dt * a
        
        # Update position
        x = x + dt * v
        
        # Periodic proximal correction (operator splitting for non-smooth term)
        # This applies the proximal operator of g at discrete intervals,
        # mimicking the splitting nature of ADMM in the continuous flow
        prox_interval = max(1, n_steps // 200)  # ~200 proximal steps
        if i % prox_interval == 0:
            x = soft_threshold(x, lam * dt * prox_interval)
        
        if i % (n_steps // 200) == 0:
            obj = lasso_obj(A, b, x, lam)
            lyap = t**2 * (obj - f_ref) + 2 * norm(x + (t/2)*v - x_ref)**2
            history['t'].append(t)
            history['obj'].append(obj)
            history['lyapunov'].append(lyap)
            history['x_norm'].append(norm(x))
    
    return x, history

# ============================================================
# LYAPUNOV FUNCTION ANALYSIS 
# ============================================================
def compute_lyapunov_convergence(A, b, lam, x0, L):
    """
    Track Lyapunov functions for both Nesterov/FISTA and ADMM.
    
    For Nesterov's method, the discrete Lyapunov function is:
        E_k = k(k+1)(f(x_k) - f*) + 2||z_k - x*||^2
    
    For ADMM, the Lyapunov function is:
        E_k = ||x_k - x*||^2 + rho||z_k - z*||^2 + (1/rho)||y_k - y*||^2
    """
    # Get reference optimal solution
    x_ref, _ = fista(A, b, lam, x0, L, max_iter=5000, tol=1e-12)
    f_ref = lasso_obj(A, b, x_ref, lam)
    
    # FISTA with Lyapunov tracking
    step = 1.0 / L
    x = x0.copy()
    y = x0.copy()
    t_val = 1.0
    
    fista_lyap = {'k': [], 'E': [], 'f_gap': []}
    
    for k in range(2000):
        x_old = x.copy()
        grad_y = lasso_grad(A, b, y)
        x = soft_threshold(y - step * grad_y, lam * step)
        
        t_new = (1 + np.sqrt(1 + 4 * t_val**2)) / 2
        y = x + ((t_val - 1) / t_new) * (x - x_old)
        t_val = t_new
        
        if k % 20 == 0:
            f_val = lasso_obj(A, b, x, lam)
            f_gap = f_val - f_ref
            z_k = x + t_val * (x - x_old)  # momentum state
            E = t_val**2 * f_gap + 2 * norm(z_k - x_ref)**2
            fista_lyap['k'].append(k)
            fista_lyap['E'].append(E)
            fista_lyap['f_gap'].append(f_gap)
    
    # ADMM with Lyapunov tracking  
    n_features = A.shape[1]
    rho = 1.0
    x_admm = x0.copy()
    z_admm = x0.copy()
    u = np.zeros(n_features)
    AtA = A.T @ A
    Atb = A.T @ b
    M = AtA + rho * np.eye(n_features)
    
    admm_lyap = {'k': [], 'E': [], 'f_gap': []}
    
    for k in range(2000):
        rhs = Atb + rho * (z_admm - u)
        x_admm, _ = cg_solve(M, rhs, x0=x_admm, max_iter=100, tol=1e-12)
        
        z_admm = soft_threshold(x_admm + u, lam / rho)
        u = u + x_admm - z_admm
        
        if k % 20 == 0:
            f_val = lasso_obj(A, b, x_admm, lam)
            f_gap = f_val - f_ref
            # ADMM Lyapunov function
            E = norm(x_admm - x_ref)**2 + rho * norm(z_admm - x_ref)**2
            admm_lyap['k'].append(k)
            admm_lyap['E'].append(E)
            admm_lyap['f_gap'].append(f_gap)
    
    f_ref_val = lasso_obj(A, b, x_ref, lam)
    
    return fista_lyap, admm_lyap, f_ref_val, x_ref

# ============================================================
# GENERALIZED VOS: PARAMETER SWEEP OVER DAMPING COEFFICIENT r
# ============================================================
def generalized_vos_sweep(A, b, lam, x0, L, r_values=[1, 2, 2.5, 3, 4, 5], n_iter=2000):
    """
    Test generalized Nesterov scheme with damping r in ODE:
    ẍ + (r/t)ẋ + ∇f(x) = 0
    
    Theory predicts O(1/t^2) convergence iff r >= 3.
    """
    step = 1.0 / L
    
    results = {}
    for r in r_values:
        x = x0.copy()
        y = x0.copy()
        t_val = 1.0
        
        obj_history = []
        
        for k in range(n_iter):
            x_old = x.copy()
            grad_y = lasso_grad(A, b, y)
            x = soft_threshold(y - step * grad_y, lam * step)
            
            # Generalized momentum with damping r
            # For discrete scheme: y_{k+1} = x_k + (k-1)/(k+r-1) * (x_k - x_{k-1})
            # For continuous limit: ẍ + (r/t)ẋ + ∇f = 0
            theta_k = (k) / (k + r) if k > 0 else 0
            y = x + theta_k * (x - x_old)
            
            if k % 20 == 0:
                obj_history.append(lasso_obj(A, b, x, lam))
        
        results[r] = np.array(obj_history)
    
    return results

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("=" * 60)
    print("Variable and Operator Splitting (VOS) Framework")
    print("Unified Nesterov Acceleration & ADMM")
    print("=" * 60)
    
    # Load data
    A, b, x_true = load_data()
    print(f"\nData: A ({A.shape[0]}x{A.shape[1]}), condition number = 10")
    print(f"x_true: {np.count_nonzero(x_true)} nonzeros out of {len(x_true)}")
    
    # Setup
    L = estimate_lipschitz(A)
    lam = 0.1 * L / A.shape[0]  # regularization parameter
    x0 = np.zeros(A.shape[1])
    max_iter = 3000
    
    print(f"\nLipschitz constant L = {L:.2f}")
    print(f"Regularization lambda = {lam:.6f}")
    
    # Run all methods
    print("\n" + "-" * 40)
    print("Running ISTA (baseline)...")
    t0 = time.time()
    x_ista, hist_ista = ista(A, b, lam, x0, L, max_iter=max_iter)
    print(f"  Time: {time.time()-t0:.2f}s, Final obj: {lasso_obj(A,b,x_ista,lam):.6f}")
    
    print("\nRunning FISTA (Nesterov acceleration)...")
    t0 = time.time()
    x_fista, hist_fista = fista(A, b, lam, x0, L, max_iter=max_iter)
    print(f"  Time: {time.time()-t0:.2f}s, Final obj: {lasso_obj(A,b,x_fista,lam):.6f}")
    print(f"  Restarts: {len(hist_fista['restart_points'])}")
    
    print("\nRunning ADMM...")
    t0 = time.time()
    x_admm, hist_admm = admm_lasso(A, b, lam, x0, max_iter=max_iter)
    print(f"  Time: {time.time()-t0:.2f}s, Final obj: {lasso_obj(A,b,x_admm,lam):.6f}")
    
    print("\nRunning VOS Continuous-Time Simulation...")
    t0 = time.time()
    x_vos, hist_vos = vos_continuous_time_simulation(A, b, lam, x0, L, T=50)
    print(f"  Time: {time.time()-t0:.2f}s, Final obj: {lasso_obj(A,b,x_vos,lam):.6f}")
    
    print("\nComputing Lyapunov analysis...")
    fista_lyap, admm_lyap, f_ref, x_ref = compute_lyapunov_convergence(A, b, lam, x0, L)
    print(f"  Reference optimal objective: {f_ref:.8f}")
    
    print("\nRunning generalized VOS parameter sweep...")
    sweep_results = generalized_vos_sweep(A, b, lam, x0, L)
    
    # Save all results
    os.makedirs('outputs', exist_ok=True)
    
    results = {
        'hist_ista': {k: (v if not isinstance(v, list) else v) for k, v in hist_ista.items()},
        'hist_fista': {k: (v if not isinstance(v, list) else v) for k, v in hist_fista.items()},
        'hist_admm': {k: (v if not isinstance(v, list) else v) for k, v in hist_admm.items()},
        'hist_vos': {k: (v if not isinstance(v, list) else v) for k, v in hist_vos.items()},
    }
    
    # Save convergence histories
    np.savez('outputs/convergence_histories.npz',
             ista_obj=hist_ista['obj'],
             ista_iter=hist_ista['iter'],
             fista_obj=hist_fista['obj'],
             fista_iter=hist_fista['iter'],
             admm_obj=hist_admm['obj'],
             admm_iter=hist_admm['iter'],
             admm_primal=hist_admm['primal_res'],
             admm_dual=hist_admm['dual_res'],
             vos_t=hist_vos['t'],
             vos_obj=hist_vos['obj'],
             vos_lyap=hist_vos['lyapunov'],
             fista_lyap_k=fista_lyap['k'],
             fista_lyap_E=fista_lyap['E'],
             admm_lyap_k=admm_lyap['k'],
             admm_lyap_E=admm_lyap['E'],
             )
    
    # Save sweep results
    sweep_data = {f'r_{r}': list(v) for r, v in sweep_results.items()}
    np.savez('outputs/parameter_sweep.npz', **sweep_data)
    
    # Save solution comparison
    np.savez('outputs/solutions.npz',
             x_true=x_true,
             x_ista=x_ista,
             x_fista=x_fista,
             x_admm=x_admm,
             x_ref=x_ref,
             )
    
    # Save key metrics
    metrics = {
        'condition_number': 10.0,
        'lipschitz_constant': float(L),
        'regularization_lambda': float(lam),
        'n_features': int(A.shape[1]),
        'n_samples': int(A.shape[0]),
        'sparsity_true': int(np.count_nonzero(x_true)),
        'ista_final_obj': float(lasso_obj(A, b, x_ista, lam)),
        'fista_final_obj': float(lasso_obj(A, b, x_fista, lam)),
        'admm_final_obj': float(lasso_obj(A, b, x_admm, lam)),
        'vos_final_obj': float(lasso_obj(A, b, x_vos, lam)),
        'reference_obj': float(f_ref),
        'ista_error_to_true': float(norm(x_ista - x_true) / norm(x_true)),
        'fista_error_to_true': float(norm(x_fista - x_true) / norm(x_true)),
        'admm_error_to_true': float(norm(x_admm - x_true) / norm(x_true)),
        'fista_restarts': len(hist_fista['restart_points']),
    }
    
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("All results saved to outputs/")
    print("=" * 60)
    
    return metrics

if __name__ == '__main__':
    main()
