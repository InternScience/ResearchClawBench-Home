"""
Unified Variable and Operator Splitting (VOS) Framework for Accelerated Optimization

This module implements and compares:
1. Gradient Descent (GD) - baseline
2. Nesterov's Accelerated Gradient / FISTA - accelerated first-order
3. ADMM - operator splitting method
4. VOS-based methods - unified framework connecting continuous-time dynamics to discrete algorithms

All methods are applied to the Lasso problem:
    min_x  (1/2)||Ax - b||^2 + lambda||x||_1

The VOS framework unifies these methods through a continuous-time perspective,
with convergence analyzed via strong Lyapunov functions.
"""

import numpy as np
from scipy.linalg import svdvals
import json
import os
import time

np.random.seed(42)

# ============================================================
# Load Data
# ============================================================
def load_data(path='data/complex_optimization_data.npy'):
    data = np.load(path, allow_pickle=True).item()
    return data['A'], data['b'], data['x_true']

# ============================================================
# Problem Setup
# ============================================================
def smooth_objective(x, A, b):
    """f(x) = (1/2)||Ax - b||^2"""
    r = A @ x - b
    return 0.5 * np.dot(r, r)

def smooth_gradient(x, A, b):
    """grad f(x) = A^T(Ax - b)"""
    return A.T @ (A @ x - b)

def nonsmooth_objective(x, lam):
    """g(x) = lambda||x||_1"""
    return lam * np.sum(np.abs(x))

def full_objective(x, A, b, lam):
    """F(x) = f(x) + g(x)"""
    return smooth_objective(x, A, b) + nonsmooth_objective(x, lam)

def prox_l1(v, threshold):
    """Proximal operator for lambda||.||_1 (soft thresholding)"""
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

def compute_lipschitz(A):
    """Compute Lipschitz constant L = lambda_max(A^T A)"""
    s = svdvals(A)
    return s[0]**2

# ============================================================
# Method 1: Proximal Gradient Descent (Baseline)
# ============================================================
def proximal_gd(A, b, lam, x0, max_iter=5000, tol=1e-12):
    """
    Proximal Gradient Descent for composite optimization.
    x_{k+1} = prox_{alpha*lambda||.||_1}(x_k - alpha*grad f(x_k))
    
    Convergence rate: O(1/k)
    """
    n = len(x0)
    x = x0.copy()
    L = compute_lipschitz(A)
    alpha = 1.0 / L
    
    obj_history = []
    
    for k in range(max_iter):
        grad = smooth_gradient(x, A, b)
        x_new = prox_l1(x - alpha * grad, alpha * lam)
        
        obj_val = full_objective(x_new, A, b, lam)
        obj_history.append(obj_val)
        
        if np.linalg.norm(x_new - x) < tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
    
    return np.array(obj_history), x

# ============================================================
# Method 2: FISTA (Nesterov's Accelerated Method for Composite)
# ============================================================
def fista(A, b, lam, x0, max_iter=5000, tol=1e-12):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    This is Nesterov's acceleration applied to composite optimization.
    
    y_0 = x_0, t_0 = 1
    x_{k+1} = prox_{alpha*lambda||.||_1}(y_k - alpha*grad f(y_k))
    t_{k+1} = (1 + sqrt(1 + 4*t_k^2))/2
    y_{k+1} = x_{k+1} + ((t_k - 1)/t_{k+1})(x_{k+1} - x_k)
    
    Convergence rate: O(1/k^2)
    """
    n = len(x0)
    x = x0.copy()
    y = x0.copy()
    L = compute_lipschitz(A)
    alpha = 1.0 / L
    
    obj_history = []
    t = 1.0
    x_prev = x.copy()
    
    for k in range(max_iter):
        grad = smooth_gradient(y, A, b)
        x_new = prox_l1(y - alpha * grad, alpha * lam)
        
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        t = t_new
        
        obj_val = full_objective(x_new, A, b, lam)
        obj_history.append(obj_val)
        
        if np.linalg.norm(x_new - x) < tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
    
    return np.array(obj_history), x

# ============================================================
# Method 2b: FISTA with Restarting (Linear Convergence on Strongly Convex)
# ============================================================
def fista_restart(A, b, lam, x0, max_iter=5000, tol=1e-12, restart_freq=50):
    """
    FISTA with periodic restarting.
    When the objective increases or momentum becomes counterproductive,
    reset the momentum term. This achieves practical linear convergence.
    """
    n = len(x0)
    x = x0.copy()
    y = x0.copy()
    L = compute_lipschitz(A)
    alpha = 1.0 / L
    
    obj_history = []
    t = 1.0
    prev_obj = np.inf
    
    for k in range(max_iter):
        # Restart if objective increased or periodically
        grad = smooth_gradient(y, A, b)
        x_new = prox_l1(y - alpha * grad, alpha * lam)
        
        obj_val = full_objective(x_new, A, b, lam)
        
        # Adaptive restart: if objective went up, reset
        if obj_val > prev_obj:
            y = x_new.copy()
            t = 1.0
        else:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        
        obj_history.append(obj_val)
        
        if np.linalg.norm(x_new - x) < tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
        prev_obj = obj_val
    
    return np.array(obj_history), x

# ============================================================
# Method 3: ADMM for Lasso
# ============================================================
def admm_lasso(A, b, lam, x0, max_iter=5000, tol=1e-12, rho=None):
    """
    ADMM for Lasso problem:
        min (1/2)||Az - b||^2 + lambda||x||_1  s.t.  z = x
    
    Updates:
    z^{k+1} = (A^T A + rho*I)^{-1}(A^T b + rho*(x^k - u^k))
    x^{k+1} = prox_{lambda/rho ||.||_1}(z^{k+1} + u^k)
    u^{k+1} = u^k + z^{k+1} - x^{k+1}
    """
    n = len(x0)
    
    if rho is None:
        L = compute_lipschitz(A)
        rho = 1.0  # Standard choice
    
    # Precompute factorization of (A^T A + rho*I)
    AtA = A.T @ A
    rhs_matrix = AtA + rho * np.eye(n)
    Atb = A.T @ b
    
    # Use Cholesky for efficiency
    try:
        L_chol = np.linalg.cholesky(rhs_matrix)
        def solve_z(rhs):
            y = np.linalg.solve(L_chol, rhs)
            return np.linalg.solve(L_chol.T, y)
    except np.linalg.LinAlgError:
        def solve_z(rhs):
            return np.linalg.solve(rhs_matrix, rhs)
    
    # Initialize
    z = x0.copy()
    x = x0.copy()
    u = np.zeros(n)
    
    obj_history = []
    
    for k in range(max_iter):
        # z-update (quadratic)
        z = solve_z(Atb + rho * (x - u))
        
        # x-update (proximal)
        x_new = prox_l1(z + u, lam / rho)
        
        # u-update (dual)
        u = u + z - x_new
        
        obj_val = full_objective(x_new, A, b, lam)
        obj_history.append(obj_val)
        
        # Check convergence via primal and dual residuals
        primal_res = np.linalg.norm(z - x_new)
        dual_res = rho * np.linalg.norm(x_new - x)
        
        if primal_res < tol and dual_res < tol:
            x = x_new
            break
        x = x_new
    
    return np.array(obj_history), x

# ============================================================
# Method 4: VOS - Nesterov ODE Discretization (Continuous-Time View)
# ============================================================
def vos_nesterov_ode(A, b, lam, x0, max_iter=5000, tol=1e-12, r=3.0):
    """
    VOS Framework: Nesterov ODE discretization.
    
    The Nesterov ODE is: X'' + (r/t)*X' + grad f(X) = 0
    
    This method derives the discrete FISTA algorithm from the ODE by
    using a semi-implicit Euler discretization with step size alpha = 1/L.
    
    The key VOS insight: the momentum coefficient beta_k = (t_k - 1)/(t_{k+1})
    arises naturally from the time-dependent damping r/t in the ODE.
    
    For composite problems, we apply the proximal operator after each
    gradient step (variable-operator splitting).
    """
    n = len(x0)
    L = compute_lipschitz(A)
    alpha = 1.0 / L
    
    x = x0.copy()
    y = x0.copy()
    
    obj_history = []
    
    # Time parameterization: t_k approx k * sqrt(alpha)
    # With r=3, the ODE gives beta_k = (k-1)/(k+2) asymptotically
    t = 1.0  # Corresponds to initial time scaling
    
    for k in range(max_iter):
        # Effective time in the ODE
        t_eff = (k + 1) * np.sqrt(alpha)
        
        # Momentum coefficient from ODE damping
        # beta(t) = (t - r/2) / (t + r/2) for large t
        # This recovers the FISTA coefficient asymptotically
        beta = max(0.0, (t_eff - r * np.sqrt(alpha) / 2.0) / 
                        (t_eff + r * np.sqrt(alpha) / 2.0))
        
        # Gradient step at extrapolation point
        grad = smooth_gradient(y, A, b)
        x_new = prox_l1(y - alpha * grad, alpha * lam)
        
        # Extrapolation (momentum)
        y = x_new + beta * (x_new - x)
        
        obj_val = full_objective(x_new, A, b, lam)
        obj_history.append(obj_val)
        
        if np.linalg.norm(x_new - x) < tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
    
    return np.array(obj_history), x

# ============================================================
# Method 5: VOS with Adaptive Restarting (Linear Convergence)
# ============================================================
def vos_adaptive_restart(A, b, lam, x0, max_iter=5000, tol=1e-12):
    """
    VOS ODE solver with adaptive restarting based on gradient alignment.
    
    From the ODE perspective, restarting corresponds to resetting the
    velocity when the trajectory enters an underdamped oscillatory regime.
    This transforms the O(1/t^2) convergence into linear convergence
    for strongly convex objectives.
    
    The restart condition: <X - x*, grad f(X)> < 0 indicates overshooting.
    In practice, we use: <y_k - x_k, grad f(y_k)> > 0 as a proxy.
    """
    n = len(x0)
    L = compute_lipschitz(A)
    alpha = 1.0 / L
    
    x = x0.copy()
    y = x0.copy()
    
    obj_history = []
    prev_obj = np.inf
    
    for k in range(max_iter):
        grad = smooth_gradient(y, A, b)
        x_new = prox_l1(y - alpha * grad, alpha * lam)
        
        obj_val = full_objective(x_new, A, b, lam)
        
        # Adaptive restart condition
        # If the objective increased or momentum is counterproductive
        momentum = x_new - x
        if obj_val > prev_obj or np.dot(momentum, grad) > 0:
            # Reset: this corresponds to velocity reset in the ODE
            y = x_new.copy()
        else:
            # Standard FISTA momentum
            t_k = k + 1.0
            t_kp1 = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
            beta = (t_k - 1.0) / t_kp1
            y = x_new + beta * (x_new - x)
        
        obj_history.append(obj_val)
        
        if np.linalg.norm(x_new - x) < tol * max(1.0, np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
        prev_obj = obj_val
    
    return np.array(obj_history), x

# ============================================================
# Method 6: VOS Unified Framework - Generalized Parameter Sweep
# ============================================================
def vos_generalized(A, b, lam, x0, max_iter=2000, tol=1e-12, r_values=None):
    """
    VOS Framework: Generalized damping parameter sweep.
    
    The generalized Nesterov ODE: X'' + (r/t)*X' + grad f(X) = 0
    
    Key theoretical result: O(1/t^2) convergence iff r >= 3.
    This method tests different values of r to verify this phase transition.
    """
    if r_values is None:
        r_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    results = {}
    
    for r in r_values:
        n = len(x0)
        L = compute_lipschitz(A)
        alpha = 1.0 / L
        
        x = x0.copy()
        y = x0.copy()
        
        obj_history = []
        t = 1.0
        
        for k in range(max_iter):
            grad = smooth_gradient(y, A, b)
            x_new = prox_l1(y - alpha * grad, alpha * lam)
            
            # Generalized momentum coefficient
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            # Scale by r/3 to get generalized coefficient
            beta = ((r / 3.0) * (t - 1.0)) / t_new
            beta = min(beta, 1.0)  # Clamp
            
            y = x_new + beta * (x_new - x)
            t = t_new
            
            obj_val = full_objective(x_new, A, b, lam)
            obj_history.append(obj_val)
        
        results[f'r={r}'] = {
            'obj_history': obj_history,
            'final_obj': obj_history[-1],
            'iterations': len(obj_history)
        }
    
    return results

# ============================================================
# Lyapunov Function Analysis
# ============================================================
def compute_lyapunov_smooth(X, V, t, x_star, f_star, A, b, r=3.0):
    """
    Strong Lyapunov function for the smooth VOS framework.
    
    E(t) = t^2*(f(X) - f*) + (r/2)*||X - x* + (t/r)*V||^2
    
    For the Nesterov ODE X'' + (r/t)*X' + grad f(X) = 0, this Lyapunov function
    satisfies dE/dt <= 0 when r >= 3.
    """
    f_X = smooth_objective(X, A, b)
    diff = X - x_star + (t / r) * V
    energy = t**2 * (f_X - f_star) + (r / 2.0) * np.dot(diff, diff)
    return energy

# ============================================================
# Main Experiment Runner
# ============================================================
def run_experiments():
    print("=" * 60)
    print("Running VOS Framework Experiments")
    print("=" * 60)
    
    # Load data
    A, b, x_true = load_data()
    n = A.shape[1]
    m = A.shape[0]
    
    # Set regularization parameter
    lam = 0.1 * np.max(np.abs(A.T @ b))
    print(f"Problem: Lasso Regression, m={m}, n={n}")
    print(f"Regularization parameter lambda = {lam:.4f}")
    
    # Initial point
    x0 = np.zeros(n)
    
    # Compute reference optimal solution using high-accuracy FISTA
    print("\nComputing reference optimal solution...")
    ref_obj, x_ref = fista(A, b, lam, x0, max_iter=50000, tol=1e-15)
    f_star = ref_obj[-1]
    print(f"Reference optimal objective: {f_star:.10f}")
    
    results = {}
    
    # ---- Method 1: Proximal Gradient Descent ----
    print("\n[1/7] Running Proximal Gradient Descent...")
    t0 = time.time()
    obj_hist_gd, x_gd = proximal_gd(A, b, lam, x0, max_iter=5000)
    gd_time = time.time() - t0
    print(f"  GD: {len(obj_hist_gd)} iterations, final obj = {obj_hist_gd[-1]:.10f}, time = {gd_time:.2f}s")
    results['gd'] = {'obj_history': obj_hist_gd.tolist(), 'final_obj': float(obj_hist_gd[-1]), 
                     'iterations': len(obj_hist_gd), 'time': gd_time, 'final_x': x_gd.tolist()}
    
    # ---- Method 2: FISTA (Nesterov) ----
    print("\n[2/7] Running FISTA (Nesterov Accelerated)...")
    t0 = time.time()
    obj_hist_fista, x_fista = fista(A, b, lam, x0, max_iter=5000)
    fista_time = time.time() - t0
    print(f"  FISTA: {len(obj_hist_fista)} iterations, final obj = {obj_hist_fista[-1]:.10f}, time = {fista_time:.2f}s")
    results['fista'] = {'obj_history': obj_hist_fista.tolist(), 'final_obj': float(obj_hist_fista[-1]),
                        'iterations': len(obj_hist_fista), 'time': fista_time, 'final_x': x_fista.tolist()}
    
    # ---- Method 2b: FISTA with Restarting ----
    print("\n[3/7] Running FISTA with Adaptive Restarting...")
    t0 = time.time()
    obj_hist_fista_r, x_fista_r = fista_restart(A, b, lam, x0, max_iter=5000)
    fista_r_time = time.time() - t0
    print(f"  FISTA-R: {len(obj_hist_fista_r)} iterations, final obj = {obj_hist_fista_r[-1]:.10f}, time = {fista_r_time:.2f}s")
    results['fista_restart'] = {'obj_history': obj_hist_fista_r.tolist(), 'final_obj': float(obj_hist_fista_r[-1]),
                                'iterations': len(obj_hist_fista_r), 'time': fista_r_time, 'final_x': x_fista_r.tolist()}
    
    # ---- Method 3: ADMM ----
    print("\n[4/7] Running ADMM...")
    t0 = time.time()
    obj_hist_admm, x_admm = admm_lasso(A, b, lam, x0, max_iter=5000)
    admm_time = time.time() - t0
    print(f"  ADMM: {len(obj_hist_admm)} iterations, final obj = {obj_hist_admm[-1]:.10f}, time = {admm_time:.2f}s")
    results['admm'] = {'obj_history': obj_hist_admm.tolist(), 'final_obj': float(obj_hist_admm[-1]),
                       'iterations': len(obj_hist_admm), 'time': admm_time, 'final_x': x_admm.tolist()}
    
    # ---- Method 4: VOS Nesterov ODE Discretization ----
    print("\n[5/7] Running VOS Nesterov ODE Discretization...")
    t0 = time.time()
    obj_hist_vos, x_vos = vos_nesterov_ode(A, b, lam, x0, max_iter=5000)
    vos_time = time.time() - t0
    print(f"  VOS-NODE: {len(obj_hist_vos)} iterations, final obj = {obj_hist_vos[-1]:.10f}, time = {vos_time:.2f}s")
    results['vos_noderiv'] = {'obj_history': obj_hist_vos.tolist(), 'final_obj': float(obj_hist_vos[-1]),
                              'iterations': len(obj_hist_vos), 'time': vos_time, 'final_x': x_vos.tolist()}
    
    # ---- Method 5: VOS with Adaptive Restarting ----
    print("\n[6/7] Running VOS with Adaptive Restarting...")
    t0 = time.time()
    obj_hist_vos_ar, x_vos_ar = vos_adaptive_restart(A, b, lam, x0, max_iter=5000)
    vos_ar_time = time.time() - t0
    print(f"  VOS-AR: {len(obj_hist_vos_ar)} iterations, final obj = {obj_hist_vos_ar[-1]:.10f}, time = {vos_ar_time:.2f}s")
    results['vos_restart'] = {'obj_history': obj_hist_vos_ar.tolist(), 'final_obj': float(obj_hist_vos_ar[-1]),
                              'iterations': len(obj_hist_vos_ar), 'time': vos_ar_time, 'final_x': x_vos_ar.tolist()}
    
    # ---- Method 6: VOS Generalized Damping Parameter Sweep ----
    print("\n[7/7] Running VOS Generalized Damping Parameter Sweep (r values)...")
    t0 = time.time()
    vos_sweep = vos_generalized(A, b, lam, x0, max_iter=2000, r_values=[1.0, 2.0, 3.0, 4.0, 5.0])
    sweep_time = time.time() - t0
    print(f"  Sweep completed in {sweep_time:.2f}s")
    results['vos_sweep'] = vos_sweep
    
    # ---- Save results ----
    os.makedirs('outputs', exist_ok=True)
    
    summary = {
        'f_star': float(f_star),
        'lambda': float(lam),
        'condition_number': 10.0,
        'lipschitz_constant': float(compute_lipschitz(A)),
        'problem_dimensions': {'m': int(m), 'n': int(n)},
        'methods': {}
    }
    for name in ['gd', 'fista', 'fista_restart', 'admm', 'vos_noderiv', 'vos_restart']:
        res = results[name]
        summary['methods'][name] = {
            'final_obj': res['final_obj'],
            'optimality_gap': res['final_obj'] - f_star,
            'iterations': res['iterations'],
            'time': res['time']
        }
    
    with open('outputs/convergence_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save full histories for plotting
    for name in ['gd', 'fista', 'fista_restart', 'admm', 'vos_noderiv', 'vos_restart']:
        hist_data = {
            'obj_history': results[name]['obj_history'],
            'iterations': results[name]['iterations']
        }
        np.save(f'outputs/{name}_history.npy', hist_data)
    
    # Save sweep results
    np.save('outputs/vos_sweep.npy', vos_sweep)
    
    # ---- Lyapunov Analysis ----
    print("\n--- Lyapunov Function Analysis ---")
    
    # Use FISTA result as approximate x_star
    x_star_approx = x_fista
    f_star_smooth = smooth_objective(x_star_approx, A, b)
    
    L = compute_lipschitz(A)
    alpha = 1.0 / L
    r = 3.0
    
    # Compute Lyapunov for VOS ODE method
    # Re-run with history tracking for Lyapunov
    n_dim = len(x0)
    x = x0.copy()
    y = x0.copy()
    lyap_vos = []
    lyap_fista = []
    t_values_lyap = []
    
    x_prev = x0.copy()
    
    for k in range(min(500, len(obj_hist_vos))):
        t_eff = (k + 1) * np.sqrt(alpha)
        
        beta = max(0.0, (t_eff - r * np.sqrt(alpha) / 2.0) / 
                        (t_eff + r * np.sqrt(alpha) / 2.0))
        
        grad = smooth_gradient(y, A, b)
        x_new = prox_l1(y - alpha * grad, alpha * lam)
        
        y = x_new + beta * (x_new - x)
        
        # Approximate velocity
        V = (x_new - x_prev) / np.sqrt(alpha)
        
        # Lyapunov energy
        f_X = smooth_objective(x_new, A, b)
        diff = x_new - x_star_approx + (t_eff / r) * V
        E = t_eff**2 * (f_X - f_star_smooth) + (r / 2.0) * np.dot(diff, diff)
        lyap_vos.append(E)
        t_values_lyap.append(t_eff)
        
        x_prev = x.copy()
        x = x_new
    
    lyap_data = {
        'vos_lyapunov': lyap_vos,
        't_values': t_values_lyap,
        'f_star_smooth': float(f_star_smooth)
    }
    
    with open('outputs/lyapunov_analysis.json', 'w') as f:
        json.dump(lyap_data, f, indent=2)
    
    print(f"\nResults saved to outputs/")
    print(f"\nMethod Comparison Summary:")
    for name, info in summary['methods'].items():
        print(f"  {name:20s}: gap={info['optimality_gap']:.2e}, iters={info['iterations']}, time={info['time']:.2f}s")
    
    return results, summary

if __name__ == '__main__':
    results, summary = run_experiments()
