"""
Unified Variable and Operator Splitting (VOS) Framework
for Nesterov's Accelerated Method and ADMM
"""
import numpy as np
from scipy.linalg import norm, cho_factor, cho_solve


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def lasso_objective(A, b, x, lam):
    return 0.5 * norm(A @ x - b)**2 + lam * norm(x, 1)


def run_all_algorithms(A, b, lam, max_iter=150):
    """Run all algorithms and return results."""
    m, n = A.shape
    x0 = np.zeros(n)
    L = np.linalg.norm(A, 2)**2  # Lipschitz constant of gradient
    step_size = 1.0 / L
    
    AtA = A.T @ A
    Atb = A.T @ b
    rho = 2.0
    M = AtA + rho * np.eye(n)
    M_chol = cho_factor(M)
    
    # Reference: run GD for many iterations (it converges well for this problem)
    x_ref, _, _ = _gd_core(AtA, Atb, A, b, x0, lam, step_size, 500)
    f_ref = lasso_objective(A, b, x_ref, lam)
    
    # 1. GD
    x_gd, obj_gd, hist_gd = _gd_core(AtA, Atb, A, b, x0, lam, step_size, max_iter)
    
    # 2. NAG
    x_nag, obj_nag, hist_nag = _nag_core(AtA, Atb, A, b, x0, lam, step_size, max_iter)
    
    # 3. NAG+Restart
    x_nag_r, obj_nag_r, hist_nag_r, restarts = _nag_restart_core(AtA, Atb, A, b, x0, lam, step_size, max_iter)
    
    # 4. ADMM (with larger rho for faster convergence)
    x_admm, obj_admm, hist_admm, primal_res, dual_res = _admm_core(AtA, Atb, A, b, x0, lam, rho, M_chol, max_iter)
    
    # 5. VOS unified (Nesterov momentum + ADMM splitting)
    x_vos, obj_vos, hist_vos = _vos_core(AtA, Atb, A, b, x0, lam, step_size, rho, max_iter, 0.5)
    
    # 6. VOS 2nd-order (continuous ODE dynamics + ADMM splitting)
    x_vos2, obj_vos2, hist_vos2, vel_norm = _vos2_core(AtA, Atb, A, b, x0, lam, step_size, rho, max_iter)
    
    return {
        'x_ref': x_ref, 'f_ref': f_ref, 'step_size': step_size, 'L': L,
        'x_gd': x_gd, 'obj_gd': obj_gd, 'hist_gd': hist_gd,
        'x_nag': x_nag, 'obj_nag': obj_nag, 'hist_nag': hist_nag,
        'x_nag_r': x_nag_r, 'obj_nag_r': obj_nag_r, 'hist_nag_r': hist_nag_r, 'restarts': restarts,
        'x_admm': x_admm, 'obj_admm': obj_admm, 'hist_admm': hist_admm,
        'primal_res': primal_res, 'dual_res': dual_res,
        'x_vos': x_vos, 'obj_vos': obj_vos, 'hist_vos': hist_vos,
        'x_vos2': x_vos2, 'obj_vos2': obj_vos2, 'hist_vos2': hist_vos2, 'vel_norm': vel_norm,
    }


def _gd_core(AtA, Atb, A, b, x0, lam, step_size, max_iter):
    x = x0.copy()
    obj = [lasso_objective(A, b, x, lam)]
    hist = [x.copy()]
    for k in range(max_iter):
        grad = AtA @ x - Atb
        x = soft_threshold(x - step_size * grad, step_size * lam)
        obj.append(lasso_objective(A, b, x, lam))
        hist.append(x.copy())
    return x, np.array(obj), hist


def _nag_core(AtA, Atb, A, b, x0, lam, step_size, max_iter):
    """Nesterov's Accelerated Gradient with momentum coefficient (k-1)/(k+2)."""
    x_prev = x0.copy()
    y = x0.copy()
    obj = [lasso_objective(A, b, x0, lam)]
    hist = [x0.copy()]
    x_curr = x0.copy()
    for k in range(1, max_iter + 1):
        grad = AtA @ y - Atb
        x_curr = soft_threshold(y - step_size * grad, step_size * lam)
        beta = (k - 1) / (k + 2)
        y = x_curr + beta * (x_curr - x_prev)
        obj.append(lasso_objective(A, b, x_curr, lam))
        hist.append(x_curr.copy())
        x_prev = x_curr.copy()
    return x_curr, np.array(obj), hist


def _nag_restart_core(AtA, Atb, A, b, x0, lam, step_size, max_iter):
    """NAG with adaptive restart based on ODE overshoot detection."""
    x_prev = x0.copy()
    y = x0.copy()
    obj = [lasso_objective(A, b, x0, lam)]
    hist = [x0.copy()]
    restarts = 0
    x_curr = x0.copy()
    for k in range(1, max_iter + 1):
        grad = AtA @ y - Atb
        x_curr = soft_threshold(y - step_size * grad, step_size * lam)
        cur_obj = lasso_objective(A, b, x_curr, lam)
        prev_obj = lasso_objective(A, b, x_prev, lam)
        if cur_obj > prev_obj and k > 2:
            y = x_curr.copy()
            restarts += 1
        else:
            beta = (k - 1) / (k + 2)
            y = x_curr + beta * (x_curr - x_prev)
        obj.append(cur_obj)
        hist.append(x_curr.copy())
        x_prev = x_curr.copy()
    return x_curr, np.array(obj), hist, restarts


def _admm_core(AtA, Atb, A, b, x0, lam, rho, M_chol, max_iter, tol=1e-6):
    """Standard ADMM for Lasso: min 0.5||Ax-b||^2 + lam||z||_1 s.t. x=z."""
    n = len(x0)
    x = x0.copy()
    z = x0.copy()
    u = np.zeros(n)
    obj = [lasso_objective(A, b, x, lam)]
    hist = [x.copy()]
    primal_res = []
    dual_res = []
    for k in range(max_iter):
        # x-update: (A^T A + rho I)x = A^T b + rho(z - u)
        rhs = Atb + rho * (z - u)
        x = cho_solve(M_chol, rhs)
        # z-update: soft thresholding
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        # u-update
        u = u + x - z
        pr = norm(x - z)
        dr = rho * norm(z - z_old)
        primal_res.append(pr)
        dual_res.append(dr)
        obj.append(lasso_objective(A, b, x, lam))
        hist.append(x.copy())
        if pr < tol and dr < tol:
            break
    return x, np.array(obj), hist, primal_res, dual_res


def _vos_core(AtA, Atb, A, b, x0, lam, step_size, rho, max_iter, alpha=0.5):
    """
    VOS: Unified Variable and Operator Splitting.
    
    Combines Nesterov acceleration with ADMM splitting:
    - Uses momentum (Nesterov) for the primal variable update
    - Uses ADMM consensus for variable splitting (smooth + non-smooth)
    - alpha controls the mix between accelerated and consensus updates
    """
    n = len(x0)
    x_prev = x0.copy()
    x_curr = x0.copy()
    z = x0.copy()
    u = np.zeros(n)
    
    # Precompute
    M = AtA + rho * np.eye(n)
    M_chol = cho_factor(M)
    
    obj = [lasso_objective(A, b, x0, lam)]
    hist = [x0.copy()]
    
    for k in range(1, max_iter + 1):
        # Momentum: Nesterov-style
        beta = (k - 1) / (k + 2)
        y = x_curr + beta * (x_curr - x_prev)
        
        # Gradient-based update (Nesterov-like)
        grad = AtA @ y - Atb
        x_nesterov = soft_threshold(y - step_size * grad, step_size * lam)
        
        # Consensus update (ADMM-like)
        rhs = Atb + rho * (z - u)
        x_admm = cho_solve(M_chol, rhs)
        
        # Unified: weighted combination
        x_new = alpha * x_nesterov + (1 - alpha) * x_admm
        
        # z-update (proximal for non-smooth)
        z = soft_threshold(x_new + u, lam / rho)
        
        # u-update (dual)
        u = u + x_new - z
        
        x_prev = x_curr.copy()
        x_curr = x_new.copy()
        obj.append(lasso_objective(A, b, x_curr, lam))
        hist.append(x_curr.copy())
    
    return x_curr, np.array(obj), hist


def _vos2_core(AtA, Atb, A, b, x0, lam, step_size, rho, max_iter):
    """
    VOS with explicit second-order ODE dynamics.
    
    Implements the continuous-time system:
    ẍ + (3/t)ẋ + ∇f(x) = 0
    
    Discretized and combined with ADMM splitting.
    """
    n = len(x0)
    x_prev = x0.copy()
    x_curr = x0.copy()
    z = x0.copy()
    u = np.zeros(n)
    
    M = AtA + rho * np.eye(n)
    M_chol = cho_factor(M)
    
    obj = [lasso_objective(A, b, x0, lam)]
    hist = [x0.copy()]
    vel_norm = []
    r = 3.0  # ODE damping parameter (optimal from Su-Boyd-Candès)
    
    for k in range(1, max_iter + 1):
        # Velocity from ODE perspective
        velocity = x_curr - x_prev
        
        # Effective damping: ẍ + (r/t)ẋ + ∇f(x) = 0
        damping = r / (k + 1e-10)
        
        # Acceleration from ODE
        grad = AtA @ x_curr - Atb
        acceleration = -damping * velocity - grad
        
        # Position update using second-order dynamics
        x_momentum = x_curr + velocity + step_size * acceleration
        x_prox = soft_threshold(x_momentum, step_size * lam)
        
        # ADMM consensus correction
        rhs = Atb + rho * (z - u)
        x_admm = cho_solve(M_chol, rhs)
        
        # Adaptive mixing: early iterations use more ADMM, later more momentum
        t_ratio = min(k / (k + 2), 0.9)
        x_new = t_ratio * x_prox + (1 - t_ratio) * x_admm
        
        # z-update
        z = soft_threshold(x_new + u, lam / rho)
        
        # u-update
        u = u + (x_new - z)
        
        x_prev = x_curr.copy()
        x_curr = x_new.copy()
        obj.append(lasso_objective(A, b, x_curr, lam))
        hist.append(x_curr.copy())
        vel_norm.append(norm(velocity))
    
    return x_curr, np.array(obj), hist, vel_norm


def compute_lyapunov_values(x_history, A, b, lam, x_star):
    """Compute Lyapunov function V(x) = f(x) - f* + c*||x-x*||^2."""
    f_star = lasso_objective(A, b, x_star, lam)
    lyap = []
    for x in x_history:
        f_val = lasso_objective(A, b, x, lam) - f_star
        dist = 0.5 * norm(x - x_star)**2
        lyap.append(f_val + dist)
    return np.array(lyap)
