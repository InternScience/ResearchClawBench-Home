"""
Variable and Operator Splitting (VOS) Framework

A unified framework for analyzing optimization algorithms through:
1. Continuous-time dynamical systems (ODE perspective)
2. Lyapunov function analysis
3. Unification of Nesterov's accelerated gradient and ADMM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')


class LassoProblem:
    """Lasso regression problem: min (1/2)||Ax - b||^2 + lambda||x||_1"""
    
    def __init__(self, A, b, lambda_reg=0.1):
        self.A = A
        self.b = b
        self.lambda_reg = lambda_reg
        self.n = A.shape[1]
        self.L = np.linalg.norm(A.T @ A, 2)  # Lipschitz constant of gradient
        self.mu = self._estimate_strong_convexity()
        
    def _estimate_strong_convexity(self):
        """Estimate strong convexity parameter from smallest singular value"""
        s = svd(self.A, compute_uv=False)
        return s[-1]**2 if len(s) > 0 else 1e-6
    
    def smooth_objective(self, x):
        """f(x) = (1/2)||Ax - b||^2"""
        residual = self.A @ x - self.b
        return 0.5 * np.dot(residual, residual)
    
    def nonsmooth_objective(self, x):
        """g(x) = lambda * ||x||_1"""
        return self.lambda_reg * np.sum(np.abs(x))
    
    def objective(self, x):
        """Full objective F(x) = f(x) + g(x)"""
        return self.smooth_objective(x) + self.nonsmooth_objective(x)
    
    def gradient(self, x):
        """Gradient of smooth part: ∇f(x) = A^T(Ax - b)"""
        return self.A.T @ (self.A @ x - self.b)
    
    def proximal_operator(self, x, gamma):
        """Proximal operator for L1 norm: soft thresholding"""
        return np.sign(x) * np.maximum(np.abs(x) - gamma * self.lambda_reg, 0)


class NesterovAcceleratedGradient:
    """
    Nesterov's Accelerated Gradient Method for Composite Optimization (FISTA)
    
    Based on:
    - Nesterov (1983): Original accelerated method
    - Beck & Teboulle (2009): FISTA for composite optimization
    - Su, Boyd, Candès (2014): ODE interpretation
    """
    
    def __init__(self, problem, step_size=None):
        self.problem = problem
        self.step_size = step_size or 1.0 / problem.L
        self.history = {'objective': [], 'iterates': [], 'lyapunov': []}
        
    def solve(self, x0, max_iter=1000, tol=1e-6):
        """Run Nesterov's accelerated gradient method"""
        x = x0.copy()
        y = x0.copy()
        x_prev = x0.copy()
        t = 1.0
        
        self.history['objective'] = []
        self.history['iterates'] = []
        self.history['lyapunov'] = []
        
        for k in range(max_iter):
            # Store previous values
            x_prev = x.copy()
            
            # Gradient step at y
            grad_y = self.problem.gradient(y)
            x = self.problem.proximal_operator(y - self.step_size * grad_y, self.step_size)
            
            # Momentum update
            t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x + ((t - 1) / t_next) * (x - x_prev)
            
            # Update t
            t = t_next
            
            # Record history
            obj_val = self.problem.objective(x)
            self.history['objective'].append(obj_val)
            self.history['iterates'].append(x.copy())
            
            # Compute Lyapunov function
            lyap = self._compute_lyapunov(x, x_prev, k)
            self.history['lyapunov'].append(lyap)
            
            # Check convergence
            if np.linalg.norm(x - x_prev) < tol:
                break
                
        return x, self.history
    
    def _compute_lyapunov(self, x, x_prev, k):
        """
        Lyapunov function for Nesterov's method:
        V_k = t_k^2 * (f(x_k) - f*) + 0.5 * ||z_k - x*||^2
        where z_k = x_{k-1} + t_k(x_k - x_{k-1})
        """
        t_k = (k + 2) / 2  # Approximation
        obj_gap = self.problem.objective(x)
        momentum_term = 0.5 * np.linalg.norm(x - x_prev)**2
        return t_k**2 * obj_gap + momentum_term


class ADMM:
    """
    Alternating Direction Method of Multipliers for Lasso
    
    Formulation: min (1/2)||Ax - b||^2 + lambda||z||_1
                 s.t. x - z = 0
    
    Augmented Lagrangian:
    L_rho(x, z, u) = (1/2)||Ax - b||^2 + lambda||z||_1 + (rho/2)||x - z + u||^2
    """
    
    def __init__(self, problem, rho=None):
        self.problem = problem
        self.rho = rho or 1.0
        self.history = {'objective': [], 'primal_residual': [], 'dual_residual': [], 'lyapunov': []}
        
    def solve(self, x0, max_iter=1000, tol=1e-6):
        """Run ADMM for Lasso"""
        x = x0.copy()
        z = x0.copy()
        u = np.zeros_like(x0)
        
        self.history['objective'] = []
        self.history['primal_residual'] = []
        self.history['dual_residual'] = []
        self.history['lyapunov'] = []
        
        AtA = self.problem.A.T @ self.problem.A
        Atb = self.problem.A.T @ self.problem.b
        
        # Precompute factorization for x-update
        n = self.problem.n
        P = AtA + self.rho * np.eye(n)
        
        for k in range(max_iter):
            x_prev = x.copy()
            z_prev = z.copy()
            
            # x-update: argmin (1/2)||Ax - b||^2 + (rho/2)||x - z + u||^2
            q = Atb + self.rho * (z - u)
            x = np.linalg.solve(P, q)
            
            # z-update: argmin lambda||z||_1 + (rho/2)||x - z + u||^2
            z = self.problem.proximal_operator(x + u, 1.0 / self.rho)
            
            # u-update (dual)
            u = u + x - z
            
            # Record history
            obj_val = self.problem.objective(z)
            self.history['objective'].append(obj_val)
            
            primal_res = np.linalg.norm(x - z)
            dual_res = np.linalg.norm(self.rho * (z - z_prev))
            self.history['primal_residual'].append(primal_res)
            self.history['dual_residual'].append(dual_res)
            
            # Lyapunov function for ADMM
            lyap = self._compute_lyapunov(x, z, u, obj_val)
            self.history['lyapunov'].append(lyap)
            
            # Check convergence
            if primal_res < tol and dual_res < tol:
                break
                
        return z, self.history
    
    def _compute_lyapunov(self, x, z, u, obj_val):
        """
        Lyapunov function for ADMM based on augmented Lagrangian
        """
        aug_lag = obj_val + (self.rho / 2) * np.linalg.norm(x - z)**2
        return aug_lag + 0.5 * self.rho * np.linalg.norm(u)**2


class GradientDescent:
    """Standard gradient descent with proximal operator for Lasso"""
    
    def __init__(self, problem, step_size=None):
        self.problem = problem
        self.step_size = step_size or 1.0 / problem.L
        self.history = {'objective': [], 'lyapunov': []}
        
    def solve(self, x0, max_iter=1000, tol=1e-6):
        """Run gradient descent with ISTA"""
        x = x0.copy()
        
        self.history['objective'] = []
        self.history['lyapunov'] = []
        
        for k in range(max_iter):
            x_prev = x.copy()
            
            # Gradient step
            grad = self.problem.gradient(x)
            x = self.problem.proximal_operator(x - self.step_size * grad, self.step_size)
            
            # Record history
            obj_val = self.problem.objective(x)
            self.history['objective'].append(obj_val)
            
            # Lyapunov
            lyap = obj_val + 0.5 * np.linalg.norm(x - x_prev)**2 / self.step_size
            self.history['lyapunov'].append(lyap)
            
            if np.linalg.norm(x - x_prev) < tol:
                break
                
        return x, self.history


class VOSFramework:
    """
    Variable and Operator Splitting (VOS) Framework
    
    Provides unified analysis through:
    1. Continuous-time ODE representation
    2. Lyapunov function analysis
    3. Connection between discrete and continuous dynamics
    """
    
    def __init__(self, problem):
        self.problem = problem
        
    def nesterov_ode(self, state, t):
        """
        ODE for Nesterov's method:
        Ẍ + (3/t)Ẋ + ∇f(X) = 0
        
        State: [X, V] where V = Ẋ
        """
        X, V = state[:self.problem.n], state[self.problem.n:]
        
        # Avoid division by zero at t=0
        damping = 3.0 / max(t, 0.001)
        
        # Ẋ = V
        dXdt = V
        
        # V̇ = -(3/t)V - ∇f(X)
        dVdt = -damping * V - self.problem.gradient(X)
        
        return np.concatenate([dXdt, dVdt])
    
    def heavy_ball_ode(self, state, t, gamma=1.0):
        """
        Heavy ball ODE (Polyak's method):
        Ẍ + γẊ + ∇f(X) = 0
        """
        X, V = state[:self.problem.n], state[self.problem.n:]
        
        dXdt = V
        dVdt = -gamma * V - self.problem.gradient(X)
        
        return np.concatenate([dXdt, dVdt])
    
    def solve_continuous_nesterov(self, x0, t_span, n_points=1000):
        """Solve Nesterov ODE continuously"""
        v0 = np.zeros_like(x0)
        state0 = np.concatenate([x0, v0])
        
        t = np.linspace(t_span[0], t_span[1], n_points)
        solution = odeint(self.nesterov_ode, state0, t)
        
        return t, solution[:, :self.problem.n], solution[:, self.problem.n:]
    
    def compute_energy_lyapunov(self, X, V, t):
        """
        Energy-based Lyapunov function for ODE:
        E(t) = t^2(f(X) - f*) + 2||X - X*||^2 + t^2||V||^2
        """
        energies = []
        for i in range(len(t)):
            x = X[i]
            v = V[i]
            ti = t[i]
            
            f_val = self.problem.smooth_objective(x)
            kinetic = 0.5 * np.linalg.norm(v)**2
            
            # Simplified Lyapunov (without optimal value)
            E = ti**2 * f_val + kinetic
            energies.append(E)
            
        return np.array(energies)


def run_experiments(data_path, lambda_reg=0.1):
    """Run all experiments and return results"""
    
    # Load data
    data = np.load(data_path, allow_pickle=True).item()
    A, b, x_true = data['A'], data['b'], data['x_true']
    
    # Create problem
    problem = LassoProblem(A, b, lambda_reg)
    
    # Initial point
    x0 = np.zeros(problem.n)
    
    # Run methods
    print("Running Gradient Descent (ISTA)...")
    gd = GradientDescent(problem)
    x_gd, hist_gd = gd.solve(x0, max_iter=500)
    
    print("Running Nesterov Accelerated Gradient (FISTA)...")
    nesterov = NesterovAcceleratedGradient(problem)
    x_nest, hist_nest = nesterov.solve(x0, max_iter=500)
    
    print("Running ADMM...")
    admm = ADMM(problem, rho=1.0)
    x_admm, hist_admm = admm.solve(x0, max_iter=500)
    
    # Compute optimal value
    f_star = min([
        problem.objective(x_gd),
        problem.objective(x_nest),
        problem.objective(x_admm),
        problem.objective(x_true)
    ])
    
    # VOS Framework analysis
    print("Running VOS continuous analysis...")
    vos = VOSFramework(problem)
    t_cont, X_cont, V_cont = vos.solve_continuous_nesterov(x0, [0.1, 50], n_points=500)
    energy_lyapunov = vos.compute_energy_lyapunov(X_cont, V_cont, t_cont)
    
    results = {
        'problem': problem,
        'f_star': f_star,
        'x_true': x_true,
        'gradient_descent': {
            'solution': x_gd,
            'history': hist_gd
        },
        'nesterov': {
            'solution': x_nest,
            'history': hist_nest
        },
        'admm': {
            'solution': x_admm,
            'history': hist_admm
        },
        'vos_continuous': {
            't': t_cont,
            'X': X_cont,
            'V': V_cont,
            'energy': energy_lyapunov
        }
    }
    
    return results


if __name__ == '__main__':
    # Test
    from data_utils import load_lasso_data
    A, b, x_true = load_lasso_data()
    problem = LassoProblem(A, b, lambda_reg=0.1)
    print(f"Problem: {A.shape}, L={problem.L:.2f}, mu={problem.mu:.6f}")
