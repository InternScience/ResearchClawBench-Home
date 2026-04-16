import numpy as np
import matplotlib.pyplot as plt
import time

# Load data
data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
A = data['A']
b = data['b']
x_true = data['x_true']

# Parameters
n_samples, n_features = A.shape
lambda_reg = 0.1 * np.max(np.abs(A.T @ b))  # Regularization parameter

def objective(x):
    return 0.5 * np.linalg.norm(A @ x - b)**2 + lambda_reg * np.linalg.norm(x, 1)

def soft_thresholding(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0)

# 1. Standard ADMM
def admm(A, b, lambda_reg, rho=1.0, max_iter=200):
    n, p = A.shape
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    
    AtA = A.T @ A
    Atb = A.T @ b
    
    # Precompute factorization
    L = np.linalg.cholesky(AtA + rho * np.eye(p))
    
    objs = []
    
    for k in range(max_iter):
        # x-update
        q = Atb + rho * (z - u)
        y = np.linalg.solve(L, q)
        x = np.linalg.solve(L.T, y)
        
        # z-update
        z_old = z.copy()
        z = soft_thresholding(x + u, lambda_reg / rho)
        
        # u-update
        u = u + x - z
        
        objs.append(objective(z))
        
    return z, objs

# 2. FISTA (Nesterov's Accelerated Proximal Gradient)
def fista(A, b, lambda_reg, max_iter=200):
    n, p = A.shape
    x = np.zeros(p)
    y = np.zeros(p)
    t = 1.0
    
    # Lipschitz constant of smooth part
    # Approximating L using power iteration
    v = np.random.randn(p)
    for _ in range(10):
        v = A.T @ (A @ v)
        v = v / np.linalg.norm(v)
    L_const = np.linalg.norm(A.T @ (A @ v))
    step_size = 1.0 / L_const
    
    AtA = A.T @ A
    Atb = A.T @ b
    
    objs = []
    
    for k in range(max_iter):
        x_old = x.copy()
        
        # Gradient step
        grad = AtA @ y - Atb
        x_temp = y - step_size * grad
        
        # Proximal step
        x = soft_thresholding(x_temp, step_size * lambda_reg)
        
        # Momentum step
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        
        objs.append(objective(x))
        
    return x, objs

# 3. Accelerated ADMM (VOS / Continuous-time perspective)
def acc_admm(A, b, lambda_reg, rho=1.0, max_iter=200, r=3.0):
    n, p = A.shape
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    
    z_hat = np.zeros(p)
    u_hat = np.zeros(p)
    
    alpha = 1.0
    
    AtA = A.T @ A
    Atb = A.T @ b
    
    L = np.linalg.cholesky(AtA + rho * np.eye(p))
    
    objs = []
    
    for k in range(max_iter):
        z_old = z.copy()
        u_old = u.copy()
        
        # x-update using z_hat and u_hat
        q = Atb + rho * (z_hat - u_hat)
        y = np.linalg.solve(L, q)
        x = np.linalg.solve(L.T, y)
        
        # z-update
        z = soft_thresholding(x + u_hat, lambda_reg / rho)
        
        # u-update
        u = u_hat + x - z
        
        # Momentum updates with parameter r
        alpha_next = (1 + np.sqrt(1 + 4 * alpha**2)) / 2
        # Using the continuous-time ODE inspired momentum: (k-1)/(k+r-1)
        beta = (k) / (k + r)
        
        z_hat = z + beta * (z - z_old)
        u_hat = u + beta * (u - u_old)
        alpha = alpha_next
        
        objs.append(objective(z))
        
    return z, objs

print("Running ADMM...")
z_admm, objs_admm = admm(A, b, lambda_reg, rho=10.0, max_iter=200)
print(f"ADMM final obj: {objs_admm[-1]}")

print("Running FISTA...")
x_fista, objs_fista = fista(A, b, lambda_reg, max_iter=200)
print(f"FISTA final obj: {objs_fista[-1]}")

print("Running Acc ADMM...")
z_acc_admm, objs_acc_admm = acc_admm(A, b, lambda_reg, rho=10.0, max_iter=200)
print(f"Acc ADMM final obj: {objs_acc_admm[-1]}")

plt.figure(figsize=(10, 6))
plt.plot(objs_admm, label='ADMM')
plt.plot(objs_fista, label='FISTA (Nesterov)')
plt.plot(objs_acc_admm, label='Accelerated ADMM (VOS)')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.title('Convergence of Optimization Algorithms for Lasso')
plt.legend()
plt.grid(True)
plt.savefig('report/images/convergence.png')

