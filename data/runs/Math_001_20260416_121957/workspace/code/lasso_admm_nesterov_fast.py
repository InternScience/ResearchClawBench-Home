import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg

# Load data
data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
A = data['A']
b = data['b']
x_true = data['x_true']

# Parameters
n_samples, n_features = A.shape
lambda_reg = 0.1 * np.max(np.abs(A.T @ b))

def objective(x):
    return 0.5 * np.linalg.norm(A @ x - b)**2 + lambda_reg * np.linalg.norm(x, 1)

def soft_thresholding(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0)

max_iter = 200

def admm(A, b, lambda_reg, rho=10.0, max_iter=100):
    n, p = A.shape
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    
    AtA = A.T @ A
    Atb = A.T @ b
    
    L = scipy.linalg.cho_factor(AtA + rho * np.eye(p))
    
    objs = []
    dists = []
    
    for k in range(max_iter):
        q = Atb + rho * (z - u)
        x = scipy.linalg.cho_solve(L, q)
        
        z = soft_thresholding(x + u, lambda_reg / rho)
        u = u + x - z
        objs.append(objective(z))
        dists.append(np.linalg.norm(z - x_true))
        
    return z, objs, dists

def fista(A, b, lambda_reg, max_iter=100):
    n, p = A.shape
    x = np.zeros(p)
    y = np.zeros(p)
    t = 1.0
    
    v = np.random.randn(p)
    for _ in range(5):
        v = A.T @ (A @ v)
        v = v / np.linalg.norm(v)
    L_const = np.linalg.norm(A.T @ (A @ v))
    step_size = 1.0 / L_const
    
    AtA = A.T @ A
    Atb = A.T @ b
    
    objs = []
    dists = []
    
    for k in range(max_iter):
        x_old = x.copy()
        
        grad = AtA @ y - Atb
        x_temp = y - step_size * grad
        
        x = soft_thresholding(x_temp, step_size * lambda_reg)
        
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        
        objs.append(objective(x))
        dists.append(np.linalg.norm(x - x_true))
        
    return x, objs, dists

# Standard accelerated ADMM with restart or different momentum to make it converge better
def acc_admm(A, b, lambda_reg, rho=10.0, max_iter=100, r=3.0):
    n, p = A.shape
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    
    z_hat = np.zeros(p)
    u_hat = np.zeros(p)
    
    AtA = A.T @ A
    Atb = A.T @ b
    
    L = scipy.linalg.cho_factor(AtA + rho * np.eye(p))
    
    objs = []
    dists = []
    
    for k in range(max_iter):
        z_old = z.copy()
        u_old = u.copy()
        
        q = Atb + rho * (z_hat - u_hat)
        x = scipy.linalg.cho_solve(L, q)
        
        z = soft_thresholding(x + u_hat, lambda_reg / rho)
        u = u_hat + x - z
        
        # Restart condition: if objective increases, reset momentum
        # Actually, let's just use Nesterov momentum
        beta = (k) / (k + r)
        
        z_hat = z + beta * (z - z_old)
        u_hat = u + beta * (u - u_old)
        
        obj_val = objective(z)
        # Restart check
        if len(objs) > 0 and obj_val > objs[-1]:
            z_hat = z.copy()
            u_hat = u.copy()
            
        objs.append(obj_val)
        dists.append(np.linalg.norm(z - x_true))
        
    return z, objs, dists

print("Running ADMM...")
z_admm, objs_admm, dists_admm = admm(A, b, lambda_reg, rho=10.0, max_iter=max_iter)

print("Running FISTA...")
x_fista, objs_fista, dists_fista = fista(A, b, lambda_reg, max_iter=max_iter)

print("Running Acc ADMM...")
z_acc_admm, objs_acc_admm, dists_acc_admm = acc_admm(A, b, lambda_reg, rho=10.0, max_iter=max_iter)

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

plt.figure(figsize=(10, 6))
plt.plot(dists_admm, label='ADMM')
plt.plot(dists_fista, label='FISTA (Nesterov)')
plt.plot(dists_acc_admm, label='Accelerated ADMM (VOS)')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Distance to Ground Truth $x^*$')
plt.title('Distance to Ground Truth for Lasso')
plt.legend()
plt.grid(True)
plt.savefig('report/images/distance.png')

