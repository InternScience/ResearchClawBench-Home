import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# Load data
data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
A = data['A']
b = data['b']
x_true = data['x_true']

lambda_reg = 0.1 * np.max(np.abs(A.T @ b))

def soft_thresholding(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0)

def fista(A, b, lambda_reg, max_iter=200):
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
    
    for k in range(max_iter):
        x_old = x.copy()
        grad = AtA @ y - Atb
        x_temp = y - step_size * grad
        x = soft_thresholding(x_temp, step_size * lambda_reg)
        
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        
    return x

x_fista = fista(A, b, lambda_reg, max_iter=200)

plt.figure(figsize=(12, 5))
plt.plot(x_true, 'o', label='True Coefficients', markersize=4, alpha=0.7)
plt.plot(x_fista, 'x', label='Recovered (FISTA)', markersize=4, alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Coefficient Value')
plt.title('Sparsity Recovery of Lasso')
plt.legend()
plt.grid(True)
plt.savefig('report/images/recovery.png')

