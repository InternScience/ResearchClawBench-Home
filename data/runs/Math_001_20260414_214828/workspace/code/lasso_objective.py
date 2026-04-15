import numpy as np
def lasso_obj(A, b, lam, x):
    axb = A @ x - b
    return 0.5 * np.dot(axb, axb) + lam * np.linalg.norm(x, 1)
def lasso_grad(A, b, lam, x):
    return A.T @ (A @ x - b)
def lasso_prox(lam, rho, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam / rho, 0)
