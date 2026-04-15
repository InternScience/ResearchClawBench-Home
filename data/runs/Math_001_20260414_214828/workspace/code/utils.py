import numpy as np
def load_data():
    d = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
    return d['A'], d['b'], d['x_true']
def lasso_obj(A, b, lam, x):
    axb = A @ x - b
    return 0.5 * np.dot(axb, axb) + lam * np.linalg.norm(x, 1)
def lasso_smooth_grad(A, b, x):
    return A.T @ (A @ x - b)
def prox_l1(tau, v):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)
print('Utils loaded')
",
<parameter name="path">code/utils.py