import numpy as np
from .utils import *
A, b, x_true = load_data()
lam = 0.01
L = np.linalg.norm(A, ord=2)**2
s = 1.0 / L
x = np.zeros_like(x_true)
y = x.copy()
t = 1.0
objs = []
iters = []
for k in range(5000):
    grad = lasso_smooth_grad(A, b, y)
    x_new = prox_l1(lam * s, y - s * grad)
    t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
    y = x_new + ((t - 1) / t_new) * (x_new - x)
    x = x_new
    t = t_new
    ob = lasso_obj(A, b, lam, x)
    objs.append(ob)
    iters.append(k)
    if k % 500 == 0:
        print(k, ob)
np.savez('outputs/fista.npz', objs=np.array(objs), iters=np.array(iters), x_final=x)
print('FISTA final obj', objs[-1])
",
<parameter name="path">code/run_fista.py