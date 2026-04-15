import numpy as np
from .utils import load_data, lasso_obj, lasso_smooth_grad, prox_l1
def fista(A, b, lam, x0, max_iter=10000, s=0.1, tol=1e-6):
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    objs = []
    errors = []
    xt = np.load('outputs/x_true.npy')
    for k in range(max_iter):
        grad = lasso_smooth_grad(A, b, y)
        x_new = prox_l1(lam * s, y - s * grad)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new
        ob = lasso_obj(A, b, lam, x)
        er = np.linalg.norm(x - xt)
        objs.append(ob)
        errors.append(er)
        if er < tol:
            break
    np.save('outputs/fista_objs.npy', np.array(objs))
    np.save('outputs/fista_errors.npy', np.array(errors))
    print('FISTA done, final obj', ob, 'error', er)
",
<parameter name="path">code/fista.py