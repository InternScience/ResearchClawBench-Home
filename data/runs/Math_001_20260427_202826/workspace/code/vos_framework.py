"""
Variable and Operator Splitting (VOS) framework.

Unifies Nesterov's accelerated gradient method (NAG) and ADMM under a single
continuous-time dynamical-system viewpoint following Su--Boyd--Candes (2015) and
Boyd--Parikh--Chu--Peleato--Eckstein (2011), with a strong Lyapunov-function
analysis in the spirit of Polyak (1964).

Two families of dynamics are considered:

    (NAG-ODE)    ẍ + (a/t) ẋ + ∇f(x) = 0,            (weakly convex)
    (NAG-SC)     ẍ + 2 sqrt(μ) ẋ + ∇f(x) = 0,        (μ-strongly convex)

and for composite problems  min_x f(x) + g(z)  s.t.  Ax + Bz = c,

    (ADMM-ODE)   ẋ = -∇f(x) - A^T λ
                 ż = -∂g(z) - B^T λ
                 λ̇ =  Ax + Bz - c

These dynamics admit strong Lyapunov functions of the form

    E(t) = t^2 (f(x) - f*) + ||v||^2 / 2     (weak convexity)
    E(t) = (f(x) - f*) + (μ/2) ||x - x*||^2 + (1/2)||v + sqrt(μ)(x-x*)||^2   (SC)
    E_admm(x,z,λ) = (1/2)||x-x*||^2 + (1/2)||z-z*||^2 + (1/(2ρ))||λ-λ*||^2

Discretizations of these ODEs reproduce, respectively, FISTA/NAG-restart and
ADMM, hence the unified "VOS" view used in this report.

This module exposes:
    * proximal_gradient (ISTA)
    * fista (NAG)
    * fista_restart (VOS-NAG, gradient-restart)
    * heavy_ball (Polyak)
    * admm_lasso (VOS-ADMM)
    * integrate_nag_ode, integrate_nag_sc_ode
    * lyapunov_nag, lyapunov_admm
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------
# Lasso problem container
# ---------------------------------------------------------------------
@dataclass
class LassoProblem:
    A: np.ndarray
    b: np.ndarray
    lam: float
    L: float = 0.0          # Lipschitz constant of grad of smooth part
    mu: float = 0.0         # strong-convexity modulus of smooth part (may be 0)

    def __post_init__(self):
        if self.L == 0.0:
            self.L = float(np.linalg.svd(self.A, compute_uv=False).max() ** 2)

    # Smooth term f(x) = 0.5 || A x - b ||^2
    def f(self, x):
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)

    def grad_f(self, x):
        return self.A.T @ (self.A @ x - self.b)

    # Non-smooth term g(x) = lam * ||x||_1
    def g(self, x):
        return self.lam * float(np.sum(np.abs(x)))

    def F(self, x):
        return self.f(x) + self.g(x)

    @staticmethod
    def soft_threshold(x, tau):
        return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


# ---------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------
@dataclass
class History:
    F:   list = field(default_factory=list)   # objective value
    err: list = field(default_factory=list)   # ||x - x_ref|| if x_ref given
    grad_norm: list = field(default_factory=list)
    time: list = field(default_factory=list)  # iteration index
    lyap: list = field(default_factory=list)  # Lyapunov function value


# ---------------------------------------------------------------------
# Proximal gradient (ISTA)
# ---------------------------------------------------------------------
def proximal_gradient(prob: LassoProblem, x0, n_iters=300, x_ref=None):
    L = prob.L
    x = x0.copy()
    hist = History()
    for k in range(n_iters):
        x = prob.soft_threshold(x - prob.grad_f(x) / L, prob.lam / L)
        hist.F.append(prob.F(x))
        hist.time.append(k)
        if x_ref is not None:
            hist.err.append(float(np.linalg.norm(x - x_ref)))
    return x, hist


# ---------------------------------------------------------------------
# FISTA (Nesterov accelerated proximal gradient)
# ---------------------------------------------------------------------
def fista(prob: LassoProblem, x0, n_iters=300, x_ref=None):
    L = prob.L
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    hist = History()
    for k in range(n_iters):
        x_new = prob.soft_threshold(y - prob.grad_f(y) / L, prob.lam / L)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        hist.F.append(prob.F(x))
        hist.time.append(k)
        if x_ref is not None:
            hist.err.append(float(np.linalg.norm(x - x_ref)))
    return x, hist


# ---------------------------------------------------------------------
# FISTA with gradient restart (VOS-NAG, Su--Boyd--Candes 2015 §5)
# ---------------------------------------------------------------------
def fista_restart(prob: LassoProblem, x0, n_iters=300, x_ref=None,
                  restart="gradient"):
    L = prob.L
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    hist = History()
    for k in range(n_iters):
        grad_y = prob.grad_f(y)
        x_new = prob.soft_threshold(y - grad_y / L, prob.lam / L)
        # gradient restart criterion: <∇f(y), x_new - x> > 0
        if restart == "gradient" and float(grad_y @ (x_new - x)) > 0:
            t = 1.0
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        hist.F.append(prob.F(x))
        hist.time.append(k)
        if x_ref is not None:
            hist.err.append(float(np.linalg.norm(x - x_ref)))
    return x, hist


# ---------------------------------------------------------------------
# Heavy-ball (Polyak 1964) for the *smooth* part only.
# Used here on the strongly-convex smooth surrogate (no regularization)
# to compare with NAG-SC.
# ---------------------------------------------------------------------
def heavy_ball_smooth(grad_f, x0, L, mu, n_iters=300, x_ref=None,
                      f=None):
    # Optimal Polyak constants
    alpha = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    x_prev = x0.copy()
    x = x0.copy()
    hist = History()
    for k in range(n_iters):
        g = grad_f(x)
        x_new = x - alpha * g + beta * (x - x_prev)
        x_prev, x = x, x_new
        hist.time.append(k)
        if f is not None:
            hist.F.append(float(f(x)))
        if x_ref is not None:
            hist.err.append(float(np.linalg.norm(x - x_ref)))
        hist.grad_norm.append(float(np.linalg.norm(g)))
    return x, hist


# ---------------------------------------------------------------------
# NAG-SC discretization for the smooth strongly-convex surrogate.
# This is the symplectic-Euler discretization of  ẍ + 2√μ ẋ + ∇f = 0.
# It coincides with the Nesterov constant-momentum scheme.
# ---------------------------------------------------------------------
def nag_sc_smooth(grad_f, x0, L, mu, n_iters=300, x_ref=None, f=None):
    s = 1.0 / L
    momentum = (1.0 - np.sqrt(mu * s)) / (1.0 + np.sqrt(mu * s))
    x_prev = x0.copy()
    x = x0.copy()
    hist = History()
    for k in range(n_iters):
        y = x + momentum * (x - x_prev)
        x_new = y - s * grad_f(y)
        x_prev, x = x, x_new
        hist.time.append(k)
        if f is not None:
            hist.F.append(float(f(x)))
        if x_ref is not None:
            hist.err.append(float(np.linalg.norm(x - x_ref)))
    return x, hist


# ---------------------------------------------------------------------
# ADMM for Lasso  min_x 0.5||A x - b||^2 + lam ||z||_1   s.t. x = z
# (this is the "VOS-ADMM" algorithmic specialization used in the paper).
# ---------------------------------------------------------------------
def admm_lasso(prob: LassoProblem, x0, n_iters=300, rho=1.0, x_ref=None):
    A, b = prob.A, prob.b
    p = A.shape[1]
    x = x0.copy()
    z = x0.copy()
    u = np.zeros(p)                # scaled dual: u = lambda / rho
    # Cache (A^T A + rho I)^{-1} via Woodbury for tall problems.
    # Here m=1000 < p=2000, so use Woodbury:
    # (A^T A + rho I)^{-1} = (1/rho) I - (1/rho^2) A^T (I + (1/rho) A A^T)^{-1} A
    AAt = A @ A.T
    M = np.eye(A.shape[0]) + AAt / rho
    L_chol = np.linalg.cholesky(M)
    Atb = A.T @ b
    hist = History()
    for k in range(n_iters):
        # x-update: minimize 0.5||A x - b||^2 + (rho/2)||x - z + u||^2
        rhs = Atb + rho * (z - u)
        # Woodbury: x = (1/rho)*rhs - (1/rho^2) A^T M^{-1} A rhs
        v = A @ rhs
        w = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, v))
        x = (rhs - A.T @ w / rho) / rho
        # z-update via soft-thresholding
        z = prob.soft_threshold(x + u, prob.lam / rho)
        # dual update
        u = u + (x - z)
        hist.time.append(k)
        hist.F.append(prob.F(z))
        if x_ref is not None:
            hist.err.append(float(np.linalg.norm(z - x_ref)))
        # primal-dual residuals as Lyapunov-like quantities
        r_p = float(np.linalg.norm(x - z))
        hist.grad_norm.append(r_p)
    return z, hist, dict(x=x, z=z, u=u)


# ---------------------------------------------------------------------
# ODE integrators for the continuous-time Nesterov dynamics.
# ---------------------------------------------------------------------
def integrate_nag_ode(grad_f, x0, t_max=10.0, dt=1e-3, a=3.0,
                      f=None, x_ref=None, t_start=1e-3):
    """Integrate ẍ + (a/t) ẋ + ∇f(x) = 0 with semi-implicit Euler."""
    x = x0.copy()
    v = np.zeros_like(x0)
    n_steps = int((t_max - t_start) / dt)
    hist = History()
    t = t_start
    for k in range(n_steps):
        g = grad_f(x)
        v = v + dt * (-(a / t) * v - g)
        x = x + dt * v
        t += dt
        if k % max(1, n_steps // 500) == 0:
            hist.time.append(t)
            if f is not None:
                hist.F.append(float(f(x)))
            if x_ref is not None:
                hist.err.append(float(np.linalg.norm(x - x_ref)))
    return x, hist


def integrate_nag_sc_ode(grad_f, x0, t_max=10.0, dt=1e-3, mu=1.0,
                         f=None, x_ref=None):
    """Integrate ẍ + 2√μ ẋ + ∇f(x) = 0 with semi-implicit Euler."""
    x = x0.copy()
    v = np.zeros_like(x0)
    damp = 2.0 * np.sqrt(mu)
    n_steps = int(t_max / dt)
    hist = History()
    for k in range(n_steps):
        g = grad_f(x)
        v = v + dt * (-damp * v - g)
        x = x + dt * v
        if k % max(1, n_steps // 500) == 0:
            hist.time.append(k * dt)
            if f is not None:
                hist.F.append(float(f(x)))
            if x_ref is not None:
                hist.err.append(float(np.linalg.norm(x - x_ref)))
    return x, hist


# ---------------------------------------------------------------------
# Lyapunov functions
# ---------------------------------------------------------------------
def lyapunov_nag_weak(t, x, v, f, f_star):
    """E(t) = t^2 (f(x) - f*) + 2 ||x - ... ||^2  (Su-Boyd-Candes Theorem 3)
    Here we use the simpler form  E = t^2 (f(x)-f*) + 0.5||v||^2
    used to certify the O(1/t^2) rate.
    """
    return t * t * (float(f(x)) - f_star) + 0.5 * float(v @ v)


def lyapunov_nag_sc(x, v, x_star, mu, f, f_star):
    """Strong Lyapunov for the strongly-convex NAG ODE
        E = (f(x) - f*) + (μ/2) ||x - x*||^2 + (1/2) || v + √μ (x - x*) ||^2
    """
    z = v + np.sqrt(mu) * (x - x_star)
    return (float(f(x)) - f_star) + 0.5 * mu * float((x - x_star) @ (x - x_star)) \
        + 0.5 * float(z @ z)


def lyapunov_admm(x, z, u, x_star, z_star, u_star, rho):
    """Boyd--Parikh ADMM Lyapunov-style residual
        V = ρ ||z - z*||^2 + (1/ρ) ||u - u*||^2
    """
    return rho * float((z - z_star) @ (z - z_star)) \
        + (1.0 / rho) * float((u - u_star) @ (u - u_star))
