"""Linear system solver with covariance handling.

This module solves the weighted least-squares problem for constraining cosmological
parameters (primarily H0 and absolute magnitudes) from distance ladder data. It
handles potentially singular covariance matrices using pseudoinverse techniques
and computes parameter uncertainties from the solution covariance.

The system being solved is: minimize (Y - AX)^T C^{-1} (Y - AX)
where:
    - A is the coefficient matrix (equations × parameters)
    - X is the parameter vector
    - Y is the observation vector
    - C is the covariance matrix

Functions:
    solve_system: Solve weighted least-squares system and compute uncertainties.
"""

import numpy as np
import scipy
from h0_constrainer import config_reader

def solve_system(equation_data):
    """Solve weighted least-squares system for cosmological parameters.
    
    Solves the linear system (A^T C^{-1} A)x = A^T C^{-1} y for parameter values,
    where A is the coefficient matrix, C is the covariance matrix, and y contains
    observed values. Uses pseudoinverse for potentially rank-deficient covariance
    matrices and computes parameter uncertainties.
    
    Args:
        equation_data (dict): Dictionary from build_equations() containing:
            - coeffs (np.ndarray): Coefficient matrix A (neq × npars).
            - covar (np.ndarray): Covariance matrix C (neq × neq).
            - yval (np.ndarray): Observed values vector y (neq).
            - npars (int): Number of parameters.
            - neq (int): Number of equations.
            - ihub (int): Index of log10(H0) parameter.
            - iabs (int or None): Index of absolute magnitude parameter.
            - icoma (int or None): Index of Coma distance modulus parameter.
    
    Returns:
        dict: Results dictionary containing:
            - params (np.ndarray): Best-fit parameter values.
            - invsolmat (np.ndarray): Parameter covariance matrix.
            - inv_covar (np.ndarray): Inverse of observation covariance.
            - covar (np.ndarray): Observation covariance matrix.
            - residuals (np.ndarray): Residuals y - A*params.
            - errmat (np.ndarray): Alternative parameter error matrix.
            - logh0_value (float): Best-fit log10(H0).
            - logh0_var (float): Variance of log10(H0).
            - h0_value (float): Best-fit H0 in km/s/Mpc.
            - h0_error (float): 1-sigma uncertainty on H0.
            - chi2 (float): Chi-squared statistic.
            - ndof (int): Degrees of freedom (adjusted for rank deficiency).
            - ndof_full (int): Nominal degrees of freedom (neq - npars).
            - npars (int): Number of parameters.
            - iabs (int or None): Absolute magnitude parameter index.
            - ihub (int): H0 parameter index.
            - mzero_value (float or None): Absolute magnitude M_B if applicable.
            - mzero_error (float or None): Uncertainty on M_B.
            - mu_coma_value (float or None): Coma distance modulus if applicable.
            - mu_coma_error (float or None): Uncertainty on mu_coma.
    
    Warnings:
        Issues warnings if covariance matrix is rank-deficient or ill-conditioned,
        and adjusts degrees of freedom accordingly.
    
    Note:
        H0 error is computed by propagating the uncertainty in log10(H0):
        σ_H0 = 10^(log10(H0) + σ_log10(H0)) - H0
    """
    coeffs = equation_data["coeffs"]
    covar = equation_data["covar"]
    yval = equation_data["yval"]
    npars = equation_data["npars"]
    neq = equation_data["neq"]
    ihub = equation_data["ihub"]
    iabs = equation_data["iabs"]
    icoma = equation_data["icoma"]



    # Use pseudoinverse to account for singular covariance
    inv_covar, covar_rank = scipy.linalg.pinv(covar, atol=1e-10, rtol=0.0, return_rank=True)
    ndof_full = neq - npars
    ndof= covar_rank - npars

    
    # Check condition number and rank of the covariance matrix
    covar_cond = np.linalg.cond(covar)
    covar_dim = covar.shape[0]


    # Warn user if singularity or near-singularity is detected
    if covar_rank < covar_dim:
        config_reader.wprint(f"\n=== WARNING === Covariance matrix is rank-deficient: rank {covar_rank} < {covar_dim}. Using pseudoinverse.")
        config_reader.wprint(f"=== WARNING === Degrees of freedom adjusted from {ndof_full} to {ndof} due to rank deficiency.")
    elif covar_cond > 1e10:
        config_reader.wprint(f"\n=== WARNING === Covariance matrix is ill-conditioned (cond={covar_cond:.2e}). Using pseudoinverse.")
        config_reader.wprint(f"=== WARNING === Degrees of freedom adjusted from {ndof_full} to {ndof} due to effective rank {covar_rank}.")


    solmat = coeffs.T @ inv_covar @ coeffs
    invsolmat = np.linalg.inv(solmat)
    solval = coeffs.T @ inv_covar @ yval
    params = invsolmat @ solval
    residuals = yval - coeffs @ params
    logh0_value = params[ihub]
    logh0_var = invsolmat[ihub, ihub]
    h0_value = 10 ** logh0_value
    h0_error = 10 ** (logh0_value + np.sqrt(logh0_var)) - h0_value
    chi2 = residuals.T @ inv_covar @ residuals

    mzero_value = None
    mzero_error = None
    if iabs is not None:
        mzero_value = params[iabs]
        mzero_error = np.sqrt(invsolmat[iabs,iabs])
    mu_coma_value = params[icoma] if icoma is not None else None
    mu_coma_error = None
    if icoma is not None:
        mu_coma_error = np.sqrt(invsolmat[icoma, icoma])

    return {
        # --- Data matrices ---
        "covar": covar,                # C: Original covariance matrix
        "inv_covar": inv_covar,        # C^-1: Inverse covariance matrix (pseudoinverse)
        "yval": yval,                  # Y: Observation vector
        "coeffs": coeffs,              # A: Coefficient matrix

        # --- Residuals ---
        "residuals": residuals,        # R = Y - AX


        # --- Solution matrices ---
        "solmat": solmat,              # S = A^T C^-1 A
        "solval": solval,              # P = A^T C^-1 Y
        "params": params,              # X = S^-1 P
        "invsolmat": invsolmat,        # S^-1: Error matrix


        # --- H0 results ---
        "logh0_value": logh0_value,
        "logh0_var": logh0_var,
        "h0_value": h0_value,
        "h0_error": h0_error,

        # --- Absolute magnitude results ---
        "mzero_value": mzero_value,
        "mzero_error": mzero_error,

        # --- Coma cluster results ---
        "mu_coma_value": mu_coma_value,
        "mu_coma_error": mu_coma_error,

        # --- Chi-squared and degrees of freedom ---
        "chi2": chi2,
        "ndof": ndof,
        "ndof_full": ndof_full,
        "npars": npars,

        # --- Parameter indices ---
        "iabs": iabs,
        "ihub": ihub,

        # --- Covariance diagnostics ---
        "covar_rank": covar_rank,
        "covar_cond": covar_cond,
        "covar_dim": covar_dim,
    }
