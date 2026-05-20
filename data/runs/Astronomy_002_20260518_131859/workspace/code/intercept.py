"""Alpha intercept computation for Hubble flow analysis.

This module computes the alpha intercept in the Hubble diagram, which is used
in the equations module to connect absolute magnitudes to the Hubble constant.
It supports various velocity correction schemes, covariance matrices, and cosmographic
corrections.

The relevant relation for the intercept (alpha) is: log10(H0) = 0.2 * M_B + intercept + 5,
where M_B is the absolute magnitude and the intercept encapsulates the cosmological and k-correction terms.

Functions:
    compute_alpha: Main function to compute alpha intercept and its uncertainty.
    _vel_to_z: Convert velocity to redshift (two methods are supported).
    _z_to_vel: Convert redshift to velocity.
    _kz: Cosmographic correction factor k(z).
"""

import numpy as np
from scipy.constants import c as c_m_s
import warnings

def _vel_to_z(v, c, optical):
    """Convert velocity to redshift.
    
    Args:
        v (float or np.ndarray): Velocity in km/s.
        c (float): Speed of light in km/s.
        optical (bool): If True, use optical definition (z = v/c).
    
    Returns:
        float or np.ndarray: Redshift z.
    """
    if optical:
        return v / c
    return np.sqrt((1 + v / c) / (1 - v / c)) - 1

def _z_to_vel(z, c, optical):
    """Convert redshift to velocity.
    
    Args:
        z (float or np.ndarray): Redshift.
        c (float): Speed of light in km/s.
        optical (bool): If True, use optical definition (v = c*z).
    
    Returns:
        float or np.ndarray: Velocity in km/s.
    """
    if optical:
        return c * z
    r = 1 + z
    return c * (r**2 - 1) / (r**2 + 1)

def _kz(z, q0, j0, ignore_cosmology):
    """Compute cosmographic correction factor k(z).
    
    Applies Taylor expansion of luminosity distance through O(z^2):
        k(z) = 1 + 0.5*(1 - q0)*z - (1/6)*(1 - q0 - 3*q0^2 + j0)*z^2
    
    Args:
        z (float or np.ndarray): Redshift.
        q0 (float): Deceleration parameter (typically -0.55 for ΛCDM).
        j0 (float): Jerk parameter (typically 1.0 for ΛCDM).
        ignore_cosmology (bool): If True, return 1 (no correction).
    
    Returns:
        float or np.ndarray: Correction factor k(z).
    """
    if ignore_cosmology:
        return np.ones_like(z, dtype=float)
    return (
        1
        + 0.5 * (1 - q0) * z
        - (1.0 / 6.0) * (1 - q0 - 3 * q0**2 + j0) * z**2
    )

def compute_alpha(
    mag, mag_err=None, vcorr=None, vpec=None, vcmb=None, vhel=None,
    q0=-0.55, j0=1.0, vdisp=240.0, covar=None,
    redshift_range=None, apply_redshift_range=False, ignore_offdiag=False, simplified=False,
    optical=False, ignore_cosmology=False, debug=False
):
    """Compute alpha intercept in Hubble flow magnitude-redshift relation.
    
    The alpha intercept relates the observed magnitudes to cosmologically-corrected
    velocities: alpha = log10(v*k(z)) - 0.2*m, where k(z) is the cosmographic
    correction factor. This function performs weighted least-squares fitting with
    support for covariance matrices and peculiar velocity uncertainties.
    
    Args:
        mag (np.ndarray): Observed magnitudes or distance moduli.
        mag_err (np.ndarray, optional): Magnitude uncertainties. Required if covar is None.
        vcorr (np.ndarray, optional): Cosmologically-corrected velocities (km/s).
            Provide either vcorr OR (vcmb and vpec).
        vpec (np.ndarray, optional): Peculiar velocities (km/s). Used with vcmb.
        vcmb (np.ndarray, optional): CMB-frame velocities (km/s). Enables advanced mode.
        vhel (np.ndarray, optional): Heliocentric velocities (km/s). Used with vcmb.
        q0 (float, optional): Deceleration parameter. Defaults to -0.55.
        j0 (float, optional): Jerk parameter. Defaults to 1.0.
        vdisp (float, optional): Peculiar velocity dispersion (km/s). Defaults to 240.0.
        covar (np.ndarray, optional): Magnitude covariance matrix. If provided, used
            instead of diagonal mag_err.
        redshift_range (tuple, optional): (zmin, zmax) for redshift selection.
        apply_redshift_range (bool, optional): Apply redshift selection. Defaults to False.
        ignore_offdiag (bool, optional): Use only diagonal of covar. Defaults to False.
        simplified (bool, optional): Use simplified velocity model. Defaults to False.
        optical (bool, optional): Use optical redshift definition. Defaults to False.
        ignore_cosmology (bool, optional): Skip cosmographic corrections. Defaults to False.
        debug (bool, optional): Enable debug output. Defaults to False.
    
    Returns:
        tuple: (alpha, alpha_err, chisq, ndof) where:
            - alpha (float): Fitted intercept value.
            - alpha_err (float): 1-sigma uncertainty on alpha.
            - chisq (float): Chi-squared statistic of the fit.
            - ndof (int): Number of degrees of freedom.
    
    Raises:
        ValueError: If neither vcorr nor vcmb is provided, or if both vpec and vcorr
            are provided simultaneously, or if no objects remain after redshift selection.
    
    Note:
        There are two modes:
        1. Simplified: Requires only vcorr. Uses simple weighted least squares.
        2. Advanced: Requires vcmb and either vpec or vcorr. Includes proper velocity transformations.
    """
    c = (c_m_s / 1000.0)  # speed of light in km/s

    if vcmb is None:
        # Simplified branch using vcorr only
        if vcorr is None:
            raise ValueError("Must provide either vcmb or vcorr.")

        zhd = _vel_to_z(vcorr, c, optical)
        useflag = np.ones_like(zhd, dtype=bool)

        if apply_redshift_range:
            if redshift_range is not None and (redshift_range[0] != 0 or redshift_range[1] != 0):
                useflag = (zhd >= redshift_range[0]) & (zhd <= redshift_range[1])

            if np.sum(useflag) == 0:
                raise ValueError(
                    "No objects for this calibrator type after redshift selection"
                )
        
        kz = _kz(zhd, q0, j0, ignore_cosmology)

        alphavec = np.log10(vcorr) + np.log10(kz) - 0.2 * mag
        errsqvec = (0.2 * mag_err) ** 2 + (np.log10(vcorr + vdisp) - np.log10(vcorr)) ** 2

        w = useflag / errsqvec
        alpha = np.sum(w * alphavec) / np.sum(w)
        alpha_err = 1.0 / np.sqrt(np.sum(w))
        chisq = np.sum(w * (alphavec - alpha) ** 2)
        ndof = np.sum(useflag) - 1
        return alpha, alpha_err, chisq, ndof

    else:
        # Advanced branch with vcmb
        zcmb = _vel_to_z(vcmb, c, optical)

        # Require exactly one of vpec, vcorr
        if (vpec is None) and (vcorr is None):
            raise ValueError("Need one of vpec or vcorr.")
        if (vpec is not None) and (vcorr is not None):
            raise ValueError("Provide exactly one of vpec or vcorr (got both).")

        if vpec is None:
            # Given vcorr
            zcorr = _vel_to_z(vcorr, c, optical)
            zpec = (1 + zcmb) / (1 + zcorr) - 1
            vpec = _z_to_vel(zpec, c, optical)
        else:
            # Given vpec
            zpec = _vel_to_z(vpec, c, optical)
            zcorr = (1 + zcmb) / (1 + zpec) - 1
            vcorr = _z_to_vel(zcorr, c, optical)

        if vhel is None:
            warnings.warn("vhel not provided; approximating vhel = vcmb (not exact).")
            vhel = vcmb

        zhel = _vel_to_z(vhel, c, optical)
        zhd = (1 + zcmb) / (1 + zpec) - 1  # equivalent to vcorr = vobs - vpec

        t1 = (1 + zhel) / (1 + zhd)
        t2 = c * zhd
        t3 = _kz(zhd, q0, j0, ignore_cosmology)

    # Model and velocity variance (velocity term already in log10 units)
    m_model = 5 * np.log10(t1 * t2 * t3)
    vel_var = (np.log10(vcorr + vdisp) - np.log10(vcorr)) ** 2

    if simplified:
        zhd = vcorr / c
        kz = _kz(zhd, q0, j0, ignore_cosmology)
        m_model = 5 * (np.log10(vcorr) + np.log10(kz))

    useflag = np.ones_like(zhd, dtype=bool)
    if apply_redshift_range:
        if redshift_range is not None and (redshift_range[0] != 0 or redshift_range[1] != 0):
            useflag = (zhd >= redshift_range[0]) & (zhd <= redshift_range[1])

        if np.sum(useflag) == 0:
            raise ValueError(
                "No objects for this calibrator type after redshift selection"
            )

    wh = np.where(useflag)[0]

    # --- Alpha-units: alphavec = 0.2 * (m_model - mag) ---
    alphavec = 0.2 * (m_model - mag)
    data = alphavec[wh]
    vel_var_used = vel_var[wh]  # already in alpha-units

    if covar is None:
        if mag_err is None:
            raise ValueError("Need mag_err if covar is not provided.")
        # Variance in alpha-units: (0.2*mag_err)^2 + vel_var
        errsq = (0.2 * mag_err[wh])**2 + vel_var_used
        weight = 1.0 / errsq
        alpha = np.sum(weight * data) / np.sum(weight)
        alpha_err = 1.0 / np.sqrt(np.sum(weight))
        chisq = np.sum(weight * (data - alpha)**2)
        ndof = len(wh) - 1
    else:
        n = len(wh)
        # Convert input mag-covariance to alpha-units: Cov_alpha = Cov_mag / 25
        cov_used = (np.diag(covar[wh, wh]) if ignore_offdiag
                    else covar[np.ix_(wh, wh)])
        cov_used = cov_used / 25.0
        # Add velocity variance (already in alpha-units) on the diagonal
        cov_used = cov_used + np.diag(vel_var_used)

        coeffs = np.ones(n)  # unit coefficient in alpha-units
        invcov = np.linalg.inv(cov_used)
        temp = coeffs @ invcov @ coeffs
        invtemp = 1.0 / temp
        alpha = invtemp * (coeffs @ invcov @ data)
        alpha_err = np.sqrt(invtemp)
        chisq = (data - alpha * coeffs) @ invcov @ (data - alpha * coeffs)
        ndof = n - 1

    return alpha, alpha_err, chisq, ndof
