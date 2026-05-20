"""Equation system construction for distance ladder analysis.

This module builds the linear system of equations that relate distance moduli,
absolute magnitudes etc., and the Hubble constant. The equation system incorporates:
    - Host galaxy distance measurements (from primary distance indicators)
    - Calibrator equations linking distances and apparent magnitudes to absolute magnitudes
    - Hubble flow constraints linking absolute magnitudes to H0 (using alpha intercept)
    - Additional constraints (megamasers, EPM, Coma cluster)
    - Galaxy group constraints

The system is represented as:
    A × X = Y  with covariance C
where:
    - A is the coefficient matrix (neq × npars)
    - X is the parameter vector (npars)
    - Y is the observation vector (neq)
    - C is the covariance matrix (neq × neq)

Parameters include (when present):
    - Distance moduli for each host galaxy (there may be more than one measurementfor each host)
    - Absolute magnitudes M for each calibrator family/secondary distance indicator
    - log_10(H0) for the Hubble constant
    - Distance moduli for galaxy groups 
    - Distance modulus to Coma cluster 

The covariance matrix includes:
    - Measurement uncertainties

    Off-diagonal elements include:
        -  Geometric measurement uncertainties for anchor distance measurements (shared among measurements using the same anchor)
        -  Uncertainties from method-anchor-source (MAS) combinations (shared among measurements using the same method-anchor-source combination)
        -  Uncertainties from host-method-source (HMS) combinations (shared among measurements using the same host-method-source combination)

Functions:
    build_equations: Main function to construct the full equation system.
    compute_vcmb: Helper to convert redshift to CMB-frame velocity.
"""

import numpy as np
import pandas as pd
from .intercept import compute_alpha
from scipy.linalg import pinv
from scipy.constants import c as c_m_s
from .data_loader import uniquify
from h0_constrainer import config_reader

def build_equations(host_df, mm_df, epm_df, coma_df, groups_df,
                    q0, j0, vpec_error, sbf_vpec_error,
                    hms, hms_index, mu_hms_error,
                    sn1a_calib_df=None, sn2_calib_df=None, tf_calib_df=None, sbf_calib_df=None,
                    sn1a_hf_df=None, sn1a_hf_cov=None, sn1a_ignore_offdiag=False, sn1a_IR=False,
                    sn2_hf_df=None, sn2_hf_cov=None, tf_hf_df=None, tf_hf_cov=None, sbf_hf_df=None, sbf_hf_cov=None,
                    sn1a_vp_column=None, tf_v_column=None, sbf_v_column=None,
                    z_range_sn1a=False, z_range_sn2=False, z_range_tf=False, z_range_sbf=False,
                    anchors=None, anchor_index=None, mu_anchor_error=None,
                    n_mas=None, mas=None, mas_index=None, mas_error=None,
                    redshift_range=None,
                    alpha_sn1a=0.714, alpha_sn1a_error=0.002, alpha_sn2=None, alpha_sn2_error=None, alpha_tf=None, alpha_tf_error=None, alpha_sbf=None, alpha_sbf_error=None,
                    sigma_mag_fp=None):
    """Build the complete system of linear equations for distance ladder analysis.

    Constructs coefficient matrix A, observation vector Y, and covariance matrix C
    The system includes:
    - One equation per host distance measurement (there can be multiple measurements for the same host)
    - Calibrator equations linking host distances to standard candle magnitudes
    - Hubble flow equations linking magnitudes to H0 (using alpha intercept)
    - Additional constraints (MM, EPM, Coma)
    - Galaxy group constraints

    Args:
        host_df (pd.DataFrame): Host galaxy distance measurements.
        mm_df (pd.DataFrame or None): Megamaser distances.
        epm_df (pd.DataFrame or None): Expanding Photosphere Method SN II distances.
        coma_df (pd.DataFrame or None): Coma cluster SNe Ia.
        groups_df (pd.DataFrame or None): Galaxy group membership.
        q0 (float): Deceleration parameter for cosmographic corrections.
        j0 (float): Jerk parameter for cosmographic corrections.
        vpec_error (float): Peculiar velocity uncertainty (km/s).
        sbf_vpec_error (float): Peculiar velocity uncertainty for SBF (km/s).
        hms (np.ndarray): Unique host-method-source combinations.
        hms_index (np.ndarray): Index mapping host rows to HMS groups.
        mu_hms_error (np.ndarray): Uncertainties for each HMS combination.
        sn1a_calib_df (pd.DataFrame, optional): SNe Ia calibrators.
        sn2_calib_df (pd.DataFrame, optional): SNe II calibrators.
        tf_calib_df (pd.DataFrame, optional): Tully-Fisher calibrators.
        sbf_calib_df (pd.DataFrame, optional): SBF calibrators.
        sn1a_hf_df (pd.DataFrame, optional): SNe Ia Hubble flow sample.
        sn1a_hf_cov (np.ndarray, optional): SNe Ia covariance matrix.
        sn1a_ignore_offdiag (bool, optional): Ignore off-diagonal covariance.
        sn1a_IR (bool, optional): SNe Ia in infrared bands.
        sn2_hf_df, sn2_hf_cov: SNe II Hubble flow data.
        tf_hf_df, tf_hf_cov: Tully-Fisher Hubble flow data.
        sbf_hf_df, sbf_hf_cov: SBF Hubble flow data.
        sn1a_vp_column, tf_v_column, sbf_v_column (str, optional): Velocity column names.
        z_range_sn1a, z_range_sn2, z_range_tf, z_range_sbf (bool): Apply redshift cuts.
        anchors (np.ndarray, optional): Unique anchor names.
        anchor_index (np.ndarray, optional): Index mapping host rows to anchors.
        mu_anchor_error (np.ndarray, optional): Geometric anchor uncertainties.
        n_mas (int, optional): Number of unique MAS combinations.
        mas (np.ndarray, optional): Unique MAS names.
        mas_index (np.ndarray, optional): Index mapping host rows to MAS groups.
        mas_error (np.ndarray, optional): Uncertainties for each MAS.
        redshift_range (tuple, optional): (zmin, zmax) for Hubble flow.
        alpha_sn1a, alpha_sn1a_error: SNe Ia Hubble intercept (if not computing from HF).
        alpha_sn2, alpha_sn2_error: SNe II Hubble intercept.
        alpha_tf, alpha_tf_error: TF Hubble intercept.
        alpha_sbf, alpha_sbf_error: SBF Hubble intercept.
        sigma_mag_fp (float, optional): Fundamental plane uncertainty for Coma constraint.

    Returns:
        dict: Equation system dictionary with keys:
            - coeffs: Coefficient matrix A (neq × npars).
            - covar: Covariance matrix C (neq × neq).
            - yval: Observation vector Y (neq).
            - eq_descr: List of equation descriptions.
            - eq_shape: List of equation forms.
            - eq_covar: List of covariance descriptions.
            - parinfo: List of parameter metadata dicts.
            - npars: Number of parameters.
            - neq: Number of equations.
            - host_index: Index mapping combined hosts to unique hosts.
            - all_hosts: Array of unique host names.
            - iabs: Index of SNe Ia absolute magnitude parameter (or None).
            - ihub: Index of log10(H0) parameter.
            - icoma: Index of Coma distance modulus (or None).
            - nhosts: Number of unique hosts.
            - nsbf, ncoma, nmm, nepm: Counts of constraint types.
            - mas_host_error, anchor_host_error: Error arrays for plotting in the future.
            - ieq_*_start: Starting equation indices for each constraint type.

    Note:
        The covariance matrix includes:
            - Measurement uncertainties
            - Geometric measurement uncertainties for anchor distances
              (shared among all measurements using the same anchor)
            - MAS (Method-Anchor-Source) uncertainties
              (shared among all measurements using the same method, anchor, and source)
            - HMS (Host-Method-Source) uncertainties
              (shared among all measurements of the same host, with the same method and source)
    """

    # Combine hosts from host data and groups
    combined_hosts = list(host_df["host"].astype(str))

    # Add extra hosts that are only in groups (not in host_df)
    extra_hosts = []
    if groups_df is not None and len(groups_df) > 0:
        host_set = set(host_df["host"].astype(str))
        extra_hosts = [h for h in groups_df["host"].astype(str).unique() if h not in host_set]
        combined_hosts += extra_hosts

    # Dynamically add any non-None calibrator DataFrames
    for df in [sn1a_calib_df, sn2_calib_df, tf_calib_df, sbf_calib_df]:
        if df is not None and len(df) > 0:
            combined_hosts += list(df["host"].astype(str))

    # Build unique host set and index mappings
    all_hosts, host_index = uniquify(combined_hosts)
    nhosts = len(all_hosts)

    # Slice index mapping for host data
    host_index_2 = host_index[:len(host_df) + len(extra_hosts)]

    # Determine host index splits for each calibrator dataset
    offset = len(host_df) + len(extra_hosts)
    host_index_sn1a = host_index_sn2 = host_index_tf = host_index_sbf = None  

    if sn1a_calib_df is not None:
        host_index_sn1a = host_index[offset:offset + len(sn1a_calib_df)]
        offset += len(sn1a_calib_df)

    if sn2_calib_df is not None:
        host_index_sn2 = host_index[offset:offset + len(sn2_calib_df)]
        offset += len(sn2_calib_df)

    if tf_calib_df is not None:
        host_index_tf = host_index[offset:offset + len(tf_calib_df)]
        offset += len(tf_calib_df)  

    if sbf_calib_df is not None:  # SBF
        host_index_sbf = host_index[offset:offset + len(sbf_calib_df)]
        offset += len(sbf_calib_df)

    # --- Create the parameters to be constrained ---

    # 1) One distance-modulus parameter per host (columns for hosts)
    parinfo = []
    for host in map(str, all_hosts):
        parinfo.append({
            "name": f"Distance modulus for {host}",
            "value": 0.0,
            "fixed": False,
            "limits": [0.0, 0.0],
            "limited": [False, False],
            "type": 0,
            "objname": host,
        })

    # 2) One distance-modulus parameter per GROUP (μ_group), if any groups present
    group_param_index = {}
    if groups_df is not None and len(groups_df) > 0:
        groups_unique = pd.unique(groups_df["group"].astype(str))
        for g in groups_unique:
            parinfo.append({
                "name": f"Distance modulus for group {g}",
                "value": 0.0,
                "fixed": False,
                "limits": [0.0, 0.0],
                "limited": [False, False],
                "type": 0,
                "objname": g,
            })
            group_param_index[g] = len(parinfo) - 1

    # 3) Absolute magnitude / offset parameters for each calibrator family (only if present)
    iabs = None
    if sn1a_calib_df is not None and len(sn1a_calib_df) > 0:
        parinfo.append({
            "name": "Reference absolute magnitude of Type Ia supernovae",
            "value": 0.0,
            "fixed": False,
            "limits": [0.0, 0.0],
            "limited": [False, False],
            "type": 0,
            "objname": "SNe Ia M_B"
        })
        iabs = len(parinfo) - 1

    iabs_sn2 = None
    if sn2_calib_df is not None and len(sn2_calib_df) > 0:
        parinfo.append({
            "name": "Reference absolute magnitude of Type II supernovae",
            "value": 0.0,
            "fixed": False,
            "limits": [0.0, 0.0],
            "limited": [False, False],
            "type": 0,
            "objname": "SNe II M_B"
        })
        iabs_sn2 = len(parinfo) - 1

    iabs_tf = None
    if tf_calib_df is not None and len(tf_calib_df) > 0:
        parinfo.append({
            "name": "Offsetted absolute magnitude of Tully-Fisher galaxy",
            "value": 0.0,
            "fixed": False,
            "limits": [0.0, 0.0],
            "limited": [False, False],
            "type": 0,
            "objname": "TF M_offset"
        })
        iabs_tf = len(parinfo) - 1

    iabs_sbf = None
    if sbf_calib_df is not None and len(sbf_calib_df) > 0:
        parinfo.append({
            "name": "SBF absolute magnitude offset (m110 = M110 + M110_offset + mu)",
            "value": 0.0,
            "fixed": False,
            "limits": [0.0, 0.0],
            "limited": [False, False],
            "type": 0,
            "objname": "SBF M_offset"
        })
        iabs_sbf = len(parinfo) - 1

    # 4) Parameter for log10(H0)
    parinfo.append({
        "name": "log_10 (Hubble constant) in km/s/Mpc",
        "value": 0.0,
        "fixed": False,
        "limits": [0.0, 0.0],
        "limited": [False, False],
        "type": 0,
        "objname": "log_10(H0)"
    })
    ihub = len(parinfo) - 1

    # 5) If Coma data exists, add the μ_coma parameter before matrix allocation
    ncoma = len(coma_df) if (coma_df is not None) else 0
    if ncoma > 0 and not any(p["objname"] == "mu_coma" for p in parinfo):
        parinfo.append({
            "name": "Distance modulus to the Coma cluster",
            "value": 0.0,
            "fixed": False,
            "limits": [0.0, 0.0],
            "limited": [False, False],
            "type": 0,
            "objname": "mu_coma",
        })

    # Final parameter count
    npars = len(parinfo)

    # Initialize alpha (Hubble intercept) variables - will be set if corresponding HF is used
    a_sn1a = a_sn1a_err = chisq_sn1a_hf = ndof_sn1a_hf = None
    a_sn2 = a_sn2_err = chisq_sn2_hf = ndof_sn2_hf = None
    a_tf = a_tf_err = chisq_tf_hf = ndof_tf_hf = None
    a_sbf = a_sbf_err = chisq_sbf_hf = ndof_sbf_hf = None

    # --- Equation counts ---

    # Base host equations
    neq = len(host_df)

    # Calibrator equations (+1 per family for the absolute-magnitude link when present)
    nsn1a = (len(sn1a_calib_df) + 1) if (sn1a_calib_df is not None and len(sn1a_calib_df) > 0) else 0
    nsn2  = (len(sn2_calib_df)  + 1) if (sn2_calib_df  is not None and len(sn2_calib_df)  > 0) else 0
    ntf   = (len(tf_calib_df)   + 1) if (tf_calib_df   is not None and len(tf_calib_df)   > 0) else 0
    nsbf  = (len(sbf_calib_df)  + 1) if (sbf_calib_df  is not None and len(sbf_calib_df)  > 0) else 0

    # Megamaser / EPM equations (if present)
    nmm  = len(mm_df)  if (mm_df  is not None) else 0
    nepm = len(epm_df) if (epm_df is not None) else 0

    # Group equations: one per (group, host) row in groups_df
    ngroups_eq = len(groups_df) if (groups_df is not None and len(groups_df) > 0) else 0

    # Coma contributes ncoma + 2 (as per your existing logic)
    neq += nsn1a + nsn2 + ntf + nsbf + nmm + nepm + ngroups_eq + (ncoma + 2 if ncoma > 0 else 0)

    # --- Allocate matrices and vectors with updated sizes ---
    covar   = np.zeros((neq, neq), dtype=float)
    coeffs  = np.zeros((neq, npars), dtype=float)
    yval    = np.zeros(neq, dtype=float)
    yvar    = np.zeros(neq, dtype=float)
    eq_descr = ["" for _ in range(neq)]
    eq_shape = ["" for _ in range(neq)]
    eq_covar = ["" for _ in range(neq)]
    eq_counter = 0


    # --- Host distance equations ---
    for k in range(len(host_df)):
        ieq = k
        ihost = host_index_2[k]
        coeffs[ieq, ihost] = 1.0
        yval[ieq] = host_df.iloc[k]["mu_host"]
        yvar[ieq] = host_df.iloc[k]["mu_error"] ** 2
        covar[ieq, ieq] = yvar[ieq]
        eq_descr[ieq] = f"Host equation for {host_df.iloc[k]['host']} with MAS: {host_df.iloc[k]['mas_name']}"
        eq_shape[ieq] = "mu_host = reported value"
        eq_covar[ieq] = "Variance of host measurement"
        eq_counter += 1


    # --- Update covariance matrix using anchor uncertainties ---
    # Only if anchors, anchor_index, and mu_anchor_error are provided.
    mas_host_error = np.zeros(len(host_df), dtype=float) # MAS reference error
    anchor_host_error = np.zeros(len(host_df), dtype=float) # Anchor geometric error
    if anchors is not None and anchor_index is not None and mu_anchor_error is not None:
        for k in range(len(anchors)):
            wh = np.where(anchor_index == k)[0]
            if len(wh) > 0:
                anchor_host_error[wh] = mu_anchor_error[k]  #Used in plots
                covar[np.ix_(wh, wh)] += mu_anchor_error[k] ** 2
                for idx in wh:
                    eq_covar[idx] += f" + geometric anchor {anchors[k]} distance"
    for k in range(n_mas):
        wh = np.where(mas_index == k)[0]
        if len(wh) > 0:
            mas_host_error[wh] = mas_error[k] #Used in plots
            covar[np.ix_(wh, wh)] += mas_error[k] ** 2
            for idx in wh:
                eq_covar[idx] += f" + MAS {mas[k]} variance"
    # --- Update covariance matrix using host-method-source (HMS) uncertainties ---
    for k in range(len(hms)):
        wh = np.where(hms_index == k)[0]
        if len(wh) > 1:
            for i in range(len(wh)):
                for j in range(len(wh)):
                    if i != j:
                        covar[wh[i], wh[j]] += mu_hms_error[k] ** 2
            for idx in wh:
                eq_covar[idx] += f" + HMS {hms[k]} variance"



    # --- GROUP equations: mu_host - mu_group = 0 with variance = sigma^2 ---
    if groups_df is not None and len(groups_df) > 0:
        # Map host -> parameter index directly from uniquify output
        host_to_par = {h: i for i, h in enumerate(all_hosts)}
        for k in range(len(groups_df)):
            ieq = eq_counter
            ihost = host_to_par[str(groups_df.iloc[k]["host"])]
            igroup = group_param_index[str(groups_df.iloc[k]["group"])]
            coeffs[ieq, ihost] = 1.0
            coeffs[ieq, igroup] = -1.0
            yval[ieq] = 0.0
            yvar[ieq] = groups_df.iloc[k]["sigma"] ** 2
            covar[ieq, ieq] = yvar[ieq]
            eq_descr[ieq] = f"Distance modulus match between host {groups_df.iloc[k]['host']} and group {groups_df.iloc[k]['group']}"
            eq_shape[ieq] = "mu_host - mu_group = 0"
            eq_covar[ieq] = "None"
            eq_counter += 1


    # --- Calibrator Equations ---
    ieq_sn1a_start = None
    if nsn1a >0:
        ieq_sn1a_start = eq_counter
        for k in range(len(sn1a_calib_df)):
            ieq = ieq_sn1a_start + k
            ihost = host_index_sn1a[k]
            coeffs[ieq, ihost] = 1.0
            coeffs[ieq, iabs] = 1.0
            yval[ieq] = sn1a_calib_df.iloc[k]["m0_Bi"]
            covar[ieq, ieq] = sn1a_calib_df.iloc[k]["sigma"] ** 2
            eq_descr[ieq] = f"SNe Ia equation for {sn1a_calib_df.iloc[k]['SN']} in host {sn1a_calib_df.iloc[k]['host']}"
            eq_shape[ieq] = "M_B + mu_host = sn_mag"
            eq_covar[ieq] = "Variance of SNe Ia magnitude (including intrinsic dispersion)"
            eq_counter += 1

    ieq_sn2_start=None
    if nsn2 >0:
        ieq_sn2_start=eq_counter
        for k in range(len(sn2_calib_df)):
            ieq = ieq_sn2_start + k
            ihost = host_index_sn2[k]
            coeffs[ieq, ihost] = 1.0
            coeffs[ieq, iabs_sn2] = 1.0
            yval[ieq] = sn2_calib_df.iloc[k]["m0_Bi"]
            covar[ieq, ieq] = sn2_calib_df.iloc[k]["sigma"] ** 2
            eq_descr[ieq] = f"SNe II equation for {sn2_calib_df.iloc[k]['SN']} in host {sn2_calib_df.iloc[k]['host']}"
            eq_shape[ieq] = "M_B + mu_host = sn_mag"
            eq_covar[ieq] = "Variance of SNe II magnitude (including intrinsic dispersion)"
            eq_counter += 1

    ieq_tf_start=None
    if ntf >0:
        ieq_tf_start=eq_counter
        for k in range(len(tf_calib_df)):
            ieq = ieq_tf_start + k
            ihost = host_index_tf[k]
            coeffs[ieq, ihost] = 1.0
            coeffs[ieq, iabs_tf] = 1.0
            yval[ieq] = tf_calib_df.iloc[k]["m"]-tf_calib_df.iloc[k]["M"] #assume that this includes phot error
            covar[ieq, ieq] = tf_calib_df.iloc[k]["M_error"] ** 2
            eq_descr[ieq] = f"TF equation for {tf_calib_df.iloc[k]['PGC']} in host {tf_calib_df.iloc[k]['host']}"
            eq_shape[ieq] = "M_offset + mu_host = m - M"
            eq_covar[ieq] = "Variance of m and M (must include velocity and photometry errors)"
            eq_counter += 1

    ieq_sbf_start = None
    if nsbf>0:
        ieq_sbf_start=eq_counter
        for k in range(len(sbf_calib_df)):
            ieq = ieq_sbf_start + k
            ihost = host_index_sbf[k]
            coeffs[ieq, ihost] = 1.0
            coeffs[ieq, iabs_sbf] = 1.0
            yval[ieq] = sbf_calib_df.iloc[k]["m110"]-sbf_calib_df.iloc[k]["M110"] 
            covar[ieq, ieq] = sbf_calib_df.iloc[k]["m110_error"] ** 2 + sbf_calib_df.iloc[k]["M110_error"] ** 2 # color + intrinsic and phot error
            eq_descr[ieq] = f"SBF calibrator distance for {sbf_calib_df.iloc[k]['host']} based on distances from all sources"
            eq_shape[ieq] = "M_offset + mu_host = m110 - M110"
            eq_covar[ieq] = "Variance of m and M (must include propagated color, intrinsic, and photometry errors)"
            eq_counter += 1





    def compute_vcmb(z):
        z = np.asarray(z)
        return  (c_m_s/1000.0) * ((1 + z)**2 - 1) / ((1 + z)**2 + 1)

    ieq_h0_m1a=None
    if nsn1a>0:
        ieq_h0_m1a=eq_counter
        if sn1a_hf_df is not None:
            # Common velocities from redshifts
            vcmb = compute_vcmb(sn1a_hf_df["zcmb"])
            vhel = compute_vcmb(sn1a_hf_df["zhel"])

            if not sn1a_IR:
                # -------- Case A: Use vpec + covariance --------
                vpec = sn1a_hf_df[sn1a_vp_column].to_numpy()

                a_sn1a, a_sn1a_err, chisq_sn1a_hf, ndof_sn1a_hf = compute_alpha(
                    mag=sn1a_hf_df["mb"].to_numpy(),
                    covar=sn1a_hf_cov,          # use covariance
                    vpec=vpec,                  # peculiar-velocity column (selected earlier)
                    vcmb=vcmb,
                    vhel=vhel,
                    vdisp=vpec_error,
                    redshift_range=redshift_range,
                    apply_redshift_range=z_range_sn1a,
                    q0=q0,
                    j0=j0,
                    ignore_offdiag=sn1a_ignore_offdiag,
                )

            else:
                # -------- Case B: IR case, use vcorr + mag_err, no covariance --------
                if sn1a_vp_column == "vp_cmb":  # Note: builds from zcmb above
                    vcorr = vcmb
                else:
                    # IR mode only allows 2m++ or CMB; for 2m++, build vcorr from zcorr
                    vcorr = compute_vcmb(sn1a_hf_df["zcorr"])

                a_sn1a, a_sn1a_err, chisq_sn1a_hf, ndof_sn1a_hf = compute_alpha(
                    mag=sn1a_hf_df["mb"].to_numpy(),
                    mag_err=sn1a_hf_df["mb_err"].to_numpy(),
                    vhel=vhel,
                    vcmb=vcmb,
                    vcorr=vcorr,
                    vdisp=vpec_error,
                    redshift_range=redshift_range,
                    apply_redshift_range=z_range_sn1a,
                    q0=q0,
                    j0=j0,
                )

            r_chisq_sn1a_hf=chisq_sn1a_hf/ndof_sn1a_hf
            config_reader.vprint(f"\nAlpha from SNe Ia is computed: a_b={a_sn1a:.4f}"+u"\u00B1"+f"{a_sn1a_err:.4f}, chi2={chisq_sn1a_hf:2f}, dof={ndof_sn1a_hf}, reduced chi2={r_chisq_sn1a_hf:2f}")
        else:
            a_sn1a=alpha_sn1a
            a_sn1a_err=alpha_sn1a_error
        ieq = ieq_h0_m1a
        coeffs[ieq, iabs] = -0.2
        coeffs[ieq, ihub] = 1.0
        yval[ieq] = a_sn1a + 5
        covar[ieq, ieq] = a_sn1a_err ** 2
        eq_descr[ieq] = "Link between M_0 and H_0 (SNe Ia HF)"
        eq_shape[ieq] = "log10(H0) - 0.2*M_0 = a_b + 5"
        eq_covar[ieq] = "Variance in SN Ia Hubble flow (a_b)"
        eq_counter += 1

    ieq_h0_m2=None
    if nsn2>0:
        ieq_h0_m2=eq_counter
        if sn2_hf_df is not None:
            vcorr = sn2_hf_df["z_corr"]*(c_m_s/1000.0)

            a_sn2, a_sn2_err, chisq_sn2_hf, ndof_sn2_hf = compute_alpha(
                mag=sn2_hf_df["m0_i"],
                mag_err=sn2_hf_df["em0_i"],
                vcorr=vcorr,
                vdisp=vpec_error,
                redshift_range=redshift_range,
                apply_redshift_range=z_range_sn2,
                q0=q0,
                j0=j0
            )

            r_chisq_sn2_hf=chisq_sn2_hf/ndof_sn2_hf
            config_reader.vprint(f"\nAlpha from SNe II is computed: a_b= {a_sn2:.4f}"+u"\u00B1"+f"{a_sn2_err:.4f}, chi2={chisq_sn2_hf:2f}, dof={ndof_sn2_hf}, reduced chi2={r_chisq_sn2_hf:2f}")
        else:
            a_sn2=alpha_sn2
            a_sn2_err= alpha_sn2_error
        ieq = ieq_h0_m2
        coeffs[ieq, iabs_sn2] = -0.2
        coeffs[ieq, ihub] = 1.0
        yval[ieq] = a_sn2 + 5
        covar[ieq, ieq] = a_sn2_err ** 2
        eq_descr[ieq] = "Link between M_0 and H_0 (SNe II HF)"
        eq_shape[ieq] = "log10(H0) - 0.2*M_0 = a_b + 5"
        eq_covar[ieq] = "Variance in SN II Hubble flow (a_b)"
        eq_counter += 1

    ieq_h0_mtf=None
    if ntf>0:
        ieq_h0_mtf=eq_counter
        if tf_hf_df is not None:
            value = tf_hf_df["m"] - tf_hf_df["M"]
            vcorr = tf_hf_df[tf_v_column]

            a_tf, a_tf_err, chisq_tf_hf, ndof_tf_hf = compute_alpha(
                value,
                mag_err=tf_hf_df["sigma"],
                vcorr=vcorr,
                vdisp=vpec_error,
                redshift_range=redshift_range,
                apply_redshift_range=z_range_tf,
                q0=q0,
                j0=j0
            )
            r_chisq_tf_hf=chisq_tf_hf/ndof_tf_hf
            config_reader.vprint(f"\nAlpha from TF is computed: a_b={a_tf:.4f}"+u"\u00B1"+f"{a_tf_err:.4f}, chi2={chisq_tf_hf:2f}, dof={ndof_tf_hf}, reduced chi2={r_chisq_tf_hf:2f}")
        else:
            a_tf=alpha_tf
            a_tf_err= alpha_tf_error       
        ieq = ieq_h0_mtf
        coeffs[ieq, iabs_tf] = -0.2
        coeffs[ieq, ihub] = 1.0
        yval[ieq] = a_tf + 5
        covar[ieq, ieq] = a_tf_err ** 2
        eq_descr[ieq] = "Link between M_offset_TF and H_0 (TF HF)"
        eq_shape[ieq] = "log10(H0) - 0.2*M_offset_TF = a_b + 5"
        eq_covar[ieq] = "Variance in TF Hubble flow (a_b)"
        eq_counter += 1

    ieq_h0_msbf = None
    if nsbf>0:
        ieq_h0_msbf=eq_counter
        if sbf_hf_df is not None:
            value = sbf_hf_df["m110"] - sbf_hf_df["M110"]
            vcorr = sbf_hf_df[sbf_v_column]

            a_sbf, a_sbf_err, chisq_sbf_hf, ndof_sbf_hf = compute_alpha(
                value,
                mag_err=sbf_hf_df["e(m110)"],
                vcorr=vcorr,
                vdisp=sbf_vpec_error,
                redshift_range=redshift_range,
                apply_redshift_range=z_range_sbf,
                q0=q0,
                j0=j0,
                optical=True,
                ignore_cosmology=True
            )
            r_chisq_sbf_hf=chisq_sbf_hf/ndof_sbf_hf
            config_reader.vprint(f"\nAlpha from SBF is computed: a_b={a_sbf:.4f}"+u"\u00B1"+f"{a_sbf_err:.4f}, chi2={chisq_sbf_hf:2f}, dof={ndof_sbf_hf}, reduced chi2={r_chisq_sbf_hf:2f}")
        else:
            a_sbf=alpha_sbf
            a_sbf_err= alpha_sbf_error       
        ieq = ieq_h0_msbf
        coeffs[ieq, iabs_sbf] = -0.2
        coeffs[ieq, ihub] = 1.0
        yval[ieq] = a_sbf + 5
        covar[ieq, ieq] = a_sbf_err ** 2
        eq_descr[ieq] = "Link between M_offset_SBF and H_0 (SBF HF)"
        eq_shape[ieq] = "log10(H0) - 0.2*M_offset_SBF = a_b + 5"
        eq_covar[ieq] = "Variance in SBF Hubble flow (a_b)"
        eq_counter += 1



    # --- mm Constraints (if available) ---
    ieq_mm_start = None
    if nmm > 0:
        ieq_mm_start=eq_counter
        for k in range(nmm):
            ieq = eq_counter
            coeffs[ieq, ihub] = 1.0
            yval[ieq] = mm_df.iloc[k]["logh_value"]
            covar[ieq, ieq] = mm_df.iloc[k]["logh_error"] ** 2
            eq_descr[ieq] = f"mm constraint from {mm_df.iloc[k]['name']}"
            eq_shape[ieq] = "log10(H0) = measured value"
            eq_covar[ieq] = "mm measurement variance"
            eq_counter += 1


    # --- EPM Constraints (if available) ---
    ieq_epm_start = None
    if nepm>0:
        ieq_epm_start = eq_counter
        for k in range(len(epm_df)):
            ieq = eq_counter
            coeffs[ieq, ihub] = 1.0
            yval[ieq] = epm_df.iloc[k]["logh_value"]
            covar[ieq, ieq] = epm_df.iloc[k]["logh_error"] ** 2
            eq_descr[ieq] = f"EPM constraint from SN II {epm_df.iloc[k]['SN']}"
            eq_shape[ieq] = "log10(H0) = measured value"
            eq_covar[ieq] = "EPM measurement variance"
            eq_counter += 1





    # --- Coma Constraints (if available) ---
    #TODO: Add hardcoded Coma values to be potential inputs from the config file
    icoma = None  # default value
    ieq_coma_start= None
    if ncoma > 0:
        ieq_coma_start=eq_counter
        # At this point, the mu_coma parameter is already included and its index is:
        icoma = next(i for i, p in enumerate(parinfo) if p["objname"] == "mu_coma")
        # Coma SN constraints
        for k in range(ncoma):
            ieq = eq_counter
            coeffs[ieq, icoma] = 1.0
            coeffs[ieq, iabs] = 1.0
            yval[ieq] = coma_df.iloc[k]["coma_sn_mag"]
            covar[ieq, ieq] = coma_df.iloc[k]["coma_sn_err"] ** 2
            eq_descr[ieq] = f"Coma SN constraint for {coma_df.iloc[k]['coma_sn_name']}"
            eq_shape[ieq] = "mu_coma + M_B = SN mag (Coma)"
            eq_covar[ieq] = "Coma SN variance"
            eq_counter += 1
        # Additional Coma constraints
        # SBF constraint for Coma
        ieq = eq_counter
        coeffs[ieq, icoma] = 1.0
        d_coma_nom = 99.1
        mu_coma_nom = 5 * np.log10(d_coma_nom) + 25
        yval[ieq] = mu_coma_nom
        error_mu_coma_nom = 0.1345
        covar[ieq, ieq] = error_mu_coma_nom ** 2
        eq_descr[ieq] = "SBF distance to Coma"
        eq_shape[ieq] = "mu_coma = SBF value"
        eq_covar[ieq] = "SBF variance (Coma)"
        eq_counter += 1
        # H0 equation from Coma
        ieq = eq_counter
        h0_nom = 76.05
        logh_nom = np.log10(h0_nom)
        coeffs[ieq, ihub] = 1.0
        coeffs[ieq, icoma] = 0.2
        yval[ieq] = logh_nom + 0.2 * mu_coma_nom
        covar[ieq, ieq] = (sigma_mag_fp/5.) ** 2
        eq_descr[ieq] = "H0 from Coma and FP"
        eq_shape[ieq] = "log10(H0) + 0.2*mu_coma = expected FP value"
        eq_covar[ieq] = "FP variance (Coma)"
        eq_counter += 1


    return {
        # =====================================================================
        # CORE EQUATION SYSTEM
        # =====================================================================
        "coeffs": coeffs,
        "covar": covar,
        "yval": yval,

        # =====================================================================
        # EQUATION METADATA
        # =====================================================================
        "eq_descr": eq_descr,
        "eq_shape": eq_shape,
        "eq_covar": eq_covar,
        "parinfo": parinfo,

        # =====================================================================
        # DIMENSIONS
        # =====================================================================
        "npars": npars,
        "neq": neq,
        "nhosts": nhosts,

        # =====================================================================
        # HOST INDEXING
        # =====================================================================
        "host_index": host_index,
        "all_hosts": all_hosts,
        "host_index_2": host_index_2,
        "host_index_sn1a": host_index_sn1a,
        "host_index_sn2": host_index_sn2,
        "host_index_tf": host_index_tf,
        "host_index_sbf": host_index_sbf,

        # =====================================================================
        # PARAMETER INDICES
        # =====================================================================
        "iabs": iabs,
        "iabs_sn2": iabs_sn2,
        "iabs_tf": iabs_tf,
        "iabs_sbf": iabs_sbf,
        "ihub": ihub,
        "icoma": icoma,

        # =====================================================================
        # CONSTRAINT COUNTS
        # =====================================================================
        "nsn1a": nsn1a,
        "nsn2": nsn2,
        "ntf": ntf,
        "nsbf": nsbf,
        "ncoma": ncoma,
        "nmm": nmm,
        "nepm": nepm,
        "ngroups_eq": ngroups_eq,

        # =====================================================================
        # ERROR ARRAYS FOR COVARIANCE
        # =====================================================================
        "mas_host_error": mas_host_error,
        "anchor_host_error": anchor_host_error,

        # =====================================================================
        # EQUATION START INDICES
        # =====================================================================
        "ieq_sn1a_start": ieq_sn1a_start,
        "ieq_sn2_start": ieq_sn2_start,
        "ieq_tf_start": ieq_tf_start,
        "ieq_sbf_start": ieq_sbf_start,
        "ieq_coma_start": ieq_coma_start,
        "ieq_mm_start": ieq_mm_start,
        "ieq_epm_start": ieq_epm_start,
        "ieq_h0_m1a": ieq_h0_m1a,
        "ieq_h0_m2": ieq_h0_m2,
        "ieq_h0_mtf": ieq_h0_mtf,
        "ieq_h0_msbf": ieq_h0_msbf,

        # =====================================================================
        # ALPHA RESULTS FROM HUBBLE FLOW FITTING
        # =====================================================================
        # SNe Ia
        "a_sn1a": a_sn1a,
        "a_sn1a_err": a_sn1a_err,
        "chisq_sn1a_hf": chisq_sn1a_hf,
        "ndof_sn1a_hf": ndof_sn1a_hf,
        # SNe II
        "a_sn2": a_sn2,
        "a_sn2_err": a_sn2_err,
        "chisq_sn2_hf": chisq_sn2_hf,
        "ndof_sn2_hf": ndof_sn2_hf,
        # Tully-Fisher
        "a_tf": a_tf,
        "a_tf_err": a_tf_err,
        "chisq_tf_hf": chisq_tf_hf,
        "ndof_tf_hf": ndof_tf_hf,
        # SBF
        "a_sbf": a_sbf,
        "a_sbf_err": a_sbf_err,
        "chisq_sbf_hf": chisq_sbf_hf,
        "ndof_sbf_hf": ndof_sbf_hf,

        # =====================================================================
        # GROUP PARAMETER MAPPING
        # =====================================================================
        "group_param_index": group_param_index,
    }

