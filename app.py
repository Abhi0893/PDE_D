#!/usr/bin/env python3
"""
Streamlit app for Diffusion Coefficient (D) analysis from 1D oxygen
cross-section data.

Upload a CSV or Excel file and get:
  - Method 1: Variance (moment) analysis  →  D_moment
  - Method 2: Numerical PDE inversion (Crank-Nicolson)  →  D_pde
  - Retention metrics (τ_half_peak, FWHM, total excess O₂)
  - Six diagnostic plots
"""

import io
import csv
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


# np.trapz was removed in NumPy 2.0 in favor of np.trapezoid.
_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data_from_bytes(raw_bytes: bytes, filename: str):
    """Parse uploaded file → positions (mm), times (s), C(x,t) matrix."""

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw_bytes), header=None)
        rows = df.astype(str).values.tolist()
    else:
        text = raw_bytes.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))

    # Row 0: timestamps (HH:MM:SS) starting at column 2
    time_strs = rows[0][2:]
    times = []
    for ts in time_strs:
        parts = ts.strip().split(":")
        times.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    times = np.array(times, dtype=float)

    # Row 1: header row (skip)
    # Rows 2+: pixel_index, position_mm, O2 values …
    positions, profiles = [], []
    for row in rows[2:]:
        if len(row) < 3:
            continue
        positions.append(float(row[1]))
        profiles.append([float(v) for v in row[2:]])

    return np.array(positions), times, np.array(profiles)


# ═══════════════════════════════════════════════════════════════════════
# AUTO-DETECT LOADING → DIFFUSION TRANSITION
# ═══════════════════════════════════════════════════════════════════════

def detect_diffusion_start(data, n_center=40):
    """Find the timestep where peak concentration stops increasing."""
    mid = data.shape[0] // 2
    half_w = n_center // 2
    center_mean = data[mid - half_w:mid + half_w, :].mean(axis=0)
    t_start = int(np.argmax(center_mean))
    return max(t_start, 0)


# ═══════════════════════════════════════════════════════════════════════
# DATASET AVERAGING
# ═══════════════════════════════════════════════════════════════════════

def compute_average_data(datasets):
    """Average multiple (x, times, data) tuples into a single dataset.

    If spatial grids differ in length, all datasets are linearly
    interpolated onto the finest common grid.  If time grids differ,
    the shortest common time span is used.

    Returns
    -------
    x_common, times_common, data_mean, data_std, n_datasets
    """
    from scipy.interpolate import interp1d

    if len(datasets) == 1:
        x, t, d = datasets[0]
        return x, t, d, np.zeros_like(d), 1

    # --- Determine common spatial grid (finest resolution) ---
    all_x = [ds[0] for ds in datasets]
    dx_min = min(np.median(np.diff(xx)) for xx in all_x)
    x_lo = max(xx[0] for xx in all_x)
    x_hi = min(xx[-1] for xx in all_x)
    n_pts = max(10, int(np.round((x_hi - x_lo) / dx_min)) + 1)
    x_common = np.linspace(x_lo, x_hi, n_pts)

    # --- Determine common time grid (shortest span) ---
    all_t = [ds[1] for ds in datasets]
    t_max = min(tt[-1] for tt in all_t)
    # Use the time vector from the dataset with most steps up to t_max
    t_common = None
    for tt in all_t:
        tt_trim = tt[tt <= t_max + 0.1]
        if t_common is None or len(tt_trim) > len(t_common):
            t_common = tt_trim

    # --- Interpolate each dataset onto common grid ---
    stack = []
    for x_i, t_i, d_i in datasets:
        # Temporal: pick columns whose time <= t_max
        t_mask = t_i <= t_max + 0.1
        d_trim = d_i[:, t_mask]
        t_trim = t_i[t_mask]

        # If time grids differ in length, interpolate temporally
        if len(t_trim) != len(t_common) or not np.allclose(t_trim, t_common, atol=1):
            d_temp = np.zeros((d_trim.shape[0], len(t_common)))
            for px in range(d_trim.shape[0]):
                f = interp1d(t_trim, d_trim[px, :], kind="linear",
                             bounds_error=False, fill_value="extrapolate")
                d_temp[px, :] = f(t_common)
            d_trim = d_temp

        # Spatial interpolation onto x_common
        d_interp = np.zeros((len(x_common), d_trim.shape[1]))
        for col in range(d_trim.shape[1]):
            f = interp1d(x_i, d_trim[:, col], kind="linear",
                         bounds_error=False, fill_value="extrapolate")
            d_interp[:, col] = f(x_common)
        stack.append(d_interp)

    cube = np.array(stack)  # shape (n_datasets, n_x, n_t)
    data_mean = cube.mean(axis=0)
    data_std = cube.std(axis=0, ddof=1) if cube.shape[0] > 1 else np.zeros_like(data_mean)

    return x_common, t_common, data_mean, data_std, len(datasets)


# ═══════════════════════════════════════════════════════════════════════
# METHOD 1: MOMENT (VARIANCE) ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def method_moments(x, data, times, n_edge=15, n_fit_steps=12):
    """Compute D from the slope of sigma^2(t)."""
    bl = 0.5 * (data[:n_edge, 0].mean() + data[-n_edge:, 0].mean())
    excess = data - bl
    dx = np.median(np.diff(x))

    ex_pos = np.maximum(excess, 0)
    M0 = np.sum(ex_pos, axis=0) * dx
    xb = np.sum(ex_pos * x[:, None], axis=0) * dx / M0
    s2 = np.sum(ex_pos * (x[:, None] - xb[None, :]) ** 2, axis=0) * dx / M0

    n = min(n_fit_steps, len(times))
    coeffs = np.polyfit(times[:n], s2[:n], 1)
    D = coeffs[0] / 2.0

    s2_pred = np.polyval(coeffs, times[:n])
    ss_res = np.sum((s2[:n] - s2_pred) ** 2)
    ss_tot = np.sum((s2[:n] - s2[:n].mean()) ** 2)
    R2 = 1 - ss_res / (ss_tot if ss_tot > 0 else 1)

    D_inst_t, D_inst_v = [], []
    step = 3
    for i in range(0, len(times) - step, step):
        ds2 = s2[i + step] - s2[i]
        dt = times[i + step] - times[i]
        if dt > 0:
            D_inst_t.append(0.5 * (times[i] + times[i + step]))
            D_inst_v.append(ds2 / (2 * dt))

    return {
        "D": D, "R2": R2, "slope": coeffs[0], "sigma2_0": coeffs[1],
        "sigma2": s2, "M0": M0, "baseline": bl, "excess": excess,
        "n_fit": n,
        "D_inst_t": np.array(D_inst_t), "D_inst_v": np.array(D_inst_v),
    }


# ═══════════════════════════════════════════════════════════════════════
# METHOD 2: PDE INVERSION (CRANK-NICOLSON)
# ═══════════════════════════════════════════════════════════════════════

def solve_CN(C0, x, times, D, bc_left, bc_right):
    """Crank-Nicolson solver for 1D diffusion with Dirichlet BCs."""
    N = len(x)
    dx = np.median(np.diff(x))
    C_all = np.zeros((N, len(times)))
    C_all[:, 0] = C0.copy()

    for n in range(len(times) - 1):
        dt = times[n + 1] - times[n]
        r = D * dt / (2.0 * dx ** 2)
        Ni = N - 2
        C_old = C_all[:, n]

        rhs = np.zeros(Ni)
        for i in range(Ni):
            j = i + 1
            rhs[i] = r * C_old[j - 1] + (1 - 2 * r) * C_old[j] + r * C_old[j + 1]
        rhs[0] += r * bc_left[n + 1]
        rhs[-1] += r * bc_right[n + 1]

        a_val, b_val, c_val = -r, 1 + 2 * r, -r
        cp = np.zeros(Ni)
        dp = np.zeros(Ni)
        cp[0] = c_val / b_val
        dp[0] = rhs[0] / b_val
        for i in range(1, Ni):
            denom = b_val - a_val * cp[i - 1]
            cp[i] = c_val / denom if i < Ni - 1 else 0
            dp[i] = (rhs[i] - a_val * dp[i - 1]) / denom

        C_new = np.zeros(Ni)
        C_new[-1] = dp[-1]
        for i in range(Ni - 2, -1, -1):
            C_new[i] = dp[i] - cp[i] * C_new[i + 1]

        C_all[0, n + 1] = bc_left[n + 1]
        C_all[1:-1, n + 1] = C_new
        C_all[-1, n + 1] = bc_right[n + 1]

    return C_all


def method_pde(x, data, times, D_bounds=(1e-6, 5e-2), n_bc=10):
    """Optimise D by minimising PDE residual."""
    bc_l = data[:n_bc, :].mean(axis=0)
    bc_r = data[-n_bc:, :].mean(axis=0)
    margin = n_bc + 5

    def objective(D):
        C_sim = solve_CN(data[:, 0], x, times, D, bc_l, bc_r)
        return np.sum((C_sim[margin:-margin, 1:] - data[margin:-margin, 1:]) ** 2)

    result = minimize_scalar(objective, bounds=D_bounds, method="bounded",
                             options={"xatol": 1e-10, "maxiter": 300})
    D_opt = result.x

    C_sim = solve_CN(data[:, 0], x, times, D_opt, bc_l, bc_r)
    n_pts = (data.shape[0] - 2 * margin) * (data.shape[1] - 1)
    rmse = np.sqrt(result.fun / n_pts)
    per_t_rmse = np.sqrt(np.mean(
        (C_sim[margin:-margin, :] - data[margin:-margin, :]) ** 2, axis=0
    ))

    return {"D": D_opt, "RMSE": rmse, "C_sim": C_sim, "per_t_rmse": per_t_rmse}


# ═══════════════════════════════════════════════════════════════════════
# RETENTION METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_retention(x, data, times, n_edge=15):
    bl = 0.5 * (data[:n_edge, 0].mean() + data[-n_edge:, 0].mean())
    excess = data - bl
    dx = np.median(np.diff(x))

    peak_excess = excess.max(axis=0)
    total_excess = np.sum(np.maximum(excess, 0), axis=0) * dx

    fwhm = np.full(len(times), np.nan)
    for t in range(len(times)):
        prof = excess[:, t]
        hm = prof.max() / 2
        above = np.where(prof > hm)[0]
        if len(above) >= 2:
            fwhm[t] = x[above[-1]] - x[above[0]]

    target = peak_excess[0] * 0.5
    tau_half = np.nan
    for i in range(1, len(times)):
        if peak_excess[i] <= target:
            frac = (target - peak_excess[i - 1]) / (peak_excess[i] - peak_excess[i - 1])
            tau_half = times[i - 1] + frac * (times[i] - times[i - 1])
            break

    return {
        "peak_excess": peak_excess, "total_excess": total_excess,
        "fwhm": fwhm, "tau_half_peak": tau_half, "baseline": bl,
    }


# ═══════════════════════════════════════════════════════════════════════
# RELEASE KINETICS METRICS
#   (AUC above therapeutic threshold, penetration depth L_p(t),
#    source depletion M_source(t), release rate -dM/dt)
# ═══════════════════════════════════════════════════════════════════════

def compute_kinetics_metrics(
    x, data, times,
    n_edge=15,
    c_threshold_excess=50.0,
    pen_threshold_frac=0.10,
    pen_threshold_abs=5.0,
):
    """Compute oxygen release kinetics metrics from C(x,t).

    Parameters
    ----------
    c_threshold_excess : float
        Therapeutic threshold, expressed as excess (% Air Sat.) ABOVE
        the GelMA baseline. AUC is integrated over any signal exceeding
        this level.
    pen_threshold_frac : float
        Penetration is measured where excess > frac * initial_peak_excess.
    pen_threshold_abs : float
        Absolute floor (% Air Sat.) for the penetration threshold, to
        avoid being fooled by late-time noise.

    Returns
    -------
    dict with:
        auc_xt        : scalar, (% Air Sat.)*mm*s   -- Metric 3
        auc_spatial_t : array[T], (% Air Sat.)*mm   -- spatial integral at each t
        L_p           : array[T], mm                 -- Metric 4 (width above thr)
        M_source      : array[T], (% Air Sat.)*mm   -- source region mass
        release_rate  : array[T], (% Air Sat.)*mm/s -- -dM/dt (Metric 5)
        src_lo, src_hi: ints, source region bounds (pixel indices)
        pen_threshold : float, actual absolute threshold used for L_p
        c_threshold_excess : float, AUC threshold echoed back
        baseline      : float, baseline C used
    """
    bl = 0.5 * (data[:n_edge, 0].mean() + data[-n_edge:, 0].mean())
    excess = data - bl
    dx = np.median(np.diff(x))
    N, T = data.shape

    # --- Metric 3: AUC above therapeutic threshold ---
    # Threshold is interpreted as "excess above baseline" so it is
    # independent of the absolute baseline value (which depends on the
    # sensor calibration).
    above_th = np.maximum(excess - c_threshold_excess, 0.0)
    auc_spatial_t = np.sum(above_th, axis=0) * dx     # (% Air Sat)*mm
    if len(times) > 1:
        auc_xt = float(_trapz(auc_spatial_t, times))  # *s
    else:
        auc_xt = 0.0

    # --- Metric 4: Penetration depth L_p(t) ---
    peak0 = float(excess[:, 0].max())
    pen_th = max(pen_threshold_frac * peak0, pen_threshold_abs)
    L_p = np.zeros(T)
    for t in range(T):
        above = np.where(excess[:, t] > pen_th)[0]
        if len(above) >= 2:
            L_p[t] = x[above[-1]] - x[above[0]]

    # --- Metric 5: Source mass & release rate ---
    # Define the "source region" as the FWHM of the initial profile.
    prof0 = excess[:, 0]
    hm0 = prof0.max() / 2.0
    above0 = np.where(prof0 > hm0)[0]
    if len(above0) >= 2:
        src_lo, src_hi = int(above0[0]), int(above0[-1])
    else:
        peak_idx = int(np.argmax(prof0))
        half_w = max(1, N // 20)
        src_lo = max(0, peak_idx - half_w)
        src_hi = min(N - 1, peak_idx + half_w)

    M_source = np.sum(np.maximum(excess[src_lo:src_hi + 1, :], 0.0), axis=0) * dx

    # Release rate = -dM/dt (central diff on interior, forward/backward on ends)
    release_rate = np.zeros(T)
    if T >= 2:
        for t in range(T):
            if t == 0:
                dt_ = times[1] - times[0]
                if dt_ > 0:
                    release_rate[t] = -(M_source[1] - M_source[0]) / dt_
            elif t == T - 1:
                dt_ = times[-1] - times[-2]
                if dt_ > 0:
                    release_rate[t] = -(M_source[-1] - M_source[-2]) / dt_
            else:
                dt_ = times[t + 1] - times[t - 1]
                if dt_ > 0:
                    release_rate[t] = -(M_source[t + 1] - M_source[t - 1]) / dt_

    return {
        "auc_xt": auc_xt,
        "auc_spatial_t": auc_spatial_t,
        "L_p": L_p,
        "M_source": M_source,
        "release_rate": release_rate,
        "src_lo": src_lo,
        "src_hi": src_hi,
        "pen_threshold": pen_th,
        "c_threshold_excess": c_threshold_excess,
        "baseline": bl,
    }


# ═══════════════════════════════════════════════════════════════════════
# RELEASE KINETICS MODEL FITS
#   (First-order, Korsmeyer-Peppas, Higuchi)
# ═══════════════════════════════════════════════════════════════════════

def _safe_r2(y, y_pred):
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_release_models(times, peak_excess):
    """Fit first-order, Korsmeyer-Peppas, and Higuchi release models to
    the peak-excess decay curve.

    peak_excess is used as a proxy for remaining "undelivered" oxygen in
    the source. Fractional release is M(t)/M_inf = 1 - C_ex(t)/C_ex(0).

    Returns dict with fitted parameters and R^2 for each model. Any model
    that fails to fit is reported with NaNs.
    """
    t = np.asarray(times, dtype=float)
    pe = np.asarray(peak_excess, dtype=float)

    nan_fo = {"C0": np.nan, "k": np.nan, "R2": np.nan, "tau_half": np.nan}
    nan_kp = {"k": np.nan, "n": np.nan, "R2": np.nan, "n_points": 0}
    nan_hi = {"k_H": np.nan, "R2": np.nan}

    if len(t) < 3 or pe[0] <= 0:
        return {"first_order": nan_fo, "korsmeyer_peppas": nan_kp, "higuchi": nan_hi}

    results = {}

    # ---- First-order: C_ex(t) = C0 * exp(-k * t) ----
    try:
        popt, _ = curve_fit(
            lambda tt, C0, k: C0 * np.exp(-k * tt),
            t, pe,
            p0=[pe[0], 1e-3],
            bounds=([0, 0], [10 * pe[0] + 1, 1.0]),
            maxfev=5000,
        )
        C0_fo, k_fo = float(popt[0]), float(popt[1])
        pred = C0_fo * np.exp(-k_fo * t)
        R2_fo = _safe_r2(pe, pred)
        tau_fo = float(np.log(2) / k_fo) if k_fo > 0 else np.nan
        results["first_order"] = {
            "C0": C0_fo, "k": k_fo, "R2": R2_fo, "tau_half": tau_fo,
        }
    except Exception:
        results["first_order"] = dict(nan_fo)

    # ---- Fractional release ----
    frac_rel = 1.0 - pe / pe[0]
    frac_rel = np.clip(frac_rel, 0.0, 1.0)

    # ---- Korsmeyer-Peppas: M/M_inf = k * t^n, valid for M/M_inf < 0.6 ----
    mask_kp = (t > 0) & (frac_rel > 1e-4) & (frac_rel < 0.6)
    if mask_kp.sum() >= 3:
        try:
            popt, _ = curve_fit(
                lambda tt, k, n: k * np.power(tt, n),
                t[mask_kp], frac_rel[mask_kp],
                p0=[1e-3, 0.5],
                bounds=([0.0, 0.05], [10.0, 2.0]),
                maxfev=5000,
            )
            k_kp, n_kp = float(popt[0]), float(popt[1])
            pred = k_kp * np.power(t[mask_kp], n_kp)
            R2_kp = _safe_r2(frac_rel[mask_kp], pred)
            results["korsmeyer_peppas"] = {
                "k": k_kp, "n": n_kp, "R2": R2_kp, "n_points": int(mask_kp.sum()),
            }
        except Exception:
            results["korsmeyer_peppas"] = dict(nan_kp)
    else:
        results["korsmeyer_peppas"] = dict(nan_kp)

    # ---- Higuchi: M/M_inf = k_H * sqrt(t) ----
    mask_h = t > 0
    if mask_h.sum() >= 3:
        try:
            popt, _ = curve_fit(
                lambda tt, k_H: k_H * np.sqrt(tt),
                t[mask_h], frac_rel[mask_h],
                p0=[1e-3],
                bounds=([0.0], [10.0]),
                maxfev=5000,
            )
            k_H = float(popt[0])
            pred = k_H * np.sqrt(t[mask_h])
            R2_h = _safe_r2(frac_rel[mask_h], pred)
            results["higuchi"] = {"k_H": k_H, "R2": R2_h}
        except Exception:
            results["higuchi"] = dict(nan_hi)
    else:
        results["higuchi"] = dict(nan_hi)

    return results


def classify_kp_mechanism(n):
    """Return a short mechanistic label for the Korsmeyer-Peppas exponent."""
    if n is None or np.isnan(n):
        return "N/A"
    if n < 0.45:
        return "Sub-Fickian (n<0.45)"
    if n < 0.55:
        return "Fickian diffusion (n\u22480.5)"
    if n < 0.95:
        return "Anomalous transport (0.5<n<1)"
    if n < 1.05:
        return "Zero-order / Case II (n\u22481)"
    return "Super Case II (n>1)"


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def make_plots(x, times, data, mom, pde, ret, t_start_min):
    """Return a matplotlib Figure with 6 diagnostic subplots."""
    t_min = times / 60

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Diffusion Analysis  |  D_moment = {mom['D']:.3e} mm\u00b2/s  |  "
        f"D_pde = {pde['D']:.3e} mm\u00b2/s  |  "
        f"\u03c4_half = {ret['tau_half_peak'] / 60:.1f} min  |  "
        f"(loading phase removed: first {t_start_min} min)",
        fontsize=11, fontweight="bold",
    )

    # 1 - Raw profiles
    ax = axes[0, 0]
    idx = np.linspace(0, len(times) - 1, 8, dtype=int)
    for i in idx:
        ax.plot(x, data[:, i], label=f"{t_min[i]:.0f} min", lw=1, alpha=0.85)
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("O\u2082 (% Air Sat.)")
    ax.set_title("Profiles (after loading)")
    ax.legend(fontsize=7, ncol=2)

    # 2 - sigma^2 vs t
    ax = axes[0, 1]
    ax.scatter(t_min, mom["sigma2"], s=10, c="steelblue", alpha=0.5, label="all")
    n = mom["n_fit"]
    ax.scatter(t_min[:n], mom["sigma2"][:n], s=20, c="red", zorder=3, label="fit range")
    t_fit = np.linspace(0, times[n - 1], 100)
    ax.plot(t_fit / 60, mom["slope"] * t_fit + mom["sigma2_0"], "r--", lw=1.5,
            label=f"D = slope/2 = {mom['D']:.3e}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("\u03c3\u00b2 (mm\u00b2)")
    ax.set_title(f"Moment Method (R\u00b2 = {mom['R2']:.3f})")
    ax.legend(fontsize=8)

    # 3 - Instantaneous D
    ax = axes[0, 2]
    mask = np.array(mom["D_inst_v"]) > 0
    if mask.any():
        ax.semilogy(np.array(mom["D_inst_t"])[mask] / 60,
                     np.array(mom["D_inst_v"])[mask],
                     "o-", ms=4, lw=1, c="teal")
    ax.axhline(mom["D"], color="red", ls="--", lw=1, label=f"D_moment = {mom['D']:.3e}")
    ax.axhline(pde["D"], color="orange", ls="--", lw=1, label=f"D_pde = {pde['D']:.3e}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("D_instantaneous (mm\u00b2/s)")
    ax.set_title("Instantaneous D (\u0394\u03c3\u00b2/2\u0394t)")
    ax.legend(fontsize=8)

    # 4 - PDE fit: data vs sim
    ax = axes[1, 0]
    for i in [0, 5, 12, 30, len(times) - 1]:
        if i < len(times):
            ax.plot(x, data[:, i], "-", lw=1.2, alpha=0.8, label=f"data {t_min[i]:.0f}m")
            ax.plot(x, pde["C_sim"][:, i], "--", lw=1, alpha=0.6)
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("O\u2082 (% Air Sat.)")
    ax.set_title(f"PDE Fit (D = {pde['D']:.3e}, RMSE = {pde['RMSE']:.1f})")
    ax.legend(fontsize=7, ncol=2)

    # 5 - Peak excess decay
    ax = axes[1, 1]
    ax.plot(t_min, ret["peak_excess"], "o-", ms=2, lw=1, c="crimson")
    if not np.isnan(ret["tau_half_peak"]):
        ax.axvline(ret["tau_half_peak"] / 60, color="gray", ls="--", lw=1,
                   label=f"\u03c4_half = {ret['tau_half_peak'] / 60:.1f} min")
        ax.axhline(ret["peak_excess"][0] / 2, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Peak Excess O\u2082")
    ax.set_title("Peak Decay \u2192 \u03c4_half_peak")
    ax.legend(fontsize=9)

    # 6 - FWHM
    ax = axes[1, 2]
    valid_fwhm = ~np.isnan(ret["fwhm"])
    ax.plot(t_min[valid_fwhm], ret["fwhm"][valid_fwhm], "s-", ms=3, lw=1, c="teal")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("FWHM (mm)")
    ax.set_title("Profile Width (FWHM of excess)")

    plt.tight_layout()
    return fig


def make_kinetics_plots(x, times, data, ret, kin, fits):
    """Figure with release kinetics diagnostics:
      1 - Peak excess decay with model fits
      2 - Fractional release + Korsmeyer-Peppas fit (log-log)
      3 - Source mass M_source(t) and release rate
      4 - Penetration depth L_p(t)
      5 - AUC spatial integral vs time (therapeutic coverage)
      6 - Source region overlay on initial profile
    """
    t_min = times / 60.0
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    kp_n = fits["korsmeyer_peppas"]["n"]
    kp_label = classify_kp_mechanism(kp_n)
    fig.suptitle(
        f"Release Kinetics  |  AUC = {kin['auc_xt']:.1f} %\u00b7mm\u00b7s  |  "
        f"first-order \u03c4\u00bd = "
        f"{(fits['first_order']['tau_half'] / 60 if not np.isnan(fits['first_order']['tau_half']) else float('nan')):.1f} min  |  "
        f"K-P n = {kp_n:.3f} ({kp_label})",
        fontsize=11, fontweight="bold",
    )

    pe = ret["peak_excess"]

    # 1 - Peak excess decay with first-order fit
    ax = axes[0, 0]
    ax.plot(t_min, pe, "o", ms=3, c="crimson", label="data")
    fo = fits["first_order"]
    if not np.isnan(fo["k"]):
        t_dense = np.linspace(times[0], times[-1], 200)
        ax.plot(t_dense / 60, fo["C0"] * np.exp(-fo["k"] * t_dense),
                "-", c="navy", lw=1.5,
                label=f"first-order (R\u00b2={fo['R2']:.3f})")
    if not np.isnan(ret["tau_half_peak"]):
        ax.axvline(ret["tau_half_peak"] / 60, color="gray", ls="--", lw=1,
                   label=f"\u03c4\u00bd (empirical) = {ret['tau_half_peak']/60:.1f} min")
        ax.axhline(pe[0] / 2, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Peak Excess O\u2082 (% Air Sat.)")
    ax.set_title("Metric 1/2: C_peak(t) & \u03c4\u00bd")
    ax.legend(fontsize=8)

    # 2 - Fractional release + Korsmeyer-Peppas + Higuchi (log-log)
    ax = axes[0, 1]
    frac_rel = np.clip(1.0 - pe / pe[0], 0.0, 1.0)
    mask = times > 0
    ax.loglog(times[mask], np.clip(frac_rel[mask], 1e-4, None),
              "o", ms=3, c="crimson", label="data")
    kp = fits["korsmeyer_peppas"]
    if not np.isnan(kp["k"]):
        t_dense = np.linspace(times[1] if len(times) > 1 else 1, times[-1], 200)
        ax.loglog(t_dense, kp["k"] * np.power(t_dense, kp["n"]),
                  "-", c="navy", lw=1.5,
                  label=f"K-P n={kp['n']:.3f} R\u00b2={kp['R2']:.3f}")
    hi = fits["higuchi"]
    if not np.isnan(hi["k_H"]):
        t_dense = np.linspace(times[1] if len(times) > 1 else 1, times[-1], 200)
        ax.loglog(t_dense, hi["k_H"] * np.sqrt(t_dense),
                  "--", c="darkgreen", lw=1.2,
                  label=f"Higuchi R\u00b2={hi['R2']:.3f}")
    ax.set_xlabel("Time (s, log)")
    ax.set_ylabel("M(t)/M\u221e (log)")
    ax.set_title("Fractional Release (K-P / Higuchi)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # 3 - Source mass M_source(t) and release rate
    ax = axes[0, 2]
    ax.plot(t_min, kin["M_source"], "o-", ms=3, lw=1, c="darkorange",
            label="M_source(t)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("M_source (% Air Sat. \u00b7 mm)", color="darkorange")
    ax.tick_params(axis="y", labelcolor="darkorange")
    ax.set_title("Metric 5: Source depletion & release rate")
    ax2 = ax.twinx()
    ax2.plot(t_min, kin["release_rate"], "s-", ms=2, lw=1, c="teal", alpha=0.7,
             label="-dM/dt")
    ax2.set_ylabel("-dM/dt (% Air Sat. \u00b7 mm / s)", color="teal")
    ax2.tick_params(axis="y", labelcolor="teal")
    ax2.axhline(0, color="gray", ls=":", lw=0.5)

    # 4 - Penetration depth L_p(t)
    ax = axes[1, 0]
    ax.plot(t_min, kin["L_p"], "o-", ms=3, lw=1, c="purple")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Penetration depth L_p (mm)")
    ax.set_title(
        f"Metric 4: L_p(t)  (thr = {kin['pen_threshold']:.1f} % Air Sat.)"
    )

    # 5 - AUC spatial integral vs time
    ax = axes[1, 1]
    ax.plot(t_min, kin["auc_spatial_t"], "o-", ms=3, lw=1, c="darkgreen")
    ax.fill_between(t_min, 0, kin["auc_spatial_t"], alpha=0.2, color="darkgreen")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("\u222b (C-C_th)\u207a dx  (% Air Sat. \u00b7 mm)")
    ax.set_title(
        f"Metric 3: Therapeutic coverage (thr = +{kin['c_threshold_excess']:.0f} above baseline)\n"
        f"AUC_xt = {kin['auc_xt']:.1f}"
    )

    # 6 - Source region overlay on initial profile
    ax = axes[1, 2]
    excess0 = data[:, 0] - kin["baseline"]
    ax.plot(x, excess0, "-", c="steelblue", lw=1.5, label="C_excess(x, t=0)")
    ax.axvspan(x[kin["src_lo"]], x[kin["src_hi"]], color="orange", alpha=0.25,
               label="source region (FWHM)")
    ax.axhline(kin["pen_threshold"], color="purple", ls="--", lw=1,
               label=f"L_p threshold")
    ax.axhline(kin["c_threshold_excess"], color="darkgreen", ls=":", lw=1,
               label="AUC threshold")
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Excess O\u2082 (% Air Sat.)")
    ax.set_title("Source region & thresholds (t=0)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════

def run_analysis(raw_bytes, filename):
    """Run full diffusion analysis and display results in Streamlit.

    Returns a dict with key metrics for cross-experiment comparison,
    or None if analysis fails.
    """
    x, times_raw, data_raw = load_data_from_bytes(raw_bytes, filename)

    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pixels", data_raw.shape[0])
    col2.metric("Timesteps", data_raw.shape[1])
    col3.metric("Position range", f"{x[0]:.2f} \u2013 {x[-1]:.2f} mm")

    dx = np.median(np.diff(x))
    dt_raw = times_raw[1] - times_raw[0]
    st.write(
        f"**\u0394x** = {dx:.4f} mm &nbsp;|&nbsp; "
        f"**\u0394t** = {dt_raw:.0f} s &nbsp;|&nbsp; "
        f"**Total time** = {times_raw[-1] / 60:.1f} min"
    )

    # Detect loading phase
    t_start = detect_diffusion_start(data_raw)
    t_start_min = t_start * 5
    st.info(
        f"Loading phase detected: first **{t_start_min} min** "
        f"(peak at t = {t_start_min} min). Using this as effective t = 0."
    )

    data = data_raw[:, t_start:]
    times = times_raw[t_start:] - times_raw[t_start]

    # --- Method 1: Moments ---
    st.subheader("Method 1: Variance (Moment) Analysis")
    with st.spinner("Computing moments..."):
        mom = method_moments(x, data, times, n_fit_steps=12)

    sig0 = np.sqrt(max(0, mom["sigma2_0"]))
    col1, col2, col3 = st.columns(3)
    col1.metric("D_moment", f"{mom['D']:.4e} mm\u00b2/s")
    col2.metric("R\u00b2", f"{mom['R2']:.4f}")
    col3.metric("\u03c3\u2080", f"{sig0:.3f} mm")
    st.caption(
        f"Baseline (fixed, from t=0 edges): {mom['baseline']:.1f} % Air Sat. &nbsp;|&nbsp; "
        f"Fit range: first {mom['n_fit']} steps ({mom['n_fit'] * 5} min) &nbsp;|&nbsp; "
        f"slope(\u03c3\u00b2 vs t) = {mom['slope']:.4e} mm\u00b2/s"
    )

    # --- Method 2: PDE ---
    st.subheader("Method 2: Numerical PDE Inversion (Crank-Nicolson)")
    with st.spinner("Optimising D over full time series (this may take a moment)..."):
        pde = method_pde(x, data, times)

    col1, col2 = st.columns(2)
    col1.metric("D_pde", f"{pde['D']:.4e} mm\u00b2/s")
    col2.metric("RMSE", f"{pde['RMSE']:.2f} % Air Sat.")

    # --- Retention ---
    st.subheader("Retention Metrics")
    ret = compute_retention(x, data, times)

    col1, col2, col3 = st.columns(3)
    col1.metric("Peak excess (t=0)", f"{ret['peak_excess'][0]:.1f}")
    col2.metric("Peak excess (t=end)", f"{ret['peak_excess'][-1]:.1f}")
    if not np.isnan(ret["tau_half_peak"]):
        col3.metric("\u03c4_half_peak", f"{ret['tau_half_peak'] / 60:.1f} min")
    fwhm_valid = ret["fwhm"][~np.isnan(ret["fwhm"])]
    if len(fwhm_valid) > 0:
        c1, c2 = st.columns(2)
        c1.metric("FWHM (t=0)", f"{ret['fwhm'][0]:.2f} mm")
        c2.metric("FWHM (t=end)", f"{fwhm_valid[-1]:.2f} mm")

    if not np.isnan(ret["tau_half_peak"]):
        st.success(
            f"\u03c4_half_peak = {ret['tau_half_peak']:.0f} s "
            f"({ret['tau_half_peak'] / 60:.1f} min) \u2014 "
            "higher means the material stays oxygenated longer."
        )

    # --- Release Kinetics (5 metrics + 3 model fits) ---
    st.subheader("Release Kinetics Metrics")

    with st.expander("Metric definitions & formulas", expanded=False):
        # --- Metric 1 ---
        st.markdown("#### :blue[Metric 1 \u2014 primary outcome]")
        st.markdown(
            "**Peak oxygen decay**\n\n"
            "Track the maximum O\u2082 concentration across the cross-section at each timestep. "
            "This is the oxygen level at the source centre."
        )
        st.latex(r"C_{\text{peak}}(t) = \max_{x}\,[\,C(x,\,t)\,]")
        st.latex(r"C_{\text{excess}}(t) = C_{\text{peak}}(t) - C_\infty")
        st.markdown(
            "$C(x, t)$ = measured O\u2082 concentration (% Air Saturation) at position $x$, time $t$\n\n"
            "$C_\\infty$ = atmospheric baseline (~100% Air Sat, from edge pixels at $t = 0$)"
        )
        st.info(
            '**In your meeting:** "We track how the peak oxygen at the PFC source decays over time. '
            "Higher methylcellulose concentrations should show a slower, flatter decay \u2014 "
            'sustained delivery rather than a burst."'
        )
        st.markdown("---")

        # --- Metric 2 ---
        st.markdown("#### :blue[Metric 2 \u2014 headline number]")
        st.markdown(
            "**Half-life of peak excess oxygen \u2014 \u03c4\u00bd**\n\n"
            "The time at which the peak excess drops to 50% of its initial value. "
            "This is the single most important number for comparing materials."
        )
        st.latex(
            r"C_{\text{excess}}(\tau_{1/2}) = \frac{1}{2} \cdot C_{\text{excess}}(0)"
        )
        st.markdown(
            "Computed by linear interpolation between consecutive timesteps where "
            "$C_{\\text{excess}}$ crosses the 50% threshold.\n\n"
            "Units: minutes (or seconds)"
        )
        st.info(
            '**In your meeting:** "\u03c4\u00bd directly answers: how long does the material stay oxygenated? '
            "If \u03c4\u00bd increases from 20 min (no MC) to 80 min (high MC), that's a 4\u00d7 improvement "
            'in delivery duration. This is our bar chart \u2014 the centrepiece result."'
        )
        st.markdown("---")

        # --- Metric 3 ---
        st.markdown("#### :blue[Metric 3 \u2014 therapeutic dose]")
        st.markdown(
            "**Area under the curve \u2014 therapeutic oxygen dose**\n\n"
            "Total oxygen delivered above a clinically meaningful threshold, integrated over "
            "both space and time. Ensures sustained release doesn\u2019t sacrifice total dose."
        )
        st.latex(
            r"\text{AUC} = \int_0^T \int_0^L \max\,[\,C(x,t) - C_{\text{th}},\; 0\,]\; dx\; dt"
        )
        st.latex(
            r"\text{AUC}_{\text{discrete}} = \sum_n \sum_i \max\,[\,C(x_i,\,t_n) - C_{\text{th}},\; 0\,]"
            r"\cdot \Delta x \cdot \Delta t"
        )
        st.markdown(
            "$C_{\\text{th}}$ = therapeutic threshold (e.g. 50% Air Sat \u2014 the level needed to overcome tumour hypoxia)\n\n"
            "$L$ = total cross-section length, $T$ = experiment duration\n\n"
            "Units: (% Air Sat) \u00b7 mm \u00b7 s"
        )
        st.info(
            '**In your meeting:** "AUC is the total oxygen dose that exceeds the therapeutic threshold. '
            "If MC increases \u03c4\u00bd but AUC drops, we\u2019re trapping oxygen \u2014 bad. "
            "If AUC is maintained or increases, we\u2019re delivering the same total dose over a "
            'longer period \u2014 exactly what we want."'
        )
        st.markdown("---")

        # --- Metric 4 ---
        st.markdown("#### :green[Metric 4 \u2014 source characterisation]")
        st.markdown(
            "**Oxygen release rate from the PFC source**\n\n"
            "Track total oxygen within the source region over time. "
            "The negative derivative is the instantaneous release rate."
        )
        st.latex(
            r"M_{\text{source}}(t) = \int_{\text{source}} C(x,\,t)\; dx"
        )
        st.latex(
            r"\text{Release rate} = -\frac{dM_{\text{source}}}{dt}"
        )
        st.latex(
            r"\text{Fractional release: } F(t) = "
            r"\frac{M_{\text{source}}(0) - M_{\text{source}}(t)}"
            r"{M_{\text{source}}(0)} = \frac{M_{\text{released}}(t)}{M_{\text{total}}(\infty)}"
        )
        st.markdown(
            "Source region = central pixels where $C > C_\\infty$ significantly at $t = 0$ (~15\u201320 pixels around peak)\n\n"
            "$F(t)$ ranges from 0 (no release) to 1 (fully depleted)"
        )
        st.info(
            '**In your meeting:** "We measure how fast the PFC plug releases its stored oxygen. '
            "A constant release rate (linear M\u209B\u2092\u1D64\u1D63\u1D9C\u1D49 decay) "
            "is pharmaceutically ideal \u2014 zero-order kinetics. "
            "If methylcellulose converts the exponential burst into linear "
            'release, that\u2019s a strong mechanistic result."'
        )
        st.markdown("---")

        # --- Metric 5 ---
        st.markdown("#### :orange[Metric 5 \u2014 mechanistic insight]")
        st.markdown(
            "**Korsmeyer-Peppas release model**\n\n"
            "Fit the fractional release curve to the power-law model. "
            "The exponent $n$ reveals whether methylcellulose changes the release mechanism."
        )
        st.latex(r"F(t) = k \cdot t^{\,n}")
        st.markdown(
            "$k$ = release rate constant\n\n"
            "$n$ = transport exponent (the key parameter)\n\n"
            "Valid for $F(t) < 0.6$ (first 60% of release)"
        )
        col_n1, col_n2, col_n3 = st.columns(3)
        col_n1.markdown("**$n = 0.5$**\n\nFickian diffusion \u2014 no barrier effect from MC")
        col_n2.markdown("**$0.5 < n < 1.0$**\n\nAnomalous transport \u2014 mixed diffusion + viscosity barrier")
        col_n3.markdown("**$n = 1.0$**\n\nZero-order release \u2014 fully barrier-controlled (ideal for therapy)")
        st.info(
            '**In your meeting:** "The Korsmeyer-Peppas exponent tells us what methylcellulose is doing mechanistically. '
            "If $n$ stays at 0.5 regardless of MC concentration, the viscosity is just slowing diffusion \u2014 "
            "not changing the mechanism. If $n$ increases toward 1.0 with higher MC, it proves that the viscosity "
            "barrier is fundamentally converting burst release into controlled, "
            'sustained release. That\u2019s the mechanistic story for the paper."'
        )
        st.markdown("---")

        # --- Complementary: first-order ---
        st.markdown("#### :gray[Complementary \u2014 first-order comparison]")
        st.markdown(
            "**First-order (exponential) release model**\n\n"
            "The baseline model \u2014 simple diffusion-limited release with no barrier. "
            "Fit this alongside Korsmeyer-Peppas; if it fits better, MC isn\u2019t changing the mechanism."
        )
        st.latex(r"C_{\text{excess}}(t) = C_0 \cdot e^{-k_1 t}")
        st.latex(r"\tau_{1/2} = \frac{\ln 2}{k_1}")
        st.markdown(
            "$C_0$ = initial peak excess\n\n"
            "$k_1$ = first-order rate constant (s\u207b\u00b9)\n\n"
            "Compare R\u00b2 of this fit vs Korsmeyer-Peppas to determine which model describes your data better."
        )

    # User-tunable thresholds
    cfg1, cfg2 = st.columns(2)
    c_th_excess = cfg1.number_input(
        "AUC threshold (% Air Sat. above baseline)",
        min_value=0.0, max_value=500.0, value=50.0, step=10.0,
        key=f"auc_th_{filename}",
        help="Therapeutic threshold expressed as excess above the GelMA baseline.",
    )
    pen_frac = cfg2.number_input(
        "Penetration depth threshold (fraction of initial peak excess)",
        min_value=0.01, max_value=0.99, value=0.10, step=0.05,
        key=f"pen_frac_{filename}",
        help="L_p is the width over which excess > frac \u00d7 initial peak excess.",
    )

    kin = compute_kinetics_metrics(
        x, data, times,
        c_threshold_excess=float(c_th_excess),
        pen_threshold_frac=float(pen_frac),
    )
    fits = fit_release_models(times, ret["peak_excess"])

    # Headline numbers
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "AUC (% Air Sat \u00b7 mm \u00b7 s)",
        f"{kin['auc_xt']:.1f}",
        help="Metric 3: total therapeutic dose above threshold, integrated over space and time.",
    )
    k2.metric(
        "L_p (t=0)",
        f"{kin['L_p'][0]:.2f} mm",
        help="Metric 4: penetration depth at t=0.",
    )
    k3.metric(
        "L_p (t=end)",
        f"{kin['L_p'][-1]:.2f} mm",
        help="Metric 4: penetration depth at the final timestep.",
    )
    rr0 = kin["release_rate"][0]
    k4.metric(
        "Initial release rate",
        f"{rr0:.3f} %\u00b7mm/s",
        help="Metric 5: -dM_source/dt at t=0.",
    )

    # Model fits
    st.markdown("**Release-kinetics model fits** (fit on peak excess decay)")
    fo = fits["first_order"]
    kp = fits["korsmeyer_peppas"]
    hi = fits["higuchi"]

    fit_rows = [
        {
            "Model": "First-order  C\u2080 e^(-kt)",
            "Param 1": f"k = {fo['k']:.4e} s\u207b\u00b9" if not np.isnan(fo["k"]) else "N/A",
            "Param 2": f"\u03c4\u00bd = {fo['tau_half']/60:.1f} min" if not np.isnan(fo["tau_half"]) else "N/A",
            "R\u00b2": f"{fo['R2']:.3f}" if not np.isnan(fo["R2"]) else "N/A",
        },
        {
            "Model": "Korsmeyer-Peppas  k t\u207f",
            "Param 1": f"k = {kp['k']:.4e}" if not np.isnan(kp["k"]) else "N/A",
            "Param 2": f"n = {kp['n']:.3f} ({classify_kp_mechanism(kp['n'])})" if not np.isnan(kp["n"]) else "N/A",
            "R\u00b2": f"{kp['R2']:.3f}" if not np.isnan(kp["R2"]) else "N/A",
        },
        {
            "Model": "Higuchi  k_H \u221at",
            "Param 1": f"k_H = {hi['k_H']:.4e} s^-\u00bd" if not np.isnan(hi["k_H"]) else "N/A",
            "Param 2": "\u2014",
            "R\u00b2": f"{hi['R2']:.3f}" if not np.isnan(hi["R2"]) else "N/A",
        },
    ]
    st.table(pd.DataFrame(fit_rows))

    if not np.isnan(kp["n"]):
        if kp["n"] > 0.85:
            st.success(
                f"K-P exponent n = {kp['n']:.2f} \u2192 release is approaching "
                "barrier-controlled (Case II) kinetics. Strong evidence the matrix "
                "is converting diffusion-limited release into sustained release."
            )
        elif kp["n"] > 0.55:
            st.info(
                f"K-P exponent n = {kp['n']:.2f} \u2192 anomalous transport "
                "(mixed Fickian + barrier control)."
            )
        else:
            st.warning(
                f"K-P exponent n = {kp['n']:.2f} \u2192 Fickian (diffusion-limited) "
                "release. The matrix is not (yet) creating a meaningful barrier."
            )

    # Kinetics plots
    fig_kin = make_kinetics_plots(x, times, data, ret, kin, fits)
    st.pyplot(fig_kin)
    buf_kin = io.BytesIO()
    fig_kin.savefig(buf_kin, format="png", dpi=180, bbox_inches="tight")
    buf_kin.seek(0)
    st.download_button(
        label="Download kinetics plot as PNG",
        data=buf_kin,
        file_name=f"kinetics_{filename}.png",
        mime="image/png",
        key=f"download_kin_{filename}",
    )
    plt.close(fig_kin)

    # --- Summary ---
    st.subheader("Summary")
    st.markdown(
        f"| Method | D (mm\u00b2/s) | Quality |\n"
        f"|--------|--------|---------|\n"
        f"| Moment (early-time) | `{mom['D']:.4e}` | R\u00b2 = {mom['R2']:.3f} |\n"
        f"| PDE (full time) | `{pde['D']:.4e}` | RMSE = {pde['RMSE']:.1f} |"
    )
    st.caption(
        "The two methods may disagree if D is concentration-dependent or if "
        "there is a source/sink term. The moment method reflects early-time "
        "behaviour; the PDE method is a global average."
    )

    # --- Plots ---
    st.subheader("Diagnostic Plots")
    fig = make_plots(x, times, data, mom, pde, ret, t_start_min)
    st.pyplot(fig)

    # Download button for the figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download plot as PNG",
        data=buf,
        file_name=f"diffusion_analysis_{filename}.png",
        mime="image/png",
        key=f"download_plot_{filename}",
    )
    plt.close(fig)

    # Return summary for cross-experiment comparison
    label = filename.rsplit(".", 1)[0]  # strip extension
    fwhm_valid = ret["fwhm"][~np.isnan(ret["fwhm"])]
    return {
        "label": label,
        "D_moment": mom["D"],
        "D_pde": pde["D"],
        "R2": mom["R2"],
        "RMSE": pde["RMSE"],
        "tau_half_peak": ret["tau_half_peak"],
        "peak_excess_0": ret["peak_excess"][0],
        "peak_excess_end": ret["peak_excess"][-1],
        "fwhm_0": ret["fwhm"][0] if len(fwhm_valid) > 0 else np.nan,
        "fwhm_end": fwhm_valid[-1] if len(fwhm_valid) > 0 else np.nan,
        "sigma2": mom["sigma2"],
        "times": times,
        "peak_excess": ret["peak_excess"],
        "fwhm": ret["fwhm"],
        # Release kinetics
        "auc_xt": kin["auc_xt"],
        "auc_spatial_t": kin["auc_spatial_t"],
        "L_p": kin["L_p"],
        "M_source": kin["M_source"],
        "release_rate": kin["release_rate"],
        "L_p_0": kin["L_p"][0],
        "L_p_end": kin["L_p"][-1],
        "release_rate_0": kin["release_rate"][0],
        "c_threshold_excess": kin["c_threshold_excess"],
        "fo_k": fits["first_order"]["k"],
        "fo_tau_half": fits["first_order"]["tau_half"],
        "fo_R2": fits["first_order"]["R2"],
        "kp_k": fits["korsmeyer_peppas"]["k"],
        "kp_n": fits["korsmeyer_peppas"]["n"],
        "kp_R2": fits["korsmeyer_peppas"]["R2"],
        "hi_k": fits["higuchi"]["k_H"],
        "hi_R2": fits["higuchi"]["R2"],
    }


# ═══════════════════════════════════════════════════════════════════════
# AVERAGED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def make_avg_profiles_plot(x, times, data_mean, data_std, n_ds, t_start_min):
    """Profiles with \u00b1 1 std band."""
    t_min = times / 60
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Averaged Data (n = {n_ds} replicates, \u00b1 1 SD shading)  |  "
        f"loading phase removed: first {t_start_min} min",
        fontsize=12, fontweight="bold",
    )

    # 1 - Profiles at selected times
    ax = axes[0]
    idx = np.linspace(0, len(times) - 1, 8, dtype=int)
    for i in idx:
        ax.plot(x, data_mean[:, i], lw=1.2, label=f"{t_min[i]:.0f} min")
        ax.fill_between(
            x,
            data_mean[:, i] - data_std[:, i],
            data_mean[:, i] + data_std[:, i],
            alpha=0.15,
        )
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("O\u2082 (% Air Sat.)")
    ax.set_title("Mean profiles \u00b1 SD")
    ax.legend(fontsize=7, ncol=2)

    # 2 - Peak excess with std band over time
    ax = axes[1]
    bl = 0.5 * (data_mean[:15, 0].mean() + data_mean[-15:, 0].mean())
    excess_mean = data_mean - bl
    peak_mean = excess_mean.max(axis=0)
    # Approximate peak std from data_std at peak location per timestep
    peak_idx = excess_mean.argmax(axis=0)
    peak_std = np.array([data_std[peak_idx[t], t] for t in range(len(times))])
    ax.plot(t_min, peak_mean, "o-", ms=3, lw=1.2, c="crimson", label="mean peak excess")
    ax.fill_between(t_min, peak_mean - peak_std, peak_mean + peak_std,
                    color="crimson", alpha=0.15, label="\u00b1 1 SD")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Peak Excess O\u2082")
    ax.set_title("Peak decay (mean \u00b1 SD)")
    ax.legend(fontsize=8)

    # 3 - Centre pixel time-trace with std band
    ax = axes[2]
    mid = data_mean.shape[0] // 2
    hw = 5
    centre_mean = data_mean[mid - hw:mid + hw, :].mean(axis=0)
    centre_std = data_std[mid - hw:mid + hw, :].mean(axis=0)
    ax.plot(t_min, centre_mean, "o-", ms=3, lw=1.2, c="steelblue",
            label="centre (mean \u00b1 SD)")
    ax.fill_between(t_min, centre_mean - centre_std, centre_mean + centre_std,
                    color="steelblue", alpha=0.15)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("O\u2082 (% Air Sat.)")
    ax.set_title("Source-centre time trace")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def run_average_analysis(datasets, filenames, all_results):
    """Run full analysis on the averaged dataset and show results.

    Parameters
    ----------
    datasets : list of (x, times, data_raw) tuples (pre-loading-trim)
    filenames : list of str
    all_results : list of per-experiment result dicts (for metric spread)
    """
    n = len(datasets)
    st.subheader(f"Averaged Analysis (n = {n} replicates)")
    st.markdown(
        f"Datasets: **{', '.join(filenames)}**\n\n"
        "The raw C(x, t) arrays are interpolated onto a common grid and "
        "element-wise averaged. All metrics below are computed on the mean "
        "data. Replicate spread (\u00b1 1 SD) is shown where applicable."
    )

    # --- Compute average ---
    x_avg, times_avg, data_mean, data_std, n_ds = compute_average_data(datasets)

    # Detect loading phase on averaged data
    t_start = detect_diffusion_start(data_mean)
    t_start_min = t_start * 5
    st.info(
        f"Loading phase (on average): first **{t_start_min} min**. "
        "Using this as effective t = 0."
    )

    data = data_mean[:, t_start:]
    data_sd = data_std[:, t_start:]
    times = times_avg[t_start:] - times_avg[t_start]

    # --- Averaged profile figure ---
    fig_avg = make_avg_profiles_plot(x_avg, times, data, data_sd, n_ds, t_start_min)
    st.pyplot(fig_avg)
    buf = io.BytesIO()
    fig_avg.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download averaged profiles (PNG)", buf,
                       "averaged_profiles.png", "image/png",
                       key="dl_avg_profiles")
    plt.close(fig_avg)

    # --- Run standard analysis on the averaged data ---
    st.markdown("---")
    st.markdown("### Metrics on Averaged Data")

    mom = method_moments(x_avg, data, times, n_fit_steps=12)
    pde = method_pde(x_avg, data, times)
    ret = compute_retention(x_avg, data, times)
    kin = compute_kinetics_metrics(x_avg, data, times)
    fits = fit_release_models(times, ret["peak_excess"])

    # Key metrics from averaged data
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("D_moment (avg)", f"{mom['D']:.4e} mm\u00b2/s")
    col2.metric("D_pde (avg)", f"{pde['D']:.4e} mm\u00b2/s")
    if not np.isnan(ret["tau_half_peak"]):
        col3.metric("\u03c4\u00bd (avg)", f"{ret['tau_half_peak'] / 60:.1f} min")
    else:
        col3.metric("\u03c4\u00bd (avg)", "N/A")
    col4.metric("RMSE (avg)", f"{pde['RMSE']:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC (avg)", f"{kin['auc_xt']:.1f}")
    col2.metric("L_p\u2080 (avg)", f"{kin['L_p'][0]:.2f} mm")
    kp_n = fits["korsmeyer_peppas"]["n"]
    col3.metric("K-P n (avg)", f"{kp_n:.3f}" if not np.isnan(kp_n) else "N/A")
    fo_tau = fits["first_order"]["tau_half"]
    col4.metric("1st-order \u03c4\u00bd (avg)",
                f"{fo_tau / 60:.1f} min" if not np.isnan(fo_tau) else "N/A")

    # --- Replicate spread table (mean \u00b1 SD from individual results) ---
    if len(all_results) >= 2:
        st.markdown("---")
        st.markdown("### Replicate Spread (mean \u00b1 SD across individual experiments)")

        def _ms(vals, fmt=".4e", scale=1.0):
            """Format mean \u00b1 SD, skipping NaN."""
            clean = [v * scale for v in vals if not np.isnan(v)]
            if not clean:
                return "N/A"
            m, s = np.mean(clean), np.std(clean, ddof=1)
            return f"{m:{fmt}} \u00b1 {s:{fmt}}"

        spread_rows = [
            {"Metric": "D_moment (mm\u00b2/s)",
             "Mean \u00b1 SD": _ms([r["D_moment"] for r in all_results])},
            {"Metric": "D_pde (mm\u00b2/s)",
             "Mean \u00b1 SD": _ms([r["D_pde"] for r in all_results])},
            {"Metric": "\u03c4\u00bd peak (min)",
             "Mean \u00b1 SD": _ms([r["tau_half_peak"] for r in all_results],
                                   fmt=".1f", scale=1 / 60)},
            {"Metric": "AUC (%\u00b7mm\u00b7s)",
             "Mean \u00b1 SD": _ms([r["auc_xt"] for r in all_results], fmt=".1f")},
            {"Metric": "L_p\u2080 (mm)",
             "Mean \u00b1 SD": _ms([r["L_p_0"] for r in all_results], fmt=".2f")},
            {"Metric": "Init. release rate (%\u00b7mm/s)",
             "Mean \u00b1 SD": _ms([r["release_rate_0"] for r in all_results], fmt=".3f")},
            {"Metric": "K-P n",
             "Mean \u00b1 SD": _ms([r["kp_n"] for r in all_results], fmt=".3f")},
            {"Metric": "1st-order \u03c4\u00bd (min)",
             "Mean \u00b1 SD": _ms([r["fo_tau_half"] for r in all_results],
                                   fmt=".1f", scale=1 / 60)},
            {"Metric": "Peak excess (t=0)",
             "Mean \u00b1 SD": _ms([r["peak_excess_0"] for r in all_results], fmt=".1f")},
            {"Metric": "FWHM\u2080 (mm)",
             "Mean \u00b1 SD": _ms([r["fwhm_0"] for r in all_results], fmt=".2f")},
        ]
        st.table(pd.DataFrame(spread_rows))

    # --- Diagnostic plots on averaged data ---
    st.markdown("---")
    st.markdown("### Diagnostic Plots (on averaged data)")
    fig = make_plots(x_avg, times, data, mom, pde, ret, t_start_min)
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download avg. diffusion plots (PNG)", buf,
                       "avg_diffusion_plots.png", "image/png",
                       key="dl_avg_diff")
    plt.close(fig)

    # Kinetics plots
    st.markdown("### Kinetics Plots (on averaged data)")
    fig_kin = make_kinetics_plots(x_avg, times, data, ret, kin, fits)
    st.pyplot(fig_kin)
    buf = io.BytesIO()
    fig_kin.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download avg. kinetics plots (PNG)", buf,
                       "avg_kinetics_plots.png", "image/png",
                       key="dl_avg_kin")
    plt.close(fig_kin)

    if not np.isnan(kp_n):
        mech = classify_kp_mechanism(kp_n)
        if kp_n > 0.85:
            st.success(f"Averaged K-P n = {kp_n:.3f} \u2192 {mech}")
        elif kp_n > 0.55:
            st.info(f"Averaged K-P n = {kp_n:.3f} \u2192 {mech}")
        else:
            st.warning(f"Averaged K-P n = {kp_n:.3f} \u2192 {mech}")


# ═══════════════════════════════════════════════════════════════════════
# MULTI-EXPERIMENT COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def make_comparison_section(all_results):
    """Display comparison plots and table for multiple experiments."""
    st.header("Cross-Experiment Comparison")

    labels = [r["label"] for r in all_results]
    n = len(all_results)

    # --- Summary table ---
    st.subheader("Comparison Table")
    rows = []
    for r in all_results:
        tau_str = f"{r['tau_half_peak'] / 60:.1f}" if not np.isnan(r["tau_half_peak"]) else "N/A"
        rows.append({
            "Experiment": r["label"],
            "D_moment (mm\u00b2/s)": f"{r['D_moment']:.4e}",
            "D_pde (mm\u00b2/s)": f"{r['D_pde']:.4e}",
            "R\u00b2": f"{r['R2']:.4f}",
            "RMSE": f"{r['RMSE']:.2f}",
            "\u03c4_half (min)": tau_str,
            "FWHM\u2080 (mm)": f"{r['fwhm_0']:.2f}" if not np.isnan(r["fwhm_0"]) else "N/A",
        })
    st.table(pd.DataFrame(rows))

    # --- Release kinetics comparison table ---
    st.subheader("Release Kinetics Comparison Table")
    kin_rows = []
    for r in all_results:
        kin_rows.append({
            "Experiment": r["label"],
            "AUC (%\u00b7mm\u00b7s)": f"{r['auc_xt']:.1f}",
            "L_p\u2080 (mm)": f"{r['L_p_0']:.2f}",
            "L_p_end (mm)": f"{r['L_p_end']:.2f}",
            "Init. release (%\u00b7mm/s)": f"{r['release_rate_0']:.3f}",
            "1st-order \u03c4\u00bd (min)": (
                f"{r['fo_tau_half']/60:.1f}" if not np.isnan(r["fo_tau_half"]) else "N/A"
            ),
            "K-P n": f"{r['kp_n']:.3f}" if not np.isnan(r["kp_n"]) else "N/A",
            "K-P R\u00b2": f"{r['kp_R2']:.3f}" if not np.isnan(r["kp_R2"]) else "N/A",
            "Mechanism": classify_kp_mechanism(r["kp_n"]),
        })
    st.table(pd.DataFrame(kin_rows))

    # --- Bar chart comparison plots ---
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))[:n]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Experiment Comparison", fontsize=13, fontweight="bold")

    # 1 - D coefficients (grouped bar)
    ax = axes[0]
    x_pos = np.arange(n)
    w = 0.35
    d_mom = [r["D_moment"] for r in all_results]
    d_pde = [r["D_pde"] for r in all_results]
    bars1 = ax.bar(x_pos - w / 2, d_mom, w, label="D_moment", color="steelblue")
    bars2 = ax.bar(x_pos + w / 2, d_pde, w, label="D_pde", color="darkorange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("D (mm\u00b2/s)")
    ax.set_title("Diffusion Coefficients")
    ax.legend(fontsize=8)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))

    # 2 - Half-life
    ax = axes[1]
    tau_vals = [r["tau_half_peak"] / 60 if not np.isnan(r["tau_half_peak"]) else 0
                for r in all_results]
    bars = ax.bar(x_pos, tau_vals, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("\u03c4_half (min)")
    ax.set_title("Peak Half-Life")
    for i, v in enumerate(tau_vals):
        if v > 0:
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

    # 3 - Initial peak excess
    ax = axes[2]
    pe_vals = [r["peak_excess_0"] for r in all_results]
    bars = ax.bar(x_pos, pe_vals, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Peak Excess O\u2082 (% Air Sat.)")
    ax.set_title("Initial Peak Excess")

    plt.tight_layout()
    st.pyplot(fig)

    # --- Overlay line plots ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Time-Series Comparison", fontsize=13, fontweight="bold")

    # 1 - sigma^2 over time
    ax = axes2[0]
    for i, r in enumerate(all_results):
        t_min = r["times"] / 60
        ax.plot(t_min, r["sigma2"], "o-", ms=2, lw=1, color=colors[i], label=r["label"])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("\u03c3\u00b2 (mm\u00b2)")
    ax.set_title("Variance Growth")
    ax.legend(fontsize=7)

    # 2 - Peak excess decay
    ax = axes2[1]
    for i, r in enumerate(all_results):
        t_min = r["times"] / 60
        ax.plot(t_min, r["peak_excess"], "o-", ms=2, lw=1, color=colors[i], label=r["label"])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Peak Excess O\u2082")
    ax.set_title("Peak Decay Comparison")
    ax.legend(fontsize=7)

    # 3 - FWHM over time
    ax = axes2[2]
    for i, r in enumerate(all_results):
        t_min = r["times"] / 60
        valid = ~np.isnan(r["fwhm"])
        ax.plot(t_min[valid], r["fwhm"][valid], "s-", ms=2, lw=1, color=colors[i],
                label=r["label"])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("FWHM (mm)")
    ax.set_title("Profile Width Comparison")
    ax.legend(fontsize=7)

    plt.tight_layout()
    st.pyplot(fig2)

    # --- Release kinetics bar charts (AUC, K-P n, 1st-order tau) ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    fig3.suptitle("Release Kinetics Comparison", fontsize=13, fontweight="bold")

    # 1 - AUC above therapeutic threshold
    ax = axes3[0]
    auc_vals = [r["auc_xt"] for r in all_results]
    ax.bar(x_pos, auc_vals, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC (% Air Sat. \u00b7 mm \u00b7 s)")
    ax.set_title("Therapeutic AUC (Metric 3)")
    for i, v in enumerate(auc_vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    # 2 - First-order half-life from fit
    ax = axes3[1]
    tau_fo = [
        (r["fo_tau_half"] / 60) if not np.isnan(r["fo_tau_half"]) else 0
        for r in all_results
    ]
    ax.bar(x_pos, tau_fo, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("\u03c4\u00bd (min)")
    ax.set_title("First-order \u03c4\u00bd (Metric 2)")
    for i, v in enumerate(tau_fo):
        if v > 0:
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # 3 - Korsmeyer-Peppas n
    ax = axes3[2]
    n_vals = [r["kp_n"] if not np.isnan(r["kp_n"]) else 0 for r in all_results]
    ax.bar(x_pos, n_vals, color=colors)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Fickian (n=0.5)")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="Zero-order (n=1)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Korsmeyer-Peppas n")
    ax.set_title("K-P transport exponent")
    ax.set_ylim(0, max(1.2, max(n_vals + [0]) * 1.1))
    ax.legend(fontsize=8)
    for i, v in enumerate(n_vals):
        if v > 0:
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig3)

    # --- Release kinetics overlay plots (L_p(t), M_source(t), spatial AUC(t)) ---
    fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
    fig4.suptitle("Release Kinetics Time Series", fontsize=13, fontweight="bold")

    # 1 - Penetration depth L_p(t)
    ax = axes4[0]
    for i, r in enumerate(all_results):
        t_min = r["times"] / 60
        ax.plot(t_min, r["L_p"], "o-", ms=2, lw=1, color=colors[i], label=r["label"])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("L_p (mm)")
    ax.set_title("Penetration Depth (Metric 4)")
    ax.legend(fontsize=7)

    # 2 - Source mass M_source(t)
    ax = axes4[1]
    for i, r in enumerate(all_results):
        t_min = r["times"] / 60
        ax.plot(t_min, r["M_source"], "o-", ms=2, lw=1, color=colors[i],
                label=r["label"])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("M_source (% Air Sat. \u00b7 mm)")
    ax.set_title("Source Depletion (Metric 5)")
    ax.legend(fontsize=7)

    # 3 - Spatial AUC at each time
    ax = axes4[2]
    for i, r in enumerate(all_results):
        t_min = r["times"] / 60
        ax.plot(t_min, r["auc_spatial_t"], "o-", ms=2, lw=1, color=colors[i],
                label=r["label"])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("\u222b (C-C_th)\u207a dx (% Air Sat. \u00b7 mm)")
    ax.set_title("Therapeutic Coverage vs Time")
    ax.legend(fontsize=7)

    plt.tight_layout()
    st.pyplot(fig4)

    # Download comparison plots
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=180, bbox_inches="tight")
    buf2.seek(0)
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format="png", dpi=180, bbox_inches="tight")
    buf3.seek(0)
    buf4 = io.BytesIO()
    fig4.savefig(buf4, format="png", dpi=180, bbox_inches="tight")
    buf4.seek(0)

    col1, col2, col3, col4 = st.columns(4)
    col1.download_button("Download D / \u03c4\u00bd bars (PNG)", buf,
                         "comparison_bars.png", "image/png")
    col2.download_button("Download time-series (PNG)", buf2,
                         "comparison_timeseries.png", "image/png")
    col3.download_button("Download kinetics bars (PNG)", buf3,
                         "kinetics_bars.png", "image/png")
    col4.download_button("Download kinetics time-series (PNG)", buf4,
                         "kinetics_timeseries.png", "image/png")
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


# ─────────────────────── PAGE CONFIG & MAIN ──────────────────────────

st.set_page_config(
    page_title="Diffusion Coefficient Analyzer",
    page_icon="\U0001f9ea",
    layout="wide",
)

st.title("\U0001f9ea Diffusion Coefficient (D) Analyzer")
st.markdown(
    "Upload one or more **CSV** or **Excel** files containing 1D oxygen "
    "cross-section data. Each file is analysed independently. "
    "When multiple files are uploaded you also get an **averaged analysis** "
    "(mean \u00b1 SD) and **cross-experiment comparison** plots."
)

with st.expander("Expected file format"):
    st.markdown(
        """
**Row 0** &mdash; timestamps (`HH:MM:SS`) at 5-min intervals (first two cells can be empty/labels).

**Row 1** &mdash; header: `Pixel, mm, measurement_IDs...`

**Rows 2+** &mdash; `pixel_index, position_mm, O2_at_t0, O2_at_t1, ...`

The oxygen profile should peak at the material centre and flatten over time
(O2 diffuses outward from a loaded core).
        """
    )

uploaded_files = st.file_uploader(
    "Upload your data file(s)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="CSV or Excel with the format described above. Upload multiple files to compare experiments.",
)

if uploaded_files:
    n_files = len(uploaded_files)

    # Pre-parse all files so we can build tabs
    raw_data_list = []   # list of (raw_bytes, filename)
    parsed_datasets = [] # list of (x, times, data_raw)  — for averaging
    for uploaded in uploaded_files:
        raw = uploaded.getvalue()
        raw_data_list.append((raw, uploaded.name))
        try:
            parsed_datasets.append(load_data_from_bytes(raw, uploaded.name))
        except Exception:
            parsed_datasets.append(None)

    # --- Build tabs ---
    tab_labels = [f"\U0001f4c4 {uploaded.name}" for uploaded in uploaded_files]
    if n_files >= 2:
        tab_labels += ["\U0001f4ca Average (mean \u00b1 SD)", "\U0001f50d Comparison"]
    tabs = st.tabs(tab_labels)

    all_results = []

    # --- Individual experiment tabs ---
    for i, tab in enumerate(tabs[:n_files]):
        with tab:
            raw_bytes, fname = raw_data_list[i]
            st.header(f"Experiment: {fname}")
            try:
                result = run_analysis(raw_bytes, fname)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                st.error(f"Analysis failed for {fname}: {e}")
                st.exception(e)

    # --- Average & Comparison tabs (only for 2+ files) ---
    if n_files >= 2:
        # Filter to successfully parsed datasets
        valid = [(ds, raw_data_list[i][1])
                 for i, ds in enumerate(parsed_datasets) if ds is not None]
        valid_datasets = [v[0] for v in valid]
        valid_names = [v[1] for v in valid]

        # Average tab
        with tabs[n_files]:
            if len(valid_datasets) >= 2:
                try:
                    run_average_analysis(valid_datasets, valid_names, all_results)
                except Exception as e:
                    st.error(f"Averaging failed: {e}")
                    st.exception(e)
            else:
                st.warning(
                    "Need at least 2 successfully parsed datasets to compute an average."
                )

        # Comparison tab
        with tabs[n_files + 1]:
            if len(all_results) >= 2:
                make_comparison_section(all_results)
            else:
                st.warning(
                    "Need at least 2 successfully analysed datasets for comparison."
                )
else:
    st.info("Upload one or more files to get started.")
