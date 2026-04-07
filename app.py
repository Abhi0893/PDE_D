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
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


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


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════

def run_analysis(raw_bytes, filename):
    """Run full diffusion analysis and display results in Streamlit."""
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
        file_name="diffusion_analysis.png",
        mime="image/png",
    )
    plt.close(fig)


# ─────────────────────── PAGE CONFIG & MAIN ──────────────────────────

st.set_page_config(
    page_title="Diffusion Coefficient Analyzer",
    page_icon="\U0001f9ea",
    layout="wide",
)

st.title("\U0001f9ea Diffusion Coefficient (D) Analyzer")
st.markdown(
    "Upload a **CSV** or **Excel** file containing 1D oxygen cross-section "
    "data to compute the diffusion coefficient using two independent methods "
    "and view retention metrics."
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

uploaded = st.file_uploader(
    "Upload your data file",
    type=["csv", "xlsx", "xls"],
    help="CSV or Excel with the format described above.",
)

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    try:
        run_analysis(raw_bytes, uploaded.name)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.exception(e)
else:
    st.info("Upload a file to get started.")
