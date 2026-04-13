#!/usr/bin/env python3
"""
Oxygen Release Kinetics Analyzer — Patent Metrics
==================================================

Upload one or more Excel/CSV files (replicates of the same condition)
and get the four patent-critical metrics:

  1. τ₁/₂           — delivery duration
  2. t_handoff       — biphasic release proof
  3. 1st-order R²    — model inadequacy proof
  4. C_peak⁰         — loading capacity
"""

import io
import csv
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data_from_bytes(raw_bytes: bytes, filename: str):
    """Parse uploaded file → positions (mm), times (s), C(x,t) matrix.

    Handles common Excel/CSV quirks:
      - Trailing empty rows
      - Rows with NaN/blank positions (skipped)
      - NaN values in data cells (forward-filled, then zero-filled)
      - Timestamps as strings ("HH:MM:SS") or datetime.time objects
    """

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw_bytes), header=None)
        rows = df.values.tolist()           # keep native types
    else:
        text = raw_bytes.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))

    # Row 0: timestamps (HH:MM:SS or datetime.time) starting at column 2
    time_strs = rows[0][2:]
    times = []
    for ts in time_strs:
        ts_str = str(ts).strip()
        if ts_str in ("", "nan", "None", "NaT"):
            continue
        # Handle datetime.time objects from Excel
        import datetime
        if isinstance(ts, datetime.time):
            times.append(ts.hour * 3600 + ts.minute * 60 + ts.second)
            continue
        parts = ts_str.split(":")
        if len(parts) == 3:
            times.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
        elif len(parts) == 2:
            times.append(int(parts[0]) * 3600 + int(parts[1]) * 60)
        else:
            # Try as raw seconds
            times.append(float(ts_str))
    times = np.array(times, dtype=float)

    # Row 1: header row (skip)
    # Rows 2+: pixel_index, position_mm, O2 values …
    n_times = len(times)
    positions, profiles = [], []
    for row in rows[2:]:
        if len(row) < 3:
            continue
        # Skip rows where position is missing/NaN
        pos_str = str(row[1]).strip()
        if pos_str in ("", "nan", "None"):
            continue
        try:
            pos = float(pos_str)
        except (ValueError, TypeError):
            continue
        if np.isnan(pos):
            continue

        # Parse data values, converting blanks/NaN to np.nan
        vals = []
        for v in row[2:2 + n_times]:
            try:
                fv = float(v)
            except (ValueError, TypeError):
                fv = np.nan
            vals.append(fv)
        # Pad if row is shorter than expected
        while len(vals) < n_times:
            vals.append(np.nan)

        positions.append(pos)
        profiles.append(vals)

    positions = np.array(positions)
    profiles = np.array(profiles)

    # Handle any remaining NaN in data: forward-fill along time, then backfill
    if np.isnan(profiles).any():
        prof_df = pd.DataFrame(profiles)
        prof_df = prof_df.ffill(axis=1).bfill(axis=1)
        # If entire rows are NaN (shouldn't happen after pos filter), fill with 0
        prof_df = prof_df.fillna(0.0)
        profiles = prof_df.values

    return positions, times, profiles


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
# PATENT METRICS (4 only)
# ═══════════════════════════════════════════════════════════════════════

def compute_patent_metrics(x, data_raw, times_raw, n_edge=15):
    """Compute the 4 patent-critical metrics from raw C(x,t).

    Returns a dict with all per-experiment values.
    """
    N = data_raw.shape[0]

    # 1. Baseline from RAW t=0 edges (before loading trim)
    bl = 0.5 * (data_raw[:n_edge, 0].mean() + data_raw[-n_edge:, 0].mean())

    # 2. Detect loading phase and trim
    t_start = detect_diffusion_start(data_raw)
    data = data_raw[:, t_start:]
    times = times_raw[t_start:] - times_raw[t_start]
    n_td = len(times)
    dt = times[1] - times[0] if n_td > 1 else 300.0

    # 3. Fixed centre region: ±5 px around peak at diffusion t=0
    excess = data - bl
    peak_px = int(np.argmax(data[:, 0]))
    half_w = 5
    lo = max(0, peak_px - half_w)
    hi = min(N, peak_px + half_w + 1)
    peak_excess = excess[lo:hi, :].mean(axis=0)
    peak_conc = data[lo:hi, :].mean(axis=0)

    # ── METRIC 1: τ₁/₂ ──
    target = peak_excess[0] * 0.5
    tau_half = np.nan
    for i in range(1, n_td):
        if peak_excess[i] <= target:
            frac = (target - peak_excess[i - 1]) / (peak_excess[i] - peak_excess[i - 1])
            tau_half = times[i - 1] + frac * (times[i] - times[i - 1])
            break

    # ── METRIC 2: t_handoff (biphasic proof) ──
    t_handoff = np.nan
    rr_at_0 = np.nan
    rr_at_peak = np.nan
    is_biphasic = False
    if n_td >= 5:
        pe_smooth = np.convolve(peak_excess, np.ones(3) / 3, mode="valid")
        t_smooth = times[1:-1]
        release_rate = -np.gradient(pe_smooth, t_smooth)
        rr_peak_idx = int(np.argmax(release_rate))
        t_handoff = float(t_smooth[rr_peak_idx])
        rr_at_0 = float(release_rate[0])
        rr_at_peak = float(release_rate[rr_peak_idx])
        is_biphasic = t_handoff > 600  # >10 min

    # ── METRIC 3: First-order fit ──
    fo_R2 = np.nan
    fo_k = np.nan
    fo_tau_half = np.nan
    fo_C0 = np.nan
    resid_early = np.nan
    resid_mid = np.nan
    resid_late = np.nan
    fo_pred = None
    try:
        popt, _ = curve_fit(
            lambda t, C0, k: C0 * np.exp(-k * t),
            times, peak_excess,
            p0=[peak_excess[0], 1e-3],
            bounds=([0, 0], [10 * peak_excess[0] + 1, 1.0]),
            maxfev=5000,
        )
        fo_C0, fo_k = float(popt[0]), float(popt[1])
        fo_pred = fo_C0 * np.exp(-fo_k * times)
        ss_res = np.sum((peak_excess - fo_pred) ** 2)
        ss_tot = np.sum((peak_excess - peak_excess.mean()) ** 2)
        fo_R2 = 1.0 - ss_res / (ss_tot if ss_tot > 0 else 1.0)
        fo_tau_half = float(np.log(2) / fo_k) if fo_k > 0 else np.nan
        n5 = max(1, n_td // 5)
        residuals = peak_excess - fo_pred
        resid_early = float(residuals[:n5].mean())
        resid_mid = float(residuals[n5:3 * n5].mean())
        resid_late = float(residuals[3 * n5:].mean())
    except Exception:
        pass

    # ── METRIC 4: C_peak⁰ ──
    C_peak_0 = float(peak_conc[0])

    return {
        "baseline": bl,
        "loading_min": t_start * 5,
        "peak_px": peak_px,
        "centre_lo": lo, "centre_hi": hi,
        "times": times,
        "peak_excess": peak_excess,
        "peak_conc": peak_conc,
        # Metric 1
        "tau_half": tau_half,
        "tau_half_min": tau_half / 60 if not np.isnan(tau_half) else np.nan,
        # Metric 2
        "t_handoff": t_handoff,
        "t_handoff_min": t_handoff / 60 if not np.isnan(t_handoff) else np.nan,
        "rr_at_0": rr_at_0,
        "rr_at_peak": rr_at_peak,
        "is_biphasic": is_biphasic,
        # Metric 3
        "fo_R2": fo_R2,
        "fo_k": fo_k,
        "fo_C0": fo_C0,
        "fo_tau_half": fo_tau_half,
        "fo_tau_half_min": fo_tau_half / 60 if not np.isnan(fo_tau_half) else np.nan,
        "fo_pred": fo_pred,
        "resid_early": resid_early,
        "resid_mid": resid_mid,
        "resid_late": resid_late,
        # Metric 4
        "C_peak_0": C_peak_0,
    }


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def make_patent_figure(m):
    """2x2 figure covering the 4 patent metrics."""
    t_min = m["times"] / 60
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    tau_str = f"{m['tau_half_min']:.1f}" if not np.isnan(m["tau_half_min"]) else "N/A"
    r2_str = f"{m['fo_R2']:.3f}" if not np.isnan(m["fo_R2"]) else "N/A"
    fig.suptitle(
        f"\u03c4\u00bd = {tau_str} min  |  "
        f"t_handoff = {m['t_handoff_min']:.1f} min  |  "
        f"1st-order R\u00b2 = {r2_str}  |  "
        f"C_peak\u2070 = {m['C_peak_0']:.1f} % Air Sat",
        fontsize=12, fontweight="bold",
    )

    pe = m["peak_excess"]

    # 1 - Peak excess decay + first-order fit + τ₁/₂
    ax = axes[0, 0]
    ax.plot(t_min, pe, "o", ms=3, c="crimson", label="data (centre avg)")
    if m["fo_pred"] is not None:
        ax.plot(t_min, m["fo_pred"], "-", c="navy", lw=1.5,
                label=f"1st-order fit (R\u00b2={r2_str})")
    if not np.isnan(m["tau_half_min"]):
        ax.axvline(m["tau_half_min"], color="gray", ls="--", lw=1,
                   label=f"\u03c4\u00bd = {tau_str} min")
        ax.axhline(pe[0] / 2, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("C_excess (% Air Sat)")
    ax.set_title("Metric 1: \u03c4\u00bd \u2014 delivery duration")
    ax.legend(fontsize=8)

    # 2 - Release rate curve + t_handoff
    ax = axes[0, 1]
    if len(m["times"]) >= 5:
        pe_smooth = np.convolve(pe, np.ones(3) / 3, mode="valid")
        t_smooth = m["times"][1:-1] / 60
        rr = -np.gradient(pe_smooth, m["times"][1:-1])
        ax.plot(t_smooth, rr, "o-", ms=2, lw=1, c="teal")
        if not np.isnan(m["t_handoff_min"]):
            ax.axvline(m["t_handoff_min"], color="red", ls="--", lw=1.5,
                       label=f"t_handoff = {m['t_handoff_min']:.1f} min")
            bip = "BIPHASIC" if m["is_biphasic"] else "monophasic"
            ax.text(0.95, 0.95, bip, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", ha="right", va="top",
                    color="green" if m["is_biphasic"] else "gray",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Release rate -dC_ex/dt (% Air Sat / s)")
    ax.set_title("Metric 2: t_handoff \u2014 biphasic proof")
    ax.legend(fontsize=8)

    # 3 - Residual pattern
    ax = axes[1, 0]
    if m["fo_pred"] is not None:
        residuals = pe - m["fo_pred"]
        ax.bar(t_min, residuals, width=(t_min[1] - t_min[0]) * 0.8,
               color=np.where(residuals >= 0, "steelblue", "salmon"),
               alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        # Shade early / mid / late regions
        n5 = max(1, len(t_min) // 5)
        for (a, b, lbl, clr) in [
            (0, n5, f"early: {m['resid_early']:+.1f}", "blue"),
            (n5, 3 * n5, f"mid: {m['resid_mid']:+.1f}", "orange"),
            (3 * n5, len(t_min), f"late: {m['resid_late']:+.1f}", "green"),
        ]:
            ax.axvspan(t_min[a], t_min[min(b, len(t_min) - 1)],
                       alpha=0.08, color=clr, label=lbl)
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Residual (data \u2212 fit)")
    ax.set_title("Metric 3: 1st-order residuals \u2014 model inadequacy")

    # 4 - Profile snapshots (context)
    ax = axes[1, 1]
    ax.axhline(m["baseline"], color="gray", ls=":", lw=1, label=f"baseline = {m['baseline']:.1f}")
    ax.axhline(m["C_peak_0"], color="crimson", ls=":", lw=1,
               label=f"C_peak\u2070 = {m['C_peak_0']:.1f}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Centre O\u2082 (% Air Sat)")
    ax.plot(t_min, m["peak_conc"], "o-", ms=3, lw=1, c="darkorange",
            label="centre region avg")
    ax.set_title("Metric 4: C_peak\u2070 \u2014 loading capacity")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT: SINGLE-FILE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_analysis(raw_bytes, filename):
    """Analyse one file and display the 4 patent metrics.

    Returns the metrics dict for cross-replicate comparison.
    """
    x, times_raw, data_raw = load_data_from_bytes(raw_bytes, filename)

    st.caption(
        f"{data_raw.shape[0]} pixels \u00d7 {data_raw.shape[1]} timesteps  |  "
        f"Position: {x[0]:.2f} \u2013 {x[-1]:.2f} mm"
    )

    m = compute_patent_metrics(x, data_raw, times_raw)

    st.caption(
        f"Baseline: **{m['baseline']:.1f}** % Air Sat (raw t=0 edges)  |  "
        f"Loading phase removed: first **{m['loading_min']}** min  |  "
        f"Peak tracked: centre \u00b15 px around pixel {m['peak_px']}"
    )

    # ── 4 headline cards ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "\u03c4\u00bd (delivery duration)",
        f"{m['tau_half_min']:.1f} min" if not np.isnan(m["tau_half_min"]) else "N/A",
    )
    c2.metric(
        "t_handoff (biphasic proof)",
        f"{m['t_handoff_min']:.1f} min" if not np.isnan(m["t_handoff_min"]) else "N/A",
        delta="BIPHASIC" if m["is_biphasic"] else "monophasic",
        delta_color="normal" if m["is_biphasic"] else "off",
    )
    c3.metric(
        "1st-order R\u00b2",
        f"{m['fo_R2']:.3f}" if not np.isnan(m["fo_R2"]) else "N/A",
        delta="model fails" if (not np.isnan(m["fo_R2"]) and m["fo_R2"] < 0.95) else None,
        delta_color="inverse" if (not np.isnan(m["fo_R2"]) and m["fo_R2"] < 0.95) else "off",
    )
    c4.metric(
        "C_peak\u2070 (loading)",
        f"{m['C_peak_0']:.1f} % Air Sat",
    )

    # ── MC extension comparison ──
    if not np.isnan(m["tau_half_min"]) and not np.isnan(m["fo_tau_half_min"]):
        actual = m["tau_half_min"]
        predicted = m["fo_tau_half_min"]
        if predicted > 0:
            ext = (actual / predicted - 1) * 100
            if ext > 10:
                st.success(
                    f"MC extends delivery by **{ext:.0f}%** beyond simple diffusion "
                    f"(actual \u03c4\u00bd = {actual:.1f} min vs 1st-order = {predicted:.1f} min)"
                )

    # ── Residual pattern detail ──
    if not np.isnan(m["resid_early"]):
        pattern = (
            f"early **{m['resid_early']:+.1f}** / "
            f"mid **{m['resid_mid']:+.1f}** / "
            f"late **{m['resid_late']:+.1f}**"
        )
        if m["resid_early"] > 0 and m["resid_mid"] < 0 and m["resid_late"] > 0:
            st.info(
                f"Residual pattern: {pattern} \u2014 "
                "systematic +/\u2212/+ confirms non-exponential release mechanism"
            )
        else:
            st.caption(f"Residual pattern: {pattern}")

    # ── Figure ──
    fig = make_patent_figure(m)
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download figure (PNG)", buf,
                       f"patent_metrics_{filename}.png", "image/png",
                       key=f"dl_{filename}")
    plt.close(fig)

    m["label"] = filename.rsplit(".", 1)[0]
    return m


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT: REPLICATE SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def show_replicate_summary(all_m):
    """Summary table and bar charts for replicates."""
    n = len(all_m)
    labels = [m["label"] for m in all_m]
    st.header(f"Replicate Summary (n = {n})")

    def _ms(vals, fmt=".1f"):
        clean = [v for v in vals if not np.isnan(v)]
        if not clean:
            return "N/A"
        m, s = np.mean(clean), (np.std(clean, ddof=1) if len(clean) > 1 else 0.0)
        return f"{m:{fmt}} \u00b1 {s:{fmt}}"

    # ── Per-replicate table ──
    rows = []
    for m in all_m:
        rows.append({
            "Replicate": m["label"],
            "\u03c4\u00bd (min)": f"{m['tau_half_min']:.1f}" if not np.isnan(m["tau_half_min"]) else "N/A",
            "t_handoff (min)": f"{m['t_handoff_min']:.1f}" if not np.isnan(m["t_handoff_min"]) else "N/A",
            "Biphasic": "YES" if m["is_biphasic"] else "NO",
            "1st-order R\u00b2": f"{m['fo_R2']:.4f}" if not np.isnan(m["fo_R2"]) else "N/A",
            "1st-order \u03c4\u00bd (min)": f"{m['fo_tau_half_min']:.1f}" if not np.isnan(m["fo_tau_half_min"]) else "N/A",
            "C_peak\u2070": f"{m['C_peak_0']:.1f}",
        })
    st.table(pd.DataFrame(rows))

    # ── Mean ± SD summary ──
    st.subheader("Patent-Ready Summary")
    n_bip = sum(m["is_biphasic"] for m in all_m)

    tau_vals = [m["tau_half_min"] for m in all_m if not np.isnan(m["tau_half_min"])]
    fo_tau_vals = [m["fo_tau_half_min"] for m in all_m if not np.isnan(m["fo_tau_half_min"])]
    if tau_vals and fo_tau_vals:
        ext = (np.mean(tau_vals) / np.mean(fo_tau_vals) - 1) * 100
        ext_str = f"{ext:+.0f}%"
    else:
        ext_str = "N/A"

    summary = [
        {
            "Metric": "\u03c4\u00bd (delivery duration)",
            "Value": _ms([m["tau_half_min"] for m in all_m]) + " min",
            "Patent claim": "Core claim \u2014 sustained delivery duration",
        },
        {
            "Metric": "t_handoff (biphasic proof)",
            "Value": _ms([m["t_handoff_min"] for m in all_m]) + f" min ({n_bip}/{n} biphasic)",
            "Patent claim": "Novelty \u2014 biphasic mechanism distinct from Fickian",
        },
        {
            "Metric": "1st-order R\u00b2",
            "Value": _ms([m["fo_R2"] for m in all_m], ".3f"),
            "Patent claim": "Non-obviousness \u2014 model fails, novel mechanism",
        },
        {
            "Metric": "MC delivery extension",
            "Value": f"{ext_str} vs simple diffusion",
            "Patent claim": "\u03c4\u00bd actual vs 1st-order predicted",
        },
        {
            "Metric": "C_peak\u2070 (loading capacity)",
            "Value": _ms([m["C_peak_0"] for m in all_m]) + " % Air Sat",
            "Patent claim": "Optimal MC range \u2014 upper bound where loading drops",
        },
    ]
    st.table(pd.DataFrame(summary))

    # ── Bar charts ──
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))[:n]
    x_pos = np.arange(n)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Replicate Comparison", fontsize=13, fontweight="bold")

    # τ₁/₂
    ax = axes[0]
    vals = [m["tau_half_min"] if not np.isnan(m["tau_half_min"]) else 0 for m in all_m]
    ax.bar(x_pos, vals, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("\u03c4\u00bd (min)")
    ax.set_title("\u03c4\u00bd \u2014 delivery duration")
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # t_handoff
    ax = axes[1]
    vals = [m["t_handoff_min"] if not np.isnan(m["t_handoff_min"]) else 0 for m in all_m]
    ax.bar(x_pos, vals, color=colors)
    ax.axhline(10, color="red", ls="--", lw=1, label="biphasic threshold (10 min)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("t_handoff (min)")
    ax.set_title("t_handoff \u2014 biphasic proof")
    ax.legend(fontsize=7)

    # R²
    ax = axes[2]
    vals = [m["fo_R2"] if not np.isnan(m["fo_R2"]) else 0 for m in all_m]
    ax.bar(x_pos, vals, color=colors)
    ax.axhline(0.95, color="red", ls="--", lw=1, label="adequacy threshold (0.95)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("R\u00b2")
    ax.set_title("1st-order R\u00b2 \u2014 model inadequacy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)

    # C_peak⁰
    ax = axes[3]
    vals = [m["C_peak_0"] for m in all_m]
    ax.bar(x_pos, vals, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("C_peak\u2070 (% Air Sat)")
    ax.set_title("C_peak\u2070 \u2014 loading capacity")

    plt.tight_layout()
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download comparison (PNG)", buf,
                       "replicate_comparison.png", "image/png",
                       key="dl_comparison")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
# PAGE CONFIG & MAIN
# ═════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="O\u2082 Release Kinetics \u2014 Patent Metrics",
    page_icon="\U0001f9ea",
    layout="wide",
)

st.title("\U0001f9ea Oxygen Release Kinetics \u2014 Patent Metrics")
st.markdown(
    "Upload one or more **CSV** or **Excel** files (replicates of the "
    "same condition). The app computes exactly **4 patent-critical metrics**:"
)
st.markdown(
    "1. **\u03c4\u00bd** \u2014 delivery duration (headline claim)\n"
    "2. **t_handoff** \u2014 biphasic release proof (novelty)\n"
    "3. **1st-order R\u00b2** \u2014 model inadequacy (non-obviousness)\n"
    "4. **C_peak\u2070** \u2014 loading capacity (optimal MC range)"
)

with st.expander("Expected file format"):
    st.markdown(
        "**Row 0** \u2014 timestamps (`HH:MM:SS`) starting at column 2.\n\n"
        "**Row 1** \u2014 header: `Pixel, mm, measurement_IDs...`\n\n"
        "**Rows 2+** \u2014 `pixel_index, position_mm, O2_at_t0, O2_at_t1, ...`"
    )

uploaded_files = st.file_uploader(
    "Upload replicate data files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if uploaded_files:
    all_m = []

    if len(uploaded_files) == 1:
        # Single file: show analysis directly
        raw = uploaded_files[0].getvalue()
        fname = uploaded_files[0].name
        st.header(f"Experiment: {fname}")
        try:
            m = run_analysis(raw, fname)
            all_m.append(m)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)
    else:
        # Multiple files: tabs per replicate + averaged + summary
        tab_labels = [f.name for f in uploaded_files] + [
            "\U0001f4ca Averaged (mean \u00b1 SD)", "\U0001f4cb Summary",
        ]
        tabs = st.tabs(tab_labels)

        n_files = len(uploaded_files)
        parsed_datasets = []  # (x, times, data_raw) for averaging

        for i, tab in enumerate(tabs[:n_files]):
            with tab:
                raw = uploaded_files[i].getvalue()
                fname = uploaded_files[i].name
                st.header(f"Replicate: {fname}")
                try:
                    m = run_analysis(raw, fname)
                    all_m.append(m)
                    # Also parse raw data for averaging
                    parsed_datasets.append(
                        load_data_from_bytes(raw, fname)
                    )
                except Exception as e:
                    st.error(f"Analysis failed for {fname}: {e}")
                    st.exception(e)

        # Averaged tab
        with tabs[n_files]:
            if len(parsed_datasets) >= 2:
                st.header(f"Averaged Analysis (n = {len(parsed_datasets)} replicates)")
                try:
                    x_avg, t_avg, d_mean, d_std, n_ds = compute_average_data(
                        parsed_datasets
                    )
                    st.caption(
                        f"Datasets interpolated onto common grid: "
                        f"{d_mean.shape[0]} pixels \u00d7 {d_mean.shape[1]} timesteps"
                    )

                    # Run patent metrics on the averaged data
                    m_avg = compute_patent_metrics(x_avg, d_mean, t_avg)

                    # Show 4 headline cards
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "\u03c4\u00bd (averaged)",
                        f"{m_avg['tau_half_min']:.1f} min" if not np.isnan(m_avg["tau_half_min"]) else "N/A",
                    )
                    c2.metric(
                        "t_handoff (averaged)",
                        f"{m_avg['t_handoff_min']:.1f} min" if not np.isnan(m_avg["t_handoff_min"]) else "N/A",
                        delta="BIPHASIC" if m_avg["is_biphasic"] else "monophasic",
                        delta_color="normal" if m_avg["is_biphasic"] else "off",
                    )
                    c3.metric(
                        "1st-order R\u00b2 (averaged)",
                        f"{m_avg['fo_R2']:.3f}" if not np.isnan(m_avg["fo_R2"]) else "N/A",
                    )
                    c4.metric(
                        "C_peak\u2070 (averaged)",
                        f"{m_avg['C_peak_0']:.1f} % Air Sat",
                    )

                    # Averaged profiles with ± 1 SD band
                    st.markdown("---")
                    t_start_avg = detect_diffusion_start(d_mean)
                    t_min = (t_avg[t_start_avg:] - t_avg[t_start_avg]) / 60

                    fig_avg, axes_avg = plt.subplots(1, 2, figsize=(14, 5))
                    fig_avg.suptitle(
                        f"Averaged Data (n = {n_ds} replicates, \u00b1 1 SD shading)",
                        fontsize=12, fontweight="bold",
                    )

                    # Profile snapshots with SD bands
                    ax = axes_avg[0]
                    idx = np.linspace(0, len(t_min) - 1, 6, dtype=int)
                    d_trim = d_mean[:, t_start_avg:]
                    s_trim = d_std[:, t_start_avg:]
                    for ii in idx:
                        ax.plot(x_avg, d_trim[:, ii], lw=1.2,
                                label=f"{t_min[ii]:.0f} min")
                        ax.fill_between(
                            x_avg,
                            d_trim[:, ii] - s_trim[:, ii],
                            d_trim[:, ii] + s_trim[:, ii],
                            alpha=0.12,
                        )
                    ax.set_xlabel("Position (mm)")
                    ax.set_ylabel("O\u2082 (% Air Sat)")
                    ax.set_title("Mean profiles \u00b1 SD")
                    ax.legend(fontsize=7, ncol=2)

                    # Peak decay with SD band
                    ax = axes_avg[1]
                    pe_avg = m_avg["peak_excess"]
                    # Approximate peak SD from std at peak pixel
                    bl_avg = m_avg["baseline"]
                    lo_a, hi_a = m_avg["centre_lo"], m_avg["centre_hi"]
                    excess_std_centre = d_std[lo_a:hi_a, t_start_avg:].mean(axis=0)
                    ax.plot(t_min, pe_avg, "o-", ms=3, lw=1.2, c="crimson",
                            label="mean C_excess")
                    ax.fill_between(t_min,
                                    pe_avg - excess_std_centre,
                                    pe_avg + excess_std_centre,
                                    color="crimson", alpha=0.15,
                                    label="\u00b1 1 SD")
                    if not np.isnan(m_avg["tau_half_min"]):
                        ax.axvline(m_avg["tau_half_min"], color="gray",
                                   ls="--", lw=1,
                                   label=f"\u03c4\u00bd = {m_avg['tau_half_min']:.1f} min")
                    ax.set_xlabel("Time (min)")
                    ax.set_ylabel("C_excess (% Air Sat)")
                    ax.set_title("Peak decay (mean \u00b1 SD)")
                    ax.legend(fontsize=8)

                    plt.tight_layout()
                    st.pyplot(fig_avg)
                    buf = io.BytesIO()
                    fig_avg.savefig(buf, format="png", dpi=180,
                                    bbox_inches="tight")
                    buf.seek(0)
                    st.download_button("Download averaged plots (PNG)", buf,
                                       "averaged_plots.png", "image/png",
                                       key="dl_avg")
                    plt.close(fig_avg)

                    # Full 2x2 patent figure on averaged data
                    st.markdown("---")
                    st.markdown("### Patent metrics on averaged data")
                    fig_pat = make_patent_figure(m_avg)
                    st.pyplot(fig_pat)
                    buf = io.BytesIO()
                    fig_pat.savefig(buf, format="png", dpi=180,
                                    bbox_inches="tight")
                    buf.seek(0)
                    st.download_button("Download avg. patent figure (PNG)", buf,
                                       "avg_patent_metrics.png", "image/png",
                                       key="dl_avg_pat")
                    plt.close(fig_pat)

                except Exception as e:
                    st.error(f"Averaging failed: {e}")
                    st.exception(e)
            else:
                st.warning(
                    "Need at least 2 successfully parsed datasets to average."
                )

        # Summary tab
        with tabs[n_files + 1]:
            if len(all_m) >= 2:
                show_replicate_summary(all_m)
            else:
                st.warning("Need at least 2 successful analyses for a summary.")
else:
    st.info("Upload one or more files to get started.")
