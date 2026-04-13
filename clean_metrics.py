#!/usr/bin/env python3
"""
Oxygen Release Kinetics — Cleaned Metrics
==========================================

Computes only the metrics agreed for patent/presentation:

  1. τ₁/₂           — delivery duration (headline metric)
  2. t_handoff       — biphasic proof (release rate peak time)
  3. 1st-order R²    — model inadequacy proof
  4. 1st-order τ₁/₂  — comparison to show MC extends delivery
  5. C_peak⁰         — loading capacity
  6. AUC             — total therapeutic dose
  7. L_p(t)          — penetration depth at 30, 60, 120 min
  8. T_eff           — time above therapeutic threshold
  9. FWHM₀           — initial spatial width

Fixes applied:
  - Baseline computed on RAW t=0 (before loading phase), not trimmed
  - Peak tracked using fixed centre region (10 pixels), not roaming max
  - KP model removed entirely
  - Trapezoidal integration for AUC (matches Streamlit app)

Usage:
  python clean_metrics.py file1.xlsx [file2.xlsx file3.xlsx ...]
"""

import sys
import csv
import io
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_trapz = getattr(np, "trapezoid", None) or np.trapz


def load_excel(filepath):
    """Load Excel or CSV → positions (mm), times (s), data matrix."""
    if filepath.endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath, header=None)
        rows = df.values.tolist()
    else:
        with open(filepath, encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))

    # Row 0: timestamps
    import datetime
    times = []
    for ts in rows[0][2:]:
        ts_str = str(ts).strip()
        if ts_str in ("", "nan", "None", "NaT"):
            continue
        if isinstance(ts, datetime.time):
            times.append(ts.hour * 3600 + ts.minute * 60 + ts.second)
            continue
        parts = ts_str.split(":")
        if len(parts) >= 2:
            times.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) if len(parts) == 3 else int(parts[0]) * 3600 + int(parts[1]) * 60)
        else:
            times.append(float(ts_str))
    times = np.array(times, dtype=float)
    n_t = len(times)

    # Rows 2+: data
    positions, profiles = [], []
    for row in rows[2:]:
        if len(row) < 3:
            continue
        try:
            pos = float(row[1])
        except (ValueError, TypeError):
            continue
        if np.isnan(pos):
            continue
        vals = []
        for v in row[2:2 + n_t]:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(np.nan)
        while len(vals) < n_t:
            vals.append(np.nan)
        positions.append(pos)
        profiles.append(vals)

    positions = np.array(positions)
    data = np.array(profiles)

    # Handle NaN
    if np.isnan(data).any():
        df_tmp = pd.DataFrame(data)
        df_tmp = df_tmp.ffill(axis=1).bfill(axis=1).fillna(0.0)
        data = df_tmp.values

    return positions, times, data


def analyze(filepath, label="", n_edge=15, c_threshold_excess=50.0, C_th_absolute=150.0):
    """Compute all cleaned metrics from a single file.

    Parameters
    ----------
    n_edge : int
        Number of edge pixels for baseline calculation.
    c_threshold_excess : float
        AUC threshold: excess above baseline (% Air Sat).
        Matches the Streamlit app default of 50.
    C_th_absolute : float
        Absolute threshold for T_eff (% Air Sat).
    """
    x, times_raw, data_raw = load_excel(filepath)
    dx = np.median(np.diff(x))
    n_pixels = data_raw.shape[0]
    n_timesteps = data_raw.shape[1]

    # ── FIX 1: Baseline from RAW t=0 (before any trimming) ──
    bl = 0.5 * (data_raw[:n_edge, 0].mean() + data_raw[-n_edge:, 0].mean())

    # ── Detect loading phase and trim ──
    mid = n_pixels // 2
    hw = min(20, mid)
    center_mean = data_raw[mid - hw:mid + hw, :].mean(axis=0)
    t_start = int(np.argmax(center_mean))
    loading_min = t_start * 5

    data = data_raw[:, t_start:]
    times = times_raw[t_start:] - times_raw[t_start]
    n_td = len(times)
    dt = times[1] - times[0] if n_td > 1 else 300.0

    # Excess using fixed baseline from raw t=0
    excess = data - bl

    # ── FIX 2: Fixed centre region instead of roaming max ──
    # Find peak pixel at t=0 of diffusion phase, then average ±5 pixels
    peak_px = int(np.argmax(data[:, 0]))
    half_w = 5
    region = slice(max(0, peak_px - half_w), min(n_pixels, peak_px + half_w + 1))
    region_width = min(n_pixels, peak_px + half_w + 1) - max(0, peak_px - half_w)

    # Peak excess: average of centre region at each timestep
    peak_excess = excess[region, :].mean(axis=0)
    peak_conc = data[region, :].mean(axis=0)

    # Also store roaming max for comparison
    peak_excess_roaming = excess.max(axis=0)

    # ═══════════════════════════════════════════════════════════════
    # METRIC 1: τ₁/₂ (Eq. 3)
    # ═══════════════════════════════════════════════════════════════
    target = peak_excess[0] * 0.5
    tau_half = np.nan
    for i in range(1, n_td):
        if peak_excess[i] <= target:
            frac = (target - peak_excess[i - 1]) / (peak_excess[i] - peak_excess[i - 1])
            tau_half = times[i - 1] + frac * (times[i] - times[i - 1])
            break

    # ═══════════════════════════════════════════════════════════════
    # METRIC 2: Release rate & handoff time (Eq. 4-5)
    # ═══════════════════════════════════════════════════════════════
    pe_smooth = np.convolve(peak_excess, np.ones(3) / 3, mode="valid")
    t_smooth = times[1:-1]
    release_rate = -np.gradient(pe_smooth, t_smooth)
    rr_peak_idx = int(np.argmax(release_rate))
    t_handoff = t_smooth[rr_peak_idx]
    rr_at_0 = release_rate[0]
    rr_at_peak = release_rate[rr_peak_idx]
    is_biphasic = t_handoff > 600  # >10 min after diffusion start

    # ═══════════════════════════════════════════════════════════════
    # METRIC 3: First-order fit (Eq. 6-7)
    # ═══════════════════════════════════════════════════════════════
    R2_fo = np.nan
    k1 = np.nan
    tau_half_fo = np.nan
    resid_early = np.nan
    resid_mid = np.nan
    resid_late = np.nan

    def first_order(t, C0, k):
        return C0 * np.exp(-k * t)

    try:
        popt, _ = curve_fit(first_order, times, peak_excess,
                            p0=[peak_excess[0], 1e-3],
                            bounds=([0, 0], [10 * peak_excess[0] + 1, 1.0]),
                            maxfev=5000)
        C0_fo, k1 = popt
        pred = first_order(times, *popt)
        ss_res = np.sum((peak_excess - pred) ** 2)
        ss_tot = np.sum((peak_excess - peak_excess.mean()) ** 2)
        R2_fo = 1 - ss_res / (ss_tot if ss_tot > 0 else 1)
        tau_half_fo = np.log(2) / k1

        # Residual pattern
        n5 = max(1, n_td // 5)
        residuals = peak_excess - pred
        resid_early = residuals[:n5].mean()
        resid_mid = residuals[n5:3 * n5].mean()
        resid_late = residuals[3 * n5:].mean()
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════════
    # METRIC 4: C_peak⁰ — loading capacity (Eq. 8)
    # ═══════════════════════════════════════════════════════════════
    C_peak_0 = peak_conc[0]
    C_excess_0 = peak_excess[0]

    # ═══════════════════════════════════════════════════════════════
    # METRIC 5: AUC — therapeutic dose (Eq. 4-5 from equations doc)
    # Using same method as Streamlit app:
    #   excess above (baseline + c_threshold_excess)
    #   trapezoidal integration over time
    # ═══════════════════════════════════════════════════════════════
    above_th = np.maximum(excess - c_threshold_excess, 0.0)
    auc_spatial_t = np.sum(above_th, axis=0) * dx
    auc_xt = float(_trapz(auc_spatial_t, times)) if n_td > 1 else 0.0

    # ═══════════════════════════════════════════════════════════════
    # METRIC 6: Penetration depth L_p (Eq. 9)
    # ═══════════════════════════════════════════════════════════════
    pen_th = max(0.10 * peak_excess[0], 5.0)  # 10% of initial or 5, whichever is larger
    Lp = {}
    Lp[0] = 0.0
    for t_target in [0, 30, 60, 120]:
        t_idx = int(round(t_target * 60 / dt)) if dt > 0 else 0
        t_idx = min(t_idx, n_td - 1)
        above = np.where(excess[:, t_idx] > pen_th)[0]
        if len(above) >= 2:
            Lp[t_target] = x[above[-1]] - x[above[0]]
        else:
            Lp[t_target] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # METRIC 7: T_eff — time above therapeutic threshold (Eq. 10)
    # ═══════════════════════════════════════════════════════════════
    T_eff = np.sum(peak_conc > C_th_absolute) * dt / 60  # minutes

    # ═══════════════════════════════════════════════════════════════
    # METRIC 8: FWHM at t=0
    # ═══════════════════════════════════════════════════════════════
    prof0 = excess[:, 0]
    hm = prof0.max() / 2
    above_hm = np.where(prof0 > hm)[0]
    fwhm0 = x[above_hm[-1]] - x[above_hm[0]] if len(above_hm) >= 2 else np.nan

    return {
        "label": label,
        "filepath": filepath,
        "n_pixels": n_pixels,
        "n_timesteps": n_timesteps,
        "loading_phase_min": loading_min,
        "baseline": bl,
        "baseline_source": "raw t=0 edges (fixed)",
        "peak_method": f"centre region avg ({region_width} px around pixel {peak_px})",
        # Metrics
        "C_peak_0": C_peak_0,
        "C_excess_0": C_excess_0,
        "tau_half_min": tau_half / 60 if not np.isnan(tau_half) else np.nan,
        "t_handoff_min": t_handoff / 60,
        "rr_at_0": rr_at_0,
        "rr_at_peak": rr_at_peak,
        "is_biphasic": is_biphasic,
        "R2_fo": R2_fo,
        "k1": k1,
        "tau_half_fo_min": tau_half_fo / 60 if not np.isnan(tau_half_fo) else np.nan,
        "resid_early": resid_early,
        "resid_mid": resid_mid,
        "resid_late": resid_late,
        "auc_xt": auc_xt,
        "auc_threshold": c_threshold_excess,
        "Lp_0": Lp.get(0, np.nan),
        "Lp_30": Lp.get(30, np.nan),
        "Lp_60": Lp.get(60, np.nan),
        "Lp_120": Lp.get(120, np.nan),
        "T_eff_min": T_eff,
        "T_eff_threshold": C_th_absolute,
        "fwhm0": fwhm0,
    }


def print_results(results):
    """Print results for one or more replicates."""

    label_group = results[0]["label"].rsplit(" ", 1)[0] if results else "Sample"
    n = len(results)

    print("=" * 70)
    print(f"  OXYGEN RELEASE KINETICS — {label_group} (n={n})")
    print("=" * 70)

    # Method info
    for r in results:
        print(f"\n  {r['label']}: {r['n_pixels']} px × {r['n_timesteps']} t"
              f"  |  loading removed: {r['loading_phase_min']} min"
              f"  |  baseline: {r['baseline']:.1f} ({r['baseline_source']})"
              f"  |  peak: {r['peak_method']}")

    def stat(key, fmt=".1f"):
        vals = [r[key] for r in results if not np.isnan(r[key])]
        if not vals:
            return "N/A", "N/A", []
        m = np.mean(vals)
        s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        return f"{m:{fmt}}", f"{s:{fmt}}", vals

    # ── Metric 1: τ₁/₂ ──
    print(f"\n{'─' * 70}")
    print(f"  Eq. 3: τ₁/₂ — delivery duration (HEADLINE METRIC)")
    print(f"{'─' * 70}")
    for r in results:
        print(f"  {r['label']:>8s}:  τ₁/₂ = {r['tau_half_min']:.1f} min")
    m, s, _ = stat("tau_half_min")
    print(f"  {'Mean±SD':>8s}:  {m} ± {s} min")

    # ── Metric 2: Biphasic proof ──
    print(f"\n{'─' * 70}")
    print(f"  Eq. 4-5: Release rate & handoff (BIPHASIC PROOF)")
    print(f"{'─' * 70}")
    for r in results:
        bip = "YES" if r["is_biphasic"] else "NO"
        print(f"  {r['label']:>8s}:  t_handoff = {r['t_handoff_min']:.1f} min"
              f"  |  R(0) = {r['rr_at_0']:.4f}  |  R(peak) = {r['rr_at_peak']:.4f}"
              f"  |  Biphasic: {bip}")
    n_bip = sum(r["is_biphasic"] for r in results)
    print(f"  Biphasic confirmed: {n_bip}/{n} replicates")

    # ── Metric 3: First-order fit ──
    print(f"\n{'─' * 70}")
    print(f"  Eq. 6-7: First-order fit (MODEL INADEQUACY PROOF)")
    print(f"{'─' * 70}")
    for r in results:
        resid = f"+{r['resid_early']:.1f} / {r['resid_mid']:+.1f} / +{r['resid_late']:.1f}" if not np.isnan(r["resid_early"]) else "N/A"
        fo_tau = f"{r['tau_half_fo_min']:.1f}" if not np.isnan(r["tau_half_fo_min"]) else "N/A"
        print(f"  {r['label']:>8s}:  R² = {r['R2_fo']:.4f}"
              f"  |  1st-order τ₁/₂ = {fo_tau} min"
              f"  |  Residuals: {resid}")
    m_r2, s_r2, _ = stat("R2_fo", ".4f")
    m_fo, s_fo, _ = stat("tau_half_fo_min")
    m_tau, _, _ = stat("tau_half_min")
    print(f"  {'Mean R²':>8s}:  {m_r2}")
    if m_fo != "N/A" and m_tau != "N/A":
        try:
            extension = (float(m_tau) / float(m_fo) - 1) * 100
            print(f"\n  MC extends delivery by {extension:.0f}% beyond simple diffusion")
            print(f"  (actual τ₁/₂ = {m_tau} min vs 1st-order prediction = {m_fo} min)")
        except (ValueError, ZeroDivisionError):
            pass

    # ── Metric 4: Loading capacity ──
    print(f"\n{'─' * 70}")
    print(f"  Eq. 8: Initial peak (LOADING CAPACITY)")
    print(f"{'─' * 70}")
    for r in results:
        print(f"  {r['label']:>8s}:  C_peak⁰ = {r['C_peak_0']:.1f}"
              f"  |  C_excess(0) = {r['C_excess_0']:.1f} % Air Sat")
    m, s, _ = stat("C_peak_0")
    print(f"  {'Mean±SD':>8s}:  {m} ± {s}")

    # ── Metric 5: AUC ──
    print(f"\n{'─' * 70}")
    print(f"  AUC — therapeutic dose (threshold = +{results[0]['auc_threshold']:.0f} above baseline)")
    print(f"{'─' * 70}")
    for r in results:
        print(f"  {r['label']:>8s}:  AUC = {r['auc_xt']:.1f} (% Air Sat · mm · s)")
    m, s, _ = stat("auc_xt", ".1f")
    print(f"  {'Mean±SD':>8s}:  {m} ± {s}")

    # ── Metric 6: Penetration depth ──
    print(f"\n{'─' * 70}")
    print(f"  Eq. 9: Penetration depth L_p(t)")
    print(f"{'─' * 70}")
    for r in results:
        print(f"  {r['label']:>8s}:  L_p(0) = {r['Lp_0']:.2f}"
              f"  |  L_p(30) = {r['Lp_30']:.2f}"
              f"  |  L_p(60) = {r['Lp_60']:.2f}"
              f"  |  L_p(120) = {r['Lp_120']:.2f} mm")

    # ── Metric 7: T_eff ──
    print(f"\n{'─' * 70}")
    print(f"  Eq. 10: T_eff — time above {results[0]['T_eff_threshold']:.0f}% Air Sat")
    print(f"{'─' * 70}")
    for r in results:
        print(f"  {r['label']:>8s}:  T_eff = {r['T_eff_min']:.0f} min")
    m, s, _ = stat("T_eff_min", ".0f")
    print(f"  {'Mean±SD':>8s}:  {m} ± {s} min")

    # ── Metric 8: FWHM ──
    print(f"\n{'─' * 70}")
    print(f"  FWHM at t=0 — initial spatial width")
    print(f"{'─' * 70}")
    for r in results:
        print(f"  {r['label']:>8s}:  FWHM₀ = {r['fwhm0']:.2f} mm")
    m, s, _ = stat("fwhm0", ".2f")
    print(f"  {'Mean±SD':>8s}:  {m} ± {s} mm")

    # ── Summary table ──
    print(f"\n{'═' * 70}")
    print(f"  PATENT-READY SUMMARY — {label_group} (n={n})")
    print(f"{'═' * 70}")

    metrics = [
        ("τ₁/₂", "tau_half_min", ".1f", "min"),
        ("1st-order τ₁/₂", "tau_half_fo_min", ".1f", "min"),
        ("1st-order R²", "R2_fo", ".3f", ""),
        ("Biphasic", None, None, None),
        ("C_peak⁰", "C_peak_0", ".1f", "% Air Sat"),
        ("AUC", "auc_xt", ".1f", "% Air Sat · mm · s"),
        ("L_p (t=0)", "Lp_0", ".2f", "mm"),
        ("L_p (60 min)", "Lp_60", ".2f", "mm"),
        ("T_eff", "T_eff_min", ".0f", "min"),
        ("FWHM₀", "fwhm0", ".2f", "mm"),
    ]

    for name, key, fmt, unit in metrics:
        if key is None:
            n_bip = sum(r["is_biphasic"] for r in results)
            print(f"  {name:<20s}  {n_bip}/{n} replicates confirmed")
            continue
        m, s, _ = stat(key, fmt)
        print(f"  {name:<20s}  {m} ± {s} {unit}")

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_metrics.py file1.xlsx [file2.xlsx ...]")
        print("       Files are treated as replicates of the same condition.")
        sys.exit(1)

    files = sys.argv[1:]
    results = []
    for i, fp in enumerate(files):
        label = f"Rep {i + 1}"
        print(f"  Processing {fp} ...", flush=True)
        r = analyze(fp, label=label)
        results.append(r)

    print()
    print_results(results)


if __name__ == "__main__":
    main()
