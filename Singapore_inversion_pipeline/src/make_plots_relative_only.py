from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from common import ensure_dir, load_toml, path_from_cfg
from geometric_conditioning import plot_geometric_conditioning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AX_MIN = -0.1
AX_MAX = 0.3
WEIGHT_THRESHOLDS = [0.5, 0.7, 0.9]

# Consistent colour for "estimated cable" elements
CABLE_BLUE = "#1f77b4"



# ---------------------------------------------------------------------------
# Naming helpers 
# ---------------------------------------------------------------------------

def format_location(raw_location: str) -> str:
    """'loc2_tx3' -> 'Location 2'"""
    m = re.match(r"loc(\d+)", str(raw_location))
    return f"Location {m.group(1)}" if m else str(raw_location)


def format_sweep(raw_anchor: str, fallback_index: int = 0) -> str:
    """'lfm35_45_rep1' -> 'Sweep 1'  |  'loc2_tx3_a1' -> 'Sweep 1'"""
    # anchor_id style: loc2_tx3_a1
    m = re.search(r"_a(\d+)$", str(raw_anchor))
    if m:
        return f"Sweep {m.group(1)}"
    # anchor_label style: lfm35_45_rep1
    m = re.search(r"rep(\d+)", str(raw_anchor))
    if m:
        return f"Sweep {m.group(1)}"
    return f"Sweep {fallback_index + 1}"


def format_anchor_label(anchor_id: str) -> str:
    """
    Convert internal anchor_id to 'Location N, Sweep N'.
    Handles both 'loc2_tx3_a1' and legacy formats.
    """
    if anchor_id is None:
        return "Unknown"
    text = str(anchor_id)
    # loc2_tx3_a1  ->  Location 2, Sweep 1
    m = re.match(r"loc(\d+).*_a(\d+)$", text)
    if m:
        return f"Location {m.group(1)}, Sweep {m.group(2)}"
    return text


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def ensure_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str).str.strip().str.upper()
        .map({"TRUE": True, "FALSE": False})
        .fillna(False)
    )


def cumulative_arclength(x, y):
    ds = np.sqrt(np.diff(np.asarray(x, float)) ** 2 + np.diff(np.asarray(y, float)) ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def latlon_to_local_xy_fixed_origin(lat_deg, lon_deg, lat0_deg, lon0_deg):
    R = 6371000.0
    lat  = np.radians(np.asarray(lat_deg, float))
    lon  = np.radians(np.asarray(lon_deg, float))
    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    return (lon - lon0) * np.cos(lat0) * R, (lat - lat0) * R


def moving_average(arr, window=31):
    arr = np.asarray(arr, dtype=float)
    if window <= 1:
        return arr.copy()
    return np.convolve(arr, np.ones(window) / window, mode="same")


def weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    values  = np.asarray(values,  dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.average(values[mask] ** 2, weights=weights[mask])))


def add_channel_labels_plan(ax, df: pd.DataFrame, label_every: int = 100) -> None:
    mask = pd.to_numeric(df["channel"], errors="coerce") % label_every == 0
    for _, row in df.loc[mask].iterrows():
        ax.text(float(row["x_m"]), float(row["y_m"]), str(int(row["channel"])),
                fontsize=7, color="0.25", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.65), zorder=10)


def add_channel_labels_depth(ax, df: pd.DataFrame, label_every: int = 100) -> None:
    mask = pd.to_numeric(df["channel"], errors="coerce") % label_every == 0
    for _, row in df.loc[mask].iterrows():
        ax.text(float(row["channel"]), float(row["z_m"]), str(int(row["channel"])),
                fontsize=7, color="0.25", ha="center", va="bottom")


def weight_mask(obs: pd.DataFrame, min_weight: float) -> np.ndarray:
    mask = np.ones(len(obs), dtype=bool)
    if "use_observation" in obs.columns:
        mask &= ensure_bool(obs["use_observation"]).to_numpy(dtype=bool)
    if "weight" in obs.columns:
        w = pd.to_numeric(obs["weight"], errors="coerce").fillna(-np.inf).to_numpy()
        mask &= w >= float(min_weight)
    return mask


def project_points_onto_polyline(px, py, x_line, y_line, s_line=None):
    px, py = np.asarray(px, float), np.asarray(py, float)
    x_line, y_line = np.asarray(x_line, float), np.asarray(y_line, float)
    if s_line is None:
        s_line = cumulative_arclength(x_line, y_line)
    dist   = np.full(len(px), np.inf)
    proj_x = np.full(len(px), np.nan)
    proj_y = np.full(len(px), np.nan)
    proj_s = np.full(len(px), np.nan)
    for i in range(len(x_line) - 1):
        dx = x_line[i+1] - x_line[i]; dy = y_line[i+1] - y_line[i]
        sl2 = dx*dx + dy*dy
        if sl2 == 0:
            t = np.zeros_like(px); qx = np.full_like(px, x_line[i]); qy = np.full_like(py, y_line[i]); sl = 0.0
        else:
            t = np.clip(((px - x_line[i])*dx + (py - y_line[i])*dy) / sl2, 0, 1)
            qx = x_line[i] + t*dx; qy = y_line[i] + t*dy; sl = np.sqrt(sl2)
        d = np.sqrt((px-qx)**2 + (py-qy)**2)
        m = d < dist
        dist[m] = d[m]; proj_x[m] = qx[m]; proj_y[m] = qy[m]
        proj_s[m] = s_line[i] + t[m]*sl
    return dist, proj_x, proj_y, proj_s


def fit_has_absolute_columns(fit: pd.DataFrame) -> bool:
    return {"residual_abs_prior_s", "residual_abs_opt_s"}.issubset(set(fit.columns))


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def build_tx_table(raw: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    tx_source = fit.copy()
    if not {"tx_x_m", "tx_y_m", "tx_u_m"}.issubset(tx_source.columns):
        tx_source = raw.copy()
    if "anchor_id" not in tx_source.columns:
        if {"location", "anchor_index"}.issubset(tx_source.columns):
            tx_source["anchor_id"] = (
                tx_source["location"].astype(str) + "_a" + tx_source["anchor_index"].astype(str)
            )
        else:
            raise KeyError("Could not build anchor_id.")
    tx_tbl = (
        tx_source.groupby("anchor_id")[["tx_x_m", "tx_y_m", "tx_u_m"]]
        .first().reset_index()
    )
    tx_tbl["anchor_label"] = tx_tbl["anchor_id"].map(format_anchor_label)
    return tx_tbl


def load_truth(truth_csv: Path, lat0: float, lon0: float) -> pd.DataFrame:
    truth = pd.read_csv(truth_csv)
    if not {"lat", "lon"}.issubset(truth.columns):
        raise ValueError("Truth CSV missing lat/lon columns.")
    x, y = latlon_to_local_xy_fixed_origin(truth["lat"].values, truth["lon"].values, lat0, lon0)
    out = truth.copy()
    out["x_m"] = x; out["y_m"] = y
    if "z" not in out.columns and "depth" in out.columns:
        out["z"] = pd.to_numeric(out["depth"], errors="coerce")
    for c in ["ch", "channel", "Channel", "CHAN", "chan"]:
        if c in out.columns:
            out["channel_like"] = pd.to_numeric(out[c], errors="coerce")
            break
    else:
        out["channel_like"] = np.arange(len(out), dtype=float)
    return out


def load_inputs(config_path, input_csv, inversion_output_dir, truth_csv):
    cfg = load_toml(config_path)
    input_csv            = input_csv            or (path_from_cfg(cfg, "inversion_dataset_output_dir") / "inversion_observations.csv")
    inversion_output_dir = inversion_output_dir or path_from_cfg(cfg, "inversion_output_dir")
    truth_csv            = truth_csv            or path_from_cfg(cfg, "cable_estimate_csv")

    layout = pd.read_csv(inversion_output_dir / "updated_cable_layout.csv")
    ctrl   = pd.read_csv(inversion_output_dir / "control_points_optimized.csv")
    fit    = pd.read_csv(inversion_output_dir / "observation_fit_diagnostics.csv")
    raw    = pd.read_csv(input_csv)
    q = (
        pd.read_csv(inversion_output_dir / "channel_control_quality.csv")
        if (inversion_output_dir / "channel_control_quality.csv").exists()
        else None
    )
    lat0 = float(raw["enu_origin_lat_deg"].dropna().iloc[0])
    lon0 = float(raw["enu_origin_lon_deg"].dropna().iloc[0])
    h0   = float(raw["enu_origin_h_m"].dropna().iloc[0]) if "enu_origin_h_m" in raw.columns else 0.0
    truth  = load_truth(truth_csv, lat0, lon0)
    tx_tbl = build_tx_table(raw, fit)
    return cfg, raw, layout, ctrl, fit, q, truth, tx_tbl, (lat0, lon0, h0), input_csv, inversion_output_dir, truth_csv


# ---------------------------------------------------------------------------
# Uncertainty tube
# Tube half-width = weighted std of per-transmission residuals, converted to
# metres via sound speed, then Gaussian-smoothed.  Computed only from
# observations with weight >= min_weight_for_tube to exclude noisy detections
# that inflate the spread without contributing to the inversion.
# ---------------------------------------------------------------------------

def build_uncertainty_tube(
    layout: pd.DataFrame,
    fit: pd.DataFrame,
    sound_speed: float = 1500.0,
    min_weight_for_tube: float = 0.5,
    gaussian_sigma: float = 30.0,
) -> pd.DataFrame:
    fit = fit.copy()
    fit["res_s"]  = pd.to_numeric(fit["residual_dt_ref_opt_s"], errors="coerce")
    fit["weight"] = pd.to_numeric(fit["weight"],                errors="coerce")
    ch_col = "channel_eff" if "channel_eff" in fit.columns else "channel"

    # Only use observations that actually entered the inversion
    fit_hc = fit[
        np.isfinite(fit["weight"]) & (fit["weight"] >= min_weight_for_tube)
    ].copy()

    def ch_weighted_std(group: pd.DataFrame) -> float:
        w = group["weight"].to_numpy(dtype=float)
        r = group["res_s"].to_numpy(dtype=float)
        mask = np.isfinite(w) & np.isfinite(r) & (w >= min_weight_for_tube)
        if np.sum(mask) < 3:          # need at least 3 observations to estimate spread
            return np.nan
        w = w[mask]; r = r[mask]
        w = w / w.sum()
        mean_r = np.sum(w * r)
        return float(np.sqrt(np.sum(w * (r - mean_r) ** 2)))

    rows = [
        {"channel": ch_val, "std_s": ch_weighted_std(grp)}
        for ch_val, grp in fit_hc.groupby(ch_col)
    ]
    ch_stats = pd.DataFrame(rows)
    ch_stats["uncertainty_m"] = float(sound_speed) * ch_stats["std_s"]

    out = layout.merge(ch_stats, on="channel", how="left").copy()
    out["uncertainty_m"] = (
        pd.to_numeric(out["uncertainty_m"], errors="coerce")
        .interpolate().bfill().ffill()
    )
    out["uncertainty_smooth_m"] = gaussian_filter1d(
        out["uncertainty_m"].to_numpy(dtype=float), sigma=gaussian_sigma
    )
    # Physical floor: tube never thinner than 1 m half-width
    out["tube_half_width_m"] = np.maximum(out["uncertainty_smooth_m"], 1.0)
    return out


def compute_tube_boundaries(df: pd.DataFrame):
    x = pd.to_numeric(df["x_m"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y_m"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df["tube_half_width_m"], errors="coerce").to_numpy(dtype=float)
    dx = np.gradient(x); dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-8
    nx = -dy / norm; ny = dx / norm
    return x, y, x + nx*w, y + ny*w, x - nx*w, y - ny*w


def build_residual_envelope(fit: pd.DataFrame, min_weight: float = 0.0) -> pd.DataFrame:
    tmp = fit.copy()
    if min_weight > 0:
        tmp = tmp.loc[weight_mask(tmp, min_weight=min_weight)].copy()
    ch_col = "channel_eff" if "channel_eff" in tmp.columns else "channel"
    tmp["rel_res_ms"] = 1000.0 * pd.to_numeric(tmp["residual_dt_ref_opt_s"], errors="coerce")
    rows = []
    for ch_val, grp in tmp.groupby(ch_col):
        vals = pd.to_numeric(grp["rel_res_ms"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append({"channel": ch_val, "p10": np.percentile(vals, 10),
                     "p50": np.percentile(vals, 50), "p90": np.percentile(vals, 90)})
    env = pd.DataFrame(rows).sort_values("channel")
    for c in ["p10", "p50", "p90"]:
        env[c] = moving_average(env[c].to_numpy(dtype=float), window=31)
    return env


def _prepare_anchor_data(fit: pd.DataFrame, min_weight: float) -> pd.DataFrame:
    hc = fit.loc[weight_mask(fit, min_weight=min_weight)].copy()
    if len(hc) == 0:
        return hc
    if "anchor_id" not in hc.columns:
        if {"location", "anchor_index"}.issubset(hc.columns):
            hc["anchor_id"] = hc["location"].astype(str) + "_a" + hc["anchor_index"].astype(str)
        else:
            hc["anchor_id"] = "all"
    hc["anchor_label"] = hc["anchor_id"].map(format_anchor_label)
    return hc


def _plot_obs_pred_panel(ax, x, y, w, title, ylabel, rmse_ms):
    ax.scatter(x, y, s=10 + 14 * np.clip(w, 0, 1), alpha=0.6, color=CABLE_BLUE)
    ax.plot([AX_MIN, AX_MAX], [AX_MIN, AX_MAX], linewidth=1.0, color="0.4")
    ax.set_xlim(AX_MIN, AX_MAX); ax.set_ylim(AX_MIN, AX_MAX)
    ax.set_title(title)
    ax.set_xlabel("$\Delta t^{\mathrm{obs}}$ (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.text(0.04, 0.92, f"weighted RMSE = {rmse_ms:.1f} ms",
            transform=ax.transAxes, fontsize=11, va="top")


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def apply_thesis_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12,
        "legend.fontsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })


# ---------------------------------------------------------------------------
# Plot 1: S_k with control points
# ---------------------------------------------------------------------------

def plot_sk_with_control_points(q, ctrl, out_png, min_separation, max_gap):
    if q is None or "S_k" not in q.columns:
        warnings.warn("No S_k data available.")
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(q["channel"], q["S_k"], linewidth=1.8, color=CABLE_BLUE, label="$S_k$ (sum of weights)")
    if len(ctrl) > 0:
        sk_at_ctrl = np.interp(ctrl["channel"], q["channel"], q["S_k"])
        ax.scatter(ctrl["channel"], sk_at_ctrl, s=30, color=CABLE_BLUE, zorder=4,
                   label=f"Control points (n={len(ctrl)})")
        threshold = 0.6*12
        ax.axhline(threshold, linestyle="--", linewidth=1.0, color=CABLE_BLUE, alpha=0.5,
                   label=f"Control quality treshold $S_k$ = {threshold:.2f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Channel"); ax.set_ylabel("$S_k$ (sum of observation weights)")
    ax.set_title("Effective observation count $S_k$ with selected control points")
    ax.grid(True, alpha=0.25)
    ax.text(0.01, 0.97, f"min separation: {min_separation} ch  |  max gap: {max_gap} ch",
            transform=ax.transAxes, fontsize=12, va="top", color="0.35")
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Plan view with uncertainty tube
# ---------------------------------------------------------------------------

def plot_plan_view_with_tube(layout, fit, truth, tx_tbl, out_png, label_every, sound_speed):
    df = build_uncertainty_tube(layout, fit, sound_speed=sound_speed)
    x, y, x_up, y_up, x_dn, y_dn = compute_tube_boundaries(df)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_betweenx(y, x_dn, x_up, alpha=0.22, color="#4C92C3",
                     label="Residual uncertainty tube", zorder=1)
    ax.plot(pd.to_numeric(layout["prior_x_m"], errors="coerce"),
            pd.to_numeric(layout["prior_y_m"], errors="coerce"),
            linewidth=1.8, alpha=0.6, color="0.45", label="Prior cable", zorder=2)
    ax.plot(truth["x_m"].values, truth["y_m"].values,
            linewidth=2.0, color="#ff7f0e", label="Reference", zorder=3)
    ax.plot(x, y, linewidth=2.4, color=CABLE_BLUE, label="Estimated cable", zorder=4)
    ax.scatter(tx_tbl["tx_x_m"].values, tx_tbl["tx_y_m"].values,
               marker="x", s=80, linewidths=2.0, color="#2ca02c", label="Transmitters", zorder=5)
    seen = set()
    for _, row in tx_tbl.iterrows():
        # Use only the location part of the label
        label = str(row["anchor_label"]).split(",")[0].strip()
        if label in seen:
            continue
        seen.add(label)
        ax.text(float(row["tx_x_m"]) + 3.0, float(row["tx_y_m"]) + 3.0,
                label, fontsize=8, weight="bold", color="0.20", zorder=6)
    
    # for _, row in tx_tbl.iterrows():
    #     ax.text(float(row["tx_x_m"]) + 3.0, float(row["tx_y_m"]) + 3.0,
    #             str(row["anchor_label"]), fontsize=8, weight="bold", color="0.20", zorder=6)
    add_channel_labels_plan(ax, df[["channel", "x_m", "y_m"]], label_every=label_every)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Plan view: estimated cable layout with uncertainty tube")
    ax.axis("equal"); ax.grid(True, alpha=0.25); ax.legend(frameon=True, loc="best")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)



def plot_plan_view_with_tube_show(
    layout,
    fit,
    truth,
    tx_tbl,
    sound_speed,
):
    df = build_uncertainty_tube(layout, fit, sound_speed=sound_speed)
    x, y, x_up, y_up, x_dn, y_dn = compute_tube_boundaries(df)

    fig, ax = plt.subplots(figsize=(10, 8))

    # uncertainty tube
    ax.fill_betweenx(
        y,
        x_dn,
        x_up,
        alpha=0.22,
        color="#4C92C3",
        zorder=1,
    )

    # prior cable
    ax.plot(
        pd.to_numeric(layout["prior_x_m"], errors="coerce"),
        pd.to_numeric(layout["prior_y_m"], errors="coerce"),
        linewidth=1.8,
        alpha=0.6,
        color="0.45",
        zorder=2,
    )

    # reference
    ax.plot(
        truth["x_m"].values,
        truth["y_m"].values,
        linewidth=2.0,
        color="#ff7f0e",
        zorder=3,
    )

    # estimated cable
    ax.plot(
        x,
        y,
        linewidth=2.4,
        color=CABLE_BLUE,
        zorder=4,
    )

    # transmitters
    ax.scatter(
        tx_tbl["tx_x_m"].values,
        tx_tbl["tx_y_m"].values,
        marker="x",
        s=80,
        linewidths=2.0,
        color="#2ca02c",
        zorder=5,
    )

    # no labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # no channel labels
    # no transmitter text labels
    # no legend

    ax.axis("equal")
    ax.grid(False)

    # remove ticks too
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

    return fig, ax



def plot_tube_vs_channel(
    layout: pd.DataFrame,
    fit: pd.DataFrame,
    out_png,
    sound_speed: float = 1500.0,
    min_weight_for_tube: float = 0.5,
    gaussian_sigma: float = 30.0,
) -> None:

    tube = build_uncertainty_tube(
        layout, fit,
        sound_speed=sound_speed,
        min_weight_for_tube=min_weight_for_tube,
        gaussian_sigma=gaussian_sigma,
    )

    ch  = tube["channel"].to_numpy(dtype=float)
    hw  = tube["tube_half_width_m"].to_numpy(dtype=float)   # half-width
    full_width = 2.0 * hw                                    # total width

    # Key statistics
    idx_max     = int(np.argmax(hw))
    ch_max      = float(ch[idx_max])
    hw_max      = float(hw[idx_max])
    hw_min      = float(np.min(hw))
    hw_mean     = float(np.mean(hw))

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # Filled area showing full tube width (±half-width)
    ax.fill_between(ch, -hw, hw, alpha=0.30, color="#4C92C3",
                    label="Tube (±half-width)")
    ax.plot(ch,  hw, linewidth=1.6, color=CABLE_BLUE)
    ax.plot(ch, -hw, linewidth=1.6, color=CABLE_BLUE)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")

    # Mark the widest point
    ax.axvline(ch_max, color="#C62828", linewidth=1.0, linestyle=":",
               label=f"Widest point  (ch {int(ch_max)},  ±{hw_max:.1f} m)")
    ax.annotate(
        f"max ±{hw_max:.1f} m",
        xy=(ch_max, hw_max),
        xytext=(ch_max + 30, hw_max + 0.5),
        fontsize=8, color="#C62828",
        arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.8),
    )

    # Annotation box with key numbers
    stats_text = (
        f"Min half-width:  {hw_min:.1f} m \n"
        f"Mean half-width: {hw_mean:.1f} m\n"
        f"Max half-width:  {hw_max:.1f} m\n"
        f"weight $\\geq$ {min_weight_for_tube:.1f},  "
        f"$c$ = {sound_speed:.0f} m/s"
    )
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes, fontsize=11, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9),
    )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Half-width (m)")
    ax.set_title(
        f"Uncertainty tube half-width along cable  "
        f"(weight $\\geq$ {min_weight_for_tube:.1f}, "
        f"Gaussian $\\sigma$ = {gaussian_sigma:.0f} ch)"
    )
    ax.legend(frameon=True, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=-(hw_max * 1.3))   # show negative mirror for context

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Plan view with control points — control points same colour as cable
# ---------------------------------------------------------------------------

def plot_plan_view_with_control_points(layout, truth, ctrl, tx_tbl, out_png, label_every):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(pd.to_numeric(layout["prior_x_m"], errors="coerce"),
            pd.to_numeric(layout["prior_y_m"], errors="coerce"),
            linewidth=1.8, alpha=0.6, color="0.45", label="Prior cable", zorder=1)
    ax.plot(pd.to_numeric(layout["x_m"], errors="coerce"),
            pd.to_numeric(layout["y_m"], errors="coerce"),
            linewidth=2.4, color=CABLE_BLUE, label="Estimated cable", zorder=3)
    # Control points in the same blue as the cable — they are part of it
    ax.scatter(pd.to_numeric(ctrl["x_m"], errors="coerce"),
               pd.to_numeric(ctrl["y_m"], errors="coerce"),
               s=20, color=CABLE_BLUE, zorder=5,
               label=f"Control points (n={len(ctrl)})")    
    ax.scatter(tx_tbl["tx_x_m"].values, tx_tbl["tx_y_m"].values,
               marker="x", s=80, linewidths=2.0, color="#2ca02c", label="Transmitters", zorder=5)
    seen = set()
    for _, row in tx_tbl.iterrows():
        # Use only the location part of the label
        label = str(row["anchor_label"]).split(",")[0].strip()
        if label in seen:
            continue
        seen.add(label)
        ax.text(float(row["tx_x_m"]) + 3.0, float(row["tx_y_m"]) + 3.0,
                label, fontsize=8, weight="bold", color="0.20", zorder=6)
    
    add_channel_labels_plan(ax, layout[["channel", "x_m", "y_m"]], label_every=label_every)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Plan view: prior and estimated cable with control points")
    ax.axis("equal"); ax.grid(True, alpha=0.25); ax.legend(frameon=True, loc="best")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Depth profile — control points same colour as cable
# ---------------------------------------------------------------------------

def plot_depth_profile(layout, truth, ctrl, out_png, label_every):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.plot(layout["channel"], layout["prior_z_m"], linewidth=1.7, alpha=0.65,
            color="0.45", label="Prior z")
    ax.plot(layout["channel"], layout["z_m"], linewidth=2.4, color=CABLE_BLUE, label="Estimated")
    ax.scatter(ctrl["channel"], ctrl["z_m"], s=20, color=CABLE_BLUE, zorder=3,
               label="Control points")
    if "channel_like" in truth.columns and "z" in truth.columns:
        ax.plot(truth["channel_like"], pd.to_numeric(truth["z"], errors="coerce"),
                linewidth=2.0, color="#ff7f0e", label="Reference")
    add_channel_labels_depth(ax, layout[["channel", "z_m"]], label_every=label_every)
    ax.set_xlabel("Channel"); ax.set_ylabel("Up (m)")
    ax.set_title("Depth profile")
    ax.grid(True, alpha=0.25); ax.legend(frameon=True, loc="best")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Depth profile with control points (alias kept for main())
# ---------------------------------------------------------------------------

def plot_depth_profile_with_control_points(layout, truth, ctrl, out_png, label_every):
        fig, ax = plt.subplots(figsize=(13, 5.6))
        ax.plot(layout["channel"], layout["prior_z_m"], linewidth=1.7, alpha=0.65,
                color="0.45", label="Prior z")
        ax.plot(layout["channel"], layout["z_m"], linewidth=2.4, color=CABLE_BLUE, label="Estimated")
        ax.scatter(ctrl["channel"], ctrl["z_m"], s=20, color=CABLE_BLUE, zorder=3,
                label="Control points")
        add_channel_labels_depth(ax, layout[["channel", "z_m"]], label_every=label_every)
        ax.set_xlabel("Channel"); ax.set_ylabel("Up (m)")
        ax.set_title("Depth profile")
        ax.grid(True, alpha=0.25); ax.legend(frameon=True, loc="best")
        fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 6: Residual histograms at three weight thresholds
# ---------------------------------------------------------------------------

def plot_residual_histograms_three_thresholds(fit, out_dir, thresholds=WEIGHT_THRESHOLDS):
    for thresh in thresholds:
        hc = fit.loc[weight_mask(fit, min_weight=thresh)].copy()
        if len(hc) == 0:
            warnings.warn(f"No observations with weight >= {thresh} for histogram.")
            continue
        rel_prior = 1000.0 * pd.to_numeric(hc["residual_dt_ref_prior_s"], errors="coerce")
        rel_opt   = 1000.0 * pd.to_numeric(hc["residual_dt_ref_opt_s"],   errors="coerce")
        rmse_prior = 1000.0 * np.sqrt(np.nanmean(pd.to_numeric(hc["residual_dt_ref_prior_s"], errors="coerce") ** 2))
        rmse_opt   = 1000.0 * np.sqrt(np.nanmean(pd.to_numeric(hc["residual_dt_ref_opt_s"],   errors="coerce") ** 2))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(rel_prior[np.isfinite(rel_prior)], bins=80, alpha=0.50, color="0.45",
                label=f"Prior  (RMSE = {rmse_prior:.3f} ms)")
        ax.hist(rel_opt[np.isfinite(rel_opt)],     bins=80, alpha=0.50, color=CABLE_BLUE,
                label=f"Inverted  (RMSE = {rmse_opt:.3f} ms)")
        ax.set_xlabel("Relative residual (ms)"); ax.set_ylabel("Count")
        ax.set_title(f"Relative-time residuals — weight $\\geq$ {thresh:.3f}  (n = {len(hc)})")
        ax.legend(frameon=True); ax.grid(True, alpha=0.20)
        ax.set_xlim(-120,120)
        fig.tight_layout()
        fig.savefig(out_dir / f"figure_rel_residual_hist_w{str(thresh).replace('.','p')}.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: Obs vs predicted 3×3 grids — panel titles use Location N, Sweep N
# ---------------------------------------------------------------------------

def _obs_pred_3x3_grid(fit, min_weight, use_prior, out_png):
    hc = _prepare_anchor_data(fit, min_weight=min_weight)
    if len(hc) == 0:
        warnings.warn(f"No observations with weight >= {min_weight}.")
        return
    pred_col     = "predicted_dt_ref_s_prior" if use_prior else "predicted_dt_ref_s_opt"
    label_suffix = "prior geometry" if use_prior else "after inversion"
    ylabel       = "$\Delta t^{\mathrm{pred}}$ (s)"    if use_prior else "$\Delta t^{\mathrm{pred}}$ (s)"

    anchor_ids = sorted(hc["anchor_id"].dropna().astype(str).unique())
    ncol = 4; nrow = int(np.ceil(len(anchor_ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0*ncol, 4.0*nrow), squeeze=False)
    fig.suptitle(
        f"Observed vs predicted relative arrival times — {label_suffix}\n"
        f"weight $\\geq$ {min_weight:.1f}", y=0.995, fontsize= 20,
    )
    for ax, aid in zip(axes.ravel(), anchor_ids):
        m     = hc["anchor_id"].astype(str).eq(aid)
        title = hc.loc[m, "anchor_label"].iloc[0]   # already "Location N, Sweep N"
        x = pd.to_numeric(hc.loc[m, "observed_dt_ref_s"], errors="coerce").values
        y = pd.to_numeric(hc.loc[m, pred_col],            errors="coerce").values
        w = pd.to_numeric(hc.loc[m, "weight"],            errors="coerce").fillna(0.0).values
        keep = np.isfinite(x) & np.isfinite(y)
        x, y, w = x[keep], y[keep], w[keep]
        if len(x) == 0:
            ax.set_title(title); ax.axis("off"); continue
        _plot_obs_pred_panel(ax, x, y, w, title, ylabel, 1000.0 * weighted_rmse(y - x, w))
    for ax in axes.ravel()[len(anchor_ids):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def plot_obs_pred_grids_three_thresholds(fit, out_dir, thresholds=WEIGHT_THRESHOLDS):
    for thresh in thresholds:
        lbl = str(thresh).replace(".", "p")
        _obs_pred_3x3_grid(fit, min_weight=thresh, use_prior=True,
                           out_png=out_dir / f"figure_obs_pred_prior_w{lbl}.png")
        _obs_pred_3x3_grid(fit, min_weight=thresh, use_prior=False,
                           out_png=out_dir / f"figure_obs_pred_inverted_w{lbl}.png")


# ---------------------------------------------------------------------------
# Plot 8: Residual vs channel — only observations that entered the inversion
# ---------------------------------------------------------------------------

def plot_residual_vs_channel(fit, out_png, min_weight=0.5):
    """
    Shows only observations with use_observation=True and weight >= min_weight,
    i.e. exactly the subset that the inversion actually used.  Colouring by
    weight highlights which observations were most influential.
    """
    hc = fit.loc[weight_mask(fit, min_weight=min_weight)].copy()
    if len(hc) == 0:
        warnings.warn(f"No observations with weight >= {min_weight}.")
        return

    ch_col = "channel_eff" if "channel_eff" in hc.columns else "channel"
    w             = pd.to_numeric(hc["weight"],                   errors="coerce").fillna(0.0).to_numpy()
    rel_prior_ms  = 1000.0 * pd.to_numeric(hc["residual_dt_ref_prior_s"], errors="coerce").to_numpy()
    rel_opt_ms    = 1000.0 * pd.to_numeric(hc["residual_dt_ref_opt_s"],   errors="coerce").to_numpy()
    ch            = pd.to_numeric(hc[ch_col], errors="coerce").to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    sc0 = axes[0].scatter(ch, rel_prior_ms, c=w, s=3, alpha=1, cmap="viridis",
                          vmin=0, vmax=1.0, rasterized=True)
    plt.colorbar(sc0, ax=axes[0], label="weight")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_ylabel("Relative residual (ms)")
    axes[0].set_title(f"Prior geometry  (weight $\\geq$ {min_weight:.1f})")
    axes[0].grid(True, alpha=0.25)

    sc1 = axes[1].scatter(ch, rel_opt_ms, c=w, s=3, alpha=1, cmap="viridis",
                          vmin=0, vmax=1.0, rasterized=True)
    plt.colorbar(sc1, ax=axes[1], label="weight")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Relative residual (ms)")
    axes[1].set_xlabel("Channel")
    axes[1].set_title(f"After inversion  (weight $\\geq$ {min_weight:.1f})")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Relative residual as a function of channel (prior vs inverted)", fontsize=13)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 9: Optimiser convergence
# ---------------------------------------------------------------------------

def plot_optimizer_convergence_from_history(inversion_output_dir, out_png):
    hist_csv = inversion_output_dir / "optimizer_history.csv"
    if not hist_csv.exists():
        warnings.warn(f"optimizer_history.csv not found in {inversion_output_dir}.")
        return
    hist  = pd.read_csv(hist_csv)
    evals = hist["eval"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(evals, hist["cost_total"], linewidth=1.8, label="Total cost")
    if "cost_best_so_far" in hist.columns:
        axes[0].plot(evals, hist["cost_best_so_far"], linestyle="--", linewidth=1.4,
                     color="#ff7f0e", label="Best so far")
    axes[0].set_ylabel("Total cost"); axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3); axes[0].legend(); axes[0].set_title("Optimiser convergence")

    cost_cols = {
        "cost_rel":     "Relative traveltime",
        "cost_prior":   "Prior penalty",
        "cost_curv":    "Curvature penalty",
        "cost_spacing": "Spacing penalty",
        "cost_abs":     "Absolute traveltime",
        "cost_bias":    "Bias penalty",
    }
    for col, lbl in cost_cols.items():
        if col in hist.columns:
            axes[1].plot(evals, hist[col], label=lbl, linewidth=1.4)
    axes[1].set_ylabel("Block cost"); axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3); axes[1].legend(ncol=2)

    if "step_norm"  in hist.columns:
        axes[2].plot(evals, hist["step_norm"],  linewidth=1.4, label="Step norm")
    if "param_norm" in hist.columns:
        axes[2].plot(evals, hist["param_norm"], linewidth=1.4, label="Parameter norm")
    axes[2].set_ylabel("Norm"); axes[2].set_xlabel("Function evaluation")
    axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3); axes[2].legend()

    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)



def plot_angular_diversity(fit: pd.DataFrame, layout: pd.DataFrame, out_png: Path) -> None:
    """
    For each channel, compute the circular standard deviation of the bearing
    angles from the estimated cable position to each transmitter that observed
    it.  High angular spread = good geometric diversity = well-constrained
    lateral position.  Low angular spread = all transmitters approached from
    the same direction = poorly constrained laterally.
    """
    ch_col = "channel_eff" if "channel_eff" in fit.columns else "channel"

    # Index estimated cable positions by channel
    cable_xy = layout.set_index("channel")[["x_m", "y_m"]]

    rows = []
    for ch_val, grp in fit.groupby(ch_col):
        if int(ch_val) not in cable_xy.index:
            continue
        cx = float(cable_xy.loc[int(ch_val), "x_m"])
        cy = float(cable_xy.loc[int(ch_val), "y_m"])

        tx_x = pd.to_numeric(grp["tx_x_m"], errors="coerce").to_numpy(dtype=float)
        tx_y = pd.to_numeric(grp["tx_y_m"], errors="coerce").to_numpy(dtype=float)
        w    = pd.to_numeric(grp["weight"],  errors="coerce").fillna(0).to_numpy(dtype=float)
        mask = np.isfinite(tx_x) & np.isfinite(tx_y) & (w > 0)

        if np.sum(mask) < 2:
            rows.append({"channel": ch_val, "angular_spread_deg": 0.0,
                         "n_tx": int(np.sum(mask))})
            continue

        dx = tx_x[mask] - cx
        dy = tx_y[mask] - cy
        angles = np.arctan2(dy, dx)

        # Circular standard deviation
        S = np.mean(np.sin(angles))
        C = np.mean(np.cos(angles))
        R = np.sqrt(S**2 + C**2)
        circ_std = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-9)))))

        rows.append({"channel": ch_val, "angular_spread_deg": circ_std,
                     "n_tx": int(np.sum(mask))})

    div = pd.DataFrame(rows).sort_values("channel")
    if div.empty:
        warnings.warn("No angular diversity data to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(div["channel"], div["angular_spread_deg"],
                 linewidth=1.8, color=CABLE_BLUE)
    axes[0].set_ylabel("Angular spread (°, circular std)")
    axes[0].set_title("Angular diversity per channel")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(div["channel"], div["n_tx"],
                 linewidth=1.4, color=CABLE_BLUE)
    axes[1].set_ylabel("Number of observations")
    axes[1].set_xlabel("Channel")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)






def plot_horizontal_shift(layout, out_png):
    fig, ax = plt.subplots(figsize=(13, 5.2))

    shift = np.hypot(
        pd.to_numeric(layout["x_m"], errors="coerce")
        - pd.to_numeric(layout["prior_x_m"], errors="coerce"),
        pd.to_numeric(layout["y_m"], errors="coerce")
        - pd.to_numeric(layout["prior_y_m"], errors="coerce"),
    )

    med = np.nanmedian(shift)
    p75 = np.nanpercentile(shift, 75)

    ax.plot(layout["channel"], shift, lw=2.1, color=CABLE_BLUE)

    ax.axhline(med, color="k", ls="--", lw=2)
    ax.axhline(p75, color="k", ls="-.", lw=2)

    ax.text(
        0.98, 0.98,
        f"Median: {med:.2f} m\n75th pct: {p75:.2f} m",
        transform=ax.transAxes,
        ha="right", va="top", fontsize = 12,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.5")
    )

    ax.set(
        xlabel="Channel",
        ylabel="Horizontal shift (m)",
        title="Horizontal displacement from prior"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)




def plot_distance_to_truth(layout, truth, out_png):
    truth_s = cumulative_arclength(truth["x_m"], truth["y_m"])

    xy_err, _, _, _ = project_points_onto_polyline(
        layout["x_m"].to_numpy(dtype=float),
        layout["y_m"].to_numpy(dtype=float),
        truth["x_m"].to_numpy(dtype=float),
        truth["y_m"].to_numpy(dtype=float),
        s_line=truth_s,
    )

    med = np.nanmedian(xy_err)
    p75 = np.nanpercentile(xy_err, 75)

    fig, ax = plt.subplots(figsize=(13, 5.0))

    ax.plot(layout["channel"], xy_err, lw=2.1, color=CABLE_BLUE)

    ax.axhline(med, color="0.2", ls="--", lw=2)
    ax.axhline(p75, color="0.2", ls="-.", lw=2)

    ax.text(
        0.98, 0.98,
        f"Median: {med:.2f} m\n75th pct: {p75:.2f} m",
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.5")
    )

    ax.set(
        xlabel="Channel",
        ylabel="Horizontal distance to reference geometry (m)",
        title="Difference between estimated cable and reference geometry",
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis-ready post-inversion plots.")
    parser.add_argument("--config",               type=Path,  required=True)
    parser.add_argument("--input_csv",            type=Path,  default=None)
    parser.add_argument("--inversion_output_dir", type=Path,  default=None)
    parser.add_argument("--truth_csv",            type=Path,  default=None)
    parser.add_argument("--output_dir",           type=Path,  default=None)
    parser.add_argument("--label_every",          type=int,   default=100)
    parser.add_argument("--min_weight",           type=float, default=0.50)
    parser.add_argument("--sound_speed",          type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    apply_thesis_style()

    (cfg, raw, layout, ctrl, fit, q,
     truth, tx_tbl, (_lat0, _lon0, _h0),
     _input_csv, inversion_output_dir, truth_csv) = load_inputs(
        config_path=args.config,
        input_csv=args.input_csv,
        inversion_output_dir=args.inversion_output_dir,
        truth_csv=args.truth_csv,
    )
    sound_speed = float(args.sound_speed) if args.sound_speed is not None else float(cfg["inversion"]["sound_speed"])
    icfg    = cfg["inversion"]
    out_dir = ensure_dir(args.output_dir or (inversion_output_dir / "thesis_plots"))

    if q is not None and "S_k" in q.columns:
        plot_sk_with_control_points(
            q=q, ctrl=ctrl,
            out_png=out_dir / "figure_sk_control_points.png",
            min_separation=int(icfg.get("control_min_separation", 5)),
            max_gap=int(icfg.get("control_max_gap", 80)),
        )

    plot_plan_view_with_tube(
        layout=layout, fit=fit, truth=truth, tx_tbl=tx_tbl,
        out_png=out_dir / "figure_planview_tube.png",
        label_every=args.label_every, sound_speed=sound_speed,
    )


    plot_plan_view_with_tube_show(
        layout=layout,
        fit=fit,
        truth=truth,
        tx_tbl=tx_tbl,
        sound_speed=sound_speed,
    )
        
    plot_tube_vs_channel(
    layout=layout,
    fit=fit,
    out_png=out_dir / "figure_tube_vs_channel.png",
    sound_speed=sound_speed,
    min_weight_for_tube=0.5,
    gaussian_sigma=30.0,
    )
    plot_plan_view_with_control_points(
        layout=layout, truth=truth, ctrl=ctrl, tx_tbl=tx_tbl,
        out_png=out_dir / "figure_planview_control_points.png",
        label_every=args.label_every,
    )
    plot_depth_profile(
        layout=layout, truth=truth, ctrl=ctrl,
        out_png=out_dir / "figure_depth_profile.png",
        label_every=args.label_every,
    )
    plot_depth_profile_with_control_points(
        layout=layout, truth=truth, ctrl=ctrl,
        out_png=out_dir / "figure_depth_profile_control_points.png",
        label_every=args.label_every,
    )
    plot_residual_histograms_three_thresholds(fit=fit, out_dir=out_dir)
    plot_obs_pred_grids_three_thresholds(fit=fit, out_dir=out_dir)
    plot_residual_vs_channel(
        fit=fit,
        out_png=out_dir / f"figure_residual_vs_channel_w{str(args.min_weight).replace('.','p')}.png",
        min_weight=args.min_weight,
    )
    plot_optimizer_convergence_from_history(
        inversion_output_dir=inversion_output_dir,
        out_png=out_dir / "figure_optimizer_convergence.png",
    )
    plot_horizontal_shift(layout=layout, out_png=out_dir / "figure_horizontal_shift.png")
    plot_distance_to_truth(layout=layout, truth=truth,
                           out_png=out_dir / "figure_distance_to_reference_vs_channel.png")
    
    plot_angular_diversity(
    fit=fit,
    layout=layout,
    out_png=out_dir / "figure_angular_diversity.png",
)
    

    plot_geometric_conditioning(
        fit=fit,
        layout=layout,
        out_dir=out_dir,
    )

    print(f"Thesis-ready plots written to: {out_dir}")


if __name__ == "__main__":
    main()
