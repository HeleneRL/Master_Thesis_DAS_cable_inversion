"""
geometric_conditioning.py
--------------------------
Functions for computing and plotting geometric conditioning metrics
(DOP, condition number kappa) and skyplots for TDOA cable localization.

Usage:
    plot_geometric_conditioning(fit=fit, layout=layout, out_dir=out_dir)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re

# ── Color scheme ──────────────────────────────────────────────────────────────
C_ORANGE = "#E07B39"
C_BLUE   = "#3B7FC4"
C_GREEN  = "#4A9B6F"
C_RED    = "#C43B3B"
C_GRAY   = "#888888"
C_PURPLE = "#8B5CA8"
C_YELLOW = "#E8C534"

# 6-color palette for source locations: blue, orange, green, red first
PALETTE  = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_YELLOW]


# ══════════════════════════════════════════════════════════════════════════════
# Core geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_channel_metrics(
    fit: pd.DataFrame,
    layout: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each channel compute DOP and kappa in both 3D and 2D (XY only).

    Returns a DataFrame with columns:
        channel, n_tx,
        dop_3d, kappa_3d,
        dop_2d, kappa_2d,
        angular_spread_deg
    """
    ch_col   = "channel_eff" if "channel_eff" in fit.columns else "channel"
    cable_xy = layout.set_index("channel")[["x_m", "y_m", "z_m"]]

    rows = []
    for ch_val, grp in fit.groupby(ch_col):
        ch_int = int(ch_val)
        if ch_int not in cable_xy.index:
            continue

        cx = float(cable_xy.loc[ch_int, "x_m"])
        cy = float(cable_xy.loc[ch_int, "y_m"])
        cz = float(cable_xy.loc[ch_int, "z_m"])

        tx_x = pd.to_numeric(grp["tx_x_m"], errors="coerce").to_numpy(float)
        tx_y = pd.to_numeric(grp["tx_y_m"], errors="coerce").to_numpy(float)
        tx_z = pd.to_numeric(grp["tx_u_m"], errors="coerce").to_numpy(float)
        w    = pd.to_numeric(grp["weight"],  errors="coerce").fillna(0).to_numpy(float)

        mask = np.isfinite(tx_x) & np.isfinite(tx_y) & np.isfinite(tx_z) & (w > 0)
        n_obs = int(np.sum(mask))

        nan_row = dict(channel=ch_val, n_tx=n_obs,
                       dop_3d=np.nan, kappa_3d=np.nan,
                       dop_2d=np.nan, kappa_2d=np.nan,
                       angular_spread_deg=np.nan)

        if n_obs < 2:
            rows.append(nan_row)
            continue

        dx = tx_x[mask] - cx
        dy = tx_y[mask] - cy
        dz = tx_z[mask] - cz   # positive = transmitter above cable

        # ── 3-D unit ray vectors ──────────────────────────────────────────
        r3 = np.sqrt(dx**2 + dy**2 + dz**2)
        ok3 = r3 > 0
        if np.sum(ok3) < 2:
            rows.append(nan_row)
            continue

        A3 = np.column_stack([dx[ok3]/r3[ok3],
                               dy[ok3]/r3[ok3],
                               dz[ok3]/r3[ok3]])
        dop_3d, kappa_3d = _dop_kappa(A3)

        # ── 2-D unit ray vectors (XY only) ────────────────────────────────
        r2 = np.sqrt(dx**2 + dy**2)
        ok2 = r2 > 0
        if np.sum(ok2) < 2:
            dop_2d, kappa_2d = np.nan, np.nan
        else:
            A2 = np.column_stack([dx[ok2]/r2[ok2], dy[ok2]/r2[ok2]])
            dop_2d, kappa_2d = _dop_kappa(A2)

        # ── Circular std of bearing angles (EN plane) ─────────────────────
        angles = np.arctan2(dy[ok2], dx[ok2]) if np.sum(ok2) >= 2 else np.array([])
        if len(angles) >= 2:
            S = np.mean(np.sin(angles))
            C_ = np.mean(np.cos(angles))
            R = np.sqrt(S**2 + C_**2)
            circ_std = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-9)))))
        else:
            circ_std = np.nan

        rows.append(dict(channel=ch_val, n_tx=n_obs,
                         dop_3d=dop_3d, kappa_3d=kappa_3d,
                         dop_2d=dop_2d, kappa_2d=kappa_2d,
                         angular_spread_deg=circ_std))

    return pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)


def _dop_kappa(A: np.ndarray) -> tuple[float, float]:
    """Return (DOP, kappa) for a unit-ray-vector matrix A."""
    AtA = A.T @ A
    eigvals = np.linalg.eigvalsh(AtA)          # ascending order
    lmin, lmax = eigvals[0], eigvals[-1]

    kappa = float(lmax / lmin) if lmin > 1e-12 else np.inf
    dop   = float(np.sqrt(np.sum(1.0 / eigvals[eigvals > 1e-12]))) \
            if lmin > 1e-12 else np.inf

    return dop, kappa


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Per-channel metrics
# ══════════════════════════════════════════════════════════════════════════════

def _plot_per_channel_metrics(
    metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Four-panel figure: 3D DOP, 2D DOP, 3D kappa, 2D kappa vs channel.

    Uses log scale on y-axis. Inf values are dropped (degenerate geometry).
    Y-axis is capped at the 99th percentile to prevent extreme outliers
    from compressing the informative range.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    ch = metrics["channel"]

    pairs = [
        ("dop_3d",   "DOP  (3D)",                       C_BLUE),
        ("dop_2d",   "HDOP  (Horizontal)",              C_ORANGE),
        ("kappa_3d", "Condition number κ  (3D)",         C_GREEN),
        ("kappa_2d", "Condition number H-κ  (Horizontal)", C_RED),
    ]

    for ax, (col, ylabel, color) in zip(axes, pairs):
        # Drop inf and nan — degenerate channels are not meaningful to plot
        vals = metrics[col].replace([np.inf, -np.inf], np.nan)

        # Cap at 99th percentile so spikes don't dominate the axis
        p99 = vals.quantile(0.99)
        p01 = vals.quantile(0.01)

        # Plot full data (clipped for display)
        vals_clipped = vals.clip(upper=p99)
        ax.plot(ch, vals_clipped, linewidth=1.4, color=color)

        # Mark clipped channels with vertical lines so the reader knows
        clipped_mask = vals > p99
        if clipped_mask.any():
            for xc in ch[clipped_mask]:
                ax.axvline(xc, color=color, alpha=0.25, linewidth=0.6,
                           linestyle="--")
            ax.annotate(
                f"{clipped_mask.sum()} channel(s) clipped above p99 = {p99:.1f}",
                xy=(0.01, 0.93), xycoords="axes fraction",
                fontsize=10, color=color, alpha=0.8,
            )

        # Log scale — only if all plotted values > 0
        plot_vals = vals_clipped.dropna()
        if len(plot_vals) > 0 and (plot_vals > 0).all():
            ax.set_yscale("log")

        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2, linewidth=0.5, which="both")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Channel", fontsize=11)
    axes[0].set_title("Geometric conditioning per channel", fontsize=12, pad=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")



# ══════════════════════════════════════════════════════════════════════════════
# DOP and kappa distribution histograms
# ══════════════════════════════════════════════════════════════════════════════
 
def _plot_conditioning_histograms(metrics: pd.DataFrame, out_path: Path) -> None:
    """
    Four-panel histogram: distribution of 2D DOP and 2D kappa across channels,
    shown as channel count and percentage. Bins reflect physically meaningful
    quality levels; the top bin catches all extreme values.
    """
    nl = chr(10)
    dop_edges  = [0, 1, 2, 5, 10, 20, 50, float('inf')]
    dop_labels = [
        '<1'  + nl + 'excellent',
        '1-2' + nl + 'good',
        '2-5' + nl + 'moderate',
        '5-10' + nl + 'poor',
        '10-20' + nl + 'very poor',
        '20-50' + nl + 'severe',
        '>50'  + nl + 'extreme',
    ]
    dop_colors = ['#2E7D32','#66BB6A','#FDD835','#FB8C00','#E53935','#B71C1C','#7B1FA2']
    kap_edges  = [0, 2, 5, 10, 50, 100, 1000, float('inf')]
    kap_labels = [
        '<2'    + nl + 'isotropic',
        '2-5'   + nl + 'good',
        '5-10'  + nl + 'moderate',
        '10-50' + nl + 'anisotropic',
        '50-100' + nl + 'strongly' + nl + 'anisotropic',
        '100-1k' + nl + 'severe',
        '>1000'  + nl + 'one-dim.',
    ]
    kap_colors = ['#2E7D32','#66BB6A','#FDD835','#FB8C00','#E53935','#B71C1C','#7B1FA2']
 
    def _bin_series(series, edges):
        vals = series.replace([np.inf, -np.inf], np.nan).dropna()
        counts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if np.isinf(hi):
                counts.append(int((vals >= lo).sum()))
            else:
                counts.append(int(((vals >= lo) & (vals < hi)).sum()))
        return np.array(counts), len(vals)
 
    dop_counts, dop_n = _bin_series(metrics['dop_2d'],   dop_edges)
    kap_counts, kap_n = _bin_series(metrics['kappa_2d'], kap_edges)
    dop_pct = dop_counts / dop_n * 100 if dop_n > 0 else dop_counts * 0.0
    kap_pct = kap_counts / kap_n * 100 if kap_n > 0 else kap_counts * 0.0
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Distribution of horizontal geometric conditioning across channels\n'
        f'(HDOP n={dop_n},  H-\u03ba, n={kap_n})',
        fontsize=13, y=1.01,
    )
 
    def _draw_bars(ax, counts, labels, colors, ylabel, title):
        x    = np.arange(len(labels))
        bars = ax.bar(x, counts, width=0.65,
                      color=colors, edgecolor='white', linewidth=0.6, alpha=0.9)
        peak = max(counts) if max(counts) > 0 else 1
        for bar, val in zip(bars, counts):
            if val > 0:
                lstr = str(int(val)) if 'Number' in ylabel else f'{val:.1f}%'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + peak * 0.01,
                        lstr, ha='center', va='bottom',
                        fontsize=8, color='#333333')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8.5,
                           va='top', ha='center')
        ax.tick_params(axis='x', pad=4)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis='y', alpha=0.2, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(-0.6, len(labels) - 0.4)
 
    _draw_bars(axes[0,0], dop_counts, dop_labels, dop_colors,
               'Number of channels', 'HDOP \u2014 channel count')
    _draw_bars(axes[0,1], dop_pct,    dop_labels, dop_colors,
               'Percentage of channels (%)', 'HDOP \u2014 percentage')
    _draw_bars(axes[1,0], kap_counts, kap_labels, kap_colors,
               'Number of channels', 'Horizontal \u03ba \u2014 channel count')
    _draw_bars(axes[1,1], kap_pct,    kap_labels, kap_colors,
               'Percentage of channels (%)', 'Horizontal \u03ba \u2014 percentage')
 
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')





     
def plot_conditioning_histograms_3d(metrics: pd.DataFrame, out_path: Path) -> None:
    """
    Four-panel histogram: distribution of 3D DOP and 3D kappa across channels,
    shown as channel count and percentage. Bins are shifted upward relative to
    the 2D version to reflect the always-present hemisphere constraint on
    vertical geometry.
    """
    nl = chr(10)
    dop_edges  = [0, 2, 4, 8, 20, 50, 200, float('inf')]
    dop_labels = [
        '<2'    + nl + 'ideal',
        '2-4'   + nl + 'good',
        '4-8'   + nl + 'moderate',
        '8-20'  + nl + 'poor',
        '20-50' + nl + 'very poor',
        '50-200' + nl + 'severe',
        '>200'  + nl + 'extreme',
    ]
    dop_colors = ['#2E7D32','#66BB6A','#FDD835','#FB8C00','#E53935','#B71C1C','#7B1FA2']
 
    kap_edges  = [0, 5, 20, 100, 500, 5000, 50000, float('inf')]
    kap_labels = [
        '<5'       + nl + 'good',
        '5-20'     + nl + 'moderate',
        '20-100'   + nl + 'anisotropic',
        '100-500'  + nl + 'strongly' + nl + 'anisotropic',
        '500-5k'   + nl + 'severe',
        '5k-50k'   + nl + 'extreme',
        '>50000'   + nl + 'one-dim.',
    ]
    kap_colors = ['#2E7D32','#66BB6A','#FDD835','#FB8C00','#E53935','#B71C1C','#7B1FA2']
 
    def _bin_series(series, edges):
        vals = series.replace([np.inf, -np.inf], np.nan).dropna()
        counts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if np.isinf(hi):
                counts.append(int((vals >= lo).sum()))
            else:
                counts.append(int(((vals >= lo) & (vals < hi)).sum()))
        return np.array(counts), len(vals)
 
    dop_counts, dop_n = _bin_series(metrics['dop_3d'],   dop_edges)
    kap_counts, kap_n = _bin_series(metrics['kappa_3d'], kap_edges)
    dop_pct = dop_counts / dop_n * 100 if dop_n > 0 else dop_counts * 0.0
    kap_pct = kap_counts / kap_n * 100 if kap_n > 0 else kap_counts * 0.0
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Distribution of 3D geometric conditioning across channels\n'
        f'(DOP n={dop_n},  \u03ba n={kap_n})',
        fontsize=13, y=1.01,
    )
 
    def _draw_bars(ax, counts, labels, colors, ylabel, title):
        x    = np.arange(len(labels))
        bars = ax.bar(x, counts, width=0.65,
                      color=colors, edgecolor='white', linewidth=0.6, alpha=0.9)
        peak = max(counts) if max(counts) > 0 else 1
        for bar, val in zip(bars, counts):
            if val > 0:
                lstr = str(int(val)) if 'Number' in ylabel else f'{val:.1f}%'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + peak * 0.01,
                        lstr, ha='center', va='bottom',
                        fontsize=8, color='#333333')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8.5,
                           va='top', ha='center')
        ax.tick_params(axis='x', pad=4)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis='y', alpha=0.2, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(-0.6, len(labels) - 0.4)
 
    _draw_bars(axes[0,0], dop_counts, dop_labels, dop_colors,
               'Number of channels', '3D DOP \u2014 channel count')
    _draw_bars(axes[0,1], dop_pct,    dop_labels, dop_colors,
               'Percentage of channels (%)', '3D DOP \u2014 percentage')
    _draw_bars(axes[1,0], kap_counts, kap_labels, kap_colors,
               'Number of channels', '3D \u03ba \u2014 channel count')
    _draw_bars(axes[1,1], kap_pct,    kap_labels, kap_colors,
               'Percentage of channels (%)', '3D \u03ba \u2014 percentage')
 
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Skyplots for selected channels
# ══════════════════════════════════════════════════════════════════════════════

def _select_channels(metrics: pd.DataFrame) -> list[int]:
    """
    Select up to 8 representative channels based on DOP and kappa.
    DOP: best, p25, p50, p75, worst.
    Kappa: best, median, worst  (drop if already selected).
    """
    valid = metrics.dropna(subset=["dop_3d", "kappa_3d"])
    valid = valid[np.isfinite(valid["dop_3d"]) & np.isfinite(valid["kappa_3d"])]
    if valid.empty:
        return []

    def _ch_at(series: pd.Series, idx: int) -> int:
        return int(valid.loc[series.index[idx], "channel"])

    dop_sorted  = valid["dop_3d"].sort_values()
    kap_sorted  = valid["kappa_3d"].sort_values()
    n = len(dop_sorted)

    selected: list[int] = []

    # DOP percentiles
    for frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
        idx = int(round(frac * (n - 1)))
        ch  = int(valid.loc[dop_sorted.index[idx], "channel"])
        if ch not in selected:
            selected.append(ch)

    # Kappa extremes + median
    for frac in [0.0, 0.50, 1.0]:
        idx = int(round(frac * (n - 1)))
        ch  = int(valid.loc[kap_sorted.index[idx], "channel"])
        if ch not in selected and len(selected) < 8:
            selected.append(ch)

    return selected


def _rays_for_channel(
    fit: pd.DataFrame,
    layout: pd.DataFrame,
    channel: int,
    global_fit: pd.DataFrame | None = None,
) -> dict:
    """
    Return dict with unit ray vectors and weights for one channel.
    Keys: dx, dy, dz, r3, e_hat (unit EN), weight, n
    """
    ch_col   = "channel_eff" if "channel_eff" in fit.columns else "channel"
    cable_xy = layout.set_index("channel")[["x_m", "y_m", "z_m"]]

    grp = fit[fit[ch_col] == channel]
    if grp.empty or channel not in cable_xy.index:
        return {}

    cx = float(cable_xy.loc[channel, "x_m"])
    cy = float(cable_xy.loc[channel, "y_m"])
    cz = float(cable_xy.loc[channel, "z_m"])

    tx_x = pd.to_numeric(grp["tx_x_m"], errors="coerce").to_numpy(float)
    tx_y = pd.to_numeric(grp["tx_y_m"], errors="coerce").to_numpy(float)
    tx_z = pd.to_numeric(grp["tx_u_m"], errors="coerce").to_numpy(float)
    w    = pd.to_numeric(grp["weight"],  errors="coerce").fillna(0).to_numpy(float)

    mask = np.isfinite(tx_x) & np.isfinite(tx_y) & np.isfinite(tx_z) & (w > 0)
    if np.sum(mask) < 1:
        return {}

    dx = tx_x[mask] - cx
    dy = tx_y[mask] - cy
    dz = tx_z[mask] - cz
    ww = w[mask]

    r3 = np.sqrt(dx**2 + dy**2 + dz**2)
    ok = r3 > 0

    # Per-ray location labels and colors
    loc_raw  = grp["location"].to_numpy() if "location" in grp.columns                else np.array(["unknown"] * len(grp))
    loc_raw  = loc_raw[mask][ok]
    color_map = _build_global_color_map(fit) if hasattr(_build_global_color_map, "__call__")                 else {}
    ray_colors = [color_map.get(_loc_number(l), C_GRAY) for l in loc_raw]
    ray_labels = [_loc_number(l) for l in loc_raw]

    return dict(dx=dx[ok], dy=dy[ok], dz=dz[ok],
                r3=r3[ok], weight=ww[ok],
                ray_colors=ray_colors, ray_labels=ray_labels)


def _draw_rays_2d(ax, xv, yv, colors, label=None):
    """Draw unit ray vectors as arrows from origin in a 2D axis.
    colors: single color string OR list of per-ray color strings.
    """
    for i, (x, y) in enumerate(zip(xv, yv)):
        col = colors[i] if isinstance(colors, (list, np.ndarray)) else colors
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=col,
                            lw=2.2, mutation_scale=12),
            zorder=3,
        )
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            color=C_GRAY, linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(0, color=C_GRAY, linewidth=0.4, linestyle=":")
    ax.axvline(0, color=C_GRAY, linewidth=0.4, linestyle=":")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.12, linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_skyplot_2d(
    rays: dict,
    channel: int,
    dop_2d: float,
    kappa_2d: float,
    out_path: Path,
) -> None:
    """
    Single-panel EN top-down skyplot with ray vectors from origin.
    Annotated with 2D DOP and kappa.
    """
    dx, dy = rays["dx"], rays["dy"]
    r2 = np.sqrt(dx**2 + dy**2)
    ok = r2 > 0
    ex = dx[ok] / r2[ok]
    ey = dy[ok] / r2[ok]

    ray_colors = rays.get("ray_colors", [C_BLUE] * len(ex))
    ray_labels = rays.get("ray_labels", [""] * len(ex))
    ray_colors_2d = [ray_colors[i] for i, v in enumerate(r2 > 0) if v]
    ray_labels_2d = [ray_labels[i] for i, v in enumerate(r2 > 0) if v]

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_rays_2d(ax, ex, ey, colors=ray_colors_2d)

    # Legend: unique sources present for this channel
    seen = {}
    for lbl, col in zip(ray_labels_2d, ray_colors_2d):
        if lbl not in seen:
            seen[lbl] = col
    handles = [plt.Line2D([0],[0], color=c, linewidth=2, label=l)
               for l, c in seen.items()]
    if handles:
        ax.legend(handles=handles, fontsize=9, loc="lower left",
                  framealpha=0.85, title="Source")

    ax.set_xlabel("East (unit ray)", fontsize=10)
    ax.set_ylabel("North (unit ray)", fontsize=10)
    ax.set_title(
        f"Channel {channel}  —  EN top-down view\n"
        f"HDOP = {dop_2d:.2f},  H-κ = {kappa_2d:.1f}",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_skyplot_3d(
    rays: dict,
    channel: int,
    dop_3d: float,
    kappa_3d: float,
    out_path: Path,
) -> None:
    """
    3-D skyplot: unit ray vectors drawn as arrows from origin in 3-D space.
    Annotated with 3D DOP and kappa.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    dx, dy, dz = rays["dx"], rays["dy"], rays["dz"]
    r3 = rays["r3"]
    ex = dx / r3
    ey = dy / r3
    ez = dz / r3

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Draw arrows as quivers from origin, colored by source location
    ray_colors = rays.get("ray_colors", [C_ORANGE] * len(ex))
    ray_labels = rays.get("ray_labels", [""] * len(ex))
    origin = np.zeros(1)
    seen_3d = {}
    for xi, yi, zi, col, lbl in zip(ex, ey, ez, ray_colors, ray_labels):
        ax.quiver(0, 0, 0, xi, yi, zi,
                  color=col, linewidth=2.2, arrow_length_ratio=0.12,
                  normalize=False)
        if lbl not in seen_3d:
            seen_3d[lbl] = col

    # Reference unit sphere wireframe (faint)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color=C_GRAY, alpha=0.08,
                      linewidth=0.4, rstride=2, cstride=2)

    # Equatorial circle
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u),
            color=C_GRAY, linewidth=0.6, linestyle="--", alpha=0.5)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel("East",  fontsize=9, labelpad=2)
    ax.set_ylabel("North", fontsize=9, labelpad=2)
    ax.set_zlabel("Up",    fontsize=9, labelpad=2)
    ax.set_title(
        f"Channel {channel}  —  3-D ray directions\n"
        f"3D DOP = {dop_3d:.2f},  3D κ = {kappa_3d:.1f}",
        fontsize=11,
    )
    if seen_3d:
        handles_3d = [plt.Line2D([0],[0], color=c, linewidth=2, label=l)
                      for l, c in seen_3d.items()]
        ax.legend(handles=handles_3d, fontsize=9, loc="upper left",
                  framealpha=0.85, title="Source")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# def _plot_angular_spread_histogram(
#     metrics: pd.DataFrame,
#     out_path: Path,
# ) -> None:
#     """
#     Two-panel histogram of angular spread (circular std) across all channels.
#     Left panel:  count of channels per 10-degree bin.
#     Right panel: percentage of channels per 10-degree bin.
#     """
#     spreads = metrics["angular_spread_deg"].dropna()
#     spreads = spreads[np.isfinite(spreads)]

#     bins = np.arange(0, 185, 10)   # 0,10,20,...,180
#     counts, edges = np.histogram(spreads, bins=bins)
#     pct = counts / counts.sum() * 100
#     centres = (edges[:-1] + edges[1:]) / 2
#     width   = 8  # bar width in degrees

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

#     axes[0].bar(centres, counts, width=width,
#                 color=C_BLUE, edgecolor="white", linewidth=0.5, alpha=0.85)
#     axes[0].set_xlabel("Angular spread (°, circular std)", fontsize=10)
#     axes[0].set_ylabel("Number of channels", fontsize=10)
#     axes[0].set_title("Angular spread distribution — channel count", fontsize=11)
#     axes[0].set_xticks(bins)
#     axes[0].grid(True, axis="y", alpha=0.2, linewidth=0.5)
#     axes[0].spines[["top", "right"]].set_visible(False)

#     axes[1].bar(centres, pct, width=width,
#                 color=C_ORANGE, edgecolor="white", linewidth=0.5, alpha=0.85)
#     axes[1].set_xlabel("Angular spread (°, circular std)", fontsize=10)
#     axes[1].set_ylabel("Percentage of channels (%)", fontsize=10)
#     axes[1].set_title("Angular spread distribution — percentage", fontsize=11)
#     axes[1].set_xticks(bins)
#     axes[1].grid(True, axis="y", alpha=0.2, linewidth=0.5)
#     axes[1].spines[["top", "right"]].set_visible(False)

#     fig.suptitle(
#         f"Distribution of transmitter angular spread across {len(spreads)} channels",
#         fontsize=12, y=1.02,
#     )
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  Saved: {out_path}")




def _plot_angular_spread_histogram(metrics: pd.DataFrame, out_path: Path) -> None:
    spreads = metrics["angular_spread_deg"].dropna()
    spreads = spreads[np.isfinite(spreads)]
    bins    = np.arange(0, 185, 10)
    counts, edges = np.histogram(spreads, bins=bins)
    pct     = counts / counts.sum() * 100
    centres = (edges[:-1] + edges[1:]) / 2
    width   = 8

    # Green-to-red color scheme: low spread = poor geometry (red),
    # high spread = good geometry (green), matching DOP/kappa histograms.
    # 18 bins from 0-10 up to 170-180
    spread_colors = [
        "#B71C1C",  # 0-10    deep red       — very poor
        "#E53935",  # 10-20   red
        "#FB8C00",  # 20-30   orange
        "#FB8C00",  # 30-40   orange
        "#FDD835",  # 40-50   yellow
        "#FDD835",  # 50-60   yellow
        "#66BB6A",  # 60-70   light green
        "#66BB6A",  # 70-80   light green
        "#2E7D32",  # 80-90   green
        "#2E7D32",  # 90-100  green
        "#2E7D32",  # 100-110 green
        "#2E7D32",  # 110-120 green
        "#2E7D32",  # 120-130 green
        "#2E7D32",  # 130-140 green
        "#2E7D32",  # 140-150 green
        "#2E7D32",  # 150-160 green
        "#2E7D32",  # 160-170 green
        "#2E7D32",  # 170-180 green
    ]

    def _draw_bars(ax, values, ylabel, title):
        peak = max(values) if max(values) > 0 else 1
        bars = ax.bar(centres, values, width=width,
                      color=spread_colors[:len(centres)],
                      edgecolor="white", linewidth=0.5, alpha=0.9)
        for bar, val in zip(bars, values):
            if val > 0:
                lstr = str(int(val)) if "Number" in ylabel else f"{val:.1f}%"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + peak * 0.01,
                        lstr, ha="center", va="bottom",
                        fontsize=7.5, color="#333333")
        axes[0].set_xlabel("Horizontal angular spread (°, circular std)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(bins)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-5, 185)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_bars(axes[0], counts,
               "Number of channels",
               "Horizontal angular spread \u2014 channel count")
    _draw_bars(axes[1], pct,
               "Percentage of channels (%)",
               "Horizontal angular spread \u2014 percentage")

    fig.suptitle(
        f"Distribution of horizontal transmitter angular spread "
        f"across {len(spreads)} channels\n"
        f"(circular standard deviation of bearing angles in the EN plane)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# All-rays planview figures
# ══════════════════════════════════════════════════════════════════════════════

def _plot_all_rays_en(
    fit: pd.DataFrame,
    layout: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    East-North planview of every detected ray between a channel and a source.
    One color per unique source location (tx_x_m, tx_y_m pair).
    Rays are drawn as semi-transparent lines from channel position to source.
    """
    ch_col   = "channel_eff" if "channel_eff" in fit.columns else "channel"
    cable_xy = layout.set_index("channel")[["x_m", "y_m"]]

    # Identify unique source locations and assign colors
    fit_valid = fit.copy()
    fit_valid["tx_x_m"] = pd.to_numeric(fit_valid["tx_x_m"], errors="coerce")
    fit_valid["tx_y_m"] = pd.to_numeric(fit_valid["tx_y_m"], errors="coerce")
    fit_valid["weight"]  = pd.to_numeric(fit_valid["weight"],  errors="coerce").fillna(0)

    fit_valid = fit_valid[
        fit_valid["tx_x_m"].notna() &
        fit_valid["tx_y_m"].notna() &
        (fit_valid["weight"] > 0)
    ]

    fit_valid["src_key"] = fit_valid["location"].apply(_loc_number)         if "location" in fit_valid.columns else         (fit_valid["tx_x_m"].round(0).astype(int).astype(str) + "_" +
         fit_valid["tx_y_m"].round(0).astype(int).astype(str))
    color_map = _build_global_color_map(fit)
    src_keys  = list(color_map.keys())

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw all rays
    plotted_keys = set()
    for _, row in fit_valid.iterrows():
        ch_int = int(row[ch_col])
        if ch_int not in cable_xy.index:
            continue
        cx = float(cable_xy.loc[ch_int, "x_m"])
        cy = float(cable_xy.loc[ch_int, "y_m"])
        tx = float(row["tx_x_m"])
        ty = float(row["tx_y_m"])
        key = row["src_key"]
        color = color_map[key]
        ax.plot([cx, tx], [cy, ty],
                color=color, alpha=0.12, linewidth=0.6,
                rasterized=True)
        plotted_keys.add(key)

    # Legend: one entry per source location
    handles = [plt.Line2D([0],[0], color=color_map[k], linewidth=1.5, alpha=0.7,
                           label=k) for k in src_keys if k in color_map]
    ax.legend(handles=handles, fontsize=9, loc="upper right",
              framealpha=0.8, title="Source location")

    ax.set_xlabel("East (m)", fontsize=11)
    ax.set_ylabel("North (m)", fontsize=11)
    ax.set_title("All detected rays — East/North planview", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_all_rays_elevation(
    fit: pd.DataFrame,
    layout: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Elevation view of every detected ray.
    X-axis: horizontal distance sqrt(dE^2 + dN^2) from channel to source,
            signed by sign of dN (north component) so rays on both sides visible.
    Y-axis: vertical offset dZ = tx_u_m - z_m  (positive = source above cable).
    One color per unique source location.
    """
    ch_col   = "channel_eff" if "channel_eff" in fit.columns else "channel"
    cable_xyz = layout.set_index("channel")[["x_m", "y_m", "z_m"]]

    fit_valid = fit.copy()
    for col in ["tx_x_m", "tx_y_m", "tx_u_m", "weight"]:
        fit_valid[col] = pd.to_numeric(fit_valid[col], errors="coerce")
    fit_valid = fit_valid[
        fit_valid["tx_x_m"].notna() &
        fit_valid["tx_y_m"].notna() &
        fit_valid["tx_u_m"].notna() &
        (fit_valid["weight"].fillna(0) > 0)
    ]

    fit_valid["src_key"] = fit_valid["location"].apply(_loc_number)         if "location" in fit_valid.columns else         (fit_valid["tx_x_m"].round(0).astype(int).astype(str) + "_" +
         fit_valid["tx_y_m"].round(0).astype(int).astype(str))
    color_map = _build_global_color_map(fit)
    src_keys  = list(color_map.keys())

    fig, ax = plt.subplots(figsize=(12, 7))

    plotted_keys = set()
    for _, row in fit_valid.iterrows():
        ch_int = int(row[ch_col])
        if ch_int not in cable_xyz.index:
            continue
        cx = float(cable_xyz.loc[ch_int, "x_m"])
        cy = float(cable_xyz.loc[ch_int, "y_m"])
        cz = float(cable_xyz.loc[ch_int, "z_m"])

        dx = float(row["tx_x_m"]) - cx
        dy = float(row["tx_y_m"]) - cy
        dz = float(row["tx_u_m"]) - cz   # positive = source above cable

        horiz = np.sqrt(dx**2 + dy**2)   # always positive — unsigned range

        key   = row["src_key"]
        color = color_map[key]
        ax.plot([0, horiz], [0, dz],
                color=color, alpha=0.12, linewidth=0.6,
                rasterized=True)
        plotted_keys.add(key)

    # Mark origin = receiver
    ax.plot(0, 0, "k+", markersize=8, zorder=5, label="Receiver (origin)")

    handles = [plt.Line2D([0],[0], color=color_map[k], linewidth=1.5, alpha=0.7,
                           label=k) for k in src_keys if k in color_map]
    ax.legend(handles=handles, fontsize=9, loc="upper right",
              framealpha=0.8, title="Source location")

    ax.set_xlabel("Horizontal range (m)  [receiver at origin]", fontsize=11)
    ax.set_ylabel("Vertical offset ΔZ (m)  [positive = source above cable]", fontsize=11)
    ax.set_title("All detected rays — elevation view", fontsize=12)
    ax.axhline(0, color=C_GRAY, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")



def _loc_number(location_str: str) -> str:
    """Extract location number from e.g. loc2_tx3 -> '2'."""
    try:
        m = re.match(r"loc(\d+)", str(location_str), re.IGNORECASE)
        return m.group(1) if m else str(location_str)
    except Exception:
        return str(location_str)


def _build_global_color_map(fit: pd.DataFrame) -> dict:
    """
    Return a stable {loc_label: color} map based on all unique location labels
    in the fit dataframe, ordered by location number.
    """
    import re as _re
    labels = fit["location"].dropna().unique().tolist() if "location" in fit.columns else []
    loc_labels = sorted(set(_loc_number(l) for l in labels),
                        key=lambda s: int(s) if s.isdigit() else 0)
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(loc_labels)}


def _build_all_rays_data(fit, layout):
    """
    Shared helper: returns unit ray components and source labels
    for every valid detection across all channels.
    Uses the location column to label sources as 'Loc N'.
    """
    ch_col    = "channel_eff" if "channel_eff" in fit.columns else "channel"
    cable_xyz = layout.set_index("channel")[["x_m", "y_m", "z_m"]]

    fv = fit.copy()
    for col in ["tx_x_m", "tx_y_m", "tx_u_m", "weight"]:
        fv[col] = pd.to_numeric(fv[col], errors="coerce")
    fv = fv[
        fv["tx_x_m"].notna() & fv["tx_y_m"].notna() &
        fv["tx_u_m"].notna() & (fv["weight"].fillna(0) > 0)
    ]
    fv["src_key"] = fv["location"].apply(_loc_number) if "location" in fv.columns                     else (fv["tx_x_m"].round(0).astype(int).astype(str) + "_" +
                          fv["tx_y_m"].round(0).astype(int).astype(str))

    color_map = _build_global_color_map(fit)
    src_keys  = [k for k in color_map]   # ordered

    ex_all, ey_all, ez_all, horiz_all, colors_all = [], [], [], [], []

    for _, row in fv.iterrows():
        ch_int = int(row[ch_col])
        if ch_int not in cable_xyz.index:
            continue
        cx = float(cable_xyz.loc[ch_int, "x_m"])
        cy = float(cable_xyz.loc[ch_int, "y_m"])
        cz = float(cable_xyz.loc[ch_int, "z_m"])

        dx = float(row["tx_x_m"]) - cx
        dy = float(row["tx_y_m"]) - cy
        dz = float(row["tx_u_m"]) - cz

        r3 = float(np.sqrt(dx**2 + dy**2 + dz**2))
        if r3 < 1e-6:
            continue

        key = row["src_key"]
        ex_all.append(dx / r3)
        ey_all.append(dy / r3)
        ez_all.append(dz / r3)
        horiz_all.append(float(np.sqrt(dx**2 + dy**2)) / r3)
        colors_all.append(color_map.get(key, C_GRAY))

    return dict(
        ex=np.array(ex_all), ey=np.array(ey_all),
        ez=np.array(ez_all), horiz=np.array(horiz_all),
        colors=colors_all,
        src_keys=src_keys, color_map=color_map,
    )


def _plot_unit_rays_en(fit, layout, out_path):
    """
    EN skyplot: all unit ray vectors plotted as lines from origin,
    colored by source location, semi-transparent.
    """
    d = _build_all_rays_data(fit, layout)
    if not d["ex"].size:
        warnings.warn("No ray data for EN unit-ray plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for ex, ey, col in zip(d["ex"], d["ey"], d["colors"]):
        ax.plot([0, ex], [0, ey], color=col, alpha=0.08, linewidth=0.7,
                rasterized=True)

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            color=C_GRAY, linewidth=0.8, linestyle="--", zorder=4)
    ax.axhline(0, color=C_GRAY, linewidth=0.4, linestyle=":")
    ax.axvline(0, color=C_GRAY, linewidth=0.4, linestyle=":")

    # Legend proxies
    handles = [plt.Line2D([0],[0], color=d["color_map"][k], linewidth=2, label=k)
               for k in d["src_keys"] if k in d["color_map"]]
    ax.legend(handles=handles, fontsize=9, loc="upper left",
              framealpha=0.85, title="Source location")

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("East (unit ray)", fontsize=11)
    ax.set_ylabel("North (unit ray)", fontsize=11)
    ax.set_title("All detected rays — EN unit-ray skyplot\n"
                , fontsize=11)
    ax.grid(True, alpha=0.12, linewidth=0.4)
    ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_unit_rays_elevation(fit, layout, out_path):
    """
    Elevation unit-ray skyplot: x = horizontal component of unit ray,
    y = vertical (Up) component of unit ray.
    All rays normalized to unit length and plotted from origin.
    """
    d = _build_all_rays_data(fit, layout)
    if not d["ex"].size:
        warnings.warn("No ray data for elevation unit-ray plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for hv, ez, col in zip(d["horiz"], d["ez"], d["colors"]):
        ax.plot([0, hv], [0, ez], color=col, alpha=0.08, linewidth=0.7,
                rasterized=True)

    # Unit circle (quarter — horiz >= 0 always)
    theta = np.linspace(0, np.pi/2, 200)
    ax.plot(np.cos(theta), np.sin(theta),
            color=C_GRAY, linewidth=0.8, linestyle="--", zorder=4)

    ax.axhline(0, color=C_GRAY, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color=C_GRAY, linewidth=0.4, linestyle=":")

    handles = [plt.Line2D([0],[0], color=d["color_map"][k], linewidth=2, label=k)
               for k in d["src_keys"] if k in d["color_map"]]
    ax.legend(handles=handles, fontsize=9, loc="upper right",
              framealpha=0.85, title="Source location")

    ax.set_xlabel("Horizontal component of unit ray  √(eE²+eN²)", fontsize=11)
    ax.set_ylabel("Vertical component of unit ray  eU", fontsize=11)
    ax.set_title("All detected rays — elevation unit-ray skyplot\n"
                 , fontsize=11)
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.12, linewidth=0.4)
    ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")



def _plot_unit_rays_by_location(fit, layout, out_path):
    """
    3x2 grid of EN unit-ray skyplots, one panel per source location.
    All rays are normalized to unit length, origin = receiver.
    Each panel is self-colored by its location color from the global palette.
    """
    d = _build_all_rays_data(fit, layout)
    if not d["ex"].size:
        warnings.warn("No ray data for per-location unit-ray plot.")
        return

    # Build per-location arrays
    loc_data = {}
    fv = fit.copy()
    fv["src_key"] = fv["location"].apply(_loc_number) if "location" in fv.columns                     else pd.Series(["?"] * len(fv))
    ch_col    = "channel_eff" if "channel_eff" in fit.columns else "channel"
    cable_xyz = layout.set_index("channel")[["x_m", "y_m", "z_m"]]

    for col in ["tx_x_m", "tx_y_m", "tx_u_m", "weight"]:
        fv[col] = pd.to_numeric(fv[col], errors="coerce")
    fv = fv[fv["tx_x_m"].notna() & fv["tx_y_m"].notna() &
            fv["tx_u_m"].notna() & (fv["weight"].fillna(0) > 0)]

    for _, row in fv.iterrows():
        ch_int = int(row[ch_col])
        if ch_int not in cable_xyz.index:
            continue
        cx = float(cable_xyz.loc[ch_int, "x_m"])
        cy = float(cable_xyz.loc[ch_int, "y_m"])
        dx = float(row["tx_x_m"]) - cx
        dy = float(row["tx_y_m"]) - cy
        dz = float(row["tx_u_m"]) - float(cable_xyz.loc[ch_int, "z_m"])
        r3 = float(np.sqrt(dx**2 + dy**2 + dz**2))
        if r3 < 1e-6:
            continue
        key = row["src_key"]
        if key not in loc_data:
            loc_data[key] = {"ex": [], "ey": []}
        loc_data[key]["ex"].append(dx / r3)
        loc_data[key]["ey"].append(dy / r3)

    color_map = _build_global_color_map(fit)
    loc_keys  = [k for k in color_map if k in loc_data]  # ordered

    n_locs = len(loc_keys)
    ncols  = 3
    nrows  = int(np.ceil(n_locs / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    theta = np.linspace(0, 2 * np.pi, 300)

    for idx, key in enumerate(loc_keys):
        r, c  = divmod(idx, ncols)
        ax    = axes[r, c]
        color = color_map[key]
        ex    = np.array(loc_data[key]["ex"])
        ey    = np.array(loc_data[key]["ey"])

        for xi, yi in zip(ex, ey):
            ax.plot([0, xi], [0, yi], color=color, alpha=0.06,
                    linewidth=0.5, rasterized=True)

        ax.plot(np.cos(theta), np.sin(theta),
                color=C_GRAY, linewidth=0.8, linestyle="--", zorder=4)
        ax.axhline(0, color=C_GRAY, linewidth=0.4, linestyle=":")
        ax.axvline(0, color=C_GRAY, linewidth=0.4, linestyle=":")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title(f"Location {key}  (n={len(ex)})", fontsize=11,
                     color=color, fontweight="bold")
        ax.set_xlabel("East", fontsize=9)
        ax.set_ylabel("North", fontsize=9)
        ax.grid(True, alpha=0.12, linewidth=0.4)
        ax.spines[["top","right"]].set_visible(False)

    # Hide unused panels
    for idx in range(n_locs, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("EN unit-ray skyplot per source location\n"
                 ,
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")



def _plot_unit_rays_elevation_by_location(fit, layout, out_path):
    """
    3x2 grid of elevation unit-ray skyplots, one panel per source location.
    X-axis: horizontal component of unit ray sqrt(eE^2+eN^2).
    Y-axis: vertical (Up) component of unit ray eU.
    Each panel colored in that location's palette color.
    """
    ch_col    = "channel_eff" if "channel_eff" in fit.columns else "channel"
    cable_xyz = layout.set_index("channel")[["x_m", "y_m", "z_m"]]

    fv = fit.copy()
    fv["src_key"] = fv["location"].apply(_loc_number) if "location" in fv.columns                     else pd.Series(["?"] * len(fv))
    for col in ["tx_x_m", "tx_y_m", "tx_u_m", "weight"]:
        fv[col] = pd.to_numeric(fv[col], errors="coerce")
    fv = fv[fv["tx_x_m"].notna() & fv["tx_y_m"].notna() &
            fv["tx_u_m"].notna() & (fv["weight"].fillna(0) > 0)]

    loc_data = {}
    for _, row in fv.iterrows():
        ch_int = int(row[ch_col])
        if ch_int not in cable_xyz.index:
            continue
        cx = float(cable_xyz.loc[ch_int, "x_m"])
        cy = float(cable_xyz.loc[ch_int, "y_m"])
        cz = float(cable_xyz.loc[ch_int, "z_m"])
        dx = float(row["tx_x_m"]) - cx
        dy = float(row["tx_y_m"]) - cy
        dz = float(row["tx_u_m"]) - cz
        r3 = float(np.sqrt(dx**2 + dy**2 + dz**2))
        if r3 < 1e-6:
            continue
        key   = row["src_key"]
        horiz = float(np.sqrt(dx**2 + dy**2)) / r3
        ez    = dz / r3
        if key not in loc_data:
            loc_data[key] = {"horiz": [], "ez": []}
        loc_data[key]["horiz"].append(horiz)
        loc_data[key]["ez"].append(ez)

    color_map = _build_global_color_map(fit)
    loc_keys  = [k for k in color_map if k in loc_data]

    ncols = 3
    nrows = int(np.ceil(len(loc_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    theta = np.linspace(0, np.pi / 2, 200)

    for idx, key in enumerate(loc_keys):
        r, c  = divmod(idx, ncols)
        ax    = axes[r, c]
        color = color_map[key]
        horiz = np.array(loc_data[key]["horiz"])
        ez    = np.array(loc_data[key]["ez"])

        for hv, ev in zip(horiz, ez):
            ax.plot([0, hv], [0, ev], color=color, alpha=0.06,
                    linewidth=0.5, rasterized=True)

        ax.plot(np.cos(theta), np.sin(theta),
                color=C_GRAY, linewidth=0.8, linestyle="--", zorder=4)
        ax.axhline(0, color=C_GRAY, linewidth=0.4, linestyle="--", alpha=0.5)
        ax.axvline(0, color=C_GRAY, linewidth=0.4, linestyle=":")
        ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.15, 1.1)
        ax.set_aspect("equal")
        ax.set_title(f"Location {key}  (n={len(horiz)})", fontsize=11,
                     color=color, fontweight="bold")
        ax.set_xlabel("Horizontal  √(eE²+eN²)", fontsize=9)
        ax.set_ylabel("Up  eU", fontsize=9)
        ax.grid(True, alpha=0.12, linewidth=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(len(loc_keys), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Elevation unit-ray skyplot per source location\n"
                 ,
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def plot_geometric_conditioning(
    fit: pd.DataFrame,
    layout: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Compute geometric conditioning metrics and produce figures.

    Saves:
        figure_conditioning_metrics.png      — per-channel DOP and kappa (4 panels)
        figure_angular_spread_histogram.png  — histogram of angular spread
        figure_all_rays_en.png               — EN planview of all detected rays
        figure_all_rays_elevation.png        — elevation view of all detected rays
        figure_skyplot_2d_ch{N}.png          — EN top-down 2D skyplot per channel
        figure_skyplot_3d_ch{N}.png          — 3D ray direction plot per channel
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing per-channel metrics...")
    metrics = _compute_channel_metrics(fit, layout)

    if metrics.empty:
        warnings.warn("No metrics computed — check fit/layout alignment.")
        return

    # Figure 1: per-channel metrics
    _plot_per_channel_metrics(
        metrics,
        out_path=out_dir / "figure_conditioning_metrics.png",
    )


    # Figure 1b: DOP and kappa distribution histograms
    _plot_conditioning_histograms(
        metrics,
        out_path=out_dir / "figure_conditioning_histograms.png",
    )


        # Figure 1c: 3D DOP and kappa distribution histograms
    plot_conditioning_histograms_3d(
        metrics,
        out_path=out_dir / 'figure_conditioning_histograms_3d.png',
    )

    # Figure 2: angular spread histogram
    _plot_angular_spread_histogram(
        metrics,
        out_path=out_dir / "figure_angular_spread_histogram.png",
    )

    # Figure 3: all-rays EN planview
    print("Plotting all-rays EN planview...")
    _plot_all_rays_en(
        fit=fit,
        layout=layout,
        out_path=out_dir / "figure_all_rays_en.png",
    )

    # Figure 4: all-rays elevation view
    print("Plotting all-rays elevation view...")
    _plot_all_rays_elevation(
        fit=fit,
        layout=layout,
        out_path=out_dir / "figure_all_rays_elevation.png",
    )

    # Figure 5: unit-ray EN skyplot (all channels)
    print("Plotting unit-ray EN skyplot...")
    _plot_unit_rays_en(
        fit=fit,
        layout=layout,
        out_path=out_dir / "figure_unit_rays_en.png",
    )

    # Figure 6: unit-ray elevation skyplot (all channels)
    print("Plotting unit-ray elevation skyplot...")
    _plot_unit_rays_elevation(
        fit=fit,
        layout=layout,
        out_path=out_dir / "figure_unit_rays_elevation.png",
    )

    # Figure 7: per-location EN unit-ray panel
    print("Plotting per-location EN unit-ray panels...")
    _plot_unit_rays_by_location(
        fit=fit,
        layout=layout,
        out_path=out_dir / "figure_unit_rays_by_location.png",
    )

    # Figure 8: per-location elevation unit-ray panels
    print("Plotting per-location elevation unit-ray panels...")
    _plot_unit_rays_elevation_by_location(
        fit=fit,
        layout=layout,
        out_path=out_dir / "figure_unit_rays_elevation_by_location.png",
    )

    # Select representative channels (excludes inf/nan)
    selected = _select_channels(metrics)
    if not selected:
        warnings.warn("Could not select representative channels.")
        return

    print(f"Selected channels for skyplots: {selected}")
    met_idx = metrics.set_index("channel")

    for ch in selected:
        rays = _rays_for_channel(fit, layout, ch, global_fit=fit)
        if not rays:
            warnings.warn(f"No ray data for channel {ch}, skipping.")
            continue

        row      = met_idx.loc[ch]
        dop_2d   = float(row["dop_2d"])
        kappa_2d = float(row["kappa_2d"])
        dop_3d   = float(row["dop_3d"])
        kappa_3d = float(row["kappa_3d"])

        # 2D EN skyplot
        _plot_skyplot_2d(
            rays=rays,
            channel=ch,
            dop_2d=dop_2d,
            kappa_2d=kappa_2d,
            out_path=out_dir / f"figure_skyplot_2d_ch{ch}.png",
        )

        # 3D skyplot
        _plot_skyplot_3d(
            rays=rays,
            channel=ch,
            dop_3d=dop_3d,
            kappa_3d=kappa_3d,
            out_path=out_dir / f"figure_skyplot_3d_ch{ch}.png",
        )

    print("Done.")