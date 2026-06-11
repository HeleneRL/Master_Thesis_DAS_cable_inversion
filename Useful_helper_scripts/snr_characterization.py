"""
snr_characterization.py
=======================
Reads  all_locations_detections.csv  (output of detector_bulk.py) and produces SNR characterisation

Usage
-----
    python snr_characterization.py \
        --csv path/to/all_locations_detections.csv \
        [--weight_csv path/to/observations_with_weights.csv] \
        [--outdir figures/snr] \
        [--fmt pdf]          # pdf (default) or png

The weight CSV is optional.  If supplied it must contain columns
  (location, anchor_label, channel, weight)   so final composite weights
  can be shown.  Without it the script falls back to pick_quality_score
  as the weight proxy (clearly labelled in the figures).


"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":        "STIXGeneral",
    "mathtext.fontset":   "stix",
    "font.size":          13,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "legend.title_fontsize": 12,
    "figure.dpi":         150,        # screen preview
    "savefig.dpi":        300,        # saved files
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "grid.linewidth":     0.5,
    "grid.alpha":         0.4,
    "lines.linewidth":    1.6,
    "patch.linewidth":    0.8,
})

# Colour palette – consistent across all figures
LOC_PALETTE = {
    "loc2": "#1f77b4",
    "loc3": "#ff7f0e",
    "loc4": "#2ca02c",
    "loc5": "#d62728",
    "loc6": "#9467bd",
    "loc7": "#8c564b",
}
TIER_COLOURS = {
    "High  (w > 0.9)":   "#2ca02c",
    "Mid   (0.5–0.9)":   "#ff7f0e",
    "Low   (w < 0.5)":   "#d62728",
}
GREY = "#555555"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, path: Path, fmt: str) -> None:
    """Save figure as pdf or png."""
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, format=fmt, bbox_inches="tight")
    print(f"  [saved] {out}")


def snr_db(df: pd.DataFrame) -> pd.Series:
    """20·log10(peak / baseline_median) – more intuitive than snr_like."""
    return 20.0 * np.log10(
        df["peak_raw_envelope"] / (df["baseline_median"].clip(lower=1e-12))
    )


def tier_label(w: float) -> str:
    if w > 0.9:
        return "High  (w > 0.9)"
    elif w >= 0.5:
        return "Mid   (0.5–0.9)"
    else:
        return "Low   (w < 0.5)"


def add_median_line(ax, data_dict: dict, positions, colour="black"):
    """Overlay median tick marks on a violin/box plot."""
    for pos, key in zip(positions, data_dict):
        med = np.median(data_dict[key])
        ax.hlines(med, pos - 0.18, pos + 0.18, colors=colour, linewidths=2.0, zorder=5)


def stat_block(series: pd.Series, label: str) -> str:
    return (
        f"{label:30s}  n={len(series):6d}  "
        f"median={series.median():6.2f}  "
        f"mean={series.mean():6.2f}  "
        f"p10={series.quantile(0.10):6.2f}  "
        f"p90={series.quantile(0.90):6.2f}  "
        f"std={series.std():5.2f}"
    )


# ---------------------------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------------------------

def load_data(
    csv_path: Path,
    weight_csv: Path | None,
    channel_min: int = 0,
    channel_max: int = 999_999,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df):,} rows from {csv_path.name}")
    print(f"Columns: {list(df.columns)}\n")

    # ---- Channel mask: keep only in-water channels -------------------------
    n_before = len(df)
    df = df[(df["channel"] >= channel_min) & (df["channel"] <= channel_max)].copy()
    n_after = len(df)
    n_dropped = n_before - n_after
    print(f"Channel filter [{channel_min}, {channel_max}]: "
          f"kept {n_after:,} rows, dropped {n_dropped:,} "
          f"({100*n_dropped/n_before:.1f}% — land/pier/out-of-water channels)")

    # Computed columns
    df["snr_dB"] = snr_db(df)

    # Anchor sweep label  e.g. "loc3_sweep1"
    df["sweep_label"] = df["anchor_label"].astype(str)

    # Normalise location names to lowercase short form
    df["loc"] = df["location"].str.lower().str.replace(r"\s+", "", regex=True)

    # Weight tier – prefer joined weight CSV, fall back to pick_quality_score
    if weight_csv is not None and weight_csv.exists():
        w = pd.read_csv(weight_csv)
        df = df.merge(w[["location", "anchor_label", "channel", "weight"]],
                      on=["location", "anchor_label", "channel"], how="left")
        df["weight"] = df["weight"].fillna(df["pick_quality_score"])
        weight_source = "composite observation weight (w_jk)"
    else:
        df["weight"] = df["pick_quality_score"]
        weight_source = "pick_quality_score (proxy; no weight CSV supplied)"

    df["tier"] = df["weight"].apply(tier_label)
    df["tier"] = pd.Categorical(
        df["tier"],
        categories=["High  (w > 0.9)", "Mid   (0.5–0.9)", "Low   (w < 0.5)"],
        ordered=True,
    )

    print(f"Weight source: {weight_source}")
    return df


# ---------------------------------------------------------------------------
# Statistics printout
# ---------------------------------------------------------------------------

def print_statistics(df: pd.DataFrame, outdir: Path) -> None:
    lines: list[str] = []

    def p(s=""):
        print(s)
        lines.append(str(s))

    p("=" * 80)
    p("SNR CHARACTERISATION  –  STATISTICS SUMMARY")
    p("=" * 80)
    p()

    # ---- Overall ----
    p("OVERALL  (all detections)")
    p("-" * 60)
    p(stat_block(df["snr_dB"], "SNR (dB)"))
    p(stat_block(df["snr_like"], "snr_like (raw)"))
    p(stat_block(df["peak_width_ms"].dropna(), "peak_width_ms"))
    p(stat_block(df["peak_ratio_best_to_second"].replace(np.inf, np.nan).dropna(),
                 "peak_ratio"))
    p(stat_block(df["pick_quality_score"], "pick_quality_score"))
    p()

    # ---- By location ----
    p("BY LOCATION")
    p("-" * 60)
    for loc, g in df.groupby("loc"):
        n_total = len(g)
        n_passed = g["passed_snr_threshold"].sum()
        frac = 100 * n_passed / n_total if n_total else 0
        p(f"  {loc:6s}  {stat_block(g['snr_dB'], 'SNR (dB)')}  "
          f"pass={n_passed}/{n_total} ({frac:.1f}%)")
    p()

    # ---- By anchor / sweep ----
    p("BY ANCHOR / SWEEP")
    p("-" * 60)
    for (loc, sw), g in df.groupby(["loc", "sweep_label"]):
        p(f"  {loc:6s} {sw:20s}  {stat_block(g['snr_dB'], 'SNR (dB)')}")
    p()

    # ---- By weight tier ----
    p("BY WEIGHT TIER")
    p("-" * 60)
    for tier, g in df.groupby("tier", observed=True):
        p(f"  {stat_block(g['snr_dB'], tier)}")
    p()

    # ---- Detection pass rates ----
    p("DETECTION PASS RATE  (passed_snr_threshold)")
    p("-" * 60)
    for loc, g in df.groupby("loc"):
        by_sweep = g.groupby("sweep_label")["passed_snr_threshold"].mean() * 100
        p(f"  {loc:6s}  overall {g['passed_snr_threshold'].mean()*100:.1f}%   "
          f"by sweep: {by_sweep.round(1).to_dict()}")
    p()

    # ---- Tier fractions ----
    p("TIER FRACTIONS")
    p("-" * 60)
    tier_counts = df["tier"].value_counts()
    for tier, cnt in tier_counts.items():
        p(f"  {tier:25s} : {cnt:6d}  ({100*cnt/len(df):.1f}%)")
    p()
    p("=" * 80)

    out_txt = outdir / "snr_statistics.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [saved] {out_txt}")


# ---------------------------------------------------------------------------
# Figure 1: Overall SNR distribution
# ---------------------------------------------------------------------------

def fig_overall_distribution(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Overall SNR Distribution – All Detections", fontsize=15, fontweight="bold")

    snr = df["snr_dB"]
    raw = df["snr_like"]

    for ax, data, xlabel, title in [
        (axes[0], snr,  "SNR  [dB]  (peak / noise floor)",     "SNR in dB  (20·log10(peak/median))"),
        (axes[1], raw,  "snr_like  [(peak−median)/MAD]",       "Raw snr_like"),
    ]:
        ax.hist(data, bins=80, color="#2c7bb6", edgecolor="white", linewidth=0.3, alpha=0.85)
        med = data.median()
        p10 = data.quantile(0.10)
        p90 = data.quantile(0.90)
        ax.axvline(med, color="#d62728", lw=1.8, ls="--", label=f"Median = {med:.1f}")
        ax.axvline(p10, color=GREY,      lw=1.2, ls=":",  label=f"P10 = {p10:.1f}")
        ax.axvline(p90, color=GREY,      lw=1.2, ls=":",  label=f"P90 = {p90:.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y")
        # Annotate n
        ax.text(0.97, 0.97, f"n = {len(data):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=GREY)

    fig.tight_layout()
    save(fig, outdir / "F1_overall_snr_distribution", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: SNR by location – violin + strip
# ---------------------------------------------------------------------------

def fig_by_location(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    locs = sorted(df["loc"].unique())
    data_by_loc = {loc: df.loc[df["loc"] == loc, "snr_dB"].values for loc in locs}

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("SNR Distribution by Source Location", fontsize=15, fontweight="bold")

    positions = np.arange(1, len(locs) + 1)
    parts = ax.violinplot(
        [data_by_loc[l] for l in locs],
        positions=positions,
        showmedians=False,
        showextrema=False,
        widths=0.7,
    )
    for i, (pc, loc) in enumerate(zip(parts["bodies"], locs)):
        colour = LOC_PALETTE.get(loc, "#888888")
        pc.set_facecolor(colour)
        pc.set_alpha(0.55)
        pc.set_edgecolor(colour)

    # Strip plot (jittered)
    rng = np.random.default_rng(42)
    for i, loc in enumerate(locs):
        vals = data_by_loc[loc]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        colour = LOC_PALETTE.get(loc, "#888888")
        ax.scatter(positions[i] + jitter, vals,
                   s=4, alpha=0.35, color=colour, linewidths=0)

    # Median ticks
    for i, loc in enumerate(locs):
        med = np.median(data_by_loc[loc])
        ax.hlines(med, positions[i] - 0.22, positions[i] + 0.22,
                  colors="black", linewidths=2.5, zorder=6,
                  label="Median" if i == 0 else "")
        ax.text(positions[i], med + 0.6, f"{med:.1f}", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold")

    # Annotate n and pass rate
    for i, loc in enumerate(locs):
        g = df[df["loc"] == loc]
        n = len(g)
        pass_frac = g["passed_snr_threshold"].mean() * 100
        ax.text(positions[i], ax.get_ylim()[0] if ax.get_ylim()[0] > -40 else -38,
                f"n={n}\n{pass_frac:.0f}% pass", ha="center", va="top",
                fontsize=8.5, color=GREY)

    ax.set_xticks(positions)
    ax.set_xticklabels([l.upper() for l in locs], fontsize=12)
    ax.set_xlabel("Source Location")
    ax.set_ylabel("SNR  [dB]")
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, outdir / "F2_snr_by_location", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: SNR by weight tier – violin + strip + CDF inset
# ---------------------------------------------------------------------------

def fig_by_tier(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    tiers = ["High  (w > 0.9)", "Mid   (0.5–0.9)", "Low   (w < 0.5)"]
    data_by_tier = {t: df.loc[df["tier"] == t, "snr_dB"].values for t in tiers}
    colours = [TIER_COLOURS[t] for t in tiers]

    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1.0], wspace=0.32)
    ax_main = fig.add_subplot(gs[0])
    ax_cdf  = fig.add_subplot(gs[1])
    fig.suptitle("SNR Distribution by Weight Tier",
                 fontsize=14, fontweight="bold")

    positions = np.arange(1, 4)
    parts = ax_main.violinplot(
        [data_by_tier[t] for t in tiers],
        positions=positions,
        showmedians=False,
        showextrema=False,
        widths=0.65,
    )
    for pc, col in zip(parts["bodies"], colours):
        pc.set_facecolor(col)
        pc.set_alpha(0.50)
        pc.set_edgecolor(col)

    rng = np.random.default_rng(0)
    for i, (t, col) in enumerate(zip(tiers, colours)):
        vals = data_by_tier[t]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax_main.scatter(positions[i] + jitter, vals,
                        s=5, alpha=0.30, color=col, linewidths=0)

    for i, t in enumerate(tiers):
        med = np.median(data_by_tier[t])
        ax_main.hlines(med, positions[i] - 0.22, positions[i] + 0.22,
                       colors="black", linewidths=2.5, zorder=6)
        ax_main.text(positions[i], med + 0.5, f"med={med:.1f} dB",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        n = len(data_by_tier[t])
        ax_main.text(positions[i], ax_main.get_ylim()[0] if ax_main.get_ylim()[0] > -40 else -38,
                     f"n = {n:,}", ha="center", va="top", fontsize=9, color=GREY)

    ax_main.set_xticks(positions)
    ax_main.set_xticklabels(["High\n(w > 0.9)", "Mid\n(0.5–0.9)", "Low\n(w < 0.5)"], fontsize=12)
    ax_main.set_xlabel("Weight tier")
    ax_main.set_ylabel("SNR  [dB]")
    ax_main.grid(axis="y")

    # CDF panel
    x_all = np.linspace(df["snr_dB"].min() - 1, df["snr_dB"].max() + 1, 500)
    for t, col in zip(tiers, colours):
        vals = np.sort(data_by_tier[t])
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ax_cdf.plot(vals, cdf, color=col, lw=2,
                    label=t.strip().split()[0])   # "High" / "Mid" / "Low"
    ax_cdf.set_xlabel("SNR  [dB]")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title("CDF by tier")
    ax_cdf.legend(title="Tier", loc="lower right")
    ax_cdf.grid()
    ax_cdf.set_ylim(0, 1)

    fig.tight_layout()
    save(fig, outdir / "F3_snr_by_tier", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Sub-component scatter matrix
# ---------------------------------------------------------------------------

def fig_scatter_matrix(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    cols = {
        "snr_dB":                  "SNR [dB]",
        "snr_like":                "snr_like\n[(peak−med)/MAD]",
        "peak_ratio_best_to_second": "Peak ratio\n(best/2nd)",
        "peak_width_ms":           "Peak width\n[ms]",
        "pick_quality_score":      "pick_quality\nscore",
    }
    keys   = list(cols.keys())
    labels = list(cols.values())
    n = len(keys)

    # Cap outliers for ratio column
    dfc = df[keys + ["loc"]].copy()
    dfc["peak_ratio_best_to_second"] = dfc["peak_ratio_best_to_second"].clip(upper=20)

    fig, axes = plt.subplots(n, n, figsize=(14, 13))
    fig.suptitle("Matched-Filter Quality Sub-Component Scatter Matrix",
                 fontsize=14, fontweight="bold", y=1.01)

    rng = np.random.default_rng(7)
    # Subsample for speed
    idx = rng.choice(len(dfc), size=min(5000, len(dfc)), replace=False)
    sub = dfc.iloc[idx]

    for i, (ki, li) in enumerate(zip(keys, labels)):
        for j, (kj, lj) in enumerate(zip(keys, labels)):
            ax = axes[i][j]
            if i == j:
                ax.hist(dfc[ki].dropna(), bins=40,
                        color="#2c7bb6", edgecolor="white", linewidth=0.2, alpha=0.8)
                ax.set_ylabel("")
            else:
                for loc, grp in sub.groupby("loc"):
                    col = LOC_PALETTE.get(loc, "#888888")
                    ax.scatter(grp[kj], grp[ki], s=3, alpha=0.3,
                               color=col, linewidths=0)
            if i == n - 1:
                ax.set_xlabel(lj, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(li, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    # Legend
    handles = [mpl.patches.Patch(color=LOC_PALETTE.get(l, "#888"), label=l.upper())
               for l in sorted(dfc["loc"].unique())]
    fig.legend(handles=handles, title="Location", loc="upper right",
               bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.tight_layout()
    save(fig, outdir / "F4_scatter_matrix", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Peak ratio vs SNR
# ---------------------------------------------------------------------------

def fig_peak_ratio_vs_snr(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Matched-Filter Ambiguity: Peak Ratio vs SNR",
                 fontsize=14, fontweight="bold")

    dfc = df.copy()
    dfc["peak_ratio_best_to_second"] = dfc["peak_ratio_best_to_second"].clip(upper=20)

    rng = np.random.default_rng(3)
    idx = rng.choice(len(dfc), size=min(8000, len(dfc)), replace=False)
    sub = dfc.iloc[idx]

    for loc, grp in sub.groupby("loc"):
        col = LOC_PALETTE.get(loc, "#888888")
        ax.scatter(grp["snr_dB"], grp["peak_ratio_best_to_second"],
                   s=8, alpha=0.35, color=col, linewidths=0, label=loc.upper())

    ax.axhline(2.0, color=GREY, lw=1.2, ls="--",
               label="ratio = 2 (unambiguous threshold)")
    ax.set_xlabel("SNR  [dB]  (peak / noise floor)")
    ax.set_ylabel("Peak ratio  (best / 2nd-best)  [clipped at 20]")
    ax.legend(title="Location", ncol=2)
    ax.grid()

    # Annotate fraction unambiguous
    n_unamb = (df["peak_ratio_best_to_second"] >= 2.0).sum()
    ax.text(0.97, 0.05,
            f"ratio ≥ 2 : {n_unamb:,} / {len(df):,} = {100*n_unamb/len(df):.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=GREY)

    fig.tight_layout()
    save(fig, outdir / "F5_peak_ratio_vs_snr", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Peak width vs SNR
# ---------------------------------------------------------------------------

def fig_peak_width_vs_snr(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    dfc = df.dropna(subset=["peak_width_ms"]).copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Peak Sharpness vs SNR  (narrower = sharper onset = better timing)",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(5)
    idx = rng.choice(len(dfc), size=min(8000, len(dfc)), replace=False)
    sub = dfc.iloc[idx]

    for loc, grp in sub.groupby("loc"):
        col = LOC_PALETTE.get(loc, "#888888")
        ax.scatter(grp["snr_dB"], grp["peak_width_ms"],
                   s=8, alpha=0.35, color=col, linewidths=0, label=loc.upper())

    ax.set_xlabel("SNR  [dB]")
    ax.set_ylabel("MF peak half-power width  [ms]")
    ax.set_yscale("log")
    ax.legend(title="Location", ncol=2)
    ax.grid(which="both")

    fig.tight_layout()
    save(fig, outdir / "F6_peak_width_vs_snr", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Heatmap – median SNR per (location × sweep)
# ---------------------------------------------------------------------------

def fig_snr_heatmap(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    pivot = df.groupby(["loc", "sweep_label"])["snr_dB"].median().unstack(fill_value=np.nan)
    pivot_pass = (df.groupby(["loc", "sweep_label"])["passed_snr_threshold"]
                  .mean().unstack(fill_value=np.nan) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(pivot) * 0.9 + 1.5)))
    fig.suptitle("Per-Transmission Quality Map", fontsize=14, fontweight="bold")

    for ax, data, title, fmt_str, cmap in [
        (axes[0], pivot,      "Median SNR  [dB]",       ".1f", "RdYlGn"),
    ]:
        im = ax.imshow(data.values, aspect="auto", cmap=cmap,
                       vmin=data.values[np.isfinite(data.values)].min(),
                       vmax=data.values[np.isfinite(data.values)].max())
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([l.upper() for l in data.index], fontsize=10)
        ax.set_xlabel("Sweep / Anchor")
        ax.set_ylabel("Location")
        ax.set_title(title)
        # Annotate cells
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data.values[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:{fmt_str}}", ha="center", va="center",
                            fontsize=7.5, color="black",
                            fontweight="bold" if val == data.values[r, :].max() else "normal")

    fig.tight_layout()
    save(fig, outdir / "F7_snr_heatmap", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Per-channel SNR profile
# ---------------------------------------------------------------------------

def fig_snr_profile(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    locs = sorted(df["loc"].unique())
    n = len(locs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("SNR Profile Along the Cable – Per Location",
                 fontsize=13, fontweight="bold")

    for ax, loc in zip(axes, locs):
        g = df[df["loc"] == loc]
        colour = LOC_PALETTE.get(loc, "#888888")

        # Median and std across sweeps for each channel
        ch_stats = g.groupby("channel")["snr_dB"].agg(["median", "std", "count"])
        ch = ch_stats.index.values
        med = ch_stats["median"].values
        std = ch_stats["std"].fillna(0).values

        ax.fill_between(ch, med - std, med + std, color=colour, alpha=0.20)
        ax.plot(ch, med, color=colour, lw=1.4, label="Median SNR across sweeps")

        ax.set_ylabel("SNR [dB]")
        ax.set_title(f"Location {loc.upper()}", fontsize=12, loc="left")
        ax.grid(axis="y", alpha=0.4)

        # Mark median SNR
        global_med = np.median(g["snr_dB"])
        ax.axhline(global_med, color="black", lw=0.9, ls="--", alpha=0.5,
                   label=f"Overall median {global_med:.1f} dB")
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("DAS channel index")
    fig.tight_layout()
    save(fig, outdir / "F8_snr_profile_per_location", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: CDF comparison by tier  (standalone, larger)
# ---------------------------------------------------------------------------

def fig_cdf(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    tiers = ["High  (w > 0.9)", "Mid   (0.5–0.9)", "Low   (w < 0.5)"]
    colours = [TIER_COLOURS[t] for t in tiers]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("CDF of SNR by Weight Tier", fontsize=14, fontweight="bold")

    for t, col in zip(tiers, colours):
        vals = np.sort(df.loc[df["tier"] == t, "snr_dB"].values)
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        n    = len(vals)
        med  = np.median(vals)
        ax.plot(vals, cdf, color=col, lw=2.2,
                label=f"{t.strip()}  (n={n:,}, med={med:.1f} dB)")

    ax.set_xlabel("SNR  [dB]  (peak / noise floor)")
    ax.set_ylabel("Cumulative fraction of detections")
    ax.legend(title="Weight tier", loc="lower right", fontsize=11)
    ax.grid()
    ax.set_ylim(0, 1)
    ax.set_xlim(left=df["snr_dB"].quantile(0.01) - 1)

    # Add vertical lines at key quantiles for the high tier
    high = np.sort(df.loc[df["tier"] == "High  (w > 0.9)", "snr_dB"].values)
    for q, ql in [(0.25, "Q1"), (0.5, "med"), (0.75, "Q3")]:
        v = np.quantile(high, q)
        ax.axvline(v, color=TIER_COLOURS["High  (w > 0.9)"],
                   lw=0.9, ls=":", alpha=0.7)
        ax.text(v + 0.2, 0.05 + q * 0.1, f"{ql}={v:.1f}", fontsize=8,
                color=TIER_COLOURS["High  (w > 0.9)"])

    fig.tight_layout()
    save(fig, outdir / "F9_cdf_by_tier", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: Per-location summary table
# ---------------------------------------------------------------------------

def fig_summary_table(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    locs = sorted(df["loc"].unique())

    rows = []
    for loc in locs:
        g = df[df["loc"] == loc]
        n_total = len(g)
        n_ch = g["channel"].nunique()
        n_sweeps = g["sweep_label"].nunique()
        n_pass = int(g["passed_snr_threshold"].sum())
        pass_pct = 100 * n_pass / n_total if n_total else 0
        med_snr = g["snr_dB"].median()
        p10_snr = g["snr_dB"].quantile(0.10)
        p90_snr = g["snr_dB"].quantile(0.90)
        med_width = g["peak_width_ms"].median()
        med_ratio = g["peak_ratio_best_to_second"].replace(np.inf, np.nan).median()
        med_pqs = g["pick_quality_score"].median()
        n_high = (g["tier"] == "High  (w > 0.9)").sum()
        pct_high = 100 * n_high / n_total if n_total else 0

        rows.append({
            "Location": loc.upper(),
            "N detections": f"{n_total:,}",
            "N channels": f"{n_ch:,}",
            "N sweeps": str(n_sweeps),
            "Pass rate": f"{pass_pct:.1f}%",
            "Med SNR [dB]": f"{med_snr:.1f}",
            "P10 SNR [dB]": f"{p10_snr:.1f}",
            "P90 SNR [dB]": f"{p90_snr:.1f}",
            "Med width [ms]": f"{med_width:.1f}" if np.isfinite(med_width) else "—",
            "Med ratio": f"{med_ratio:.2f}" if np.isfinite(med_ratio) else "—",
            "Med PQS": f"{med_pqs:.3f}",
            "% High tier": f"{pct_high:.1f}%",
        })

    tdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(16, len(tdf.columns) * 1.4), len(tdf) * 0.65 + 2.0))
    ax.axis("off")
    fig.suptitle("Per-Location SNR Summary", fontsize=14, fontweight="bold", y=0.98)

    col_labels = list(tdf.columns)
    cell_text  = tdf.values.tolist()

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8)

    # Shade header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c7bb6")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Shade alternate rows
    for i in range(1, len(tdf) + 1):
        colour = "#f0f4fa" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(colour)

    fig.tight_layout()
    save(fig, outdir / "F10_summary_table", fmt)
    plt.close(fig)

    # Also save as CSV for easy LaTeX import
    tdf.to_csv(outdir / "snr_per_location_table.csv", index=False)
    print(f"  [saved] {outdir / 'snr_per_location_table.csv'}")


# ---------------------------------------------------------------------------
# Figure 11: Detection-rate heatmap (passed_snr_threshold)
# ---------------------------------------------------------------------------

def fig_detection_rate(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    pivot = (df.groupby(["loc", "sweep_label"])["passed_snr_threshold"]
               .mean().unstack(fill_value=np.nan) * 100)

    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 1.2),
                                    max(4, pivot.shape[0] * 0.85 + 1.5)))
    fig.suptitle("Detection Pass-Rate per (Location × Sweep)  [% channels passing SNR threshold]",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, shrink=0.85, label="Pass rate [%]")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([l.upper() for l in pivot.index], fontsize=11)
    ax.set_xlabel("Sweep / Anchor label")
    ax.set_ylabel("Location")

    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, color="black")

    fig.tight_layout()
    save(fig, outdir / "F11_detection_rate_heatmap", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: pick_quality_score vs snr_like
# ---------------------------------------------------------------------------

def fig_pqs_vs_snr(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("pick_quality_score vs Raw SNR – Detector Score Calibration",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(9)
    idx = rng.choice(len(df), size=min(10000, len(df)), replace=False)
    sub = df.iloc[idx]

    # Left: coloured by location
    ax = axes[0]
    for loc, grp in sub.groupby("loc"):
        col = LOC_PALETTE.get(loc, "#888")
        ax.scatter(grp["snr_dB"], grp["pick_quality_score"],
                   s=6, alpha=0.35, color=col, linewidths=0, label=loc.upper())
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("pick_quality_score  [0–1]")
    ax.set_title("Coloured by location")
    ax.legend(title="Location", ncol=2, fontsize=9)
    ax.grid(alpha=0.4)

    # Right: coloured by tier
    ax = axes[1]
    for tier in ["High  (w > 0.9)", "Mid   (0.5–0.9)", "Low   (w < 0.5)"]:
        grp = sub[sub["tier"] == tier]
        col = TIER_COLOURS[tier]
        ax.scatter(grp["snr_dB"], grp["pick_quality_score"],
                   s=6, alpha=0.35, color=col, linewidths=0,
                   label=tier.strip().split()[0])
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("pick_quality_score  [0–1]")
    ax.set_title("Coloured by weight tier")
    ax.legend(title="Tier", fontsize=9)
    ax.grid(alpha=0.4)

    # Score thresholds from thesis (0.45*snr + 0.25*ratio + 0.20*timing + 0.10*edge)
    # snr_score saturates at snr_like=10 → snr_dB ≈ 20*log10(1+10/…) depends on baseline
    # Just draw horizontal lines at 0.5 and 0.9 (the tier boundaries)
    for axi in axes:
        axi.axhline(0.9, color=TIER_COLOURS["High  (w > 0.9)"],
                    lw=1.2, ls="--", alpha=0.7)
        axi.axhline(0.5, color=TIER_COLOURS["Low   (w < 0.5)"],
                    lw=1.2, ls="--", alpha=0.7)

    fig.tight_layout()
    save(fig, outdir / "F12_pqs_vs_snr", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13: SNR distribution per anchor sweep, small multiples
# ---------------------------------------------------------------------------

def fig_per_sweep(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    """One histogram per (loc, sweep_label) small-multiples grid."""
    locs = sorted(df["loc"].unique())
    sweeps = sorted(df["sweep_label"].unique())

    nrow = len(locs)
    ncol = len(sweeps)

    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(max(14, ncol * 2.5), max(10, nrow * 2.2)),
                              sharex=True, sharey=False)
    if nrow == 1:
        axes = axes[np.newaxis, :]
    if ncol == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("SNR Distribution per (Location × Sweep)  – Small Multiples",
                 fontsize=13, fontweight="bold", y=1.01)

    bins = np.linspace(df["snr_dB"].quantile(0.01),
                       df["snr_dB"].quantile(0.99), 35)

    for i, loc in enumerate(locs):
        for j, sw in enumerate(sweeps):
            ax = axes[i][j]
            g = df[(df["loc"] == loc) & (df["sweep_label"] == sw)]
            colour = LOC_PALETTE.get(loc, "#888")
            if len(g) > 0:
                ax.hist(g["snr_dB"], bins=bins, color=colour,
                        edgecolor="none", alpha=0.75)
                med = g["snr_dB"].median()
                ax.axvline(med, color="black", lw=1.2, ls="--")
                ax.text(0.97, 0.97, f"med={med:.1f}\nn={len(g)}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="black")
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color=GREY)
                ax.set_facecolor("#f5f5f5")

            if i == 0:
                ax.set_title(sw, fontsize=8, rotation=30, ha="left")
            if j == 0:
                ax.set_ylabel(loc.upper(), fontsize=10)
            ax.tick_params(labelsize=7)

    fig.text(0.5, -0.01, "SNR  [dB]", ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, outdir / "F13_snr_per_sweep_small_multiples", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNR characterisation plots from all_locations_detections.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    parser.add_argument("--csv",        type=Path, required=True,
                        help="Path to all_locations_detections.csv")
    parser.add_argument("--weight_csv", type=Path, default=None,
                        help="Optional CSV with composite weights (location, anchor_label, channel, weight)")
    parser.add_argument("--outdir",     type=Path, default=Path("figures_snr"),
                        help="Output directory for figures (default: figures_snr/)")
    parser.add_argument("--fmt",        choices=["pdf", "png"], default="pdf",
                        help="Output format: pdf (vector, default) or png (300 dpi)")
    parser.add_argument("--channel_min", type=int, default=0,
                        help="First in-water channel (inclusive). "
                             "Rows below this are dropped before plotting. "
                             "Match your [trust] channel_min, e.g. 348.")
    parser.add_argument("--channel_max", type=int, default=999_999,
                        help="Last in-water channel (inclusive). "
                             "Rows above this are dropped before plotting. "
                             "Match your [trust] channel_max, e.g. 2267.")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"ERROR: CSV not found: {args.csv}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SNR Characterisation – Thesis Figure Generator")
    print(f"{'='*60}")
    print(f"Input CSV    : {args.csv}")
    print(f"Weight CSV   : {args.weight_csv or '(none – using pick_quality_score as proxy)'}")
    print(f"Output dir   : {args.outdir}")
    print(f"Format       : {args.fmt.upper()}")
    print(f"Channel range: [{args.channel_min}, {args.channel_max}]\n")

    df = load_data(args.csv, args.weight_csv,
                   channel_min=args.channel_min,
                   channel_max=args.channel_max)

    print_statistics(df, args.outdir)

    print("\nGenerating figures…")
    fig_overall_distribution(df, args.outdir, args.fmt)   # F1
    fig_by_location(df, args.outdir, args.fmt)             # F2
    fig_by_tier(df, args.outdir, args.fmt)                 # F3
    fig_scatter_matrix(df, args.outdir, args.fmt)          # F4
    fig_peak_ratio_vs_snr(df, args.outdir, args.fmt)       # F5
    fig_peak_width_vs_snr(df, args.outdir, args.fmt)       # F6
    fig_snr_heatmap(df, args.outdir, args.fmt)             # F7
    fig_snr_profile(df, args.outdir, args.fmt)             # F8
    fig_cdf(df, args.outdir, args.fmt)                     # F9
    fig_summary_table(df, args.outdir, args.fmt)           # F10
    fig_detection_rate(df, args.outdir, args.fmt)          # F11
    fig_pqs_vs_snr(df, args.outdir, args.fmt)              # F12
    fig_per_sweep(df, args.outdir, args.fmt)               # F13

    print(f"\nAll done. {len(list(args.outdir.glob('*.' + args.fmt)))} figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()