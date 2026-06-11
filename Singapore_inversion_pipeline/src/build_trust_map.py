from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_toml, ensure_dir, path_from_cfg


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})


def ensure_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def normalize_anchor_column(df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["anchor_label", "anchor_name", "anchor", "anchor_id", "replicate"]
    for c in candidates:
        if c in df.columns:
            out = df.copy()
            out["anchor"] = out[c].astype(str)
            return out
    raise ValueError(f"Could not find an anchor column. Tried: {candidates}")


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def smooth_series(
    channels: np.ndarray,
    values_ms: np.ndarray,
    med_win: int,
    mean_win: int,
    min_periods: int,
) -> np.ndarray:
    s = pd.Series(values_ms, index=channels, dtype=float)
    s = s.interpolate(method="index", limit=12, limit_direction="both")
    s = s.rolling(med_win, center=True, min_periods=min_periods).median()
    s = s.rolling(mean_win, center=True, min_periods=min_periods).mean()
    return s.to_numpy(dtype=float)


def format_location(raw_location: str) -> str:
    """'loc2_tx3' -> 'Location 2'"""
    m = re.match(r"loc(\d+)", str(raw_location))
    return f"Location {m.group(1)}" if m else str(raw_location)


def format_sweep(raw_anchor: str, fallback_index: int) -> str:
    """'lfm35_45_rep1' -> 'Sweep 1'"""
    m = re.search(r"rep(\d+)", str(raw_anchor))
    return f"Sweep {m.group(1)}" if m else f"Sweep {fallback_index + 1}"


def standardize_input(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        [
            "location", "reference_channel", "anchor_index", "channel",
            "peak_time_s_from_sequence_start", "snr_like",
            "passed_snr_threshold", "near_window_edge", "pick_quality_score",
        ],
    )

    out = df.copy()
    out = normalize_anchor_column(out)
    out = out.rename(columns={"peak_time_s_from_sequence_start": "t_peak_s", "snr_like": "snr"})

    out["channel"] = out["channel"].astype(int)
    out["reference_channel"] = out["reference_channel"].astype(int)
    out["anchor_index"] = out["anchor_index"].astype(int)
    out["t_peak_s"] = pd.to_numeric(out["t_peak_s"], errors="coerce")
    out["snr"] = pd.to_numeric(out["snr"], errors="coerce")
    out["pick_quality_score"] = pd.to_numeric(out["pick_quality_score"], errors="coerce")
    out["passed_snr_threshold"] = out["passed_snr_threshold"].fillna(False).astype(bool)
    out["near_window_edge"] = out["near_window_edge"].fillna(False).astype(bool)

    out["relative_to_reference_s"] = np.nan
    for (loc, anchor_idx), g in out.groupby(["location", "anchor_index"], sort=False):
        ref_ch = int(g["reference_channel"].iloc[0])
        ref_rows = g[g["channel"] == ref_ch]
        if ref_rows.empty:
            continue
        t_ref = pd.to_numeric(ref_rows["t_peak_s"], errors="coerce").iloc[0]
        if not np.isfinite(t_ref):
            continue
        idx = (out["location"] == loc) & (out["anchor_index"] == anchor_idx)
        out.loc[idx, "relative_to_reference_s"] = out.loc[idx, "t_peak_s"] - t_ref

    out["is_valid"] = (
        (out["snr"] > 0)
        & out["passed_snr_threshold"]
        & (~out["near_window_edge"])
        & np.isfinite(out["t_peak_s"])
        & np.isfinite(out["relative_to_reference_s"])
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build smooth arrival curves and sweep disagreement from detector output."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_toml(args.config)
    tcfg = cfg["trust"]

    med_win = ensure_odd(int(tcfg["rolling_median_win"]))
    mean_win = ensure_odd(int(tcfg["rolling_mean_win"]))
    min_periods = max(3, int(tcfg.get("min_periods", max(med_win // 5, mean_win // 5))))

    in_csv = path_from_cfg(cfg, "raw_detection_output_dir") / "all_locations_detections.csv"
    outdir = ensure_dir(path_from_cfg(cfg, "trust_output_dir"))

    df = pd.read_csv(in_csv)
    df = standardize_input(df)
    df = df[
        (df["channel"] >= int(tcfg["channel_min"]))
        & (df["channel"] <= int(tcfg["channel_max"]))
    ].copy()
    if df.empty:
        raise ValueError("No rows left after channel filtering.")

    df["offset_ms"] = 1000.0 * df["relative_to_reference_s"].astype(float)
    df["base_valid"] = (
        df["is_valid"]
        & (~df["near_window_edge"])
        & np.isfinite(df["offset_ms"])
        & np.isfinite(df["snr"])
    )

    channels = np.arange(int(tcfg["channel_min"]), int(tcfg["channel_max"]) + 1)
    locations = sorted(df["location"].astype(str).unique())
    anchors = sorted(df["anchor"].astype(str).unique())

    # ------------------------------------------------------------------
    # Build per-(location, anchor) smooth curves
    # ------------------------------------------------------------------
    smooth_rows: list[pd.DataFrame] = []
    for loc in locations:
        loc_df = df[df["location"] == loc].copy()
        for anchor in anchors:
            grp = loc_df[loc_df["anchor"] == anchor].copy().sort_values("channel")
            merged = pd.DataFrame({"channel": channels}).merge(grp, on="channel", how="left")
            merged["location"] = loc
            merged["anchor"] = anchor
            merged["base_valid"] = merged["base_valid"].fillna(False).astype(bool)
            merged["near_window_edge"] = merged["near_window_edge"].fillna(False).astype(bool)

            vals = merged["offset_ms"].to_numpy(dtype=float)
            valid_mask = merged["base_valid"].to_numpy(dtype=bool)
            smooth_input = np.where(valid_mask, vals, np.nan)

            if np.sum(np.isfinite(smooth_input)) >= int(tcfg["min_valid_points_per_group"]):
                smooth_ms = smooth_series(channels, smooth_input, med_win, mean_win, min_periods)
            else:
                smooth_ms = np.full_like(channels, np.nan, dtype=float)

            smooth_rows.append(
                pd.DataFrame({
                    "location": loc,
                    "anchor": anchor,
                    "channel": channels,
                    "offset_ms": vals,
                    "smooth_offset_ms": smooth_ms,
                    "base_valid": valid_mask,
                    "pick_quality_score": merged["pick_quality_score"].to_numpy(dtype=float),
                    "snr_like": merged["snr"].to_numpy(dtype=float),
                })
            )

    smooth_df = pd.concat(smooth_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Location median smooth and sweep disagreement
    # ------------------------------------------------------------------
    loc_median = (
        smooth_df.groupby(["location", "channel"], as_index=False)
        .agg(location_median_smooth_ms=("smooth_offset_ms", lambda s: np.nanmedian(s.to_numpy(dtype=float))))
    )
    smooth_df = smooth_df.merge(loc_median, on=["location", "channel"], how="left")

    if len(anchors) >= 2:
        wide = smooth_df.pivot_table(
            index=["location", "channel"],
            columns="anchor",
            values="smooth_offset_ms",
            aggfunc="first",
        )
        anchor_list = sorted(wide.columns.tolist())
        a1, a2 = anchor_list[0], anchor_list[1]
        disagreement = (
            (wide[a1] - wide[a2]).abs().rename("anchor_disagreement_ms").reset_index()
        )
    else:
        disagreement = pd.DataFrame(columns=["location", "channel", "anchor_disagreement_ms"])

    smooth_df = smooth_df.merge(disagreement, on=["location", "channel"], how="left")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    smooth_df.to_csv(outdir / "channel_smooth_curves.csv", index=False)
    loc_median.to_csv(outdir / "location_median_smooth.csv", index=False)
    disagreement.to_csv(outdir / "anchor_disagreement.csv", index=False)

    # ------------------------------------------------------------------
    # Diagnostic: smoothed arrival curves
    # ------------------------------------------------------------------
    plot_dir = ensure_dir(outdir / "plots")
    n_loc = len(locations)
    fig, axes = plt.subplots(n_loc, 1, figsize=(16, 3 * n_loc), sharex=True)
    if n_loc == 1:
        axes = [axes]

    sweep_colors = ["#E87D2B", "#C62828", "#4878CF"]
    median_color = "#5B2C8C"

    for ax, loc in zip(axes, locations):
        loc_df = smooth_df[smooth_df["location"] == loc]
        raw = loc_df[loc_df["base_valid"] == True]
        ax.scatter(raw["channel"], raw["offset_ms"],
                   s=2, alpha=0.15, color="teal", rasterized=True, label="_nolegend_")

        for i, anchor in enumerate(anchors):
            sweep_label = format_sweep(anchor, i)
            a_df = loc_df[loc_df["anchor"] == anchor].sort_values("channel")
            ax.plot(a_df["channel"], a_df["smooth_offset_ms"],
                    color=sweep_colors[i % len(sweep_colors)],
                    linewidth=1.4, label=f"{sweep_label} smooth")

        med_df = loc_df.drop_duplicates("channel").sort_values("channel")
        ax.plot(med_df["channel"], med_df["location_median_smooth_ms"],
                color=median_color, linewidth=1.4, linestyle="--", label="Median smooth")

        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_ylim(-100, 400)
        ax.set_ylabel("Offset (ms)")
        ax.set_title(format_location(loc))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Channel")
    fig.suptitle("Smoothed arrival curves by location")
    fig.tight_layout()
    fig.savefig(plot_dir / "smoothed_arrival_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(plot_dir / "smoothed_arrival_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Diagnostic: sweep disagreement by location
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(n_loc, 1, figsize=(16, 2.5 * n_loc), sharex=True)
    if n_loc == 1:
        axes = [axes]

    for ax, loc in zip(axes, locations):
            dis_df = disagreement[disagreement["location"] == loc].sort_values("channel")
            ax.plot(dis_df["channel"], dis_df["anchor_disagreement_ms"],
                    color="#C62828", linewidth=1.0)
            ax.set_ylabel("|S_d| (ms)")
            ax.set_title(format_location(loc))
            ax.grid(True, alpha=0.3)

            # add this block
            vals = dis_df["anchor_disagreement_ms"].dropna()
            if len(vals) > 0:
                lowerpct=vals.quantile(0.25)
                med = vals.median()
                upperpct = vals.quantile(0.75)
                ax.text(0.98, 0.95,
                        f"25th pct = {lowerpct:.1f} ms\nmedian = {med:.1f} ms\n75th pct = {upperpct:.1f}ms",
                        transform=ax.transAxes,
                        fontsize=13, va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                ec="0.75", alpha=0.8))

    axes[-1].set_xlabel("Channel")
    fig.suptitle("Sweep disagreement by location")
    fig.tight_layout()
    fig.savefig(plot_dir / "anchor_disagreement.png", dpi=300, bbox_inches="tight")
    fig.savefig(plot_dir / "anchor_disagreement.pdf",bbox_inches="tight")
    plt.close(fig)

    all_vals = disagreement["anchor_disagreement_ms"].dropna()
    print(f"\nSweep disagreement across all locations:")
    print(f"  Overall median: {all_vals.median():.1f} ms")
    print(f"  Overall mean:   {all_vals.mean():.1f} ms")
    print(f"  Overall 75pct:   {all_vals.quantile(0.75):.1f} ms")
    print(f"  Overall 25pct:   {all_vals.quantile(0.25):.1f} ms")
    for loc in locations:
        vals = disagreement[disagreement["location"] == loc]["anchor_disagreement_ms"].dropna()
        print(f"  {format_location(loc):12s}  median = {vals.median():.1f} ms  "
              f"mean = {vals.mean():.1f} ms  "
              f"75th pct = {vals.quantile(0.75):.1f} ms  "  f"25th pct = {vals.quantile(0.25):.1f} ms")

    print(f"Saved outputs to: {outdir}")
    print(f"  channel_smooth_curves.csv: {len(smooth_df)} rows")
    print(f"  anchor_disagreement.csv:   {len(disagreement)} rows")


if __name__ == "__main__":
    main()
