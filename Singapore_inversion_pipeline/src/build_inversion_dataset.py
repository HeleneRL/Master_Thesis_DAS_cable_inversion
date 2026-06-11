from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymap3d import geodetic2enu

from common import load_toml, ensure_dir, path_from_cfg


# ── Color scheme ──────────────────────────────────────────────────────────────
C_ORANGE = "#E07B39"
C_BLUE   = "#3B7FC4"
C_GREEN  = "#4A9B6F"
C_RED    = "#C43B3B"
C_GRAY   = "#888888"
C_PURPLE = "#8B5CA8"
C_YELLOW = "#E8C534"


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})





# ---------------------------------------------------------------------------
# Naming helpers — single source of truth for reader-facing labels
# ---------------------------------------------------------------------------

def format_location(raw_location: str) -> str:
    """'loc2_tx3' -> 'Location 2'"""
    m = re.match(r"loc(\d+)", str(raw_location))
    return f"Location {m.group(1)}" if m else str(raw_location)


def format_sweep(raw_anchor: str, fallback_index: int = 0) -> str:
    """'lfm35_45_rep1' -> 'Sweep 1'"""
    m = re.search(r"rep(\d+)", str(raw_anchor))
    return f"Sweep {m.group(1)}" if m else f"Sweep {fallback_index + 1}"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csvs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrivals     = pd.read_csv(path_from_cfg(cfg, "raw_detection_output_dir") / "all_locations_detections.csv")
    tx           = pd.read_csv(path_from_cfg(cfg, "transmitter_output_dir")   / "transmission_times_with_tx_positions.csv")
    smooth_df    = pd.read_csv(path_from_cfg(cfg, "trust_output_dir")         / "channel_smooth_curves.csv")
    disagreement = pd.read_csv(path_from_cfg(cfg, "trust_output_dir")         / "anchor_disagreement.csv")
    prior        = pd.read_csv(path_from_cfg(cfg, "prior_output_dir")         / "prior_cable_by_channel.csv")
    return arrivals, tx, smooth_df, disagreement, prior


# ---------------------------------------------------------------------------
# Relative arrival time
# ---------------------------------------------------------------------------

def compute_relative_arrival(arrivals: pd.DataFrame, channel_min: int, channel_max: int) -> pd.DataFrame:
    df = arrivals.copy()
    df = df[(df["channel"] >= channel_min) & (df["channel"] <= channel_max)].copy()
    df["observed_t_s"] = pd.to_numeric(df["peak_time_s_from_sequence_start"], errors="coerce")
    df["observed_dt_ref_s"] = np.nan

    for (loc, anchor), g in df.groupby(["location", "anchor_index"]):
        ref_ch = int(g["reference_channel"].iloc[0])
        ref_rows = g[g["channel"] == ref_ch]
        if ref_rows.empty:
            continue
        t_ref = float(ref_rows["observed_t_s"].iloc[0])
        idx = (df["location"] == loc) & (df["anchor_index"] == anchor)
        df.loc[idx, "observed_dt_ref_s"] = df.loc[idx, "observed_t_s"] - t_ref

    return df


# ---------------------------------------------------------------------------
# Transmitter table
# ---------------------------------------------------------------------------

def prepare_tx_table(tx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in tx.iterrows():
        rows.append({
            "location":          r["location"],
            "anchor_index":      1,
            "anchor_label":      "lfm35_45_rep1",
            "reference_channel": int(r["reference_channel"]),
            "tx_lat":            float(r["tx_lat_peak1"]),
            "tx_lon":            float(r["tx_lon_peak1"]),
            "tx_z_m":            float(r["tx_depth_m"]),
        })
        rows.append({
            "location":          r["location"],
            "anchor_index":      2,
            "anchor_label":      "lfm35_45_rep2",
            "reference_channel": int(r["reference_channel"]),
            "tx_lat":            float(r["tx_lat_peak2"]),
            "tx_lon":            float(r["tx_lon_peak2"]),
            "tx_z_m":            float(r["tx_depth_m"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_all(
    arrivals_rel: pd.DataFrame,
    tx_long: pd.DataFrame,
    smooth_df: pd.DataFrame,
    disagreement: pd.DataFrame,
    prior: pd.DataFrame,
) -> pd.DataFrame:
    df = arrivals_rel.merge(
        tx_long,
        on=["location", "anchor_index", "anchor_label", "reference_channel"],
        how="left",
        validate="many_to_one",
    )

    smooth_small = smooth_df[
        ["location", "anchor", "channel", "smooth_offset_ms", "location_median_smooth_ms"]
    ].copy()
    df = df.merge(
        smooth_small.rename(columns={"anchor": "anchor_label"}),
        on=["location", "anchor_label", "channel"],
        how="left",
    )

    dis_small = disagreement[["location", "channel", "anchor_disagreement_ms"]].copy()
    df = df.merge(dis_small, on=["location", "channel"], how="left")

    prior_small = prior.rename(
        columns={"lat": "prior_lat", "lon": "prior_lon", "depth": "prior_z_m"}
    )[["channel", "prior_lat", "prior_lon", "prior_z_m"]].copy()
    df = df.merge(prior_small, on="channel", how="left", validate="many_to_one")

    passed    = df["passed_snr_threshold"].astype(str).str.upper().eq("TRUE")
    near_edge = df["near_window_edge"].astype(str).str.upper().eq("TRUE")
    df["base_valid"] = passed & (~near_edge)

    return df


# ---------------------------------------------------------------------------
# ENU coordinates
# ---------------------------------------------------------------------------

def add_local_enu_coordinates(df: pd.DataFrame, lat0_deg: float, lon0_deg: float, h0_m: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    tx_e, tx_n, tx_u = geodetic2enu(
        out["tx_lat"].to_numpy(dtype=float), out["tx_lon"].to_numpy(dtype=float),
        out["tx_z_m"].to_numpy(dtype=float), lat0_deg, lon0_deg, h0_m,
    )
    prior_e, prior_n, prior_u = geodetic2enu(
        out["prior_lat"].to_numpy(dtype=float), out["prior_lon"].to_numpy(dtype=float),
        out["prior_z_m"].to_numpy(dtype=float), lat0_deg, lon0_deg, h0_m,
    )
    out["tx_x_m"],    out["tx_y_m"],    out["tx_u_m"]    = tx_e,    tx_n,    tx_u
    out["prior_x_m"], out["prior_y_m"], out["prior_u_m"] = prior_e, prior_n, prior_u
    return out


# ---------------------------------------------------------------------------
# Three-factor weight
# ---------------------------------------------------------------------------

# def make_weight(df: pd.DataFrame, wcfg: dict) -> pd.DataFrame:
#     out = df.copy()

#     sigma_smooth = float(wcfg["sigma_smooth_ms"])
#     sigma_anchor = float(wcfg["sigma_anchor_ms"])
#     pick_floor   = float(wcfg.get("pick_quality_floor", 0.05))
#     min_weight   = float(wcfg.get("use_observation_min_weight", 0.02))

#     q_pick = pd.to_numeric(out["pick_quality_score"], errors="coerce").fillna(pick_floor)
#     q_pick = q_pick.clip(lower=pick_floor, upper=1.0)
#     q_pick = np.where(out["base_valid"].astype(bool), q_pick, pick_floor)

#     observed_ms   = 1000.0 * pd.to_numeric(out["observed_dt_ref_s"], errors="coerce").to_numpy(dtype=float)
#     smooth_ref_ms = pd.to_numeric(out["location_median_smooth_ms"], errors="coerce").to_numpy(dtype=float)
#     residual_ms   = observed_ms - smooth_ref_ms
#     q_smooth = np.exp(-0.5 * (residual_ms / sigma_smooth) ** 2)
#     q_smooth = np.where(np.isfinite(residual_ms), q_smooth, 1.0)

#     disagreement_ms = pd.to_numeric(out["anchor_disagreement_ms"], errors="coerce").to_numpy(dtype=float)
#     q_anchor = np.exp(-0.5 * (disagreement_ms / sigma_anchor) ** 2)
#     q_anchor = np.where(np.isfinite(disagreement_ms), q_anchor, 1.0)

#     w = q_pick * q_smooth * q_anchor
#     w = np.clip(w, 0.0, 1.0)

#     out["q_pick"]   = q_pick
#     out["q_smooth"] = q_smooth
#     out["q_anchor"] = q_anchor
#     out["weight"]   = w

#     out["use_observation"] = (
#         out["observed_dt_ref_s"].notna()
#         & out["tx_lat"].notna()
#         & out["tx_lon"].notna()
#         & out["prior_lat"].notna()
#         & out["prior_lon"].notna()
#         & out["tx_x_m"].notna()
#         & out["tx_y_m"].notna()
#         & out["prior_x_m"].notna()
#         & out["prior_y_m"].notna()
#         & (out["weight"] > min_weight)
#     )

#     return out


def make_weight(df: pd.DataFrame, wcfg: dict) -> pd.DataFrame:
    out = df.copy()

    sigma_smooth = float(wcfg["sigma_smooth_ms"])
    sigma_anchor = float(wcfg["sigma_anchor_ms"])
    pick_floor   = float(wcfg.get("pick_quality_floor", 0.05))
    min_weight   = float(wcfg.get("use_observation_min_weight", 0.02))

    # ------------------------------------------------------------------
    # Factor 1: pick quality — product of three sub-scores (logical AND).
    # Raw ingredients were saved by the detector; we re-apply the same
    # linear ramp scoring functions and combine as a product rather than
    # the weighted sum the detector used internally.
    # Breakpoint values match the detector exactly.
    # ------------------------------------------------------------------
    def score_high_good(x, bad, good):
        return np.clip((x - bad) / (good - bad + 1e-12), 0.0, 1.0)

    def score_low_good(x, good, bad):
        return np.clip((bad - x) / (bad - good + 1e-12), 0.0, 1.0)

    snr_raw  = pd.to_numeric(out["snr_like"],            errors="coerce").fillna(0.0).to_numpy()
    prom_raw = pd.to_numeric(out["peak_ratio_best_to_second"], errors="coerce").fillna(0.0).to_numpy()
    width_ms = pd.to_numeric(out["peak_width_ms"],       errors="coerce").fillna(40.0).to_numpy()
    #near_edge = ensure_bool(out["near_window_edge"]).to_numpy(dtype=bool)

    q_snr   = score_high_good(snr_raw,                   bad=3.0,  good=10.0)
    q_prom  = score_high_good(np.minimum(prom_raw, 8.0), bad=1.05, good=1.5)
    q_sharp = np.where(
        np.isfinite(width_ms),
        score_low_good(width_ms, good=4.0, bad=12.0),
        0.0,
    )

    # Near-window-edge detections are zeroed — the edge guard from the
    # methods chapter is implemented here as a multiplicative zero rather
    # than a separate exclusion step.
    q_pick = q_snr * q_prom * q_sharp
    #q_pick = np.where(near_edge, 0.0, q_pick)
    q_pick = np.clip(q_pick, pick_floor, 1.0)

   


    # ------------------------------------------------------------------
    # Factor 2: smooth-curve residual
    # ------------------------------------------------------------------
    observed_ms   = 1000.0 * pd.to_numeric(out["observed_dt_ref_s"], errors="coerce").to_numpy(dtype=float)
    smooth_ref_ms = pd.to_numeric(out["location_median_smooth_ms"],  errors="coerce").to_numpy(dtype=float)
    residual_ms   = observed_ms - smooth_ref_ms
    q_smooth = np.exp(-0.5 * (residual_ms / sigma_smooth) ** 2)
    q_smooth = np.where(np.isfinite(residual_ms), q_smooth, 1.0)

    # ------------------------------------------------------------------
    # Factor 3: anchor (sweep) disagreement
    # ------------------------------------------------------------------
    disagreement_ms = pd.to_numeric(out["anchor_disagreement_ms"], errors="coerce").to_numpy(dtype=float)
    q_anchor = np.exp(-0.5 * (disagreement_ms / sigma_anchor) ** 2)
    q_anchor = np.where(np.isfinite(disagreement_ms), q_anchor, 1.0)

    # ------------------------------------------------------------------
    # Composite weight
    # ------------------------------------------------------------------
    w = np.clip(q_pick * q_smooth * q_anchor, 0.0, 1.0)

    out["q_snr"]    = q_snr
    out["q_prom"]   = q_prom
    out["q_sharp"]  = q_sharp
    out["q_pick"]   = q_pick
    out["q_smooth"] = q_smooth
    out["q_anchor"] = q_anchor
    out["weight"]   = w

    out["use_observation"] = (
        out["observed_dt_ref_s"].notna()
        & out["tx_lat"].notna()
        & out["tx_lon"].notna()
        & out["prior_lat"].notna()
        & out["prior_lon"].notna()
        & out["tx_x_m"].notna()
        & out["tx_y_m"].notna()
        & out["prior_x_m"].notna()
        & out["prior_y_m"].notna()
        & (out["weight"] > min_weight)
    )

    # for (loc, anchor), sub in out.groupby(["location", "anchor_label"]):
    #     sub = sub.sort_values("channel")

    #     x = sub["channel"].to_numpy()

    #     snr_raw_sub  = pd.to_numeric(sub["snr_like"], errors="coerce").fillna(0.0).to_numpy()
    #     prom_raw_sub = pd.to_numeric(sub["peak_ratio_best_to_second"], errors="coerce").fillna(0.0).to_numpy()
    #     width_ms_sub = pd.to_numeric(sub["peak_width_ms"], errors="coerce").fillna(40.0).to_numpy()

    #     q_snr_sub   = score_high_good(snr_raw_sub, bad=3.0, good=10.0)
    #     q_prom_sub  = score_high_good(np.minimum(prom_raw_sub, 8.0), bad=1.05, good=1.5)
    #     q_sharp_sub = score_low_good(width_ms_sub, good=4.0, bad=12.0)

    #     fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)

    #     axes[0, 0].scatter(x, snr_raw_sub, s=2, alpha=0.5)
    #     axes[0, 1].scatter(x, q_snr_sub, s=2, alpha=0.5)

    #     axes[1, 0].scatter(x, prom_raw_sub, s=2, alpha=0.5)
    #     axes[1, 1].scatter(x, q_prom_sub, s=2, alpha=0.5)

    #     axes[2, 0].scatter(x, width_ms_sub, s=2, alpha=0.5)
    #     axes[2, 1].scatter(x, q_sharp_sub, s=2, alpha=0.5)

    #     axes[0, 0].set_title("Raw SNR")
    #     axes[0, 1].set_title("q_snr")
    #     axes[1, 0].set_title("Raw dominance")
    #     axes[1, 1].set_title("q_dom")
    #     axes[2, 0].set_title("Raw width")
    #     axes[2, 1].set_title("q_sharp")

    #     for ax in axes[:, 1]:
    #         ax.set_ylim(-0.05, 1.05)

    #     axes[2, 0].set_xlabel("Channel")
    #     axes[2, 1].set_xlabel("Channel")

    #     fig.suptitle(f"{loc} — {anchor}")
    #     plt.tight_layout()
    #     plt.show()

    return out




# ---------------------------------------------------------------------------
# Relative arrival stacked plot
# ---------------------------------------------------------------------------

def make_relative_arrival_by_location_plot(
    merged: pd.DataFrame,
    outdir: Path,
    anchor_label: str = "lfm35_45_rep1",
) -> None:
    locations = sorted(merged["location"].astype(str).unique())
    sub = merged[merged["anchor_label"].astype(str) == anchor_label].copy()

    sweep_label = format_sweep(anchor_label)
    loc_labels  = [format_location(loc) for loc in locations]

    fig, axes = plt.subplots(len(locations), 1, figsize=(16, 2.8 * len(locations)), sharex=True)
    if len(locations) == 1:
        axes = [axes]

    sc = None
    for ax, loc, loc_label in zip(axes, locations, loc_labels):
        loc_df = sub[sub["location"] == loc].sort_values("channel")
        if loc_df.empty:
            ax.set_title(loc_label)
            continue

        ch = pd.to_numeric(loc_df["channel"],         errors="coerce").to_numpy()
        dt = 1000.0 * pd.to_numeric(loc_df["observed_dt_ref_s"], errors="coerce").to_numpy()
        w  = pd.to_numeric(loc_df["weight"],           errors="coerce").fillna(0.0).to_numpy()

        sc = ax.scatter(ch, dt, c=w, s=2, alpha=1, cmap="viridis",
                        vmin=0.0, vmax=1.0, rasterized=True)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_ylim(-50, 300)
        ax.set_ylabel("Relative arrival (ms)")
        ax.set_title(loc_label)
        ax.grid(True, alpha=0.2)


    axes[-1].set_xlabel("Channel")
    fig.suptitle(
        f"Relative arrival times coloured by weight — {sweep_label}",
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 0.90, 0.97])

    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Weight", fontsize=12)
    cbar.ax.tick_params(labelsize=12)


    # axes[-1].set_xlabel("Channel")
    # fig.suptitle(f"Relative arrival times coloured by weight — {sweep_label}")

    # fig.tight_layout(rect=[0, 0, 0.96, 1.0])  
    fname = f"relative_arrival_by_location_{anchor_label}.png"
    fig.savefig(outdir / "weight_diagnostics" / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)




def make_relative_arrival_by_location_weight_plot(
    merged: pd.DataFrame,
    outdir: Path,
    anchor_label: str = "lfm35_45_rep1",
    min_weight: float = 0.0,          # <-- new parameter, default keeps old behaviour
) -> None:
    locations = sorted(merged["location"].astype(str).unique())
    sub = merged[merged["anchor_label"].astype(str) == anchor_label].copy()

    # Filter by weight threshold
    if min_weight > 0.0:
        sub = sub[pd.to_numeric(sub["weight"], errors="coerce").fillna(0.0) >= min_weight]

    sweep_label = format_sweep(anchor_label)
    loc_labels  = [format_location(loc) for loc in locations]

    fig, axes = plt.subplots(len(locations), 1, figsize=(16, 2.8 * len(locations)), sharex=True)
    if len(locations) == 1:
        axes = [axes]

    sc = None
    for ax, loc, loc_label in zip(axes, locations, loc_labels):
        loc_df = sub[sub["location"] == loc].sort_values("channel")
        if loc_df.empty:
            ax.set_title(loc_label)
            continue

        ch = pd.to_numeric(loc_df["channel"],          errors="coerce").to_numpy()
        dt = 1000.0 * pd.to_numeric(loc_df["observed_dt_ref_s"], errors="coerce").to_numpy()
        w  = pd.to_numeric(loc_df["weight"],            errors="coerce").fillna(0.0).to_numpy()

        sc = ax.scatter(ch, dt, c=w, s=2, alpha=1, cmap="viridis",
                        vmin=0, vmax=1.0, rasterized=True)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_ylim(-50, 300)
        ax.set_ylabel("Relative arrival (ms)")
        ax.set_title(loc_label)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Channel")

    weight_str = f"weight $\\geq$ {min_weight:.1f}" if min_weight > 0 else "all weights"
    fig.suptitle(
        f"Relative arrival times coloured by weight — {sweep_label} — {weight_str}",
        
    )

    fig.tight_layout(rect=[0, 0, 0.96, 1.0])
    suffix = f"_w{str(min_weight).replace('.', 'p')}" if min_weight > 0 else ""
    fname  = f"relative_arrival_by_location_weight_{anchor_label}{suffix}.png"
    fig.savefig(outdir / "weight_diagnostics" / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def make_weight_diagnostic_plots(df: pd.DataFrame, outdir: Path) -> None:
    plot_dir = ensure_dir(outdir / "weight_diagnostics")

    locations = sorted(df["location"].astype(str).unique())
    anchors   = sorted(df["anchor_label"].astype(str).unique())

    factor_cols   = ["q_pick", "q_smooth", "q_anchor"]
    factor_labels = [
        "Pick quality  $q_{j,k}$",
        "Smooth residual  $q_{j,k}^{\\mathrm{smooth}}$",
        "Sweep dis.  $q_k^{\\mathrm{anchor}}$",
    ]
    factor_colors = [C_GREEN, C_BLUE, C_RED]

    for loc in locations:
        loc_label = format_location(loc)
        for anchor in anchors:
            sweep_label = format_sweep(anchor)
            sub = df[(df["location"] == loc) & (df["anchor_label"] == anchor)].sort_values("channel")
            if sub.empty:
                continue

            fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

            
            axes[3].scatter(sub["channel"], sub["weight"],
                             c=sub["weight"], s=2, alpha=1, cmap="viridis", vmin=0, vmax=1, rasterized=True,)
            axes[3].set_ylim(-0.05, 1.05)
            axes[3].set_ylabel("Final weight $w_{j,k}$")
            axes[3].axhline(0, color="gray", linewidth=0.5)
            axes[3].grid(True, alpha=0.3)

            axes[0].set_title(f"{loc_label}  —  {sweep_label}")

            for ax, col, label, color in zip(axes[0:3], factor_cols, factor_labels, factor_colors):
                ax.scatter(sub["channel"], sub[col], s=2, alpha=1, color=color, rasterized=True)
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel(label)
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.grid(True, alpha=0.3)




            sc = axes[4].scatter(
                sub["channel"], 
                1000.0 * pd.to_numeric(sub["observed_dt_ref_s"], errors="coerce"),
                c=sub["weight"], s=2, alpha=1, cmap="viridis", vmin=0, vmax=1, rasterized=True,
            )
            axes[4].set_ylabel("Relative arrival (ms)")
            axes[4].set_xlabel("Channel")
            axes[4].grid(True, alpha=0.3)

            fig.tight_layout()
            # Use clean readable name for the file as well
            fname = f"weights_{loc_label.replace(' ', '_')}_{sweep_label.replace(' ', '_')}.png"
            fig.savefig(plot_dir / fname, dpi=300, bbox_inches="tight")
            plt.close(fig)

    # Combined summary
    fig, axes = plt.subplots(len(locations), 1, figsize=(16, 3 * len(locations)), sharex=True)
    if len(locations) == 1:
        axes = [axes]

    for ax, loc in zip(axes, locations):
        loc_label = format_location(loc)
        loc_df = df[df["location"] == loc].sort_values("channel")
        for anchor in anchors:
            sweep_label = format_sweep(anchor)
            sub = loc_df[loc_df["anchor_label"] == anchor]
            if sub.empty:
                continue
            ax.scatter(sub["channel"], sub["weight"], s=2, alpha=1,
                       rasterized=True, label=sweep_label)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Weight")
        ax.set_title(loc_label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Channel")
    fig.suptitle("Final observation weights by location and sweep")
    fig.tight_layout()
    fig.savefig(plot_dir / "weights_all_locations_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Per-factor summary grid
    fig, big_axes = plt.subplots(
        len(locations), 3,
        figsize=(18, 3 * len(locations)),
        sharex=True, sharey=True,
    )
    if len(locations) == 1:
        big_axes = [big_axes]

    for row_axes, loc in zip(big_axes, locations):
        loc_label = format_location(loc)
        loc_df = df[df["location"] == loc].sort_values("channel")
        for ax, col, label, color in zip(row_axes, factor_cols, factor_labels, factor_colors):
            ax.scatter(loc_df["channel"], loc_df[col], s=2, alpha=0.3, color=color, rasterized=True)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{loc_label}\n{label}")
            ax.grid(True, alpha=0.3)

    fig.supxlabel("Channel")
    fig.supylabel("Factor value")
    fig.suptitle("Individual weight factors by location")
    fig.tight_layout()
    fig.savefig(plot_dir / "weight_factors_all_locations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    make_relative_arrival_by_location_plot(df, outdir, anchor_label="lfm35_45_rep1")
    make_relative_arrival_by_location_plot(df, outdir, anchor_label="lfm35_45_rep2")

    print(f"  Weight diagnostic plots saved to: {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build inversion_observations.csv from all upstream products."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg  = load_toml(args.config)
    ocfg = cfg["inversion_dataset"]
    outdir = ensure_dir(path_from_cfg(cfg, "inversion_dataset_output_dir"))

    arrivals, tx, smooth_df, disagreement, prior = load_csvs(cfg)

    arrivals_rel = compute_relative_arrival(
        arrivals, int(ocfg["channel_min"]), int(ocfg["channel_max"]),
    )
    tx_long = prepare_tx_table(tx)
    merged  = merge_all(arrivals_rel, tx_long, smooth_df, disagreement, prior)
    merged  = add_local_enu_coordinates(
        merged,
        lat0_deg=float(ocfg["enu_lat0_deg"]),
        lon0_deg=float(ocfg["enu_lon0_deg"]),
        h0_m=float(ocfg["enu_h0_m"]),
    )
    merged = make_weight(merged, ocfg)

    merged["enu_origin_lat_deg"] = float(ocfg["enu_lat0_deg"])
    merged["enu_origin_lon_deg"] = float(ocfg["enu_lon0_deg"])
    merged["enu_origin_h_m"]     = float(ocfg["enu_h0_m"])

    out_csv = outdir / "inversion_observations.csv"
    merged.to_csv(out_csv, index=False)

    make_weight_diagnostic_plots(merged, outdir)

    make_relative_arrival_by_location_weight_plot(merged, outdir, anchor_label="lfm35_45_rep1", min_weight=0.5)
    make_relative_arrival_by_location_weight_plot(merged, outdir, anchor_label="lfm35_45_rep2", min_weight=0.5)

    print(f"Saved: {out_csv}")
    print(f"Rows total:  {len(merged)}")
    print(f"Rows usable: {int(merged['use_observation'].sum())}")

    usable = merged[merged["use_observation"]]
    print("\nUsable rows by location and sweep:")
    usable = usable.copy()
    usable["location_label"] = usable["location"].map(format_location)
    usable["sweep_label"]    = usable["anchor_label"].map(format_sweep)
    print(
        usable.groupby(["location_label", "sweep_label"])
        .agg(n=("channel", "size"), mean_w=("weight", "mean"))
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
