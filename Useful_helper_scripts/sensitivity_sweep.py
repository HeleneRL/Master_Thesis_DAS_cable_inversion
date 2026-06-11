"""

OBS this needs to be placed in the src folder if it is to be ran!!


sensitivity_sweep.py
====================
Runs a hyperparameter sensitivity analysis on the DAS cable inversion by
varying one parameter at a time while keeping everything else at the values
in the config file.

Two sweeps are performed:
  1. prior_sigma_xy  – how far the cable is allowed to move from the prior
  2. curvature_sigma_xy – how sharply the cable is allowed to bend

All data loading, observation building, and control-point selection are done
exactly as in invert_cable_diagnostics.py (same config, same CSV), so the
only thing that changes between runs is the single swept parameter.

Usage
-----
    python sensitivity_sweep.py --config path/to/pipeline_config.toml [options]

    --config        Path to the pipeline config (required)
    --outdir        Where to write results (default: <inversion_output_dir>/sensitivity)
    --fmt           Figure format: pdf (default) or png
    --max_nfev      Override max function evaluations per run (default: from config)
    --skip_plots    Only save the summary CSV, skip figure generation

Each inversion run saves:
  sensitivity/<sweep_name>/run_<tag>/inversion_summary.csv   (scalar metrics)
  sensitivity/<sweep_name>/run_<tag>/updated_cable_layout.csv
  sensitivity/<sweep_name>/summary.csv                        (all runs combined)

Final figures:
  sensitivity/sensitivity_prior_sigma_xy.pdf
  sensitivity/sensitivity_curvature_sigma_xy.pdf
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse everything from the inversion script directly
# (assumes sensitivity_sweep.py lives in the same directory, or the project
#  root is on PYTHONPATH)
from invert_cable_diagnostics import (
    build_observation_table,
    build_prior_geometry,
    choose_control_channels,
    choose_control_channels_by_quality,
    compute_fit_diagnostics,
    linear_fill_to_full_channels,
    save_outputs,
    solve_inversion,
    summarize_channel_control_quality,
)
from common import load_toml, ensure_dir, path_from_cfg

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":       "STIXGeneral",
    "mathtext.fontset":  "stix",
    "font.size":         13,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.4,
    "lines.linewidth":   2.0,
})

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

SWEEPS = {
    "prior_sigma_xy": {
        "label":      r"$\sigma_{xy}^{\mathrm{base}}$  [m]",
        "values":     [5, 15, 40, 80, 150, 300, 600],
        "baseline":   80,
        "config_key": "prior_sigma_xy",           # key inside [inversion] block
        "x_scale":    "log",
        "annotate_x": True,                        # show bend-radius on top axis?
        "bend_radius": False,
        "description": (
            "How far the cable is allowed to move from the prior geometry. "
            "Baseline = 80 m (the median cable correction is ~73.7 m). "
            "Values below ~15 m prevent the cable from reaching the correct position; "
            "values above ~300 m make the prior penalty negligible."
        ),
    },
    "curvature_sigma_xy": {
        "label":      r"$\sigma_{\mathrm{curv},xy}$  [m$^{-1}$]",
        "values":     [0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 2.0],
        "baseline":   0.1,
        "config_key": "curvature_sigma_xy",
        "x_scale":    "log",
        "annotate_x": False,
        "bend_radius": True,          # show 1/sigma on secondary axis
        "description": (
            "Maximum plausible cable curvature (1/bend-radius). "
            "Baseline = 0.1 m^-1 (10 m bend radius). "
            "The Singapore cable diameter ~7.7 mm gives a minimum safe bend radius "
            "of ~77 mm, so even sigma=2.0 m^-1 (0.5 m radius) is physically plausible."
        ),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_shared_data(cfg: dict) -> tuple:
    """Load CSV, build obs table and prior geometry — shared across all runs."""
    icfg = cfg["inversion"]
    df = pd.read_csv(
        path_from_cfg(cfg, "inversion_dataset_output_dir") / "inversion_observations.csv"
    )

    origin_lat = float(df["enu_origin_lat_deg"].dropna().iloc[0])
    origin_lon = float(df["enu_origin_lon_deg"].dropna().iloc[0])
    origin_h   = float(df["enu_origin_h_m"].dropna().iloc[0])

    obs = build_observation_table(df, int(icfg["channel_offset"]))
    prior_geom_sparse = build_prior_geometry(df, int(icfg["channel_offset"]))
    prior_full = linear_fill_to_full_channels(prior_geom_sparse)

    min_ch = prior_full["channel"].min()
    max_ch = prior_full["channel"].max()
    obs = obs[(obs["channel_eff"] >= min_ch) & (obs["channel_eff"] <= max_ch)].copy()
    obs = obs[(obs["reference_channel_eff"] >= min_ch) & (obs["reference_channel_eff"] <= max_ch)].copy()

    channel_quality_df = summarize_channel_control_quality(obs)

    # Control channels — chosen once from the config, identical for every run
    if str(icfg["control_selection_mode"]) == "spacing":
        control_channels = choose_control_channels(
            prior_full["channel"].values,
            obs["reference_channel_eff"].unique(),
            int(icfg["control_spacing"]),
        )
    else:
        control_channels = choose_control_channels_by_quality(
            full_channels=prior_full["channel"].values,
            channel_quality_df=channel_quality_df,
            quality_threshold=float(icfg["control_quality_threshold"]),
            min_separation=int(icfg["control_min_separation"]),
            max_gap=int(icfg["control_max_gap"]),
            reference_channels=obs["reference_channel_eff"].unique(),
        )

    return obs, prior_full, channel_quality_df, control_channels, origin_lat, origin_lon, origin_h


def build_solve_kwargs(icfg: dict, max_nfev_override: int | None = None) -> dict:
    """Extract all solve_inversion kwargs from the [inversion] config block."""
    return dict(
        sound_speed       = float(icfg["sound_speed"]),
        channel_spacing   = float(icfg["channel_spacing"]),
        abs_scale         = float(icfg.get("abs_scale", 0.003)),
        rel_scale         = float(icfg["rel_scale"]),
        prior_sigma_xy    = float(icfg["prior_sigma_xy"]),
        prior_sigma_z     = float(icfg["prior_sigma_z"]),
        curvature_sigma_xy= float(icfg["curvature_sigma_xy"]),
        curvature_sigma_z = float(icfg["curvature_sigma_z"]),
        spacing_sigma     = float(icfg["spacing_sigma"]),
        anchor_bias_sigma = float(icfg.get("anchor_bias_sigma", 0.02)),
        huber_delta_abs   = float(icfg.get("huber_delta_abs", 3.0)),
        huber_delta_rel   = float(icfg["huber_delta_rel"]),
        max_nfev          = max_nfev_override or int(icfg["max_nfev"]),
        relative_only     = bool(icfg.get("relative_only", False)),
    )


def rmse_ms(residuals_s: np.ndarray) -> float:
    r = residuals_s[np.isfinite(residuals_s)]
    return 1000.0 * float(np.sqrt(np.mean(r ** 2)))


def weighted_rmse_ms(residuals_s: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(residuals_s) & np.isfinite(weights) & (weights > 0)
    r = residuals_s[mask]
    w = weights[mask]
    return 1000.0 * float(np.sqrt(np.average(r ** 2, weights=w)))


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    sweep_name: str,
    param_value: float,
    obs: pd.DataFrame,
    prior_full: pd.DataFrame,
    channel_quality_df: pd.DataFrame,
    control_channels: np.ndarray,
    solve_kwargs: dict,
    run_dir: Path,
    origin_lat: float,
    origin_lon: float,
    origin_h: float,
    tag: str,
) -> dict:
    """Run one inversion with a single overridden parameter. Returns scalar metrics."""

    kw = copy.deepcopy(solve_kwargs)
    kw[SWEEPS[sweep_name]["config_key"]] = float(param_value)

    print(f"\n{'─'*60}")
    print(f"  {sweep_name} = {param_value}   (tag: {tag})")
    print(f"{'─'*60}")

    t0 = time.perf_counter()
    try:
        solution = solve_inversion(
            obs=obs,
            prior_full=prior_full,
            channel_quality_df=channel_quality_df,
            control_channels=control_channels,
            **kw,
        )
        elapsed = time.perf_counter() - t0
        diagnostics = compute_fit_diagnostics(solution)

        # Save full outputs into the run subdirectory
        save_outputs(
            obs, solution, diagnostics, str(run_dir),
            origin_lat, origin_lon, origin_h, channel_quality_df,
        )

        # --- Scalar metrics ---
        fit_prior = diagnostics["pred_rel_prior"]
        fit_opt   = diagnostics["pred_rel_opt"]
        obs_dt    = obs["observed_dt_ref_s"].values.astype(float)
        weights   = obs["weight"].values.astype(float)

        res_prior = obs_dt - fit_prior
        res_opt   = obs_dt - fit_opt

        cable = pd.read_csv(run_dir / "updated_cable_layout.csv")
        hshift = cable["horizontal_shift_m"]

        converged = bool(solution["result"].success) or solution["result"].status in (1, 2, 3, 4)

        metrics = {
            "sweep":                  sweep_name,
            "param_value":            float(param_value),
            "tag":                    tag,
            "converged":              converged,
            "status":                 int(solution["result"].status),
            "nfev":                   int(solution["result"].nfev),
            "final_cost":             float(solution["result"].cost),
            "elapsed_s":              round(elapsed, 1),
            "n_control_points":       int(len(control_channels)),
            "rmse_prior_ms":          rmse_ms(res_prior),
            "rmse_opt_ms":            rmse_ms(res_opt),
            "weighted_rmse_opt_ms":   weighted_rmse_ms(res_opt, weights),
            "median_hshift_m":        float(hshift.median()),
            "p75_hshift_m":           float(hshift.quantile(0.75)),
            "p95_hshift_m":           float(hshift.quantile(0.95)),
        }

        # If a NUS reference file exists alongside the cable layout, compute
        # mean distance to it.  Optional — skipped silently if not present.
        nus_path = run_dir.parent.parent / "nus_reference.csv"
        if nus_path.exists():
            try:
                nus = pd.read_csv(nus_path)
                merged = cable.merge(nus, on="channel", suffixes=("", "_nus"))
                dist = np.sqrt(
                    (merged["x_m"] - merged["x_m_nus"]) ** 2
                    + (merged["y_m"] - merged["y_m_nus"]) ** 2
                )
                metrics["mean_nus_dist_m"] = float(dist.mean())
                metrics["median_nus_dist_m"] = float(dist.median())
            except Exception as exc:
                warnings.warn(f"Could not compute NUS distance: {exc}")

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        warnings.warn(f"Run failed: {exc}")
        metrics = {
            "sweep": sweep_name, "param_value": float(param_value), "tag": tag,
            "converged": False, "status": -99, "nfev": 0,
            "final_cost": np.nan, "elapsed_s": round(elapsed, 1),
            "n_control_points": int(len(control_channels)),
            "rmse_prior_ms": np.nan, "rmse_opt_ms": np.nan,
            "weighted_rmse_opt_ms": np.nan,
            "median_hshift_m": np.nan, "p75_hshift_m": np.nan,
            "p95_hshift_m": np.nan,
        }

    # Save per-run metrics immediately so a crash mid-sweep doesn't lose data
    pd.DataFrame([metrics]).to_csv(run_dir / "run_metrics.csv", index=False)
    print(f"  RMSE prior={metrics['rmse_prior_ms']:.2f} ms  "
          f"opt={metrics['rmse_opt_ms']:.2f} ms  "
          f"hshift_med={metrics['median_hshift_m']:.1f} m  "
          f"nfev={metrics['nfev']}  t={metrics['elapsed_s']}s")
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_sensitivity_plot(
    sweep_df: pd.DataFrame,
    sweep_name: str,
    baseline_value: float,
    baseline_rmse: float,
    outpath: Path,
    fmt: str,
    has_nus: bool,
) -> None:
    sweep_info = SWEEPS[sweep_name]
    x = sweep_df["param_value"].values.astype(float)
    rmse = sweep_df["rmse_opt_ms"].values.astype(float)
    hshift = sweep_df["median_hshift_m"].values.astype(float)

    n_panels = 3 if has_nus else 2
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(9, 3.5 * n_panels),
                             sharex=True)
    fig.suptitle(
        f"Sensitivity to {sweep_info['label']}\n"
        f"(all other parameters at baseline values)",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: RMSE ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, rmse, "o-", color="#1f77b4", zorder=5, label="Optimised RMSE")
    ax.axvline(baseline_value, color="black", lw=1.2, ls="--",
               label=f"Baseline ({baseline_value})")

    # Stable band: ±0.5 ms around the baseline-run RMSE
    if np.isfinite(baseline_rmse):
        ax.axhspan(baseline_rmse - 0.5, baseline_rmse + 0.5,
                   color="#1f77b4", alpha=0.10, label="±0.5 ms stable band")
        ax.axhline(baseline_rmse, color="#1f77b4", lw=0.8, ls=":",
                   alpha=0.6)

    # Annotate failed runs
    for _, row in sweep_df.iterrows():
        if not row["converged"]:
            ax.annotate("✗ failed", xy=(row["param_value"], row["rmse_opt_ms"]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color="red")

    ax.set_ylabel("RMSE  [ms]")
    ax.set_xscale(sweep_info["x_scale"])
    ax.legend(fontsize=10)
    ax.grid()

    # ── Panel 2: median horizontal shift ───────────────────────────────────
    ax = axes[1]
    ax.plot(x, hshift, "s-", color="#d62728", zorder=5)
    ax.axvline(baseline_value, color="black", lw=1.2, ls="--")
    ax.set_ylabel("Median horizontal\nshift from prior  [m]")
    ax.set_xscale(sweep_info["x_scale"])
    ax.grid()

    # For sigma_xy sweep: draw a reference line at the known median correction
    # (73.7 m from thesis) so we can see whether the cable reaches it
    if sweep_name == "prior_sigma_xy":
        ax.axhline(73.7, color="#d62728", lw=0.9, ls=":",
                   label="Expected correction ~73.7 m")
        ax.legend(fontsize=9)

    # ── Panel 3 (optional): NUS distance ───────────────────────────────────
    if has_nus:
        ax = axes[2]
        nus_dist = sweep_df["mean_nus_dist_m"].values.astype(float)
        ax.plot(x, nus_dist, "^-", color="#2ca02c", zorder=5)
        ax.axvline(baseline_value, color="black", lw=1.2, ls="--")
        ax.set_ylabel("Mean distance from\nNUS reference  [m]")
        ax.set_xscale(sweep_info["x_scale"])
        ax.grid()

    # ── Secondary x-axis: bend radius (curvature sweep only) ───────────────
    if sweep_info["bend_radius"]:
        ax_top = axes[0].twiny()
        ax_top.set_xscale("log")
        ax_top.set_xlim(axes[0].get_xlim())
        tick_vals = x[np.isfinite(x) & (x > 0)]
        ax_top.set_xticks(tick_vals)
        ax_top.set_xticklabels([f"{1/v:.1f}" for v in tick_vals], fontsize=9)
        ax_top.set_xlabel("Corresponding min bend radius  [m]", fontsize=10)

    # ── Shared x-label (on bottom panel) ───────────────────────────────────
    axes[-1].set_xlabel(sweep_info["label"])

    # ── Nfev annotation as text under each point ───────────────────────────
    for _, row in sweep_df.iterrows():
        axes[0].text(row["param_value"], row["rmse_opt_ms"] - 0.15,
                     f"{int(row['nfev'])}", ha="center", va="top",
                     fontsize=7.5, color="grey")

    fig.tight_layout()
    out = outpath.with_suffix(f".{fmt}")
    fig.savefig(out, format=fmt, bbox_inches="tight",
                dpi=300 if fmt == "png" else None)
    print(f"  [saved] {out}")
    plt.close(fig)


def make_combined_plot(
    all_results: pd.DataFrame,
    outdir: Path,
    fmt: str,
) -> None:
    """One-page overview: both sweeps side by side."""
    sweeps = [s for s in SWEEPS if s in all_results["sweep"].values]
    if len(sweeps) == 0:
        return

    fig, axes = plt.subplots(len(sweeps), 2,
                             figsize=(14, 4.5 * len(sweeps)))
    if len(sweeps) == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Hyperparameter Sensitivity – Summary", fontsize=15, fontweight="bold")

    for row_i, sweep_name in enumerate(sweeps):
        df = all_results[all_results["sweep"] == sweep_name].copy()
        info = SWEEPS[sweep_name]
        x = df["param_value"].values.astype(float)
        baseline = info["baseline"]

        # RMSE panel
        ax = axes[row_i, 0]
        ax.plot(x, df["rmse_opt_ms"].values, "o-", color="#1f77b4")
        ax.axvline(baseline, color="black", lw=1.2, ls="--", label="Baseline")
        ax.set_xscale(info["x_scale"])
        ax.set_xlabel(info["label"])
        ax.set_ylabel("RMSE  [ms]")
        ax.set_title(f"{sweep_name} → RMSE")
        ax.legend(fontsize=9)
        ax.grid()

        # Horizontal shift panel
        ax = axes[row_i, 1]
        ax.plot(x, df["median_hshift_m"].values, "s-", color="#d62728")
        ax.axvline(baseline, color="black", lw=1.2, ls="--")
        ax.set_xscale(info["x_scale"])
        ax.set_xlabel(info["label"])
        ax.set_ylabel("Median hshift  [m]")
        ax.set_title(f"{sweep_name} → horizontal shift")
        ax.grid()

    fig.tight_layout()
    out = outdir / f"sensitivity_combined.{fmt}"
    fig.savefig(out, format=fmt, bbox_inches="tight",
                dpi=300 if fmt == "png" else None)
    print(f"  [saved] {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity sweep for DAS cable inversion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",    type=Path, required=True,
                        help="Path to pipeline_config.toml")
    parser.add_argument("--outdir",    type=Path, default=None,
                        help="Output directory (default: <inversion_output_dir>/sensitivity)")
    parser.add_argument("--fmt",       choices=["pdf", "png"], default="pdf",
                        help="Figure format (default: pdf)")
    parser.add_argument("--max_nfev",  type=int, default=None,
                        help="Override max function evaluations per run")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip figure generation, save CSVs only")
    parser.add_argument("--sweeps",    nargs="+",
                        choices=list(SWEEPS.keys()),
                        default=list(SWEEPS.keys()),
                        help="Which sweeps to run (default: all)")
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"ERROR: config not found: {args.config}")

    cfg  = load_toml(args.config)
    icfg = cfg["inversion"]

    # Output root
    if args.outdir is None:
        base_out = path_from_cfg(cfg, "inversion_output_dir")
        outdir = Path(base_out) / "sensitivity"
    else:
        outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  DAS Cable Inversion – Hyperparameter Sensitivity Sweep")
    print(f"{'='*65}")
    print(f"  Config  : {args.config}")
    print(f"  Output  : {outdir}")
    print(f"  Format  : {args.fmt.upper()}")
    print(f"  Sweeps  : {args.sweeps}")
    if args.max_nfev:
        print(f"  max_nfev override: {args.max_nfev}")
    print()

    # ── Load shared data once ───────────────────────────────────────────────
    print("Loading data and building prior geometry…")
    (obs, prior_full, channel_quality_df,
     control_channels, origin_lat, origin_lon, origin_h) = load_shared_data(cfg)

    baseline_kwargs = build_solve_kwargs(icfg, args.max_nfev)
    print(f"  Observations : {len(obs):,}")
    print(f"  Control pts  : {len(control_channels)}")
    print(f"  Baseline hyperparameters from config:")
    for k in ("prior_sigma_xy", "prior_sigma_z",
               "curvature_sigma_xy", "curvature_sigma_z",
               "rel_scale", "spacing_sigma"):
        print(f"    {k:25s} = {baseline_kwargs[k]}")

    all_metrics: list[dict] = []

    # ── Run sweeps ──────────────────────────────────────────────────────────
    for sweep_name in args.sweeps:
        sweep_info = SWEEPS[sweep_name]
        sweep_dir  = outdir / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'━'*65}")
        print(f"  SWEEP: {sweep_name}")
        print(f"  {sweep_info['description']}")
        print(f"  Values: {sweep_info['values']}")
        print(f"  Baseline: {sweep_info['baseline']}")
        print(f"{'━'*65}")

        sweep_metrics: list[dict] = []

        for val in sweep_info["values"]:
            # Human-readable tag for directory name
            tag = str(val).replace(".", "p")
            run_dir = sweep_dir / f"run_{tag}"
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics = run_single(
                sweep_name=sweep_name,
                param_value=val,
                obs=obs,
                prior_full=prior_full,
                channel_quality_df=channel_quality_df,
                control_channels=control_channels,
                solve_kwargs=baseline_kwargs,
                run_dir=run_dir,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                origin_h=origin_h,
                tag=tag,
            )
            sweep_metrics.append(metrics)
            all_metrics.append(metrics)

        # Save sweep summary immediately
        sweep_df = pd.DataFrame(sweep_metrics)
        sweep_csv = sweep_dir / "summary.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"\n  [saved] {sweep_csv}")

        # Print sweep summary table
        print(f"\n  {'─'*55}")
        print(f"  Sweep summary: {sweep_name}")
        print(f"  {'─'*55}")
        cols = ["param_value", "rmse_opt_ms", "median_hshift_m", "nfev", "converged"]
        if "mean_nus_dist_m" in sweep_df.columns:
            cols.insert(2, "mean_nus_dist_m")
        print(sweep_df[cols].to_string(index=False, float_format="{:.2f}".format))

        # Identify stable range
        baseline_row = sweep_df[
            np.isclose(sweep_df["param_value"], sweep_info["baseline"], rtol=0.01)
        ]
        if len(baseline_row) > 0:
            baseline_rmse = float(baseline_row["rmse_opt_ms"].iloc[0])
            stable = sweep_df[
                (sweep_df["rmse_opt_ms"] - baseline_rmse).abs() < 0.5
            ]["param_value"].values
            if len(stable) >= 2:
                print(f"\n  Stable range (RMSE within ±0.5 ms of baseline {baseline_rmse:.2f} ms): "
                      f"[{stable.min()}, {stable.max()}]")
        else:
            baseline_rmse = np.nan

        # Figures
        if not args.skip_plots:
            has_nus = "mean_nus_dist_m" in sweep_df.columns and sweep_df["mean_nus_dist_m"].notna().any()
            make_sensitivity_plot(
                sweep_df=sweep_df,
                sweep_name=sweep_name,
                baseline_value=sweep_info["baseline"],
                baseline_rmse=baseline_rmse,
                outpath=outdir / f"sensitivity_{sweep_name}",
                fmt=args.fmt,
                has_nus=has_nus,
            )

    # ── Combined summary ────────────────────────────────────────────────────
    all_df = pd.DataFrame(all_metrics)
    all_csv = outdir / "all_sweeps_summary.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"\n  [saved] {all_csv}")

    if not args.skip_plots:
        make_combined_plot(all_df, outdir, args.fmt)

    # ── Final printed summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SENSITIVITY SWEEP COMPLETE")
    print(f"{'='*65}")
    for sweep_name in args.sweeps:
        df = all_df[all_df["sweep"] == sweep_name]
        if df.empty:
            continue
        baseline = SWEEPS[sweep_name]["baseline"]
        base = df[np.isclose(df["param_value"], baseline, rtol=0.01)]
        rmse_range = df["rmse_opt_ms"].max() - df["rmse_opt_ms"].min()
        print(f"\n  {sweep_name}")
        if len(base):
            print(f"    Baseline RMSE    : {base['rmse_opt_ms'].iloc[0]:.3f} ms")
        print(f"    RMSE range       : {df['rmse_opt_ms'].min():.3f} – "
              f"{df['rmse_opt_ms'].max():.3f} ms  (Δ = {rmse_range:.3f} ms)")
        stable = df[(df["rmse_opt_ms"] - df["rmse_opt_ms"].min()).abs() < 0.5]
        if len(stable) >= 2:
            print(f"    Stable values    : {stable['param_value'].values}")
    print(f"\n  All outputs in: {outdir}\n")


if __name__ == "__main__":
    main()
