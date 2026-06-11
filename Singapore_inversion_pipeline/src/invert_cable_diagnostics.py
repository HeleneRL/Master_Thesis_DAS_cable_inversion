from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import least_squares

from common import load_toml, ensure_dir, path_from_cfg


CHANNEL_SPACING_DEFAULT = 1.02
SOUND_SPEED_DEFAULT = 1500.0
CHANNEL_OFFSET_DEFAULT = 0


CABLE_BLUE = "#1f77b4"
PRIOR_GRAY = "0.45"


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return values[np.searchsorted(cdf, 0.5)]


def huber_scale(x, delta):
    ax = np.abs(x)
    return np.where(ax <= delta, x, delta * np.sign(x) * np.sqrt(ax / delta))


def build_prior_geometry(df, channel_offset):
    geom = (
        df.groupby("channel")[["prior_x_m", "prior_y_m", "prior_u_m"]]
        .first()
        .reset_index()
        .rename(columns={"channel": "raw_channel"})
    )
    geom["channel"] = geom["raw_channel"] + channel_offset
    geom = geom.sort_values("channel").reset_index(drop=True)
    return geom[["channel", "prior_x_m", "prior_y_m", "prior_u_m"]]


def linear_fill_to_full_channels(prior_geom):
    full_ch = np.arange(prior_geom["channel"].min(), prior_geom["channel"].max() + 1)
    out = pd.DataFrame({"channel": full_ch})
    for col in ["prior_x_m", "prior_y_m", "prior_u_m"]:
        f = interp1d(
            prior_geom["channel"].values,
            prior_geom[col].values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        out[col] = f(full_ch)
    return out


def build_observation_table(df, channel_offset):
    obs = df.copy()

    # Coerce only the bool columns that still exist in the new pipeline
    bool_cols = ["use_observation", "passed_snr_threshold", "near_window_edge", "base_valid"]
    for c in bool_cols:
        if c in obs.columns and obs[c].dtype == object:
            obs[c] = obs[c].astype(str).str.upper().map({"TRUE": True, "FALSE": False})

    obs["channel_eff"] = obs["channel"] + channel_offset
    obs["reference_channel_eff"] = obs["reference_channel"] + channel_offset
    obs["anchor_id"] = obs["location"].astype(str) + "_a" + obs["anchor_index"].astype(str)

    keep = np.ones(len(obs), dtype=bool)
    if "use_observation" in obs.columns:
        keep &= obs["use_observation"].fillna(False).values.astype(bool)
    if "weight" in obs.columns:
        keep &= np.isfinite(obs["weight"].values)
        keep &= obs["weight"].values > 0

    numeric_needed = [
        "channel_eff",
        "reference_channel_eff",
        "observed_t_s",
        "observed_dt_ref_s",
        "tx_x_m",
        "tx_y_m",
        "tx_u_m",
        "weight",
    ]
    for c in numeric_needed:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")
        keep &= np.isfinite(obs[c].values)

    obs = obs.loc[keep].copy().reset_index(drop=True)
    return obs


def summarize_channel_control_quality(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute S_k = sum of observation weights for each channel.

    This is the control quality score used to rank candidate control points.
    A channel seen by many transmissions with high-quality detections scores
    high; a channel with few or low-weight observations scores low.
    Channels with high S_k make good control points because they combine
    coverage with quality — the two things a good representative channel needs.
    """
    grp = (
        obs.groupby("channel_eff")
        .agg(
            S_k=("weight", "sum"),
            n_obs=("weight", "size"),
            mean_weight=("weight", "mean"),
            n_unique_anchors=("anchor_id", pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"channel_eff": "channel"})
    )
    # Expose S_k as control_quality_score so downstream code
    # (choose_control_channels_by_quality, plots) works unchanged.
    #grp["control_quality_score"] = grp["S_k"]
    grp["control_quality_score"] = grp["S_k"] / max(float(grp["S_k"].max()), 1e-9)
    return grp.sort_values("channel").reset_index(drop=True)


def choose_control_channels(full_channels, reference_channels, spacing):
    start = int(full_channels.min())
    end = int(full_channels.max())
    ctrl = list(range(start, end + 1, int(spacing)))
    if ctrl[-1] != end:
        ctrl.append(end)
    ctrl = set(ctrl)
    ctrl.add(start)
    ctrl.add(end)
    for rc in reference_channels:
        rc = int(rc)
        nearest = int(round((rc - start) / spacing) * spacing + start)
        nearest = min(max(nearest, start), end)
        ctrl.add(rc)
        ctrl.add(nearest)
    return np.array(sorted(ctrl), dtype=int)


def choose_control_channels_by_quality(
    full_channels,
    channel_quality_df,
    quality_threshold,
    min_separation=5,
    max_gap=40,
    reference_channels=None,
):
    """
    Greedy selection of control channels ranked by S_k (sum of weights).

    Selects the highest-scoring channel that is at least min_separation
    channels from any already-selected channel, then repeats. Channels
    with S_k below quality_threshold are skipped. After selection, a
    gap-filling pass ensures no gap exceeds max_gap channels.

    quality_threshold is expressed in the same units as S_k (sum of weights),
    not a normalised [0,1] score.
    """
    full_channels = np.asarray(full_channels, dtype=int)
    start = int(full_channels.min())
    end = int(full_channels.max())

    qdf = channel_quality_df.copy()
    qdf["channel"] = qdf["channel"].astype(int)
    qdf = qdf[qdf["channel"].between(start, end)].copy()
    # Sort descending by S_k; break ties by n_obs then n_unique_anchors
    qdf = qdf.sort_values(
        ["control_quality_score", "n_obs", "n_unique_anchors"],
        ascending=False,
    )

    selected = []
    for _, row in qdf.iterrows():
        ch = int(row["channel"])
        score = float(row["control_quality_score"])
        if not np.isfinite(score) or score < quality_threshold:
            continue
        if len(selected) == 0 or min(abs(ch - s) for s in selected) >= int(min_separation):
            selected.append(ch)

    selected = set(selected)
    selected.add(start)
    selected.add(end)

    if reference_channels is not None:
        for rc in np.asarray(reference_channels, dtype=int):
            if start <= rc <= end:
                selected.add(int(rc))

    selected = np.array(sorted(selected), dtype=int)

    # Gap-filling pass
    if max_gap is not None and max_gap > 0 and len(selected) > 1:
        filled = [int(selected[0])]
        for ch in selected[1:]:
            prev = filled[-1]
            gap = int(ch - prev)
            if gap > int(max_gap):
                extra = list(range(prev + int(max_gap), ch, int(max_gap)))
                filled.extend(extra)
            filled.append(int(ch))
        selected = np.array(sorted(set(filled)), dtype=int)

    return selected


def interpolate_curve(ctrl_channels, ctrl_xyz, eval_channels):
    x = CubicSpline(ctrl_channels, ctrl_xyz[:, 0], bc_type="natural")(eval_channels)
    y = CubicSpline(ctrl_channels, ctrl_xyz[:, 1], bc_type="natural")(eval_channels)
    z = interp1d(ctrl_channels, ctrl_xyz[:, 2], kind="linear", bounds_error=False, fill_value="extrapolate")(eval_channels)

    return np.column_stack([x, y, z])


def make_channel_lookup(channels):
    channels = np.asarray(channels, dtype=int)
    return {int(ch): i for i, ch in enumerate(channels)}


def maybe_add_latlon(geom_df, origin_lat, origin_lon, origin_h):
    try:
        from pyproj import Transformer
        to_ecef = Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
        from_ecef = Transformer.from_crs("epsg:4978", "epsg:4979", always_xy=True)
        lon0, lat0, h0 = origin_lon, origin_lat, origin_h
        x0, y0, z0 = to_ecef.transform(lon0, lat0, h0)
        lat0r = np.deg2rad(lat0)
        lon0r = np.deg2rad(lon0)
        slat, clat = np.sin(lat0r), np.cos(lat0r)
        slon, clon = np.sin(lon0r), np.cos(lon0r)
        R = np.array([[-slon, -slat * clon, clat * clon],
                      [clon, -slat * slon, clat * slon],
                      [0.0, clat, slat]])
        Rt = R.T
        enu = geom_df[["x_m", "y_m", "z_m"]].values
        ecef = enu @ Rt + np.array([x0, y0, z0])
        lon, lat, h = from_ecef.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
        geom_df["lat_deg"] = lat
        geom_df["lon_deg"] = lon
        geom_df["h_m"] = h
    except Exception as exc:
        warnings.warn(f"Could not compute lat/lon output: {exc}")
    return geom_df


def print_jacobian_diagnostics(result):
    try:
        s = np.linalg.svd(result.jac, compute_uv=False)
        smin = float(s[-1])
        smax = float(s[0])
        cond = smax / max(smin, 1e-16)
        print(f"[Jacobian] singular values: min={smin:.3e}, max={smax:.3e}, cond≈{cond:.3e}")
        print("[Jacobian] smallest 10 singular values:", s[-10:])
    except Exception as exc:
        print(f"[Jacobian] Could not compute SVD: {exc}")


def solve_inversion(
    obs,
    prior_full,
    channel_quality_df,
    control_channels,
    sound_speed=1500.0,
    channel_spacing=1.02,
    abs_scale=0.003,
    rel_scale=0.0015,
    prior_sigma_xy=60.0,
    prior_sigma_z=0.025,
    curvature_sigma_xy=8,
    curvature_sigma_z=0.025,
    spacing_sigma=0.08,
    anchor_bias_sigma=0.02,
    huber_delta_abs=3.0,
    huber_delta_rel=3.0,
    max_nfev=250,
    relative_only=False,
):
    full_channels = prior_full["channel"].values.astype(int)
    full_lookup = make_channel_lookup(full_channels)

    ctrl_lookup_full_idx = np.array([full_lookup[int(c)] for c in control_channels], dtype=int)
    prior_xyz_full = prior_full[["prior_x_m", "prior_y_m", "prior_u_m"]].values.astype(float)
    prior_xyz_ctrl = prior_xyz_full[ctrl_lookup_full_idx]

    obs_ch_idx = np.array([full_lookup[int(c)] for c in obs["channel_eff"].values], dtype=int)
    ref_ch_idx = np.array([full_lookup[int(c)] for c in obs["reference_channel_eff"].values], dtype=int)
    weights = obs["weight"].values.astype(float)
    sqrtw = np.sqrt(np.clip(weights, 1e-8, None))

    tx_xyz = obs[["tx_x_m", "tx_y_m", "tx_u_m"]].values.astype(float)
    obs_t_rel = obs["observed_dt_ref_s"].values.astype(float)

    # ------------------------------------------------------------------
    # Adaptive per-control-point prior sigma based on S_k.
    #
    # prior_sigma_xy / prior_sigma_z are the base values, interpreted as
    # the allowed displacement at a control point with *average* observation
    # density.  Control points in well-observed regions (high S_k) get a
    # proportionally larger sigma (more freedom); control points in poorly-
    # observed or gap-filled regions (low S_k) get a smaller sigma (tighter
    # anchor to the prior).
    #
    # Gap-filled control points that have no observations at all receive
    # the minimum S_k seen among observed control points, making them
    # maximally conservative.
    # ------------------------------------------------------------------
    sk_lookup = channel_quality_df.set_index("channel")["S_k"]
    sk_ctrl = np.array(
        [float(sk_lookup.get(int(c), 0.0)) for c in control_channels],
        dtype=float,
    )

    # Separate truly observed from gap-filled (S_k == 0)
    observed_mask = sk_ctrl > 1e-6
    if np.any(observed_mask):
        sk_min_observed = float(sk_ctrl[observed_mask].min())
        sk_mean = float(sk_ctrl[observed_mask].mean())
    else:
        sk_min_observed = 1.0
        sk_mean = 1.0

    # Gap-filled points get the minimum observed S_k → tightest prior
    sk_ctrl = np.where(observed_mask, sk_ctrl, sk_min_observed)

    # Scale relative to the mean so that prior_sigma_xy is the sigma at
    # an average-density control point
    sk_scale = sk_ctrl / sk_mean
    prior_sigma_xy_ctrl = prior_sigma_xy * sk_scale   # shape (n_ctrl,)
    prior_sigma_z_ctrl  = prior_sigma_z  * sk_scale   # shape (n_ctrl,)

    print(
        f"[prior sigma] S_k range at control points: "
        f"{sk_ctrl.min():.2f} – {sk_ctrl.max():.2f}  "
        f"(mean {sk_mean:.2f})\n"
        f"  => sigma_xy range: "
        f"{prior_sigma_xy_ctrl.min():.1f} – {prior_sigma_xy_ctrl.max():.1f} m"
    )

    # ------------------------------------------------------------------

    anchors = None
    anchor_idx = None
    init_biases = None
    obs_t_abs = None

    if not relative_only:
        anchors = np.array(sorted(obs["anchor_id"].unique()))
        anchor_to_idx = {a: i for i, a in enumerate(anchors)}
        anchor_idx = np.array([anchor_to_idx[a] for a in obs["anchor_id"].values], dtype=int)
        obs_t_abs = obs["observed_t_s"].values.astype(float)
        pred_abs_prior = np.linalg.norm(tx_xyz - prior_xyz_full[obs_ch_idx], axis=1) / sound_speed
        init_biases = np.zeros(len(anchors))
        for a_i in range(len(anchors)):
            m = anchor_idx == a_i
            init_biases[a_i] = weighted_median(obs_t_abs[m] - pred_abs_prior[m], weights[m])
            if not np.isfinite(init_biases[a_i]):
                init_biases[a_i] = 0.0

    n_ctrl = len(control_channels)

    if relative_only:
        x0 = np.zeros(3 * n_ctrl)
    else:
        x0 = np.concatenate([np.zeros(3 * n_ctrl), init_biases])

    history = {
        "eval": [], "cost_total": [], "cost_best_so_far": [],
        "cost_rel": [], "cost_prior": [], "cost_curv": [], "cost_spacing": [],
        "param_norm": [], "step_norm": [], "dx_norm": [], "dy_norm": [], "dz_norm": [],
    }
    if not relative_only:
        history["cost_abs"] = []
        history["cost_bias"] = []

    prev_p = {"value": None}
    best_cost = {"value": np.inf}

    def unpack(p):
        dx = p[0:n_ctrl]
        dy = p[n_ctrl:2 * n_ctrl]
        dz = p[2 * n_ctrl:3 * n_ctrl]
        ctrl_xyz = prior_xyz_ctrl + np.column_stack([dx, dy, dz])
        full_xyz = interpolate_curve(control_channels, ctrl_xyz, full_channels)
        bias = None if relative_only else p[3 * n_ctrl:]
        return dx, dy, dz, ctrl_xyz, full_xyz, bias

    def residual_vector(p):
        dx, dy, dz, ctrl_xyz, full_xyz, bias = unpack(p)
        xyz_obs = full_xyz[obs_ch_idx]
        xyz_ref = full_xyz[ref_ch_idx]

        pred_rel = (
            np.linalg.norm(tx_xyz - xyz_obs, axis=1)
            - np.linalg.norm(tx_xyz - xyz_ref, axis=1)
        ) / sound_speed
        rel_res = sqrtw * huber_scale((obs_t_rel - pred_rel) / rel_scale, huber_delta_rel)

        # Adaptive prior penalty: sigma scales with local S_k
        dxyz = ctrl_xyz - prior_xyz_ctrl
        prior_pen = np.concatenate([
            dxyz[:, 0] / prior_sigma_xy_ctrl,
            dxyz[:, 1] / prior_sigma_xy_ctrl,
            dxyz[:, 2] / prior_sigma_z_ctrl,
        ])

        # Arc-length distances between consecutive control points
        ctrl_s = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(ctrl_xyz, axis=0), axis=1)
        )])
        h1 = ctrl_s[1:-1] - ctrl_s[:-2]
        h2 = ctrl_s[2:] - ctrl_s[1:-1]

        # Proper second derivative (curvature) in m^-1, spacing-aware
        scale = 2.0 / (h1 + h2)
        d2x = scale * ((ctrl_xyz[2:, 0] - ctrl_xyz[1:-1, 0]) / h2
                     - (ctrl_xyz[1:-1, 0] - ctrl_xyz[:-2, 0]) / h1)
        d2y = scale * ((ctrl_xyz[2:, 1] - ctrl_xyz[1:-1, 1]) / h2
                     - (ctrl_xyz[1:-1, 1] - ctrl_xyz[:-2, 1]) / h1)
        d2z = scale * ((ctrl_xyz[2:, 2] - ctrl_xyz[1:-1, 2]) / h2
                     - (ctrl_xyz[1:-1, 2] - ctrl_xyz[:-2, 2]) / h1)

        curv_pen = np.concatenate([
            d2x / curvature_sigma_xy,
            d2y / curvature_sigma_xy,
            d2z / curvature_sigma_z,
        ])

        seg = np.linalg.norm(np.diff(full_xyz, axis=0), axis=1)
        spacing_pen = (seg - channel_spacing) / spacing_sigma

        blocks = [rel_res, prior_pen, curv_pen, spacing_pen]
        cost_rel     = 0.5 * np.sum(rel_res ** 2)
        cost_prior   = 0.5 * np.sum(prior_pen ** 2)
        cost_curv    = 0.5 * np.sum(curv_pen ** 2)
        cost_spacing = 0.5 * np.sum(spacing_pen ** 2)
        cost_total   = cost_rel + cost_prior + cost_curv + cost_spacing

        if not relative_only:
            pred_abs = np.linalg.norm(tx_xyz - xyz_obs, axis=1) / sound_speed + bias[anchor_idx]
            abs_res  = sqrtw * huber_scale((obs_t_abs - pred_abs) / abs_scale, huber_delta_abs)
            bias_pen = bias / anchor_bias_sigma
            blocks   = [abs_res, rel_res, prior_pen, curv_pen, spacing_pen, bias_pen]
            cost_abs  = 0.5 * np.sum(abs_res ** 2)
            cost_bias = 0.5 * np.sum(bias_pen ** 2)
            cost_total = cost_abs + cost_rel + cost_prior + cost_curv + cost_spacing + cost_bias
            history["cost_abs"].append(cost_abs)
            history["cost_bias"].append(cost_bias)

        if cost_total < best_cost["value"]:
            best_cost["value"] = cost_total

        step_norm = np.nan if prev_p["value"] is None else np.linalg.norm(p - prev_p["value"])
        history["eval"].append(len(history["eval"]))
        history["cost_total"].append(cost_total)
        history["cost_best_so_far"].append(best_cost["value"])
        history["cost_rel"].append(cost_rel)
        history["cost_prior"].append(cost_prior)
        history["cost_curv"].append(cost_curv)
        history["cost_spacing"].append(cost_spacing)
        history["param_norm"].append(np.linalg.norm(p))
        history["step_norm"].append(step_norm)
        history["dx_norm"].append(np.linalg.norm(dx))
        history["dy_norm"].append(np.linalg.norm(dy))
        history["dz_norm"].append(np.linalg.norm(dz))
        prev_p["value"] = p.copy()
        return np.concatenate(blocks)

    result = least_squares(
        residual_vector, x0=x0, method="trf", loss="linear",
        max_nfev=max_nfev, verbose=2,
    )
    print_jacobian_diagnostics(result)

    dx_opt, dy_opt, dz_opt, ctrl_xyz_opt, full_xyz_opt, bias_opt = unpack(result.x)

    out = {
        "result": result,
        "control_channels": control_channels,
        "control_xyz_prior": prior_xyz_ctrl,
        "control_xyz_opt": ctrl_xyz_opt,
        "full_channels": full_channels,
        "prior_xyz_full": prior_xyz_full,
        "full_xyz_opt": full_xyz_opt,
        "obs_indices": obs_ch_idx,
        "ref_indices": ref_ch_idx,
        "tx_xyz": tx_xyz,
        "weights": weights,
        "obs_t_rel": obs_t_rel,
        "sound_speed": sound_speed,
        "history": history,
        "mode": "relative_only" if relative_only else "full",
        # Expose per-control-point sigmas for diagnostics/plotting
        "prior_sigma_xy_ctrl": prior_sigma_xy_ctrl,
        "prior_sigma_z_ctrl":  prior_sigma_z_ctrl,
        "sk_ctrl": sk_ctrl,
    }
    if not relative_only:
        out.update({
            "anchors": anchors,
            "anchor_bias_s": bias_opt,
            "anchor_bias_init_s": init_biases,
            "anchor_idx": anchor_idx,
            "obs_t_abs": obs_t_abs,
        })
    return out


def compute_fit_diagnostics(solution):
    full_xyz = solution["full_xyz_opt"]
    prior_xyz = solution["prior_xyz_full"]
    obs_idx = solution["obs_indices"]
    ref_idx = solution["ref_indices"]
    tx_xyz = solution["tx_xyz"]
    sound_speed = solution["sound_speed"]
    mode = solution.get("mode", "full")

    pred_t_geom_opt = np.linalg.norm(tx_xyz - full_xyz[obs_idx], axis=1) / sound_speed
    pred_t_geom_prior = np.linalg.norm(tx_xyz - prior_xyz[obs_idx], axis=1) / sound_speed
    pred_rel_opt = (
        np.linalg.norm(tx_xyz - full_xyz[obs_idx], axis=1)
        - np.linalg.norm(tx_xyz - full_xyz[ref_idx], axis=1)
    ) / sound_speed
    pred_rel_prior = (
        np.linalg.norm(tx_xyz - prior_xyz[obs_idx], axis=1)
        - np.linalg.norm(tx_xyz - prior_xyz[ref_idx], axis=1)
    ) / sound_speed

    diag = {
        "pred_t_geom_prior": pred_t_geom_prior,
        "pred_t_geom_opt": pred_t_geom_opt,
        "pred_rel_prior": pred_rel_prior,
        "pred_rel_opt": pred_rel_opt,
    }

    if mode != "relative_only":
        anchor_idx = solution["anchor_idx"]
        bias_opt = solution["anchor_bias_s"]
        bias_init = solution["anchor_bias_init_s"]
        diag.update({
            "pred_abs_prior_init_bias": pred_t_geom_prior + bias_init[anchor_idx],
            "pred_abs_prior_opt_bias": pred_t_geom_prior + bias_opt[anchor_idx],
            "pred_abs_opt": pred_t_geom_opt + bias_opt[anchor_idx],
            "anchor_bias_init_row": bias_init[anchor_idx],
            "anchor_bias_opt_row": bias_opt[anchor_idx],
        })
    return diag


def plot_optimizer_history(solution, out_png: str):
    hist = solution.get("history", None)
    if hist is None or len(hist["eval"]) == 0:
        return

    evals = np.asarray(hist["eval"], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(evals, hist["cost_total"], label="Total cost")
    axes[0].plot(evals, hist["cost_best_so_far"], linestyle="--", label="Best so far")
    axes[0].set_ylabel("Total cost")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title("Optimiser convergence")

    for key, label in [
        ("cost_abs", "Absolute traveltime"),
        ("cost_rel", "Relative traveltime"),
        ("cost_prior", "Prior penalty"),
        ("cost_curv", "Curvature penalty"),
        ("cost_spacing", "Spacing penalty"),
        ("cost_bias", "Bias penalty"),
    ]:
        if key in hist:
            axes[1].plot(evals, hist[key], label=label)
    axes[1].set_ylabel("Block cost")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)

    axes[2].plot(evals, hist["step_norm"], label="Step norm")
    axes[2].plot(evals, hist["param_norm"], label="Parameter norm")
    axes[2].set_ylabel("Norm")
    axes[2].set_xlabel("Function evaluation")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_outputs(obs, solution, diagnostics, output_dir, origin_lat, origin_lon, origin_h, channel_quality_df):
    os.makedirs(output_dir, exist_ok=True)

    full_channels = solution["full_channels"]
    prior_xyz = solution["prior_xyz_full"]
    full_xyz = solution["full_xyz_opt"]
    mode = solution.get("mode", "full")

    cable = pd.DataFrame({
        "channel": full_channels,
        "prior_x_m": prior_xyz[:, 0], "prior_y_m": prior_xyz[:, 1], "prior_z_m": prior_xyz[:, 2],
        "x_m": full_xyz[:, 0], "y_m": full_xyz[:, 1], "z_m": full_xyz[:, 2],
    })
    cable["dx_m"] = cable["x_m"] - cable["prior_x_m"]
    cable["dy_m"] = cable["y_m"] - cable["prior_y_m"]
    cable["dz_m"] = cable["z_m"] - cable["prior_z_m"]
    cable["horizontal_shift_m"] = np.sqrt(cable["dx_m"] ** 2 + cable["dy_m"] ** 2)
    cable = maybe_add_latlon(cable, origin_lat, origin_lon, origin_h)
    cable.to_csv(os.path.join(output_dir, "updated_cable_layout.csv"), index=False)

    ctrl = pd.DataFrame({
        "channel": solution["control_channels"],
        "prior_x_m": solution["control_xyz_prior"][:, 0],
        "prior_y_m": solution["control_xyz_prior"][:, 1],
        "prior_z_m": solution["control_xyz_prior"][:, 2],
        "x_m": solution["control_xyz_opt"][:, 0],
        "y_m": solution["control_xyz_opt"][:, 1],
        "z_m": solution["control_xyz_opt"][:, 2],
    })
    ctrl.to_csv(os.path.join(output_dir, "control_points_optimized.csv"), index=False)

    if "anchors" in solution and "anchor_bias_s" in solution:
        pd.DataFrame({
            "anchor_id": solution["anchors"],
            "anchor_bias_s": solution["anchor_bias_s"],
        }).to_csv(os.path.join(output_dir, "anchor_biases.csv"), index=False)

    fit = obs.copy()
    fit["predicted_t_geom_s_prior"] = diagnostics["pred_t_geom_prior"]
    fit["predicted_t_geom_s_opt"] = diagnostics["pred_t_geom_opt"]
    fit["predicted_dt_ref_s_prior"] = diagnostics["pred_rel_prior"]
    fit["predicted_dt_ref_s_opt"] = diagnostics["pred_rel_opt"]
    fit["residual_dt_ref_prior_s"] = fit["observed_dt_ref_s"] - fit["predicted_dt_ref_s_prior"]
    fit["residual_dt_ref_opt_s"] = fit["observed_dt_ref_s"] - fit["predicted_dt_ref_s_opt"]

    has_absolute = "pred_abs_prior_init_bias" in diagnostics
    if has_absolute:
        fit["anchor_bias_s_init"] = diagnostics["anchor_bias_init_row"]
        fit["anchor_bias_s_opt"] = diagnostics["anchor_bias_opt_row"]
        fit["predicted_t_abs_s_prior_init_bias"] = diagnostics["pred_abs_prior_init_bias"]
        fit["predicted_t_abs_s_prior_opt_bias"] = diagnostics["pred_abs_prior_opt_bias"]
        fit["predicted_t_abs_s_opt"] = diagnostics["pred_abs_opt"]
        fit["predicted_t_abs_s_prior"] = fit["predicted_t_abs_s_prior_init_bias"]
        fit["residual_abs_prior_init_bias_s"] = fit["observed_t_s"] - fit["predicted_t_abs_s_prior_init_bias"]
        fit["residual_abs_prior_opt_bias_s"] = fit["observed_t_s"] - fit["predicted_t_abs_s_prior_opt_bias"]
        fit["residual_abs_opt_s"] = fit["observed_t_s"] - fit["predicted_t_abs_s_opt"]
        fit["residual_abs_prior_s"] = fit["residual_abs_prior_init_bias_s"]

    fit.to_csv(os.path.join(output_dir, "observation_fit_diagnostics.csv"), index=False)

    q = channel_quality_df.copy()
    q["is_control_point"] = q["channel"].isin(solution["control_channels"])
    q.to_csv(os.path.join(output_dir, "channel_control_quality.csv"), index=False)

    summary = pd.DataFrame({
        "metric": [
            "mode", "n_observations", "n_control_points", "cost",
            "success", "status", "message", "optimality", "nfev",
            "rmse_rel_prior_ms", "rmse_rel_opt_ms", "weighted_rmse_rel_opt_ms",
            "median_horizontal_shift_m", "p95_horizontal_shift_m",
        ],
        "value": [
            mode, len(obs), len(solution["control_channels"]),
            solution["result"].cost, bool(solution["result"].success),
            solution["result"].status, solution["result"].message,
            solution["result"].optimality, solution["result"].nfev,
            1000.0 * np.sqrt(np.mean(fit["residual_dt_ref_prior_s"] ** 2)),
            1000.0 * np.sqrt(np.mean(fit["residual_dt_ref_opt_s"] ** 2)),
            1000.0 * np.sqrt(np.average(fit["residual_dt_ref_opt_s"] ** 2, weights=fit["weight"])),
            cable["horizontal_shift_m"].median(),
            cable["horizontal_shift_m"].quantile(0.95),
        ],
    })
    summary.to_csv(os.path.join(output_dir, "inversion_summary.csv"), index=False)

    if solution.get("history"):
        pd.DataFrame(solution["history"]).to_csv(
            os.path.join(output_dir, "optimizer_history.csv"), index=False
        )




def make_plots(obs, solution, diagnostics, output_dir, channel_quality_df):
    """Lightweight diagnostic plots generated during inversion run."""
    os.makedirs(output_dir, exist_ok=True)

    full_channels = solution["full_channels"]
    prior_xyz = solution["prior_xyz_full"]
    full_xyz = solution["full_xyz_opt"]
    ctrl_prior = solution["control_xyz_prior"]
    ctrl_opt = solution["control_xyz_opt"]
    ctrl_ch = solution["control_channels"]

    tx_tbl = obs.groupby("anchor_id")[["tx_x_m", "tx_y_m", "tx_u_m"]].first().reset_index()

    fit = obs.copy()
    fit["predicted_dt_ref_s_prior"] = diagnostics["pred_rel_prior"]
    fit["predicted_dt_ref_s_opt"] = diagnostics["pred_rel_opt"]
    fit["residual_dt_ref_prior_s"] = fit["observed_dt_ref_s"] - fit["predicted_dt_ref_s_prior"]
    fit["residual_dt_ref_opt_s"] = fit["observed_dt_ref_s"] - fit["predicted_dt_ref_s_opt"]

    # Plan view
    plt.figure(figsize=(10, 8))
    plt.plot(prior_xyz[:, 0], prior_xyz[:, 1], label="Prior cable", color=PRIOR_GRAY)
    plt.plot(full_xyz[:, 0], full_xyz[:, 1], label="Inverted cable", color=CABLE_BLUE)
    plt.scatter(ctrl_prior[:, 0], ctrl_prior[:, 1], s=18, label="Prior control pts", color=PRIOR_GRAY)
    plt.scatter(ctrl_opt[:, 0], ctrl_opt[:, 1], s=18, label="Optimized control pts", color=CABLE_BLUE)
    plt.scatter(tx_tbl["tx_x_m"], tx_tbl["tx_y_m"], marker="x", s=70, label="Transmitters", color="green")
    plt.xlabel("East (m)"); plt.ylabel("North (m)")
    plt.title("Cable layout: prior vs inverted")
    plt.axis("equal"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_plan_view.png"), dpi=200)
    plt.close()

    # Depth profile
    plt.figure(figsize=(12, 5))
    plt.plot(full_channels, prior_xyz[:, 2], label="Prior z",  color=PRIOR_GRAY)
    plt.plot(full_channels, full_xyz[:, 2], label="Inverted z",color=CABLE_BLUE)
    plt.scatter(ctrl_ch, ctrl_prior[:, 2], s=15, label="Prior control pts", color=PRIOR_GRAY)
    plt.scatter(ctrl_ch, ctrl_opt[:, 2], s=15, label="Optimized control pts", color=CABLE_BLUE)
    plt.xlabel("Channel"); plt.ylabel("Up (m)")
    plt.title("Depth profile"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_depth_profile.png"), dpi=200)
    plt.close()

    # Residual histogram
    rel_prior = fit["residual_dt_ref_prior_s"].to_numpy(dtype=float)
    rel_opt = fit["residual_dt_ref_opt_s"].to_numpy(dtype=float)
    plt.figure(figsize=(10, 6))
    plt.hist(1000 * rel_prior[np.isfinite(rel_prior)], bins=80, alpha=0.5, label="Relative prior",color=PRIOR_GRAY )
    plt.hist(1000 * rel_opt[np.isfinite(rel_opt)], bins=80, alpha=0.5, label="Relative inverted",color=CABLE_BLUE )
    plt.xlabel("Residual (ms)"); plt.ylabel("Count")
    plt.title("Relative-time residuals"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_rel_residual_hist.png"), dpi=200)
    plt.close()

    # S_k and control points
    q = channel_quality_df.copy()
    q["is_control_point"] = q["channel"].isin(ctrl_ch)
    plt.figure(figsize=(13, 4.5))
    plt.plot(q["channel"], q["S_k"], label="$S_k$ (sum of weights)")
    sel = q.loc[q["is_control_point"]]
    if len(sel) > 0:
        plt.scatter(sel["channel"], sel["S_k"], s=22, zorder=3, label="Selected control points")
    plt.xlabel("Channel"); plt.ylabel("$S_k$")
    plt.title("Control-point effective observation count $S_k$")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_control_quality_score.png"), dpi=200)
    plt.close()

    plot_optimizer_history(solution, os.path.join(output_dir, "plot_optimizer_history.png"))


def main():
    parser = argparse.ArgumentParser(description="Invert DAS cable layout from inversion_observations.csv.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_toml(args.config)
    icfg = cfg["inversion"]
    df = pd.read_csv(path_from_cfg(cfg, "inversion_dataset_output_dir") / "inversion_observations.csv")
    output_dir = ensure_dir(path_from_cfg(cfg, "inversion_output_dir"))

    origin_lat = float(df["enu_origin_lat_deg"].dropna().iloc[0])
    origin_lon = float(df["enu_origin_lon_deg"].dropna().iloc[0])
    origin_h = float(df["enu_origin_h_m"].dropna().iloc[0])

    obs = build_observation_table(df, int(icfg["channel_offset"]))
    prior_geom_sparse = build_prior_geometry(df, int(icfg["channel_offset"]))
    prior_full = linear_fill_to_full_channels(prior_geom_sparse)

    min_ch, max_ch = prior_full["channel"].min(), prior_full["channel"].max()
    obs = obs[(obs["channel_eff"] >= min_ch) & (obs["channel_eff"] <= max_ch)].copy()
    obs = obs[(obs["reference_channel_eff"] >= min_ch) & (obs["reference_channel_eff"] <= max_ch)].copy()

    channel_quality_df = summarize_channel_control_quality(obs)

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

    relative_only = bool(icfg.get("relative_only", False))

    solution = solve_inversion(
        obs=obs, prior_full=prior_full, channel_quality_df=channel_quality_df, control_channels=control_channels,
        sound_speed=float(icfg["sound_speed"]), channel_spacing=float(icfg["channel_spacing"]),
        abs_scale=float(icfg.get("abs_scale", 0.003)), rel_scale=float(icfg["rel_scale"]),
        prior_sigma_xy=float(icfg["prior_sigma_xy"]), prior_sigma_z=float(icfg["prior_sigma_z"]),
        curvature_sigma_xy=float(icfg["curvature_sigma_xy"]), curvature_sigma_z=float(icfg["curvature_sigma_z"]),
        spacing_sigma=float(icfg["spacing_sigma"]),
        anchor_bias_sigma=float(icfg.get("anchor_bias_sigma", 0.02)),
        huber_delta_abs=float(icfg.get("huber_delta_abs", 3.0)), huber_delta_rel=float(icfg["huber_delta_rel"]),
        max_nfev=int(icfg["max_nfev"]), relative_only=relative_only,
    )
    diagnostics = compute_fit_diagnostics(solution)
    save_outputs(obs, solution, diagnostics, str(output_dir), origin_lat, origin_lon, origin_h, channel_quality_df)
    make_plots(obs, solution, diagnostics, str(output_dir), channel_quality_df)

    print(f"Saved inversion outputs to: {output_dir}")


if __name__ == "__main__":
    main()
