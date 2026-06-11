from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d as pm
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, rfftfreq

from scipy.signal import find_peaks, detrend


from common import load_toml, ensure_dir, path_from_cfg

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_boattrack(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Boattrack CSV is missing columns: {missing}")
    out = pd.DataFrame({
        "lon": pd.to_numeric(df["X"], errors="coerce"),
        "lat": pd.to_numeric(df["Y"], errors="coerce"),
    })
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("Boattrack must contain at least 2 valid points.")
    return out


def read_cable_estimate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"lat", "lon", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cable estimate CSV is missing columns: {missing}")
    out = pd.DataFrame({
        "lat":   pd.to_numeric(df["lat"], errors="coerce"),
        "lon":   pd.to_numeric(df["lon"], errors="coerce"),
        "depth": pd.to_numeric(df["z"],   errors="coerce"),
    })
    # Keep channel index if present
    if "ch" in df.columns:
        out["ch"] = pd.to_numeric(df["ch"], errors="coerce")
    out = out.dropna(subset=["lat", "lon", "depth"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def latlon_to_local_xy(lat, lon, lat0, lon0):
    e, n, _ = pm.geodetic2enu(lat, lon, 0.0, lat0, lon0, 0.0)
    return np.asarray(e), np.asarray(n)


def local_xy_to_latlon(e, n, lat0, lon0):
    lat, lon, _ = pm.enu2geodetic(e, n, 0.0, lat0, lon0, 0.0)
    return np.asarray(lat), np.asarray(lon)


def remove_duplicate_consecutive_points(x, y, z=None):
    pts = np.column_stack([x, y]) if z is None else np.column_stack([x, y, z])
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts2 = pts[keep]
    if z is None:
        return pts2[:, 0], pts2[:, 1]
    return pts2[:, 0], pts2[:, 1], pts2[:, 2]


def cumulative_length_2d(x, y):
    seglen = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return s, seglen


def cumulative_length_3d(x, y, z):
    seglen = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return s, seglen


# ---------------------------------------------------------------------------
# Horizontal prior: interpolate along boat track
# ---------------------------------------------------------------------------

def interpolate_along_polyline_2d(x, y, distances):
    """
    Interpolate (x, y) positions along a 2-D polyline at the given arc-length
    distances.  Points beyond the end of the polyline are extrapolated along
    the last segment direction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    distances = np.asarray(distances, dtype=float)

    s, seglen = cumulative_length_2d(x, y)
    total_len = s[-1]

    valid = np.where(seglen > 0)[0]
    if len(valid) == 0:
        raise ValueError("All polyline segments have zero length.")
    last_i = valid[-1]
    p0 = np.array([x[last_i], y[last_i]])
    p1 = np.array([x[last_i + 1], y[last_i + 1]])
    last_dir = (p1 - p0) / np.linalg.norm(p1 - p0)

    xi = np.empty_like(distances)
    yi = np.empty_like(distances)

    inside  = distances <= total_len
    outside = ~inside

    if np.any(inside):
        d  = distances[inside]
        j  = np.searchsorted(s, d, side="right") - 1
        j  = np.clip(j, 0, len(s) - 2)
        sl = s[j + 1] - s[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (d - s[j]) / sl
        t = np.where(sl > 0, t, 0.0)
        xi[inside] = x[j] + t * (x[j + 1] - x[j])
        yi[inside] = y[j] + t * (y[j + 1] - y[j])

    if np.any(outside):
        extra = distances[outside] - total_len
        end   = np.array([x[-1], y[-1]])
        pts   = end[None, :] + extra[:, None] * last_dir[None, :]
        xi[outside] = pts[:, 0]
        yi[outside] = pts[:, 1]

    return xi, yi, total_len


# ---------------------------------------------------------------------------
# Depth prior: smooth the Singapore team's estimate, assign per channel
# ---------------------------------------------------------------------------

def build_smoothed_depth_prior(
    channel_ids: np.ndarray,
    cable_df: pd.DataFrame,
    smooth_sigma_channels: float,
) -> np.ndarray:
    # Direct channel-to-channel lookup
    if "ch" not in cable_df.columns:
        raise ValueError("cable_df must have a 'ch' column for channel-indexed depth assignment.")

    ch_to_depth = cable_df.set_index("ch")["depth"]

    depth_raw = np.array([
        float(ch_to_depth.get(int(ch), np.nan))
        for ch in channel_ids
    ])

    # Fill any channels not present in the estimate by linear interpolation
    finite = np.isfinite(depth_raw)
    if not np.all(finite):
        x_all = np.arange(len(depth_raw))
        depth_raw = np.interp(x_all, x_all[finite], depth_raw[finite])

    depth_smooth = gaussian_filter1d(depth_raw, sigma=float(smooth_sigma_channels))
    return depth_smooth


def build_3d_consistent_horizontal_distances(
    depth_by_channel: np.ndarray,
    channel_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert nominal DAS channel spacing along the 3-D fibre into the
    corresponding horizontal walking distances along the 2-D prior track.

    The depth prior is already channel-indexed.  Between neighbouring
    channels i and i+1, the physical fibre spacing is assumed to be
    channel_spacing_m, so the horizontal increment is

        sqrt(channel_spacing_m**2 - dz**2)

    where dz = depth[i+1] - depth[i].  The returned cumulative distances
    can be passed to interpolate_along_polyline_2d.
    """
    depth_by_channel = np.asarray(depth_by_channel, dtype=float)
    dz = np.diff(depth_by_channel)

    max_abs_dz = float(np.max(np.abs(dz))) if len(dz) else 0.0
    if np.any(np.abs(dz) >= channel_spacing_m):
        bad = np.where(np.abs(dz) >= channel_spacing_m)[0]
        first_bad = int(bad[0])
        raise ValueError(
            "Depth change between adjacent channels is too large for the "
            "requested 3-D channel spacing. This would make the horizontal "
            "step imaginary. Consider increasing depth smoothing. "
            f"First offending channel step index: {first_bad}; "
            f"|dz| = {abs(dz[first_bad]):.3f} m, "
            f"spacing = {channel_spacing_m:.3f} m."
        )

    horizontal_step = np.sqrt(channel_spacing_m**2 - dz**2)
    horizontal_s = np.concatenate([[0.0], np.cumsum(horizontal_step)])

    return horizontal_s, horizontal_step, dz

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build channel-indexed prior cable geometry."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg   = load_toml(args.config)
    pcfg  = cfg["prior_geometry"]
    outdir = ensure_dir(path_from_cfg(cfg, "prior_output_dir"))

    boat_df  = read_boattrack(Path(cfg["paths"]["boattrack_csv"]))
    cable_df = read_cable_estimate(Path(cfg["paths"]["cable_estimate_csv"]))

    # Shared ENU origin: centroid of both datasets
    all_lats = np.r_[boat_df["lat"].values, cable_df["lat"].values]
    all_lons = np.r_[boat_df["lon"].values, cable_df["lon"].values]
    lat0 = float(np.mean(all_lats))
    lon0 = float(np.mean(all_lons))

    boat_x, boat_y = latlon_to_local_xy(boat_df["lat"].values, boat_df["lon"].values, lat0, lon0)
    boat_x, boat_y = remove_duplicate_consecutive_points(boat_x, boat_y)

    # ------------------------------------------------------------------
    # Channel and depth prior
    # ------------------------------------------------------------------
    first_channel     = int(pcfg["first_channel"])
    last_channel      = int(pcfg["last_channel"])
    channel_spacing_m = float(pcfg["channel_spacing_m"])

    n_channels  = last_channel - first_channel + 1
    channel_ids = np.arange(first_channel, last_channel + 1)

    # The Singapore depth estimate is channel-indexed.  We therefore first
    # assign/smooth depth by channel, then use adjacent depth differences to
    # choose how far to walk horizontally along the 2-D prior track so that
    # neighbouring channels are separated by channel_spacing_m in 3-D.
    smooth_sigma = float(pcfg.get("depth_smooth_sigma_channels", 20.0))
    depth_smooth = build_smoothed_depth_prior(
        channel_ids=channel_ids,
        cable_df=cable_df,
        smooth_sigma_channels=smooth_sigma,
    )

    # ------------------------------------------------------------------
    # Horizontal prior: 3-D-consistent sampling along the boat track
    # ------------------------------------------------------------------
    horizontal_s, horizontal_step, dz_channel = build_3d_consistent_horizontal_distances(
        depth_by_channel=depth_smooth,
        channel_spacing_m=channel_spacing_m,
    )

    interp_x, interp_y, total_len_2d = interpolate_along_polyline_2d(
        boat_x, boat_y, horizontal_s
    )
    interp_lat, interp_lon = local_xy_to_latlon(interp_x, interp_y, lat0, lon0)
    channel_xy = np.column_stack([interp_x, interp_y])

    # Diagnostic only: what the old purely-horizontal construction would have
    # used.  This is not used to build the output geometry.
    channel_s_2d_nominal = np.arange(n_channels, dtype=float) * channel_spacing_m

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------

    # The inversion expects columns: channel, lat, lon, depth
    # It also needs x_m, y_m, u_m (ENU) which are added here for convenience
    # so that build_inversion_dataset does not need to reproject.
    interp_df = pd.DataFrame({
        "channel": channel_ids,
        "lat":     interp_lat,
        "lon":     interp_lon,
        "depth":   depth_smooth,
        "x_m":     interp_x,
        "y_m":     interp_y,
        "u_m":     depth_smooth,          # z = depth in ENU
        "enu_origin_lat_deg": lat0,
        "enu_origin_lon_deg": lon0,
        "enu_origin_h_m":     0.0,
    })

    # Save the channel-indexed prior (the canonical output consumed downstream)
    interp_df[["channel", "lat", "lon", "depth",
               "x_m", "y_m", "u_m",
               "enu_origin_lat_deg", "enu_origin_lon_deg", "enu_origin_h_m"]
    ].to_csv(outdir / "prior_cable_by_channel.csv", index=False)

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    cable_x, cable_y = latlon_to_local_xy(
        cable_df["lat"].values, cable_df["lon"].values, lat0, lon0
    )

    # Raw depth from nearest-neighbour lookup (before smoothing) for comparison
    if HAVE_SCIPY:
        from scipy.spatial import cKDTree as _KDTree
        tree = _KDTree(np.column_stack([cable_x, cable_y]))
        _, idx_raw = tree.query(channel_xy, k=1)
    else:
        idx_raw = np.array([
            int(np.argmin(np.sum((np.column_stack([cable_x, cable_y]) - p) ** 2, axis=1)))
            for p in channel_xy
        ])
    depth_raw_assigned = cable_df["depth"].values[idx_raw]

    # Plan view
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(boat_df["lon"], boat_df["lat"], label="Boat track", linewidth=1.2, color="0.55")
    ax.plot(cable_df["lon"], cable_df["lat"], label="Cable estimate (Singapore)", linewidth=1.2, color="#ff7f0e")
    ax.plot(interp_df["lon"], interp_df["lat"], label="Interpolated prior", linewidth=2.0, color="#1f77b4")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Prior cable geometry")
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "prior_geometry_map.png", dpi=200)
    plt.close(fig)

    # Depth profile: raw assigned, smoothed prior, original cable estimate
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(channel_ids, depth_smooth, linewidth=2.2, color="#1f77b4",
            label=f"Smoothed prior depth ($\\sigma$ = {smooth_sigma:.0f} ch)")
    ax.plot(cable_df["ch"], cable_df["depth"].values,
            linewidth=0.7, alpha=1, color="#ff7f0e",
            label="Singapore cable estimate (original)")
    #ax.set_ylim(-20,-3)
    #ax.set_xlim(2080,2140)
    #ax.set_aspect('equal')   # <- key line
    ax.set_xlabel("Channel")
    ax.set_ylabel("Depth (m, negative = below surface)")
    ax.set_title("Prior depth profile")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "prior_geometry_depth.png", dpi=200)
    #plt.close(fig)



    dx = 1.02  # meters per channel

    # Use original orange depth
    ch = cable_df["ch"].values.astype(float)
    depth = cable_df["depth"].values.astype(float)

    # Remove NaNs
    valid = np.isfinite(ch) & np.isfinite(depth)
    ch = ch[valid]
    depth = depth[valid]

    # Remove large-scale bathymetry trend
    # choose sigma larger than the suspected roll wiggle
    trend_sigma_ch = 30
    depth_trend = gaussian_filter1d(depth, sigma=trend_sigma_ch)

    residual = depth - depth_trend
    residual = detrend(residual)  # removes leftover linear trend

    # Optional window to reduce edge effects
    window = np.hanning(len(residual))
    residual_win = residual * window

    # FFT
    N = len(residual_win)
    freq = rfftfreq(N, d=dx)      # cycles per meter
    fft_vals = rfft(residual_win)

    amplitude = np.abs(fft_vals) / N
    amplitude[1:-1] *= 2

    # Ignore zero frequency
    valid_freq = freq > 0
    freq_plot = freq[valid_freq]
    amp_plot = amplitude[valid_freq]
    wavelength_plot = 1 / freq_plot

    # Find strongest peaks
    peaks, props = find_peaks(amp_plot, prominence=0.02)

    # Sort peaks by amplitude
    peak_order = np.argsort(amp_plot[peaks])[::-1]
    top_peaks = peaks[peak_order[:20]]

    print("Strongest spatial-frequency peaks:")
    for p in top_peaks:
        f = freq_plot[p]
        wavelength = 1 / f
        amp = amp_plot[p]
        print(f"freq = {f:.5f} cycles/m, wavelength = {wavelength:.1f} m, amplitude = {amp:.3f} m")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(ch, residual, linewidth=0.8, color="#ff7f0e")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Depth residual (m)")
    ax.set_title("High-frequency depth residual after removing smooth bathymetry")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "depth_residual_highfreq.png", dpi=200)
    #plt.close(fig)


    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(freq_plot, amp_plot, linewidth=1.0)

    for p in top_peaks[:10]:
        ax.axvline(freq_plot[p], linestyle="--", alpha=0.4)

    ax.set_xlabel("Spatial frequency (cycles/m)")
    ax.set_ylabel("Amplitude (m)")
    ax.set_title("FFT of high-frequency depth residual")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "depth_residual_fft.png", dpi=200)
    #plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(wavelength_plot, amp_plot, linewidth=1.0)

    for p in top_peaks[:10]:
        ax.axvline(wavelength_plot[p], linestyle="--", alpha=0.4)

    ax.set_xlabel("Spatial wavelength (m)")
    ax.set_ylabel("Amplitude (m)")
    ax.set_title("FFT of high-frequency depth residual, shown as wavelength")
    ax.set_xscale("log")
    ax.invert_xaxis()  # short wavelengths on the right/left depending on taste
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "depth_residual_fft_wavelength.png", dpi=200)
    #plt.close(fig)

    plt.show()


    

    print(f"Saved: {outdir / 'prior_cable_by_channel.csv'}")
    print(f"Channels: {first_channel} – {last_channel}  ({n_channels} total)")
    print(f"2D polyline length available: {total_len_2d:.2f} m")
    print(f"3D fibre length needed:       {(n_channels - 1) * channel_spacing_m:.2f} m")
    print(f"Horizontal length used:       {horizontal_s[-1]:.2f} m")
    print(f"Old 2D-only length would be:  {channel_s_2d_nominal[-1]:.2f} m")
    print(f"3D correction in plan view:   {channel_s_2d_nominal[-1] - horizontal_s[-1]:.3f} m")
    print(f"Max adjacent |dz|:            {np.max(np.abs(dz_channel)):.3f} m")
    print(f"Min horizontal step:          {np.min(horizontal_step):.3f} m")
    print(f"Depth smooth sigma:           {smooth_sigma:.1f} channels")
    print(f"Depth range (smoothed):       {depth_smooth.min():.1f} – {depth_smooth.max():.1f} m")


if __name__ == "__main__":
    main()