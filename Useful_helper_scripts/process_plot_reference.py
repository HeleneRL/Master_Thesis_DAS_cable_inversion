from __future__ import annotations

from pathlib import Path
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import spectrogram, chirp, correlate, hilbert, find_peaks
import h5py


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing sequential .hdf5 files
FOLDER = Path(r"D:\Singapore Data\loc6_tx1")

# Plot spectrogram for this channel
CHANNEL = 1150

# Run matched filter on this reference channel
REFERENCE_CHANNEL = 1150

LOCATION = "loc6_tx1"

# Singapore local time = UTC+8
UTC_OFFSET_HOURS = 8

# ---------------- Spectrogram settings ----------------
WINDOW = "hann"
NPERSEG = 4096
NOVERLAP = 3584

FREQ_MIN = 3400
FREQ_MAX = 7000

DB_FLOOR_PERCENTILE = 5
DB_CEIL_PERCENTILE = 99.8

# ---------------- Matched filter settings ----------------
LFM_F0_HZ = 3500.0
LFM_F1_HZ = 4500.0
LFM_DURATION_S = 5.0

# Peaks must be above this normalized matched-filter envelope threshold
PEAK_THRESHOLD = 0.30

# Minimum spacing between peaks
MIN_PEAK_SPACING_SEC = 3.0

# Optional save paths
SAVE_SPECTROGRAM_PATH = None
SAVE_MATCHED_FILTER_PATH = None

# Example:
# SAVE_SPECTROGRAM_PATH = r"D:\Singapore Data\loc3_tx1\spectrogram.png"
# SAVE_MATCHED_FILTER_PATH = r"D:\Singapore Data\loc3_tx1\matched_filter.png"


# ============================================================
# HDF5 HELPERS
# ============================================================

def list_hdf5_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {folder}")
    return files


def read_one_file_one_channel(filepath: Path, channel: int):
    """
    Read one channel directly from one raw OptoDAS HDF5 file.

    Returns:
        y          : 1D np.ndarray, shape (n_samples,)
        fs         : float
        dx         : float
        start_utc  : datetime.datetime
        n_channels : int
    """
    with h5py.File(filepath, "r") as f:
        data = f["data"]
        n_samples, n_channels = data.shape

        if not (0 <= channel < n_channels):
            raise IndexError(
                f"Requested channel {channel}, but file has channels 0..{n_channels-1}"
            )

        y = data[:, channel].astype(np.float64)

        header = f["header"]
        dt_s = float(header["dt"][()])
        fs = 1.0 / dt_s
        dx = float(header["dx"][()]) if "dx" in header else np.nan

        start_unix = float(header["time"][()])
        start_utc = dt.datetime.utcfromtimestamp(start_unix)

        return y, fs, dx, start_utc, n_channels


def load_sequence_one_channel(filepaths: list[Path], channel: int):
    """
    Load a single channel from several sequential HDF5 files and concatenate.
    """
    signals = []
    starts = []
    fs_list = []
    dx_list = []
    n_channels_list = []

    for fp in filepaths:
        y, fs, dx, start_utc, n_channels = read_one_file_one_channel(fp, channel)
        signals.append(y)
        starts.append(start_utc)
        fs_list.append(fs)
        dx_list.append(dx)
        n_channels_list.append(n_channels)
        print(f"Loaded {fp.name}: {len(y)} samples, fs={fs:.3f} Hz")

    fs0 = fs_list[0]
    if not np.allclose(fs_list, fs0):
        raise ValueError(f"Sampling rate differs across files: {fs_list}")

    dx0 = dx_list[0]
    if not np.allclose(dx_list, dx0, equal_nan=True):
        raise ValueError(f"dx differs across files: {dx_list}")

    nch0 = n_channels_list[0]
    if any(n != nch0 for n in n_channels_list):
        raise ValueError(f"n_channels differs across files: {n_channels_list}")

    x = np.concatenate(signals)
    return x, fs0, dx0, starts[0], nch0


# ============================================================
# MATCHED FILTER HELPERS
# ============================================================

def make_lfm_reference(fs: float, f0: float, f1: float, duration: float) -> np.ndarray:
    n = int(round(duration * fs))
    t = np.arange(n, dtype=np.float64) / fs

    ref = chirp(t, f0=f0, f1=f1, t1=duration, method="linear")
    ref *= np.hanning(n)
    ref -= np.mean(ref)
    ref /= np.linalg.norm(ref) + 1e-12

    return ref.astype(np.float64)


def matched_filter_envelope(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Returns normalized envelope of valid matched-filter output.
    Length = len(x) - len(ref) + 1
    """
    xc = correlate(x, ref, mode="valid")
    env = np.abs(hilbert(xc))
    env /= np.max(env) + 1e-12
    return env


def detect_peaks(
    env: np.ndarray,
    fs: float,
    threshold: float,
    min_spacing_sec: float,
) -> list[dict]:
    min_distance_samples = max(1, int(round(min_spacing_sec * fs)))

    peaks, props = find_peaks(
        env,
        height=threshold,
        prominence=0.0,
        distance=min_distance_samples,
    )

    out = []
    for i, p in enumerate(peaks):
        out.append(
            {
                "peak_index_samples": int(p),
                "peak_time_s": float(p / fs),
                "peak_height": float(props["peak_heights"][i]),
                "prominence": float(props["prominences"][i]),
            }
        )

    return out


# ============================================================
# PLOTTING
# ============================================================

def plot_spectrogram(
    x: np.ndarray,
    fs: float,
    dx: float,
    start_utc: dt.datetime,
    n_channels: int,
) -> plt.Figure:
    dist_km = (CHANNEL * dx) / 1000.0 if np.isfinite(dx) else np.nan

    f, t_sec, Sxx = spectrogram(
        x,
        fs=fs,
        window=WINDOW,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        detrend=False,
        scaling="density",
        mode="magnitude",
    )

    Sxx_db = 20 * np.log10(Sxx + 1e-20)

    fmask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
    f_plot = f[fmask]
    S_plot = Sxx_db[fmask, :]

    local_start = start_utc + dt.timedelta(hours=UTC_OFFSET_HOURS)
    t_local = [local_start + dt.timedelta(seconds=float(s)) for s in t_sec]
    t_local_num = mdates.date2num(t_local)

    vmin = np.percentile(S_plot, DB_FLOOR_PERCENTILE)
    vmax = np.percentile(S_plot, DB_CEIL_PERCENTILE)

    fig, ax = plt.subplots(figsize=(7, 12))

    pcm = ax.pcolormesh(
        f_plot,
        t_local_num,
        S_plot.T,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    title = (
        f"{LOCATION}, channel {CHANNEL}, at dist {dist_km:.6f} km"
        if np.isfinite(dist_km)
        else f"{LOCATION}, channel {CHANNEL}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (local)")
    ax.set_xlim(FREQ_MIN, FREQ_MAX)

    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Magnitude (dB)")

    plt.tight_layout()

    print()
    print("=== Spectrogram info ===")
    print(f"Channel           : {CHANNEL}")
    print(f"n_channels/file   : {n_channels}")
    print(f"dx                : {dx:.6f} m")
    print(f"distance          : {dist_km:.6f} km")
    print(f"fs                : {fs:.3f} Hz")
    print(f"UTC start         : {start_utc}")
    print(f"Local start       : {local_start}")
    print(f"Duration          : {len(x)/fs:.3f} s")

    return fig


def plot_matched_filter(
    env: np.ndarray,
    fs: float,
    peaks: list[dict],
    start_utc: dt.datetime,
) -> plt.Figure:
    t = np.arange(len(env), dtype=np.float64) / fs
    local_start = start_utc + dt.timedelta(hours=UTC_OFFSET_HOURS)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, env, linewidth=1.0, label="Matched filter envelope")
    ax.axhline(PEAK_THRESHOLD, linestyle="--", linewidth=1.0, label=f"Threshold = {PEAK_THRESHOLD:.3f}")

    if peaks:
        peak_times = [p["peak_time_s"] for p in peaks]
        peak_vals = [p["peak_height"] for p in peaks]
        ax.scatter(peak_times, peak_vals, s=45, marker="o", label="Detected peaks")

        for i, p in enumerate(peaks, start=1):
            ax.axvline(p["peak_time_s"], linestyle="--", alpha=0.5)
            ax.text(
                p["peak_time_s"],
                p["peak_height"],
                f"{i}",
                rotation=90,
                va="bottom",
                ha="left",
            )

    ax.set_title(
        f"{LOCATION}, reference channel {REFERENCE_CHANNEL}, matched filter "
        f"({LFM_F0_HZ/1000:.1f}-{LFM_F1_HZ/1000:.1f} kHz, {LFM_DURATION_S:.1f} s)"
    )
    ax.set_xlabel("Time since concatenated start (s)")
    ax.set_ylabel("Normalized envelope")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    print()
    print("=== Matched filter info ===")
    print(f"Reference channel : {REFERENCE_CHANNEL}")
    print(f"LFM sweep         : {LFM_F0_HZ:.1f} Hz -> {LFM_F1_HZ:.1f} Hz")
    print(f"LFM duration      : {LFM_DURATION_S:.3f} s")
    print(f"Threshold         : {PEAK_THRESHOLD:.3f}")
    print(f"Min peak spacing  : {MIN_PEAK_SPACING_SEC:.3f} s")
    print(f"Start UTC         : {start_utc}")
    print(f"Start local       : {local_start}")

    print()
    print("Detected peaks:")
    if not peaks:
        print("  No peaks found.")
    else:
        for i, p in enumerate(peaks, start=1):
            peak_local = local_start + dt.timedelta(seconds=p["peak_time_s"])
            print(
                f"  Peak {i:02d}: "
                f"sample={p['peak_index_samples']}, "
                f"time_s={p['peak_time_s']:.6f}, "
                f"height={p['peak_height']:.6f}, "
                f"prominence={p['prominence']:.6f}, "
                f"local_time={peak_local.strftime('%H:%M:%S.%f')[:-3]}"
            )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    files = list_hdf5_files(FOLDER)

    print(f"Folder: {FOLDER}")
    print(f"Found {len(files)} .hdf5 files")
    print("Files used:")
    for fp in files:
        print(f"  {fp.name}")

    # Spectrogram channel
    x_spec, fs_spec, dx, start_utc_spec, n_channels_spec = load_sequence_one_channel(files, CHANNEL)

    # Reference channel for matched filter
    x_ref, fs_ref, dx_ref, start_utc_ref, n_channels_ref = load_sequence_one_channel(files, REFERENCE_CHANNEL)

    if not np.isclose(fs_spec, fs_ref):
        raise RuntimeError(
            f"Sampling rate mismatch between channels: {fs_spec} vs {fs_ref}"
        )

    if start_utc_spec != start_utc_ref:
        raise RuntimeError(
            f"Start time mismatch between channels: {start_utc_spec} vs {start_utc_ref}"
        )

    # ---------------- Plot 1: spectrogram ----------------
    fig1 = plot_spectrogram(
        x=x_spec,
        fs=fs_spec,
        dx=dx,
        start_utc=start_utc_spec,
        n_channels=n_channels_spec,
    )

    if SAVE_SPECTROGRAM_PATH is not None:
        out1 = Path(SAVE_SPECTROGRAM_PATH)
        out1.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out1, dpi=220, bbox_inches="tight")
        print(f"Saved spectrogram to: {out1}")

    # ---------------- Plot 2: matched filter ----------------
    x_ref = x_ref - np.mean(x_ref)

    ref = make_lfm_reference(
        fs=fs_ref,
        f0=LFM_F0_HZ,
        f1=LFM_F1_HZ,
        duration=LFM_DURATION_S,
    )

    env = matched_filter_envelope(x_ref, ref)

    peaks = detect_peaks(
        env=env,
        fs=fs_ref,
        threshold=PEAK_THRESHOLD,
        min_spacing_sec=MIN_PEAK_SPACING_SEC,
    )

    fig2 = plot_matched_filter(
        env=env,
        fs=fs_ref,
        peaks=peaks,
        start_utc=start_utc_ref,
    )

    if SAVE_MATCHED_FILTER_PATH is not None:
        out2 = Path(SAVE_MATCHED_FILTER_PATH)
        out2.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out2, dpi=220, bbox_inches="tight")
        print(f"Saved matched-filter plot to: {out2}")

    plt.show()


if __name__ == "__main__":
    main()