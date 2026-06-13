# Position Estimation of Submarine Fiber-Optic Cables using Distributed Acoustic Sensing and Acoustic Sources

Code accompanying the master's thesis:

> **"Position Estimation of Submarine Fiber-Optic Cables Using Distributed Acoustic Sensing and Acoustic Sources"**
> 
> Helene Randem Lunde
> Signal Processing and Communications,
> Department of Electronic Systems, NTNU, Spring 2026.

This repository is referenced in the appendix of the thesis.

## Scope of this repository

This repository contains **only the Singapore inversion pipeline.**

The thesis evaluates the inversion framework on two datasets: the Singapore
2 km cable loop and the Trondheimsfjord deployment. Only the Singapore
pipeline is published here. This is a deliberate choice.

The framework is built around a dataset-**independent** inversion core (the
"inversion block" in Figure 4.1.1 of the thesis) that is identical for every
dataset. What differs from one dataset to the next is only the front-end:
how raw DAS recordings are turned into the three inputs the inversion block
needs (a prior cable geometry, source positions per transmission, and
weighted relative arrival times).

If you want to run this on your **own** data, you need to adapt the Singapore
front-end to your acquisition (your detector template, your source-position
source, your prior construction) and then you can reuse the same inversion
core. See the per-dataset notes in `Singapore_inversion_pipeline/README.md`.

## Input data not included

The raw DAS recordings, GPS tracks and prior geometry files are **not** part of
this repository. The Singapore dataset was shared by collaborators at NUS and
is not redistributed here. To reproduce the thesis results you need the
original data. To run on your own data, supply the equivalent files and point
the config at them.

## Repository layout

```text
Singapore_inversion_pipeline/     The full, runnable inversion pipeline.
    config/pipeline_config.toml   Single config file for every stage.
    src/                          All pipeline stages + the solver.
    README.md                     Detailed pipeline documentation.

Useful_helper_scripts/            Stand-alone analysis / QC scripts.
                                  These are NOT part of the core inversion.

README.md                         This file.
```

## What the pipeline does (very high level)

The pipeline estimates the 3D geometry of a submarine fiber-optic cable from
the direct-wave arrival times of known acoustic transmissions recorded by a
DAS system. In order, it:

1. Detects arrivals in the raw DAS recordings with a matched filter
   (`detector_bulk.py`).
2. Matches transmission timestamps to source GPS positions
   (`build_transmitter_positions.py`).
3. Builds a channel-indexed prior cable geometry
   (`build_prior_cable_3d_consistent.py`).
4. Builds per-channel/per-location trust ("smooth arrival curves")
   (`build_trust_map.py`).
5. Merges everything into one weighted observation table
   (`build_inversion_dataset.py`).
6. Solves the inversion: the cable is parameterized by a sparse set of
   control points joined by a cubic spline, and a Trust-Region Reflective
   least-squares solver balances data fit against prior, curvature and
   spacing regularization (`invert_cable_diagnostics.py`).
7. Produces figures (`make_plots_relative_only.py`).

Full method details are in Chapter 4 of the thesis.

## Quick start

Requires Python 3.11+. Install dependencies:

```bash
pip install numpy pandas scipy matplotlib h5py pymap3d pyproj
```

Edit the file paths in `Singapore_inversion_pipeline/config/pipeline_config.toml`
to point at your data (see the **Config** section below), then run:

```bash
cd Singapore_inversion_pipeline
python src/run_pipeline.py --config config/pipeline_config.toml
```

The detailed pipeline README explains how to run individual stages.

## Config: what you must change before running on your own data

Everything that is dataset-specific lives in
`Singapore_inversion_pipeline/config/pipeline_config.toml`. To run on your own
data, the items below **must** be reviewed.

### 1. File paths — `[paths]` section

These are Windows paths from the original machine (e.g. `D:\Singapore Data\...`)
and **will not** exist on your system. Change every one of them:

| Key | What it points to |
| --- | --- |
| `data_root` | Root folder holding the per-location DAS recording folders (the `.hdf5` sequences). |
| `raw_detection_output_dir` | Where detector output is written. |
| `trust_output_dir` | Where trust summaries are written. |
| `transmitter_output_dir` | Where matched source positions are written. |
| `prior_output_dir` | Where the prior geometry is written. |
| `inversion_dataset_output_dir` | Where the merged observation table goes. |
| `inversion_output_dir` | Where the inversion result + plots go. |
| `sweep_times_csv` | Transmission times file. |
| `tx_gps_csv` | Source/vessel GPS track file. |
| `boattrack_csv` | Boat-track estimate used to build the prior. |
| `cable_estimate_csv` | Reference/prior cable estimate (also used as the "truth-like" geometry in the plots). |

> **Note:** the input files above must exist and have the expected columns.

### 2. ENU origin — `[inversion_dataset]` section

The local East–North–Up coordinate origin is set to Singapore:

```toml
enu_lat0_deg = 1.2160
enu_lon0_deg = 103.8518
enu_h0_m     = 0.0
```

Set this to a point near **your** cable.

### 3. Channel range

`channel_min` / `channel_max` (and `first_channel` / `last_channel` in
`[prior_geometry]`) are set to the Singapore in-water channel span (348–2267).
Set these to your own cable's channel range.

### 4. Source locations

The `[locations.*]` blocks at the bottom are entirely Singapore-specific: each
gives a folder name, a reference channel, and the transmission ("anchor") times
in seconds for that location. Replace these with your own transmission sites,
reference channels and times.

### 5. Detector template — `[detector]` section

Assumes the Singapore 3.5–4.5 kHz LFM sweep (`lfm_f0_hz`, `lfm_f1_hz`,
`lfm_duration_s`) and a 25 kHz sample rate. Change to match your transmitted
signal and acquisition.

### 6. Physical / solver parameters

| Parameter | Meaning |
| --- | --- |
| `sound_speed` | 1500 m/s nominal seawater (Singapore). |
| `channel_spacing` | 1.02 m, the DAS gauge spacing. |
| `prior_sigma_xy` / `prior_sigma_z` | How far the cable may move from the prior, in metres. |
| `curvature_sigma_xy` / `curvature_sigma_z` | How sharply it may bend. |
| `spacing_sigma` | Allowed deviation in channel spacing. |
| `rel_scale`, `huber_delta_rel` | Residual scale / robust-loss knee. |

The meaning of each is documented inline in the config and in Section 4.8 and
Table 4.8.1 of the thesis.

## Citation

If you use this code, please cite the thesis:

> H. R. Lunde, "Position Estimation of Submarine Fiber-Optic Cables Using
> Distributed Acoustic Sensing and Acoustic Sources," Master's thesis,
> Department of Electronic Systems, NTNU, 2026.
