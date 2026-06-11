# Singapore DAS Cable-Inversion Pipeline

This folder contains the full, runnable pipeline that estimates a submarine
fiber-optic cable's 3D geometry from DAS-recorded acoustic arrival times, as
described in Chapter 4 of the thesis. It is the Singapore branch of the
framework; the dataset-independent inversion core (`invert_cable_diagnostics.py`)
is reusable for any dataset (see the top-level `README.txt` for scope).

## Folder layout

```text
config/
  pipeline_config.toml      Single config file for every stage.

src/
  common.py                         Shared helpers (config loading, paths).
  detector_bulk.py                  Matched-filter detection + pick-quality scoring.
  build_transmitter_positions.py    Matches transmission times to source GPS.
  build_prior_cable_3d_consistent.py  Builds channel-indexed prior geometry.
  build_trust_map.py                Per-location smooth arrival curves / trust.
  build_inversion_dataset.py        Merges everything into one weighted table.
  invert_cable_diagnostics.py       The inversion solver (relative arrivals).
  make_plots_relative_only.py       Thesis figures from the inversion output.
  geometric_conditioning.py         DOP / conditioning metrics + skyplots (used
                                    by the inversion diagnostics).
  run_pipeline.py                   Runs every stage in order.
```

Output directories (raw detections, trust, transmitter, prior geometry,
inversion dataset, inversion) are created automatically under the paths set in
the config.

## Running the pipeline

Requires Python 3.11+ (3.12 tested).

```bash
pip install numpy pandas scipy matplotlib h5py pymap3d
# optional: pyproj (lat/lon output on inverted cable), tomli (Python < 3.11)
```

First edit the paths and dataset-specific values in
`config/pipeline_config.toml` (see the top-level `README.txt` for the full
checklist of what must change). Then run the whole chain:

```bash
python src/run_pipeline.py --config config/pipeline_config.toml
```

`run_pipeline.py` executes the stages in this exact order:

1. `detector_bulk.py`
2. `build_transmitter_positions.py`
3. `build_prior_cable_3d_consistent.py`
4. `build_trust_map.py`
5. `build_inversion_dataset.py`
6. `invert_cable_diagnostics.py`
7. `make_plots_relative_only.py`

You can also run any stage on its own with the same `--config` argument, e.g.:

```bash
python src/detector_bulk.py --config config/pipeline_config.toml
python src/invert_cable_diagnostics.py --config config/pipeline_config.toml
python src/make_plots_relative_only.py --config config/pipeline_config.toml
```

The plotting stage is post-inversion only: it reads the solver output and can
be rerun freely (e.g. after styling changes) without recomputing the inversion.

## What each stage does

- **detector_bulk.py** reads the per-location HDF5 DAS sequences and writes
  per-channel arrival detections, each with a composite pick-quality score
  (peak-to-noise, peak dominance, peak sharpness). See Section 4.4.

- **build_transmitter_positions.py** matches each transmission timestamp to a
  source GPS position. See Section 4.5.

- **build_prior_cable_3d_consistent.py** interpolates a channel-indexed prior
  cable geometry from the boat-track and cable-estimate inputs, resampled so
  neighbouring channels are one DAS gauge length apart. See Section 4.2.

- **build_trust_map.py** builds the smoothed (rolling median + mean) arrival
  curves per location used to down-weight noise picks. See Section 4.5.2.

- **build_inversion_dataset.py** merges arrivals, source positions, prior
  geometry and trust into one table of weighted relative observations. The
  composite weight multiplies pick quality, smooth-curve residual and
  sweep-disagreement penalties. See Section 4.5.3.

- **invert_cable_diagnostics.py** runs the inversion. The cable is
  parameterized by a sparse set of control points joined by a cubic spline
  (linear in depth); a Trust-Region Reflective least-squares solver minimizes a
  data term plus prior, curvature and spacing regularization, using a Huber
  robust loss. Outputs the updated geometry plus diagnostics. See Sections
  4.7-4.8.

- **make_plots_relative_only.py** produces the thesis figure set (plan view
  with uncertainty tube, depth profile, observed-vs-predicted, residual
  histograms, etc.) from the inversion output.

## Recursive (coarse-to-fine) use

As described in Section 5.1.2, the inversion can be applied recursively: run a
coarse pass (sparse control points, loose prior penalty), then feed its output
geometry back in as the prior for a finer pass (denser control points, tighter
regularization). This recovers the large-scale shift cheaply before refining
local detail. To do this, point `cable_estimate_csv` / the prior at the
first-pass output and tighten `prior_sigma_xy` and the control-point density in
the config for the second run.

## Notes

- This is a clean operational baseline, not a full research framework.
- The observation weight uses pick-level quality explicitly, not only
  channel-level trust.
- The uncertainty tube in the plots is a diagnostic derived from relative
  residual misfit, not a formal posterior confidence interval.
