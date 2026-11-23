# SCOUT: self-driving scan paths in AFM — example datasets (lite, v4)

Measured datasets for *SCOUT: Self-Driving Scan Paths in Atomic Force Microscopy* (M. Checa,
Center for Nanophase Materials Sciences, Oak Ridge National Laboratory), recorded with the
`selfdriving_afm` (SCOUT) package v0.4.0. Every self-driving run is stored with its
**equivalent raster**: a raster with the same number of measured pixels, taken right after
the run over the same area.

These are **lite** copies: every setting, calibration, prior, traced feature and image of the
original files is kept, and the long sample streams are thinned (`pts_stride`, see below). On
the GP data the lite copies reproduce the full-data numbers (Fig. 4 example: 16.3 % of pixels
on objects and a 16.3 nm vs 17.5 nm TV error for SCOUT vs its raster, from both full and lite
files).

## Contents

| folder | mode | what |
|---|---|---|
| `00_ground_truth/` | raster | 256-line and 150-line rasters of the shapes sample (SiO₂ disc + 4 hexagons, 116 nm): the reference for every 2026-11-09 score |
| `01_grid_refine_stepwise/` | grid_refine | stage 1 survey spiral → stage 2 one spiral per object → stage 3 dense spiral per object (PyScanner spirals, no tip jumps), 2026-11-03 |
| `02_edge_explorer/` | edge_explorer | contour following with closed-loop position sensors, 2026-10-14 |
| `03_edge_explorer_L2_contour/` | edge_explorer | contour following on the shapes sample, 2026-11-09 |
| `04_gp_path_lambda_sweep/` | gp_path | GP pathfinder (preset `pure_gp_smooth`, 240 V of path), λ = 0, 0.25, 0.5, 0.75, 1 |
| `05_gp_path_edge_focus/` | gp_path | λ = 1, γ = 2 (preset `pure_gp_edge_focus`): the most edge-focused GP run |
| `06_gp_path_edges_preset/` | gp_path | preset `pure_gp_edges_smooth` |
| `07_hybrids/` | gp_edge_explorer, grid_tracker, gp_spiral | combining levels: GP or survey exploration between objects + edge tracing or spirals on them |
| `08_pfm_domain_walls/` | domain_wall_explorer | PFM: following ferroelectric domain walls (phase flip / amplitude dip), 2026-10-16 and 2026-11-04; amplitude and phase along the path (and in the raster) |

`manifest.csv` / `manifest.json`: one row per file (mode, description, samples, stride, samples
in the original, raster samples, channels, units, µm per V, size, SHA-256, source file name).
`previews/`: one image per file (path start → end, binned map; same for its raster).

## File format (`.npz`, NumPy)

| array | meaning |
|---|---|
| `pts` (N × 3, float32) | x, y (drive volts, at the **measured** positions: position sensors) and z along the tip path, in time order |
| `pts_index`, `pts_stride`, `pts_n` | the sample index of each kept point in the saved stream, the stride (1 kept in `pts_stride`), and the stream's length |
| `pts_ch_amplitude`, `pts_ch_phase` | PFM channels along the path (folder 08) |
| `pts_level`, `pts_scan_index` | grid_refine stage (0–2) and scan of each point (folder 01) |
| `raster_xs`, `raster_ys`, `raster_zs` | the equivalent raster at the measured positions (1 in `raster_stride`; `raster_dt` s between kept samples), `raster_ch_*_vs` its PFM channels |
| `xs`, `ys`, `zs` | ground-truth / reference rasters (folder 00) |
| `binned_Z*`, `recon_*`, `raster_binned_Z*`, `raster_recon_Z*` | the maps saved at the end of the run (bounds: `recon_bounds`) |
| `modality`, `kinds`, `segment_lengths` | the mode, and every waveform's kind and sample count (of the original stream) |
| `priors_json`, `features_json`, `substrate_json`, `sensor_calibration_json`, `z_signal_json`, `pfm_json` | settings, priors, traced features, substrate model, calibrations (JSON strings) |
| `um_per_V`, `um_at_0V` | position-sensor calibration: µm = `um_at_0V` + `um_per_V` × V (the x sign carries the Asylum orientation: AR images are mirrored in x) |
| `z_unit` | nm (ZSensor height) or deg (PFM phase as the primary channel) |
| `sample_rate_hz`, `tip_velocity_V_s`, `gp_timing`, `run_wall_time_s`, `raster_*` scalars | acquisition and timing |

Instrument: Asylum Cypher, X/Y driven by PyScanner's FPGA waveforms (1 MS/s, 30 V/s on the
2026-11 data), X/Y position sensors on, Z = ZSensor. The saved stream of a full run already
keeps one block-mean sample per `store_every` at the instrument's rate for the long runs; the
lite stride is on top of that. Folder 02 (2026-10-14) is an older lite copy with block means
of 26 samples instead of every 26th sample.

## Use

```bash
python examples/quickstart.py                      # numpy + matplotlib only: every file, summary + quick-look image
```

```python
import sys; sys.path.insert(0, "examples")
import scout_datasets as sd                        # = selfdriving_afm/datasets.py (numpy only)
run = sd.load("data/05_gp_path_edge_focus/<file>.npz")
run["x"], run["y"], run["z"], run["raster"], run["channels"], run["priors"]
xu, yu = sd.to_um(run, run["x"], run["y"])        # micrometres
```

With the SCOUT package (https://doi.org/10.5281/zenodo.24482057, https://github.com/mchecanu/SCOUT-selfdriving_afm; `pip install ".[mcp]"`):

- `python examples/reconstruct_with_scout.py data/07_hybrids/<file>.npz` — levelled TV maps,
  run vs raster, in µm.
- The paper's figure scripts read these files, e.g.
  `python analysis/fig4_gp.py data/05_gp_path_edge_focus/<file>.npz --dense data/00_ground_truth/ground_truth_256lines_20261109_105402_lite.npz --out fig4`.
- From an LLM client through the MCP server: `list_scans(directory="<this folder>/data")`,
  `scan_summary`, `reconstruct_scan(method="tv", units="um")`, `animate_scan`, or
  `set_sample(dataset=".../ground_truth_256lines_..._lite.npz")` to **simulate any mode on the
  measured sample**.

## Citation and license

Please cite the paper (reference to be added) and this record. License: to be set at deposit
(CC BY 4.0 is the usual choice for data; subject to the ORNL release process).
