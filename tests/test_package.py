"""Smoke tests: no hardware, a few minutes on a laptop.  python -m pytest -q"""
import importlib
import sys

import numpy as np
import pytest

MODULES = ["config", "trajectory", "edge_detect", "execution", "modes", "priors", "reconstruction", "simulation",
           "topography", "gp_models", "gp_presets", "pfm", "sensors", "stepwise", "substrate", "user_raster",
           "reference_raster", "pyscanner_scans", "viz", "compare_animation", "datasets", "asylum",
           "backends", "planners"]


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(f"selfdriving_afm.{name}")


def test_version():
    import selfdriving_afm
    assert selfdriving_afm.__version__.count(".") == 2


def test_every_mode_registered_and_builds():
    from selfdriving_afm import modes
    from selfdriving_afm.config import ExperimentConfig
    from selfdriving_afm.backends.simulate_backend import SimulateBackend
    from selfdriving_afm.topography import sample_from_text
    expected = {"edge_follow", "edge_explorer", "domain_wall_explorer", "grid_refine", "gp_path",
                "gp_edge_explorer", "grid_tracker", "gp_spiral"}
    assert expected <= set(modes.MODES)
    smp = sample_from_text("field 4\n1 circle height=0.3 size=1.2 at=(0,0) edge=0.03\n")
    for m in modes.MODES:
        if m == "domain_wall_explorer":
            continue                      # needs a PFM backend (pfm.PFMSampleBackend)
        s = modes.build(m, ExperimentConfig(), SimulateBackend(smp, noise_std=0.0, seed=1), live=False)
        assert s.planner is not None


def test_gp_presets_reset_every_field():
    from selfdriving_afm import gp_presets
    from selfdriving_afm.config import ExperimentConfig
    names = list(gp_presets.GP_PRESETS)
    keys = {n: set(gp_presets.preset_fields(n)) for n in names}
    assert all(k == keys[names[0]] for k in keys.values())          # same fields in every preset
    cfg = ExperimentConfig()
    gp_presets.apply_gp_preset(cfg, "pure_gp_edge_focus")
    assert cfg.gp_path.ivr_interest_power == 2.0 and cfg.gp_path.ivr_lambda == 1.0
    gp_presets.apply_gp_preset(cfg, "pure_gp")
    assert cfg.gp_path.ivr_interest_power == 1.0                       # gamma does not leak


def _quick_sim(mode, tmp_path, preset=None, **ov):
    from selfdriving_afm import modes, simulation as sim, priors, gp_presets
    from selfdriving_afm.config import ExperimentConfig
    from selfdriving_afm.topography import sample_from_text
    smp = sample_from_text("field 3\nbase -6\n1 circle height=0.3 size=1.2 at=(0.8,0.5) edge=0.03\n"
                           "1 rectangle height=0.3 size=1.0x0.6 at=(-1,-1) edge=0.03\n")
    cfg = ExperimentConfig()
    cfg.readback.mode = "raw"
    # the package default (0.05 V/s at 50 kS/s = 1 uV per sample) is for slow instrument scans;
    # a test needs few samples: 0.5 mV per sample
    cfg.velocity.tip_velocity_V_s, cfg.velocity.sample_rate_hz = 10.0, 2.0e4
    priors.apply(cfg, priors.from_sample(smp))
    if preset:
        gp_presets.apply_gp_preset(cfg, preset)
    ev = sim.simulate(mode, cfg, smp, noise_std=0.0004, seed=1, overrides=ov)
    prefix = modes.save(ev["session"], tmp_path, prefix="t", save_png=False)
    return ev, f"{prefix}.npz"


def test_simulate_gp_path_and_lite_roundtrip(tmp_path):
    from selfdriving_afm import datasets
    ev, npz = _quick_sim("gp_path", tmp_path, preset="pure_gp_smooth", **{"gp_path.max_path_length_V": 3.0})
    assert ev["session"].executor.history
    full = datasets.load(npz)
    assert full["format"] == datasets.FORMAT_FULL and len(full["z"]) > 100
    lite = datasets.make_lite(npz, tmp_path / "t_lite.npz", stride=10)
    r = datasets.load(lite)
    assert r["format"] == datasets.FORMAT_LITE and r["stride"] == 10
    assert np.allclose(r["x"], full["x"][::10], atol=1e-5)
    assert datasets.summary(r)["samples_original"] == len(full["z"])


def test_simulate_edge_explorer(tmp_path):
    from selfdriving_afm import datasets
    ev, npz = _quick_sim("edge_explorer", tmp_path)
    assert len(ev["session"].executor.history) > 5
    feats = datasets.load(npz)["features"]
    assert feats is not None and len(feats) >= 1                      # it found and traced something


def test_reconstructions():
    from selfdriving_afm.config import LiveViewConfig
    from selfdriving_afm.reconstruction import bin_image, inpaint_binned, flattened_reconstruction
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (5000, 2))
    z = 0.1 * xy[:, 0] + (np.hypot(*xy.T) < 0.4)
    B, b = bin_image(xy, z, 50)
    for m in ("linear", "tv", "compressed_sensing", "fast"):
        R = inpaint_binned(B, b, LiveViewConfig(), mode=m)
        assert R.shape == (50, 50) and np.isfinite(R).all()
    cfg = LiveViewConfig(); cfg.n_image_pixels = 50; cfg.reconstruction = "tv"
    _, R, plane = flattened_reconstruction(xy, z, b, cfg)
    assert abs(plane[1] - 0.1) < 0.05                                   # the tilt is found


@pytest.mark.skipif(sys.version_info < (3, 10), reason="the mcp SDK needs Python 3.10+")
def test_mcp_tools_registered():
    pytest.importorskip("mcp")
    import asyncio
    from selfdriving_afm import mcp_server
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    for n in ("server_info", "simulate", "compare", "run_scan", "acquire_reference_raster", "stepwise_grid_refine",
              "scan_summary", "reconstruct_scan", "run_analysis", "set_sample", "apply_gp_preset"):
        assert n in names
    def hint(t):
        a = t.annotations
        return bool(a and (getattr(a, "destructive_hint", None) or getattr(a, "destructiveHint", None)))
    destructive = {t.name for t in tools if hint(t)}
    assert {"run_scan", "microscope_setup", "acquire_reference_raster", "withdraw_and_park",
            "stepwise_grid_refine", "probe_pyscanner_scans"} <= destructive
