"""
Simulation backend: drop-in replacement for PyScannerBackend, backed by a synthetic
topography function so the planners can be developed/tested/tuned without hardware.

Same default shapes as your original notebooks (a cylinder + a square), but you can
pass any callable `topography_fn(x, y) -> z`.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional

from .base import ScanBackend


def default_topography(x, y):
    """Cylinder centered at (0.15, 0.15) radius 0.10 (V), square at (-0.20,-0.20)
    side 0.10 (V) -- same layout as your original notebook, just rescaled from nm
    to a plausible bias-volts range for a ~0.6 V field of view.
    """
    r = np.sqrt((x - 0.15) ** 2 + (y - 0.15) ** 2)
    cylinder = np.where(r <= 0.10, 0.10, 0.0)

    square = np.where((x >= -0.25) & (x <= -0.15) & (y >= -0.25) & (y <= -0.15), 0.15, 0.0)

    return np.maximum(cylinder, square)


class SimulateBackend(ScanBackend):
    # Z is measured exactly where the path is: no time shift on fast rasters (the
    # executor's starting estimate; readback.fast_raster_delay_s is the instrument's)
    fast_raster_delay_guess_s = 0.0

    def __init__(self, topography_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_topography,
                 noise_std: float = 0.0015, seed: Optional[int] = None):
        self.topography_fn = topography_fn
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self.current_pos = np.array([0.0, 0.0])
        # bookkeeping for visualization / analysis, mirrors what real acquisition would log
        self.history_x = []
        self.history_y = []
        self.history_z = []
        # PyScanner's AI_vs_time can come back SHORT at the end (35k-154k samples seen on
        # the instrument): (lo, hi) samples dropped from the end of every waveform's
        # record, drawn uniformly -- None = record everything (the default)
        self.ai_tail_loss_samples = None

    def measure_path(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        x_vec = np.asarray(x_vec, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float)
        z_vec = self.topography_fn(x_vec, y_vec)
        if self.noise_std > 0:
            z_vec = z_vec + self._rng.normal(0.0, self.noise_std, size=z_vec.shape)

        self.history_x.append(x_vec)
        self.history_y.append(y_vec)
        self.history_z.append(z_vec)
        if len(x_vec) > 0:
            self.current_pos = np.array([x_vec[-1], y_vec[-1]])
        if self.ai_tail_loss_samples:
            lo, hi = self.ai_tail_loss_samples
            k = int(self._rng.integers(int(lo), int(hi) + 1))
            z_vec = z_vec[:max(len(z_vec) - k, 0)]
        return z_vec

    def move_to(self, x: float, y: float, transition_time_s: float = 0.2) -> None:
        # transition_time_s is accepted (not used -- simulation has no transition
        # dynamics) purely so notebook code can call backend.move_to(...) identically
        # regardless of which backend is active, matching PyScannerBackend's signature.
        self.current_pos = np.array([float(x), float(y)])

    def read_ao_xy(self):
        """Where the (simulated) X/Y outputs are now."""
        return float(self.current_pos[0]), float(self.current_pos[1])

    def run_pyscanner_scan(self, scan, sample_rate_hz: float):
        """Simulated PyScanner-defined scan: the path pyscanner_scans.predict_path
        expects PyScanner to run, measured like any other path."""
        from ..pyscanner_scans import predict_path
        x, y = predict_path(scan, sample_rate_hz)
        return x, y, self.measure_path(x, y)

    def read_position_sensors(self, x_vec: np.ndarray, y_vec: np.ndarray):
        """Illustrative-only open-loop error model (fixed gain error + offset +
        the same noise floor as Z) -- just enough to exercise
        PositionSensorConfig in simulation, not a real piezo model. Real hardware
        gets its actual sensor readback from PyScannerBackend instead."""
        x_vec = np.asarray(x_vec, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float)
        gain_x, offset_x = 0.98, 0.0015
        gain_y, offset_y = 1.03, -0.0010
        x_sensor = x_vec * gain_x + offset_x + self._rng.normal(0.0, self.noise_std, size=x_vec.shape)
        y_sensor = y_vec * gain_y + offset_y + self._rng.normal(0.0, self.noise_std, size=y_vec.shape)
        return x_sensor, y_sensor

    # convenience for plotting
    def all_points(self):
        if not self.history_x:
            return np.empty((0, 3))
        x = np.concatenate(self.history_x)
        y = np.concatenate(self.history_y)
        z = np.concatenate(self.history_z)
        return np.column_stack([x, y, z])
