"""
Backend interface.

A "backend" is anything that can take an (x_vec, y_vec) trajectory (already-clamped,
in bias volts) and return a z_vec of the same length -- the measured height at each
sample. That's the ONLY thing the planners depend on, which is what lets the exact
same planner code run against:
  - `simulate_backend.SimulateBackend`  (synthetic topography, no hardware needed)
  - `pyscanner_backend.PyScannerBackend` (the real instrument)

Both readback strategies you asked for ("full binned image" vs "raw per-sample
AI_vs_time") are implemented INSIDE PyScannerBackend and selected via
config.ReadbackConfig.mode -- the planner never needs to know which one is active.
"""

from __future__ import annotations
from typing import Protocol, Tuple
import numpy as np


class ScanBackend(Protocol):
    def measure_path(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """Execute the trajectory (x_vec, y_vec) and return z_vec, same length."""
        ...

    def move_to(self, x: float, y: float, transition_time_s: float = 0.2) -> None:
        """Move the tip to a single point without acquiring data (repositioning),
        smoothly over transition_time_s. Bypasses ContinuousExecutor entirely -- this
        is PyScanner's native line-scan move mechanism (line_scan_control_cluster /
        do_probe_move), not a data-acquiring scan primitive, so it isn't recorded in
        ContinuousExecutor.history and doesn't go through the live view."""
        ...

    def read_position_sensors(self, x_vec: np.ndarray, y_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optional: read back the actual (sensor) X/Y position for the segment
        `measure_path` just executed, same length as x_vec/y_vec -- for correcting
        open-loop discrepancies (config.PositionSensorConfig). Only called by
        ContinuousExecutor when that config's `enabled` is True; a backend that
        doesn't support it should raise NotImplementedError (the default here),
        which is exactly what happens if you enable the option against a backend
        that doesn't override this."""
        raise NotImplementedError(f"{type(self).__name__} does not support position-sensor readback")
