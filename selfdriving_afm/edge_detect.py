"""
Edge detection along a measured z-trace.

Two use cases:
  - `find_first_edge` / `find_edge_indices` walk a 1D trace (a search spiral, ordered
    by arc length) and return sample(s) that look like a step onto/off a feature.
  - `find_circle_crossings` walks a closed loop (a tracing circle, ordered by angle)
    and returns every index where the trace crosses the "on feature" / "off feature"
    threshold -- there can be 0, 2, or more crossings depending on how the circle
    sits relative to the boundary (e.g. straddling a corner).

Both support two threshold modes (see config.EdgeDetectConfig):
  - "manual":   a single user-supplied threshold, in the same units as z.
  - "auto_mad": threshold = k * MAD, with the baseline (median) and MAD estimated
                robustly from the trace itself -- no prior knowledge of feature
                height needed, useful for unknown samples.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple

from .config import EdgeDetectConfig


def robust_baseline(z: np.ndarray):
    """Median and MAD (scaled to be a consistent std-dev estimate), computed over
    the WHOLE trace rather than a leading window.

    Median/MAD have a 50% breakdown point, so as long as a MINORITY of the samples
    are "on feature" -- true for a search spiral seeded away from a known edge,
    since it only just barely reaches the edge by the time it triggers, and true for
    a tracing circle sitting on a boundary rather than deep inside a feature -- the
    estimate stays a good proxy for the substrate noise floor. Using the whole trace
    (hundreds-thousands of samples) instead of a small leading window also avoids the
    threshold coming out anomalously tight from an unlucky low-noise realization in
    just a handful of points.
    """
    median = np.nanmedian(z)
    mad = np.nanmedian(np.abs(z - median)) * 1.4826  # scale factor -> ~std dev for Gaussian noise
    return median, mad


def get_threshold(z: np.ndarray, cfg: EdgeDetectConfig):
    """Resolve a concrete (baseline, threshold) pair from the config + local data."""
    median, mad = robust_baseline(z)
    if cfg.mode == "manual":
        if cfg.manual_threshold is None:
            raise ValueError("EdgeDetectConfig.mode == 'manual' requires manual_threshold to be set")
        return median, cfg.manual_threshold
    elif cfg.mode == "auto_mad":
        mad = max(mad, 1e-12)  # avoid a degenerate zero-noise threshold
        return median, cfg.mad_k * mad
    else:
        raise ValueError(f"Unknown EdgeDetectConfig.mode: {cfg.mode}")


def start_level_threshold(z: np.ndarray, cfg: EdgeDetectConfig, frac: float = 0.02,
                          min_samples: int = 200):
    """(baseline, threshold) for a search spiral, from its FIRST samples only -- the
    tiny patch right around where the tip started (the first 2% of an Archimedean
    spiral's samples stay within ~14% of its radius), so a single level. Median and
    MAD over the whole spiral instead break when a big part of it runs over a
    feature: the MAD then measures the feature height, not the noise, and the
    threshold ends up nearly as tall as the feature (seen in simulation: 0.0875 V
    threshold on a 0.1 V step)."""
    n0 = min(len(z), max(min_samples, int(frac * len(z))))
    return get_threshold(z[:n0], cfg)


def find_edge_indices(z: np.ndarray, cfg: EdgeDetectConfig) -> np.ndarray:
    """All indices along an ordered 1D trace where |z - baseline| exceeds threshold,
    in trace order (i.e. in order of increasing arc length / spiral radius).
    Empty array if no edge is found anywhere in the trace.
    """
    baseline, threshold = get_threshold(z, cfg)
    deviation = np.abs(z - baseline)
    return np.nonzero(deviation > threshold)[0]


def find_first_edge(z: np.ndarray, cfg: EdgeDetectConfig) -> Optional[int]:
    """First index along an ordered 1D trace where |z - baseline| exceeds threshold.
    Returns None if no edge is found anywhere in the trace.
    """
    hits = find_edge_indices(z, cfg)
    if len(hits) == 0:
        return None
    return int(hits[0])


def _runs(mask: np.ndarray) -> List[list]:
    """Consecutive runs of equal value in a 1D bool array: [start, end_inclusive, value]."""
    if len(mask) == 0:
        return []
    idx = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.concatenate(([0], idx))
    ends = np.concatenate((idx - 1, [len(mask) - 1]))
    return [[int(s), int(e), bool(mask[s])] for s, e in zip(starts, ends)]


def clean_mask(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Remove noise flicker from an on/off-feature mask: interior runs shorter than
    `min_run` samples are flipped to their neighbours' state (short spikes first,
    then short dropouts), so a single noisy sample can't register as two edges."""
    if min_run <= 1 or len(mask) == 0:
        return mask
    m = mask.copy()
    for value in (True, False):
        for s, e, v in _runs(m):
            if v == value and (e - s + 1) < min_run and s > 0 and e < len(m) - 1:
                m[s:e + 1] = not value
    return m


def transitions(mask: np.ndarray) -> np.ndarray:
    """Indices where an ordered trace changes state (index of the first sample of
    the new state) -- i.e. the actual edges along a spiral, as opposed to every
    sample that happens to sit on a feature plateau."""
    return np.flatnonzero(mask[1:] != mask[:-1]) + 1


def on_arcs(mask: np.ndarray, closed: bool) -> List[Tuple[int, int]]:
    """(start, end_inclusive) index of every on-feature run. With closed=True (a
    full tracing circle) a run touching both ends is merged across the wraparound,
    giving start > end for that arc."""
    runs = _runs(mask)
    if closed and len(runs) > 1 and runs[0][2] == runs[-1][2]:
        runs[0][0] = runs[-1][0]
        runs.pop()
    return [(s, e) for s, e, v in runs if v]


def find_circle_crossings(z: np.ndarray, cfg: EdgeDetectConfig, wrap: bool = True) -> List[int]:
    """Indices around a closed loop where the trace crosses the on/off-feature
    threshold. `wrap=True` also checks the boundary between the last and first
    sample (since a circle is a closed loop).
    """
    baseline, threshold = get_threshold(z, cfg)
    above = np.abs(z - baseline) > threshold

    if not above.any() or above.all():
        # entirely off-feature or entirely on-feature -> no boundary crossing on this circle
        return []

    crossings = []
    n = len(above)
    limit = n if wrap else n - 1
    for i in range(limit):
        j = (i + 1) % n
        if above[i] != above[j]:
            crossings.append(j)
    return crossings


def hysteresis_mask(h: np.ndarray, lo: float, hi: float, initial: Optional[bool] = None) -> np.ndarray:
    """Two-threshold ("Schmitt trigger") on/off-feature mask along an ordered trace of
    |height above the substrate|: switches ON only above `hi`, back OFF only below
    `lo` -- in between it keeps its state. A path running along a sloping sidewall,
    where the height hovers right at a single threshold, gives one edge instead of a
    burst of flickering ones (seen on the instrument: 3-18 crossings within one
    segment). `initial`: the state before the first sample (e.g. where the previous,
    continuous segment ended); None = decided by the midpoint of lo and hi."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if initial is None:
        initial = bool(h[0] > 0.5 * (lo + hi))
    mark = np.where(h > hi, 1, np.where(h < lo, -1, 0))
    idx = np.where(mark != 0, np.arange(n), -1)
    last = np.maximum.accumulate(idx)   # index of the latest decisive sample so far
    state = np.where(last >= 0, mark[np.maximum(last, 0)] > 0, initial)
    return state.astype(bool)


def refine_crossing(h: np.ndarray, i: int, level: float, entering: bool, max_back: int) -> int:
    """With a two-threshold mask a crossing registers where the height passes the FAR
    threshold -- past the edge. Move it back to where the height passes `level` (the
    single threshold, mid-way), searching at most `max_back` samples back from i."""
    j0 = max(0, i - max_back)
    seg = h[j0:i + 1]
    beyond = seg <= level if entering else seg >= level   # still on the old side of `level`
    k = np.flatnonzero(beyond)
    return int(j0 + k[-1] + 1) if len(k) else int(i)
