"""
Synthetic and loaded sample topographies for simulation (Simulation.ipynb).

A "topography" is anything callable as z = topo(x, y) on arrays of bias-volt
coordinates -- exactly what backends.SimulateBackend takes -- so every scan mode runs
against it unchanged. Two ways to make one:

1. From a TEXT description (`sample_from_text`), one feature group per line:

       2 triangles  height=0.3 size=1.0
       1 square     height=0.5 size=0.8 at=(1.5, -1)
       1 rectangle  height=0.2 size=1.2x0.4 angle=30
       3 circles    height=0.4 size=0.6 edge=0.05
       1 step       height=0.2 at=(0, -2) angle=0

   - count first (optional, default 1; "2", "2x" or "2 x" all work), then the shape
     (plural or singular), then key=value settings in any order.
   - shapes: square, rectangle, triangle, circle (disk/cylinder), ellipse, ring,
     bump (Gaussian), pyramid, step (terrace edge), polygon.
   - keys (lengths in bias volts, the same units as every other length in the package;
     height in Z units, the same as AI0):
       height=  (default 0.3; negative = a pit/hole)
       size=    lateral size: one number (square side / circle diameter / equilateral
                triangle side / bump FWHM) or WxH (rectangle, ellipse, isosceles triangle
                base x height, ring outer diameter)
       at=(x, y)  centre; omitted -> placed at random inside the field without touching
                  other features (seeded, reproducible)
       angle=   rotation, degrees counter-clockwise (step: direction the ground RISES)
       edge=    sidewall width (0 = vertical step); rounds the feature's walls, like a
                blunt tip would
       width=   ring wall width (default 0.2 x size)
       points=(x1,y1),(x2,y2),...  polygon vertices, relative to `at`
   - global lines: "field 4" (half-width of the sample square, V), "base -5.7"
     (substrate level), "tilt 0.01 -0.02" (plane slope dz/dx dz/dy), "seed 3",
     "roughness 0.007 1 4" (smooth random substrate texture: RMS in Z units, then
     optionally its shortest and longest wavelength in V, default 1 and 4 -- the
     low-frequency waviness a real substrate has under the fine noise; seeded),
     "gap 0.3" (minimum clearance between randomly placed features).
   - '#' starts a comment.
   Features ADD up, so a square on top of a step is a stacked terrace.

2. From an Igor binary wave (`sample_from_ibw`): an Asylum Research .ibw file (the
   instrument's own raster images), one channel (e.g. "HeightRetrace"), mapped onto a
   bias-volt square you choose, optionally flattened and rescaled. `read_ibw` is a
   small numpy-only reader (IBW v5, what Asylum writes), so nothing extra needs
   installing on the instrument PC.

Both return a `Sample`: callable like a topography function, plus `.image(n)` for a
ground-truth image, `.bounds`, and (text samples) `.features` for per-feature scoring.
"""

from __future__ import annotations
import re
import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ====================================================================================
# Shapes -- each is a signed distance function (negative inside) in local coordinates,
# so rotation, soft sidewalls and "is this pixel on the edge" all come for free
# ====================================================================================
def _sd_box(u, v, w, h):
    qx, qy = np.abs(u) - 0.5 * w, np.abs(v) - 0.5 * h
    out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return out + np.minimum(np.maximum(qx, qy), 0)


def _sd_ellipse(u, v, a, b):
    # a, b = semi-axes. Scaled-circle approximation: exact for circles, within a few %
    # of the true distance near the boundary for moderate aspect ratios -- plenty here
    k = np.hypot(u / a, v / b)
    return (k - 1.0) * min(a, b)


def _sd_polygon(u, v, verts: np.ndarray):
    """Exact signed distance to a simple polygon (vectorized over points, loop over
    edges) -- the standard winding-parity formulation."""
    d = np.full(np.shape(u), np.inf)
    s = np.ones(np.shape(u))
    n = len(verts)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[i - 1]
        ex, ey = bx - ax, by - ay
        wx, wy = u - ax, v - ay
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        d = np.minimum(d, np.hypot(wx - ex * t, wy - ey * t))
        c1 = v >= ay
        c2 = v < by
        c3 = ex * wy > ey * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
    return s * d


def _profile(sd, edge):
    """0..1 height fraction from a signed distance: a vertical wall for edge=0,
    otherwise a smooth (cosine) wall `edge` wide, centred on the boundary."""
    if edge <= 0:
        return (sd <= 0).astype(float)
    t = np.clip(0.5 - sd / edge, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


SHAPES = ("square", "rectangle", "triangle", "circle", "ellipse", "ring", "bump", "pyramid", "step", "polygon")
_ALIASES = {"squares": "square", "rectangles": "rectangle", "rect": "rectangle", "rects": "rectangle",
            "triangles": "triangle", "tri": "triangle", "circles": "circle", "disk": "circle",
            "disks": "circle", "disc": "circle", "discs": "circle", "cylinder": "circle",
            "cylinders": "circle", "dot": "circle", "dots": "circle", "ellipses": "ellipse",
            "rings": "ring", "bumps": "bump", "gaussian": "bump", "gaussians": "bump",
            "pyramids": "pyramid", "steps": "step", "terrace": "step", "terraces": "step",
            "polygons": "polygon", "poly": "polygon"}


@dataclass
class Feature:
    shape: str
    height: float = 0.3
    size: Tuple[float, float] = (1.0, 1.0)   # lateral W x H (V)
    at: Optional[Tuple[float, float]] = None  # centre (V); None = placed at random
    angle_deg: float = 0.0
    edge: float = 0.0                         # sidewall width (V)
    width: Optional[float] = None             # ring wall width
    points: Optional[np.ndarray] = None       # polygon vertices, relative to `at`
    name: str = ""

    # --- geometry ---------------------------------------------------------------
    def radius(self) -> float:
        """Bounding-circle radius around `at` (for placement and fast evaluation)."""
        w, h = self.size
        if self.shape == "polygon" and self.points is not None:
            r = float(np.max(np.hypot(self.points[:, 0], self.points[:, 1])))
        elif self.shape == "bump":
            r = 1.5 * max(w, h)          # Gaussian: ~0 beyond 1.5 FWHM
        else:
            r = 0.5 * np.hypot(w, h)
        return r + self.edge

    def footprint(self) -> float:
        """Radius the feature visibly occupies -- used to keep random placements apart
        (a Gaussian bump's tail is excluded; radius() keeps it for evaluation)."""
        return 0.5 * max(self.size) + self.edge if self.shape == "bump" else self.radius()

    def _local(self, x, y):
        a = np.deg2rad(self.angle_deg)
        dx, dy = x - self.at[0], y - self.at[1]
        c, s = np.cos(a), np.sin(a)
        return c * dx + s * dy, -s * dx + c * dy

    def signed_distance(self, x, y):
        """Distance to this feature's edge (V), negative inside. `step`: distance to the
        step line (negative on the high side)."""
        u, v = self._local(np.asarray(x, float), np.asarray(y, float))
        w, h = self.size
        sh = self.shape
        if sh in ("square", "rectangle"):
            return _sd_box(u, v, w, h)
        if sh in ("circle", "ellipse", "bump"):
            return _sd_ellipse(u, v, 0.5 * w, 0.5 * h)
        if sh == "ring":
            t = self.width if self.width is not None else 0.2 * min(w, h)
            outer = _sd_ellipse(u, v, 0.5 * w, 0.5 * h)
            inner = _sd_ellipse(u, v, max(0.5 * w - t, 1e-9), max(0.5 * h - t, 1e-9))
            return np.maximum(outer, -inner)
        if sh == "triangle":
            return _sd_polygon(u, v, self._triangle())
        if sh == "pyramid":
            return _sd_box(u, v, w, h)
        if sh == "polygon":
            return _sd_polygon(u, v, self.points)
        if sh == "step":
            return -u
        raise ValueError(f"Unknown shape {sh!r} (known: {', '.join(SHAPES)})")

    def _triangle(self):
        w, h = self.size
        # isosceles, base w along x, apex +h; centred on its bounding box
        return np.array([[-0.5 * w, -0.5 * h], [0.5 * w, -0.5 * h], [0.0, 0.5 * h]])

    def z(self, x, y):
        """This feature's height contribution at (x, y)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if self.shape == "step":
            return self.height * _profile(self.signed_distance(x, y), self.edge)
        out = np.zeros(x.shape)
        r = self.radius()
        # only evaluate near the feature -- a run can be tens of millions of samples
        m = (np.abs(x - self.at[0]) <= r) & (np.abs(y - self.at[1]) <= r)
        if not m.any():
            return out
        xs, ys = x[m], y[m]
        if self.shape == "bump":
            u, v = self._local(xs, ys)
            w, h = self.size
            k = 4 * np.log(2)
            out[m] = self.height * np.exp(-k * ((u / w) ** 2 + (v / h) ** 2))
        elif self.shape == "pyramid":
            u, v = self._local(xs, ys)
            w, h = self.size
            out[m] = self.height * np.clip(1 - np.maximum(np.abs(u) / (0.5 * w), np.abs(v) / (0.5 * h)), 0, 1)
        else:
            out[m] = self.height * _profile(self.signed_distance(xs, ys), self.edge)
        return out

    def describe(self) -> str:
        w, h = self.size
        sz = f"{w:g}" if w == h else f"{w:g}x{h:g}"
        at = "random" if self.at is None else f"({self.at[0]:.3g}, {self.at[1]:.3g})"
        extra = f", angle {self.angle_deg:g} deg" if self.angle_deg else ""
        extra += f", edge {self.edge:g} V" if self.edge else ""
        return f"{self.name or self.shape}: height {self.height:g}, size {sz} V, at {at}{extra}"


# ====================================================================================
# Sample = sum of features + substrate (+ tilt), or an image
# ====================================================================================
class Sample:
    """z = sample(x, y). `bounds` = (xmin, xmax, ymin, ymax) of the defined area (V);
    outside it a text sample is just substrate, an image sample repeats its edge."""

    def __init__(self, fn: Callable, bounds, features: Sequence[Feature] = (), description: str = "",
                 z_label: str = "Z (AI0 units)"):
        self._fn = fn
        self.bounds = tuple(float(b) for b in bounds)
        self.features = list(features)
        self.description = description
        self.z_label = z_label

    def __call__(self, x, y):
        return self._fn(np.asarray(x, float), np.asarray(y, float))

    def grid(self, n: int = 300, bounds=None):
        b = bounds or self.bounds
        return np.meshgrid(np.linspace(b[0], b[1], n), np.linspace(b[2], b[3], n))

    def image(self, n: int = 300, bounds=None):
        """Ground-truth image [row = y, col = x] over `bounds` (default: the sample's)."""
        X, Y = self.grid(n, bounds)
        return self(X, Y)

    def plot(self, ax=None, n: int = 300, cmap: str = "copper", label_features: bool = True):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        im = ax.imshow(self.image(n), origin="lower", extent=self.bounds, cmap=cmap, interpolation="nearest")
        ax.figure.colorbar(im, ax=ax, label=self.z_label)
        if label_features:
            for i, f in enumerate(self.features):
                if f.shape != "step":
                    ax.annotate(str(i + 1), f.at, color="cyan", ha="center", va="center", fontsize=9,
                                fontweight="bold")
        ax.set_xlabel("X (V)")
        ax.set_ylabel("Y (V)")
        ax.set_title(self.description.splitlines()[0] if self.description else "Sample")
        ax.set_aspect("equal")
        return ax


def _texture(rms: float, lam_min: float, lam_max: float, seed: int, center, half: float,
             n_waves: int = 24) -> Callable:
    """Smooth random substrate texture: a sum of plane waves (random directions, phases,
    wavelengths between lam_min and lam_max), scaled to `rms` over the field."""
    rng = np.random.default_rng(seed + 7919)
    ang = rng.uniform(0, 2 * np.pi, n_waves)
    k = 2 * np.pi / rng.uniform(min(lam_min, lam_max), max(lam_min, lam_max), n_waves)
    ph = rng.uniform(0, 2 * np.pi, n_waves)
    amp = rng.uniform(0.5, 1.5, n_waves)
    kx, ky = k * np.cos(ang), k * np.sin(ang)

    def raw(x, y):
        return np.sum(amp * np.cos(kx * x[:, None] + ky * y[:, None] + ph), axis=1)

    X, Y = np.meshgrid(np.linspace(center[0] - half, center[0] + half, 150),
                       np.linspace(center[1] - half, center[1] + half, 150))
    g = raw(X.ravel(), Y.ravel())
    scale = rms / max(float(g.std()), 1e-12)
    mean = float(g.mean())

    def fn(x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        xf, yf = x.ravel(), y.ravel()
        out = np.empty(len(xf))
        for i in range(0, len(xf), 200_000):   # in chunks: waveforms are millions of samples
            out[i:i + 200_000] = scale * (raw(xf[i:i + 200_000], yf[i:i + 200_000]) - mean)
        return out.reshape(x.shape)
    return fn


def make_sample(features: Sequence[Feature], field_half_V: float = 4.0, base: float = 0.0,
                tilt: Tuple[float, float] = (0.0, 0.0), seed: int = 0, gap_V: float = 0.3,
                center: Tuple[float, float] = (0.0, 0.0),
                roughness: Optional[Tuple[float, float, float]] = None) -> Sample:
    """Place every feature without a position at random (seeded, not overlapping other
    features by less than gap_V, fully inside the field), and build the Sample."""
    rng = np.random.default_rng(seed)
    feats = [Feature(**{**f.__dict__}) for f in features]  # copies -- don't mutate the caller's
    cx, cy = center
    H = field_half_V
    placed = [f for f in feats if f.at is not None and f.shape != "step"]
    # biggest first -- small ones fit into the gaps left over, not the other way round
    for f in sorted(feats, key=lambda g: -g.footprint()):
        if f.at is not None:
            continue
        if f.shape == "step":
            f.at = (cx, cy)
            continue
        r = f.footprint()
        if r >= H:
            raise ValueError(f"{f.describe()} doesn't fit in a {2 * H:g} V field")
        for _ in range(5000):
            p = (cx + rng.uniform(-H + r, H - r), cy + rng.uniform(-H + r, H - r))
            if all(np.hypot(p[0] - g.at[0], p[1] - g.at[1]) >= r + g.footprint() + gap_V for g in placed):
                f.at = p
                placed.append(f)
                break
        else:
            raise ValueError(f"Couldn't place {f.describe()} -- the field is too crowded. Make the field "
                             f"bigger, the features smaller or fewer, lower `gap`, or give positions with at=(x, y).")
    tx, ty = tilt
    tex = _texture(*roughness, seed, center, H) if roughness and roughness[0] > 0 else None

    def fn(x, y):
        z = base + tx * (x - cx) + ty * (y - cy)
        z = np.broadcast_to(np.asarray(z, float), np.shape(x)).copy()
        for f in feats:
            z += f.z(x, y)
        if tex is not None:
            z += tex(x, y)
        return z

    desc = "\n".join(f"{i + 1}. {f.describe()}" for i, f in enumerate(feats))
    head = f"{len(feats)} feature(s) on a {2 * H:g} x {2 * H:g} V field"
    extra = [f"tilt {tx:g}, {ty:g} (dz/dx, dz/dy)"] if (tx, ty) != (0.0, 0.0) else []
    if tex is not None:
        extra.append(f"substrate texture {roughness[0]:g} RMS, wavelengths {roughness[1]:g}-{roughness[2]:g} V")
    if extra:
        head += " -- " + ", ".join(extra)
    return Sample(fn, (cx - H, cx + H, cy - H, cy + H), feats, head + "\n" + desc)


# ------------------------------------------------------------------------------------
# text description -> features
# ------------------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _parse_pair(s: str) -> Tuple[float, float]:
    nums = re.findall(_NUM, s)
    if len(nums) != 2:
        raise ValueError(f"expected two numbers in {s!r}")
    return float(nums[0]), float(nums[1])


def _parse_size(s: str) -> Tuple[float, float]:
    nums = re.findall(_NUM, s)
    if len(nums) == 1:
        return float(nums[0]), float(nums[0])
    if len(nums) == 2:
        return float(nums[0]), float(nums[1])
    raise ValueError(f"size must be one number or WxH, got {s!r}")


def parse_text(text: str):
    """Text description -> (features, settings dict). See the module docstring."""
    feats: List[Feature] = []
    settings: Dict[str, object] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            low = line.lower()
            head = low.split()[0]
            if head in ("field", "base", "tilt", "seed", "gap", "center", "centre", "roughness"):
                nums = [float(v) for v in re.findall(_NUM, line)]
                if head == "roughness":
                    settings["roughness"] = (nums[0], nums[1] if len(nums) > 1 else 1.0,
                                             nums[2] if len(nums) > 2 else 4.0)
                elif head == "tilt":
                    settings["tilt"] = (nums[0], nums[1] if len(nums) > 1 else 0.0)
                elif head in ("center", "centre"):
                    settings["center"] = (nums[0], nums[1])
                elif head == "seed":
                    settings["seed"] = int(nums[0])
                else:
                    settings[{"field": "field_half_V", "gap": "gap_V"}.get(head, head)] = nums[0]
                continue
            # keys: pull out key=value pairs first (values may contain spaces/commas)
            kv = {}
            # value = (a, b)[, (c, d) ...]  |  W x H  |  a plain token
            key_re = re.compile(r"(\w+)\s*=\s*(\([^()]*\)(?:\s*,\s*\([^()]*\))*|[^\s=]+(?:\s*[x×]\s*[^\s=]+)?)")
            for m in key_re.finditer(line):
                kv[m.group(1).lower()] = m.group(2)
            rest = key_re.sub(" ", line).split()
            count = 1
            if rest and re.fullmatch(r"\d+x?", rest[0].lower()):
                count = int(rest[0].lower().rstrip("x"))
                rest = rest[1:]
            if rest and rest[0].lower() in ("x", "×"):
                rest = rest[1:]
            if not rest:
                raise ValueError("no shape name")
            shape = _ALIASES.get(rest[0].lower(), rest[0].lower())
            if shape not in SHAPES:
                raise ValueError(f"unknown shape {rest[0]!r} (known: {', '.join(SHAPES)})")
            unknown = set(kv) - {"height", "h", "size", "at", "angle", "edge", "width", "points", "name",
                                 "diameter", "side"}
            if unknown:
                raise ValueError(f"unknown setting(s) {sorted(unknown)}")
            size_s = kv.get("size", kv.get("diameter", kv.get("side")))
            f = Feature(shape=shape,
                        height=float(kv.get("height", kv.get("h", 0.3))),
                        size=_parse_size(size_s) if size_s else (1.0, 1.0),
                        at=_parse_pair(kv["at"]) if "at" in kv else None,
                        angle_deg=float(kv.get("angle", 0.0)),
                        edge=float(kv.get("edge", 0.0)),
                        width=float(kv["width"]) if "width" in kv else None,
                        name=kv.get("name", ""))
            if shape == "triangle" and size_s and len(re.findall(_NUM, size_s)) == 1:
                s = f.size[0]
                f.size = (s, s * np.sqrt(3) / 2)   # one number = equilateral, side s
            if shape == "polygon":
                if "points" not in kv:
                    raise ValueError("polygon needs points=(x1,y1),(x2,y2),...")
                nums = [float(v) for v in re.findall(_NUM, kv["points"])]
                if len(nums) < 6 or len(nums) % 2:
                    raise ValueError("polygon needs at least 3 (x, y) points")
                pts = np.array(nums).reshape(-1, 2)
                mid = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
                f.points = pts - mid          # at=(x, y) is the centre of its bounding box
                f.size = tuple(float(v) for v in np.ptp(pts, axis=0))
            if count > 1 and f.at is not None:
                raise ValueError("a fixed position at=(x, y) only makes sense for one feature")
            for _ in range(count):
                feats.append(Feature(**{**f.__dict__}))
        except Exception as e:
            raise ValueError(f"Sample text, line {lineno} ({raw.strip()!r}): {e}") from None
    return feats, settings


def sample_from_text(text: str, **defaults) -> Sample:
    """Build a Sample from a text description (module docstring). Keyword arguments
    (field_half_V, base, tilt, seed, gap_V, center, roughness) are defaults the text can override."""
    feats, settings = parse_text(text)
    if not feats:
        raise ValueError("The sample text describes no features")
    return make_sample(feats, **{**defaults, **settings})


# ====================================================================================
# Images: any 2D array, or a channel of an Asylum .ibw file
# ====================================================================================
def image_sample(img: np.ndarray, bounds, description: str = "", z_label: str = "Z",
                 order: int = 1) -> Sample:
    """A Sample that interpolates an image [row = y, col = x] spanning `bounds`
    (bilinear by default, order=0 nearest). Outside the image it repeats the edge."""
    from scipy.ndimage import map_coordinates
    img = np.asarray(img, dtype=float)
    ny, nx = img.shape
    x0, x1, y0, y1 = bounds

    def fn(x, y):
        shape = np.shape(x)
        c = (np.ravel(x) - x0) / (x1 - x0) * (nx - 1)
        r = (np.ravel(y) - y0) / (y1 - y0) * (ny - 1)
        out = np.empty(c.size)
        for i in range(0, c.size, 2_000_000):  # chunked -- keeps the index arrays small
            out[i:i + 2_000_000] = map_coordinates(img, [r[i:i + 2_000_000], c[i:i + 2_000_000]],
                                                  order=order, mode="nearest")
        return out.reshape(shape)

    return Sample(fn, bounds, (), description, z_label)


# IBW wave data types -> numpy
_IBW_TYPES = {2: np.float32, 4: np.float64, 8: np.int8, 16: np.int16, 32: np.int32,
              8 | 64: np.uint8, 16 | 64: np.uint16, 32 | 64: np.uint32}


@dataclass
class IBWWave:
    name: str
    data: np.ndarray                    # as stored: axis 0 = x (fast), axis 1 = y, axis 2 = channel
    channels: List[str]                 # labels of the last dimension (Asylum: channel names)
    delta: Tuple[float, ...]            # sfA: step per point, per dimension
    offset: Tuple[float, ...]           # sfB
    units: str = ""
    dim_units: List[str] = field(default_factory=list)
    note: Dict[str, str] = field(default_factory=dict)

    def channel(self, name_or_index) -> np.ndarray:
        """One channel as an image [row = y, col = x] (ready for imshow(origin="lower"))."""
        if self.data.ndim == 2:
            return self.data.T.astype(float)
        if isinstance(name_or_index, str):
            names = [c.lower() for c in self.channels]
            if name_or_index.lower() not in names:
                raise KeyError(f"No channel {name_or_index!r} -- this file has {self.channels}")
            i = names.index(name_or_index.lower())
        else:
            i = int(name_or_index)
        return self.data[:, :, i].T.astype(float)

    def physical_size(self) -> Tuple[float, float]:
        """(width, height) of the image in its own units (Asylum: metres) -- first to
        last pixel centre, which is what Asylum's ScanSize note reports."""
        return abs(self.delta[0]) * (self.data.shape[0] - 1), abs(self.delta[1]) * (self.data.shape[1] - 1)


def read_ibw(path) -> IBWWave:
    """Minimal reader for Igor binary wave files, version 5 (Asylum Research AFM
    images) -- and v2/v3 1D/2D waves without labels. Numpy only."""
    b = open(path, "rb").read()
    endian = "<" if struct.unpack("<h", b[:2])[0] in (1, 2, 3, 5) else ">"
    version = struct.unpack(endian + "h", b[:2])[0]
    if version != 5:
        raise NotImplementedError(f"{path}: IBW version {version} -- only version 5 (Asylum) is supported")
    (_, _, wfm_size, formula_size, note_size, data_e_units_size) = struct.unpack(endian + "hhiiii", b[:20])
    dim_e_units_size = struct.unpack(endian + "4i", b[20:36])
    dim_labels_size = struct.unpack(endian + "4i", b[36:52])
    wh = b[64:64 + 320]
    npnts = struct.unpack(endian + "i", wh[12:16])[0]
    wtype = struct.unpack(endian + "h", wh[16:18])[0]
    name = wh[28:60].split(b"\0", 1)[0].decode("latin-1")
    n_dim = [d for d in struct.unpack(endian + "4i", wh[68:84])]
    sfA = struct.unpack(endian + "4d", wh[84:116])
    sfB = struct.unpack(endian + "4d", wh[116:148])
    units = wh[148:152].split(b"\0", 1)[0].decode("latin-1")
    dim_units = [wh[152 + 4 * i:156 + 4 * i].split(b"\0", 1)[0].decode("latin-1") for i in range(4)]
    if wtype not in _IBW_TYPES:
        raise NotImplementedError(f"{path}: unsupported IBW data type {wtype}")
    dt = np.dtype(_IBW_TYPES[wtype]).newbyteorder(endian)
    shape = [d for d in n_dim if d > 0] or [npnts]
    start = 64 + 320
    data = np.frombuffer(b, dtype=dt, count=npnts, offset=start).reshape(shape, order="F")
    pos = 64 + wfm_size + formula_size
    note_raw = b[pos:pos + note_size].decode("latin-1", errors="replace")
    pos += note_size + data_e_units_size
    if data_e_units_size:
        units = b[pos - data_e_units_size:pos].decode("latin-1")
    for i in range(4):
        if dim_e_units_size[i]:
            dim_units[i] = b[pos:pos + dim_e_units_size[i]].decode("latin-1")
        pos += dim_e_units_size[i]
    labels = []
    for i in range(4):
        blk = b[pos:pos + dim_labels_size[i]]
        pos += dim_labels_size[i]
        labels.append([blk[j:j + 32].split(b"\0", 1)[0].decode("latin-1") for j in range(0, len(blk), 32)])
    last = len(shape) - 1
    channels = labels[last][1:] if len(labels[last]) > 1 else [f"channel{i}" for i in range(shape[-1])]
    note = {}
    for ln in re.split(r"[\r\n]+", note_raw):
        if ":" in ln:
            k, v = ln.split(":", 1)
            note[k.strip()] = v.strip()
    return IBWWave(name, data, channels, tuple(sfA[:len(shape)]), tuple(sfB[:len(shape)]), units,
                   dim_units[:len(shape)], note)


def flatten(img: np.ndarray, method: str = "plane") -> np.ndarray:
    """AFM-style levelling: "none", "plane" (subtract the best-fit plane), "line"
    (subtract each scan line's median -- removes line-to-line offsets), or
    "plane+line"."""
    img = np.asarray(img, dtype=float)
    if method in ("none", None):
        return img
    out = img.copy()
    if "plane" in method:
        ny, nx = out.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        M = np.isfinite(out)
        A = np.column_stack([X[M], Y[M], np.ones(M.sum())])
        co, *_ = np.linalg.lstsq(A, out[M], rcond=None)
        out = out - (co[0] * X + co[1] * Y + co[2])
    if "line" in method:
        out = out - np.nanmedian(out, axis=1, keepdims=True)
    if method not in ("plane", "line", "plane+line"):
        raise ValueError(f"Unknown flatten method {method!r} (use 'none', 'plane', 'line', 'plane+line')")
    return out


def sample_from_ibw(path, channel="HeightRetrace", half_size_V: float = 4.0, center_V=(0.0, 0.0),
                    flatten_method: str = "plane+line", z_scale="auto", z_range: float = 1.0,
                    z_offset: float = 0.0, order: int = 1) -> Sample:
    """One channel of an .ibw image, laid out over a (2 half_size_V)^2 bias-volt square
    centred on center_V (the package works in open-loop volts, so the mapping from the
    file's metres is your choice). z: flattened, then either multiplied by z_scale (a
    number, e.g. 1e9 for metres -> nm) or, with z_scale="auto", rescaled so the image's
    1st-99th percentile range becomes z_range -- so thresholds tuned on text samples
    carry over. z_offset is added last (e.g. -5.7 for an AI0-like baseline)."""
    w = read_ibw(path)
    img = flatten(w.channel(channel), flatten_method)
    lo, hi = np.nanpercentile(img, [1, 99])
    if z_scale == "auto":
        s = z_range / max(hi - lo, 1e-30)
        img = (img - lo) * s
        zlab = f"{channel} (rescaled: 1-99% range = {z_range:g})"
    else:
        img = img * float(z_scale)
        s = float(z_scale)
        zlab = f"{channel} x {z_scale:g}"
    img = img + z_offset
    cx, cy = center_V
    bounds = (cx - half_size_V, cx + half_size_V, cy - half_size_V, cy + half_size_V)
    px, py = w.physical_size()
    desc = (f"{pathlib_name(path)} -- {channel} ({w.data.shape[0]} x {w.data.shape[1]} px, "
            f"{px * 1e6:.3g} x {py * 1e6:.3g} um{'' if w.dim_units[:1] in (['m'], ['']) else ' ?'}) "
            f"mapped onto {2 * half_size_V:g} x {2 * half_size_V:g} V\n"
            f"flatten: {flatten_method}; z: raw 1-99% range {hi - lo:.3g} {w.units} x {s:.3g}"
            f"{f' + {z_offset:g}' if z_offset else ''}\nchannels in file: {', '.join(w.channels)}")
    smp = image_sample(img, bounds, desc, zlab, order=order)
    smp.ibw = w
    return smp


def pathlib_name(p) -> str:
    import pathlib
    return pathlib.Path(p).name
