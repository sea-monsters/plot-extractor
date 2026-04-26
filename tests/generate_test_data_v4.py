"""Generate 500 realistic test images (set 4) with full metadata tags.

v4 moves beyond per-type batches by randomly mixing, per image:
  - Chart type:  expanded pool (bar, histogram, pie, area, step, box plot +
                 all original v1-v3 types)
  - Multi-plot:  1-4 subplots per image with independent chart types
  - Combo:       line+scatter, line+bar, line+area on shared axes
  - Distortions: rotation, perspective warp, defocus / motion blur, JPEG,
                 Gaussian / salt-&-pepper noise, resolution degradation,
                 sharpening artifacts, moiré, uneven lighting, line break
                 (fold crease), curl/fold, edge shadow, partial crop,
                 ink smudge, grayscale conversion

Every image's _meta.json includes a "tags" block with:
  - dataset: "v4"
  - chart_types:  list of chart types in the image
  - distortions:  list of applied distortions
  - subplot_layout:  (rows, cols) or null
  - chart_count:  number of subplots / visible chart panels
  - resolution:  (height, width) of saved image
  - is_grayscale:  bool
"""
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

TEST_DATA_V4_DIR = Path(__file__).parent.parent / "test_data_v4"
N_IMAGES = 500

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

FIG_SIZES = [(4, 3), (5, 3.5), (6, 4), (7, 5), (8, 5.5), (9, 6), (10, 7), (12, 8), (6, 6)]
DPIS = [80, 100, 120, 150, 200]
LINE_WIDTHS = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
GRID_STYLES = [
    {"linestyle": "--", "alpha": 0.5},
    {"linestyle": "-", "alpha": 0.3},
    {"linestyle": ":", "alpha": 0.6},
    {"linestyle": "-.", "alpha": 0.4},
]
BG_COLORS = ["white", "#f8f8f8", "#fffff0", "#f0f0f0", "#f5f5dc"]
MARKER_TYPES = ["o", "s", "^", "D", "x", "+", "*", "v"]
LINE_STYLES = ["-", "--", "-.", ":"]
LINE_COLORS = [
    "blue", "red", "green", "orange", "purple", "brown", "teal", "navy",
    "darkgreen", "crimson", "dodgerblue", "darkorange", "forestgreen", "indigo",
    "steelblue", "tomato", "seagreen", "sienna", "slateblue", "darkcyan",
    "olive", "maroon", "coral", "cadetblue", "darkviolet", "goldenrod",
]
SCATTER_COLORS = [
    "teal", "crimson", "navy", "darkgreen", "purple",
    "dodgerblue", "darkorange", "brown", "steelblue", "tomato", "seagreen", "indigo",
]
COLOR_PAIRS = [
    ("blue", "orange"), ("red", "green"), ("purple", "teal"),
    ("navy", "crimson"), ("dodgerblue", "darkorange"),
    ("steelblue", "tomato"), ("forestgreen", "sienna"),
    ("slateblue", "goldenrod"), ("darkcyan", "coral"),
]
COLOR_TRIPLES = [
    ("blue", "red", "green"), ("orange", "purple", "teal"),
    ("navy", "crimson", "forestgreen"), ("dodgerblue", "darkorange", "indigo"),
    ("steelblue", "tomato", "seagreen"), ("slateblue", "coral", "olive"),
]
COLOR_QUADS = [
    ("blue", "red", "green", "orange"),
    ("navy", "crimson", "forestgreen", "darkorange"),
    ("steelblue", "tomato", "seagreen", "goldenrod"),
    ("purple", "teal", "brown", "dodgerblue"),
]
ALL_CHART_TYPES = [
    "simple_linear", "log_y", "loglog", "dual_y", "inverted_y",
    "scatter", "multi_series", "log_x", "no_grid", "dense",
    "bar", "bar_h", "histogram", "area", "step", "pie",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick(rng: np.random.Generator, choices):
    return choices[int(rng.integers(len(choices)))]

def _pick_fig(rng: np.random.Generator):
    return _pick(rng, FIG_SIZES)

def _pick_dpi(rng: np.random.Generator):
    return _pick(rng, DPIS)

def _pick_lw(rng: np.random.Generator):
    return float(_pick(rng, LINE_WIDTHS))

def _pick_grid(rng: np.random.Generator):
    return _pick(rng, GRID_STYLES)

def _pick_bg(rng: np.random.Generator):
    return _pick(rng, BG_COLORS)

def _pick_marker(rng: np.random.Generator):
    return _pick(rng, MARKER_TYPES)

def _make_fig_ax(rng: np.random.Generator, **kwargs):
    figsize = _pick_fig(rng)
    dpi = _pick_dpi(rng)
    bg = _pick_bg(rng)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bg, **kwargs)
    ax.set_facecolor(bg)
    tick_size = int(rng.integers(7, 13))
    label_size = int(rng.integers(9, 15))
    ax.tick_params(labelsize=tick_size)
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(label_size)
    return fig, ax

# ---------------------------------------------------------------------------
# Chart generators — each returns (fig, ax, data_dict, axes_dict)
# ---------------------------------------------------------------------------

def _gen_simple_linear(rng, ax=None, return_fig=True):
    """Generate a simple linear chart. Returns (fig, ax, data, axes) or (data, axes)."""
    n_pts = int(rng.integers(25, 120))
    x_min = float(rng.uniform(-5, 10))
    x_max = float(x_min + rng.uniform(5, 40))
    x = np.linspace(x_min, x_max, n_pts)
    func_type = int(rng.integers(0, 8))
    if func_type == 0:
        amp, freq = float(rng.uniform(3, 30)), float(rng.uniform(0.3, 5))
        y = np.sin((x - x_min) * freq) * amp + amp + float(rng.uniform(0, 20))
    elif func_type == 1:
        amp, freq = float(rng.uniform(3, 30)), float(rng.uniform(0.3, 5))
        y = np.cos((x - x_min) * freq) * amp + amp + float(rng.uniform(0, 20))
    elif func_type == 2:
        y = float(rng.uniform(-5, 8)) * (x - x_min) + float(rng.uniform(0, 50))
    elif func_type == 3:
        y = float(rng.uniform(-1, 2)) * (x - x_min) ** 2 + float(rng.uniform(-10, 30))
    elif func_type == 4:
        y = np.sqrt(np.abs(x - x_min) + 1) * float(rng.uniform(2, 15))
    elif func_type == 5:
        y = float(rng.uniform(-0.05, 0.1)) * (x - x_min) ** 3 + float(rng.uniform(-10, 30))
    elif func_type == 6:
        y = float(rng.uniform(10, 40)) * np.tanh((x - (x_min + x_max) / 2) * 0.5)
    else:
        period = float(rng.uniform(2, 8))
        y = ((x - x_min) % period) / period * float(rng.uniform(5, 20))
    y_min = float(np.floor(y.min())) - float(rng.uniform(1, 5))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(1, 5))
    if y_min >= y_max: y_max = y_min + 10
    color, lw, ls = _pick(rng, LINE_COLORS), _pick_lw(rng), _pick_linestyle(rng)
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "linear", "min": y_min, "max": y_max}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_log_y(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(30, 150))
    x_min = float(rng.uniform(-5, 5)); x_max = float(x_min + rng.uniform(10, 100))
    x = np.linspace(x_min, x_max, n_pts)
    func_type = int(rng.integers(0, 5))
    if func_type == 0:
        y = np.exp(float(rng.uniform(0.01, 0.2)) * (x - x_min)) + float(rng.uniform(0, 3))
    elif func_type == 1:
        y = np.exp(float(rng.uniform(0.01, 0.1)) * (x - x_min)) + np.exp(float(rng.uniform(0.1, 0.3)) * (x - x_min)) * 0.1
    elif func_type == 2:
        y = (np.abs(x - x_min) + 0.1) ** float(rng.uniform(1.5, 4))
    elif func_type == 3:
        y = np.exp(float(rng.uniform(0.02, 0.15)) * (x - x_min)) * float(rng.uniform(1, 10)) + 1
    else:
        y = np.exp(0.05 * (x - x_min) ** 1.5) + 1
    y_floor = max(0.01, float(y.min()) * float(rng.uniform(0.3, 0.8)))
    y_ceil = float(y.max()) * float(rng.uniform(1.5, 5))
    color, lw = _pick(rng, LINE_COLORS), _pick_lw(rng)
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.semilogy(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_floor, y_ceil)
    ax.set_xlabel("X"); ax.set_ylabel("Y (log)")
    ax.grid(True, which="both", **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "log", "min": y_floor, "max": y_ceil}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_loglog(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(30, 150))
    power = float(rng.uniform(0.3, 4.0))
    x_min_exp = float(rng.integers(-1, 3)); x_max_exp = float(x_min_exp + rng.integers(2, 6))
    x = np.logspace(x_min_exp, x_max_exp, n_pts)
    func_type = int(rng.integers(0, 4))
    if func_type == 0: y = x ** power
    elif func_type == 1: y = float(rng.uniform(0.5, 5)) * x ** power
    elif func_type == 2: y = x ** power + float(rng.uniform(1, 10))
    else: y = x ** power + 0.1 * x ** (power * 0.5)
    y_floor = float(y.min()) * float(rng.uniform(0.3, 0.8))
    y_ceil = float(y.max()) * float(rng.uniform(1.5, 5))
    x_floor = float(x.min()) * float(rng.uniform(0.3, 0.8))
    x_ceil = float(x.max()) * float(rng.uniform(1.5, 5))
    color, lw = _pick(rng, LINE_COLORS), _pick_lw(rng)
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.loglog(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_floor, x_ceil); ax.set_ylim(y_floor, y_ceil)
    ax.set_xlabel("X (log)"); ax.set_ylabel("Y (log)")
    ax.grid(True, which="both", **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "log", "min": x_floor, "max": x_ceil},
            "y": {"type": "log", "min": y_floor, "max": y_ceil}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_dual_y(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(25, 100))
    x_min = float(rng.uniform(-5, 10)); x_max = float(x_min + rng.uniform(5, 30))
    x = np.linspace(x_min, x_max, n_pts)
    func_type = int(rng.integers(0, 4))
    if func_type == 0:
        amp1, freq1 = float(rng.uniform(50, 200)), float(rng.uniform(0.3, 3))
        y1 = np.sin((x - x_min) * freq1) * amp1 + amp1 + 10
        amp2, freq2 = float(rng.uniform(0.5, 5)), float(rng.uniform(0.5, 4))
        y2 = np.cos((x - x_min) * freq2) * amp2 + amp2 + 1
    elif func_type == 1:
        slope = float(rng.uniform(2, 10))
        y1 = slope * (x - x_min) + float(rng.uniform(0, 20))
        y2 = float(rng.uniform(0.01, 0.1)) * (x - x_min) ** 2 + float(rng.uniform(0, 2))
    elif func_type == 2:
        y1 = np.exp(float(rng.uniform(0.05, 0.15)) * (x - x_min)) * 10
        y2 = float(rng.uniform(1, 3)) * (x - x_min) + float(rng.uniform(0, 5))
    else:
        amp1 = float(rng.uniform(20, 80))
        y1 = np.sin((x - x_min) * 2) * amp1 + amp1 + 5
        y2 = np.log(np.abs(x - x_min) + 2) * float(rng.uniform(1, 5))
    y1_min = float(np.floor(y1.min())) - float(rng.uniform(1, 5))
    y1_max = float(np.ceil(y1.max())) + float(rng.uniform(1, 10))
    y2_min = float(np.floor(y2.min())) - float(rng.uniform(0.5, 2))
    y2_max = float(np.ceil(y2.max())) + float(rng.uniform(0.5, 3))
    colors = _pick(rng, COLOR_PAIRS)
    lw1, lw2 = _pick_lw(rng), _pick_lw(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax1 = _make_fig_ax(rng)
    else:
        ax1 = ax
    ax1.plot(x, y1, color=colors[0], linewidth=lw1)
    ax1.set_xlim(x_min, x_max); ax1.set_ylim(y1_min, y1_max)
    ax1.set_ylabel("Left Y", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=colors[1], linewidth=lw2, linestyle="--")
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel("Right Y", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])
    ax1.set_xlabel("X")
    data = {"series1": {"x": x.tolist(), "y": y1.tolist()},
            "series2": {"x": x.tolist(), "y": y2.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y_left": {"type": "linear", "min": y1_min, "max": y1_max},
            "y_right": {"type": "linear", "min": y2_min, "max": y2_max}}
    return ((fig, ax1, data, axes) if own_fig else (data, axes))

def _gen_inverted_y(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(25, 100))
    x_min = float(rng.uniform(-3, 5)); x_max = float(x_min + rng.uniform(5, 25))
    x = np.linspace(x_min, x_max, n_pts)
    func_type = int(rng.integers(0, 5))
    if func_type == 0: y = (x - x_min) ** 2 * float(rng.uniform(0.5, 3))
    elif func_type == 1: y = np.exp((x - x_min) * float(rng.uniform(0.1, 0.5)))
    elif func_type == 2: y = np.sqrt(x - x_min + 1) * float(rng.uniform(5, 20))
    elif func_type == 3: y = np.log(x - x_min + 2) * float(rng.uniform(5, 15))
    else: y = (x - x_min) * float(rng.uniform(2, 10))
    y_top = 0; y_bottom = float(np.ceil(y.max())) + float(rng.uniform(2, 10))
    color, lw, ls = _pick(rng, LINE_COLORS), _pick_lw(rng), _pick_linestyle(rng)
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel("X"); ax.set_ylabel("Y (inverted)"); ax.grid(True, **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "linear", "min": float(y_top), "max": float(y_bottom), "inverted": True}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_scatter(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(15, 80))
    dist_type = int(rng.integers(0, 5))
    if dist_type == 0:
        x_lo = float(rng.uniform(-20, 10))
        x_hi = float(x_lo + rng.uniform(20, 200))
        y_lo = float(rng.uniform(-10, 10))
        y_hi = float(y_lo + rng.uniform(20, 100))
        x = rng.uniform(x_lo, x_hi, n_pts)
        y = rng.uniform(y_lo, y_hi, n_pts)
    elif dist_type == 1:
        x = rng.uniform(-10, 100, n_pts)
        y = float(rng.uniform(0.2, 5)) * x + rng.normal(0, float(rng.uniform(3, 20)), n_pts)
    elif dist_type == 2:
        centers = int(rng.integers(2, 6)); pts_per = max(3, n_pts // centers)
        x = np.concatenate([rng.normal(float(rng.uniform(10, 90)), float(rng.uniform(5, 20)), pts_per) for _ in range(centers)])
        y = np.concatenate([rng.normal(float(rng.uniform(10, 80)), float(rng.uniform(3, 15)), pts_per) for _ in range(centers)])
    elif dist_type == 3:
        x = rng.uniform(-5, 15, n_pts)
        y = float(rng.uniform(0.5, 3)) * x ** 2 + rng.normal(0, float(rng.uniform(5, 15)), n_pts)
    else:
        theta = rng.uniform(0, 2 * np.pi, n_pts)
        r = float(rng.uniform(10, 40)) + rng.normal(0, float(rng.uniform(2, 8)), n_pts)
        x, y = r * np.cos(theta), r * np.sin(theta)
    x_min = float(np.floor(x.min())) - float(rng.uniform(2, 10))
    x_max = float(np.ceil(x.max())) + float(rng.uniform(2, 10))
    y_min = float(np.floor(y.min())) - float(rng.uniform(2, 10))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(2, 10))
    color, marker = _pick(rng, SCATTER_COLORS), _pick_marker(rng)
    marker_size = float(rng.uniform(15, 80))
    edge_width = float(rng.uniform(0.5, 2))
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.scatter(x, y, color=color, s=marker_size, marker=marker, edgecolors="black", linewidths=edge_width)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "linear", "min": y_min, "max": y_max}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_multi_series(rng, ax=None, return_fig=True):
    n_series = int(rng.integers(2, 6)); n_pts = int(rng.integers(30, 100))
    x_min = float(rng.uniform(-5, 5)); x_max = float(x_min + rng.uniform(5, 30))
    x = np.linspace(x_min, x_max, n_pts)
    if n_series <= 2: colors = list(_pick(rng, COLOR_PAIRS))
    elif n_series == 3: colors = list(_pick(rng, COLOR_TRIPLES))
    else: colors = list(_pick(rng, COLOR_QUADS))
    all_y, data_dict = [], {}
    for s in range(n_series):
        amp = float(rng.uniform(2, 20)); freq = float(rng.uniform(0.3, 4))
        phase = float(rng.uniform(0, 2 * np.pi))
        func_type = int(rng.integers(0, 4))
        if func_type == 0: y = np.sin((x - x_min) * freq + phase) * amp
        elif func_type == 1: y = np.cos((x - x_min) * freq + phase) * amp
        elif func_type == 2: y = float(rng.uniform(-3, 5)) * (x - x_min) + amp * np.sin(freq * (x - x_min) + phase) * 0.3
        else: y = amp * np.tanh((x - (x_min + x_max) / 2) * freq * 0.2) + phase
        all_y.append(y); data_dict[f"series{s + 1}"] = {"x": x.tolist(), "y": y.tolist()}
    all_y_arr = np.concatenate(all_y)
    y_min = float(np.floor(all_y_arr.min())) - float(rng.uniform(2, 8))
    y_max = float(np.ceil(all_y_arr.max())) + float(rng.uniform(2, 8))
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    for s in range(n_series):
        c = colors[s] if s < len(colors) else _pick(rng, LINE_COLORS)
        ax.plot(x, all_y[s], color=c, linewidth=_pick_lw(rng), linestyle=_pick_linestyle(rng), label=f"S{s + 1}")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    legend_loc = _pick(rng, ["best", "upper right", "upper left", "lower right", "lower left"])
    ax.legend(loc=legend_loc, fontsize=int(rng.integers(7, 11)))
    ax.grid(True, **_pick_grid(rng))
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "linear", "min": y_min, "max": y_max}}
    return ((fig, ax, data_dict, axes) if own_fig else (data_dict, axes))

def _gen_log_x(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(30, 150))
    x_min_exp = float(rng.integers(-2, 3)); x_max_exp = float(x_min_exp + rng.integers(2, 6))
    x = np.logspace(x_min_exp, x_max_exp, n_pts)
    func_type = int(rng.integers(0, 5))
    if func_type == 0: y = np.sqrt(x) * float(rng.uniform(2, 10))
    elif func_type == 1: y = np.log(x + 1) * float(rng.uniform(3, 20))
    elif func_type == 2: y = x ** float(rng.uniform(0.1, 0.8))
    elif func_type == 3: y = np.log(x) * float(rng.uniform(5, 15)) + float(rng.uniform(-10, 10))
    else: y = float(rng.uniform(20, 60)) / (1 + np.exp(-0.5 * (np.log10(x) - (x_min_exp + x_max_exp) / 2)))
    x_floor = float(x.min()) * float(rng.uniform(0.3, 0.8))
    x_ceil = float(x.max()) * float(rng.uniform(1.5, 5))
    y_min = float(np.floor(y.min())) - float(rng.uniform(2, 10))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(2, 10))
    color, lw = _pick(rng, LINE_COLORS), _pick_lw(rng)
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.semilogx(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_floor, x_ceil); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (log)"); ax.set_ylabel("Y")
    ax.grid(True, which="both", **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "log", "min": x_floor, "max": x_ceil},
            "y": {"type": "linear", "min": y_min, "max": y_max}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_no_grid(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(15, 80))
    x_min = float(rng.uniform(-5, 5)); x_max = float(x_min + rng.uniform(3, 25))
    x = np.linspace(x_min, x_max, n_pts)
    func_type = int(rng.integers(0, 6))
    if func_type == 0: y = np.exp(-(x - x_min) * float(rng.uniform(0.1, 1.5))) * float(rng.uniform(5, 40))
    elif func_type == 1: y = np.sqrt(x - x_min + 1) * float(rng.uniform(2, 15))
    elif func_type == 2: y = np.log(x - x_min + 1) * float(rng.uniform(3, 20))
    elif func_type == 3: y = (x - x_min) * float(rng.uniform(1, 8)) + float(rng.uniform(-5, 10))
    elif func_type == 4:
        center = (x_min + x_max) / 2
        y = float(rng.uniform(10, 40)) / (1 + np.exp(-(x - center) * float(rng.uniform(0.5, 3))))
    else: y = np.sin((x - x_min) * float(rng.uniform(0.5, 3))) * float(rng.uniform(3, 15)) + 15
    y_min = float(np.floor(y.min())) - float(rng.uniform(1, 5))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(1, 5))
    color, lw, ls = _pick(rng, LINE_COLORS), _pick_lw(rng), _pick_linestyle(rng)
    has_marker = bool(rng.integers(0, 2))
    marker = _pick_marker(rng) if has_marker else None
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    if marker:
        ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, marker=marker, markersize=float(rng.uniform(3, 8)))
    else:
        ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "linear", "min": y_min, "max": y_max}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_dense(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(200, 1200))
    x_min = 0; x_max = float(rng.uniform(1, 8)) * np.pi
    x = np.linspace(x_min, x_max, n_pts)
    func_type = int(rng.integers(0, 5))
    if func_type == 0:
        y = np.sin(float(rng.uniform(1, 6)) * x) * float(rng.uniform(0.5, 2)) + np.cos(float(rng.uniform(5, 15)) * x) * float(rng.uniform(0.1, 0.8))
    elif func_type == 1:
        y = np.sin(float(rng.uniform(2, 5)) * x) * float(rng.uniform(0.5, 1.5)) + np.cos(float(rng.uniform(8, 15)) * x) * float(rng.uniform(0.2, 0.6)) + np.sin(float(rng.uniform(20, 30)) * x) * float(rng.uniform(0.05, 0.2))
    elif func_type == 2:
        f0, f1 = float(rng.uniform(1, 3)), float(rng.uniform(10, 20))
        phase = 2 * np.pi * (f0 * x + (f1 - f0) / (2 * x_max) * x ** 2)
        y = np.sin(phase) * float(rng.uniform(0.5, 1.5))
    elif func_type == 3:
        y = np.sin(float(rng.uniform(10, 20)) * x) * np.sin(float(rng.uniform(0.5, 2)) * x) * float(rng.uniform(0.5, 1.5))
    else:
        y = np.sin(x * float(rng.uniform(5, 15))) + 0.3 * np.sin(x * float(rng.uniform(30, 50)))
    y_min = float(np.floor(y.min())) - float(rng.uniform(0.5, 2))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(0.5, 2))
    lw = float(rng.uniform(0.3, 2.0))
    color = _pick(rng, ["navy", "darkblue", "black", "darkgreen", "indigo", "brown", "steelblue", "darkslategray", "midnightblue"])
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.plot(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "linear", "min": y_min, "max": y_max}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

# -- New chart types for v4 --

def _gen_bar(rng, ax=None, return_fig=True):
    n_bars = int(rng.integers(5, 16))
    labels = [f"G{rng.integers(1, 8):d}" for _ in range(n_bars)]
    values = rng.uniform(5, 100, n_bars)
    color = _pick(rng, LINE_COLORS)
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.bar(labels, values, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", **grid_kw)
    ax.tick_params(axis="x", rotation=0)
    data = {"series1": {"x": list(range(n_bars)), "y": values.tolist(), "labels": labels}}
    axes = {"x": {"type": "category", "min": 0, "max": n_bars - 1},
            "y": {"type": "linear", "min": 0, "max": float(np.ceil(values.max())) + 5}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_bar_h(rng, ax=None, return_fig=True):
    n_bars = int(rng.integers(5, 16))
    labels = [f"G{rng.integers(1, 8):d}" for _ in range(n_bars)]
    values = rng.uniform(5, 100, n_bars)
    color = _pick(rng, LINE_COLORS)
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.barh(labels, values, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Value")
    ax.grid(True, axis="x", **grid_kw)
    data = {"series1": {"x": values.tolist(), "y": list(range(n_bars)), "labels": labels}}
    axes = {"x": {"type": "linear", "min": 0, "max": float(np.ceil(values.max())) + 5},
            "y": {"type": "category", "min": 0, "max": n_bars - 1}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_histogram(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(100, 500))
    dist_type = int(rng.integers(0, 4))
    if dist_type == 0: data = rng.normal(float(rng.uniform(20, 80)), float(rng.uniform(5, 20)), n_pts)
    elif dist_type == 1: data = rng.uniform(0, float(rng.uniform(50, 200)), n_pts)
    elif dist_type == 2: data = np.concatenate([rng.normal(float(rng.uniform(20, 40)), float(rng.uniform(3, 8)), n_pts // 2), rng.normal(float(rng.uniform(60, 80)), float(rng.uniform(3, 8)), n_pts - n_pts // 2)])
    else: data = rng.exponential(float(rng.uniform(10, 40)), n_pts)
    n_bins = int(rng.integers(10, 30))
    color = _pick(rng, LINE_COLORS)
    alpha = float(rng.uniform(0.6, 0.9))
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.hist(data, bins=n_bins, color=color, alpha=alpha, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", **grid_kw)
    data_dict = {"series1": {"values": data.tolist()}}
    axes = {"x": {"type": "linear"}, "y": {"type": "linear"}}
    return ((fig, ax, data_dict, axes) if own_fig else (data_dict, axes))

def _gen_area(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(30, 100))
    x_min = float(rng.uniform(0, 5)); x_max = float(x_min + rng.uniform(5, 30))
    x = np.linspace(x_min, x_max, n_pts)
    trend = (x - x_min) / (x_max - x_min) * float(rng.uniform(10, 50))
    noise = rng.normal(0, float(rng.uniform(1, 5)), n_pts)
    y = trend + noise + float(rng.uniform(5, 20))
    y = np.maximum(y, 0)
    y_min = 0; y_max = float(np.ceil(y.max())) + float(rng.uniform(2, 10))
    color = _pick(rng, LINE_COLORS)
    lw = _pick_lw(rng)
    alpha = float(rng.uniform(0.2, 0.4))
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.fill_between(x, y, 0, color=color, alpha=alpha)
    ax.plot(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "linear", "min": y_min, "max": y_max}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_step(rng, ax=None, return_fig=True):
    n_pts = int(rng.integers(6, 20))
    x_min = float(rng.uniform(0, 5)); x_max = float(x_min + rng.uniform(5, 25))
    x = np.sort(rng.uniform(x_min, x_max, n_pts))
    y = np.cumsum(rng.uniform(2, 15, n_pts)) + float(rng.uniform(5, 20))
    where = _pick(rng, ["pre", "post"])
    color, lw = _pick(rng, LINE_COLORS), _pick_lw(rng)
    grid_kw = _pick_grid(rng)
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.step(x, y, color=color, linewidth=lw, where=where)
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(float(np.floor(y.min())) - 5, float(np.ceil(y.max())) + 5)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, **grid_kw)
    data = {"series1": {"x": x.tolist(), "y": y.tolist()}}
    axes = {"x": {"type": "linear", "min": x_min, "max": x_max},
            "y": {"type": "linear", "min": y.min(), "max": y.max()}}
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _gen_pie(rng, ax=None, return_fig=True):
    n_slices = int(rng.integers(3, 8))
    values = np.abs(rng.normal(50, 20, n_slices))
    labels = [f"Cat{chr(65+i)}" for i in range(n_slices)]
    colors = [_pick(rng, LINE_COLORS) for _ in range(n_slices)]
    explode = [0] * n_slices
    if rng.random() < 0.3:
        explode[int(rng.integers(0, n_slices))] = float(rng.uniform(0.05, 0.2))
    own_fig = ax is None
    if own_fig:
        fig, ax = _make_fig_ax(rng)
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, explode=explode, shadow=bool(rng.integers(0, 2)))
    ax.axis("equal")
    data = {"series1": {"labels": labels, "values": values.tolist()}}
    axes = {}  # pie: no cartesian axes
    return ((fig, ax, data, axes) if own_fig else (data, axes))

def _pick_linestyle(rng):
    return _pick(rng, LINE_STYLES)

# ---------------------------------------------------------------------------
# Generator lookup
# ---------------------------------------------------------------------------

CHART_GENERATORS = {
    "simple_linear": _gen_simple_linear,
    "log_y": _gen_log_y,
    "loglog": _gen_loglog,
    "dual_y": _gen_dual_y,
    "inverted_y": _gen_inverted_y,
    "scatter": _gen_scatter,
    "multi_series": _gen_multi_series,
    "log_x": _gen_log_x,
    "no_grid": _gen_no_grid,
    "dense": _gen_dense,
    "bar": _gen_bar,
    "bar_h": _gen_bar_h,
    "histogram": _gen_histogram,
    "area": _gen_area,
    "step": _gen_step,
    "pie": _gen_pie,
}

# ---------------------------------------------------------------------------
# Distortion pipeline
# ---------------------------------------------------------------------------

def _rotation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(0.5, 15)) * (1 if rng.random() < 0.5 else -1)
    pil_img = Image.fromarray(img)
    rotated = pil_img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return np.array(rotated)

def _perspective_warp(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    margin = float(rng.uniform(0.05, 0.15))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.uniform(-margin, margin) * w, rng.uniform(-margin, margin) * h],
        [w + rng.uniform(-margin, margin) * w, rng.uniform(-margin, margin) * h],
        [w + rng.uniform(-margin, margin) * w, h + rng.uniform(-margin, margin) * h],
        [rng.uniform(-margin, margin) * w, h + rng.uniform(-margin, margin) * h],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def _gaussian_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(0.5, 3.0))
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

def _motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = int(rng.uniform(5, 30))
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size
    angle = rng.uniform(0, 180)
    center = (size / 2, size / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
    return cv2.filter2D(img, -1, kernel)

def _jpeg_compression(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(15, 85))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

def _gaussian_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(5, 30))
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _salt_pepper_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    amount = float(rng.uniform(0.005, 0.04))
    result = img.copy()
    n_salt = int(amount * h * w * 0.5)
    n_pepper = int(amount * h * w * 0.5)
    if n_salt > 0:
        sr, sc = rng.integers(0, h, n_salt), rng.integers(0, w, n_salt)
        result[sr, sc] = 255
    if n_pepper > 0:
        pr, pc = rng.integers(0, h, n_pepper), rng.integers(0, w, n_pepper)
        result[pr, pc] = 0
    return result

def _resolution_degradation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    scale = float(rng.uniform(0.1, 0.35))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    interp = _pick(rng, [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    return cv2.resize(small, (w, h), interpolation=interp)

def _sharpening_artifact(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    strength = float(rng.uniform(1.5, 4.0))
    kernel = np.array([[-1, -1, -1], [-1, 9 + strength, -1], [-1, -1, -1]]) / (1 + strength / 8)
    return cv2.filter2D(img, -1, kernel)

def _moire_pattern(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    freq_x = float(rng.uniform(1, 5))
    freq_y = float(rng.uniform(1, 5))
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pattern = 128 + 127 * np.sin(xx * freq_x * np.pi / 180 + yy * freq_y * np.pi / 180 * float(rng.uniform(0.5, 1.5)))
    pattern = np.stack([pattern] * 3, axis=-1)
    alpha = float(rng.uniform(0.05, 0.2))
    return np.clip(img.astype(np.float32) * (1 - alpha) + pattern * alpha, 0, 255).astype(np.uint8)

def _uneven_lighting(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    grad_type = int(rng.integers(0, 4))
    strength = float(rng.uniform(0.15, 0.5))
    if grad_type == 0:  # radial vignette
        cy, cx = h * float(rng.uniform(0.3, 0.7)), w * float(rng.uniform(0.3, 0.7))
        dist = np.sqrt(((xx - cx) / (w * 0.5)) ** 2 + ((yy - cy) / (h * 0.5)) ** 2)
        grad = 1 - strength * np.clip(dist, 0, 1)
    elif grad_type == 1:  # horizontal
        grad = 1 - strength * (xx / w)
    elif grad_type == 2:  # vertical
        grad = 1 - strength * (yy / h)
    else:  # diagonal
        grad = 1 - strength * ((xx + yy) / (w + h))
    grad_3ch = np.stack([grad] * 3, axis=-1)
    return np.clip(img.astype(np.float32) * grad_3ch, 0, 255).astype(np.uint8)

def _line_break(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a fold crease: white horizontal/vertical band across the image."""
    h, w = img.shape[:2]
    result = img.copy()
    is_horizontal = rng.random() < 0.5
    n_breaks = int(rng.integers(1, 4))
    for _ in range(n_breaks):
        width = int(rng.uniform(3, 20))
        if is_horizontal:
            y = int(rng.uniform(0.1, 0.9) * h)
            result[max(0, y - width):min(h, y + width), :] = 255
        else:
            x = int(rng.uniform(0.1, 0.9) * w)
            result[:, max(0, x - width):min(w, x + width)] = 255
    return result

def _curl_fold(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a dog-eared corner fold."""
    h, w = img.shape[:2]
    corner = int(rng.integers(0, 4))
    fold_size = int(rng.uniform(0.12, 0.35) * min(h, w))
    result = img.copy()
    pts = {
        0: np.array([[0, 0], [fold_size, 0], [0, fold_size]]),
        1: np.array([[w, 0], [w - fold_size, 0], [w, fold_size]]),
        2: np.array([[0, h], [fold_size, h], [0, h - fold_size]]),
        3: np.array([[w, h], [w - fold_size, h], [w, h - fold_size]]),
    }
    cv2.fillPoly(result, [pts[corner]], (255, 255, 255))
    return result

def _edge_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    result = img.astype(np.float32)
    side = int(rng.integers(0, 4))
    shadow_width = int(float(rng.uniform(0.05, 0.2)) * (h if side < 2 else w))
    darkness = float(rng.uniform(0.2, 0.6))
    for i in range(shadow_width):
        alpha = darkness * (1 - i / shadow_width)
        if side == 0: result[i, :, :] *= (1 - alpha)
        elif side == 1: result[h - 1 - i, :, :] *= (1 - alpha)
        elif side == 2: result[:, i, :] *= (1 - alpha)
        else: result[:, w - 1 - i, :] *= (1 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)

def _ink_smudge(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    result = img.astype(np.float32)
    n_blobs = int(rng.integers(1, 6))
    for _ in range(n_blobs):
        cx = int(rng.uniform(0.05, 0.95) * w)
        cy = int(rng.uniform(0.05, 0.95) * h)
        radius = int(rng.uniform(8, 50))
        alpha = float(rng.uniform(0.08, 0.3))
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
        result[mask] = result[mask] * (1 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)

def _grayscale_conversion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def _partial_crop(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Crop a random region (simulate partial document screenshot)."""
    h, w = img.shape[:2]
    ratio = float(rng.uniform(0.4, 0.9))
    new_h, new_w = int(h * ratio), int(w * ratio)
    x0 = int(rng.uniform(0, max(1, w - new_w)))
    y0 = int(rng.uniform(0, max(1, h - new_h)))
    return img[y0:y0 + new_h, x0:x0 + new_w]

# Registry: name → (function, probability_weight)
DISTORTIONS = [
    ("rotation", _rotation, 1.0),
    ("perspective_warp", _perspective_warp, 0.6),
    ("gaussian_blur", _gaussian_blur, 1.0),
    ("motion_blur", _motion_blur, 0.6),
    ("jpeg_compression", _jpeg_compression, 1.0),
    ("gaussian_noise", _gaussian_noise, 1.0),
    ("salt_pepper_noise", _salt_pepper_noise, 0.8),
    ("resolution_degradation", _resolution_degradation, 0.8),
    ("sharpening_artifact", _sharpening_artifact, 0.6),
    ("moire_pattern", _moire_pattern, 0.5),
    ("uneven_lighting", _uneven_lighting, 0.8),
    ("line_break", _line_break, 0.5),
    ("curl_fold", _curl_fold, 0.4),
    ("edge_shadow", _edge_shadow, 0.5),
    ("ink_smudge", _ink_smudge, 0.4),
    ("grayscale", _grayscale_conversion, 0.3),
    ("partial_crop", _partial_crop, 0.3),
]

# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------

def _fig_to_array(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))

def _save_meta(out_dir: Path, name: str, data_dict: dict, axes: dict, tags: dict):
    meta = {"data": data_dict, "axes": axes, "tags": tags}
    with open(out_dir / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def _apply_distortions(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Apply a random subset of distortions. Returns (processed_img, tags)."""
    applied: list[str] = []
    n_distort = int(rng.integers(0, 5))
    if n_distort == 0:
        return img, applied
    selected = rng.choice(len(DISTORTIONS), min(n_distort, len(DISTORTIONS)), replace=False)
    for idx in selected:
        name, func, weight = DISTORTIONS[idx]
        if rng.random() < weight:
            img = func(img, rng)
            applied.append(name)
    return img, applied

# ---------------------------------------------------------------------------
# Multi-plot generator
# ---------------------------------------------------------------------------

def _generate_multi_plot(rng: np.random.Generator) -> tuple[np.ndarray, list[str], dict]:
    """Generate an image with multiple subplots. Returns (image, chart_types, data_dict)."""
    n_plots = int(rng.integers(2, 5))
    if n_plots == 2:
        rows, cols = _pick(rng, [(1, 2), (2, 1)])
    elif n_plots == 3:
        rows, cols = 1, 3
    else:
        rows, cols = 2, 2

    chart_types: list[str] = []
    for _ in range(n_plots):
        ct = _pick(rng, [t for t in ALL_CHART_TYPES if t != "dual_y"])  # dual_y is special
        chart_types.append(ct)

    figsize = _pick_fig(rng)
    dpi = _pick_dpi(rng)
    bg = _pick_bg(rng)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, facecolor=bg)
    axes_flat = np.array(axes).flatten() if hasattr(axes, 'flatten') else np.array([axes]).flatten()

    all_data: dict[str, dict] = {}
    all_axes: dict[str, dict] = {}

    for i, (ct, ax) in enumerate(zip(chart_types, axes_flat)):
        gen_fn = CHART_GENERATORS.get(ct)
        if gen_fn is None:
            gen_fn = _gen_simple_linear
        ax.set_facecolor(bg)
        # Pass existing ax so generator draws into it
        result = gen_fn(rng, ax=ax, return_fig=False)
        if result:
            data, axes_info = result
            for k, v in data.items():
                all_data[f"sub{i+1}_{k}"] = v
            all_axes[f"sub{i+1}"] = axes_info

    fig.tight_layout()
    img = _fig_to_array(fig)
    return img, chart_types, all_data, all_axes

# ---------------------------------------------------------------------------
# Combo chart (line + scatter / line + bar on shared axes)
# ---------------------------------------------------------------------------

def _generate_combo(rng: np.random.Generator) -> tuple[np.ndarray, list[str], dict, dict]:
    """Generate a combo chart with two different representations on shared axes."""
    n_pts = int(rng.integers(25, 80))
    x_min = float(rng.uniform(0, 5)); x_max = float(x_min + rng.uniform(8, 30))
    x = np.linspace(x_min, x_max, n_pts)

    fig, ax = _make_fig_ax(rng)
    combo_type = int(rng.integers(0, 3))

    chart_types: list[str] = []
    all_data: dict[str, dict] = {}
    y_min_all, y_max_all = float('inf'), float('-inf')

    if combo_type == 0:  # line + scatter
        amp = float(rng.uniform(5, 30)); freq = float(rng.uniform(0.5, 3))
        y_line = np.sin((x - x_min) * freq) * amp + amp + 10
        y_scatter = y_line + rng.normal(0, float(rng.uniform(1, 5)), n_pts)
        ax.plot(x, y_line, color=_pick(rng, LINE_COLORS), linewidth=_pick_lw(rng), label="Line")
        ax.scatter(x, y_scatter, color=_pick(rng, SCATTER_COLORS), s=30, marker="o", label="Data", zorder=5)
        all_data = {"series1": {"x": x.tolist(), "y": y_line.tolist()},
                    "series2": {"x": x.tolist(), "y": y_scatter.tolist()}}
        chart_types = ["line", "scatter"]
        y_min_all = min(y_line.min(), y_scatter.min())
        y_max_all = max(y_line.max(), y_scatter.max())

    elif combo_type == 1:  # line + bar
        n_bars = int(rng.integers(6, 12))
        bar_x = np.linspace(x_min + 1, x_max - 1, n_bars)
        y_line = np.sin((x - x_min) * float(rng.uniform(0.5, 2))) * float(rng.uniform(5, 20)) + 20
        bar_vals = rng.uniform(5, 35, n_bars)
        ax.plot(x, y_line, color=_pick(rng, LINE_COLORS), linewidth=_pick_lw(rng), label="Trend")
        ax.bar(bar_x, bar_vals, width=(x_max - x_min) / n_bars * 0.6, color=_pick(rng, LINE_COLORS), alpha=0.6, label="Bars")
        all_data = {"series1": {"x": x.tolist(), "y": y_line.tolist()},
                    "series2": {"x": bar_x.tolist(), "y": bar_vals.tolist()}}
        chart_types = ["line", "bar"]
        y_min_all = min(y_line.min(), bar_vals.min())
        y_max_all = max(y_line.max(), bar_vals.max())

    else:  # line + area
        y_line = np.sin((x - x_min) * float(rng.uniform(0.5, 2))) * float(rng.uniform(5, 20)) + 20
        y_area = y_line * float(rng.uniform(0.3, 0.7)) + float(rng.uniform(5, 15))
        color = _pick(rng, LINE_COLORS)
        ax.plot(x, y_line, color=color, linewidth=_pick_lw(rng), label="Line")
        ax.fill_between(x, 0, y_area, color=color, alpha=0.25, label="Area")
        ax.plot(x, y_area, color=color, linewidth=0.5, alpha=0.5)
        all_data = {"series1": {"x": x.tolist(), "y": y_line.tolist()},
                    "series2": {"x": x.tolist(), "y": y_area.tolist()}}
        chart_types = ["line", "area"]
        y_min_all = min(y_line.min(), y_area.min(), 0)
        y_max_all = max(y_line.max(), y_area.max())

    ax.set_xlim(x_min, x_max)
    y_margin = float(rng.uniform(2, 5))
    ax.set_ylim(float(np.floor(y_min_all)) - y_margin, float(np.ceil(y_max_all)) + y_margin)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=int(rng.integers(7, 10)))
    ax.grid(True, **_pick_grid(rng))

    axes_info = {"x": {"type": "linear", "min": x_min, "max": x_max},
                 "y": {"type": "linear", "min": float(np.floor(y_min_all)) - y_margin, "max": float(np.ceil(y_max_all)) + y_margin}}
    img = _fig_to_array(fig)
    return img, chart_types, all_data, axes_info

# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def main():
    TEST_DATA_V4_DIR.mkdir(parents=True, exist_ok=True)

    # Decide per-image mode
    modes = (["single"] * 325) + (["combo"] * 50) + (["multi"] * 125)
    random.shuffle(modes)

    generated = 0
    for idx in range(N_IMAGES):
        seed = hash(("v4", idx)) % (2**31)
        rng = np.random.default_rng(seed)
        mode = modes[idx]

        name = f"{idx + 1:04d}"
        chart_types: list[str] = []
        distortions_tags: list[str] = []
        data_dict: dict = {}
        axes_dict: dict = {}

        if mode == "single":
            ct = _pick(rng, ALL_CHART_TYPES)
            chart_types = [ct]
            gen_fn = CHART_GENERATORS.get(ct, _gen_simple_linear)
            fig, _, dd, ad = gen_fn(rng)
            img = _fig_to_array(fig)
            data_dict, axes_dict = dd, ad

        elif mode == "combo":
            img, chart_types, data_dict, axes_dict = _generate_combo(rng)

        else:  # multi
            img, chart_types, data_dict, axes_dict = _generate_multi_plot(rng)

        _h_before = img.shape[0]
        img, distortions_tags = _apply_distortions(img, rng)

        # Build tags
        tags = {
            "dataset": "v4",
            "chart_types": chart_types,
            "distortions": distortions_tags,
            "chart_count": len(chart_types),
            "resolution": list(img.shape[:2]),
        }

        # Determine output directory
        # Group by primary chart type for organisation
        primary_type = chart_types[0] if chart_types else "unknown"
        out_dir = TEST_DATA_V4_DIR / primary_type
        out_dir.mkdir(parents=True, exist_ok=True)

        img_pil = Image.fromarray(img)
        img_pil.save(out_dir / f"{name}.png")
        _save_meta(out_dir, name, data_dict, axes_dict, tags)
        generated += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{N_IMAGES}] generated")

    print(f"Done. {generated} images in {TEST_DATA_V4_DIR}")


if __name__ == "__main__":
    main()
