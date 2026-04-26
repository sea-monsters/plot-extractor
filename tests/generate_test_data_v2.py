"""Generate 50 diverse test images per chart type (set 2) with ground truth.

Compared to set 1, this set introduces wider variation in:
- Figure sizes (4x3 to 12x8)
- DPI (80 to 150)
- Line widths (0.8 to 4.0)
- Axis label fonts, tick label sizes
- Marker types and sizes (o, s, ^, D, x, +)
- Grid styles (solid, dotted, varying alpha)
- Background colors (white, lightgray, ivory)
- Color palettes (wider range including pastels, darks)
- Data ranges (wider x/y spans, more extreme values)
- Function types (more mathematical functions per type)
- Line styles (solid, dashed, dash-dot)
- Axis tick formatting (scientific notation, custom formatters)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, EngFormatter

TEST_DATA_V2_DIR = Path(__file__).parent.parent / "test_data_v2"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FIG_SIZES = [(4, 3), (5, 3.5), (6, 4), (7, 5), (8, 5.5), (9, 6), (10, 7), (12, 8)]
DPIS = [80, 100, 120, 150]
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
    "blue", "red", "green", "orange", "purple",
    "brown", "teal", "navy", "darkgreen", "crimson",
    "dodgerblue", "darkorange", "forestgreen", "indigo",
    "steelblue", "tomato", "seagreen", "sienna",
    "slateblue", "darkcyan", "olive", "maroon",
    "coral", "cadetblue", "darkviolet", "goldenrod",
]
SCATTER_COLORS = [
    "teal", "crimson", "navy", "darkgreen", "purple",
    "dodgerblue", "darkorange", "brown",
    "steelblue", "tomato", "seagreen", "indigo",
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


def _save_meta(out_dir: Path, name: str, data_dict: dict, axes: dict):
    meta = {"data": data_dict, "axes": axes}
    with open(out_dir / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


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


def _pick_linestyle(rng: np.random.Generator):
    return _pick(rng, LINE_STYLES)


def _make_fig_ax(rng: np.random.Generator, **kwargs):
    figsize = _pick_fig(rng)
    dpi = _pick_dpi(rng)
    bg = _pick_bg(rng)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bg)
    ax.set_facecolor(bg)
    tick_size = int(rng.integers(7, 13))
    label_size = int(rng.integers(9, 15))
    ax.tick_params(labelsize=tick_size)
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(label_size)
    return fig, ax


# ---------------------------------------------------------------------------
# Generators — each type with expanded variation
# ---------------------------------------------------------------------------

def generate_simple_linear(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(25, 120))
    x_min = float(rng.uniform(-5, 10))
    x_max = float(x_min + rng.uniform(5, 40))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 8))
    if func_type == 0:  # sine
        amp = float(rng.uniform(3, 30))
        freq = float(rng.uniform(0.3, 5))
        y = np.sin((x - x_min) * freq) * amp + amp + float(rng.uniform(0, 20))
    elif func_type == 1:  # cosine
        amp = float(rng.uniform(3, 30))
        freq = float(rng.uniform(0.3, 5))
        y = np.cos((x - x_min) * freq) * amp + amp + float(rng.uniform(0, 20))
    elif func_type == 2:  # linear
        slope = float(rng.uniform(-5, 8))
        y = slope * (x - x_min) + float(rng.uniform(0, 50))
    elif func_type == 3:  # quadratic
        a = float(rng.uniform(-1, 2))
        y = a * (x - x_min) ** 2 + float(rng.uniform(-10, 30))
    elif func_type == 4:  # sqrt growth
        y = np.sqrt(np.abs(x - x_min) + 1) * float(rng.uniform(2, 15))
    elif func_type == 5:  # cubic
        a = float(rng.uniform(-0.05, 0.1))
        y = a * (x - x_min) ** 3 + float(rng.uniform(-10, 30))
    elif func_type == 6:  # tanh
        amp = float(rng.uniform(10, 40))
        y = amp * np.tanh((x - (x_min + x_max) / 2) * 0.5)
    else:  # sawtooth
        period = float(rng.uniform(2, 8))
        y = ((x - x_min) % period) / period * float(rng.uniform(5, 20))

    y_min = float(np.floor(y.min())) - float(rng.uniform(1, 5))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(1, 5))
    if y_min >= y_max:
        y_max = y_min + 10

    color = _pick(rng, LINE_COLORS)
    lw = _pick_lw(rng)
    ls = _pick_linestyle(rng)
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "simple_linear"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_log_y(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(30, 150))
    x_min = float(rng.uniform(-5, 5))
    x_max = float(x_min + rng.uniform(10, 100))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 5))
    if func_type == 0:  # exponential
        rate = float(rng.uniform(0.01, 0.2))
        y = np.exp(rate * (x - x_min)) + float(rng.uniform(0, 3))
    elif func_type == 1:  # double exponential
        rate1 = float(rng.uniform(0.01, 0.1))
        rate2 = float(rng.uniform(0.1, 0.3))
        y = np.exp(rate1 * (x - x_min)) + np.exp(rate2 * (x - x_min)) * 0.1
    elif func_type == 2:  # power law
        power = float(rng.uniform(1.5, 4))
        y = (np.abs(x - x_min) + 0.1) ** power
    elif func_type == 3:  # exponential with offset
        rate = float(rng.uniform(0.02, 0.15))
        y = np.exp(rate * (x - x_min)) * float(rng.uniform(1, 10)) + 1
    else:  # factorial-like
        y = np.exp(0.05 * (x - x_min) ** 1.5) + 1

    y_floor = max(0.01, float(y.min()) * float(rng.uniform(0.3, 0.8)))
    y_ceil = float(y.max()) * float(rng.uniform(1.5, 5))

    color = _pick(rng, LINE_COLORS)
    lw = _pick_lw(rng)
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)
    ax.semilogy(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_floor, y_ceil)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (log)")
    ax.grid(True, which="both", **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "log_y"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "log", "min": y_floor, "max": y_ceil}})


def generate_loglog(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(30, 150))
    power = float(rng.uniform(0.3, 4.0))

    x_min_exp = float(rng.integers(-1, 3))
    x_max_exp = float(x_min_exp + rng.integers(2, 6))
    x = np.logspace(x_min_exp, x_max_exp, n_pts)

    func_type = int(rng.integers(0, 4))
    if func_type == 0:  # pure power
        y = x ** power
    elif func_type == 1:  # power with multiplier
        y = float(rng.uniform(0.5, 5)) * x ** power
    elif func_type == 2:  # power + constant
        y = x ** power + float(rng.uniform(1, 10))
    else:  # fractional power
        y = x ** power + 0.1 * x ** (power * 0.5)

    y_floor = float(y.min()) * float(rng.uniform(0.3, 0.8))
    y_ceil = float(y.max()) * float(rng.uniform(1.5, 5))
    x_floor = float(x.min()) * float(rng.uniform(0.3, 0.8))
    x_ceil = float(x.max()) * float(rng.uniform(1.5, 5))

    color = _pick(rng, LINE_COLORS)
    lw = _pick_lw(rng)
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)
    ax.loglog(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_floor, x_ceil)
    ax.set_ylim(y_floor, y_ceil)
    ax.set_xlabel("X (log)")
    ax.set_ylabel("Y (log)")
    ax.grid(True, which="both", **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "loglog"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "log", "min": x_floor, "max": x_ceil},
                "y": {"type": "log", "min": y_floor, "max": y_ceil}})


def generate_dual_y(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(25, 100))
    x_min = float(rng.uniform(-5, 10))
    x_max = float(x_min + rng.uniform(5, 30))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 4))
    if func_type == 0:  # sin vs cos, very different scales
        amp1 = float(rng.uniform(50, 200))
        freq1 = float(rng.uniform(0.3, 3))
        y1 = np.sin((x - x_min) * freq1) * amp1 + amp1 + 10
        amp2 = float(rng.uniform(0.5, 5))
        freq2 = float(rng.uniform(0.5, 4))
        y2 = np.cos((x - x_min) * freq2) * amp2 + amp2 + 1
    elif func_type == 1:  # linear vs quadratic
        slope = float(rng.uniform(2, 10))
        y1 = slope * (x - x_min) + float(rng.uniform(0, 20))
        a = float(rng.uniform(0.01, 0.1))
        y2 = a * (x - x_min) ** 2 + float(rng.uniform(0, 2))
    elif func_type == 2:  # exp vs linear
        rate = float(rng.uniform(0.05, 0.15))
        y1 = np.exp(rate * (x - x_min)) * 10
        y2 = float(rng.uniform(1, 3)) * (x - x_min) + float(rng.uniform(0, 5))
    else:  # sin vs log
        amp1 = float(rng.uniform(20, 80))
        y1 = np.sin((x - x_min) * 2) * amp1 + amp1 + 5
        y2 = np.log(np.abs(x - x_min) + 2) * float(rng.uniform(1, 5))

    y1_min = float(np.floor(y1.min())) - float(rng.uniform(1, 5))
    y1_max = float(np.ceil(y1.max())) + float(rng.uniform(1, 10))
    y2_min = float(np.floor(y2.min())) - float(rng.uniform(0.5, 2))
    y2_max = float(np.ceil(y2.max())) + float(rng.uniform(0.5, 3))

    colors = _pick(rng, COLOR_PAIRS)
    lw1, lw2 = _pick_lw(rng), _pick_lw(rng)

    fig, ax1 = _make_fig_ax(rng)
    ax1.plot(x, y1, color=colors[0], linewidth=lw1)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y1_min, y1_max)
    ax1.set_ylabel("Left Y", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=colors[1], linewidth=lw2, linestyle="--")
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel("Right Y", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    ax1.set_xlabel("X")

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "dual_y"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y1.tolist()},
                "series2": {"x": x.tolist(), "y": y2.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y_left": {"type": "linear", "min": y1_min, "max": y1_max},
                "y_right": {"type": "linear", "min": y2_min, "max": y2_max}})


def generate_inverted_y(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(25, 100))
    x_min = float(rng.uniform(-3, 5))
    x_max = float(x_min + rng.uniform(5, 25))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 5))
    if func_type == 0:  # quadratic (depth profile)
        y = (x - x_min) ** 2 * float(rng.uniform(0.5, 3))
    elif func_type == 1:  # exponential decay
        y = np.exp((x - x_min) * float(rng.uniform(0.1, 0.5)))
    elif func_type == 2:  # sqrt
        y = np.sqrt(x - x_min + 1) * float(rng.uniform(5, 20))
    elif func_type == 3:  # log
        y = np.log(x - x_min + 2) * float(rng.uniform(5, 15))
    else:  # linear growth
        y = (x - x_min) * float(rng.uniform(2, 10))

    y_top = 0
    y_bottom = float(np.ceil(y.max())) + float(rng.uniform(2, 10))

    color = _pick(rng, LINE_COLORS)
    lw = _pick_lw(rng)
    ls = _pick_linestyle(rng)
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (inverted)")
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "inverted_y"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": float(y_top), "max": float(y_bottom), "inverted": True}})


def generate_scatter(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(15, 80))

    dist_type = int(rng.integers(0, 5))
    if dist_type == 0:  # uniform
        x_lo = float(rng.uniform(-20, 10))
        x_hi = float(x_lo + rng.uniform(20, 200))
        y_lo = float(rng.uniform(-10, 10))
        y_hi = float(y_lo + rng.uniform(20, 100))
        x = rng.uniform(x_lo, x_hi, n_pts)
        y = rng.uniform(y_lo, y_hi, n_pts)
    elif dist_type == 1:  # linear trend + noise
        x = rng.uniform(-10, 100, n_pts)
        slope = float(rng.uniform(0.2, 5))
        noise_std = float(rng.uniform(3, 20))
        y = slope * x + rng.normal(0, noise_std, n_pts)
    elif dist_type == 2:  # clustered
        centers = int(rng.integers(2, 6))
        pts_per = max(3, n_pts // centers)
        x = np.concatenate([rng.normal(float(rng.uniform(10, 90)), float(rng.uniform(5, 20)), pts_per) for _ in range(centers)])
        y = np.concatenate([rng.normal(float(rng.uniform(10, 80)), float(rng.uniform(3, 15)), pts_per) for _ in range(centers)])
        n_pts = len(x)
    elif dist_type == 3:  # quadratic trend + noise
        x = rng.uniform(-5, 15, n_pts)
        a = float(rng.uniform(0.5, 3))
        y = a * x ** 2 + rng.normal(0, float(rng.uniform(5, 15)), n_pts)
    else:  # circular / ring
        theta = rng.uniform(0, 2 * np.pi, n_pts)
        r = float(rng.uniform(10, 40)) + rng.normal(0, float(rng.uniform(2, 8)), n_pts)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    x_min = float(np.floor(x.min())) - float(rng.uniform(2, 10))
    x_max = float(np.ceil(x.max())) + float(rng.uniform(2, 10))
    y_min = float(np.floor(y.min())) - float(rng.uniform(2, 10))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(2, 10))

    color = _pick(rng, SCATTER_COLORS)
    marker = _pick_marker(rng)
    marker_size = float(rng.uniform(15, 80))
    edge_width = float(rng.uniform(0.5, 2))
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)
    ax.scatter(x, y, color=color, s=marker_size, marker=marker,
               edgecolors="black", linewidths=edge_width)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_multi_series(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_series = int(rng.integers(2, 6))
    n_pts = int(rng.integers(30, 100))
    x_min = float(rng.uniform(-5, 5))
    x_max = float(x_min + rng.uniform(5, 30))
    x = np.linspace(x_min, x_max, n_pts)

    if n_series <= 2:
        colors = list(_pick(rng, COLOR_PAIRS))
    elif n_series == 3:
        colors = list(_pick(rng, COLOR_TRIPLES))
    else:
        colors = list(_pick(rng, COLOR_QUADS))

    all_y = []
    data_dict = {}
    for s in range(n_series):
        func_type = int(rng.integers(0, 4))
        amp = float(rng.uniform(2, 20))
        freq = float(rng.uniform(0.3, 4))
        phase = float(rng.uniform(0, 2 * np.pi))
        if func_type == 0:
            y = np.sin((x - x_min) * freq + phase) * amp
        elif func_type == 1:
            y = np.cos((x - x_min) * freq + phase) * amp
        elif func_type == 2:
            slope = float(rng.uniform(-3, 5))
            y = slope * (x - x_min) + amp * np.sin(freq * (x - x_min) + phase) * 0.3
        else:
            y = amp * np.tanh((x - (x_min + x_max) / 2) * freq * 0.2) + phase

        all_y.append(y)
        data_dict[f"series{s + 1}"] = {"x": x.tolist(), "y": y.tolist()}

    all_y_arr = np.concatenate(all_y)
    y_min = float(np.floor(all_y_arr.min())) - float(rng.uniform(2, 8))
    y_max = float(np.ceil(all_y_arr.max())) + float(rng.uniform(2, 8))

    fig, ax = _make_fig_ax(rng)
    for s in range(n_series):
        c = colors[s] if s < len(colors) else _pick(rng, LINE_COLORS)
        lw = _pick_lw(rng)
        ls = _pick_linestyle(rng)
        ax.plot(x, all_y[s], color=c, linewidth=lw, linestyle=ls, label=f"S{s + 1}")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    legend_loc = _pick(rng, ["best", "upper right", "upper left", "lower right", "lower left"])
    ax.legend(loc=legend_loc, fontsize=int(rng.integers(7, 11)))

    grid_kw = _pick_grid(rng)
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "multi_series"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name, data_dict,
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_log_x(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(30, 150))

    x_min_exp = float(rng.integers(-2, 3))
    x_max_exp = float(x_min_exp + rng.integers(2, 6))
    x = np.logspace(x_min_exp, x_max_exp, n_pts)

    func_type = int(rng.integers(0, 5))
    if func_type == 0:  # sqrt
        y = np.sqrt(x) * float(rng.uniform(2, 10))
    elif func_type == 1:  # log
        y = np.log(x + 1) * float(rng.uniform(3, 20))
    elif func_type == 2:  # fractional power
        y = x ** float(rng.uniform(0.1, 0.8))
    elif func_type == 3:  # log with offset
        y = np.log(x) * float(rng.uniform(5, 15)) + float(rng.uniform(-10, 10))
    else:  # sigmoid on log x
        y = float(rng.uniform(20, 60)) / (1 + np.exp(-0.5 * (np.log10(x) - (x_min_exp + x_max_exp) / 2)))

    x_floor = float(x.min()) * float(rng.uniform(0.3, 0.8))
    x_ceil = float(x.max()) * float(rng.uniform(1.5, 5))
    y_min_val = float(np.floor(y.min())) - float(rng.uniform(2, 10))
    y_max_val = float(np.ceil(y.max())) + float(rng.uniform(2, 10))

    color = _pick(rng, LINE_COLORS)
    lw = _pick_lw(rng)
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)
    ax.semilogx(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_floor, x_ceil)
    ax.set_ylim(y_min_val, y_max_val)
    ax.set_xlabel("X (log)")
    ax.set_ylabel("Y")
    ax.grid(True, which="both", **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "log_x"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "log", "min": x_floor, "max": x_ceil},
                "y": {"type": "linear", "min": y_min_val, "max": y_max_val}})


def generate_no_grid(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(15, 80))
    x_min = float(rng.uniform(-5, 5))
    x_max = float(x_min + rng.uniform(3, 25))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 6))
    if func_type == 0:  # exponential decay
        y = np.exp(-(x - x_min) * float(rng.uniform(0.1, 1.5))) * float(rng.uniform(5, 40))
    elif func_type == 1:  # sqrt
        y = np.sqrt(x - x_min + 1) * float(rng.uniform(2, 15))
    elif func_type == 2:  # log
        y = np.log(x - x_min + 1) * float(rng.uniform(3, 20))
    elif func_type == 3:  # linear
        y = (x - x_min) * float(rng.uniform(1, 8)) + float(rng.uniform(-5, 10))
    elif func_type == 4:  # sigmoid
        center = (x_min + x_max) / 2
        y = float(rng.uniform(10, 40)) / (1 + np.exp(-(x - center) * float(rng.uniform(0.5, 3))))
    else:  # sine
        amp = float(rng.uniform(3, 15))
        freq = float(rng.uniform(0.5, 3))
        y = np.sin((x - x_min) * freq) * amp + amp

    y_min = float(np.floor(y.min())) - float(rng.uniform(1, 5))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(1, 5))

    color = _pick(rng, LINE_COLORS)
    lw = _pick_lw(rng)
    ls = _pick_linestyle(rng)
    has_marker = bool(rng.integers(0, 2))
    marker = _pick_marker(rng) if has_marker else None
    marker_size = float(rng.uniform(3, 8))

    fig, ax = _make_fig_ax(rng)
    if marker:
        ax.plot(x, y, color=color, linewidth=lw, linestyle=ls,
                marker=marker, markersize=marker_size)
    else:
        ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # No grid — that's the defining characteristic

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "no_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_dense(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(200, 1200))
    x_min = 0
    x_max = float(rng.uniform(1, 8)) * np.pi
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 5))
    if func_type == 0:  # dual frequency
        freq_a = float(rng.uniform(1, 6))
        freq_b = float(rng.uniform(5, 15))
        amp_a = float(rng.uniform(0.5, 2))
        amp_b = float(rng.uniform(0.1, 0.8))
        y = np.sin(freq_a * x) * amp_a + np.cos(freq_b * x) * amp_b
    elif func_type == 1:  # triple frequency
        y = (np.sin(float(rng.uniform(2, 5)) * x) * float(rng.uniform(0.5, 1.5))
             + np.cos(float(rng.uniform(8, 15)) * x) * float(rng.uniform(0.2, 0.6))
             + np.sin(float(rng.uniform(20, 30)) * x) * float(rng.uniform(0.05, 0.2)))
    elif func_type == 2:  # chirp
        f0 = float(rng.uniform(1, 3))
        f1 = float(rng.uniform(10, 20))
        phase = 2 * np.pi * (f0 * x + (f1 - f0) / (2 * x_max) * x ** 2)
        y = np.sin(phase) * float(rng.uniform(0.5, 1.5))
    elif func_type == 3:  # modulated sine
        carrier = np.sin(float(rng.uniform(10, 20)) * x)
        envelope = np.sin(float(rng.uniform(0.5, 2)) * x) * float(rng.uniform(0.5, 1.5))
        y = carrier * envelope
    else:  # noise-like
        y = np.sin(x * float(rng.uniform(5, 15))) + 0.3 * np.sin(x * float(rng.uniform(30, 50)))

    y_min = float(np.floor(y.min())) - float(rng.uniform(0.5, 2))
    y_max = float(np.ceil(y.max())) + float(rng.uniform(0.5, 2))
    lw = float(rng.uniform(0.3, 2.0))

    color = _pick(rng, ["navy", "darkblue", "black", "darkgreen", "indigo",
                         "brown", "steelblue", "darkslategray", "midnightblue"])
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)
    ax.plot(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V2_DIR / "dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    _save_meta(out_dir, name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "simple_linear": generate_simple_linear,
    "log_y": generate_log_y,
    "loglog": generate_loglog,
    "dual_y": generate_dual_y,
    "inverted_y": generate_inverted_y,
    "scatter": generate_scatter,
    "multi_series": generate_multi_series,
    "log_x": generate_log_x,
    "no_grid": generate_no_grid,
    "dense": generate_dense,
}

N_PER_TYPE = 50


def main():
    for chart_type, gen_func in GENERATORS.items():
        type_dir = TEST_DATA_V2_DIR / chart_type
        type_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N_PER_TYPE):
            seed = hash(("v2", chart_type, i)) % (2**31)
            gen_func(i, seed)
        print(f"  {chart_type}: {N_PER_TYPE} images generated")
    total = len(GENERATORS) * N_PER_TYPE
    print(f"Done. {total} images in {TEST_DATA_V2_DIR}")


if __name__ == "__main__":
    main()
