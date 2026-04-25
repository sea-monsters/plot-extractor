"""Generate 30 random test images per chart type with ground truth."""
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
SAMPLES_DIR = Path(__file__).parent.parent / "samples"

LINE_COLORS = [
    "blue", "red", "green", "orange", "purple",
    "brown", "teal", "navy", "darkgreen", "crimson",
    "dodgerblue", "darkorange", "forestgreen", "indigo",
]
SCATTER_COLORS = [
    "teal", "crimson", "navy", "darkgreen", "purple",
    "dodgerblue", "darkorange", "brown",
]
COLOR_PAIRS = [
    ("blue", "orange"), ("red", "green"), ("purple", "teal"),
    ("navy", "crimson"), ("dodgerblue", "darkorange"),
]
COLOR_TRIPLES = [
    ("blue", "red", "green"), ("orange", "purple", "teal"),
    ("navy", "crimson", "forestgreen"), ("dodgerblue", "darkorange", "indigo"),
]

SAMPLE_TO_TYPE = {
    "01_simple_linear": "simple_linear",
    "02_log_y": "log_y",
    "03_loglog": "loglog",
    "04_dual_y": "dual_y",
    "05_inverted_y": "inverted_y",
    "06_scatter": "scatter",
    "07_multi_series": "multi_series",
    "08_log_x": "log_x",
    "09_no_grid": "no_grid",
    "10_dense": "dense",
}


def _save_meta(out_dir: Path, name: str, data_dict: dict, axes: dict):
    meta = {"data": data_dict, "axes": axes}
    with open(out_dir / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _pick(rng: np.random.Generator, choices):
    return choices[rng.integers(len(choices))]


def copy_existing_samples():
    """Copy existing sample images into their type directories."""
    for sample_name, chart_type in SAMPLE_TO_TYPE.items():
        type_dir = TEST_DATA_DIR / chart_type
        type_dir.mkdir(parents=True, exist_ok=True)
        for ext in ["png"]:
            src = SAMPLES_DIR / f"{sample_name}.{ext}"
            if src.exists():
                dst = type_dir / f"000_original.{ext}"
                shutil.copy2(src, dst)
        src_meta = SAMPLES_DIR / f"{sample_name}_meta.json"
        if src_meta.exists():
            dst_meta = type_dir / "000_original_meta.json"
            shutil.copy2(src_meta, dst_meta)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_simple_linear(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(30, 81))
    x_min, x_max = float(rng.uniform(0, 5)), float(rng.uniform(10, 25))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 5))
    if func_type == 0:  # sine
        amp = float(rng.uniform(5, 20))
        freq = float(rng.uniform(0.5, 3))
        y = np.sin((x - x_min) * freq) * amp + amp + 5
    elif func_type == 1:  # cosine
        amp = float(rng.uniform(5, 20))
        freq = float(rng.uniform(0.5, 3))
        y = np.cos((x - x_min) * freq) * amp + amp + 5
    elif func_type == 2:  # linear
        slope = float(rng.uniform(-2, 4))
        y = slope * (x - x_min) + float(rng.uniform(10, 30))
    elif func_type == 3:  # quadratic
        a = float(rng.uniform(0.1, 1))
        y = a * (x - x_min) ** 2 + float(rng.uniform(0, 10))
    else:  # sqrt growth
        y = np.sqrt(x - x_min + 1) * float(rng.uniform(3, 10))

    y_min, y_max = float(np.floor(y.min()) - 2), float(np.ceil(y.max()) + 2)
    if y_min == y_max:
        y_max = y_min + 10

    color = _pick(rng, LINE_COLORS)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "simple_linear" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "simple_linear", name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_log_y(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(50, 121))
    x_min, x_max = float(rng.uniform(0, 5)), float(rng.uniform(20, 100))
    x = np.linspace(x_min, x_max, n_pts)

    rate = float(rng.uniform(0.02, 0.15))
    base_offset = float(rng.uniform(0, 5))
    y = np.exp(rate * (x - x_min)) + base_offset

    y_floor = max(0.1, float(y.min()) * 0.5)
    y_ceil = float(y.max()) * 2

    color = _pick(rng, LINE_COLORS)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.semilogy(x, y, color=color, linewidth=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_floor, y_ceil)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "log_y" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "log_y", name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "log", "min": y_floor, "max": y_ceil}})


def generate_loglog(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(50, 120))
    power = float(rng.uniform(0.5, 3.0))

    x_min_exp = float(rng.integers(0, 2))
    x_max_exp = float(rng.integers(3, 6))
    x = np.logspace(x_min_exp, x_max_exp, n_pts)
    y = x ** power

    y_floor = float(y.min()) * 0.5
    y_ceil = float(y.max()) * 2
    x_floor = float(x.min()) * 0.5
    x_ceil = float(x.max()) * 2

    color = _pick(rng, LINE_COLORS)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.loglog(x, y, color=color, linewidth=2)
    ax.set_xlim(x_floor, x_ceil)
    ax.set_ylim(y_floor, y_ceil)
    ax.set_xlabel("X (log)")
    ax.set_ylabel("Y (log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "loglog" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "loglog", name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "log", "min": x_floor, "max": x_ceil},
                "y": {"type": "log", "min": y_floor, "max": y_ceil}})


def generate_dual_y(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(30, 80))
    x_min, x_max = float(rng.uniform(0, 5)), float(rng.uniform(8, 20))
    x = np.linspace(x_min, x_max, n_pts)

    amp1 = float(rng.uniform(20, 80))
    freq1 = float(rng.uniform(0.5, 2))
    y1 = np.sin((x - x_min) * freq1) * amp1 + amp1 + 10
    y1_min, y1_max = 0, float(np.ceil(y1.max())) + 10

    amp2 = float(rng.uniform(1, 5))
    freq2 = float(rng.uniform(0.5, 3))
    y2 = np.cos((x - x_min) * freq2) * amp2 + amp2 + 1
    y2_min, y2_max = 0, float(np.ceil(y2.max())) + 2

    colors = _pick(rng, COLOR_PAIRS)
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
    ax1.plot(x, y1, color=colors[0], linewidth=2)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y1_min, y1_max)
    ax1.set_ylabel("Left Y", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=colors[1], linewidth=2)
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel("Right Y", color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    ax1.set_xlabel("X")
    ax1.set_title("Dual Y-Axis")
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "dual_y" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "dual_y", name,
               {"series1": {"x": x.tolist(), "y": y1.tolist()},
                "series2": {"x": x.tolist(), "y": y2.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y_left": {"type": "linear", "min": y1_min, "max": y1_max},
                "y_right": {"type": "linear", "min": y2_min, "max": y2_max}})


def generate_inverted_y(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(30, 80))
    x_min, x_max = float(rng.uniform(0, 5)), float(rng.uniform(8, 20))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 3))
    if func_type == 0:
        y = (x - x_min) ** 2
    elif func_type == 1:
        y = np.exp((x - x_min) * 0.3)
    else:
        y = np.sqrt(x - x_min + 1) * 10

    y_floor = float(np.ceil(y.max())) + 5
    y_ceil = 0

    color = _pick(rng, LINE_COLORS)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_floor, y_ceil)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (inverted)")
    ax.grid(True, linestyle="--", alpha=0.5)
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "inverted_y" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "inverted_y", name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": float(y_ceil), "max": float(y_floor), "inverted": True}})


def generate_scatter(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(20, 61))

    dist_type = int(rng.integers(0, 3))
    if dist_type == 0:  # uniform
        x_lo, x_hi = float(rng.uniform(0, 10)), float(rng.uniform(50, 150))
        y_lo, y_hi = float(rng.uniform(0, 10)), float(rng.uniform(30, 80))
        x = rng.uniform(x_lo, x_hi, n_pts)
        y = rng.uniform(y_lo, y_hi, n_pts)
    elif dist_type == 1:  # linear trend + noise
        x = rng.uniform(0, 100, n_pts)
        slope = float(rng.uniform(0.3, 2.0))
        y = slope * x + rng.normal(0, 10, n_pts)
    else:  # clustered
        centers = int(rng.integers(2, 5))
        x = np.concatenate([rng.normal(float(rng.uniform(20, 80)), 8, n_pts // centers) for _ in range(centers)])
        y = np.concatenate([rng.normal(float(rng.uniform(20, 60)), 6, n_pts // centers) for _ in range(centers)])
        n_pts = len(x)

    x_min, x_max = float(np.floor(x.min())) - 5, float(np.ceil(x.max())) + 5
    y_min, y_max = float(np.floor(y.min())) - 5, float(np.ceil(y.max())) + 5

    color = _pick(rng, SCATTER_COLORS)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.scatter(x, y, color=color, s=50 + int(rng.integers(0, 40)), edgecolors="black")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "scatter" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "scatter", name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_multi_series(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_series = int(rng.integers(2, 5))
    n_pts = int(rng.integers(40, 80))
    x_min, x_max = float(rng.uniform(0, 3)), float(rng.uniform(8, 20))
    x = np.linspace(x_min, x_max, n_pts)

    all_y = []
    data_dict = {}
    colors = _pick(rng, COLOR_TRIPLES) if n_series == 3 else [
        _pick(rng, LINE_COLORS) for _ in range(n_series)
    ]
    for s in range(n_series):
        amp = float(rng.uniform(3, 15))
        freq = float(rng.uniform(0.5, 3))
        phase = float(rng.uniform(0, 2 * np.pi))
        if s % 2 == 0:
            y = np.sin((x - x_min) * freq + phase) * amp
        else:
            y = np.cos((x - x_min) * freq + phase) * amp
        all_y.append(y)
        data_dict[f"series{s + 1}"] = {"x": x.tolist(), "y": y.tolist()}

    all_y_arr = np.concatenate(all_y)
    y_min = float(np.floor(all_y_arr.min())) - 5
    y_max = float(np.ceil(all_y_arr.max())) + 5

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    for s in range(n_series):
        c = colors[s] if s < len(colors) else _pick(rng, LINE_COLORS)
        ax.plot(x, all_y[s], color=c, linewidth=2, label=f"S{s + 1}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "multi_series" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "multi_series", name, data_dict,
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_log_x(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(50, 120))

    x_min_exp = float(rng.integers(0, 2))
    x_max_exp = float(rng.integers(3, 6))
    x = np.logspace(x_min_exp, x_max_exp, n_pts)

    func_type = int(rng.integers(0, 3))
    if func_type == 0:
        y = np.sqrt(x)
    elif func_type == 1:
        y = np.log(x + 1) * float(rng.uniform(5, 20))
    else:
        y = x ** float(rng.uniform(0.2, 0.6))

    x_floor = float(x.min()) * 0.5
    x_ceil = float(x.max()) * 2
    y_min = float(np.floor(y.min())) - 5
    y_max = float(np.ceil(y.max())) + 5

    color = _pick(rng, LINE_COLORS)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.semilogx(x, y, color=color, linewidth=2)
    ax.set_xlim(x_floor, x_ceil)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (log)")
    ax.set_ylabel("Y")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "log_x" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "log_x", name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "log", "min": x_floor, "max": x_ceil},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_no_grid(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(20, 60))
    x_min, x_max = float(rng.uniform(0, 3)), float(rng.uniform(5, 15))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 4))
    if func_type == 0:
        y = np.exp(-(x - x_min) * float(rng.uniform(0.2, 1.0))) * float(rng.uniform(10, 30))
    elif func_type == 1:
        y = np.sqrt(x - x_min + 1) * float(rng.uniform(3, 10))
    elif func_type == 2:
        y = np.log(x - x_min + 1) * float(rng.uniform(5, 15))
    else:
        y = (x - x_min) * float(rng.uniform(1, 5)) + float(rng.uniform(0, 5))

    y_min = float(np.floor(y.min())) - 2
    y_max = float(np.ceil(y.max())) + 2

    color = _pick(rng, LINE_COLORS)
    has_marker = bool(rng.integers(0, 2))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    if has_marker:
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4)
    else:
        ax.plot(x, y, color=color, linewidth=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # No grid
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "no_grid" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "no_grid", name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_dense(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(300, 801))
    x_min = 0
    x_max = float(rng.uniform(2, 6)) * np.pi
    x = np.linspace(x_min, x_max, n_pts)

    freq_a = float(rng.uniform(1, 5))
    freq_b = float(rng.uniform(5, 12))
    amp_a = float(rng.uniform(0.5, 1.5))
    amp_b = float(rng.uniform(0.2, 0.8))
    y = np.sin(freq_a * x) * amp_a + np.cos(freq_b * x) * amp_b

    y_min = float(np.floor(y.min())) - 1
    y_max = float(np.ceil(y.max())) + 1
    lw = float(rng.uniform(0.5, 1.5))

    color = _pick(rng, ["navy", "darkblue", "black", "darkgreen", "indigo", "brown"])
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(x, y, color=color, linewidth=lw)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)
    name = f"{idx + 1:03d}"
    fig.savefig(TEST_DATA_DIR / "dense" / f"{name}.png")
    plt.close(fig)
    _save_meta(TEST_DATA_DIR / "dense", name,
               {"series1": {"x": x.tolist(), "y": y.tolist()}},
               {"x": {"type": "linear", "min": x_min, "max": x_max},
                "y": {"type": "linear", "min": y_min, "max": y_max}})


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


def main():
    copy_existing_samples()
    for chart_type, gen_func in GENERATORS.items():
        type_dir = TEST_DATA_DIR / chart_type
        type_dir.mkdir(parents=True, exist_ok=True)
        for i in range(30):
            seed = hash((chart_type, i)) % (2**31)
            gen_func(i, seed)
        print(f"  {chart_type}: 30 images generated")
    print(f"Done. Test data in {TEST_DATA_DIR}")


if __name__ == "__main__":
    main()
