"""Generate 50 hard-mode test images per chart type (set 3) with ground truth.

Compared to v1/v2, this set simulates real-world scanned/photographed charts:
- Axis rotation (0.5-3 deg, simulating skewed scans)
- Gaussian noise, salt & pepper noise
- JPEG compression artifacts
- Slight blur (scanner/camera out of focus)
- Combo chart types (line+scatter, line+errorbar, etc.)
- Scanning artifacts (watermarks, paper texture, edge shadow)
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

TEST_DATA_V3_DIR = Path(__file__).parent.parent / "test_data_v3"

# ---------------------------------------------------------------------------
# Constants (same diversity as v2)
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


# ---------------------------------------------------------------------------
# Shared helpers (same as v2)
# ---------------------------------------------------------------------------

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


def _fig_to_array(fig):
    """Convert matplotlib figure to RGB numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Post-processing pipeline (scanning simulation)
# ---------------------------------------------------------------------------

def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees with white fill."""
    pil_img = Image.fromarray(img)
    rotated = pil_img.rotate(angle, expand=True, resample=Image.BICUBIC,
                             fillcolor=(255, 255, 255))
    return np.array(rotated)


def _add_gaussian_noise(img: np.ndarray, rng: np.random.Generator,
                        sigma: float) -> np.ndarray:
    """Add Gaussian noise."""
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _add_salt_pepper(img: np.ndarray, rng: np.random.Generator,
                     amount: float) -> np.ndarray:
    """Add salt and pepper noise."""
    result = img.copy()
    h, w = img.shape[:2]
    n_salt = int(amount * h * w * 0.5)
    n_pepper = int(amount * h * w * 0.5)

    salt_rows = rng.integers(0, h, n_salt)
    salt_cols = rng.integers(0, w, n_salt)
    result[salt_rows, salt_cols] = 255

    pepper_rows = rng.integers(0, h, n_pepper)
    pepper_cols = rng.integers(0, w, n_pepper)
    result[pepper_rows, pepper_cols] = 0

    return result


def _jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression artifacts."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", img_bgr, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def _add_watermark(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add diagonal semi-transparent watermark text."""
    pil_img = Image.fromarray(img)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text = _pick(rng, ["CONFIDENTIAL", "DRAFT", "SAMPLE", "COPY"])
    h, w = img.shape[:2]
    font_size = max(16, int(min(h, w) * 0.08))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    alpha = int(rng.integers(20, 50))
    draw.text((w * 0.15, h * 0.45), text, fill=(128, 128, 128, alpha), font=font)
    composite = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
    return np.array(composite.convert("RGB"))


def _add_paper_texture(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add subtle paper-like texture noise to background."""
    h, w = img.shape[:2]
    # Low-frequency noise via small array upscaled
    small_h, small_w = max(4, h // 32), max(4, w // 32)
    texture_small = rng.uniform(-8, 8, (small_h, small_w)).astype(np.float32)
    texture = cv2.resize(texture_small, (w, h), interpolation=cv2.INTER_CUBIC)
    texture_3ch = np.stack([texture] * 3, axis=-1)
    # Only add to near-white pixels (background)
    brightness = np.mean(img.astype(np.float32), axis=-1, keepdims=True)
    mask = (brightness > 200).astype(np.float32)
    return np.clip(img.astype(np.float32) + texture_3ch * mask, 0, 255).astype(np.uint8)


def _add_edge_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a gradient shadow on one edge (simulating book spine shadow)."""
    h, w = img.shape[:2]
    result = img.astype(np.float32)
    side = int(rng.integers(0, 4))  # 0=top, 1=bottom, 2=left, 3=right
    shadow_width = int(rng.uniform(0.1, 0.3) * (h if side < 2 else w))
    darkness = float(rng.uniform(0.3, 0.7))

    if side == 0:  # top
        for i in range(shadow_width):
            alpha = darkness * (1 - i / shadow_width)
            result[i, :, :] = result[i, :, :] * (1 - alpha)
    elif side == 1:  # bottom
        for i in range(shadow_width):
            alpha = darkness * (1 - i / shadow_width)
            result[h - 1 - i, :, :] = result[h - 1 - i, :, :] * (1 - alpha)
    elif side == 2:  # left
        for i in range(shadow_width):
            alpha = darkness * (1 - i / shadow_width)
            result[:, i, :] = result[:, i, :] * (1 - alpha)
    else:  # right
        for i in range(shadow_width):
            alpha = darkness * (1 - i / shadow_width)
            result[:, w - 1 - i, :] = result[:, w - 1 - i, :] * (1 - alpha)

    return np.clip(result, 0, 255).astype(np.uint8)


def _postprocess(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random scanning-like artifacts to a rendered chart."""
    # Rotation (70% chance)
    if rng.random() < 0.7:
        angle = float(rng.uniform(-3, 3))
        if abs(angle) > 0.3:
            img = _rotate_image(img, angle)

    # Gaussian noise (50%)
    if rng.random() < 0.5:
        sigma = float(rng.uniform(3, 15))
        img = _add_gaussian_noise(img, rng, sigma)

    # Salt & pepper (30%)
    if rng.random() < 0.3:
        amount = float(rng.uniform(0.005, 0.03))
        img = _add_salt_pepper(img, rng, amount)

    # JPEG compression (40%)
    if rng.random() < 0.4:
        quality = int(rng.integers(40, 80))
        img = _jpeg_compress(img, quality)

    # Slight blur (30%)
    if rng.random() < 0.3:
        kernel = float(rng.uniform(0.3, 1.2))
        img = _gaussian_blur(img, kernel)

    # Watermark (10%)
    if rng.random() < 0.1:
        img = _add_watermark(img, rng)

    # Paper texture (20%)
    if rng.random() < 0.2:
        img = _add_paper_texture(img, rng)

    # Edge shadow (10%)
    if rng.random() < 0.1:
        img = _add_edge_shadow(img, rng)

    return img


def _render_and_save(fig, ax, out_dir: Path, name: str, rng: np.random.Generator,
                     data_dict: dict, axes: dict):
    """Render figure, apply postprocessing, save image and meta."""
    img = _fig_to_array(fig)
    img = _postprocess(img, rng)
    pil_img = Image.fromarray(img)
    pil_img.save(out_dir / f"{name}.png")
    _save_meta(out_dir, name, data_dict, axes)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_simple_linear(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(25, 120))
    x_min = float(rng.uniform(-5, 10))
    x_max = float(x_min + rng.uniform(5, 40))
    x = np.linspace(x_min, x_max, n_pts)

    func_type = int(rng.integers(0, 8))
    if func_type == 0:
        y = np.sin((x - x_min) * float(rng.uniform(0.3, 5))) * float(rng.uniform(3, 30)) + 20
    elif func_type == 1:
        y = np.cos((x - x_min) * float(rng.uniform(0.3, 5))) * float(rng.uniform(3, 30)) + 20
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
    if y_min >= y_max:
        y_max = y_min + 10

    color = _pick(rng, LINE_COLORS)
    lw = _pick_lw(rng)
    ls = _pick_linestyle(rng)
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)

    # ~30% combo: line + error bars
    has_errorbar = rng.random() < 0.3
    if has_errorbar:
        yerr = np.abs(rng.normal(0, float(np.std(y)) * 0.1, n_pts))
        marker = _pick_marker(rng)
        ax.errorbar(x, y, yerr=yerr, color=color, linewidth=lw, linestyle=ls,
                    marker=marker, markersize=3, capsize=2, elinewidth=0.8)
    else:
        ax.plot(x, y, color=color, linewidth=lw, linestyle=ls)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V3_DIR / "simple_linear"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng,
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
    if func_type == 0:
        y = np.exp(float(rng.uniform(0.01, 0.2)) * (x - x_min)) + float(rng.uniform(0, 3))
    elif func_type == 1:
        y = (np.exp(float(rng.uniform(0.01, 0.1)) * (x - x_min))
             + np.exp(float(rng.uniform(0.1, 0.3)) * (x - x_min)) * 0.1)
    elif func_type == 2:
        y = (np.abs(x - x_min) + 0.1) ** float(rng.uniform(1.5, 4))
    elif func_type == 3:
        y = np.exp(float(rng.uniform(0.02, 0.15)) * (x - x_min)) * float(rng.uniform(1, 10)) + 1
    else:
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
    out_dir = TEST_DATA_V3_DIR / "log_y"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng,
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
    if func_type == 0:
        y = x ** power
    elif func_type == 1:
        y = float(rng.uniform(0.5, 5)) * x ** power
    elif func_type == 2:
        y = x ** power + float(rng.uniform(1, 10))
    else:
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
    out_dir = TEST_DATA_V3_DIR / "loglog"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng,
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
    if func_type == 0:
        amp1 = float(rng.uniform(50, 200))
        freq1 = float(rng.uniform(0.3, 3))
        y1 = np.sin((x - x_min) * freq1) * amp1 + amp1 + 10
        amp2 = float(rng.uniform(0.5, 5))
        freq2 = float(rng.uniform(0.5, 4))
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
    out_dir = TEST_DATA_V3_DIR / "dual_y"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax1, out_dir, name, rng,
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
    if func_type == 0:
        y = (x - x_min) ** 2 * float(rng.uniform(0.5, 3))
    elif func_type == 1:
        y = np.exp((x - x_min) * float(rng.uniform(0.1, 0.5)))
    elif func_type == 2:
        y = np.sqrt(x - x_min + 1) * float(rng.uniform(5, 20))
    elif func_type == 3:
        y = np.log(x - x_min + 2) * float(rng.uniform(5, 15))
    else:
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
    out_dir = TEST_DATA_V3_DIR / "inverted_y"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng,
                     {"series1": {"x": x.tolist(), "y": y.tolist()}},
                     {"x": {"type": "linear", "min": x_min, "max": x_max},
                      "y": {"type": "linear", "min": float(y_top), "max": float(y_bottom), "inverted": True}})


def generate_scatter(idx: int, seed: int):
    rng = np.random.default_rng(seed)
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
        slope = float(rng.uniform(0.2, 5))
        y = slope * x + rng.normal(0, float(rng.uniform(3, 20)), n_pts)
    elif dist_type == 2:
        centers = int(rng.integers(2, 6))
        pts_per = max(3, n_pts // centers)
        x = np.concatenate([rng.normal(float(rng.uniform(10, 90)), float(rng.uniform(5, 20)), pts_per) for _ in range(centers)])
        y = np.concatenate([rng.normal(float(rng.uniform(10, 80)), float(rng.uniform(3, 15)), pts_per) for _ in range(centers)])
        n_pts = len(x)
    elif dist_type == 3:
        x = rng.uniform(-5, 15, n_pts)
        y = float(rng.uniform(0.5, 3)) * x ** 2 + rng.normal(0, float(rng.uniform(5, 15)), n_pts)
    else:
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

    # ~30% combo: scatter + trend line
    has_trend = rng.random() < 0.3
    if has_trend:
        trend_color = _pick(rng, ["gray", "black", "dimgray"])
        z = np.polyfit(x, y, min(3, n_pts - 1))
        p = np.poly1d(z)
        x_sort = np.sort(x)
        ax.plot(x_sort, p(x_sort), color=trend_color, linewidth=1, linestyle="--", alpha=0.7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V3_DIR / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng,
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
    scatter_indices = set()

    # ~30% combo: some series are scatter
    has_scatter_combo = rng.random() < 0.3
    if has_scatter_combo:
        n_scatter = int(rng.integers(1, max(2, n_series)))
        scatter_indices = set(rng.choice(n_series, min(n_scatter, n_series), replace=False))

    for s in range(n_series):
        amp = float(rng.uniform(2, 20))
        freq = float(rng.uniform(0.3, 4))
        phase = float(rng.uniform(0, 2 * np.pi))
        func_type = int(rng.integers(0, 4))
        if func_type == 0:
            y = np.sin((x - x_min) * freq + phase) * amp
        elif func_type == 1:
            y = np.cos((x - x_min) * freq + phase) * amp
        elif func_type == 2:
            slope = float(rng.uniform(-3, 5))
            y = slope * (x - x_min) + amp * np.sin(freq * (x - x_min) + phase) * 0.3
        else:
            y = amp * np.tanh((x - (x_min + x_max) / 2) * freq * 0.2) + phase

        # Add jitter for scatter series
        if s in scatter_indices:
            x_noisy = x + rng.normal(0, float(rng.uniform(0.1, 0.5)), n_pts)
            y_noisy = y + rng.normal(0, float(rng.uniform(0.5, 3)), n_pts)
            data_dict[f"series{s + 1}"] = {"x": x_noisy.tolist(), "y": y_noisy.tolist()}
            all_y.append(y_noisy)
        else:
            data_dict[f"series{s + 1}"] = {"x": x.tolist(), "y": y.tolist()}
            all_y.append(y)

    all_y_arr = np.concatenate(all_y)
    y_min = float(np.floor(all_y_arr.min())) - float(rng.uniform(2, 8))
    y_max = float(np.ceil(all_y_arr.max())) + float(rng.uniform(2, 8))

    fig, ax = _make_fig_ax(rng)
    for s in range(n_series):
        c = colors[s] if s < len(colors) else _pick(rng, LINE_COLORS)
        series_data = data_dict[f"series{s + 1}"]
        sx = np.array(series_data["x"])
        sy = np.array(series_data["y"])

        if s in scatter_indices:
            ax.scatter(sx, sy, color=c, s=20 + int(rng.integers(0, 30)),
                       label=f"S{s + 1}", edgecolors="black", linewidths=0.5)
        else:
            lw = _pick_lw(rng)
            ls = _pick_linestyle(rng)
            ax.plot(sx, sy, color=c, linewidth=lw, linestyle=ls, label=f"S{s + 1}")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    legend_loc = _pick(rng, ["best", "upper right", "upper left", "lower right", "lower left"])
    ax.legend(loc=legend_loc, fontsize=int(rng.integers(7, 11)))
    grid_kw = _pick_grid(rng)
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V3_DIR / "multi_series"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng, data_dict,
                     {"x": {"type": "linear", "min": x_min, "max": x_max},
                      "y": {"type": "linear", "min": y_min, "max": y_max}})


def generate_log_x(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pts = int(rng.integers(30, 150))

    x_min_exp = float(rng.integers(-2, 3))
    x_max_exp = float(x_min_exp + rng.integers(2, 6))
    x = np.logspace(x_min_exp, x_max_exp, n_pts)

    func_type = int(rng.integers(0, 5))
    if func_type == 0:
        y = np.sqrt(x) * float(rng.uniform(2, 10))
    elif func_type == 1:
        y = np.log(x + 1) * float(rng.uniform(3, 20))
    elif func_type == 2:
        y = x ** float(rng.uniform(0.1, 0.8))
    elif func_type == 3:
        y = np.log(x) * float(rng.uniform(5, 15)) + float(rng.uniform(-10, 10))
    else:
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
    out_dir = TEST_DATA_V3_DIR / "log_x"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng,
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
    if func_type == 0:
        y = np.exp(-(x - x_min) * float(rng.uniform(0.1, 1.5))) * float(rng.uniform(5, 40))
    elif func_type == 1:
        y = np.sqrt(x - x_min + 1) * float(rng.uniform(2, 15))
    elif func_type == 2:
        y = np.log(x - x_min + 1) * float(rng.uniform(3, 20))
    elif func_type == 3:
        y = (x - x_min) * float(rng.uniform(1, 8)) + float(rng.uniform(-5, 10))
    elif func_type == 4:
        center = (x_min + x_max) / 2
        y = float(rng.uniform(10, 40)) / (1 + np.exp(-(x - center) * float(rng.uniform(0.5, 3))))
    else:
        y = np.sin((x - x_min) * float(rng.uniform(0.5, 3))) * float(rng.uniform(3, 15)) + 15

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

    # ~20% combo: add text annotations
    if rng.random() < 0.2:
        n_annotations = int(rng.integers(1, 4))
        for _ in range(n_annotations):
            ann_x = float(rng.uniform(x_min + (x_max - x_min) * 0.15, x_max - (x_max - x_min) * 0.15))
            idx_near = int(np.argmin(np.abs(x - ann_x)))
            ann_y = float(y[idx_near])
            text = _pick(rng, ["peak", "max", "note", "A", "B", "C", "val", "inflection"])
            ax.annotate(text, xy=(ann_x, ann_y),
                        xytext=(ann_x + float(rng.uniform(-1, 1)),
                                ann_y + float(rng.uniform(1, 5))),
                        fontsize=int(rng.integers(7, 11)),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V3_DIR / "no_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng,
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
    if func_type == 0:
        y = (np.sin(float(rng.uniform(1, 6)) * x) * float(rng.uniform(0.5, 2))
             + np.cos(float(rng.uniform(5, 15)) * x) * float(rng.uniform(0.1, 0.8)))
    elif func_type == 1:
        y = (np.sin(float(rng.uniform(2, 5)) * x) * float(rng.uniform(0.5, 1.5))
             + np.cos(float(rng.uniform(8, 15)) * x) * float(rng.uniform(0.2, 0.6))
             + np.sin(float(rng.uniform(20, 30)) * x) * float(rng.uniform(0.05, 0.2)))
    elif func_type == 2:
        f0 = float(rng.uniform(1, 3))
        f1 = float(rng.uniform(10, 20))
        phase = 2 * np.pi * (f0 * x + (f1 - f0) / (2 * x_max) * x ** 2)
        y = np.sin(phase) * float(rng.uniform(0.5, 1.5))
    elif func_type == 3:
        carrier = np.sin(float(rng.uniform(10, 20)) * x)
        envelope = np.sin(float(rng.uniform(0.5, 2)) * x) * float(rng.uniform(0.5, 1.5))
        y = carrier * envelope
    else:
        y = np.sin(x * float(rng.uniform(5, 15))) + 0.3 * np.sin(x * float(rng.uniform(30, 50)))

    y_min_val = float(np.floor(y.min())) - float(rng.uniform(0.5, 2))
    y_max_val = float(np.ceil(y.max())) + float(rng.uniform(0.5, 2))
    lw = float(rng.uniform(0.3, 2.0))

    color = _pick(rng, ["navy", "darkblue", "black", "darkgreen", "indigo",
                         "brown", "steelblue", "darkslategray", "midnightblue"])
    grid_kw = _pick_grid(rng)

    fig, ax = _make_fig_ax(rng)
    ax.plot(x, y, color=color, linewidth=lw)

    # ~20% combo: fill between
    if rng.random() < 0.2:
        fill_color = _pick(rng, ["lightblue", "lightyellow", "honeydew", "mistyrose"])
        ax.fill_between(x, y, y_min_val, alpha=0.2, color=fill_color)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_val, y_max_val)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, **grid_kw)

    name = f"{idx + 1:03d}"
    out_dir = TEST_DATA_V3_DIR / "dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_and_save(fig, ax, out_dir, name, rng,
                     {"series1": {"x": x.tolist(), "y": y.tolist()}},
                     {"x": {"type": "linear", "min": x_min, "max": x_max},
                      "y": {"type": "linear", "min": y_min_val, "max": y_max_val}})


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
        type_dir = TEST_DATA_V3_DIR / chart_type
        type_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N_PER_TYPE):
            seed = hash(("v3", chart_type, i)) % (2**31)
            gen_func(i, seed)
        print(f"  {chart_type}: {N_PER_TYPE} images generated")
    total = len(GENERATORS) * N_PER_TYPE
    print(f"Done. {total} images in {TEST_DATA_V3_DIR}")


if __name__ == "__main__":
    main()
