# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install dependencies and activate the virtual environment before running anything:

```bash
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Unix
pip install -r requirements.txt
```

### Run extraction on a single image
```bash
python plot_extractor/main.py input_chart.png --output extracted_data.csv --debug debug_output/
```

### Validate accuracy against test data
```bash
# Validate all 10 chart types
python tests/validate_by_type.py --debug

# Validate specific types only
python tests/validate_by_type.py --types simple_linear log_y inverted_y --debug

# Validate against a custom dataset (e.g., test_data_v2, test_data_v3, test_data_v4)
python tests/validate_by_type.py --data-dir test_data_v2 --debug
```

Validation outputs:
- Console: per-file relative error, SSIM, pass/fail status
- `report_by_type.csv` or `report_<data_dir>.csv`: detailed results
- `debug_type/<type>/`: rebuilt plots and extracted CSVs when `--debug` is used

### Generate test data
```bash
# Default dataset (test_data/)
python tests/generate_test_data.py

# Additional variants exist for different test sets
python tests/generate_test_data_v2.py
python tests/generate_test_data_v3.py
python tests/generate_test_data_v4.py
```

Each generator creates 30 PNGs per chart type plus `_meta.json` files with ground-truth axes and data points.

### Validate a single image manually
```python
from pathlib import Path
from plot_extractor.main import extract_from_image
result = extract_from_image(Path("test_data/simple_linear/001.png"), debug_dir=Path("debug_single"))
print(result["data"])        # extracted series
print(result["ssim_crop"])   # SSIM against rebuilt plot
```

## Architecture

### Pipeline Overview

The extraction pipeline is a linear sequence of 5 stages in `plot_extractor/main.py::extract_from_image()`:

1. **Image loading** (`core/image_loader.py`) — loads PNG, denoises with `fastNlMeansDenoisingColored`, converts to grayscale
2. **Axis detection** (`core/axis_detector.py`) — Canny + HoughLinesP to find horizontal/vertical axes, then detects tick marks via 1D edge projection and peak finding
3. **Axis calibration** (`core/axis_calibrator.py`) — OCR tick labels via pytesseract, fits pixel-to-data mapping (linear or log), auto-detects inversion from correlation sign
4. **Data extraction** (`core/data_extractor.py`) — foreground mask via background color subtraction, grid line removal, then extracts points by vertical median scan or connected-component centroids
5. **Plot rebuilding + SSIM** (`core/plot_rebuilder.py`, `utils/ssim_compare.py`) — reconstructs matplotlib plot from extracted data, compares to original with pure-NumPy SSIM

### Key Data Structures

**`Axis`** (`core/axis_detector.py`):
- `direction`: "x" | "y"
- `side`: "bottom" | "top" | "left" | "right"
- `position`: pixel coordinate of the axis line
- `plot_start` / `plot_end`: plot bounds along the orthogonal axis
- `ticks`: list of `(pixel, raw_value_or_None)`

**`CalibratedAxis`** (`core/axis_calibrator.py`):
- `axis`: the underlying `Axis`
- `axis_type`: "linear" | "log"
- `a`, `b`: fit coefficients for `pixel = a * f(value) + b` where `f` is identity or log10
- `inverted`: bool — Y-axis default has pixels increasing downward while values increase upward (corr < 0 = normal)
- `tick_map`: list of `(pixel, data_value)` pairs actually used for fitting
- `to_data(pixel)` / `to_pixel(value)`: conversion methods

**Metadata JSON format** (used for ground truth and fallback calibration):
```json
{
  "data": {
    "series1": {"x": [...], "y": [...]},
    "series2": {"x": [...], "y": [...]}
  },
  "axes": {
    "x": {"type": "linear", "min": 0, "max": 10},
    "y": {"type": "log", "min": 1, "max": 1000},
    "y_left": {"type": "linear", "min": 0, "max": 100},
    "y_right": {"type": "linear", "min": 0, "max": 10}
  }
}
```

### Multi-Series and Dual-Y Logic

The most complex logic lives in `core/data_extractor.py::extract_all_data()`:

- **Color separation**: HSV hue k-means clustering separates series by color. Low-saturation pixels (grays, anti-aliasing) are filtered out before clustering.
- **Dual-Y assignment**: When both left and right Y axes are detected with significantly different value ranges (`range_diff > 0.5 * max(range)`), the extractor tries both `(left, right)` and `(right, left)` assignments against ground-truth metadata and picks the lower-error one.
- **Same-color multi-series**: If color clustering yields only 1 cluster but metadata expects multiple series, `_extract_layered_series_from_mask()` uses per-column vertical layering with permutation-based continuity tracking to separate overlapping lines.
- **Series merging**: `_merge_series_fragments()` deduplicates overlapping traces. For multi-series metadata, the extractor scores raw color-separated traces vs merged traces against ground truth and keeps the better set.

### Grid Removal

`_remove_grid_lines()` in `core/data_extractor.py` uses HoughLinesP on the foreground mask to find long horizontal/vertical lines. Outer-edge lines are treated as axis borders; interior lines are classified as grid and erased. Grid detection runs on the **raw** (un-preprocessed) image because denoising blurs faint grid lines.

### Axis Classification

`utils/math_utils.py::classify_axis()` compares R² of linear fit vs log fit on tick pixel/value pairs. It automatically chooses the better model. If residual is too high (> 1e6), calibration falls back to linear.

### Validation Pass/Fail Criteria

`tests/validate_by_type.py` uses **data-level relative MAE** (not SSIM) as the pass/fail criterion:

| Type | Threshold |
|------|-----------|
| simple_linear, log_y, loglog, dual_y, inverted_y, log_x, no_grid | 5% |
| scatter | 8% |
| multi_series, dense | 6% |

SSIM is computed for reference only. Series matching uses permutation search over ground-truth series when both extracted and expected counts are ≤ 5.

## Project Conventions

- Config thresholds and constants live in `plot_extractor/config.py`
- `print()` is used for CLI output (not logging)
- Image arrays are RGB (OpenCV BGR loaded then converted)
- Matplotlib is used with `Agg` backend for headless plot generation
- Test data versions: `test_data/` (original), `test_data_v2/`, `test_data_v3/`, `test_data_v4/` — newer versions may have different generation parameters for validation robustness

## Key Files

| File | Purpose |
|------|---------|
| `plot_extractor/main.py` | CLI entry point and pipeline orchestration |
| `plot_extractor/config.py` | Thresholds and hyperparameters |
| `plot_extractor/core/axis_detector.py` | Hough-based axis/tick detection |
| `plot_extractor/core/axis_calibrator.py` | Pixel-to-data calibration (linear/log) |
| `plot_extractor/core/data_extractor.py` | Foreground masking, grid removal, point extraction, series separation |
| `plot_extractor/core/ocr_reader.py` | pytesseract tick label reading + meta JSON loader |
| `plot_extractor/core/plot_rebuilder.py` | Matplotlib reconstruction for validation |
| `plot_extractor/utils/math_utils.py` | Numeric parsing, linear/log fitting, axis classification |
| `plot_extractor/utils/image_utils.py` | Background detection, foreground mask |
| `plot_extractor/utils/ssim_compare.py` | Pure NumPy SSIM implementation |
| `tests/validate_by_type.py` | Per-type accuracy validation with CSV reporting |
| `tests/generate_test_data.py` | Ground-truth test image generator (30 per type) |

## Release Workflow

Before submitting any PR or pushing to `main`, run the lint gate:

```bash
pylint --fail-under=9 $(git ls-files '*.py')
```

This is the exact command executed by `.github/workflows/pylint.yml` on every push. The CI matrix tests Python 3.8 through 3.14. A failing lint gate blocks merge.

**Pre-release checklist:**

1. Lint passes with score ≥ 9.0 (`pylint --fail-under=9`)
2. Validation passes against supported datasets (`python tests/validate_by_type.py`)
3. No regressions in baseline pass rates (reference `docs/BASELINE_EVALUATION.md`)

Pylint configuration lives in `pyproject.toml` under `[tool.pylint]`. OpenCV (`cv2`) and PIL are in `ignored-modules` to suppress false-positive `no-member` errors. Refactoring hints (too-many-locals, too-many-branches, etc.) and docstring requirements are disabled — fix actual bugs, not noise.
