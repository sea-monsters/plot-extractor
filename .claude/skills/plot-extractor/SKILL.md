---
name: plot-extractor
description: >-
  Extract structured data points (x, y series) from chart images.
  Supports linear, log, inverted, dual-y, scatter, dense, and multi-series
  chart types. Reads a PNG/JPG chart image and outputs a CSV with columns
  [series, x, y]. Uses Hough line detection for axis finding, pytesseract OCR
  for tick label reading, and color clustering for multi-series separation.
  TRIGGER when: user asks to extract data from a chart, plot, or graph image;
  user wants to digitize a figure; user mentions "get data from chart",
  "extract points from graph", "digitize plot", or similar.
  DO NOT TRIGGER when: the input is structured data (CSV, JSON, Excel);
  the chart is a hand-drawn sketch with no printed axes; the image is a
  photograph with severe perspective distortion.
origin: project
---

# Plot Extractor — Chart Data Extraction Tool

Extract numerical data points from chart images into structured CSV output.

## When to Use

- User provides a chart image and wants the underlying data as CSV/numbers
- User needs to digitize a published figure for re-analysis
- User asks to "get data from this chart" or "extract points from this graph"
- User wants to recover data series from a saved plot image

**Skip** when the input is already tabular data, hand-drawn with no axes, or the user only wants a visual description of the chart.

## Quick Start

The primary entry point is `plot_extractor.main:extract_from_image()`. Invoke it from Python:

```python
from pathlib import Path
from plot_extractor.main import extract_from_image

result = extract_from_image(
    Path("chart.png"),
    output_csv=Path("output.csv"),
    debug_dir=Path("debug/"),   # optional: SSIM rebuild + comparison
)
```

Or run the CLI:

```bash
python -m plot_extractor.main chart.png -o output.csv -d debug/
```

A meta JSON file with ground-truth axes or series can optionally guide calibration:

```bash
python -m plot_extractor.main chart.png -o output.csv --meta metadata.json
```

## Pipeline (How It Works)

The extraction runs a 5-stage pipeline inside `extract_from_image()`:

1. **Load & preprocess** — denoise with `fastNlMeansDenoisingColored`, convert to grayscale
2. **Detect axes** — Canny edges + HoughLinesP to find horizontal/vertical axis lines, then 1D edge projection for tick marks
3. **Calibrate axes** — pytesseract OCR on tick labels, fit pixel-to-data mapping (linear or log via R² comparison), auto-detect inversion
4. **Extract data** — background-color subtraction for foreground mask, Hough-based grid line removal, centroid or vertical-median point extraction, HSV hue clustering for multi-series separation
5. **Rebuild + SSIM** (debug mode) — reconstruct a matplotlib plot from extracted data, compare to original image

## Return Value

`extract_from_image()` returns a dict or `None`:

```python
{
    "data": {                    # extracted data series
        "series1": {"x": [...], "y": [...]},
        "series2": {"x": [...], "y": [...]},
    },
    "calibrated_axes": [...],    # CalibratedAxis objects
    "ssim": 0.9234,             # full-image SSIM (if debug_dir set)
    "ssim_crop": 0.9512,        # crop-to-plot SSIM (if debug_dir set)
    "csv": Path("output.csv"),  # output CSV path
    "plot_bounds": (l, t, r, b),# pixel bounds of the plot area
    "is_scatter": False,        # whether scatter markers were detected
    "has_grid": True,           # whether grid lines were found+removed
    "diagnostics": {...},       # per-axis fit residuals, point counts, ranges
}
```

Output CSV format:

| series | x | y |
|--------|---|---|
| series1 | 0.0 | 1.0 |
| series1 | 1.0 | 2.5 |
| series2 | 0.0 | 3.0 |

## Supported Chart Types

| Type | Notes |
|------|-------|
| `simple_linear` | Standard Cartesian with linear axes |
| `log_y` / `log_x` / `loglog` | Logarithmic axes |
| `inverted_y` | Y-axis values decreasing upward |
| `dual_y` | Two independent y-axes (left/right) |
| `scatter` | Point markers instead of lines |
| `multi_series` | Multiple colored data series |
| `no_grid` | Charts without grid lines |
| `dense` | High data-point density |

## Limitations

- **Severe perspective distortion** (angled photos) will break axis detection
- **Hand-drawn axes** without printed tick marks cannot be calibrated
- **Multi-series with crossing lines** or near-identical colors may merge incorrectly
- **Charts with subplots** or faceted layouts are not supported
- **pie charts, bar charts, heatmaps** — only 2D Cartesian line/scatter plots
- OCR requires pytesseract and Tesseract binaries to be installed locally

## Installation Requirements

```bash
pip install -e .                    # install in editable mode
# plus: Tesseract OCR engine for tick label reading
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# macOS:   brew install tesseract
# Linux:   apt install tesseract-ocr
```
