# Plot Extractor — User Guide

> Version: Beta 1.6.0

## Overview

Plot Extractor reads a chart image, detects axes, reads tick labels via OCR, extracts plotted data points, and writes structured CSV output. It supports ten chart types including linear, logarithmic, inverted, dual-axis, scatter, dense, and multi-series families.

## Installation

### Prerequisites

- Python 3.10+
- Tesseract OCR (required for tick label reading)

### Setup

```bash
git clone https://github.com/sea-monsters/plot-extractor.git
cd plot-extractor

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -e .
```

### Tesseract OCR

| OS | Command |
|----|---------|
| Ubuntu/Debian | `sudo apt-get install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) |

If Tesseract is unavailable, the pipeline falls back to heuristic tick generation (accuracy degrades significantly; use only for shape analysis).

## Quick Start

### CLI

```bash
# Basic extraction
plot-extractor chart.png -o output.csv

# With debug output (SSIM comparison, debug images)
plot-extractor chart.png -o output.csv -d debug_output/

# With LLM enhancement for ambiguous charts
plot-extractor chart.png --use-llm --use-ocr -o output.csv
```

Or via module:

```bash
python -m plot_extractor.main chart.png --output output.csv --debug debug_output/
```

### Python API

```python
from pathlib import Path
from plot_extractor.main import extract_from_image

result = extract_from_image(
    Path("chart.png"),
    output_csv=Path("output.csv"),
    use_ocr=True,
    debug_dir=Path("debug/"),
)

# Access extracted data
for series_name, series_data in result["data"].items():
    print(f"{series_name}: {len(series_data['x'])} points")

# SSIM score (when debug_dir is set)
print(f"SSIM: {result['ssim']}")
```

### Flags

| Flag | Purpose |
|------|---------|
| `--output`, `-o` | CSV output path |
| `--debug`, `-d` | Debug output directory (SSIM comparison, debug images) |
| `--use-ocr` | Enable Tesseract OCR for tick label reading (**recommended for accuracy**) |
| `--use-llm` | Enable LLM vision enhancement for ambiguous chart classification |
| `--meta` | Path to metadata JSON for axis range hints |

## Supported Chart Types

| Type | Description |
|------|-------------|
| `simple_linear` | Single-series line chart with linear axes |
| `log_y` | Y-axis logarithmic, X-axis linear |
| `log_x` | X-axis logarithmic, Y-axis linear |
| `loglog` | Both axes logarithmic |
| `inverted_y` | Y-axis inverted (values decrease upward) |
| `dual_y` | Two Y-axes with different scales |
| `scatter` | Scatter plot (point markers, no connecting lines) |
| `multi_series` | Multiple overlaid line series |
| `no_grid` | Chart without visible grid lines |
| `dense` | High-density oscillating curve |

## Output Format

Each extraction produces a CSV file with columns:

```
x_series1, y_series1, x_series2, y_series2, ...
```

The Python API returns a dict:

```python
{
    "data": {
        "series1": {"x": [1.0, 2.0, ...], "y": [10.0, 20.0, ...]},
        "series2": {"x": [1.0, 2.0, ...], "y": [5.0, 15.0, ...]}
    },
    "ssim": 0.9324,      # SSIM score (when debug_dir is set)
    "policy": {...},      # ExtractionPolicy used
    "routing": {...}      # Routing metrics
}
```

## Architecture

The extraction pipeline uses a three-layer strategy ensemble:

1. **Layer 1 — Rule-based routing** (default): lightweight image features → softmax type probabilities → `ExtractionPolicy` via decision tree (`adaptive_strategy.py`).
2. **Layer 2 — LLM vision enhancement** (opt-in via `--use-llm`): vision-capable LLM classifies ambiguous charts when rule-based confidence is low.
3. **Layer 3 — OCR tick reading** (opt-in via `--use-ocr`): Tesseract reads tick labels for axis calibration. Falls back to heuristic synthetic ticks when unavailable.

**New in Beta 1.6.0:**
- **Chart structural decomposition** (`layout/chart_structure.py`): decomposes chart into plot_area, x_axis_area, y_axis_area, and legend_area using detected axis positions. Replaces hardcoded thresholds (`h*0.8`, `w*0.2`) with structural context.
- **Junction-aware skeleton path tracing** (`core/skeleton_path.py`): for dense charts, traces continuous curves through thinned skeletons by detecting endpoints and branch points, resolving branches by direction continuity.
- **Adaptive strategy selector** (`core/adaptive_strategy.py`): replaces fixed `POLICY_WEIGHTS` matrix with a decision tree using measurable image features (foreground density, saturation, connected-component statistics) for routing.
- **Overlapping scatter separation** (`core/scatter_overlap.py`): detects abnormally large connected components (indicating overlapping markers) and splits them using greedy shape matching.

## Known Limitations

- **OCR-dependent accuracy**: without `--use-ocr`, extracted values are in arbitrary units. With OCR, accuracy is still sensitive to tick label quality (superscripts, small fonts, blur).
- **Multi-series charts**: HSV color clustering can fail when series have similar colors or heavy anti-aliasing. Junction-aware separation is planned but not yet integrated.
- **Degraded images**: scan/photo noise, blur, rotation, and JPEG artifacts reduce axis detection, thinning quality, and OCR accuracy.
- **Dense curves on noisy images**: Zhang-Suen thinning fragments on anti-aliased or thick curves; performance drops significantly on degraded data.
- **Log axes**: OCR superscript misread (e.g., 10² → "102") remains a critical bottleneck for log_x and loglog charts.
- **Unsupported chart types**: bar charts, pie charts, histograms, area charts, and step charts are not yet supported.
- **Multi-panel/subplot charts**: only single-panel charts are supported.
- **Extraction quality varies by chart type**: scatter and simple_linear charts extract most reliably; multi-series and log_x remain challenging.

## Validation

To evaluate extraction accuracy against ground truth:

```bash
# Standard evaluation (OCR required for meaningful metrics)
python tests/validate_by_type.py --use-ocr --workers 4

# V2/V3/V4 datasets
python tests/validate_by_type.py --data-dir test_data_v2 --use-ocr --workers 4
python tests/validate_by_type.py --data-dir test_data_v3 --use-ocr --workers 4
python tests/validate_by_type.py --data-dir test_data_v4 --v4-special --use-ocr --workers 4

# Unit tests for new modules (Beta 1.6.0)
pytest tests/test_chart_structure.py       # 20 tests — structural decomposition
pytest tests/test_skeleton_path.py         # 18 tests — path tracing
pytest tests/test_scatter_overlap.py       # 7 tests — overlap separation
pytest tests/test_adaptive_strategy.py     # 10 tests — decision tree routing
```

See `docs/CHANGELOG.md` for version history and baseline metrics.

## License

MIT License. See `LICENSE` for details.
