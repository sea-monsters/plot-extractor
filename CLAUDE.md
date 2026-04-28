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
# Default extraction (Layer 1 rule-based only)
python plot_extractor/main.py input_chart.png --output extracted_data.csv --debug debug_output/

# Enable LLM enhancement for ambiguous charts (Layer 1 + Layer 2)
python plot_extractor/main.py input_chart.png --use-llm --output extracted_data.csv

# Enable OCR for tick label reading (Layer 3)
python plot_extractor/main.py input_chart.png --use-ocr --output extracted_data.csv

# Enable both LLM and OCR
python plot_extractor/main.py input_chart.png --use-llm --use-ocr --output extracted_data.csv
```

### Validate accuracy against test data
```bash
# Validate all 10 chart types against v1 baseline
python tests/validate_by_type.py --debug

# Validate with LLM enhancement
python tests/validate_by_type.py --use-llm --debug

# Validate with OCR
python tests/validate_by_type.py --use-ocr --debug

# Validate with OCR in parallel (4 workers, much faster)
python tests/validate_by_type.py --use-ocr --workers 4 --debug

# Validate specific types only
python tests/validate_by_type.py --types simple_linear log_y inverted_y --debug

# Validate against different datasets
python tests/validate_by_type.py --data-dir test_data_v2 --debug
python tests/validate_by_type.py --data-dir test_data_v3 --debug
python tests/validate_by_type.py --data-dir test_data_v4 --v4-special --debug

# Validate v4a route profiles
python tests/validate_v4a_routes.py --data-dir test_data_v4a
```

Validation outputs:
- Console: per-file relative error, SSIM, pass/fail status, routing metrics (density/color/noise strategies)
- `report_by_type.csv` or `report_<data_dir>.csv`: detailed results
- `debug_type/<type>/`: rebuilt plots and extracted CSVs when `--debug` is used

### Generate test data
```bash
# Default dataset (test_data/)
python tests/generate_test_data.py

# Additional variants for different test sets
python tests/generate_test_data_v2.py
python tests/generate_test_data_v3.py
python tests/generate_test_data_v4.py
python tests/generate_test_data_v4a.py
```

Each generator creates 30 PNGs per chart type plus `_meta.json` files with ground-truth axes and data points.

### Validate a single image manually
```python
from pathlib import Path
from plot_extractor.main import extract_from_image

# Default extraction (rule-based only)
result = extract_from_image(Path("test_data/simple_linear/001.png"), debug_dir=Path("debug_single"))

# With LLM enhancement
result = extract_from_image(Path("chart.png"), use_llm=True, use_ocr=True)

print(result["data"])        # extracted series dict
print(result["ssim_crop"])   # SSIM against rebuilt plot (when debug_dir is set)
print(result["policy"])      # ExtractionPolicy used
print(result["routing"])     # routing metrics if available
```

### Unit tests for routing components
```bash
# Test axis candidate generation
python tests/test_axis_candidates.py

# Test series candidate generation
python tests/test_series_candidates.py
```

## Architecture

### Three-Layer Strategy Ensemble

The extraction pipeline uses a difficulty-stratified routing system:

**Layer 1 (default) — Rule-based ChartTypeGuesser + PolicyRouter**
- `core/chart_type_guesser.py`: lightweight image features → softmax type probabilities
- `core/policy_router.py`: type probabilities → `ExtractionPolicy` via weighted ensemble
- No external dependencies beyond OpenCV + NumPy
- Always runs; produces extraction config based on visual features

**Layer 2 (uncertain cases) — LLM Vision Enhancement**
- `core/llm_policy_router.py`: triggers when `top1_prob - top2_prob < 0.15` (low confidence)
- Calls vision-capable LLM to classify chart image
- Environment variables: `ANTHROPIC_API_KEY` (Claude), `OPENAI_API_KEY` (GPT), or custom endpoint
- Optional; activated via `--use-llm` flag

**Layer 3 (opt-in) — OCR (tesseract)**
- `core/ocr_reader.py`: reads tick label text for axis calibration
- Requires system tesseract binary + pytesseract
- Falls back to heuristic synthetic ticks when unavailable
- **CRITICAL for accuracy**: Without OCR, relative error can approach 100% on real-world charts
- Optional; activated via `--use-ocr` flag

### Policy Router Architecture

The `ExtractionPolicy` dataclass (in `core/policy_router.py`) controls these pipeline stages:

| Field | Values | Affected Module |
|-------|--------|----------------|
| `noise_strategy` | `clean`, `salt_pepper`, `jpeg`, `blur`, `rotation_noise` | `image_loader.py` preprocessing |
| `rotation_correct` | `True` / `False` | `main.py` rotation correction |
| `color_strategy` | `hue_only`, `hsv3d`, `layered`, `none` | `data_extractor.py` series separation |
| `density_strategy` | `standard`, `thinning`, `scatter` | `data_extractor.py` point extraction |
| `ocr_block_size` | odd integer | `ocr_reader.py` adaptive threshold |
| `hough_threshold` | integer | `axis_detector.py` HoughLinesP threshold |
| `cal_residual_threshold_linear` | float | `axis_calibrator.py` residual gate |

Policy is computed from chart-type probabilities via a weighted matrix (see `POLICY_WEIGHTS` in `policy_router.py`).

### Extraction Pipeline (5 Stages)

Once `ExtractionPolicy` is determined, `plot_extractor/main.py::extract_from_image()` executes:

1. **Image loading** (`core/image_loader.py`) — loads PNG, applies policy-specified preprocessing (denoising, rotation correction if enabled), converts to grayscale

2. **Axis detection** (`core/axis_detector.py`) — Canny + HoughLinesP with policy-specified thresholds, detects tick marks via 1D edge projection and peak finding; rotation detection runs on raw image

3. **Axis calibration** (`core/axis_calibrator.py`) — OCR tick labels (if enabled) via pytesseract with policy-specified preprocessing, fits pixel-to-data mapping (linear or log) using RANSAC robust regression to reject outlier OCR misreads, auto-detects inversion

4. **Data extraction** (`core/data_extractor.py`) — foreground mask via background subtraction, policy-specified color separation (hue-only, HSV-3D, or layered), grid removal, policy-specified point extraction (standard vertical scan, thinning for dense, or connected components for scatter)

5. **Plot rebuilding + SSIM** (`core/plot_rebuilder.py`, `utils/ssim_compare.py`) — reconstructs matplotlib plot from extracted data, compares to original with pure-NumPy SSIM

### New Architecture Modules (v4a Phase)

**Core routing and candidate generation**:
- `core/chart_type_guesser.py` — lightweight feature extraction + softmax type classifier
- `core/policy_router.py` — policy ensemble from type probabilities
- `core/llm_policy_router.py` — LLM vision fallback for ambiguous cases
- `core/confidence.py` — confidence scoring and quality gates
- `core/axis_candidates.py` — axis candidate generation with ranking
- `core/series_candidates.py` — series candidate generation with quality gates

**Geometry and layout**:
- `geometry/grid_suppress.py` — directional morphology for grid line removal
- `geometry/legend_bind.py` — legend detection and binding to series
- `layout/plot_area.py` — plot area detection within panels
- `layout/text_roi.py` — text region-of-interest detection
- `layout/panel_split.py` — multi-panel chart splitting

**Service layer**:
- `service/mcp_server.py` — MCP tool interface for Claude Code integration
- `service/schemas.py` — structured extraction result schemas
- `service/debug_overlay.py` — visual debug overlay generation

### Key Data Structures

**`ImageFeatures`** (`core/chart_type_guesser.py`):
- Geometric: Hough line counts (horiz/vert/diag), edge density, aspect ratio
- Color: hue peak count, saturation mean, color dominance
- Texture/noise: Laplacian variance, extreme pixel ratio, FFT high-frequency ratio, block variance ratio
- Structural: foreground column density, connected component stats, axis count, tick regularity, rotation estimate

**`ExtractionPolicy`** (`core/policy_router.py`):
- Preprocessing: noise_strategy, rotation_correct, median/bilateral/unsharp params
- Extraction: color_strategy, density_strategy, min_clusters, thinning_quality_gate
- OCR: block_size, C constant, deskew flag
- Axis detection: Hough thresholds
- Calibration: residual thresholds, meta fallback

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
- `inverted`: bool — Y-axis default has pixels increasing downward while values increase upward
- `tick_map`: list of `(pixel, data_value)` pairs used for fitting
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
    "y": {"type": "log", "min": 1, "max": 1000}
  }
}
```

### Algorithm-Level Changes (v4a)

Five algorithm-level optimizations were implemented after threshold-tuning failures:

1. **HSV 3D clustering** — full H+S+V k-means fallback when hue-only fails (quality gates: compactness, min_hue_dist, x-coverage)
2. **OCR preprocessing** — crop-level grayscale + upscale + median blur + adaptive thresholding before tesseract
3. **Zhang-Suen thinning** — contrib-gated `cv2.ximgproc.thinning` with pure NumPy fallback for dense charts
4. **Rotation correction** — strict rotation estimator (edge lines only, median aggregation, angle agreement gate) with coordinate-space rotation before axis detection
5. **RANSAC robust regression** — Scatteract-style RANSAC for axis calibration outlier rejection; custom pure-NumPy implementation with pixel-grounded threshold and fallback to ordinary least squares

See `docs/ARCHITECTURAL_CHANGES_IMPL.md` for detailed implementation notes.

### Validation Pass/Fail Criteria

`tests/validate_by_type.py` uses **data-level relative MAE** (not SSIM) as pass/fail:

| Type | Threshold |
|------|-----------|
| simple_linear, log_y, loglog, dual_y, inverted_y, log_x, no_grid | 5% |
| scatter | 8% |
| multi_series, dense | 6% |

SSIM is computed for reference only. Series matching uses permutation search when both extracted and expected counts are ≤ 5.

## Project Conventions

- Config thresholds and constants live in `plot_extractor/config.py`
- `print()` is used for CLI output (not logging)
- Image arrays are RGB (OpenCV BGR loaded then converted)
- Matplotlib uses `Agg` backend for headless plot generation
- Test data versions: `test_data/` (v1), `test_data_v2/`, `test_data_v3/`, `test_data_v4/`, `test_data_v4a/`
- OpenCV contrib features are gated by `hasattr(cv2, "ximgproc")` with fallbacks

## Key Files

| File | Purpose |
|------|---------|
| `plot_extractor/main.py` | CLI entry point, pipeline orchestration, rotation correction |
| `plot_extractor/config.py` | Thresholds and hyperparameters |
| `plot_extractor/core/chart_type_guesser.py` | Lightweight feature extraction + type classifier |
| `plot_extractor/core/policy_router.py` | Policy ensemble from type probabilities |
| `plot_extractor/core/llm_policy_router.py` | LLM vision fallback for ambiguous charts |
| `plot_extractor/core/axis_detector.py` | Hough-based axis/tick detection + rotation estimator |
| `plot_extractor/core/axis_calibrator.py` | Pixel-to-data calibration (linear/log) with OCR preprocessing |
| `plot_extractor/core/data_extractor.py` | Foreground masking, HSV clustering, thinning, point extraction |
| `plot_extractor/core/ocr_reader.py` | pytesseract tick reading with crop-level preprocessing |
| `plot_extractor/core/plot_rebuilder.py` | Matplotlib reconstruction for validation |
| `plot_extractor/core/axis_candidates.py` | Axis candidate generation and ranking |
| `plot_extractor/core/series_candidates.py` | Series candidate generation with quality gates |
| `plot_extractor/geometry/grid_suppress.py` | Grid line removal via directional morphology |
| `plot_extractor/geometry/legend_bind.py` | Legend detection and binding |
| `plot_extractor/layout/plot_area.py` | Plot area detection within panels |
| `plot_extractor/utils/math_utils.py` | Numeric parsing, linear/log fitting, axis classification |
| `plot_extractor/utils/image_utils.py` | Background detection, foreground mask, rotation |
| `plot_extractor/utils/ssim_compare.py` | Pure NumPy SSIM implementation |
| `plot_extractor/service/mcp_server.py` | MCP tool interface |
| `tests/validate_by_type.py` | Per-type accuracy validation with routing metrics |
| `tests/validate_v4a_routes.py` | v4a route-profile validation |
| `tests/test_axis_candidates.py` | Unit tests for axis candidates |
| `tests/test_series_candidates.py` | Unit tests for series candidates |
| `tests/generate_test_data.py` | Ground-truth test image generator (30 per type) |
| `docs/BASELINE_EVALUATION.md` | Validation baselines and true pass rates (meta-isolated) |
| `docs/ARCHITECTURAL_CHANGES_IMPL.md` | Implementation details for algorithm-level changes |
| `docs/SCATTERACT_COMPARISON.md` | Workflow and design comparison with Scatteract (Bloomberg) |

## Release Workflow

Before submitting any PR or pushing to `main`, run the lint gate:

```bash
pylint --fail-under=9 $(git ls-files '*.py')
```

This is the exact command executed by `.github/workflows/pylint.yml` on every push. The CI matrix tests Python 3.8 through 3.14. A failing lint gate blocks merge.

**Pre-release checklist:**

1. Lint passes with score ≥ 9.0 (`pylint --fail-under=9`)
2. Validation passes against v1-v4 datasets (`python tests/validate_by_type.py --data-dir test_data`)
3. No regressions in baseline pass rates (reference `docs/BASELINE_EVALUATION.md`)

Pylint configuration lives in `pyproject.toml` under `[tool.pylint]`. OpenCV (`cv2`) and PIL are in `ignored-modules` to suppress false-positive `no-member` errors. Refactoring hints and docstring requirements are disabled.

## Environment Variables

### LLM Configuration (optional, for `--use-llm`)

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Use Claude vision models | `claude-haiku-4-5` |
| `OPENAI_API_KEY` | Use GPT vision models | `gpt-5.4-nano` |
| `LLM_BASE_URL` | Custom OpenAI-compatible endpoint | — |
| `LLM_API_KEY` | API key for custom endpoint | — |
| `LLM_MODEL` | Model name override | Provider-specific default |

Provider priority: Anthropic → OpenAI → Custom endpoint. When running in Claude Code, `ANTHROPIC_API_KEY` is typically already set.

### OCR Configuration (optional, for `--use-ocr`)

Tesseract must be installed on the system:
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

If missing, pipeline falls back to heuristic tick generation (accuracy degrades significantly).