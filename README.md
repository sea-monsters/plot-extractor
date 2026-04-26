# Plot Extractor

Automated data extraction from chart images with axis calibration and validation.

> Status: **Beta**. Core extraction paths are working for supported chart types, but robustness on diverse and degraded real-world images is still under active optimization.

## Overview

Plot Extractor detects chart axes, calibrates pixel-to-data coordinates, extracts plotted data, and writes structured CSV output. It supports linear, logarithmic, inverted, dual-axis, scatter, dense, and multi-series chart families.

## Supported Chart Types

- `simple_linear`
- `log_y`
- `loglog`
- `log_x`
- `inverted_y`
- `dual_y`
- `scatter`
- `multi_series`
- `no_grid`
- `dense`

## Installation

### Standard (pip)

```bash
git clone https://github.com/sea-monsters/plot-extractor.git
cd plot-extractor

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

Dependencies include Python 3.10+, OpenCV, NumPy, Matplotlib, and pytesseract for OCR.

### Tesseract OCR (required for tick label reading)

```bash
# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS
brew install tesseract
# Linux
apt install tesseract-ocr
```

## Usage

### CLI

```bash
plot-extractor input_chart.png -o output.csv -d debug_output/
```

Or via module:

```bash
python -m plot_extractor.main input_chart.png --output output.csv --debug debug_output/
```

### Python API

```python
from pathlib import Path
from plot_extractor.main import extract_from_image

result = extract_from_image(
    Path("chart.png"),
    output_csv=Path("output.csv"),
    debug_dir=Path("debug/"),
)
print(result["data"])         # dict of series_name → {x: [...], y: [...]}
print(result["ssim"])         # SSIM score (when debug_dir is set)
```

## Claude Code Plugin

This project includes a Claude Code skill that lets Claude Code directly extract data from chart images.

### Setup

1. Clone and install the repo as above.
2. The skill at `.claude/skills/plot-extractor/SKILL.md` is auto-discovered by Claude Code.
3. In a Claude Code session, the skill activates automatically when you ask Claude to extract data from a chart image.

### Example in Claude Code

```
User: extract data from this chart and save as CSV
[attach: chart.png]

Claude: [invokes plot-extractor skill, runs extraction]
        Extracted 2 series with 245 points. CSV saved to chart.csv.
        SSIM: 0.9324
```

You can also invoke it explicitly:

```
/plot-extractor  # triggers the skill directly
```

### Requirements in Claude Code

- The Python environment used by Claude Code must have the package installed (`pip install -e /path/to/plot-extractor`).
- Tesseract OCR must be on PATH for tick label reading.
- If Tesseract is unavailable, provide axis metadata via `--meta metadata.json` as a fallback.

## Release Workflow

Every push triggers the Pylint CI check (`.github/workflows/pylint.yml`). Before submitting or merging a PR, ensure the lint gate passes locally.

### Pre-Release Checklist

1. **Lint** — must score ≥ 9.0 with no errors or warnings:

   ```bash
   pip install pylint
   pylint --fail-under=9 $(git ls-files '*.py')
   ```

   Configuration is in `pyproject.toml` under `[tool.pylint]`. The CI matrix tests Python 3.8 through 3.14.

2. **Validation** — run the evaluator against supported datasets (see below).

3. **Review** — reference the baseline in `docs/BASELINE_EVALUATION.md` for expected pass rates.

Failures in the lint gate block merge. Fix issues locally before pushing:

```bash
# Run the exact command the CI uses
pylint --fail-under=9 $(git ls-files '*.py')

# Check only the core package
pylint plot_extractor/
```

## Validation

Run the standard evaluator:

```bash
python tests/validate_by_type.py
python tests/validate_by_type.py --data-dir test_data_v2
python tests/validate_by_type.py --data-dir test_data_v3
```

Run the v4 supported-domain evaluator:

```bash
python tests/validate_by_type.py --data-dir test_data_v4 --v4-special
```

Current baseline details are maintained in [docs/BASELINE_EVALUATION.md](docs/BASELINE_EVALUATION.md). Bottleneck analysis and lightweight skill candidates are tracked in [docs/EXTRACTION_BOTTLENECKS_SKILL_PLAN.md](docs/EXTRACTION_BOTTLENECKS_SKILL_PLAN.md).

## Current Baseline

| Dataset | Scope | Pass | Rate |
|---------|-------|------|------|
| v1 | supported chart types | 288/310 | 92.9% |
| v2 | wider generated variation | 383/500 | 76.6% |
| v3 | scan/photo degradation simulation | 184/500 | 36.8% |
| v4 | supported single-chart subset | 91/204 | 44.6% |

v4 contains 500 mixed-scope images; 296 are currently outside the extractor's supported domain, such as combo charts, multi-subplot charts, unsupported chart types, or partial crops.

## Architecture

```text
plot_extractor/
├── core/
│   ├── image_loader.py
│   ├── axis_detector.py
│   ├── axis_calibrator.py
│   ├── data_extractor.py
│   ├── plot_rebuilder.py
│   └── ocr_reader.py
├── utils/
│   ├── image_utils.py
│   ├── math_utils.py
│   └── ssim_compare.py
└── config.py
```

## Known Beta Limitations

- Multi-series charts still struggle with crossings, nearby colors, and degraded line quality.
- Scan/photo degradation can break axis detection and calibration.
- Dense curves can be over-fragmented or confused with grid/text artifacts.
- v4 mixed charts are evaluated only within the currently supported single-chart domain.
- Tesseract OCR must be installed separately for tick label reading.

## License

MIT License. See `LICENSE` for details.
