# Plot Extractor

Automated data extraction from chart images with axis calibration and OCR tick reading.

> Status: **Beta 1.6.0**. Core extraction paths are working for supported chart types. Recent additions: chart structural decomposition, junction-aware skeleton path tracing, overlapping scatter separation, and adaptive strategy routing. Active development targets OCR calibration robustness, log-scale detection, and multi-series color separation.

## Supported Chart Types

`simple_linear` · `log_y` · `log_x` · `loglog` · `inverted_y` · `dual_y` · `scatter` · `multi_series` · `no_grid` · `dense`

## Installation

```bash
git clone https://github.com/sea-monsters/plot-extractor.git
cd plot-extractor
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -e .
```

**Tesseract OCR** is required for tick label reading:

| OS | Command |
|----|---------|
| Ubuntu/Debian | `sudo apt-get install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) |

## Quick Start

```bash
# CLI
plot-extractor chart.png -o output.csv -d debug_output/
python -m plot_extractor.main chart.png --use-ocr --output output.csv

# Python API
from pathlib import Path
from plot_extractor.main import extract_from_image

result = extract_from_image(Path("chart.png"), use_ocr=True)
for name, s in result["data"].items():
    print(f"{name}: {len(s['x'])} points")
```

## Claude Code Plugin

This project includes a Claude Code skill for chart data extraction. In a Claude Code session:

```
/extract data from this chart and save as CSV
[attach: chart.png]
```

Or invoke explicitly: `/plot-extractor`

Requirements: the Python environment used by Claude Code must have the package installed (`pip install -e .`), and Tesseract must be on PATH.

## Architecture Highlights

- **Structural decomposition** (`layout/chart_structure.py`): CACHED-inspired 4-area decomposition (plot, x-axis, y-axis, legend) replaces hardcoded position thresholds.
- **Junction-aware path tracing** (`core/skeleton_path.py`): Traces continuous curves through thinned skeletons for dense chart extraction.
- **Adaptive strategy selector** (`core/adaptive_strategy.py`): Decision-tree policy routing from measurable image features instead of fixed weight matrices.
- **Overlap separation** (`core/scatter_overlap.py`): Greedy shape-matching for overlapping scatter markers.

## Documentation

| Document | Audience |
|----------|----------|
| **[User Guide](docs/USER_GUIDE.md)** | Users — installation, CLI/Python API, supported types, output format, limitations |
| **[Changelog](docs/CHANGELOG.md)** | Users & contributors — version history, baseline metrics, feature additions per release |
| **[Architecture Details](docs/ARCHITECTURAL_CHANGES_IMPL.md)** | Contributors — implementation notes for algorithm-level changes |

## CI & Validation

Every push triggers a Pylint check (`.github/workflows/pylint.yml`, score ≥ 9.0 across Python 3.8–3.14).

```bash
# Lint gate
pylint --fail-under=9 $(git ls-files '*.py')

# Accuracy evaluation (OCR required for meaningful metrics)
python tests/validate_by_type.py --use-ocr --workers 4
python tests/validate_by_type.py --data-dir test_data_v2 --use-ocr --workers 4
python tests/validate_by_type.py --data-dir test_data_v3 --use-ocr --workers 4
python tests/validate_by_type.py --data-dir test_data_v4 --v4-special --use-ocr --workers 4

# Unit tests for new modules
pytest tests/test_chart_structure.py
pytest tests/test_skeleton_path.py
pytest tests/test_scatter_overlap.py
pytest tests/test_adaptive_strategy.py
```

## License

MIT License. See `LICENSE` for details.
