# Plot Extractor

Automated data extraction from chart images with axis calibration and OCR tick reading.

> Status: **Beta 1.5.0**. Core extraction paths are working for supported chart types. Active development targets OCR calibration robustness, log-scale detection, and multi-series color separation.

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

## Documentation

| Document | Audience |
|----------|----------|
| **[User Guide](docs/USER_GUIDE.md)** | Users — installation, CLI/Python API, supported types, output format, limitations |
| **[Changelog](docs/CHANGELOG.md)** | Users & contributors — version history, baseline metrics, feature additions per release |

## CI & Validation

Every push triggers a Pylint check (`.github/workflows/pylint.yml`, score ≥ 9.0 across Python 3.8–3.14).

```bash
# Lint gate
pylint --fail-under=9 $(git ls-files '*.py')

# Accuracy evaluation (OCR required)
python tests/validate_by_type.py --use-ocr --workers 4
python tests/validate_by_type.py --data-dir test_data_v2 --use-ocr --workers 4
```

## License

MIT License. See `LICENSE` for details.
