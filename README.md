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

```bash
git clone https://github.com/sea-monsters/plot-extractor.git
cd plot-extractor

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies include Python 3.8+, OpenCV, NumPy, Matplotlib, and pytesseract for optional OCR.

## Usage

Extract data from a chart image:

```bash
python plot_extractor/main.py input_chart.png --output extracted_data.csv
```

Run with debug outputs:

```bash
python plot_extractor/main.py input_chart.png --debug debug_output/
```

Use metadata when ground-truth axes or labels are available:

```bash
python plot_extractor/main.py input_chart.png --meta metadata.json
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

Current baseline details are maintained in [docs/BASELINE_EVALUATION.md](docs/BASELINE_EVALUATION.md).

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

## License

MIT License. See `LICENSE` for details.
