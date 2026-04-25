# Plot Extractor

**Automated data extraction from chart images with axis calibration and validation**

## Overview

Plot Extractor is a Python tool that automatically extracts numerical data from chart images. It detects axes, calibrates coordinate systems (linear, logarithmic, inverted), and extracts data points with high accuracy, outputting structured CSV files.

## Features

- **Axis Detection**: Automatically identifies X/Y axes, tick marks, and plot boundaries
- **Axis Calibration**: Supports linear, logarithmic (log y, log x, loglog), and inverted Y axes
- **Data Extraction**: Handles single-line, multi-line, scatter plots, and dense charts
- **Grid Removal**: Detects and removes grid lines to improve extraction accuracy
- **Multi-Series Separation**: Color-based clustering to separate multiple data series
- **Output**: Generates CSV files with extracted (x, y) coordinates
- **Validation**: Rebuilds plots for visual comparison (SSIM-based)

## Supported Chart Types

1. **simple_linear**: Single continuous line with linear axes
2. **log_y**: Exponential growth curves with log Y axis
3. **loglog**: Power-law relationships with both axes logarithmic
4. **log_x**: Logarithmic X axis with linear Y axis
5. **inverted_y**: Y axis inverted (y_max at bottom)
6. **dual_y**: Dual Y-axis charts (left and right axes)
7. **scatter**: Disconnected scatter points
8. **multi_series**: Multiple colored lines (2-4 series)
9. **no_grid**: Charts without grid lines
10. **dense**: High-frequency oscillating curves

## Installation

```bash
git clone https://github.com/sea-monsters/plot-extractor.git
cd plot-extractor

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib
- pytesseract (OCR)

## Usage

### Basic Extraction

```bash
python plot_extractor/main.py input_chart.png --output extracted_data.csv
```

### With Debug Output

```bash
python plot_extractor/main.py input_chart.png --debug debug_output/
```

This generates:
- `extracted_data.csv`: Extracted (x, y) coordinates
- `rebuilt_chart.png`: Reconstructed plot for validation
- SSIM comparison scores

### With Ground Truth Metadata

If you have ground truth axis labels in JSON format:

```bash
python plot_extractor/main.py input_chart.png --meta metadata.json
```

Metadata format:
```json
{
  "axes": {
    "x": {"type": "linear", "min": 0, "max": 10},
    "y": {"type": "log", "min": 1, "max": 1000}
  },
  "data": {
    "series1": {"x": [0, 1, 2, ...], "y": [1, 10, 100, ...]}
  }
}
```

## Validation

Run per-type validation on test datasets:

```bash
python tests/validate_by_type.py --types simple_linear log_y --debug
```

Generates validation report showing:
- Pass rate per type
- Average/max relative error
- Per-image SSIM scores

## Current Status

**Development Phase**: Active optimization for accuracy across 10 chart types

| Type | Pass Rate | Avg Error | Status |
|------|-----------|-----------|--------|
| simple_linear | 51.6% | 6.1% | Improving (grid interference) |
| log_y | 41.9% | 9.4% | Log axis calibration refinement |
| inverted_y | 77.4% | 3.6% | Stable |
| no_grid | 45.2% | 6.4% | Marker detection needed |
| scatter | 6.5% | 19.3% | Evaluation method needs DTW |
| dense | 6.5% | 24.0% | Curve tracing needed |
| loglog | 3.2% | 100.2% | Log axis tick fallback |
| log_x | 3.2% | 65.3% | Log axis tick fallback |
| dual_y | 0.0% | 1390% | Right axis binding missing |
| multi_series | 0.0% | 63.6% | Color separation at crossings |

See detailed analysis in `docs/TYPE_CALIBRATION_EVALUATION_2026-04-25.md`.

## Architecture

```
plot_extractor/
├── core/
│   ├── image_loader.py      # Image loading and preprocessing
│   ├── axis_detector.py     # Hough line detection for axes
│   ├── axis_calibrator.py   # Pixel-to-data mapping (linear/log)
│   ├── data_extractor.py    # Foreground mask and point extraction
│   ├── plot_rebuilder.py    # Reconstruct plot for validation
│   └── ocr_reader.py        # Tick label OCR (optional)
├── utils/
│   ├── image_utils.py       # Background detection, foreground mask
│   ├── math_utils.py        # Linear/log fitting, axis classification
│   └── ssim_compare.py      # SSIM visual comparison
└── config.py                # Thresholds and constants
```

## Testing

Generate random test images with ground truth:

```bash
python tests/generate_test_data.py
```

Creates 30 random charts per type in `test_data/<type>/` with metadata JSON files.

## Known Limitations

1. **Grid interference**: Dense grid lines can bias median-based extraction
2. **Dual Y-axis binding**: Right axis series not correctly calibrated (under development)
3. **Log axis OCR**: Tick label recognition may fail for log axes
4. **Crossing points**: Multi-series charts with line crossings may have color mixing
5. **Marker overlap**: Charts with point markers (circles, squares) need special handling

## Roadmap

### Phase 1 (Current)
- Fix dual_y right axis binding
- Improve simple_linear grid removal via color separation
- Add log axis fallback when OCR fails

### Phase 2
- Implement curve tracing for dense charts
- Add marker detection for no_grid charts
- Improve scatter evaluation with DTW matching

### Phase 3
- Multi-series crossing point handling
- Color palette standardization
- Edge case handling (partial axes, rotated plots)

## Contributing

This project is under active development. For collaboration:
- Review optimization logs in `docs/`
- Test extraction on your chart images
- Submit issues with sample images for new chart types

## License

MIT License - See LICENSE file for details

## Credits

Developed using:
- OpenCV for image processing and line detection
- NumPy for numerical computations
- Matplotlib for plot reconstruction
- pytesseract for OCR (optional)

---

**Status**: Beta - Core functionality working, accuracy optimization in progress

**Last Updated**: 2026-04-25