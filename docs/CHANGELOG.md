# Changelog

## Beta 1.5.0 — Scale Detector (2026-04-28)

### Added

- **Hierarchical log scale detection** (`scale_detector.py`): 4-level classifier that inspects grid-line and tick spacing patterns to determine whether an axis uses a logarithmic or linear scale. Prevents the OCR superscript fix from incorrectly converting linear values (e.g., 105 → 10⁵).
- **X→Y staged axis evaluation**: axes are now evaluated in order (X-axes first, then Y-axes) with cross-axis signal propagation for loglog chart detection.
- **`--workers N`** flag on `validate_by_type.py` for parallel validation (~3× speedup with `--workers 4`).

### Changed

- **OCR is now mandatory for baseline evaluation.** Non-OCR mode produces synthetic tick values in arbitrary units and must not be used to judge extraction quality. All validation commands should use `--use-ocr`.
- **README aligned** with current OCR baselines and validation workflow.

### Fixed

- `_fix_log_superscript_ocr` was defined but never wired into the calibration pipeline. It is now gated behind the scale detector and only activates when visual evidence supports a log scale.

### Log Detection Accuracy (v1, per-axis)

| Type | Recall | Method |
|------|--------|--------|
| log_y | 100.0% | Grid spacing |
| log_x | 41.9% | Windowed geometric |
| loglog | 12.1% | Mixed (dense minor grid) |
| **Log overall** | **44.3%** | — |
| Linear false-positive | 0.7% | — |

---

## Beta 1.4.0 — Policy Ensemble & Meta Isolation (2026-04-27)

### Added

- **Chart type guesser** (`chart_type_guesser.py`): lightweight image features → softmax type probabilities.
- **Policy router** (`policy_router.py`): type probabilities → `ExtractionPolicy` via weighted ensemble for preprocessing, color separation, density handling, and OCR parameters.
- **LLM policy router** (`llm_policy_router.py`): vision-LLM fallback for ambiguous chart classification (opt-in via `--use-llm`).
- **Meta isolation boundary**: ground-truth metadata is no longer passed into the extraction pipeline. Extraction now runs blind; meta is used only for evaluation.
- **RANSAC robust regression** for axis calibration: custom pure-NumPy implementation with pixel-grounded threshold and OLS fallback.
- Four algorithm-level optimizations: Zhang-Suen thinning, OCR crop preprocessing, HSV 3D clustering fallback, rotation detection + correction.

### Changed

- Validation now uses per-image `*_meta.json` files instead of a single `_meta.json` per directory.

---

## Beta 1.3.0 — Multi-Series & Dual-Y Hardening (2026-04-26)

### Added

- Deterministic color cluster ordering (seeded k-means + KMEANS_PP_CENTERS).
- Meta-aware dual-axis assignment for two-series charts.
- Series merge candidate self-selection for multi-series charts.
- Permutation search for series matching in evaluation.
- 2D nearest-neighbor matching for scatter evaluation.

### Changed

- Dual-Y assignment now validates per-series axis fit quality before assignment.
- Log-axis scatter misclassification fixed: log charts no longer fall into scatter centroid extraction.

---

## Beta 1.2.0 — Endpoint Calibration (2026-04-26)

### Added

- Meta endpoint calibration: plot-area endpoints used as anchors when meta axis min/max are available.
- Diagnostic layer: axis count, tick count, calibration residual, plot bounds exposed via `--debug`.

### Changed

- Deterministic color cluster ordering via seeded OpenCV k-means.

---

## Beta 1.1.0 — Core Fixes (2026-04-25)

### Fixed

- y_right axis detection: edge spine check now takes priority over tick pattern matching.
- Series-to-Y-axis assignment: dual-Y detection now validates different data ranges before assigning.
- Crash in validator CSV writer when extraction throws exceptions.

---

## Beta 1.0.0 — Initial Release (2026-04-24)

### Features

- 10 supported chart types: simple_linear, log_y, log_x, loglog, inverted_y, dual_y, scatter, multi_series, no_grid, dense.
- 5-stage extraction pipeline: image loading → axis detection → calibration → data extraction → rebuild/SSIM.
- OCR tick label reading via Tesseract (opt-in).
- Hough-based axis and tick detection.
- Hue-based color separation for multi-series charts.
- Per-column median extraction for line charts.
- Connected-component centroid extraction for scatter charts.
- Pure-NumPy SSIM implementation for validation.
- Test data generator (30 samples per type).
- `validate_by_type.py` for per-type accuracy validation.
