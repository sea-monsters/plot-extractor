# Changelog

## Beta 1.7.0 — Wave 4 Axis Calibration Closure (2026-05-08)

### Added

- **Y-axis decades guard bypass** (`axis_calibrator.py` §12.3): single-anchor superscript-loss candidates (e.g., OCR reads "102" for 10²) can now bypass the decades > 1.5 guard on Y-axis when the candidate is a superscript-loss interpretation. log_y max error 3.54 → 2.43.
- **Sequence-level superscript-loss scoring** (`axis_calibrator.py` §12.2): detects multi-anchor monotonic exponent sequences (10, 100, 1000 → 10¹, 10², 10³) and down-weights literal anchor scores when sequence evidence is present. Structural improvement for future samples.
- **Loglog cross-axis safety gate** (`axis_calibrator.py` §12.4): `_loglog_replace_gate` rejects sparse-tesseract loglog promotions when span changes exceed 1.5 decades without formula evidence.
- **Axis evidence micro-benchmark** (`tests/axis_evidence_micro_benchmark.py` §12.5/§12.6): compact summary table with per-sample dominant failure classification (`x_axis`, `y_axis`, `both_axes`, `series_geometry`), `--samples` subset filtering, and best-vs-runner-up candidate deltas.
- **Axis-only error metrics** (`tests/axis_evidence_micro_benchmark.py` §12.6): `axis_log_endpoint_err` and `axis_endpoint_err` separate calibration errors from series extraction errors.

### Validation Baselines (Beta 1.7.0)

| Dataset | Images | Pass Rate | Delta vs 1.6.0 | Notes |
|---------|--------|-----------|----------------|-------|
| V1 (clean synthetic) | 310 | **43.2%** | −8.7pp | log_x +25.8pp, log_y +19.4pp; multi_series 0% |
| V2 (degraded) | 500 | **26.6%** | −2.6pp | Style variation degrades log axes |
| V3 (noisy) | 500 | **14.4%** | −1.0pp | Rotation + noise still challenging |
| V4 (real-world, in-scope) | 204 | **14.7%** | −1.5pp | Multi-panel, combo charts out-of-scope |

### Convergence Decision

Per §12.7 convergence rule, axis calibration strategy is **frozen** after two consecutive patches (§12.2 no improvement, §12.1 reverted) produced no net gain. Remaining failures are **input-limited** (wrong OCR reads), not scoring-limited.

### Known Issues

- **multi_series**: 0% across all datasets — routing excellent (80–98% top1) but extraction quality near-zero.
- **OCR anchor unreliability**: log_x/020 and 026 fail because OCR reads mantissas ("1, 2, 10, 75") instead of full scientific notation values (~5, ~200, ~5000, ~200000).
- **Dense curves**: oscillating-line extraction still fragile on degraded images (0–5.6%).
- **Dual-Y**: no dedicated secondary-axis handling; routed as multi_series with poor extraction.

---

## Beta 1.6.0 — Structural Decomposition & Adaptive Routing (2026-04-30)

### Added

- **Chart structural decomposition** (`layout/chart_structure.py`): CACHED-inspired (2305.04151) 4-area decomposition — plot_area, x_axis_area, y_axis_area, legend_area. Replaces hardcoded position thresholds (`h*0.8`, `w*0.2`) with structural context-based element role resolution.
- **Junction-aware skeleton path tracing** (`core/skeleton_path.py`): based on 1802.05902 sketch vectorization. Detects endpoints and branch points in thinned skeletons, traces continuous paths, and resolves branches by direction continuity. Replaces column-median extraction for dense charts.
- **Adaptive strategy selector** (`core/adaptive_strategy.py`): decision-tree policy routing from measurable image features (foreground density, saturation, CC statistics) replacing fixed `POLICY_WEIGHTS` matrix.
- **Overlapping scatter separation** (`core/scatter_overlap.py`): based on 0809.1802 pure-CV extraction. Detects abnormally large connected components and splits them using greedy shape matching.
- **55 unit tests** across 4 new test modules: `test_chart_structure.py`, `test_skeleton_path.py`, `test_scatter_overlap.py`, `test_adaptive_strategy.py`.

### Validation Baselines (Beta 1.6.0)

| Dataset | Images | Pass Rate | Notes |
|---------|--------|-----------|-------|
| V1 (clean synthetic) | 310 | **43.2%** | Post-Wave 4 re-evaluation |
| V2 (degraded) | 500 | **26.6%** | OCR baseline, no meta leakage |
| V3 (noisy) | 500 | **14.4%** | OCR baseline, no meta leakage |
| V4 (real-world, in-scope) | 204 | **14.7%** | 59.2% out-of-scope (multi-panel, combo, bar/pie) |

### Known Issues

- OCR superscript misread (10² → "102") remains critical bottleneck for log_x (25.8%) and loglog.
- HSV clustering fails on multi_series across all datasets (0%).
- Dense curve extraction degrades severely on noise (0–5.6%).

---

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
