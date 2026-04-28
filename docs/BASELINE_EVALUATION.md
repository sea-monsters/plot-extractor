# Baseline Evaluation

> **CRITICAL: Baselines recorded before 2026-04-28 are INVALID due to meta-information leakage.**
>
> The old pipeline passed `_meta.json` ground truth directly into `extract_from_image()`, which used meta-supplied axis ranges, series counts, and data values to artificially boost pass rates. See [META_ISOLATION_BOUNDARY.md](./META_ISOLATION_BOUNDARY.md) for the full audit and the hard boundary rule.
>
> **Old headline numbers (pre-2026-04-28):** v1 92.9%, v2 75.6%, v3 35.8%, v4 44.1% — all invalid.
> **True headline numbers (meta-isolated, post-2026-04-28):** see True Baselines section below.

Current project phase: beta.

This document records validation baselines. The detailed CSV reports are generated locally and are not committed by default because repository rules ignore generated CSV outputs.

## Evaluation Commands

```powershell
python tests\validate_by_type.py
python tests\validate_by_type.py --data-dir test_data_v2
python tests\validate_by_type.py --data-dir test_data_v3
python tests\validate_by_type.py --data-dir test_data_v4 --v4-special
```

v4 uses a special supported-domain evaluator because the set contains single charts, combo charts, multi-subplot charts, unsupported chart types, and partial crops. The v4 headline score below is therefore the supported-domain score, not the full 500-image mixed-scope score.

## True Baselines (Meta-Isolated, Post-2026-04-28)

### V1 Result (2026-04-28) — Pre-RANSAC

| Type | Pass | Rate | AvgErr | MaxErr | Top1 | Top2 |
|------|------|------|--------|--------|------|------|
| dense | 0/31 | 0.0% | 1.3816 | 2.6749 | 25.8% | 32.3% |
| dual_y | 0/31 | 0.0% | 0.4680 | 1.0421 | 0.0% | 6.5% |
| inverted_y | 0/31 | 0.0% | 0.5646 | 1.0579 | 0.0% | 0.0% |
| log_x | 0/31 | 0.0% | 0.3015 | 0.8258 | 100.0% | 100.0% |
| log_y | 0/31 | 0.0% | 0.2819 | 1.3135 | 96.8% | 96.8% |
| loglog | 5/31 | 16.1% | 0.0961 | 0.2537 | 0.0% | 100.0% |
| multi_series | 0/31 | 0.0% | 0.3531 | 0.5241 | 87.1% | 87.1% |
| no_grid | 0/31 | 0.0% | 0.3508 | 0.7585 | 0.0% | 93.5% |
| scatter | 0/31 | 0.0% | 0.5886 | 2.0883 | 61.3% | 77.4% |
| simple_linear | 0/31 | 0.0% | 0.5317 | 0.9458 | 45.2% | 64.5% |
| **TOTAL** | **5/310** | **1.6%** | — | — | **41.3%** | **64.8%** |

### V1 Result (2026-04-28) — With RANSAC + Relaxed Plausibility + OCR

Scatteract-style RANSAC robust regression implemented. See [ARCHITECTURAL_CHANGES_IMPL.md](./ARCHITECTURAL_CHANGES_IMPL.md) for implementation details.

| Type | Pass | Rate | AvgErr | MaxErr | Top1 | Top2 | Delta |
|------|------|------|--------|--------|------|------|-------|
| dense | 5/31 | 16.1% | 165205439.3 | 556843147.2 | 25.8% | 32.3% | +16.1pp |
| dual_y | 0/31 | 0.0% | 0.3061 | 1.1762 | 0.0% | 6.5% | 0.0pp |
| inverted_y | 28/31 | 90.3% | 0.0460 | 0.8767 | 0.0% | 0.0% | **+90.3pp** |
| log_x | 0/31 | 0.0% | 1.6135 | 20.7868 | 100.0% | 100.0% | 0.0pp |
| log_y | 1/31 | 3.2% | 2.5170 | 44.9613 | 96.8% | 96.8% | +3.2pp |
| loglog | 5/31 | 16.1% | 0.1030 | 0.3771 | 0.0% | 100.0% | 0.0pp |
| multi_series | 0/31 | 0.0% | 0.3499 | 2.5093 | 87.1% | 87.1% | 0.0pp |
| no_grid | 22/31 | 71.0% | 1.1334 | 12.5050 | 0.0% | 93.5% | **+71.0pp** |
| scatter | 28/31 | 90.3% | 0.0468 | 0.5222 | 61.3% | 77.4% | **+90.3pp** |
| simple_linear | 25/31 | 80.6% | 0.0953 | 0.8600 | 45.2% | 64.5% | **+80.6pp** |
| **TOTAL** | **114/310** | **36.8%** | — | — | **41.3%** | **64.8%** | **+35.2pp** |

**Key improvements**: simple_linear (+80.6pp), inverted_y (+90.3pp), scatter (+90.3pp), no_grid (+71.0pp).

**Still failing**: log_x, log_y, dual_y, multi_series. These types have correct chart-type routing (high top1/top2) but fail in calibration or data extraction. Log axes may have a coordinate-transform bug; dual_y and multi_series need separate algorithmic fixes.

### V2, V3, V4

Pending. See task outputs below.

> **Why is the true rate so low?** The extraction pipeline without ground-truth meta cannot reliably calibrate axes. Relative errors are 25%-135% because pixel-to-data mapping is arbitrary. Chart-type **routing** works well (41% top1, 65% top2), but the calibration and data-extraction stages need fundamental rebuilding. The old 92.9% was entirely fabricated by meta-supplied axis ranges and series counts.

## Invalid Pre-Fix Baselines (Pre-2026-04-28)

**The following data is preserved for historical reference only. These numbers are NOT valid baselines because the extraction pipeline received ground-truth meta.**

### Headline

| Dataset | Scope | Pass | Rate | Notes |
|---------|-------|------|------|-------|
| v1 | 10 supported chart types | 288/310 | 92.9% | **INVALID** — meta leakage |
| v2 | 10 supported chart types | 378/500 | 75.6% | **INVALID** — meta leakage |
| v3 | 10 supported chart types | 179/500 | 35.8% | **INVALID** — meta leakage |
| v4 | supported single-chart subset | 90/204 | 44.1% | **INVALID** — meta leakage |

### V1 Results (Invalid)

| Type | Pass | Rate | AvgErr | MaxErr |
|------|------|------|--------|--------|
| dense | 30/31 | 96.8% | 0.0270 | 0.2825 |
| dual_y | 26/31 | 83.9% | 0.0512 | 0.3273 |
| inverted_y | 31/31 | 100.0% | 0.0038 | 0.0085 |
| log_x | 31/31 | 100.0% | 0.0053 | 0.0123 |
| log_y | 31/31 | 100.0% | 0.0065 | 0.0154 |
| loglog | 31/31 | 100.0% | 0.0050 | 0.0103 |
| multi_series | 17/31 | 54.8% | 0.0990 | 0.5183 |
| no_grid | 29/31 | 93.5% | 0.0070 | 0.0816 |
| scatter | 31/31 | 100.0% | 0.0092 | 0.0231 |
| simple_linear | 31/31 | 100.0% | 0.0067 | 0.0200 |
| **TOTAL** | **288/310** | **92.9%** | — | — |

## V2 Results

| Type | Pass | Rate | AvgErr | MaxErr |
|------|------|------|--------|--------|
| dense | 20/50 | 40.0% | 0.1730 | 1.0000 |
| dense | 20/50 | 40.0% | 0.1730 | 1.0000 |
| dual_y | 34/50 | 68.0% | 0.1309 | 2.6321 |
| inverted_y | 40/50 | 80.0% | 0.0741 | 1.0000 |
| log_x | 47/50 | 94.0% | 0.0281 | 0.8657 |
| log_y | 50/50 | 100.0% | 0.0065 | 0.0286 |
| loglog | 46/50 | 92.0% | 0.1689 | 4.1160 |
| multi_series | 14/50 | 28.0% | 0.1689 | 0.5874 |
| no_grid | 43/50 | 86.0% | 0.0636 | 1.0000 |
| scatter | 47/50 | 94.0% | 0.0185 | 0.2215 |
| simple_linear | 37/50 | 74.0% | 0.0992 | 1.0000 |
| **TOTAL** | **378/500** | **75.6%** | — | — |

## V3 Results

| Type | Pass | Rate | AvgErr | MaxErr |
|------|------|------|--------|--------|
| dense | 9/50 | 18.0% | 0.5964 | 1.8399 |
| dual_y | 13/50 | 26.0% | 0.2412 | 1.8795 |
| inverted_y | 21/50 | 42.0% | 0.2799 | 1.1436 |
| log_x | 20/50 | 40.0% | 0.6315 | 6.8889 |
| log_y | 21/50 | 42.0% | 0.9628 | 4.0929 |
| loglog | 18/50 | 36.0% | 1.4203 | 4.5861 |
| multi_series | 10/50 | 20.0% | 0.2075 | 1.0569 |
| no_grid | 11/50 | 22.0% | 0.2949 | 1.0000 |
| scatter | 40/50 | 80.0% | 0.0975 | 0.7231 |
| simple_linear | 16/50 | 32.0% | 0.2312 | 0.7438 |
| **TOTAL** | **179/500** | **35.8%** | — | — |

## V4 Supported-Domain Results

Scope accounting:

| Scope | Count |
|-------|------:|
| supported / in-scope | 204 |
| out-of-scope | 296 |
| total | 500 |

Supported-domain score:

| Type | Pass | Rate | AvgErr | MaxErr |
|------|------|------|--------|--------|
| dense | 6/18 | 33.3% | 0.3694 | 1.0000 |
| dual_y | 8/23 | 34.8% | 0.2823 | 1.0000 |
| inverted_y | 7/18 | 38.9% | 0.2669 | 1.0000 |
| log_x | 12/18 | 66.7% | 0.9575 | 14.4071 |
| log_y | 10/24 | 41.7% | 0.4610 | 1.5647 |
| loglog | 9/20 | 45.0% | 0.9671 | 4.0887 |
| multi_series | 6/28 | 21.4% | 0.3396 | 1.1519 |
| no_grid | 10/17 | 58.8% | 0.1780 | 1.0000 |
| scatter | 16/20 | 80.0% | 0.1000 | 1.0000 |
| simple_linear | 6/18 | 33.3% | 0.4597 | 1.0000 |
| **SUPPORTED TOTAL** | **90/204** | **44.1%** | — | — |

## Current Interpretation

The controlled v1 set is mostly stable, but v2 and v3 show that style variation and scan-like degradation remain hard. v4 confirms that real-world pressure is not only a multi-series issue: axis detection and calibration under geometric and visual degradation are broad bottlenecks across chart types.

The strongest current supported area is scatter extraction. The weakest supported areas are multi-series, dense curves, and degraded simple/no-grid line charts where axes or foreground masks are unstable.

## Optimization History

### Threshold-Tuning Phase (2026-04-26 to 2026-04-27)

**7 optimization attempts, all reverted due to regressions on v2/v3:**

1. **Log axis integer power fallback** — stretched calibration range beyond actual data
2. **Grayscale intensity-weighted centroid** — calibration errors dominate, not extraction precision
3. **Multi_series layered extraction fallback** — mixed different-color series on full mask
4. **Median blur preprocessing** — catastrophic (288→7), broke OCR and axis detection
5. **Morphological opening on mask** — eroded thin oscillating lines in dense charts
6. **Small CC removal** — net negative, removed valid thin features
7. **Dense curve detection** — v3 scatter regression, classification fragile under noise

**Key finding**: Threshold/filter changes cannot fix the root causes:
- Dense: thick oscillating lines → need **thinning** (algorithm change)
- v3 rotation: degrades axis/tick reading → need **rotation correction** (pipeline stage)
- Calibration: OCR errors → need **OCR-specific preprocessing** (separate from general denoising)
- Multi_series: hue-only fails → need **full HSV clustering** (algorithm change)

### Implement Plan Review Findings (2026-04-26)

Critical corrections identified while auditing code and dependencies:

1. **Sauvola in core OpenCV is incorrect**
   - Current plan wording implied Sauvola baseline, but with existing dependency (`opencv-python-headless`) baseline should be adaptive threshold.
2. **Thinning needs contrib-aware gating**
   - `cv2.ximgproc.thinning` requires opencv-contrib; must provide capability gate and fallback path.
3. **Pillow usage should be explicit**
   - If rotation helper imports PIL directly, dependency should be treated explicitly rather than assumed transitively.
4. **Rotation coordinate strategy should be simplified first**
   - Initial implementation can run fully in rotated coordinates by rotating both `image` and `raw_image`, avoiding mixed-coordinate complexity.
5. **HSV fallback trigger must use quality checks**
   - Triggering only on hue K=1 misses other failure cases; use expected count + cluster quality + x-coverage gates.

### Algorithm-Level Changes (Updated Next Phase)

See [ARCHITECTURAL_CHANGES_IMPL.md](./ARCHITECTURAL_CHANGES_IMPL.md) for detailed implementation guidance.

**Execution order (risk-increasing):**

1. **Full HSV clustering (quality-gated fallback)** — low risk, isolated to multi_series path
2. **OCR-specific preprocessing (OpenCV core baseline)** — medium risk, improves calibration path
3. **Zhang-Suen thinning (contrib-gated + fallback)** — medium risk, targeted to dense path
4. **Rotation detection + correction** — high risk, broad impact on v3 robustness

Each change requires validation against v1-v4 (v4 in-scope) with no regressions before commit.

---

## Algorithm-Level Changes Implemented (2026-04-27)

### Change 1: HSV Fallback with Quality Gates

**File**: `plot_extractor/core/data_extractor.py`
**Function**: `_separate_series_by_color`

Added `meta` parameter for expected series count. After hue-only clustering, checks quality gates (`K < expected`, poor compactness, `min_hue_dist < 15`, x-coverage failure). If triggered, runs normalized H+S+V 3D k-means with `K = expected_series_count` and compactness-based selection.

**Validation**: v1 multi_series stable (17/31), no regression on single-series types.

### Change 2: OCR Crop Preprocessing

**File**: `plot_extractor/core/ocr_reader.py`
**Function**: `_preprocess_tick_crop`, `read_tick_label`

Per-crop pipeline: grayscale → 2-3x upscale (based on crop size) → `cv2.medianBlur(gray, 3)` → `cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block_size, 10)` with odd `block_size` and `cv2.error` fallback to raw crop.

**Validation**: v1 stable, no calibration regression on any type.

### Change 3: Zhang-Suen Thinning with Contrib Gate

**File**: `plot_extractor/core/data_extractor.py`
**Function**: `_apply_thinning`

Tries `cv2.ximgproc.thinning` first (contrib-gated via `hasattr(cv2, "ximgproc")`), falls back to pure NumPy vectorized Zhang-Suen with two subiterations per iteration. Applied only in dense single-color path when `fg_cols > w * 0.7`.

**Validation**: v1 dense stable (30/31), no scatter regression.

### Change 4: Rotation Detection + Correction

**Files**: `plot_extractor/core/axis_detector.py`, `plot_extractor/core/image_loader.py`, `plot_extractor/main.py`

Strict rotation estimator:
- Only lines ≥ 40% of image dimension
- Only lines near image edges (top/bottom 15% for horizontal, left/right 15% for vertical)
- Median aggregation of raw line angles
- Both horizontal and vertical estimates must agree within 3°
- Only corrects when `|angle| >= 0.5°`

`rotate_image()` in `image_loader.py` uses `cv2.getRotationMatrix2D` + `cv2.warpAffine` with `BORDER_CONSTANT` and white fill, computing expanded output size to avoid clipping.

Integration in `main.py` rotates both `image` and `raw_image` before axis detection so downstream runs entirely in rotated coordinates.

**Validation**: v1 simple_linear 31/31 (no false-positive regression). v2-v4 baseline collected.

### Bug Fix: Validator CSV Crash

**File**: `tests/validate_by_type.py`

Added `extrasaction="ignore"` to the CSV DictWriter so exception rows containing an `"error"` field no longer crash the report writer.

### Lint Gate

`pylint --fail-under=9` → **9.82/10** ✅


