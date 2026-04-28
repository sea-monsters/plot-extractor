# Deep Architectural Changes — Implementation Guide

> This document provides detailed implementation guidance for the algorithm-level changes identified in `optimization_attempts_complete.md`.

## Context

After 7 failed threshold-tuning attempts, analysis shows the dominant bottlenecks require algorithm-level changes:

1. **Dense charts**: Thick oscillating lines → need thinning (not detection)
2. **v3 distortions**: Rotation degrades axis/tick reliability → need rotation correction stage
3. **Calibration**: OCR errors → need OCR-specific preprocessing (separate from general denoising)
4. **Multi_series**: Color separation fails → need full HSV fallback with quality gates

## Plan Audit Corrections (2026-04-26)

Before implementation, the original plan needs several corrections to match current code and dependency constraints:

1. **Sauvola is not in OpenCV core**. OpenCV core supports adaptive thresholding; Sauvola/Niblack is in `cv2.ximgproc.niBlackThreshold` (opencv-contrib).
2. **Thinning API availability is dependency-bound**. `cv2.ximgproc.thinning` requires `opencv-contrib-python`; current requirements only pin `opencv-python-headless`.
3. **Pillow should not be treated as guaranteed**. It may be present transitively, but if used directly it should be an explicit dependency.
4. **Rotation can stay in rotated-coordinate space**. In this pipeline, if both `image` and `raw_image` are rotated before axis detection, no original-coordinate back-projection is required for extraction correctness.
5. **HSV fallback trigger must be stronger than “K=1” only**. Similar-hue or low-saturation cases can still fail with `K>1`; add cluster-quality gates.

These corrections are reflected in the updated change definitions below.

## Change 1: Zhang-Suen Thinning for Dense Charts

### Target
- **dense**: v1 96.8%, v2 40%, v3 18%
- Thinning reduces thick line masks to near-1px skeletons so vertical scan is less ambiguous

### Implementation Location
- **File**: `plot_extractor/core/data_extractor.py`
- **Function**: helper before `_extract_from_mask`

### API Notes (from Context7 / OpenCV 4.x docs)
- Signature: `cv2.ximgproc.thinning(src, dst, thinningType)`
- `thinningType`: `cv2.ximgproc.THINNING_ZHANGSUEN` (default) or `THINNING_GUOHALL`
- Input must be 8-bit single-channel image with binary blobs at 255
- Can work in-place (`dst` == `src`)

### Dependencies
- **Primary path**: OpenCV contrib (`opencv-contrib-python`) — provides `cv2.ximgproc.thinning`
- **Fallback path**: pure NumPy thinning helper if contrib unavailable
- Runtime capability gate:
  ```python
  has_ximgproc = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning")
  ```

### Integration Strategy
- Only apply in dense-like single-color path (not scatter path)
- Gate by x-coverage of foreground columns (`fg_cols > w * 0.7`)
- If thinned mask is empty or too sparse, fallback to current extraction path

```python
# Capability-gated usage
if has_ximgproc:
    thinned = cv2.ximgproc.thinning(mask, None, cv2.ximgproc.THINNING_ZHANGSUEN)
else:
    thinned = _fallback_numpy_thinning(mask)
```

### Validation Criteria
- **v1**: dense remains stable (no regression)
- **v2/v3**: dense improves materially
- **Other types**: no regression (especially scatter)

---

## Change 2: OCR-Specific Preprocessing

### Target
- Improve tick-label OCR robustness under noise/compression/light rotation

### Implementation Location
- **File**: `plot_extractor/core/ocr_reader.py`
- **Function**: new crop-level preprocessing helper used by `read_all_tick_labels`

### Baseline Preprocessing Recommendation (OpenCV Core Only)
1. Convert crop to grayscale
2. Upscale 2-3x (`cv2.INTER_CUBIC`)
3. Optional median blur (`ksize=3`) on crop only
4. `cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=10)`
   - `blockSize` must be odd (e.g. 25). `C` subtracted from weighted mean.
5. OCR with numeric whitelist and `--psm 7`

### API Notes (from Context7 / OpenCV 4.x docs)
- `cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)`
- `adaptiveMethod`: `cv2.ADAPTIVE_THRESH_GAUSSIAN_C` = Gaussian-weighted sum − C
- `blockSize`: odd integer defining neighborhood size for threshold calculation
- `C`: constant subtracted from the computed mean/weighted sum

### Optional Contrib Variant
If contrib is available, evaluate `cv2.ximgproc.niBlackThreshold` (Sauvola/Niblack family):
- `cv2.ximgproc.niBlackThreshold(src, dst, maxValue, type, blockSize, k, binarizationMethod, r=128)`
- `binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA` (enum value 1)
- This is optional second phase, not baseline

### Dependencies
- OpenCV core (`opencv-python-headless`) + pytesseract + system Tesseract
- No contrib dependency for baseline version

### Validation Criteria
- **v1**: no regression
- **v2/v3**: improved labeled tick count and reduced calibration residual outliers
- Track `diagnostics.axes[].labeled_tick_count` and `diagnostics.axes[].residual`

---

## Change 3: Full HSV Clustering for Multi_series

### Target
- **multi_series** remains weakest supported type across v1-v4

### Implementation Location
- **File**: `plot_extractor/core/data_extractor.py`
- **Function**: `_separate_series_by_color`

### API Notes (from Context7 / OpenCV 4.x docs)
- `cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)` returns `(compactness, labels, centers)`
- `data` must be `np.float32`, shape `(N, D)` where D = 3 for HSV
- `criteria`: `(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)`
- `flags`: `cv2.KMEANS_PP_CENTERS` (recommended) or `KMEANS_RANDOM_CENTERS`
- `attempts`: number of times to run with different initial labels (e.g. 5–10)
- Hue circularity: H range 0–179 (wraps); evaluate center distances using `min(|h1−h2|, 180−|h1−h2|)`

### Updated Trigger Strategy
Do not trigger fallback only when `K == 1`. Trigger full-HSV fallback when any of:
- `K < expected_series_count`
- hue-cluster compactness is poor
- inter-center hue distance below threshold
- cluster masks fail x-coverage checks

### Integration Strategy
- Keep current hue-first path for low risk
- Add quality-gated HSV fallback, then reuse existing downstream extraction

### Validation Criteria
- Improve multi_series on v2/v3 while keeping v1 stable
- No regression on single-series families

---

## Change 4: Rotation Detection + Correction

### Target
- Improve v3 rotation/noise-compound cases by reducing axis/tick slant impact

### Implementation Location
- **Axis angle estimation**: `plot_extractor/core/axis_detector.py`
- **Rotation utility**: either `core/image_loader.py` or a dedicated geometry helper
- **Pipeline hookup**: `plot_extractor/main.py::extract_from_image`

### API Notes (from Context7 / OpenCV 4.x docs)
- `cv2.getRotationMatrix2D(center, angle, scale)` where angle > 0 = counter-clockwise
- `cv2.warpAffine(src, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))`
- `borderMode=BORDER_CONSTANT` + `borderValue=(255,255,255)` produces white fill
- No `expand=True` in OpenCV; instead compute `dsize` from rotated bounding box or use PIL

### Dependencies
- If using Pillow directly, add explicit dependency declaration
- No other external dependencies required

### Coordinate Handling Strategy
For this repo, initial implementation should:
1. detect angle on preprocessed image,
2. rotate both `image` and `raw_image`,
3. run axis detection/calibration/extraction entirely on rotated images,
4. avoid mixed original/rotated coordinate logic in first phase.

### Validation Criteria
- v3 rotation bucket improves with no v1/v2 regression
- log and dual-axis families remain stable

---

## Change 5: RANSAC Robust Regression for Axis Calibration

### Research Foundation

**Paper**: *Scatteract: Automated extraction of data from scatter plots*  
**Authors**: Mathieu Cliche, David Rosenberg, Dhruv Madeka, Connie Yee  
**Venue**: ECML/PKDD 2017  
**arXiv**: [1704.06687](https://arxiv.org/abs/1704.06687)  
**Citations**: 75

**Key insight from the paper**:
> "We use optical character recognition together with **robust regression** to map from pixels to the coordinate system of the chart."
>
> "The experiment shows that Scatteract achieves an average precision of 88%, average recall of 87%, and an overall success rate of **89.2%** in data extraction from scatter plots, with **RANSAC outperforming other methods by 5.4%** and significantly improving accuracy over alternatives."

Scatteract's pipeline (Section 3):
1. Object detection → bounding boxes for tick marks, tick values, and points
2. OCR on tick value bounding boxes
3. Find closest tick mark to each tick value
4. Cluster (tick mark, tick value) pairs into X/Y axis
5. **Apply robust regression (RANSAC) to determine pixel-to-chart mapping**
6. Apply mapping to detected points

### Problem in Current Code

The original calibration used ordinary least squares (`np.polyfit`) without outlier rejection:
- A single misread OCR value (e.g., "8" read as "0") would distort the entire fit
- `_is_plausible_ocr_tick_sequence()` was overly strict, requiring 3+ ticks, arithmetic/geometric sequence patterns, and correlation > 0.3
- When OCR failed plausibility, the system fell back to `_build_heuristic_ticks()` generating arbitrary synthetic values (`0,1,2,...` or `1,10,100,...`), causing ~100% relative error

### Implementation

**Files modified**:
- `plot_extractor/utils/math_utils.py`: Added `fit_linear_ransac()` and `fit_log_ransac()`
- `plot_extractor/core/axis_calibrator.py`: Relaxed `_is_plausible_ocr_tick_sequence()`
- `plot_extractor/core/axis_candidates.py`: Updated `_solve_from_ocr()` to use RANSAC

**RANSAC design** (Scatteract-style):
- **Minimal sample**: 2 points per iteration (linear model has 2 DOF)
- **Residual threshold**: Auto-computed as 15% of median tick spacing (`_compute_ransac_threshold()`)
- **Max trials**: 100 with deterministic seed (42)
- **Refinement**: After finding best inlier set, refit with all inliers via standard least squares
- **Fallback**: If fewer than 2 inliers found, fall back to regular `fit_linear()`/`fit_log()`

**Plausibility relaxation**:
- Lowered minimum from 3 ticks to 2 ticks
- Removed strict arithmetic/geometric sequence enforcement (RANSAC handles outliers)
- Removed correlation threshold (RANSAC handles noisy correlation)
- Kept basic sanity checks: finite values, non-repeated values (>50% unique), monotonic trend (>60% consistency), extreme range detection

### Code-Level Comparison with Scatteract Reference Implementation

The reference implementation was studied from `D:/Codex_lib/code_reference/scatteract` (Bloomberg's official repo, Apache 2.0).

#### Scatteract's RANSAC (`scatter_extract.py:get_conversion`, lines 412-448)

```python
from sklearn.linear_model import RANSACRegressor

# Threshold derived from label-value spread, not pixel spacing
ransac_threshold = np.median(np.abs(np.array(labels) - np.median(labels)))**2 / 50.0

reg = RANSACRegressor(
    random_state=0,
    loss=lambda x, y: np.sum((np.abs(x-y))**2, axis=1),
    residual_threshold=ransac_threshold
)
reg.fit(
    np.reshape(positions, (len(positions), 1)),
    np.reshape(labels, (len(labels), 1))
)
```

**Key differences**:
| Aspect | Scatteract | Our Implementation |
|--------|-----------|-------------------|
| **Library** | sklearn `RANSACRegressor` | Custom pure-NumPy |
| **Threshold basis** | Label-value spread: `median(\|labels - median\|)² / 50` | Pixel spacing: `median(tick_spacing) × 0.15` |
| **Loss function** | Custom squared-L1 | Standard squared residual in pixel space |
| **Minimal sample** | 2 points (handled by sklearn) | 2 points (explicit loop) |
| **Direction** | `pixel → data_value` | `pixel → data_value` (same) |
| **Fallback** | `None` if < 2 points (cannot predict) | Regular `np.polyfit` on all points |
| **Log support** | Linear only (scatter plots only) | Separate `fit_log_ransac` for log axes |

**Assessment**: Our threshold is more physically grounded (pixel distance) and handles log axes. Scatteract's threshold is statistically motivated but assumes label values have meaningful spread (true for their synthetic data, may not hold for real charts with small ranges).

#### Scatteract's OCR Preprocessing (`tesseract.py`)

Scatteract's pipeline: grayscale → resize to height 130 → **deskew** (minAreaRect angle) → crop to content bounding box → resize → tesseract with `DigitBuilder(layout=6)`.

**Key differences**:
| Aspect | Scatteract | Our Implementation (`ocr_reader.py`) |
|--------|-----------|-------------------------------------|
| **Deskew** | Yes: `cv2.minAreaRect` + `scipy.ndimage.rotate` | No deskew |
| **Pre-threshold** | `cv2.adaptiveThreshold` + OTSU | `cv2.adaptiveThreshold` only |
| **Upscale** | Maintain aspect ratio to height 130 | 2-3× based on crop size |
| **Blur** | None | `cv2.medianBlur(gray, 3)` |
| **Negative sign hack** | `contours_len==2` → prepend `-` | Not implemented |
| **Tesseract config** | `DigitBuilder(tesseract_layout=6)` | `config='--psm 7 -c tessedit_char_whitelist=0123456789.,-eEkKmMbB%'` |

**Assessment**: Scatteract's deskew is a significant advantage for rotated or slanted tick labels. Our median blur helps with noise but does not correct rotation. The negative-sign hack is a pragmatic workaround for tesseract's weakness on single negative digits.

#### Scatteract's Tick-Label Matching (`scatter_extract.py:get_closest_ticks`, lines 320-363)

Scatteract uses **bidirectional nearest-neighbor validation**:
1. Compute distance matrix between all labels and all ticks
2. For each label, find nearest tick (`min_index_labels`)
3. For each tick, find nearest label (`min_index_ticks`)
4. Only keep pairs where `min_index_ticks[min_index] == k` (mutual nearest neighbors)

**Our approach**: OCR is applied directly at detected tick pixel positions. There is no explicit tick-to-label matching step — the OCR either reads a value at the tick position or it doesn't.

**Assessment**: Scatteract's bidirectional matching is more robust when object detection produces spurious or missing ticks/labels. Our approach is simpler but fails if tick detection and OCR are not perfectly aligned.

#### Scatteract's Axis Separation (`scatter_extract.py:split_labels_XY`, lines 366-409)

Scatteract uses **DBSCAN clustering**:
- Cluster label positions by Y-coordinate (for X-axis) and X-coordinate (for Y-axis)
- `eps = std_dev / 25.0`
- Labels assigned to the dominant cluster on one axis and excluded from the other

**Our approach**: Axes are detected by HoughLinesP (horizontal/vertical line detection). Ticks are naturally associated with their axis by geometry.

**Assessment**: Scatteract's DBSCAN is more robust for charts with unconventional axis placement or missing axis lines. Our Hough-based approach is faster but assumes standard axis geometry.

### Validation Criteria

- v1 simple_linear: target > 70% (previously ~3% without RANSAC)
- No regression on log, scatter, multi_series families
- Reduced dependency on heuristic synthetic ticks

## Implementation Status

All four changes were implemented on 2026-04-27. See `docs/BASELINE_EVALUATION.md` and `docs/OPTIMIZATION_PROGRESS.md` (Chapter 22) for detailed validation results.

| Change | Status | Files | Key Result |
|--------|--------|-------|------------|
| 1. Zhang-Suen thinning | ✅ Completed | `data_extractor.py` | v1 dense stable |
| 2. OCR preprocessing | ✅ Completed | `ocr_reader.py` | v1 stable, no regression |
| 3. HSV fallback | ✅ Completed | `data_extractor.py` | v1 multi_series stable |
| 4. Rotation detection | ✅ Completed | `axis_detector.py`, `image_loader.py`, `main.py` | v1 simple_linear 31/31 |
| 5. RANSAC calibration | ✅ Completed | `math_utils.py`, `axis_calibrator.py`, `axis_candidates.py` | v1 simple_linear 25/31 (80.6%) |

**Lint score**: 9.82/10 (threshold: 9.0)

---

## Implementation Order (Risk-Increasing)

1. **HSV fallback with quality gates** (isolated, low blast radius)
2. **OCR crop preprocessing (core OpenCV baseline)**
3. **Thinning path with contrib-capability gate + NumPy fallback**
4. **Rotation angle detection + correction**

---

## Validation Discipline (Per Change)

1. Implement isolated change
2. Run v1 sanity (no regression gate)
3. Run v2, v3, and v4 in-scope evaluator
4. Revert immediately if net regression
5. Update `docs/BASELINE_EVALUATION.md` with measured impact

Threshold tuning alone has already been exhausted; prioritize algorithmic changes only.
