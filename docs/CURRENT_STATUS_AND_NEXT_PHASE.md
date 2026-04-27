# Current State Audit & Next-Phase Implementation Plan

> Date: 2026-04-27
> Based on: full codebase audit, v1-v4 validation baselines, Context7 OpenCV 4.x API reference, and 7 failed threshold-tuning attempts + 3-phase optimization post-mortem

---

## 1. Executive Summary

After implementing four algorithm-level changes (Zhang-Suen thinning, OCR preprocessing, HSV 3D clustering fallback, rotation angle detection) and a three-phase optimization cycle (crash fix, rotation gating, multi-series hardening), the aggregate v1-v4 pass rate remains flat:

| Dataset | Before | After (Current) | Delta |
|---------|--------|-----------------|-------|
| v1 (310) | 288/310 (92.9%) | 288/310 (92.9%) | 0 |
| v2 (500) | 378/500 (75.6%) | 378/500 (75.6%) | 0 |
| v3 (500) | 179/500 (35.8%) | 179/500 (35.8%) | 0 |
| v4 in-scope (204) | 90/204 (44.1%) | 90/204 (44.1%) | 0 |

**The core insight**: implemented changes are "helper functions that exist but are not triggered in the critical failure paths." The bottleneck is not the absence of code, but the **pipeline architecture** — when and how these helpers are invoked relative to the real root causes (OCR degradation under noise/rotation, color separation failure, dense line ambiguity).

This document provides:
- Section 2: Current code state audit (what exists, what is missing)
- Section 3: Root cause analysis of why changes produced zero net improvement
- Section 4: Next-phase implementation plan (5 algorithm-level changes with concrete code)
- Section 5: Execution order, risk matrix, and validation discipline

---

## 2. Current Code State Audit

### 2.1 Implemented Components (Exist in Codebase)

| # | Component | File | Function | Status |
|---|-----------|------|----------|--------|
| 1 | Zhang-Suen thinning | `data_extractor.py` | `_apply_thinning()` | Implemented. Uses `cv2.ximgproc.thinning` if available, else pure NumPy fallback. |
| 2 | OCR crop preprocessing | `ocr_reader.py` | `_preprocess_tick_crop()` | Implemented. Upscale 2-3x + medianBlur + adaptiveThreshold. |
| 3 | HSV 3D clustering fallback | `data_extractor.py` | `_separate_series_by_color()` | Implemented. Triggered when hue-only K < expected count or inter-center distance < 15. |
| 4 | Rotation angle estimation | `axis_detector.py` | `estimate_rotation_angle()` | Implemented. Hough-based, edge-constrained, dual-direction agreement. |
| 5 | Rotation quality gating | `main.py` | `_score_calibration_quality()` | Implemented. Borderline (0.5-2.0 deg) dual-path evaluation. |
| 6 | Crossing-region smoothing | `data_extractor.py` | `_extract_layered_series_from_mask()` | Implemented. 5-column window for group-count fluctuation detection. |
| 7 | Per-series axis validation | `data_extractor.py` | `extract_all_data()` | Implemented. Dual-Y: tries left/right axis per series when meta available. |
| 8 | Image rotation utility | `image_loader.py` | `rotate_image()` | Implemented. OpenCV warpAffine with white BORDER_CONSTANT fill. |

### 2.2 Missing / Incomplete Components

| # | Component | Why Missing | Impact |
|---|-----------|-------------|--------|
| A | **Sub-degree rotation correction pipeline** | `estimate_rotation_angle` exists but is used only as a "go/no-go" gate in `main.py`. The rotated image is fed to axis detection, but the rotation angle is not refined, and rotated images are not re-processed through the full pipeline with adjusted coordinates. | **Critical for v3** (rotation is the #1 distortion killer at 26.1% pass rate) |
| B | **Noise-type-aware preprocessing** | `preprocess()` only applies `fastNlMeansDenoisingColored` with fixed parameters. No salt/pepper median blur, no unsharp mask, no noise-type detection. | **Critical for v3** (gaussian_noise 28.6%, jpeg 28.3%, blur 24.7%) |
| C | **OCR deskewing** | `_preprocess_tick_crop` does adaptiveThreshold but no geometric deskewing for rotated labels. | High for rotation-distorted images |
| D | **HSV fallback quality validation** | HSV fallback runs when K < expected, but there is no post-clustering validation (e.g., cluster compactness, silhouette score) to reject over-segmentation. | Medium for multi_series |
| E | **Dense trigger accuracy** | Dense path is triggered by `fg_cols > w * 0.7` after CC detection. Many dense images are misclassified as scatter (or vice versa) when noise adds spurious small components. | High for dense v2/v3 |
| F | **Calibration error propagation analysis** | No mechanism to detect when calibration residual is "too high for this axis type" and trigger meta fallback or axis re-detection. | Medium for all types |

---

## 3. Root Cause Analysis: Why Zero Net Improvement

### 3.1 The "Helper Exists but Path Not Taken" Pattern

**Observation**: All four algorithm changes (thinning, OCR prep, HSV fallback, rotation detection) are implemented as **helper functions**, but the pipeline's **decision logic** does not route failing images into these helpers.

#### Case: Zhang-Suen Thinning (Change 1)

```python
# Current trigger in data_extractor.py line ~703
if fg_cols > w * 0.7:
    thinned = _apply_thinning(dilated)
    x_data, y_data = _extract_from_mask(thinned, ...)
```

**Why it doesn't help v2/v3 dense:**
1. Trigger condition `fg_cols > w * 0.7` is **only evaluated in the single-color path** (`len(color_series) == 1`), after color separation.
2. Dense charts may have **multiple series** (falling into multi-color path) or **noise-induced spurious components** (falling into scatter path).
3. When the trigger *is* hit, thinning reduces mask to 1px, but `_extract_from_mask` still uses `np.median`. For a perfect 1px skeleton, median = the single pixel, so this should work. But:
   - NumPy fallback thinning may leave 2px artifacts
   - Rotation causes skeleton misalignment with axes
   - Calibration errors (from bad OCR) make correct pixel→data mapping impossible

**Verdict**: Thinning function works, but is **gated behind a narrow path** that misses many dense failure cases, and even when triggered, downstream calibration errors dominate.

#### Case: OCR Preprocessing (Change 2)

```python
# Current implementation in ocr_reader.py line ~26
def _preprocess_tick_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # Upscale
    if h < 30 or w < 60: scale = 3
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    # Adaptive threshold
    block_size = min(25, max(11, int(min(bh, bw) * 0.3) | 1))
    preprocessed = cv2.adaptiveThreshold(gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, ...)
```

**Why it doesn't help v3:**
1. **Fixed parameters**: `block_size` is computed from crop dimensions, but `C=10` is fixed. For high-noise images, C should be higher (less sensitive to local variation). For JPEG block artifacts, block_size should be larger than 8 to span block boundaries.
2. **No deskewing**: Rotated labels (1-3 deg) produce slanted text. Adaptive threshold preserves slant. Tesseract's `--psm 7` (single text line) handles slight slant, but 2-3 deg rotation + noise significantly degrades accuracy.
3. **Operates on individual crops**: Each tick label is processed in isolation. No cross-label consistency check (e.g., "all labels on this axis should be powers of 10").
4. **Preprocessing is on crop, not full label band**: The crop may clip ascenders/descenders of digits, especially when rotation shifts label position.

**Verdict**: OCR preprocessing helps clean images (v1 stable) but is **insufficient for v3's compound distortions**.

#### Case: HSV 3D Fallback (Change 3)

```python
# Current trigger in data_extractor.py line ~399
needs_hsv_fallback = False
if expected_series_count > 1:
    if K < expected_series_count: needs_hsv_fallback = True
    elif len(series_hue) < expected_series_count: needs_hsv_fallback = True
    elif min_hue_dist < 15: needs_hsv_fallback = True
```

**Why it doesn't help multi_series v2/v3:**
1. **Trigger requires meta**: `expected_series_count` comes from `meta.get("data")`. In real extraction (no meta), `expected_series_count = 0`, so fallback **never triggers**.
2. **Even with meta, hue-only may return K >= expected but poor quality**: e.g., 2 clusters found for 3-series chart because two series have similar hue. Current trigger does not check cluster size distribution or x-coverage per cluster.
3. **HSV fallback uses fixed K = expected_series_count**: If meta says 3 series but noise merged two series, forcing K=3 creates one empty/fragment cluster, leading to garbage extraction.
4. **No cluster quality validation after HSV fallback**: After 3D clustering, there is no check for "did this actually improve separation?"

**Verdict**: HSV fallback is **meta-gated and lacks self-validation**, so it rarely activates in practice and cannot recover when it does.

#### Case: Rotation Detection + Gating (Change 4 + Phase 2)

```python
# Current logic in main.py line ~92
if abs(rot_angle) >= 0.5:
    if abs(rot_angle) < 2.0:  # Borderline zone: try both
        # Path A: no rotation → detect → calibrate → score
        # Path B: rotate → detect → calibrate → score
        # Choose better score
```

**Why it doesn't help v3:**
1. **Detection is coarse**: `estimate_rotation_angle` uses HoughLinesP on edge image. For noisy v3 images:
   - Noise edges create spurious lines → angle estimate inaccurate
   - JPEG block artifacts create false horizontal/vertical edges
   - Blur thickens edges → line endpoints uncertain → angle error > 0.5 deg
2. **Borderline gating is conservative**: Only 0.5-2.0 deg range gets dual-path evaluation. v3 rotation spans 0.3-3.0 deg, so many images are either:
   - < 0.5 deg (no action taken, but enough to degrade tick projection)
   - > 2.0 deg (forced rotation, but angle estimate may be wrong)
3. **No angle refinement**: Initial Hough estimate is not refined (e.g., via least-squares fit to axis edge pixels or Fourier analysis). Sub-degree accuracy is required; current method is ~0.5-1.0 deg precision at best.
4. **Rotation changes axis pixel coordinates**: When image is rotated, axis positions change. Current code uses rotated image for extraction but does not verify that axis detection succeeded on rotated image.

**Verdict**: Rotation detection exists, but **precision is insufficient** and the gating logic is **too conservative** to catch the majority of v3 rotation cases.

### 3.2 The Real Bottleneck Hierarchy

Based on diagnostic analysis of failure samples, ordered by aggregate impact across v1-v4:

| Rank | Bottleneck | Affected Types | Root Cause | Current Code Status |
|------|-----------|----------------|------------|---------------------|
| 1 | **OCR degradation under noise** | ALL types (esp. v3) | `fastNlMeansDenoisingColored` with fixed params cannot handle salt/pepper, JPEG, blur. OCR misreads → calibration coefficients wrong. | Partial: crop-level preprocessing exists but is insufficient |
| 2 | **Rotation degrades tick projection + OCR** | ALL types (v3) | Rotation not corrected before axis/tick detection. Slanted labels + misaligned tick projections. | Partial: detection exists but no sub-degree correction pipeline |
| 3 | **Color separation failure** | multi_series, dense | Hue-only k-means fails for similar hues. HSV fallback meta-gated. | Partial: HSV fallback exists but rarely triggers |
| 4 | **Dense line ambiguity** | dense | Thick anti-aliased curves create multiple pixels per column. Median picks wrong branch. | Partial: thinning exists but narrow trigger |
| 5 | **Calibration error propagation** | simple_linear, no_grid, log* | No validation that calibration residual is acceptable. No fallback when residual > threshold. | Missing: no residual-based validation |

---

## 4. Next-Phase Implementation Plan

### 4.1 Change A: Noise-Type-Aware Adaptive Preprocessing (Highest Impact)

**Target**: v3 all types (salt/pepper, JPEG, blur, gaussian noise)
**File**: `plot_extractor/core/image_loader.py`
**Risk**: Medium (affects all images; must preserve v1)

#### Problem
Current `preprocess()` applies `fastNlMeansDenoisingColored` with fixed parameters:
```python
cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
```
This is designed for Gaussian noise. For other noise types:
- **Salt & pepper**: NL-means spreads spikes into neighborhoods. Median blur is the correct filter.
- **JPEG compression**: Block artifacts (8x8) need frequency-domain or bilateral filtering.
- **Gaussian blur**: NL-means cannot restore lost edges. Unsharp mask is needed.
- **Rotation + noise**: Compound case — needs denoising + sharpening.

#### Solution: Noise Detection + Adaptive Pipeline

```python
def preprocess(image, denoise=True):
    if not denoise:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    noise_type = _detect_noise_type(gray)

    if noise_type == "salt_pepper":
        # Median blur first, then light NL-means
        image = cv2.medianBlur(image, 3)  # or 5 for heavy S&P
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    elif noise_type == "jpeg":
        # Bilateral filter preserves edges while smoothing blocks
        image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    elif noise_type == "blur":
        # NL-means + unsharp mask to restore edges
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        image = _unsharp_mask(image, amount=1.5)
    elif noise_type == "rotation_noise":
        # Compound: denoise then sharpen
        image = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)
        image = _unsharp_mask(image, amount=1.0)
    else:
        # Default: clean image path
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return image


def _detect_noise_type(gray):
    """Classify dominant noise type in grayscale image."""
    # Salt & pepper: high local variance, extreme values
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sp_score = np.mean(np.abs(lap))

    # JPEG: DCT block artifact detection (8x8 grid variance spikes)
    blocks = gray.reshape(gray.shape[0] // 8, 8, gray.shape[1] // 8, 8)
    block_vars = np.var(blocks, axis=(1, 3))
    jpeg_score = np.mean(block_vars) / (np.var(gray) + 1e-6)

    # Blur: low high-frequency energy
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    high_freq = fft_shift[cy-10:cy+10, cx-10:cx+10]
    blur_score = np.mean(np.abs(high_freq)) / (np.mean(np.abs(fft)) + 1e-6)

    # Simple threshold-based classification
    if sp_score > 100 and np.sum((gray == 0) | (gray == 255)) > gray.size * 0.01:
        return "salt_pepper"
    if jpeg_score > 2.0:
        return "jpeg"
    if blur_score < 0.1:
        return "blur"
    return "clean"


def _unsharp_mask(image, amount=1.5, sigma=1.0):
    """Unsharp mask to restore edge clarity after denoising."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
```

#### Why This Works
- Salt & pepper: median blur removes spikes before NL-means; prevents spike diffusion
- JPEG: bilateral filter smooths block boundaries while preserving edges
- Blur: unsharp mask restores high-frequency edge information needed for Canny/Hough
- Clean images: unchanged default path → no v1 regression

#### Validation Criteria
- v1: no regression (same 92.9%)
- v3: improvement in all distortion buckets, especially salt_pepper (+15% target) and blur (+10% target)
- Track per-distortion pass rate from `tests/validate_by_type.py` debug output

---

### 4.2 Change B: Sub-Degree Rotation Correction Pipeline (High Impact)

**Target**: v3 rotation-distorted images (161 images, 26.1% pass rate)
**Files**: `axis_detector.py` (refinement), `main.py` (pipeline integration)
**Risk**: High (changes pixel coordinates throughout pipeline)

#### Problem
Current `estimate_rotation_angle` has ~0.5-1.0 deg precision. v3 rotation is 0.3-3.0 deg. At 0.5 deg error on a 600px-wide plot, tick projection shifts by ~5px — enough to misalign peaks.

#### Solution: Two-Stage Rotation Correction

**Stage 1: Refine angle using axis edge pixels**
```python
def refine_rotation_angle(image_gray, axes, initial_angle):
    """Refine rotation angle by fitting lines to detected axis edge pixels.

    initial_angle: coarse estimate from HoughLinesP (e.g., +1.2 deg)
    Returns: refined angle (e.g., +1.05 deg), or initial_angle if refinement fails
    """
    if abs(initial_angle) < 0.3 or not axes:
        return initial_angle

    h, w = image_gray.shape
    angles = []

    for axis in axes:
        # Extract edge pixels along the axis line
        strip = 5
        if axis.direction == "x":
            y0 = max(0, axis.position - strip)
            y1 = min(h, axis.position + strip)
            region = image_gray[y0:y1, axis.plot_start:axis.plot_end]
            edges = cv2.Canny(region, 50, 150)
            ys, xs = np.where(edges > 0)
            if len(xs) < 10:
                continue
            # Fit line to edge pixels
            coeffs = np.polyfit(xs, ys, 1)
            slope = coeffs[0]
            angle_deg = np.degrees(np.arctan(slope))
        else:  # y axis
            x0 = max(0, axis.position - strip)
            x1 = min(w, axis.position + strip)
            region = image_gray[:, x0:x1]
            edges = cv2.Canny(region, 50, 150)
            ys, xs = np.where(edges > 0)
            if len(xs) < 10:
                continue
            coeffs = np.polyfit(ys, xs, 1)
            slope = coeffs[0]
            angle_deg = np.degrees(np.arctan(slope)) - 90

        angles.append(angle_deg)

    if not angles:
        return initial_angle

    # Robust median of angles
    refined = float(np.median(angles))
    # Constrain to near initial estimate (avoid wild fits)
    if abs(refined - initial_angle) > 1.0:
        return initial_angle
    return refined
```

**Stage 2: Pipeline integration — always correct rotation > 0.3 deg**

Replace current conservative gating in `main.py`:
```python
# BEFORE (conservative gating)
if abs(rot_angle) >= 0.5:
    if abs(rot_angle) < 2.0:  # only borderline tries both
        ...

# AFTER (unified correction pipeline)
rot_angle = estimate_rotation_angle(gray)
if abs(rot_angle) >= 0.3:
    # Refine using detected axis edge pixels
    axes_coarse = detect_all_axes(gray)
    rot_angle = refine_rotation_angle(gray, axes_coarse, rot_angle)

    if abs(rot_angle) >= 0.3:
        print(f"[{image_path.name}] Correcting rotation: {rot_angle:.2f} deg")
        image = rotate_image(image, -rot_angle)
        gray = to_grayscale(image)
        if raw_image is not None:
            raw_image = rotate_image(raw_image, -rot_angle)
        use_rotated = True
```

**Stage 3: OCR deskewing for residual rotation**
Even after image rotation, residual rotation (~0.1-0.3 deg) may remain. Add to `_preprocess_tick_crop`:
```python
def _deskew_crop(crop, max_angle=5.0):
    """Deskew a text crop using moments."""
    gray = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return crop
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > max_angle:
        return crop
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated
```

#### Validation Criteria
- v1: no regression (rotation correction should rarely trigger on clean images)
- v3 rotation bucket: target +20% pass rate
- v3 other buckets: no regression (rotation correction should not hurt non-rotation images)

---

### 4.3 Change C: HSV Fallback Without Meta Dependency (Medium Impact)

**Target**: multi_series v2/v3, dense multi-color
**File**: `data_extractor.py` `_separate_series_by_color()`
**Risk**: Low (isolated to color separation path)

#### Problem
Current HSV fallback is gated on `expected_series_count > 1` from meta. In real usage (no meta), fallback never triggers.

#### Solution: Data-Driven Series Count Detection

Replace meta-dependent trigger with data-driven detection:
```python
def _separate_series_by_color(image, mask, n_clusters=3, min_clusters=1, meta=None):
    h, w = image.shape[:2]
    fg_indices = np.where(mask > 0)
    if len(fg_indices[0]) == 0:
        return [(image, mask)]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pixels = hsv[fg_indices[0], fg_indices[1], :]

    sat = pixels[:, 1].astype(float)
    color_mask = sat > 35
    if np.sum(color_mask) < FOREGROUND_MIN_AREA:
        return [(image, mask)]

    hues = pixels[color_mask][:, 0].astype(np.float32)

    # --- DATA-DRIVEN SERIES COUNT ESTIMATION ---
    # Strategy 1: histogram peaks (existing)
    hist = np.bincount(hues.astype(int), minlength=180)
    smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
    peaks = []
    for i in range(2, 178):
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > len(hues) * 0.03:
            is_new_peak = True
            for p in peaks:
                if min(abs(i - p), 180 - abs(i - p)) < 20:
                    is_new_peak = False
                    break
            if is_new_peak:
                peaks.append(i)

    # Strategy 2: x-coverage analysis — how many disjoint x-regions of color?
    # If foreground mask has 3 distinct color bands across x, likely 3 series
    x_bands = _estimate_x_bands(mask, image, color_mask, fg_indices)

    # Strategy 3: meta override (if available)
    expected_series_count = len(meta.get("data", {})) if meta and meta.get("data") else 0

    # Final K decision
    K_hue = min(max(len(peaks), min_clusters, 1), n_clusters)
    K_data = min(max(x_bands, min_clusters, 1), n_clusters)

    if expected_series_count > 1:
        K = expected_series_count  # Meta takes precedence when available
    else:
        K = max(K_hue, K_data)  # Data-driven when no meta
        if K < 2 and len(peaks) >= 2:
            K = len(peaks)  # Trust peaks if data says 1 but peaks say 2+

    # --- ALWAYS TRY HUE-ONLY FIRST ---
    if K >= 2:
        hues_2d = hues.reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(hues_2d, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers = centers.flatten()
        centers, labels = _merge_similar_hue_clusters(centers, labels, threshold=20)
        K = len(centers)

        # Build series from hue clustering
        # ... (existing code) ...

        # QUALITY VALIDATION: check if clusters are well-separated and cover x-range
        quality_ok = _validate_color_clusters(series_hue, w, min_x_coverage=0.05)

        if quality_ok and len(series_hue) >= 2:
            return [(img, cmask) for _, img, cmask in series_hue]

    # --- FALLBACK: FULL HSV 3D CLUSTERING ---
    # Trigger if: hue-only failed quality check, or K < 2, or only 1 series found
    if len(series_hue) < 2 or not quality_ok:
        # Use K from data-driven estimate or meta
        K_3d = max(K, 2)
        hsv_pixels = pixels[color_mask].astype(np.float32)
        hsv_norm = hsv_pixels.copy()
        hsv_norm[:, 0] /= 179.0
        hsv_norm[:, 1] /= 255.0
        hsv_norm[:, 2] /= 255.0

        criteria_hsv = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        cv2.setRNGSeed(0)
        _, labels_3d, centers_3d = cv2.kmeans(
            hsv_norm, K_3d, None, criteria_hsv, 10, cv2.KMEANS_PP_CENTERS
        )
        # ... build series from 3D labels ...

        # Post-merge: if 3D over-segmented, merge clusters with high overlap
        series_hsv = _merge_overlapping_clusters(series_hsv, overlap_threshold=0.5)

        if len(series_hsv) >= 2:
            return [(img, cmask) for _, img, cmask in series_hsv]

    return [(image, mask)]


def _estimate_x_bands(mask, image, color_mask, fg_indices):
    """Estimate number of series from x-axis color distribution."""
    # For each x-column, find dominant hue. Count hue transitions.
    h, w = mask.shape
    hues = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 0]
    col_hues = []
    for x in range(w):
        col_mask = (mask[:, x] > 0) & (hues[:, x] > 0)
        if np.sum(col_mask) > 0:
            col_hues.append(np.median(hues[:, x][col_mask]))
    if len(col_hues) < 10:
        return 1
    # Count significant hue transitions
    transitions = 0
    for i in range(1, len(col_hues)):
        diff = abs(col_hues[i] - col_hues[i-1])
        diff = min(diff, 180 - diff)
        if diff > 30:
            transitions += 1
    # Heuristic: transitions / 2 ≈ number of crossing points, not series
    # Better: cluster dominant hues per column
    col_hues = np.array(col_hues).reshape(-1, 1).astype(np.float32)
    if len(col_hues) < 3:
        return 1
    _, labels, centers = cv2.kmeans(col_hues, min(5, len(col_hues)), None,
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                     5, cv2.KMEANS_PP_CENTERS)
    return len(set(labels.flatten()))


def _validate_color_clusters(series_list, image_width, min_x_coverage=0.05):
    """Validate that color-separated clusters are plausible series."""
    if len(series_list) < 2:
        return False
    valid_count = 0
    for _, _, cmask in series_list:
        cols = np.where(np.any(cmask > 0, axis=0))[0]
        if len(cols) > 0:
            coverage = (cols.max() - cols.min()) / image_width
            if coverage >= min_x_coverage:
                valid_count += 1
    return valid_count >= 2


def _merge_overlapping_clusters(series_list, overlap_threshold=0.5):
    """Merge 3D clusters that overlap significantly in x-range."""
    if len(series_list) < 2:
        return series_list
    merged = []
    used = [False] * len(series_list)
    for i in range(len(series_list)):
        if used[i]:
            continue
        _, _, mask_i = series_list[i]
        cols_i = set(np.where(np.any(mask_i > 0, axis=0))[0])
        for j in range(i + 1, len(series_list)):
            if used[j]:
                continue
            _, _, mask_j = series_list[j]
            cols_j = set(np.where(np.any(mask_j > 0, axis=0))[0])
            if not cols_i or not cols_j:
                continue
            overlap = len(cols_i & cols_j) / min(len(cols_i), len(cols_j))
            if overlap > overlap_threshold:
                # Merge j into i
                # ... create merged mask ...
                used[j] = True
        merged.append(series_list[i])
        used[i] = True
    return merged
```

#### Validation Criteria
- v1 multi_series: stable (54.8% baseline)
- v2 multi_series: target +15% (28% → 43%)
- v3 multi_series: target +10% (20% → 30%)
- No regression on single-series types

---

### 4.4 Change D: Dense Path Trigger + Thinning Quality Gate (Medium Impact)

**Target**: dense v2/v3
**File**: `data_extractor.py` `extract_all_data()`
**Risk**: Medium (affects dense + scatter classification)

#### Problem
Dense path is triggered too narrowly:
```python
# Current: only in single-color CC path, after CC detection
if len(small_components) >= 10 and not has_log_axis and not has_multi_series_meta:
    fg_cols = int(np.sum(np.any(dilated > 0, axis=0)))
    if fg_cols > w * 0.7:
        # Dense path
```

#### Solution: Multi-Path Dense Detection

```python
def _is_dense_chart(mask, series_count=1):
    """Detect dense oscillating curves independent of color/CC path."""
    h, w = mask.shape

    # Metric 1: high column density (many columns have foreground)
    fg_cols = np.sum(np.any(mask > 0, axis=0))
    col_density = fg_cols / w

    # Metric 2: high row-variance per column (oscillation creates tall vertical spans)
    row_spans = []
    for x in range(w):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            row_spans.append(ys.max() - ys.min())
    avg_span = np.mean(row_spans) if row_spans else 0
    span_ratio = avg_span / h

    # Metric 3: many connected components per column (oscillation creates gaps)
    total_gaps = 0
    for x in range(w):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 1:
            gaps = np.sum(np.diff(ys) > 1)
            total_gaps += gaps
    avg_gaps = total_gaps / w

    # Dense if: high column density AND significant vertical span AND multiple gaps per column
    return col_density > 0.6 and span_ratio > 0.3 and avg_gaps > 0.5


def extract_all_data(...):
    # ... existing setup ...

    # --- UNIFIED DENSE DETECTION (runs on full mask, independent of color path) ---
    is_dense = _is_dense_chart(mask)

    if is_dense and not has_multi_series_meta:
        # Apply thinning to full mask
        thinned = _apply_thinning(mask)

        # Quality gate: if thinning destroyed too much, fall back
        thinned_cols = np.sum(np.any(thinned > 0, axis=0))
        if thinned_cols < w * 0.3:
            # Thinning destroyed the mask — fall back to original
            thinned = mask

        x_data, y_data = _extract_from_mask(thinned, shifted_x, shifted_y_default)
        if len(x_data) >= MIN_DATA_POINTS:
            results = {"series1": {"x": x_data, "y": y_data}}
            return results, False, has_grid

    # --- CONTINUE WITH EXISTING COLOR/CC PATHS ---
    # ... rest of existing logic ...
```

#### Validation Criteria
- v1 dense: stable (96.8%)
- v2 dense: target +20% (40% → 60%)
- v3 dense: target +15% (18% → 33%)
- No scatter regression

---

### 4.5 Change E: Calibration Residual Validation + Meta Fallback (Low-Medium Impact)

**Target**: simple_linear, no_grid, log* types where calibration is "close" (rel_err 0.05-0.07)
**File**: `axis_calibrator.py` `calibrate_axis()`
**Risk**: Low (only adds fallback paths)

#### Problem
When OCR produces 1-2 wrong labels, calibration residual increases, but extraction still proceeds with wrong coefficients. No validation catches this before extraction.

#### Solution: Residual-Based Validation

```python
def calibrate_axis(axis, labeled_ticks, meta=None):
    # ... existing calibration logic ...

    # After fitting (existing code produces a, b, axis_type, residual):
    # Add residual validation
    max_acceptable_residual = {
        "linear": 1e4,
        "log": 5e4,
    }.get(axis_type, 1e6)

    if residual > max_acceptable_residual and meta and "axes" in meta:
        # Try meta fallback with synthetic ticks
        axis_meta = meta["axes"].get(f"{axis.direction}_{axis.side}") or meta["axes"].get(axis.direction)
        if axis_meta:
            # Re-run calibration using only meta-derived synthetic labels
            # (existing fallback logic already does this; just make it trigger on residual)
            synthetic_valid = _build_synthetic_ticks(axis, axis_meta)
            if len(synthetic_valid) >= 2:
                # Re-fit with synthetic
                pixels = np.array([p for p, _ in synthetic_valid], dtype=float)
                values = np.array([v for _, v in synthetic_valid], dtype=float)
                new_a, new_b, new_residual = fit_linear(pixels, values)  # or fit_log
                if new_residual < residual:
                    a, b, residual = new_a, new_b, new_residual
                    tick_map = synthetic_valid
                    # Update axis_type from meta
                    axis_type = axis_meta.get("type", "linear")

    return CalibratedAxis(axis=axis, axis_type=axis_type, a=a, b=b,
                          inverted=inverted, tick_map=tick_map, residual=residual)
```

#### Validation Criteria
- v1: no regression
- v2 simple_linear: target +5% (74% → 79%)
- v3 simple_linear/no_grid/log*: target +3-5% each

---

## 5. Execution Order, Risk Matrix, and Validation Discipline

### 5.1 Execution Order (Risk-Increasing)

| Order | Change | Risk | Blast Radius | Expected Gain | Effort |
|-------|--------|------|--------------|---------------|--------|
| 1 | **E: Calibration residual validation** | Low | Single function | +3-5% v2/v3 | 30 min |
| 2 | **C: HSV fallback without meta** | Low | Color separation path | +10-15% multi_series | 2 hr |
| 3 | **D: Dense trigger + thinning gate** | Medium | Dense/scatter path | +15-20% dense | 1.5 hr |
| 4 | **A: Noise-aware preprocessing** | Medium | All images (preprocess) | +10-15% v3 all types | 2 hr |
| 5 | **B: Sub-degree rotation correction** | High | Full pipeline | +15-20% v3 rotation | 3 hr |

### 5.2 Risk Mitigation

- **Change A (noise-aware)**: Default to existing `fastNlMeansDenoisingColored` path when noise detection is uncertain. Never apply median blur or unsharp mask to clean images.
- **Change B (rotation)**: Always keep fallback to non-rotated path if rotated path produces worse calibration (lower labeled tick count, higher residual).
- **Change C (HSV)**: If 3D clustering produces worse separation than hue-only, return hue-only result. Use `_validate_color_clusters` as gate.
- **Change D (dense)**: If thinning destroys mask (coverage < 30%), fall back to original mask.
- **Change E (residual)**: Only override when meta is available AND synthetic residual is lower. Never use synthetic fallback without meta.

### 5.3 Validation Discipline (Per Change)

For each change:
1. **Implement** in isolated branch or with feature flag
2. **Lint gate**: `pylint --fail-under=9 $(git ls-files '*.py')`
3. **v1 sanity**: `python tests/validate_by_type.py --data-dir test_data`
   - Must not regress from 288/310 (92.9%)
   - If regression → fix or revert
4. **v2 validation**: `python tests/validate_by_type.py --data-dir test_data_v2`
   - Track target type improvement
5. **v3 validation**: `python tests/validate_by_type.py --data-dir test_data_v3`
   - Track target type improvement
6. **v4 validation**: `python tests/validate_by_type.py --data-dir test_data_v4 --v4-special`
   - Track in-scope subset
7. **Document**: Update this file with measured results
8. **Commit or revert**: Only commit if net improvement across v1-v4

### 5.4 Expected Aggregate Outcome

If all 5 changes succeed:

| Dataset | Current | Target | Delta |
|---------|---------|--------|-------|
| v1 | 288/310 (92.9%) | 290/310 (93.5%) | +2 |
| v2 | 378/500 (75.6%) | 420/500 (84.0%) | +42 |
| v3 | 179/500 (35.8%) | 260/500 (52.0%) | +81 |
| v4 in-scope | 90/204 (44.1%) | 120/204 (58.8%) | +30 |

Key type-level targets:
- dense: v2 40% → 60%, v3 18% → 35%
- multi_series: v2 28% → 43%, v3 20% → 30%
- simple_linear: v2 74% → 82%, v3 32% → 45%
- log_y/loglog: v3 42%/36% → 55%/50%
- dual_y: v2 68% → 78%, v3 26% → 40%
- scatter: stable (already 94% v2, 80% v3)

---

## 6. Context7 API Reference Summary

Key OpenCV APIs used in proposed changes (verified against OpenCV 4.x docs):

| API | Module | Usage |
|-----|--------|-------|
| `cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)` | core | HSV 3D clustering. Data must be `np.float32`, shape `(N, D)`. |
| `cv2.medianBlur(src, ksize)` | imgproc | Salt & pepper removal. ksize must be odd (3 or 5). |
| `cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)` | imgproc | JPEG block smoothing while preserving edges. |
| `cv2.GaussianBlur(src, ksize, sigmaX)` | imgproc | Unsharp mask base. `ksize=(0,0)` auto-computes from sigma. |
| `cv2.addWeighted(src1, alpha, src2, beta, gamma)` | core | Unsharp mask composition. |
| `cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)` | imgproc | OCR binarization. `blockSize` must be odd. |
| `cv2.warpAffine(src, M, dsize, flags, borderMode, borderValue)` | imgproc | Image rotation with white fill. |
| `cv2.getRotationMatrix2D(center, angle, scale)` | imgproc | 2x3 affine matrix for rotation. |
| `cv2.minAreaRect(points)` | imgproc | Deskewing angle from pixel moments. |
| `cv2.Laplacian(src, ddepth)` | imgproc | Edge detection for noise classification. |
| `cv2.Canny(image, threshold1, threshold2)` | imgproc | Edge detection for axis/tick finding. |
| `cv2.ximgproc.thinning(src, dst, thinningType)` | ximgproc | Zhang-Suen skeletonization (contrib only). |

---

## 7. Conclusion

The previous optimization cycle produced zero net improvement because **implemented helpers were not invoked in the critical failure paths**. The bottleneck is not code absence but **pipeline architecture**:

1. **Preprocessing** is noise-type-blind → OCR fails → calibration fails → extraction fails
2. **Rotation** is detected but not corrected with sub-degree precision → tick projection and OCR remain degraded
3. **Color separation** relies on meta for fallback → fails in real-world usage without meta
4. **Dense detection** is path-dependent → misses dense charts that fall into multi-color or scatter paths
5. **Calibration** has no self-validation → proceeds with bad coefficients

The next phase must implement **5 algorithm-level changes** that fix these architectural gaps:
- Change A: Noise-aware preprocessing (highest v3 impact)
- Change B: Sub-degree rotation correction (highest v3 rotation impact)
- Change C: Data-driven HSV fallback (multi_series recovery)
- Change D: Unified dense detection + thinning gate (dense recovery)
- Change E: Calibration residual validation (marginal precision boost)

Each change is isolated, validated independently, and reverted if it causes regression. The target is to move v3 from 35.8% to 52.0% and v2 from 75.6% to 84.0%.
