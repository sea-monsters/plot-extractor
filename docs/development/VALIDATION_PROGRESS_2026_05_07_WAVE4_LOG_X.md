# Wave 4 Progress — Log-X Coverage Gate & Pivot to Log-Y

> Date: 2026-05-07
> Focus: Wave 4 Chapter 12 executable tracks (§12.1 log-X span-selection, §12.5 benchmark infra)
> Datasets: v1-v4 (310 images total)
> Status: **Log-X improvements landed; pivoting to Log-Y (§12.3) and LogLog (§12.4)**

---

## 1. Round Overview

Following the Wave 4 strategy (`AXIS_EVALUATION_STRATEGY_20260507.md`), this round targeted §12.1 (log-X span-selection scoring) and §12.5 (benchmark infrastructure). After solver-weight tuning proved ineffective, a direct pipeline-level fix (coverage gate + direct log fit) was implemented.

---

## 2. Development Content

### 2.1 Coverage Gate + Direct Log Fit (`axis_calibrator.py`)

**Problem**: `_select_best_ocr_calibration` selected log ranges that excluded the majority of OCR anchors (e.g., `log_x/020` selected `1000..1e6` but anchors were `1,2,10,75`, coverage = 0%). When the coverage gate called `_build_heuristic_ticks` recursively, it got the same wrong answer because the P0 solver followed the same path.

**Fix**:
- In `_select_best_ocr_calibration`, when coverage < 0.5:
  - **Bypass `calibrate_axis`** entirely (avoids `_snap_to_power_of_ten` and `_filter_non_standard_log_anchors` destroying inferred values)
  - Call `infer_log_values_from_spacing(ticks, anchors=anchor_values)` directly for P1 inference
  - Fit with `fit_log_ransac` and construct `CalibratedAxis` directly

**Lines**: ~2479-2598 in `axis_calibrator.py`

### 2.2 P0 Coverage Score (`_solve_absolute_log_ticks`)

Added `coverage_score` (weight 0.17) to the P0 solver scoring:
```python
score = (
    tmlog_consistency * 0.25
    + span_score * 0.15
    + canonical_score * 0.10
    + alignment_score * 0.08
    + position_score * 0.15
    + literal_score * 0.10
    + coverage_score * 0.17
)
```

This penalizes candidates whose inferred range excludes the original OCR anchors.

### 2.3 Benchmark Infrastructure (`tests/axis_evidence_micro_benchmark.py`)

Completed §12.5/§12.6 requirements:
- `--samples` comma-separated subset filtering
- Compact summary table (RelErr / Pass / X_range / Y_range / Dominant)
- Axis-only error metrics: `axis_endpoint_err`, `axis_span_ratio`, `axis_log_endpoint_err`
- `dominant_failure` label (`x_axis`, `y_axis`, `both_axes`, `none`)
- Candidate deltas: `solver_best_score`, `solver_best_delta`, `solver_runnerup_score`

---

## 3. Test Results

### 3.1 Full Validation (v1-v4, 310 images, `--use-ocr`)

| Type | Pass | Rate | vs Prior |
|------|------|------|----------|
| **Overall** | 126/310 | **40.6%** | ~stable |
| log_x | 7/31 | **22.6%** | ↑ from ~15% |
| log_y | 0/31 | **0.0%** | — disaster |
| loglog | 2/31 | **6.5%** | — |
| simple_linear | — | baseline | stable |

### 3.2 Micro-Benchmark — Log-X Problem Samples

| Sample | RelErr | Pass | Selected X_range | Expected X_range | Dominant | Notes |
|--------|--------|------|------------------|------------------|----------|-------|
| log_x/004 | 0.1067 | False | 1.0..1000 | 5.0..2000 | x_axis | Single anchor + wrong dw |
| log_x/007 | 0.0819 | False | 1.0..1000 | 5.0..2000 | x_axis | Same |
| log_x/009 | **0.0479** | **True** | 10.0..10000 | 5.0..20000 | none | Passes |
| log_x/020 | **0.7072** | False | 0.11..63.4 | 5.0..200000 | x_axis | OCR catastrophic |
| log_x/026 | **0.7114** | False | 0.11..63.4 | 5.0..200000 | x_axis | OCR catastrophic |

### 3.3 Micro-Benchmark — Log-Y Problem Samples

| Sample | RelErr | Pass | Y_range | Expected Y_range | Dominant |
|--------|--------|------|---------|------------------|----------|
| log_y/002 | 0.4012 | False | 0.284..284.2 | 0.923..291.8 | both_axes |
| log_y/013 | 0.3223 | False | 1.0..10.0 | 0.186..58.94 | both_axes |
| log_y/024 | 0.0998 | False | 1.0..10.0 | 2.307..67379.5 | y_axis |
| log_y/030 | 1.3874 | False | 1.0..10.0 | 0.387..122.4 | both_axes |

**Passed: 0/4**

---

## 4. Key Findings

### 4.1 Log-X: Two Distinct Failure Classes

**Class A — Fixable (004, 007, 009)**
- Single OCR anchor + wrong TMLOG decade width produces systematic scale offset
- P0 coverage score helped; remaining errors are sub-11% (near threshold)

**Class B — OCR-Catastrophic (020, 026)**
- OCR reads: `1, 2, 10, 75` at pixels 77, 237, 386, 538
- Expected labels: ~`10, 100, 1000, 100000` (range 5..200000, 4.5 decades)
- Coverage gate is **ineffective** because the wrong range `0.11..63.4` already fully covers the misread anchors (coverage = 1.0)
- P0 solver delta = 0.059 (best vs runner-up), extremely narrow
- **Verdict**: Known limitation under current OCR quality. P0/P1 cannot recover from fundamentally wrong anchors.

### 4.2 Log-Y: Complete Disaster at 0% — Biggest ROI

**Root cause identified**: Y-axis log labels use superscript notation (`10²`, `10³`, etc.) which tesseract systematically misreads as plain integers:
- `10²` → `"102"` → value 102
- `10¹` → `"101"` → value 101
- `10⁴` → `"104"` → value 104

**Impact on scoring**:
- `_literal_log_anchor_score` has `suspicious_superscript_read` penalty for values in [10,19] and [100,110]
- Penalty reduces literal score to 0.25 for BOTH the literal misread AND the corrected canonical candidate (e.g., 102→100)
- This gives canonical candidates **zero advantage** over literal misreads
- When P0 has <3 anchors, Y-axis guard rejects >1.5 decade spans, collapsing to default 1..10

**Failure modes**:
- **Mode A** (log_y/002): 3 anchors, P0 runs, span ~correct but absolute offset off by ~0.5 decade
- **Mode B** (log_y/013, 024, 030): 1-2 anchors, P0 returns no candidates, fallback to 1..10

---

## 5. Next Round Priorities

Per user direction: **stop adjusting log_x; proceed to §12.3 (log-Y) then §12.4 (loglog).**

### §12.3 Log-Y Disaster Fixes (P0)
1. Fix `_literal_log_anchor_score` to **boost canonical candidates** for superscript-misread anchors instead of penalizing them equally
2. Relax Y-axis `decades > 1.5` guard when TMLOG consistency is high
3. Add explicit superscript-loss candidate generation (e.g., [100,110] → also try 100, 10)

### §12.4 LogLog Cross-Axis Safety (P1)
1. Add cross-axis consistency scoring when both axes are log
2. Prevent one-axis disaster from corrupting the other

### Acceptance Criteria
- log_y pass rate > 0% (target: 20-30%)
- No regression in overall 40.6% or log_x 22.6%
