# Optimization Milestone Log

## Milestone 1: Dual Y-Axis Right Axis Binding

**Date**: 2026-04-25
**Priority**: P0 (Critical)
**Issue**: dual_y 0% pass rate on 310-sample dataset

### Root Cause
- Right Y-axis series were calibrated using left Y-axis scale
- `extract_all_data` defaulted to `y_cal = left_axis` for all series
- No mechanism to assign different Y axes to different series

### Solution
Modified `plot_extractor/core/data_extractor.py`:
1. Detect both left and right Y axes (`y_left`, `y_right`)
2. In multi-color extraction, assign series to Y axes:
   - Series 0 (first color) → left Y axis
   - Series 1 (second color) → right Y axis (if exists)
3. For single-color and other cases, use default Y axis

### Code Changes
```python
# Detect dual Y axes
y_left = next((ca for ca in y_cals if ca.axis.side == "left"), None)
y_right = next((ca for ca in y_cals if ca.axis.side == "right"), None)

# Multi-color: assign Y axis per series
for series_idx, (_, cmask) in enumerate(color_series):
    if y_left and y_right and series_idx < 2:
        y_cal_for_series = y_left if series_idx == 0 else y_right
    else:
        y_cal_for_series = y_cal_default

    shifted_y = ShiftedCal(y_cal_for_series, dx=0, dy=top)
    x_d, y_d = _extract_from_mask(dilated, shifted_x, shifted_y)
```

### Baseline Results (310-sample dataset)

| Type | Pass Rate | Avg Err | Max Err |
|------|-----------|---------|---------|
| dense | 6.5% | 0.2396 | 0.4778 |
| **dual_y** | **0.0%** | **4.5310** | **105.4670** |
| inverted_y | 77.4% | 0.0348 | 0.1538 |
| log_x | 71.0% | 0.0419 | 0.0891 |
| log_y | 48.4% | 0.0833 | 0.4883 |
| loglog | 48.4% | 0.0638 | 0.3956 |
| multi_series | 0.0% | 0.6475 | 2.2402 |
| no_grid | 48.4% | 0.0633 | 0.3666 |
| scatter | 6.5% | 0.1927 | 0.4484 |
| simple_linear | 51.6% | 0.0599 | 0.2197 |

**Overall**: 111/310 (35.8%)

### Analysis

**dual_y baseline (0.0%)**:
- Milestone 1 commit message claimed 3.2% improvement, but that was on old 155-sample dataset
- On new 310-sample dataset, dual_y still fails completely (0/31)
- Series order matching heuristic (series_idx 0→left) doesn't align with ground truth naming
- Extraction precision issues persist

**Other types**:
- inverted_y, log_x stable at ≥71%
- log_y, loglog, no_grid, simple_linear at ~48-51%
- dense, scatter, multi_series at <7%

---

## Milestone 2: Simple Linear Grid Separation (FAILED)

**Date**: 2026-04-26
**Priority**: P0-2 (Critical)
**Issue**: simple_linear 51.6%, grid+anti-aliasing interference

### Attempted Solution
Enable color separation for single-color charts (K<2) by separating high-saturation (data line) vs low-saturation (grid):

```python
if K < 2 and np.sum(color_mask) >= FOREGROUND_MIN_AREA:
    high_sat_ratio = np.sum(color_mask) / len(fg_indices[0])
    if high_sat_ratio > 0.40:
        high_sat_mask = np.zeros_like(mask)
        high_sat_indices = np.where(color_mask)[0]
        rows = fg_indices[0][high_sat_indices]
        cols = fg_indices[1][high_sat_indices]
        high_sat_mask[rows, cols] = 255
        return [(image, high_sat_mask)]
```

### Results

| Type | Baseline | Attempted | Change |
|------|----------|-----------|--------|
| dense | 6.5% | 3.2% | ⚪ **Regression -3.3%** |
| dual_y | 0.0% | 0.0% | ⚪ No change |
| log_y | 48.4% | 45.2% | ⚪ **Regression -3.2%** |
| loglog | 48.4% | 54.8% | ✅ Improvement +6.4% |
| scatter | 6.5% | 9.7% | ✅ Improvement +3.2% |
| simple_linear | 51.6% | 51.6% | ⚪ No change |

**Overall**: 111/310 (35.8%) → 112/310 (36.1%) — marginal, but dense regression outweighs gains

### Failure Analysis

**Why it failed**:

1. **simple_linear unchanged**: Grid removal doesn't help when axis calibration is already accurate — the issue is not grid interference, but axis calibration precision or series-to-axis matching

2. **log_y regression**: Thin data lines have low high-sat ratios (<20%), so threshold doesn't trigger. But for borderline cases where it does trigger, too few pixels remain for reliable extraction

3. **dense regression**: High-frequency patterns need full mask for dense point detection; high-sat mask loses too much data

4. **loglog/scatter improvement**: Coincidental — loglog benefits from better single-series handling, scatter from reduced grid noise

**Root cause misdiagnosis**:
- Original hypothesis: "grid lines interfere with data extraction"
- Reality: simple_linear's bottleneck is **axis calibration precision** and **series-to-axis matching**, not grid interference
- Grid removal helps only when data lines are thick and high-saturation ratio >40%

### Decision

**Milestone 2 abandoned**. High-saturation mask approach does not reliably improve simple_linear and causes regressions elsewhere.

---

## Milestone 3: (Pending - Log Axis Fallback)

**Planned Priority**: P1 (Urgent)
**Issue**: loglog/log_x 48.4%/71.0%, log axis calibration failures

**Planned Solution**:
- Force meta fallback when tick_map length < 3
- Validate log axis sanity (avoid x_min=1, y_min=0.001 issues)
- Improve log tick recognition

---

## Validation Protocol

Each milestone must:
1. ✅ Improve target type pass rate on 310-sample dataset
2. ✅ No regression on other types (run full `validate_by_type.py`)
3. ✅ Document changes and rationale
4. ✅ Commit to git with descriptive message
5. ✅ Baseline established before optimization