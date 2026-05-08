# Wave 4 Progress — Log-Y Single-Anchor Confidence (§12.3)

> Date: 2026-05-08
> Focus: §12.3 Log-Y single-anchor confidence cap
> Datasets: v1 (test_data), types `log_y` + regression check
> Status: **Acceptance criteria met**

---

## 1. Round Overview

Following the §12.7 execution order in `AXIS_EVALUATION_STRATEGY_20260507.md`, this round implemented a targeted relaxation of the Y-axis decades guard in `_solve_absolute_log_ticks` for superscript-loss OCR patterns.

---

## 2. Development Content

### 2.1 Problem Diagnosis

Micro-benchmark evidence for `log_y/013` and `log_y/030`:

| Sample | Old Y Range | Expected Y Range | Solver Candidates | Root Cause |
|--------|-------------|------------------|-------------------|------------|
| log_y/013 | 1..10 | 0.186..58.9 | 0 | Single anchor `102` (superscript-loss of 10²); Y-axis guard blocks all candidates with decades > 1.5 |
| log_y/030 | 1..10 | 0.387..122.4 | 0 | Same pattern: single anchor `102`, Y-axis guard blocks |

Both samples share the same failure chain:
1. OCR reads `102` → detected as superscript-loss candidate (100..110 range)
2. `_candidate_log_anchor_values(102)` generates candidates [102.0, 100.0]
3. For Y-axis, candidates are filtered to literal + superscript-loss only
4. Both `cand=102.0` (literal) and `cand=100.0` (10²) produce inferred ranges spanning > 1.5 decades
5. Y-axis guard: `decades > 1.5 AND reliable_anchor_count < 3` → discards all candidates
6. Solver returns None → P1 also blocked → P2 fallback: `logspace(0, 1, n)` = 1..10

### 2.2 Fix: Conditional Y-Axis Guard Bypass

In `_solve_absolute_log_ticks` (line ~1437), relaxed the Y-axis decades guard for superscript-loss candidates:

```python
_is_superscript_anchor = (
    100 <= int(round(float(anchor_value))) <= 110
    or 10 <= int(round(float(anchor_value))) <= 19
)
_is_superscript_cand = _is_superscript_loss_candidate(
    float(anchor_value), float(cand_value)
)
if (
    axis.direction == "y"
    and preferred_type == "log"
    and reliable_anchor_count < 3
    and decades > 1.5
    and not (_is_superscript_anchor and _is_superscript_cand)
):
    continue
```

**Safety**: Only the superscript-loss interpretation (e.g., 100.0 = 10²) bypasses the guard. The literal interpretation (e.g., 102.0) is still blocked. The candidate must still pass TMLOG consistency, canonical value, alignment, and span plausibility scoring.

**File**: `plot_extractor/core/axis_calibrator.py:1437-1450`

---

## 3. Test Results

### 3.1 Targeted Samples

| Sample | Old RelErr | New RelErr | Δ | Old Y Range | New Y Range | Dominant |
|--------|-----------|-----------|---|-------------|-------------|----------|
| log_y/013 | 3.5396 | **2.4274** | **-1.11** | 1..10 | 0.138..1377 | y_axis |
| log_y/024 | 1.6995 | 1.6995 | 0 | 4.96..60619 | 4.96..60619 | x_axis |
| log_y/030 | 1.3854 | **1.2000** | **-0.19** | 1..10 | 0.299..299 | series_geometry → no longer y_axis |

### 3.2 Focused Type Validation

| Type | Pass | Rate | AvgErr | MaxErr | vs Prior |
|------|------|------|--------|--------|----------|
| **log_y** | 6/31 | 19.4% | 0.3647 | **2.4274** | pass stable; max −1.11 |
| **simple_linear** | 29/31 | 93.5% | 0.0608 | 1.1495 | stable — no regression |

### 3.3 Cross-Type Regression Check

| Type | Pass | AvgErr | MaxErr | Status |
|------|------|--------|--------|--------|
| log_x | 8/31 | 0.1343 | 0.4096 | stable |
| loglog | 2/31 | 0.3125 | 1.7130 | stable |

---

## 4. Key Findings

### 4.1 Y-Axis Range Still Over-Shifts

For log_y/013, the Y range expanded from 1..10 to 0.138..1377 (4 decades), but the expected range is ~2.5 decades (0.186..58.9). The max value (1377) is ~23x the expected max (58.9). This is because TMLOG decade detection counts more decades than actually exist in the tick spacing pattern. The superscript-loss bypass allows the solver to explore wider ranges, but the TMLOG decade width estimation still produces inflated counts.

### 4.2 Dominant Failure Shift for log_y/030

log_y/030's dominant failure shifted from `y_axis` to `series_geometry`, meaning the Y-axis calibration is now close enough that series extraction error dominates. The Y range (0.299..299) is within 2.4x of the expected range (0.387..122.4), a significant improvement over the original 1..10 single-decade collapse.

### 4.3 Remaining log_y Disaster Cases

log_y/024 is an X-axis-dominant failure (X: 100..1000 vs expected 3.95..89), not a Y-axis issue. The X axis is wrongly treated as log in a secondary candidate. This is a different failure mode not addressed by §12.3.

### 4.4 Solver Diagnostics Gap

`solver_candidate_count` remains 0 in the micro-benchmark because `_build_heuristic_ticks` discards the diagnostics from `_solve_absolute_log_ticks` (variable `_diagnostics` is captured but not forwarded to the calibration debug_trace). This does not affect correctness but makes the benchmark less useful for diagnosing future solver issues.

---

## 5. Acceptance Criteria Status

### §12.3

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| One of 013/024/030 improves > 0.20 | Yes | 013 improves by 1.11 | **Pass** |
| log_y pass count ≥ 6/31 | 6/31 | 6/31 | **Pass** |
| simple_linear unchanged | 29/31 | 29/31 | **Pass** |
| No new linear→log promotion | Yes | Yes | **Pass** |

---

## 6. §12.2 Superscript-Loss Sequence Detection

### 6.1 Implementation

Added sequence-level superscript-loss evidence scoring inside `_solve_absolute_log_ticks`:

1. **`_seq_superscript_score`**: computed from all anchors before the candidate loop
   - Collects anchors in [10,19] or [100,110] range
   - If ≥ 2 such anchors exist, converts each to its implied exponent
   - Checks monotonic ordering of exponents by pixel position
   - Returns 0.5–1.0 confidence based on sequence length

2. **`_literal_log_anchor_score` enhancement**: accepts `sequence_superscript_score` parameter
   - When sequence score > 0.5, suspicious literal reads (10–19, 100–110) get score 0.10 instead of 0.25
   - This makes the superscript-loss candidate interpretation more likely to win

**Files changed**:
- `plot_extractor/core/axis_calibrator.py`: `_solve_absolute_log_ticks` (sequence scoring), `_literal_log_anchor_score` (sequence-aware literal penalty)

### 6.2 Validation

The sequence detector does NOT fire on any of the 13 benchmark samples because none have ≥ 2 anchors in the suspicious ranges forming monotonic exponent sequences. All samples have 0–1 such anchors.

| Type | Pass | AvgErr | MaxErr | vs Prior |
|------|------|--------|--------|----------|
| log_y | 6/31 | 0.3647 | 2.4274 | stable |
| loglog | 2/31 | 0.3125 | 1.7130 | stable |
| simple_linear | 29/31 | 0.0608 | 1.1495 | stable |

### 6.3 Conclusion

§12.2 is implemented as a structural improvement for future samples with multi-anchor superscript-loss patterns. It does not affect current benchmark results because the trigger condition (≥ 2 suspicious anchors in monotonic exponent order) is not met by any current test sample.

---

## 7. §12.1 Log-X Absolute Span Selection — Attempted and Reverted

### 7.1 Problem Analysis

Direct solver debugging for `log_x/020` and `log_x/026` revealed:

- **OCR anchors**: (77, 1.0), (237, 2.0), (386, 10.0), (538, 75.0) — all literal reads
- **Expected X range**: 5..200000 (~4.6 decades)
- **Selected range**: 1000..1000000 (3 decades) — from P0 solver, best candidate cand=10^5 at anchor (386, 10.0)
- **TMLOG decade width**: 115.25 pixels → total decades = 461/115.25 = 4.0
- **No candidate produces the correct 4.6-decade range** — solver produces 2.5-3.6 decades

The OCR reads "1, 2, 10, 75" do not correspond to the actual axis values (~5, ~200, ~5000, ~200000). The reads are off by 1-3 orders of magnitude. This is NOT a superscript-loss issue — the OCR is simply misreading the chart labels (possibly scientific notation like "5×10^0" → "1", "2×10^2" → "2", etc.).

### 7.2 Attempted Fix: Multi-Anchor Coverage Scoring

Added `_coverage` term to `_solve_absolute_log_ticks` candidate scoring:
- Penalises candidates where other OCR anchors fall far outside the inferred range
- Intended to shift ranking from cand=10^5 (all other anchors outside range) to candidates with better anchor coverage

**Result: REGRESSION**
- log_x/020: rel_err 0.1851 → 0.4331 (2.4x worse)
- log_x/026: rel_err 0.1816 → 0.4355 (2.4x worse)

The coverage term favoured cand=1000 with range [3.066..1000] (2.5 decades) because all other anchors (1, 2, 10) fit inside. But this range is even further from the expected 5..200000 than the original [1000..1000000].

### 7.3 Root Cause

When OCR anchors are unreliable (wrong by orders of magnitude), multi-anchor coverage is counterproductive. The coverage term amplifies the error by preferring ranges that fit the wrong anchors.

### 7.4 Reverted and Conclusion

All §12.1 scoring changes reverted. Baseline restored: log_x 8/31, avg 0.1343, max 0.4096.

**§12.1 acceptance criteria cannot be met** with current OCR quality. The OCR reads for log_x/020 and log_x/026 are fundamentally incorrect (reading mantissas instead of full values, or misreading scientific notation). No solver scoring change can fix wrong inputs — the fix would need OCR preprocessing improvements (better label cropping, scientific notation parsing).

---

## 8. Wave 4 Overall Summary

### 8.1 Completed Tasks

| § | Task | Status | Impact |
|---|------|--------|--------|
| §12.5 | Benchmark summary/subset support | **Done** (prior session) | Compact table, `--samples` filtering, candidate deltas |
| §12.6 | Axis-only error metrics | **Done** (prior session) | `dominant_failure` classification, axis-log-endpoint errors |
| §12.4 | Loglog cross-axis safety | **Done** (prior session) | Gate safe but narrow; no loglog improvement |
| §12.3 | Log-Y single-anchor confidence | **Done** | log_y max 3.54→2.43; log_y/013 improves by 1.11 |
| §12.2 | Superscript-loss sequence detection | **Done** | Structural improvement; no current benchmark impact |
| §12.1 | Log-X absolute span selection | **Attempted, reverted** | OCR anchors too unreliable; cannot meet criteria |

### 8.2 Current Validation Baselines (after all Wave 4 work)

| Type | Pass | AvgErr | MaxErr |
|------|------|--------|--------|
| log_x | 8/31 | 0.1343 | 0.4096 |
| log_y | 6/31 | 0.3647 | 2.4274 |
| loglog | 2/31 | 0.3125 | 1.7130 |
| simple_linear | 29/31 | 0.0608 | 1.1495 |

### 8.3 Code Changes Summary

| File | Changes |
|------|---------|
| `axis_calibrator.py:1442-1457` | §12.3: Y-axis decades guard bypass for superscript-loss candidates |
| `axis_calibrator.py:1430-1457` | §12.2: Sequence-level superscript-loss scoring (`_seq_superscript_score`) |
| `axis_calibrator.py:1271-1302` | §12.2: `_literal_log_anchor_score` accepts `sequence_superscript_score` parameter |

### 8.4 Recommended Next Wave

Per the convergence rule in §12.7: "If two consecutive patches improve neither axis-only metrics nor pass counts, freeze the axis calibration strategy and shift to series extraction."

§12.2 (no measurable improvement) and §12.1 (reverted) are two consecutive patches with no net improvement. The axis calibration strategy should be frozen, and the next development wave should focus on:

1. **OCR preprocessing improvements**: Better label cropping and scientific notation parsing for charts like log_x/020 and log_x/026 where OCR reads mantissas instead of full values
2. **Series extraction quality**: Many high-error samples now have acceptable axis calibration but poor series extraction (e.g., log_y/030 dominant failure shifted from y_axis to series_geometry)
3. **TMLOG decade width accuracy**: The decade detection often under/over-counts decades by 1-2, limiting solver accuracy even when anchors are correct
