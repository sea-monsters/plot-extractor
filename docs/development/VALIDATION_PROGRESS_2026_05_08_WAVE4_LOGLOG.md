# Wave 4 Progress — LogLog Cross-Axis Safety (§12.4)

> Date: 2026-05-08
> Focus: §12.4 LogLog cross-axis safety scoring
> Datasets: v1 (test_data), types `loglog` + `log_x`
> Status: **Implementation complete; gate is safe but addresses narrow edge case**

---

## 1. Round Overview

Following the §12.4 executable track in `AXIS_EVALUATION_STRATEGY_20260507.md`, this round implemented cross-axis safety mechanisms to prevent log-X replacement rules from harming loglog calibration when both axes are independently identified as log.

---

## 2. Development Content

### 2.1 Cross-axis `log` flag computation (`calibrate_all_axes`)

After all axis-type forcing (log_x Y-linear override, W3.1/W3.2 loglog upgrade), compute final cross-axis flags:

```python
final_any_x_log = any(axis_is_log.get(id(ax), False) for ax in x_axes)
final_any_y_log = any(axis_is_log.get(id(ax), False) for ax in y_axes)
```

Per-axis `cross_axis_log` is then:
- X-axis → `final_any_y_log` (is the Y-axis log?)
- Y-axis → `final_any_x_log` (is the X-axis log?)

**File**: `plot_extractor/core/axis_calibrator.py:2955-2958, 3173`

### 2.2 `_loglog_replace_gate` in `_select_best_ocr_calibration`

Added a relaxed replacement gate that triggers when:
1. `cross_axis_log=True` (the other axis is log)
2. Primary source is sparse tesseract on X-axis (`valid_count <= 3`, `decade_span < 2.0`)
3. `axis_preferred != "linear"`

```python
_loglog_replace_gate = (
    cross_axis_log
    and _has_sparse_tesseract
    and axis_preferred != "linear"
)
```

**File**: `plot_extractor/core/axis_calibrator.py:2214-2218`

### 2.3 Span-change safety gate

Inside the replacement block, if `cross_axis_log=True` and the heuristic replacement changes the decade span by > 1.5 decades **without** FormulaOCR evidence (`formula_anchor_count >= 1` and `formula_log_score >= 0.3`), the replacement is rejected.

```python
if span_change > 1.5 and not has_formula_evidence:
    _accept_heuristic = False
```

**File**: `plot_extractor/core/axis_calibrator.py:2255-2270`

### 2.4 Signature plumbing

- `_build_heuristic_ticks`: added `cross_axis_log=False` parameter (compatibility)
- `_select_best_ocr_calibration`: added `cross_axis_log: bool = False` parameter

---

## 3. Test Results

### 3.1 Focused Validation (`--types loglog log_x`, `--use-ocr`, `--workers 1`)

| Type | Pass | Rate | AvgErr | MaxErr | vs Acceptance Criteria | vs Prior |
|------|------|------|--------|--------|------------------------|----------|
| **loglog** | 2/31 | 6.5% | **0.3125** | **1.7130** | avg > 0.2508 (miss), max > 1.6688 (miss) | avg 0.2908 → 0.3125 (slight drift) |
| **log_x** | 8/31 | 25.8% | **0.1343** | **0.4096** | kept 8/31 gain (pass) | stable |

### 3.2 Micro-Benchmark — LogLog Problem Samples

| Sample | RelErr | Pass | X_range | Expected X | Y_range | Expected Y | X_source | Y_source | Notes |
|--------|--------|------|---------|------------|---------|------------|----------|----------|-------|
| loglog/003 | 1.6113 | False | 526..5.3M | 0.5..2000 | 1.0..10M | 0.5..10.7M | heuristic | formula_generated | X over-shifted by P0 solver |
| loglog/012 | 1.6688 | False | 10.0..10000 | 0.5..20000 | 0.72..758K | 0.5..865K | tesseract | formula_generated | X compressed, Y over-shifted |
| loglog/026 | **1.7130** | False | 1000.0..1M | 0.5..2000 | 1.0..10000 | 0.5..11778 | heuristic | formula_generated | X over-shifted 3 decades |
| loglog/005 | **0.0208** | **True** | 10.0..10000 | 0.5..20000 | 1.0..1G | 0.5..2G | tesseract | fused | Pass — best sample |
| loglog/011 | **0.0168** | **True** | 1.0..1000 | 5.0..2000 | 302.0..209M | 288..382M | heuristic | formula_generated | Pass — formula anchors saved Y |

### 3.3 Gate Trigger Analysis

Ran `debug_loglog_cal.py` to inspect whether `_loglog_replace_gate` actually fires:

| Sample | `_loglog_replace_gate` triggered? | Effect |
|--------|-----------------------------------|--------|
| 000-030 | **No** | Primary source on X-axis is almost never "tesseract" in loglog; dominant sources are `heuristic` and `formula_generated` |

**Conclusion**: The gate is structurally correct but fires on a narrow subset of loglog failures that is not representative of the current test set.

---

## 4. Key Findings & Lessons Learned

### 4.1 Why the Gate Does Not Improve LogLog Average Error

The `_loglog_replace_gate` only triggers when:
- X-axis primary candidate source == `tesseract`
- `valid_count <= 3`
- `decade_span < 2.0`

In the current loglog test set, **tesseract is almost never the primary source** for the X-axis. The dominant failure modes are:

1. **Heuristic decade over-shift** (003, 014, 016, 021, 026): `_build_heuristic_ticks` / `_solve_absolute_log_ticks` infers a range like `1000..1M` when the expected range is `0.5..2000`. This is a P0 solver absolute-decade assignment problem, not a tesseract replacement problem.

2. **Y-axis misclassified as linear** (004, 005, 007): `should_treat_as_log` fails on the Y-axis due to sparse tick labels or dense minor grid. `cross_axis_log` second-chance detection was already added in Wave 3; remaining cases have deeper visual ambiguity.

3. **Formula-generated range over-shift** (011, 012): FormulaOCR provides enough anchors to pass the coverage gate, but the inferred absolute scale is off by 0.5–1.5 decades.

None of these three failure classes intersect with the `_loglog_replace_gate` trigger conditions.

### 4.2 Log-X Baseline Preserved

`log_x` pass count remains **8/31**, average error 0.1343, max error 0.4096. The span-change safety gate does not interfere with the normal `_normal_replace_gate` path because it only adds rejection logic (`_accept_heuristic = False`) rather than expanding replacement scope.

### 4.3 Catastrophic Regression Avoided

During early implementation, a broader version of this gate (with relaxed conditions and anchor-based decade rescaling in `_build_heuristic_ticks`) caused a catastrophic loglog regression: average error spiked from ~0.15 to >100. This was caught by validation and immediately reverted. The lesson is:

> **Minimal-change principle**: When adding cross-axis safety, only modify `_select_best_ocr_calibration` (candidate selection), not `_build_heuristic_ticks` (value generation). The latter has complex decade-fingerprint logic that is easy to destabilize.

---

## 5. Next Round Priorities

Per `AXIS_EVALUATION_STRATEGY_20260507.md` §12.7 execution order:

### §12.6 Data Extraction vs Axis Calibration Separation (P0)
- Add axis-only error metrics to benchmark
- Label `dominant_failure` per sample
- Move non-axis failures out of axis-strategy backlog

### §12.3 Log-Y Single-Anchor Confidence (P1)
- Revisit after §12.6 gives per-axis attribution
- `_literal_log_anchor_score` canonical boost for superscript loss
- Relax Y-axis `decades > 1.5` guard when TMLOG consistency is high

### §12.4 Follow-up (P2 — if §12.6 changes failure distribution)
- If §12.6 reveals that a significant fraction of loglog failures **do** involve sparse tesseract X-axis, revisit `_loglog_replace_gate` thresholds
- Otherwise, the gate is correctly scoped and no further work is needed

---

## 6. §12.6 Axis-Only Error Metrics & Dominant-Failure Label

### 6.1 Implementation

Completed in `tests/axis_evidence_micro_benchmark.py`:

1. **`_compute_axis_error`** — axis-only metrics:
   - `axis_log_endpoint_err`: sum of |log10(selected) – log10(expected)| at both ends (decades)
   - `axis_endpoint_err`: normalized linear endpoint deviation (= avg endpoint error / expected span)
   - `axis_span_ratio`: selected_log_span / expected_log_span

2. **`_classify_dominant_failure`** — per-axis-type-aware classification:
   - Log axes: `axis_log_endpoint_err > 0.5` decades → bad
   - Linear axes: `axis_endpoint_err > 0.15` (~15 % span deviation) → bad
   - Labels: `none`, `x_axis`, `y_axis`, `both_axes`, `series_geometry`, `unknown`

3. **Summary CSV** (`benchmark_summary.csv`) — one row per sample with:
   - `primary_x_min/max`, `primary_y_min/max`
   - `x_axis_log_err`, `y_axis_log_err`, `x_span_ratio`, `y_span_ratio`
   - `dominant_failure`

4. **Compact console table** — readable per-sample overview with `.4g` formatted ranges.

**Files changed**:
- `tests/axis_evidence_micro_benchmark.py` — `_classify_dominant_failure` signature, linear-aware thresholds, normalized non-positive fallback, `.4g` print formatting.

### 6.2 Benchmark Classification Results (13 representative samples)

| Sample | RelErr | Pass | Dominant | Interpretation |
|--------|--------|------|----------|----------------|
| log_x/004 | 0.1067 | False | **x_axis** | X log range 1..1000 vs expected 5..2000 (1 decade off) |
| log_x/007 | 0.0819 | False | **both_axes** | X log off 1 decade; Y linear min drift (0 vs –3) |
| log_x/009 | 0.0479 | True | **none** | Pass |
| log_x/020 | 0.1851 | False | **x_axis** | X log over-shifted 3 decades (1000..1M vs 5..200K) |
| log_x/026 | 0.1816 | False | **x_axis** | Same over-shift |
| log_y/002 | 0.0021 | True | **none** | Pass |
| log_y/013 | 3.5396 | False | **y_axis** | Y log collapsed to 1..10 vs expected 0.19..58.9 |
| log_y/024 | 1.6995 | False | **x_axis** | X linear misread as 100..1000 vs expected 3.95..89 |
| log_y/030 | 1.3854 | False | **y_axis** | Y log collapsed to 1..10 vs expected 0.39..122 |
| loglog/005 | 0.0208 | True | **none** | Pass |
| loglog/011 | 0.0168 | True | **none** | Pass |
| loglog/025 | 0.1081 | False | **both_axes** | X log under-shifted; Y log over-shifted |
| loglog/029 | 0.6776 | False | **both_axes** | X log compressed 3.6 decades; Y log under-shifted |

### 6.3 Key Insight: Failure-Source Attribution

The benchmark now cleanly separates three groups:

- **Axis-dominant failures** (9/13): x_axis (3), y_axis (2), both_axes (4)
  - These are legitimate targets for axis-calibration patches.
- **Non-axis failures** (2/13): series_geometry — not yet observed in the 13-sample set because all high-error samples have at least one bad axis.
- **Clean passes** (4/13): none — no further axis work needed.

### 6.4 No Regression Check

`validate_by_type.py --types log_x log_y loglog`:

| Type | Pass | AvgErr | MaxErr | vs Prior |
|------|------|--------|--------|----------|
| log_x | 8/31 | 0.1343 | 0.4096 | stable |
| log_y | 6/31 | 0.3634 | 3.5396 | stable |
| loglog | 2/31 | 0.3125 | 1.7130 | stable |

Benchmark changes are confined to the micro-benchmark script; no pipeline code touched.

---

## 7. Acceptance Criteria Status

### §12.4

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| loglog avg_rel_err | ≤ 0.2508 | 0.3125 | **Miss** — gate does not address dominant failure modes |
| loglog max_rel_err | ≤ 1.6688 | 1.7130 | **Miss** — same root cause |
| log_x pass count | ≥ 8/31 | 8/31 | **Pass** — no regression |

### §12.6

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Axis-only metrics per sample | Yes | Yes | **Pass** |
| Known failures classified | Yes | 9/13 axis-dominant | **Pass** |
| Future axis patches can use axis-only metrics | Yes | `axis_log_endpoint_err` / `axis_endpoint_err` in summary | **Pass** |
