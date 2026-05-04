# Development Progress Report (2026-05-02): Log Axis + Throughput Round

## 1. Round Scope

This round focused on two goals:

1. Stabilize log-axis OCR routing (`tesseract` vs `FormulaOCR`) and tick-anchor usage.
2. Reduce throughput bottlenecks in dense log samples, especially crop-planning latency.

Validation was run with a consistent scope:

- command family: `tests/validate_by_type.py`
- types: `log_x`, `log_y`, `loglog`
- data: `test_data`, `test_data_v2`, `test_data_v3`
- config: `--use-ocr --workers 1 --max-images 4 --formula-batch-max-crops 4`

All runs are recorded in:

- `docs/development/validation_ledger_2026_05_02.jsonl`

## 2. Implemented Changes (This Round)

### 2.1 Traceability and Progress Discipline

- Added validation ledger append support in `tests/validate_by_type.py`
  - run metadata: commit, git status, diff stat, args, summary
- Added richer axis debug trace for calibration decision auditing

### 2.2 Log Calibration and Source Ownership

- Enabled `preferred_type` propagation into heuristic tick building path
- Improved candidate-source scoring trace
- Strengthened FormulaOCR candidate participation and log evidence handling
- Added conservative guards around implausible log spans in multi-hypothesis fitting

### 2.3 Tick-Guided Crop and Throughput Guards

- Added geometry fallback support in label crop planner for log-axis paths
- Added dynamic geometry probe budget tied to tick count (sub-linear growth)
- Reduced dense-axis OCR explosion by skipping full per-tick direct fallback on unsampled ticks

## 3. v1-v3 Before/After Comparison (Same Validation Scope)

Baseline timestamps (same-day earlier runs):

- `test_data`: `2026/5/2 8:38:05`
- `test_data_v2`: `2026/5/2 8:40:42`
- `test_data_v3`: `2026/5/2 9:05:26`

Post-fix timestamps:

- `test_data`: `2026/5/2 19:47:05`
- `test_data_v2`: `2026/5/2 19:49:08`
- `test_data_v3`: `2026/5/2 19:51:51`

### 3.1 `test_data` (v1)

| Type | Before | After | Delta |
|------|--------|-------|-------|
| `log_x` | `0/4`, avg `0.2203` | `1/4`, avg `0.2272` | pass +1, avg err slightly worse |
| `log_y` | `1/4`, avg `9.5931` | `1/4`, avg `21.8894` | pass flat, avg err worse |
| `loglog` | `0/4`, avg `1.4950` | `0/4`, avg `0.9174` | pass flat, avg err improved |

### 3.2 `test_data_v2` (v2)

| Type | Before | After | Delta |
|------|--------|-------|-------|
| `log_x` | `1/4`, avg `0.2488` | `1/4`, avg `0.5764` | pass flat, avg err worse |
| `log_y` | `1/4`, avg `1826.2887` | `1/4`, avg `1826.4165` | effectively unchanged |
| `loglog` | `0/4`, avg `58.1302` | `0/4`, avg `327.8727` | pass flat, avg err worse |

### 3.3 `test_data_v3` (v3)

| Type | Before | After | Delta |
|------|--------|-------|-------|
| `log_x` | `0/4`, avg `0.4227` | `0/4`, avg `0.5142` | pass flat, avg err worse |
| `log_y` | `1/4`, avg `26.5130` | `1/4`, avg `34.4127` | pass flat, avg err worse |
| `loglog` | `4/4`, avg `0.0449` | `4/4`, avg `0.0449` | stable |

## 4. Throughput Observations

Dense-log bottlenecks remain dominated by crop-planning stage, but catastrophic latency classes were reduced.

Current `report_test_data_v3.csv` / `report_test_data_v4.csv` show:

- `test_data_v3/log_x/004.png`: total ~`12.34s` (`timing_total_ms=12339.98`)
- `test_data_v4/loglog/0038.png`: total ~`26.66s` (`timing_total_ms=26663.93`)

Compared with earlier same-day dense-sample behavior (round logs), this is a substantial throughput stabilization, though still expensive.

## 5. Round Verdict

### 5.1 What Improved

- Process discipline improved:
  - reproducible run ledger
  - explicit traceability for axis calibration source decisions
- Throughput risk improved on dense samples (fewer extreme stalls)
- `v3 loglog` remained stable at `4/4` under this validation slice

### 5.2 What Did Not Improve Enough

- `v1-v3` `log_x` and `log_y` accuracy did not show broad convergence.
- `v2` still has catastrophic outliers (`log_y`/`loglog`) that dominate average error.
- Crop-planning still contributes the largest share of total latency on dense charts.

## 6. Next Priorities

1. Separate optimization tracks:
   - Track A: accuracy (`tick anchor -> absolute decade offset`)
   - Track B: throughput (cross-image batching, OCR call coalescing)
2. For accuracy, prioritize:
   - robust single-anchor exponent inference with bounded decade candidates
   - stronger constraints when Formula evidence is sparse but explicit
3. For throughput, prioritize:
   - cross-image batch queue path as default validation path
   - avoid repeated OCR on near-identical axis strips in same run

## 7. Execution Guardrails For Future Rounds (Must Repeat)

To avoid drifting into single-axis optimization, future execution must keep
throughput and accuracy as separate tracks but enforce a shared decision gate.

### 7.1 Round Start Checklist

1. Declare current round intent:
   - A: accuracy-focused
   - B: throughput-focused
   - C: mixed optimization
2. Run lint preflight first and clean lint issues:
   - CI-aligned gate: `pylint --fail-under=9`
   - Validation must not start before lint gate passes
3. Lock validation scope before coding:
   - `types` (`log_x`, `log_y`, `loglog`)
   - `data_dir`
   - `workers`
   - `formula_batch_max_crops`
4. Record baseline for both dimensions:
   - accuracy: pass-rate, avg/max `rel_err`
   - throughput: `timing_crop_ms`, `timing_formula_ms`, `timing_total_ms`

### 7.2 In-Round Strategy (Co-Optimization)

1. Apply low-cost path first (sampling/probing budget control).
2. Add conditional rescue when evidence is weak (do not hard-disable fallback).
3. Validate both axes in the same run slice; do not infer from one-axis-only wins.
4. Prefer adaptive gates over one-way pruning:
   - sparse probe by default
   - targeted re-probe only when anchor confidence is insufficient

### 7.3 Merge Gate (Go / No-Go)

Candidate change can be merged only when:

- Throughput improves or is stable on dense samples, and
- Cross-axis accuracy does not materially regress, and
- No new catastrophic latency/error class appears.

If any gate fails: rollback or tune adaptive thresholds, then re-run the same
scope. This guardrail must be repeated in each subsequent engineering round.

## 8. Throughput Checkpoint (2026-05-03)

Validation was re-run under lint-gated flow:

- lint preflight: `pylint --fail-under=9.00` (pass)
- commands:
  - `python tests/validate_by_type.py --types log_x log_y loglog --data-dir test_data_v3 --use-ocr --workers 1 --max-images 4 --formula-batch-max-crops 4`
  - `python tests/validate_by_type.py --types loglog --data-dir test_data_v4 --use-ocr --workers 1 --max-images 2 --formula-batch-max-crops 4`

### 8.1 v3 Throughput Summary (`report_test_data_v3.csv`)

| Type | N | crop avg (ms) | crop max (ms) | formula avg (ms) | total avg (ms) | total max (ms) |
|------|---|---------------|---------------|------------------|----------------|----------------|
| `log_x` | 4 | `11671.17` | `17297.08` | `0.74` | `12562.72` | `18411.99` |
| `log_y` | 4 | `9249.90` | `12725.31` | `0.63` | `9735.60` | `13337.25` |
| `loglog` | 4 | `9102.01` | `13807.68` | `0.83` | `9537.88` | `14324.57` |

### 8.2 v4 Throughput Summary (`report_test_data_v4.csv`)

| Type | N | crop avg (ms) | crop max (ms) | formula avg (ms) | total avg (ms) | total max (ms) |
|------|---|---------------|---------------|------------------|----------------|----------------|
| `loglog` | 2 | `16538.87` | `20007.79` | `5.33` | `17902.21` | `21432.59` |

### 8.3 Comparison Notes

- Relative to prior same-report reference sample:
  - `test_data_v3/log_x/004.png` improved from ~`12.34s` to ~`10.89s`.
  - `test_data_v4/loglog/0038.png` improved from ~`26.66s` to ~`21.43s`.
- Formula batch stage is no longer dominant in this slice; crop planning remains the primary bottleneck.
- A remaining hotspot is `test_data_v3/log_x/002.png` (`timing_crop_ms=17297.08`), which should be prioritized for next targeted profiling.

## 9. Consolidated Optimization Merge (2026-05-03, End-of-Day)

This section merges all optimization work completed today, including strategy,
implementation, and joint throughput+accuracy validation.

### 9.1 Strategy-Level Changes (Anti-Drift)

1. Enforced dual-objective loop in docs:
   - throughput and accuracy are separate tracks, but merge decisions use a shared gate.
2. Added mandatory lint-first discipline:
   - each round must pass lint before validation starts.
3. Added repeated execution checklist:
   - baseline -> minimal patch -> same-scope re-validation -> go/no-go.

### 9.2 Code-Level Changes (Today)

1. `tests/validate_by_type.py`
   - Added lint preflight gate (`pylint --fail-under=9`) before running validation.
   - Added controlled bypass flags (`--skip-lint-preflight`, `--lint-fail-under`, `--lint-report`).
2. `plot_extractor/core/ocr_reader.py`
   - Tightened geometry probe budgets on dense axes.
   - Added sampled second-pass OCR gate for planned probes.
   - Added adaptive rescue probing when numeric anchors are insufficient.
   - Skipped label-instance OCR sweep when tick-anchored numeric evidence is already sufficient.
3. `plot_extractor/core/axis_calibrator.py`
   - Skipped expensive OCR anchor planning on secondary duplicate axes (`top/right`)
     when corresponding primary direction (`bottom/left`) is present.

### 9.3 Targeted Profiling Evidence (`test_data_v3/log_x/002.png`)

- Before today’s throughput cuts:
  - `crop_planning_ms` ~ `10762`
  - total profile ~ `12.0s`
  - `pytesseract.run_and_get_output` calls: `106`
- After today’s patches:
  - `crop_planning_ms` ~ `3598`
  - total profile ~ `6.0s`
  - `pytesseract.run_and_get_output` calls: `51`

Interpretation: primary latency reduction came from eliminating duplicate-axis OCR
and suppressing redundant label-instance / second-pass probe calls.

### 9.4 Latest Joint Validation (Lint-Gated)

Commands:

- `python tests/validate_by_type.py --types log_x log_y loglog --data-dir test_data_v3 --use-ocr --workers 1 --max-images 4 --formula-batch-max-crops 4`
- `python tests/validate_by_type.py --types loglog --data-dir test_data_v4 --use-ocr --workers 1 --max-images 2 --formula-batch-max-crops 4`

Both runs passed lint preflight (`pylint --fail-under=9.00`) before execution.

#### `test_data_v3` Summary

| Type | N | pass_rate | avg_rel_err | max_rel_err | crop_avg (ms) | total_avg (ms) |
|------|---|-----------|-------------|-------------|---------------|----------------|
| `log_x` | 4 | `0.00` | `0.39` | `0.58` | `3614.57` | `4016.52` |
| `log_y` | 4 | `0.25` | `8.98` | `35.29` | `4359.84` | `4779.82` |
| `loglog` | 4 | `1.00` | `0.04` | `0.05` | `4678.38` | `5023.08` |

#### `test_data_v4` Summary

| Type | N | pass_rate | avg_rel_err | max_rel_err | crop_avg (ms) | total_avg (ms) |
|------|---|-----------|-------------|-------------|---------------|----------------|
| `loglog` | 2 | `0.00` | `0.05` | `0.05` | `5415.00` | `6129.18` |

### 9.5 Throughput Delta vs Section 8 Checkpoint

| Dataset/Type | crop avg delta | total avg delta |
|--------------|----------------|-----------------|
| `v3/log_x` | `11671 -> 3615` (~`-69%`) | `12563 -> 4017` (~`-68%`) |
| `v3/log_y` | `9250 -> 4360` (~`-53%`) | `9736 -> 4780` (~`-51%`) |
| `v3/loglog` | `9102 -> 4678` (~`-49%`) | `9538 -> 5023` (~`-47%`) |
| `v4/loglog` | `16539 -> 5415` (~`-67%`) | `17902 -> 6129` (~`-66%`) |

### 9.6 Accuracy/Throughput Joint Verdict

1. Throughput target: achieved substantial additional reduction on dense/log slices.
2. Accuracy target: no new regression introduced in the current validation scope;
   existing `log_x`/`log_y` accuracy limitations remain the next primary quality task.
3. Next round should focus on main-axis calibration robustness while preserving the
   current throughput gains and lint-first execution discipline.

### 9.7 Main-Axis Calibration Robustness Patch (2026-05-04)

This follow-up patch implements the next-round requirement from 9.6:
improve main-axis calibration robustness while keeping the same joint gate.

#### 9.7.1 Code Change

1. `plot_extractor/core/axis_calibrator.py`
   - Added a primary-axis robustness guard in `calibrate_axis`:
     - scope: only `bottom/left` axes when `preferred_type="log"`.
     - behavior: if the current best candidate is unstable and a log candidate has
       materially better residual, promote that log candidate.
   - Added trace fields for audit:
     - `candidate_primary_log_promoted`
     - `candidate_primary_log_reason`

2. `tests/test_axis_candidates.py`
   - Added unit test: primary axis can trigger the log promotion guard.
   - Added unit test: secondary axis (`top/right`) does not trigger this guard.

#### 9.7.2 Validation (Same Joint Gate as 9.4)

Lint-gated commands (unchanged scope):

- `python tests/validate_by_type.py --types log_x log_y loglog --data-dir test_data_v3 --use-ocr --workers 1 --max-images 4 --formula-batch-max-crops 4`
- `python tests/validate_by_type.py --types loglog --data-dir test_data_v4 --use-ocr --workers 1 --max-images 2 --formula-batch-max-crops 4`

Both runs passed lint preflight:
- `pylint --fail-under=9.00` (PASS)

Targeted unit tests:
- `python -m pytest tests/test_axis_candidates.py -q`
- result: `6 passed`

#### 9.7.3 Joint-Gate Result Snapshot

`test_data_v3`:

| Type | N | pass_rate | avg_rel_err | max_rel_err |
|------|---|-----------|-------------|-------------|
| `log_x` | 4 | `0.00` | `0.3803` | `0.5788` |
| `log_y` | 4 | `0.25` | `8.9847` | `35.2944` |
| `loglog` | 4 | `1.00` | `0.0449` | `0.0484` |

`test_data_v4`:

| Type | N | pass_rate | avg_rel_err | max_rel_err |
|------|---|-----------|-------------|-------------|
| `loglog` | 2 | `0.00` | `0.0525` | `0.0535` |

#### 9.7.4 Patch Verdict

1. The robustness guard is implemented with bounded scope (primary axes only),
   with explicit traceability for future audits.
2. Joint-gate discipline remains intact (same lint + same validation slice).
3. No new catastrophic latency/error class was introduced in this gate slice;
   `log_x` / `log_y` absolute accuracy remains the next optimization focus.
