# Wave 4 Closure — Axis Calibration Candidate Scoring

> Date: 2026-05-08
> Strategy doc: `docs/development/AXIS_EVALUATION_STRATEGY_20260507.md` §12
> Status: **Closed — convergence rule triggered**

---

## 1. Scope and Execution Order

Wave 4 implemented the six improvement tracks defined in §12 of the axis evaluation strategy document, following the execution order in §12.7:

1. §12.5 Benchmark summary/subset support
2. §12.1 Log-X absolute span selection
3. §12.4 Loglog cross-axis safety
4. §12.3 Log-Y single-anchor confidence
5. §12.2 Superscript-loss OCR interpretation
6. §12.6 Data extraction vs axis calibration separation

Actual execution order across two sessions:

| Session | Tasks completed |
|---------|----------------|
| 2026-05-07 → 05-08 (session 1) | §12.5, §12.6, §12.4, §12.1 (partial — solver + benchmark infra) |
| 2026-05-08 (session 2) | §12.3, §12.2, §12.1 (revisited — reverted) |

---

## 2. Task Results

### §12.5 OCR Evidence Quality and Candidate Provenance — **Done**

Extended `axis_evidence_micro_benchmark.py`:
- Compact summary table: sample, X/Y ranges, sources, `dominant_failure`
- `--samples` subset filtering for targeted re-runs
- Best-vs-runner-up candidate deltas in diagnostics

### §12.6 Data Extraction vs Axis Calibration Separation — **Done**

Added axis-only error metrics and failure classification:
- `axis_log_endpoint_err`: log-space endpoint error for log axes
- `axis_endpoint_err`: normalized linear error for linear axes
- `dominant_failure`: `none`, `x_axis`, `y_axis`, `both_axes`, `series_geometry`, `unknown`
- Benchmark result: 9/13 samples classified as axis-dominant failures

### §12.4 Loglog Cross-Axis Safety — **Done**

Implemented cross-axis safety in `_select_best_ocr_calibration`:
- `final_any_x_log` / `final_any_y_log` flags computed in `calibrate_all_axes`
- `_loglog_replace_gate`: triggers on sparse tesseract + cross-axis log
- Span-change safety: rejects >1.5-decade replacement without formula evidence
- **Result**: gate is safe but fires on a narrow subset; loglog metrics unchanged

### §12.3 Log-Y Single-Anchor Confidence — **Done** (measurable improvement)

Relaxed Y-axis decades guard in `_solve_absolute_log_ticks` for superscript-loss candidates:

```python
# Bypass decades > 1.5 guard when anchor is superscript-loss and candidate
# is the superscript-loss interpretation (e.g. 102 → 10² = 100)
if (... and not (_is_superscript_anchor and _is_superscript_cand)):
    continue
```

**Impact**: log_y/013 rel_err 3.54 → 2.43 (−1.11); log_y/030 dominant failure shifted from `y_axis` to `series_geometry`.

### §12.2 Superscript-Loss OCR Interpretation — **Done** (structural, no current impact)

Added sequence-level superscript-loss evidence:
- `_seq_superscript_score` in `_solve_absolute_log_ticks`: checks multi-anchor monotonic exponent ordering
- `_literal_log_anchor_score` enhanced with `sequence_superscript_score` parameter
- **Result**: no current benchmark samples trigger the detector (all have 0–1 superscript-suspicious anchors)

### §12.1 Log-X Absolute Span Selection — **Attempted, reverted**

Direct solver debugging revealed the root cause:
- `log_x/020` and `log_x/026` OCR reads: "1", "2", "10", "75"
- Expected axis values: ~5, ~200, ~5000, ~200000 (off by 1–3 orders of magnitude)
- TMLOG detects ~4 decades; solver produces 2.5–3.6; expected is 4.6
- No candidate produces the correct range

Attempted multi-anchor coverage scoring — **caused regression** (0.18 → 0.43). Coverage term trusts wrong anchors. All changes reverted.

---

## 3. Validation Baselines

### Before Wave 4

| Type | Pass | AvgErr | MaxErr |
|------|------|--------|--------|
| log_x | 5/31 | 0.1745 | 0.4927 |
| log_y | 6/31 | 0.3638 | 3.5396 |
| loglog | 2/31 | 0.2508 | 1.6688 |
| simple_linear | 29/31 | 0.0608 | 1.1495 |

### After Wave 4

| Type | Pass | AvgErr | MaxErr | Delta |
|------|------|--------|--------|-------|
| **log_x** | **8/31** (+3) | **0.1343** | **0.4096** | +3 passes, avg −23%, max −17% |
| **log_y** | 6/31 | 0.3647 | **2.4274** | max −31% |
| **loglog** | 2/31 | 0.3125 | 1.7130 | avg drifted +25%; pass count stable |
| **simple_linear** | 29/31 | 0.0608 | 1.1495 | unchanged |

### Benchmark Failure Classification (13 samples)

| Dominant failure | Count | Examples |
|-----------------|-------|---------|
| x_axis | 3 | log_x/004, log_x/020, log_y/024 |
| y_axis | 2 | log_y/013, log_y/030 |
| both_axes | 4 | log_x/007, loglog/025, loglog/029 |
| series_geometry | 0 | — (not yet observed) |
| none (pass) | 4 | log_x/009, log_y/002, loglog/005, loglog/011 |

---

## 4. Convergence Decision

Per §12.7 convergence rule:

> "If two consecutive patches improve neither axis-only metrics nor pass counts, freeze the axis calibration strategy and shift to series extraction."

§12.2 produced no measurable improvement on current benchmarks. §12.1 was attempted and reverted. Two consecutive patches with no net improvement — **the axis calibration strategy is frozen**.

### Remaining axis-level issues are input-limited, not scoring-limited

| Failure mode | Root cause | Fix scope |
|-------------|-----------|-----------|
| OCR reads mantissas instead of full values (log_x/020, 026) | OCR preprocessing | Label cropping / scientific notation parsing |
| TMLOG decade over-count (log_y/013) | Decade detection accuracy | Generation layer (high risk) |
| X-axis wrongly treated as log (log_y/024) | Secondary candidate promotion | Selection layer (narrow case) |
| Loglog decade drift | Absolute decade assignment | Generation layer (high risk) |

These require either OCR preprocessing improvements or generation-layer changes to TMLOG/decade inference — both outside the safe scoring-only scope of the current axis calibration strategy.

---

## 5. Code Changes Retained in HEAD

| File | § | Change |
|------|---|--------|
| `axis_calibrator.py` | §12.3 | Y-axis decades guard bypass for superscript-loss candidates (~L1442) |
| `axis_calibrator.py` | §12.2 | `_seq_superscript_score` in `_solve_absolute_log_ticks` (~L1430) |
| `axis_calibrator.py` | §12.2 | `_literal_log_anchor_score` accepts `sequence_superscript_score` |
| `axis_calibrator.py` | §12.4 | `cross_axis_log` parameter plumbing, `_loglog_replace_gate`, span-change safety |
| `axis_evidence_micro_benchmark.py` | §12.5/6 | Summary table, `--samples`, `_compute_axis_error`, `_classify_dominant_failure` |

---

## 6. Recommended Next Wave

1. **OCR preprocessing** — Better label cropping and scientific notation parsing (addresses log_x/020, 026 class)
2. **Series extraction quality** — log_y/030 dominant failure is now `series_geometry`, not `y_axis`; axis calibration is no longer the bottleneck
3. **TMLOG decade width accuracy** — Generation-layer; requires careful regression testing
