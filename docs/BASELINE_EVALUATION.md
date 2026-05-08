# Baseline Evaluation

> **CRITICAL: Baselines recorded before 2026-04-28 are INVALID due to meta-information leakage.**
>
> The old pipeline passed `_meta.json` ground truth directly into `extract_from_image()`, which used meta-supplied axis ranges, series counts, and data values to artificially boost pass rates. See [META_ISOLATION_BOUNDARY.md](./META_ISOLATION_BOUNDARY.md) for the full audit and the hard boundary rule.
>
> **Old headline numbers (pre-2026-04-28):** v1 92.9%, v2 75.6%, v3 35.8%, v4 44.1% — all invalid.
> **True headline numbers (meta-isolated):** see True Baselines section below.

Current project phase: beta.

This document records validation baselines. The detailed CSV reports are generated locally and are not committed by default because repository rules ignore generated CSV outputs.

## Evaluation Commands

```powershell
python tests\validate_by_type.py --use-ocr --workers 4
python tests\validate_by_type.py --data-dir test_data_v2 --use-ocr --workers 4
python tests\validate_by_type.py --data-dir test_data_v3 --use-ocr --workers 4
python tests\validate_by_type.py --data-dir test_data_v4 --v4-special --use-ocr --workers 4
```

v4 uses a special supported-domain evaluator because the set contains single charts, combo charts, multi-subplot charts, unsupported chart types, and partial crops. The v4 headline score below is therefore the supported-domain score, not the full 500-image mixed-scope score.

---

## True Baselines (Meta-Isolated, Wave 4 — 2026-05-08)

### V1 Result (test_data, 310 images)

| Type | Pass | Rate | AvgErr | MaxErr | Top1 | Top2 |
|------|------|------|--------|--------|------|------|
| dense | 0/31 | 0.0% | 3.1004 | 19.4756 | 29.0% | 38.7% |
| dual_y | 2/31 | 6.5% | 0.7066 | 11.9217 | 0.0% | 3.2% |
| inverted_y | 30/31 | 96.8% | 0.0313 | 0.8527 | 0.0% | 0.0% |
| log_x | 8/31 | 25.8% | 0.1343 | 0.4096 | 90.3% | 100.0% |
| log_y | 6/31 | 19.4% | 0.3647 | 2.4274 | 100.0% | 100.0% |
| loglog | 2/31 | 6.5% | 0.3125 | 1.7130 | 0.0% | 100.0% |
| multi_series | 0/31 | 0.0% | 1076.3931 | 33355.9819 | 87.1% | 87.1% |
| no_grid | 27/31 | 87.1% | 0.0200 | 0.2384 | 0.0% | 100.0% |
| scatter | 30/31 | 96.8% | 0.0275 | 0.0893 | 61.3% | 71.0% |
| simple_linear | 29/31 | 93.5% | 0.0608 | 1.1495 | 38.7% | 61.3% |
| **TOTAL** | **134/310** | **43.2%** | — | — | **39.7%** | **65.5%** |

### V2 Result (test_data_v2, 500 images)

| Type | Pass | Rate | AvgErr | MaxErr | Top1 | Top2 |
|------|------|------|--------|--------|------|------|
| dense | 2/50 | 4.0% | 1.3811 | 10.5732 | 38.0% | 40.0% |
| dual_y | 7/50 | 14.0% | 0.7998 | 9.7677 | 0.0% | 66.0% |
| inverted_y | 27/50 | 54.0% | 0.5096 | 9.0879 | 0.0% | 0.0% |
| log_x | 2/50 | 4.0% | 0.6722 | 7.7623 | 16.0% | 24.0% |
| log_y | 6/50 | 12.0% | 0.4696 | 5.3120 | 22.0% | 26.0% |
| loglog | 10/50 | 20.0% | 1.3275 | 59.2408 | 6.0% | 48.0% |
| multi_series | 0/50 | 0.0% | 5565.3349 | 258979.3325 | 98.0% | 98.0% |
| no_grid | 23/50 | 46.0% | 2055.6248 | 77154.7708 | 0.0% | 20.0% |
| scatter | 37/50 | 74.0% | 0.0817 | 0.5544 | 34.0% | 50.0% |
| simple_linear | 19/50 | 38.0% | 68.3937 | 1095.4487 | 28.0% | 52.0% |
| **TOTAL** | **133/500** | **26.6%** | — | — | **24.2%** | **42.4%** |

### V3 Result (test_data_v3, 500 images)

| Type | Pass | Rate | AvgErr | MaxErr | Top1 | Top2 |
|------|------|------|--------|--------|------|------|
| dense | 0/50 | 0.0% | 2.6364 | 27.6559 | 34.0% | 42.0% |
| dual_y | 0/50 | 0.0% | 1.1101 | 16.9236 | 0.0% | 22.0% |
| inverted_y | 7/50 | 14.0% | 0.7182 | 8.2527 | 0.0% | 0.0% |
| log_x | 0/50 | 0.0% | 64.2459 | 3157.7237 | 24.0% | 38.0% |
| log_y | 14/50 | 28.0% | 78.6114 | 3897.7410 | 30.0% | 60.0% |
| loglog | 11/50 | 22.0% | 0.2823 | 6.8248 | 8.0% | 40.0% |
| multi_series | 0/50 | 0.0% | 9.0059 | 220.1587 | 80.0% | 86.0% |
| no_grid | 7/50 | 14.0% | 163.4275 | 7479.6164 | 0.0% | 22.0% |
| scatter | 29/50 | 58.0% | 0.1399 | 0.7851 | 16.0% | 26.0% |
| simple_linear | 4/50 | 8.0% | 143158022.1709 | 7106513921.3490 | 18.0% | 48.0% |
| **TOTAL** | **72/500** | **14.4%** | — | — | **21.0%** | **38.4%** |

### V4 Supported-Domain Result (test_data_v4, 204 in-scope images)

Scope accounting:

| Scope | Count |
|-------|------:|
| supported / in-scope | 204 |
| out-of-scope | 296 |
| total | 500 |

Supported-domain score:

| Type | Pass | Rate | AvgErr | MaxErr | Top1 | Top2 |
|------|------|------|--------|--------|------|------|
| dense | 1/18 | 5.6% | 622.7029 | 11132.2099 | 33.3% | 44.4% |
| dual_y | 2/23 | 8.7% | 0.6883 | 3.3871 | 0.0% | 17.4% |
| inverted_y | 1/18 | 5.6% | 24.8138 | 405.1714 | 0.0% | 0.0% |
| log_x | 1/18 | 5.6% | 2.8086 | 38.2118 | 16.7% | 22.2% |
| log_y | 3/24 | 12.5% | 0.4854 | 1.8745 | 8.3% | 12.5% |
| loglog | 6/20 | 30.0% | 4500.1711 | 87845.6270 | 0.0% | 20.0% |
| multi_series | 0/28 | 0.0% | 67.9812 | 1871.3895 | 89.3% | 89.3% |
| no_grid | 3/17 | 17.6% | 4.6531 | 62.5434 | 5.9% | 29.4% |
| scatter | 11/20 | 55.0% | 0.1380 | 1.0000 | 30.0% | 50.0% |
| simple_linear | 2/18 | 11.1% | 53692.9442 | 966410.6918 | 0.0% | 61.1% |
| **SUPPORTED TOTAL** | **30/204** | **14.7%** | — | — | **19.1%** | **36.3%** |

---

## Historical True Baselines (Pre-Wave 4)

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

**Key improvements from RANSAC era**: simple_linear (+80.6pp), inverted_y (+90.3pp), scatter (+90.3pp), no_grid (+71.0pp).

### V1 Result (2026-05-08) — Post-Wave 4

Wave 4 focused on axis calibration candidate scoring (§12). Key delta from RANSAC era:
- log_x: 0→8/31 (+25.8pp) — TMLOG decade fingerprint + solver improvements
- log_y: 1→6/31 (+16.1pp) — superscript-loss bypass, cross-axis safety
- simple_linear: 25→29/31 (+12.9pp)
- scatter: 28→30/31 (+6.5pp)
- **TOTAL: 114→134/310 (+6.5pp)**

Still failing: dense, dual_y, multi_series (0%), loglog (6.5%).

---

## Headline Comparison

| Dataset | Date | Pass | Rate | Notes |
|---------|------|------|------|-------|
| v1 | 2026-04-28 (pre-RANSAC) | 5/310 | 1.6% | Meta-isolated |
| v1 | 2026-04-28 (RANSAC) | 114/310 | 36.8% | Meta-isolated |
| v1 | 2026-05-08 (Wave 4) | 134/310 | **43.2%** | Current |
| v2 | 2026-05-08 | 133/500 | **26.6%** | Current |
| v3 | 2026-05-08 | 72/500 | **14.4%** | Current |
| v4 | 2026-05-08 | 30/204 | **14.7%** | Supported domain |

---

## Current Interpretation

The controlled v1 set is now at 43.2% (up from 36.8% after RANSAC). The biggest remaining gaps are:

1. **multi_series**: 0% across all datasets — routing is excellent (87-98% top1) but extraction quality is near-zero. HSV clustering and series separation need fundamental work.
2. **dense**: 0-5.6% — thinning helps but calibration is still fragile on oscillating lines.
3. **dual_y**: 0-14% — routed as multi_series, but secondary Y-axis handling is missing.
4. **log_x/log_y**: 25.8%/19.4% on v1, collapsing on v2-v4 — OCR quality degradation under style variation is the dominant cause.
5. **simple_linear collapses on v3/v4** (8.0%/11.1%) — rotation, noise, and grid removal failures under real-world degradation.

Scatter remains the strongest type (55-97% depending on dataset). Inverted_y and no_grid are strong on v1 but degrade significantly on v2-v4.

---

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

### Algorithm-Level Changes (2026-04-27 to 2026-04-28)

See [ARCHITECTURAL_CHANGES_IMPL.md](./ARCHITECTURAL_CHANGES_IMPL.md) for detailed implementation guidance.

1. **Full HSV clustering** (quality-gated fallback) — low risk, isolated to multi_series path
2. **OCR-specific preprocessing** (OpenCV core baseline) — medium risk, improves calibration path
3. **Zhang-Suen thinning** (contrib-gated + fallback) — medium risk, targeted to dense path
4. **Rotation detection + correction** — high risk, broad impact on v3 robustness

### Wave 4: Axis Calibration Candidate Scoring (2026-05-07 to 2026-05-08)

See [WAVE4_CLOSURE_2026_05_08.md](./development/WAVE4_CLOSURE_2026_05_08.md) for full details.

Focus: §12 of the axis evaluation strategy — improve log-axis calibration through better candidate scoring.

**Retained improvements**:
- Y-axis decades guard bypass for superscript-loss candidates (log_y max 3.54→2.43)
- Sequence-level superscript-loss scoring (structural, future-proofing)
- Cross-axis log detection safety for loglog
- Benchmark infrastructure: axis-only error metrics, dominant failure classification

**Convergence rule triggered**: Two consecutive patches (§12.2 no improvement, §12.1 reverted) with no net gain. Axis calibration strategy frozen. Remaining failures are input-limited (wrong OCR reads), not scoring-limited.

**Next priorities** (from Wave 4 closure):
1. OCR preprocessing (scientific notation parsing, better label cropping)
2. Series extraction quality (multi_series dominant failure shifted from axis to series_geometry)
3. TMLOG decade width accuracy (generation layer, high risk)

---

## Lint Gate

`pylint --fail-under=9` → **9.82/10** (2026-05-08)
