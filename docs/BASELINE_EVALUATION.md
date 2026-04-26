# Baseline Evaluation

Current project phase: beta.

This document records the current v1-v4 validation baseline collected on 2026-04-26. The detailed CSV reports are generated locally and are not committed by default because repository rules ignore generated CSV outputs.

## Evaluation Commands

```powershell
python tests\validate_by_type.py
python tests\validate_by_type.py --data-dir test_data_v2
python tests\validate_by_type.py --data-dir test_data_v3
python tests\validate_by_type.py --data-dir test_data_v4 --v4-special
```

v4 uses a special supported-domain evaluator because the set contains single charts, combo charts, multi-subplot charts, unsupported chart types, and partial crops. The v4 headline score below is therefore the supported-domain score, not the full 500-image mixed-scope score.

## Headline Baseline

| Dataset | Scope | Pass | Rate | Notes |
|---------|-------|------|------|-------|
| v1 | 10 supported chart types | 288/310 | 92.9% | Original controlled set |
| v2 | 10 supported chart types | 383/500 | 76.6% | Wider style and data variation |
| v3 | 10 supported chart types | 184/500 | 36.8% | Scanned/photographed degradation simulation |
| v4 | supported single-chart subset | 91/204 | 44.6% | 296/500 samples are out of current extractor scope |

## V1 Results

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
| dual_y | 37/50 | 74.0% | 1.0052 | 47.3437 |
| inverted_y | 40/50 | 80.0% | 0.0742 | 1.0000 |
| log_x | 47/50 | 94.0% | 0.0281 | 0.8657 |
| log_y | 50/50 | 100.0% | 0.0065 | 0.0286 |
| loglog | 46/50 | 92.0% | 0.1689 | 4.1160 |
| multi_series | 17/50 | 34.0% | 0.1375 | 0.5162 |
| no_grid | 43/50 | 86.0% | 0.0652 | 1.0000 |
| scatter | 47/50 | 94.0% | 0.0187 | 0.2215 |
| simple_linear | 36/50 | 72.0% | 0.1028 | 1.0000 |
| **TOTAL** | **383/500** | **76.6%** | — | — |

## V3 Results

| Type | Pass | Rate | AvgErr | MaxErr |
|------|------|------|--------|--------|
| dense | 9/50 | 18.0% | 0.6520 | 1.8399 |
| dual_y | 18/50 | 36.0% | 0.9694 | 16.2814 |
| inverted_y | 22/50 | 44.0% | 0.2498 | 1.1436 |
| log_x | 20/50 | 40.0% | 0.6315 | 6.8889 |
| log_y | 21/50 | 42.0% | 0.9628 | 4.0929 |
| loglog | 18/50 | 36.0% | 1.4203 | 4.5861 |
| multi_series | 9/50 | 18.0% | 0.2353 | 0.8545 |
| no_grid | 11/50 | 22.0% | 0.2842 | 1.0000 |
| scatter | 40/50 | 80.0% | 0.0952 | 0.7231 |
| simple_linear | 16/50 | 32.0% | 0.2837 | 0.7663 |
| **TOTAL** | **184/500** | **36.8%** | — | — |

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
| dense | 5/18 | 27.8% | 0.4646 | 1.3198 |
| dual_y | 8/23 | 34.8% | 0.2823 | 1.0000 |
| inverted_y | 8/18 | 44.4% | 0.2658 | 1.0000 |
| log_x | 12/18 | 66.7% | 0.9575 | 14.4071 |
| log_y | 10/24 | 41.7% | 0.6410 | 4.4666 |
| loglog | 9/20 | 45.0% | 0.9671 | 4.0887 |
| multi_series | 6/28 | 21.4% | 0.3114 | 1.0000 |
| no_grid | 10/17 | 58.8% | 0.1816 | 1.0000 |
| scatter | 17/20 | 85.0% | 0.0958 | 1.0000 |
| simple_linear | 6/18 | 33.3% | 0.4625 | 1.0000 |
| **SUPPORTED TOTAL** | **91/204** | **44.6%** | — | — |

## Current Interpretation

The controlled v1 set is mostly stable, but v2 and v3 show that style variation and scan-like degradation remain hard. v4 confirms that real-world pressure is not only a multi-series issue: axis detection and calibration under geometric and visual degradation are broad bottlenecks across chart types.

The strongest current supported area is scatter extraction. The weakest supported areas are multi-series, dense curves, and degraded simple/no-grid line charts where axes or foreground masks are unstable.
