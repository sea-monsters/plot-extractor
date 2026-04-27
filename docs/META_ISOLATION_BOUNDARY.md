# Meta Information Isolation Boundary

## Rule (Hard Boundary)

**Ground-truth `_meta.json` files may NEVER be passed into the extraction pipeline.**

- Test scripts may load `_meta.json` **only for scoring/comparison** after extraction completes.
- The extraction pipeline (`plot_extractor/main.py` and all modules it calls) must receive **only the image** plus optional policy/CLI flags.
- Any code that passes `meta=...` into `extract_from_image()` or any core module is a **bug** and must be removed immediately.

## Why This Exists

During a 2026-04-28 audit, we discovered that the old validation pipeline was passing `meta=meta` directly to `extract_from_image()`. The meta dictionary contained ground-truth axis ranges (`axes.min`, `axes.max`, `axes.type`), series counts, and data values. The extraction code used this information to:

1. **Calibrate axes** — falling back to meta-supplied min/max when OCR failed (`cal_meta_fallback` strategy).
2. **Determine series count** — using `meta["data"]` to know how many series to expect for color clustering.
3. **Score and select candidates** — using meta ground truth to pick the "best" axis mapping or series separation.

This produced **artificially high pass rates** that did not reflect real-world extraction quality:

| Dataset | Leaked (with meta) | True (meta-isolated) | Delta |
|---------|-------------------|----------------------|-------|
| v1 | 92.9% | ~7% | **-85.9pp** |
| v2 | 75.6% | ~10% | **-65.6pp** |
| v3 | 37.4% | ~7% | **-30.4pp** |
| v4 | 44.1% | ~7% | **-37.1pp** |

The old `docs/BASELINE_EVALUATION.md` numbers (dated 2026-04-26) are **invalid** because they were collected with meta leakage.

## Where the Boundary Lives

### Extraction side (MUST NOT receive meta)

| File | Responsibility |
|------|---------------|
| `plot_extractor/main.py` | Entry point. `extract_from_image()` signature has no `meta` parameter. |
| `plot_extractor/core/axis_detector.py` | Detects axes from image pixels only. |
| `plot_extractor/core/axis_calibrator.py` | Calibrates from OCR + heuristics only. No meta fallback. |
| `plot_extractor/core/data_extractor.py` | Extracts data from image pixels only. Series count estimated from visual features. |
| `plot_extractor/core/axis_candidates.py` | Candidates from OCR and heuristic only. No meta source. |
| `plot_extractor/core/series_candidates.py` | Candidates from image analysis only. |
| `plot_extractor/core/policy_router.py` | Policy has no `cal_meta_fallback` field. |
| `plot_extractor/core/confidence.py` | Scoring does not reference meta reliability. |
| `plot_extractor/core/ocr_reader.py` | Reads tick labels from image crops only. |

### Validation side (MAY load meta for scoring)

| File | Responsibility |
|------|---------------|
| `tests/validate_by_type.py` | Loads meta at line 206. Uses for `evaluate_data_accuracy()` only. Comment: "Load meta for scoring ONLY; never pass to extraction". |
| `tests/validate_v4a_routes.py` | Loads meta to read `tags` (route profiles, targets, confusers) for router evaluation. Never passes to extraction. |
| `tests/validate_loop.py` | Validation loop. Loads meta for threshold lookup only. |
| `tests/generate_test_data*.py` | Generates `_meta.json` as ground truth. This is correct. |

## How to Verify Compliance

Run this grep. If any matches appear in `plot_extractor/` (not `tests/`), investigate immediately:

```bash
# MUST return zero results inside plot_extractor/core/ and plot_extractor/main.py
grep -rn "meta=" plot_extractor/core/ plot_extractor/main.py
grep -rn "load_meta" plot_extractor/core/ plot_extractor/main.py
```

Test scripts may match, but only in the context of **scoring after extraction**, never as a function argument to extraction.

## Historical Context

- **2026-04-26**: Old baselines collected with meta leakage. `docs/BASELINE_EVALUATION.md` recorded v1 92.9%, v2 75.6%, v3 35.8%, v4 44.1%.
- **2026-04-27**: Algorithm-level changes implemented (HSV clustering, OCR preprocessing, Zhang-Suen thinning, rotation correction). Validation still passing meta.
- **2026-04-28 01:19**: Audit revealed meta leakage. Immediate cleanup across 7+ files.
- **2026-04-28**: True baselines established. V1 ~7%, V2 ~10%, V3 ~7%, V4 ~7%. These are the honest starting points for improvement.

## Enforcement

1. **Code review**: Any PR that adds a `meta` parameter to an extraction function must be blocked.
2. **Lint**: No automated lint rule exists yet; rely on the grep check above.
3. **Tests**: If a test passes meta to extraction, it is a bad test and must be fixed.
4. **Documentation**: This file is the canonical reference. If you move or rename it, update all references.
