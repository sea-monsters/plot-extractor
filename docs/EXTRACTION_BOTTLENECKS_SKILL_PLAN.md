# Extraction Bottlenecks And Skill Plan

Current phase: beta.

This document records the v1-v4 extraction bottlenecks and evaluates which fixes can be implemented as small, robust, skill-like modules. A "skill" here means a narrow, reusable capability with clear inputs, outputs, and validation, not a broad rewrite of the extractor.

## Evidence Base

Baseline source: `docs/BASELINE_EVALUATION.md`.

| Dataset | Scope | Pass | Rate | Main Signal |
|---------|-------|------|------|-------------|
| v1 | controlled supported chart types | 288/310 | 92.9% | Current pipeline works on clean generated charts |
| v2 | wider generated variation | 383/500 | 76.6% | Style and data variation expose generalization gaps |
| v3 | scan/photo degradation simulation | 184/500 | 36.8% | Image degradation breaks broad extraction paths |
| v4 | supported single-chart subset | 91/204 | 44.6% | Realistic mixed pressure breaks more than multi-series |

v4 also contains 296 out-of-scope samples among 500 total images. These include combo charts, multi-subplot charts, unsupported chart types, and partial crops.

## Bottleneck 1: Axis Detection And Calibration Robustness

### Evidence

Simple and otherwise supported chart types degrade sharply under v3/v4 pressure:

| Type | v1 | v2 | v3 | v4 supported |
|------|----|----|----|--------------|
| simple_linear | 100.0% | 72.0% | 32.0% | 33.3% |
| no_grid | 93.5% | 86.0% | 22.0% | 58.8% |
| log_y | 100.0% | 100.0% | 42.0% | 41.7% |
| log_x | 100.0% | 94.0% | 40.0% | 66.7% |
| loglog | 100.0% | 92.0% | 36.0% | 45.0% |

This points to plot-area localization, axis line detection, tick inference, and pixel-to-data mapping as a cross-type bottleneck. Once calibration is wrong, even a good foreground extraction produces bad numeric data.

### Skill Candidate: `axis-recovery`

Goal: produce a robust plot rectangle and axis model before curve extraction.

Inputs:
- original image
- optional metadata axes
- optional current `AxisDetector` output

Outputs:
- plot area rectangle
- x/y axis line candidates with confidence
- fallback tick/scale hints
- diagnostic reason when confidence is low

Implementation shape:
- Try the existing Hough-based detector first.
- Add a fallback that runs contrast normalization, adaptive thresholding, morphological line extraction, and contour/connected-component scoring.
- Score candidates by geometry: near-horizontal/near-vertical, border proximity, long span, intersection consistency, and tick density.
- Return confidence instead of silently accepting a weak axis.

Why it is a good skill:
- Small surface area.
- Benefits all chart types.
- Can be validated on v3/v4 without changing series extraction.
- Easy rollback: use only when the current detector confidence is low.

Risks:
- Over-aggressive morphology may confuse grid lines with axes.
- Partial crops may produce high-confidence false axes unless scope detection is checked first.

Validation:
- Add a report mode that records axis confidence and plot-area error where metadata is available.
- First target: improve v3/v4 `simple_linear`, `no_grid`, and log-family scores without regressing v1.

## Bottleneck 2: Multi-Series Separation And Ownership

### Evidence

| Dataset | Multi-series Pass | Rate |
|---------|-------------------|------|
| v1 | 17/31 | 54.8% |
| v2 | 17/50 | 34.0% |
| v3 | 9/50 | 18.0% |
| v4 | 6/28 | 21.4% |

The failures are consistent with color mixing, crossings, nearby colors, anti-aliasing, legends/markers/grid artifacts, and fragmented lines.

### Skill Candidate: `series-separation`

Goal: assign foreground pixels or curve segments to stable series IDs.

Inputs:
- plot-area image crop
- foreground mask
- optional expected series count from metadata or legend inference
- optional previous color clusters

Outputs:
- per-series masks or point clouds
- cluster confidence
- merge/split diagnostics

Implementation shape:
- Cluster only foreground pixels inside the plot area.
- Use perceptual color space such as Lab/HSV rather than raw RGB alone.
- Seed with high-saturation, non-grid pixels.
- Use local continuity to repair short gaps.
- At crossings, preserve ambiguous points as shared candidates until ordering/continuity resolves ownership.

Why it is a good skill:
- Narrowly targets the weakest supported type.
- Can be called only for multi-series or when multiple color clusters are detected.
- Does not require changing axis calibration.

Risks:
- Similar colors and grayscale charts remain hard.
- Metadata-assisted expected series count may overfit generated datasets if not guarded by confidence.

Validation:
- Track per-series count accuracy, not only final numeric error.
- First target: v2/v4 multi-series before attempting degraded v3 recovery.

## Bottleneck 3: Dense Curve And Fragment Repair

### Evidence

| Dataset | Dense Pass | Rate |
|---------|------------|------|
| v1 | 30/31 | 96.8% |
| v2 | 20/50 | 40.0% |
| v3 | 9/50 | 18.0% |
| v4 | 5/18 | 27.8% |

Clean dense curves are mostly solved, but style variation and degradation break foreground masks, grid suppression, and line continuity.

### Skill Candidate: `curve-continuity-repair`

Goal: convert noisy foreground fragments into one or more plausible continuous curve traces.

Inputs:
- foreground mask
- plot area
- chart type hint

Outputs:
- repaired mask
- ordered curve samples
- confidence and gap statistics

Implementation shape:
- Remove tiny components outside plausible curve geometry.
- Use skeletonization or thinning when available.
- Join short gaps using local direction and distance limits.
- For dense charts, sample columns with robust median/quantile envelopes instead of trusting every connected pixel.

Why it is a good skill:
- Local to mask post-processing.
- Useful for dense and degraded simple line charts.
- Can be validated by point-count stability and numeric error.

Risks:
- Can incorrectly bridge separate series if used before series separation.
- Must be gated by chart type or cluster count.

Validation:
- First target: v2/v4 dense and v3 simple_linear.
- Reject if v1 dense regresses.

## Bottleneck 4: Foreground And Grid/Text Separation

### Evidence

The v2-v4 drop across simple, no-grid, dense, and log-family charts suggests that the extractor sometimes treats grid/text/noise as data or removes real data with the background.

### Skill Candidate: `foreground-cleanup`

Goal: create a cleaner plot-area foreground mask before data extraction.

Inputs:
- plot-area image
- axis/grid hints
- optional detected text/tick regions

Outputs:
- cleaned foreground mask
- removed grid/text/noise mask
- cleanup diagnostics

Implementation shape:
- Estimate background from border and low-saturation regions.
- Suppress long straight grid-like lines before curve sampling.
- Remove text-like components by size, aspect ratio, and location.
- Preserve colored/high-contrast curve pixels even when close to grid lines.

Why it is a good skill:
- Reusable preprocessor.
- Does not change output schema.
- Can be introduced behind a config flag or confidence gate.

Risks:
- Grid removal can erase thin data lines.
- Text-component heuristics may remove small markers.

Validation:
- Compare foreground pixel retention against metadata-rendered expected regions where available.
- Track pass-rate change on v2 simple/no_grid/dense.

## Bottleneck 5: Scope Classification Before Extraction

### Evidence

In v4, 296/500 images are outside the current supported single-chart domain. Treating these as normal extraction failures hides the real boundary between unsupported input and extractor mistakes.

### Skill Candidate: `scope-classifier`

Goal: decide whether an image should enter the standard extractor, a specialized future path, or an explicit unsupported response.

Inputs:
- original image
- optional metadata tags during evaluation

Outputs:
- scope label: supported_single_chart, partial_crop, multi_subplot, combo_chart, unsupported_type, uncertain
- confidence
- reason codes

Implementation shape:
- Start metadata-driven for tests.
- Add image-only heuristics: multiple plot rectangles, missing axes, multiple panel grids, bar/pie-like geometry, crop boundaries.
- For uncertain cases, prefer "unsupported/needs review" over low-confidence extraction.

Why it is a good skill:
- Very small first version.
- Improves evaluation honesty immediately.
- Lets future combo/multi-subplot support be added as separate paths.

Risks:
- It does not improve extraction accuracy directly.
- Over-filtering could hide supported difficult samples if confidence rules are too strict.

Validation:
- v4 scope report should remain explicit and auditable.
- Measure supported-domain pass rate separately from total dataset coverage.

## Suggested Skill Implementation Order

1. `axis-recovery`
   - Highest leverage because it affects every chart type.
   - First validation target: v3/v4 simple_linear and log-family.

2. `foreground-cleanup`
   - Supports axis recovery and curve extraction.
   - First validation target: v2/v3 simple_linear, no_grid, dense.

3. `series-separation`
   - Directly addresses the weakest supported user-visible class.
   - First validation target: v2/v4 multi_series.

4. `curve-continuity-repair`
   - Best after foreground and series ownership are less noisy.
   - First validation target: v2/v4 dense and degraded simple lines.

5. `scope-classifier`
   - Keep evaluation honest and prepare for future real-world expansion.
   - First validation target: v4 scope accounting and unsupported reason stability.

## Light Skill Interface Pattern

Each skill should stay small and callable from the existing pipeline:

```python
result = skill.run(input_image, context)
if result.confidence >= threshold:
    context.update(result.outputs)
else:
    context.add_diagnostic(result.reason)
```

Recommended result fields:

- `outputs`: changed masks, rectangles, clusters, or traces
- `confidence`: numeric confidence in `[0, 1]`
- `reason`: short diagnostic reason
- `metrics`: skill-specific counters for reports

This keeps the skills robust, testable, and easy to disable if a validation round shows regression.

## Context7 Check

Context7 MCP lookup was attempted for OpenCV and scikit-image implementation guidance on 2026-04-26, but the connector returned `fetch failed`. The fallback research direction remains standard OpenCV/scikit-image primitives: morphology, Hough line detection, contours, connected components, color-space clustering, and skeletonization/thinning. Re-run Context7 before implementation if connectivity recovers.

## Next Action

Start with a minimal `axis-recovery` diagnostic pass:

1. Add axis confidence reporting without changing behavior.
2. Run v1-v4 and identify low-confidence false passes/failures.
3. Add fallback axis candidate scoring only for low-confidence cases.
4. Commit the round and compare v1-v4 deltas.
