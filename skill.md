# Plot Extractor Plugin — Skill Reference

> This document helps Claude Code / OpenClaw understand the plugin architecture, optional features, and environment dependencies. Read this before invoking extraction or debugging failures.

## Architecture Overview

The extraction pipeline uses a **three-layer difficulty-stratified strategy ensemble**:

```
Layer 1 (default) — Rule-based ChartTypeGuesser + PolicyRouter
  Lightweight image features → softmax type probabilities → policy ensemble
  No external dependencies beyond OpenCV + NumPy.

Layer 2 (uncertain cases) — LLM Vision Enhancement
  Triggers ONLY when top1_prob - top2_prob < 0.15 (low confidence gap)
  Calls a vision-capable LLM to classify the chart image.
  Reuses the API key already configured for Claude Code / OpenClaw.

Layer 3 (opt-in) — OCR (tesseract)
  Explicitly enabled via --use-ocr flag or use_ocr=True.
  Reads tick label text from the image for axis calibration.
  Falls back to heuristic synthetic ticks when OCR is unavailable.

> **CRITICAL: OCR has a massive impact on accuracy.**
> Without OCR, the pipeline falls back to heuristic synthetic ticks that assign
> arbitrary scales (e.g., 0,1,2,... or logspace(0,1,n)), producing ~100% relative
> error on real-world charts. With OCR, typical pass rates improve dramatically:
> - simple_linear: ~90% → ~93%
> - dense: ~0% → ~65%
> - log_y: ~0% → ~85% (with LLM fallback for narrow ranges)
> **Always enable --use-ocr unless you have ground-truth meta JSON.**
```

## Environment Variables

### LLM Configuration (optional)

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Use Anthropic Claude vision models | `claude-haiku-4-5` |
| `OPENAI_API_KEY` | Use OpenAI GPT vision models | `gpt-5.4-nano` |
| `LLM_BASE_URL` | Custom OpenAI-compatible endpoint URL | — |
| `LLM_API_KEY` | API key for the custom endpoint | — |
| `LLM_MODEL` | **Model name override for all providers** | Provider-specific default |

**Provider priority:** Anthropic → OpenAI → Custom endpoint.

When running inside Claude Code, `ANTHROPIC_API_KEY` is typically already set — the plugin reuses it automatically. No extra configuration is required unless you want to override the model name (`LLM_MODEL`).

### OCR Configuration (optional)

OCR requires **tesseract** to be installed on the system:

- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
- **macOS:** `brew install tesseract`
- **Windows:** Download installer from https://github.com/UB-Mannheim/tesseract/wiki

If tesseract is missing, the pipeline falls back to heuristic tick generation. Accuracy may degrade for charts with non-uniform tick spacing.

## CLI Usage

```bash
# Default extraction (Layer 1 only)
python plot_extractor/main.py input_chart.png --output extracted.csv --debug debug/

# Enable LLM enhancement for ambiguous charts (Layer 1 + Layer 2)
python plot_extractor/main.py input_chart.png --use-llm

# Enable OCR for tick label reading (Layer 3)
python plot_extractor/main.py input_chart.png --use-ocr

# Enable both
python plot_extractor/main.py input_chart.png --use-llm --use-ocr
```

## Validation Usage

```bash
# Validate baseline dataset (no meta passed to extractor)
python tests/validate_by_type.py --debug

# Validate with LLM enhancement
python tests/validate_by_type.py --use-llm --debug

# Validate with OCR
python tests/validate_by_type.py --use-ocr --debug

# Validate custom dataset
python tests/validate_by_type.py --data-dir test_data_v2 --use-llm
```

Validation reports include per-type routing metrics:
- `top1_acc` / `top2_acc` — chart type guess accuracy
- `density_strategies` — how often thinning/standard/scatter extraction was chosen
- `color_strategies` — how often HSV3D/layered/hue-only color separation was chosen
- `noise_strategies` — how often salt-pepper/jpeg/blur/rotation_noise preprocessing was chosen

## Policy Router Behavior

The `ExtractionPolicy` dataclass controls these pipeline stages:

| Field | Values | Affected Stage |
|-------|--------|----------------|
| `noise_strategy` | `clean`, `salt_pepper`, `jpeg`, `blur`, `rotation_noise` | `image_loader.py` preprocessing |
| `rotation_correct` | `True` / `False` | `main.py` rotation correction |
| `color_strategy` | `hue_only`, `hsv3d`, `layered`, `none` | `data_extractor.py` series separation |
| `density_strategy` | `standard`, `thinning`, `scatter` | `data_extractor.py` point extraction |
| `ocr_block_size` | odd integer | `ocr_reader.py` adaptive threshold block size |
| `ocr_C` | integer | `ocr_reader.py` adaptive threshold constant |
| `hough_threshold` | integer | `axis_detector.py` HoughLinesP threshold |
| `cal_residual_threshold_linear` | float | `axis_calibrator.py` residual gate |
| `cal_residual_threshold_log` | float | `axis_calibrator.py` residual gate |

## LLM Axis Fallback

When `--use-llm` and `--use-ocr` are both enabled, the pipeline uses a two-tier
axis reading strategy:

1. **OCR first** — tesseract reads each tick label independently.
2. **LLM fallback** — If OCR yields < 2 valid labels (or < 40% coverage on a
dense 6+ tick axis), the pipeline crops the axis region and sends it to the
vision LLM with the prompt:
   > "Read the numeric label at each tick mark. If a tick has no visible label, use null."

The LLM returns an ordered array of tick values that is merged with OCR results
(OCR values take priority where available). This is especially effective for:

- **Log axes with narrow ranges** (e.g., 2.7–36.3): matplotlib may label only one
major tick. The LLM infers the remaining major ticks from the visual pattern.
- **Dense axes**: OCR often reads only endpoint labels. The LLM fills in middle
values from the cropped axis image.

## Common Failure Modes & Actions

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| "No axes detected" | Rotation, heavy noise, or non-standard layout | Try `--use-llm`; verify image is a standard chart |
| "Axis calibration failed" | OCR missing + heuristic tick guess wrong | Install tesseract and use `--use-ocr`; or verify tick spacing is uniform |
| "Axis calibration failed" (log axes) | Only 1 major tick labeled (narrow log range) | Use `--use-llm --use-ocr`; LLM reads major ticks from axis crop |
| "No data extracted" | Dense chart not thinned, or scatter misrouted as line | Check routing summary for density_strategy; use `--use-llm` |
| LLM not triggering (policy) | Confidence gap >= 0.15 (guesser is already certain) | Normal — LLM only activates on ambiguous chart classification |
| LLM not triggering (axis) | OCR read >= 2 labels or LLM unavailable | Normal — LLM axis fallback only fires when OCR is insufficient |
| `ModuleNotFoundError: pytesseract` | OCR imported but package missing | `pip install pytesseract` or avoid `--use-ocr` |
| `TesseractNotFoundError` | tesseract binary not on PATH | Install tesseract (see OCR Configuration above) |

## Dependency Checklist

Before enabling optional features, verify these are available:

- [ ] `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` or (`LLM_BASE_URL` + `LLM_API_KEY`) — for `--use-llm`
- [ ] `tesseract` binary on system PATH — for `--use-ocr`
- [ ] `LLM_MODEL` set if you want a non-default model for your provider
