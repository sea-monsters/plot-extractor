"""Axis evidence micro-benchmark for log calibration failures.

This script runs a fixed, representative set of log-axis samples and records
axis-level evidence alongside data-level accuracy.  It is intentionally small:
use it before full validation when changing absolute decade inference.

Extended (Wave 4, §12.5): compact summary table, --samples subset filtering,
best-vs-runner-up candidate deltas, axis-only error metrics, and dominant_failure.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.main import extract_from_image
from tests.validate_by_type import evaluate_data_accuracy


DEFAULT_SAMPLES = [
    "log_x/004.png",
    "log_x/007.png",
    "log_x/009.png",
    "log_x/020.png",
    "log_x/026.png",
    "log_y/002.png",
    "log_y/013.png",
    "log_y/024.png",
    "log_y/030.png",
    "loglog/005.png",
    "loglog/011.png",
    "loglog/025.png",
    "loglog/029.png",
]


def _load_meta(image_path: Path) -> dict:
    meta_path = image_path.with_name(f"{image_path.stem}_meta.json")
    if not meta_path.exists() and image_path.stem == "000_original":
        meta_path = image_path.with_name("000_original_meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _compute_axis_error(
    selected_min: float | None,
    selected_max: float | None,
    expected_min: float | None,
    expected_max: float | None,
    axis_type: str | None,
) -> dict:
    """Compute axis-only error metrics from selected vs expected ranges."""
    result = {
        "axis_endpoint_err": None,
        "axis_span_ratio": None,
        "axis_log_endpoint_err": None,
    }
    if any(v is None for v in [selected_min, selected_max, expected_min, expected_max]):
        return result
    if any(v <= 0 for v in [selected_min, selected_max, expected_min, expected_max]):
        # Fall back to linear for non-positive ranges
        exp_span = expected_max - expected_min
        if exp_span > 0:
            result["axis_endpoint_err"] = (
                abs(selected_max - expected_max) / exp_span
                + abs(selected_min - expected_min) / exp_span
            ) / 2.0
        else:
            result["axis_endpoint_err"] = abs(selected_max - expected_max) + abs(selected_min - expected_min)
        return result

    # Log-space endpoint error
    log_sel_min = math.log10(selected_min)
    log_sel_max = math.log10(selected_max)
    log_exp_min = math.log10(expected_min)
    log_exp_max = math.log10(expected_max)
    result["axis_log_endpoint_err"] = abs(log_sel_max - log_exp_max) + abs(log_sel_min - log_exp_min)

    # Linear endpoint error
    sel_span = selected_max - selected_min
    exp_span = expected_max - expected_min
    if exp_span > 0:
        result["axis_endpoint_err"] = (
            abs(selected_max - expected_max) / exp_span
            + abs(selected_min - expected_min) / exp_span
        ) / 2.0
    else:
        result["axis_endpoint_err"] = None

    # Span ratio
    exp_log_span = log_exp_max - log_exp_min
    sel_log_span = log_sel_max - log_sel_min
    if exp_log_span > 0:
        result["axis_span_ratio"] = sel_log_span / exp_log_span
    else:
        result["axis_span_ratio"] = None

    return result


def _classify_dominant_failure(
    x_axis_err: dict,
    y_axis_err: dict,
    rel_err: float,
    x_type: str | None = None,
    y_type: str | None = None,
) -> str:
    """Classify the dominant failure source for a sample.

    Uses log-space endpoint error (>0.5 decades) for log axes and
    linear endpoint error (>0.15 = ~15 % span deviation) for linear axes.
    """
    if rel_err <= 0.05:
        return "none"

    x_log = x_axis_err.get("axis_log_endpoint_err")
    x_lin = x_axis_err.get("axis_endpoint_err")
    y_log = y_axis_err.get("axis_log_endpoint_err")
    y_lin = y_axis_err.get("axis_endpoint_err")

    if x_type == "log" and x_log is not None:
        x_bad = x_log > 0.5
    elif x_lin is not None:
        x_bad = x_lin > 0.15
    else:
        x_bad = False

    if y_type == "log" and y_log is not None:
        y_bad = y_log > 0.5
    elif y_lin is not None:
        y_bad = y_lin > 0.15
    else:
        y_bad = False

    if x_bad and y_bad:
        return "both_axes"
    if x_bad:
        return "x_axis"
    if y_bad:
        return "y_axis"
    # Axis errors are small but data error is high → series extraction
    if rel_err > 0.10:
        return "series_geometry"
    return "unknown"


def _extract_best_runnerup_delta(candidates: list[dict]) -> dict:
    """Extract score delta between best and runner-up candidate."""
    if len(candidates) < 2:
        return {"delta": None, "best_score": None, "runnerup_score": None}
    sorted_cands = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
    best = sorted_cands[0].get("score", 0)
    runner = sorted_cands[1].get("score", 0)
    return {
        "delta": round(best - runner, 4),
        "best_score": round(best, 4),
        "runnerup_score": round(runner, 4),
    }


def _axis_rows(sample: str, result: dict, meta: dict, rel_err: float, passed: bool) -> list[dict]:
    rows = []
    diagnostics = result.get("diagnostics", {})
    expected_axes = meta.get("axes", {})
    for axis in diagnostics.get("axes", []):
        direction = axis.get("direction")
        expected = expected_axes.get(direction, {})
        trace = axis.get("debug_trace", {})
        evidence = trace.get("axis_evidence", {})
        anchors = evidence.get("anchors", [])
        observed = [
            {
                "pixel": a.get("tick_pixel"),
                "text": a.get("tesseract_text"),
                "value": a.get("tesseract_value"),
                "source": a.get("source"),
                "format": a.get("value_format"),
            }
            for a in anchors
            if a.get("source") != "missing"
        ]
        candidate_sources = trace.get("candidate_sources", [])

        # Extract solver diagnostics for candidate delta
        solver_diag = trace.get("solver_diagnostics", {})
        candidates_detail = solver_diag.get("candidates", [])
        cand_delta = _extract_best_runnerup_delta(candidates_detail)

        # Axis-only error metrics
        val_range = axis.get("value_range") or [None, None]
        axis_errors = _compute_axis_error(
            val_range[0], val_range[1],
            expected.get("min"), expected.get("max"),
            axis.get("axis_type"),
        )

        rows.append({
            "sample": sample,
            "direction": direction,
            "side": axis.get("side"),
            "axis_type": axis.get("axis_type"),
            "tick_source": axis.get("tick_source"),
            "tick_count": axis.get("tick_count"),
            "labeled_tick_count": axis.get("labeled_tick_count"),
            "value_min": val_range[0],
            "value_max": val_range[1],
            "expected_type": expected.get("type"),
            "expected_min": expected.get("min"),
            "expected_max": expected.get("max"),
            "axis_endpoint_err": axis_errors["axis_endpoint_err"],
            "axis_span_ratio": axis_errors["axis_span_ratio"],
            "axis_log_endpoint_err": axis_errors["axis_log_endpoint_err"],
            "rel_err": rel_err,
            "passed": passed,
            "candidate_sources": json.dumps(candidate_sources, ensure_ascii=True),
            "observed_anchors": json.dumps(observed, ensure_ascii=True),
            "solver_candidate_count": solver_diag.get("candidate_count", 0),
            "solver_best_score": solver_diag.get("best_score"),
            "solver_best_delta": cand_delta["delta"],
            "solver_runnerup_score": cand_delta["runnerup_score"],
        })
    return rows


def _build_summary_row(
    sample: str,
    chart_type: str,
    result: dict,
    meta: dict,
    rel_err: float,
    passed: bool,
) -> dict:
    """Build a one-row-per-sample summary with primary axis ranges and failure classification."""
    diagnostics = result.get("diagnostics", {})
    expected_axes = meta.get("axes", {})

    primary_x = {}
    primary_y = {}
    for axis in diagnostics.get("axes", []):
        d = axis.get("direction")
        side = axis.get("side")
        val_range = axis.get("value_range") or [None, None]
        if d == "x" and side in ("bottom", None):
            primary_x = {
                "x_range": val_range,
                "x_type": axis.get("axis_type"),
                "x_source": axis.get("tick_source"),
            }
        elif d == "y" and side in ("left", None):
            primary_y = {
                "y_range": val_range,
                "y_type": axis.get("axis_type"),
                "y_source": axis.get("tick_source"),
            }

    # Fallback: take first seen of each direction
    for axis in diagnostics.get("axes", []):
        d = axis.get("direction")
        val_range = axis.get("value_range") or [None, None]
        if d == "x" and not primary_x:
            primary_x = {
                "x_range": val_range,
                "x_type": axis.get("axis_type"),
                "x_source": axis.get("tick_source"),
            }
        elif d == "y" and not primary_y:
            primary_y = {
                "y_range": val_range,
                "y_type": axis.get("axis_type"),
                "y_source": axis.get("tick_source"),
            }

    exp_x = expected_axes.get("x", {})
    exp_y = expected_axes.get("y", {})

    x_axis_err = _compute_axis_error(
        (primary_x.get("x_range") or [None, None])[0],
        (primary_x.get("x_range") or [None, None])[1],
        exp_x.get("min"), exp_x.get("max"),
        primary_x.get("x_type"),
    )
    y_axis_err = _compute_axis_error(
        (primary_y.get("y_range") or [None, None])[0],
        (primary_y.get("y_range") or [None, None])[1],
        exp_y.get("min"), exp_y.get("max"),
        primary_y.get("y_type"),
    )

    dominant = _classify_dominant_failure(
        x_axis_err, y_axis_err, rel_err,
        x_type=primary_x.get("x_type"),
        y_type=primary_y.get("y_type"),
    )

    return {
        "sample": sample,
        "chart_type": chart_type,
        "rel_err": round(rel_err, 4),
        "passed": passed,
        "primary_x_min": (primary_x.get("x_range") or [None, None])[0],
        "primary_x_max": (primary_x.get("x_range") or [None, None])[1],
        "primary_y_min": (primary_y.get("y_range") or [None, None])[0],
        "primary_y_max": (primary_y.get("y_range") or [None, None])[1],
        "expected_x_min": exp_x.get("min"),
        "expected_x_max": exp_x.get("max"),
        "expected_y_min": exp_y.get("min"),
        "expected_y_max": exp_y.get("max"),
        "x_source": primary_x.get("x_source"),
        "y_source": primary_y.get("y_source"),
        "x_type": primary_x.get("x_type"),
        "y_type": primary_y.get("y_type"),
        "x_axis_log_err": x_axis_err.get("axis_log_endpoint_err"),
        "y_axis_log_err": y_axis_err.get("axis_log_endpoint_err"),
        "x_span_ratio": x_axis_err.get("axis_span_ratio"),
        "y_span_ratio": y_axis_err.get("axis_span_ratio"),
        "dominant_failure": dominant,
    }


def run_benchmark(data_dir: Path, output_dir: Path, samples: list[str]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
    summary_rows = []
    for sample in samples:
        chart_type = sample.split("/", 1)[0]
        image_path = data_dir / sample
        meta = _load_meta(image_path)
        result = extract_from_image(
            image_path,
            output_csv=output_dir / f"{chart_type}_{image_path.stem}.csv",
            debug_dir=output_dir / "debug",
            use_ocr=True,
        )
        rel_err, _errors = evaluate_data_accuracy(result.get("data"), meta, chart_type)
        passed = bool(rel_err <= {
            "log_x": 0.05,
            "log_y": 0.05,
            "loglog": 0.05,
        }.get(chart_type, 0.05))
        summary.append({
            "sample": sample,
            "chart_type": chart_type,
            "rel_err": rel_err,
            "passed": passed,
        })
        all_rows.extend(_axis_rows(sample, result, meta, rel_err, passed))
        summary_rows.append(_build_summary_row(sample, chart_type, result, meta, rel_err, passed))

    csv_path = output_dir / "axis_evidence_micro_benchmark.csv"
    json_path = output_dir / "axis_evidence_micro_benchmark.json"
    summary_csv_path = output_dir / "benchmark_summary.csv"

    # Detailed axis-level CSV
    fieldnames = [
        "sample", "direction", "side", "axis_type", "tick_source",
        "tick_count", "labeled_tick_count", "value_min", "value_max",
        "expected_type", "expected_min", "expected_max",
        "axis_endpoint_err", "axis_span_ratio", "axis_log_endpoint_err",
        "rel_err", "passed",
        "candidate_sources", "observed_anchors",
        "solver_candidate_count", "solver_best_score",
        "solver_best_delta", "solver_runnerup_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary CSV (one row per sample)
    summary_fieldnames = [
        "sample", "chart_type", "rel_err", "passed",
        "primary_x_min", "primary_x_max", "primary_y_min", "primary_y_max",
        "expected_x_min", "expected_x_max", "expected_y_min", "expected_y_max",
        "x_source", "y_source", "x_type", "y_type",
        "x_axis_log_err", "y_axis_log_err", "x_span_ratio", "y_span_ratio",
        "dominant_failure",
    ]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path.write_text(json.dumps({
        "summary": summary,
        "summary_table": summary_rows,
        "axes": all_rows,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("test_data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "debug" / "axis_evidence_micro_benchmark",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help=(
            "Comma-separated sample list, e.g. 'log_x/020.png,log_x/026.png'. "
            "Defaults to the full DEFAULT_SAMPLES set."
        ),
    )
    args = parser.parse_args()

    samples = DEFAULT_SAMPLES
    if args.samples:
        samples = [s.strip() for s in args.samples.split(",") if s.strip()]

    csv_path, json_path = run_benchmark(args.data_dir, args.output_dir, samples)
    print(f"Axis evidence CSV: {csv_path}")
    print(f"Axis evidence JSON: {json_path}")
    print(f"Summary CSV: {csv_path.parent / 'benchmark_summary.csv'}")

    # Print compact summary table
    summary_csv = csv_path.parent / "benchmark_summary.csv"
    if summary_csv.exists():
        rows = []
        with summary_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        print(f"\n{'Sample':<20} {'RelErr':>8} {'Pass':>5} {'X_range':>20} {'Y_range':>20} {'Dominant':>16}")
        print("-" * 95)
        for r in rows:
            def _fmt(v):
                try:
                    return f"{float(v):.4g}"
                except (ValueError, TypeError):
                    return str(v) if v else "?"
            x_min = _fmt(r.get("primary_x_min"))
            x_max = _fmt(r.get("primary_x_max"))
            y_min = _fmt(r.get("primary_y_min"))
            y_max = _fmt(r.get("primary_y_max"))
            x_range = f"{x_min}..{x_max}" if x_min != "?" and x_max != "?" else "?"
            y_range = f"{y_min}..{y_max}" if y_min != "?" and y_max != "?" else "?"
            print(
                f"{r['sample']:<20} {float(r['rel_err']):>8.4f} {r['passed']:>5} "
                f"{x_range:>20} {y_range:>20} {r['dominant_failure']:>16}"
            )
        passed_count = sum(1 for r in rows if r["passed"] == "True")
        print(f"\nPassed: {passed_count}/{len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
