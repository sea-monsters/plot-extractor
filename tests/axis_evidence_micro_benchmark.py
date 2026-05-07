"""Axis evidence micro-benchmark for log calibration failures.

This script runs a fixed, representative set of log-axis samples and records
axis-level evidence alongside data-level accuracy.  It is intentionally small:
use it before full validation when changing absolute decade inference.
"""
import argparse
import csv
import json
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
        rows.append({
            "sample": sample,
            "direction": direction,
            "side": axis.get("side"),
            "axis_type": axis.get("axis_type"),
            "tick_source": axis.get("tick_source"),
            "tick_count": axis.get("tick_count"),
            "labeled_tick_count": axis.get("labeled_tick_count"),
            "value_min": (axis.get("value_range") or [None, None])[0],
            "value_max": (axis.get("value_range") or [None, None])[1],
            "expected_type": expected.get("type"),
            "expected_min": expected.get("min"),
            "expected_max": expected.get("max"),
            "rel_err": rel_err,
            "passed": passed,
            "candidate_sources": json.dumps(candidate_sources, ensure_ascii=True),
            "observed_anchors": json.dumps(observed, ensure_ascii=True),
        })
    return rows


def run_benchmark(data_dir: Path, output_dir: Path, samples: list[str]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
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

    csv_path = output_dir / "axis_evidence_micro_benchmark.csv"
    json_path = output_dir / "axis_evidence_micro_benchmark.json"
    fieldnames = [
        "sample", "direction", "side", "axis_type", "tick_source",
        "tick_count", "labeled_tick_count", "value_min", "value_max",
        "expected_type", "expected_min", "expected_max", "rel_err", "passed",
        "candidate_sources", "observed_anchors",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    json_path.write_text(json.dumps({
        "summary": summary,
        "axes": all_rows,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("test_data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_type") / "axis_evidence_micro_benchmark",
    )
    parser.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    args = parser.parse_args()

    csv_path, json_path = run_benchmark(args.data_dir, args.output_dir, args.samples)
    print(f"Axis evidence CSV: {csv_path}")
    print(f"Axis evidence JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
