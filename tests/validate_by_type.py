"""Per-type validation: run extraction on all images in test_data/<type>/."""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.main import extract_from_image
from plot_extractor.config import (
    SSIM_THRESHOLD_SIMPLE,
    SSIM_THRESHOLD_LOG,
    SSIM_THRESHOLD_SCATTER,
    SSIM_THRESHOLD_MULTI,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
DEBUG_DIR = Path(__file__).parent.parent / "debug_type"
REPORT_PATH = Path(__file__).parent.parent / "report_by_type.csv"

TYPE_THRESHOLDS = {
    "simple_linear": SSIM_THRESHOLD_SIMPLE,
    "log_y": SSIM_THRESHOLD_LOG,
    "loglog": SSIM_THRESHOLD_LOG,
    "dual_y": SSIM_THRESHOLD_LOG,
    "inverted_y": SSIM_THRESHOLD_SIMPLE,
    "scatter": SSIM_THRESHOLD_SCATTER,
    "multi_series": SSIM_THRESHOLD_MULTI,
    "log_x": SSIM_THRESHOLD_LOG,
    "no_grid": SSIM_THRESHOLD_SIMPLE,
    "dense": SSIM_THRESHOLD_SIMPLE,
}

SUPPORTED_V4_SINGLE_TYPES = set(TYPE_THRESHOLDS)

# Data-level accuracy thresholds (relative MAE in y)
DATA_ACCURACY_THRESHOLDS = {
    "simple_linear": 0.05,   # 5% relative error
    "log_y": 0.05,
    "loglog": 0.05,
    "dual_y": 0.05,
    "inverted_y": 0.05,
    "scatter": 0.08,         # scatter has more jitter
    "multi_series": 0.06,
    "log_x": 0.05,
    "no_grid": 0.05,
    "dense": 0.06,
}


def compare_series_accuracy(x_gt, y_gt, x_ext, y_ext, match_mode: str = "x"):
    """Compare extracted series to ground truth."""
    if not x_ext or not y_ext or len(x_ext) < 3:
        return {
            "mae_y": float("inf"),
            "rmse_y": float("inf"),
            "rel_err": 1.0,
            "max_err": float("inf"),
        }

    x_gt = np.asarray(x_gt, dtype=float)
    y_gt = np.asarray(y_gt, dtype=float)
    x_ext = np.asarray(x_ext, dtype=float)
    y_ext = np.asarray(y_ext, dtype=float)

    y_range = float(np.max(y_gt) - np.min(y_gt))
    if y_range == 0:
        y_range = 1.0

    x_range = float(np.max(x_gt) - np.min(x_gt))
    if x_range == 0:
        x_range = 1.0

    # For line charts, each x maps to one y. For scatter charts, repeated or
    # nearby x values are normal, so use normalized 2D nearest-neighbor matching.
    y_pred = []
    for xg, yg in zip(x_gt, y_gt):
        if match_mode == "xy":
            dist = ((x_ext - xg) / x_range) ** 2 + ((y_ext - yg) / y_range) ** 2
            idx = int(np.argmin(dist))
        else:
            idx = int(np.argmin(np.abs(x_ext - xg)))
        y_pred.append(y_ext[idx])
    y_pred = np.array(y_pred)

    abs_err = np.abs(y_gt - y_pred)
    mae_y = float(np.mean(abs_err))
    rmse_y = float(np.sqrt(np.mean(abs_err ** 2)))
    max_err = float(np.max(abs_err))
    rel_err = mae_y / y_range

    return {
        "mae_y": mae_y,
        "rmse_y": rmse_y,
        "rel_err": rel_err,
        "max_err": max_err,
        "y_range": y_range,
        "n_gt": len(x_gt),
        "n_ext": len(x_ext),
    }


def evaluate_data_accuracy(data, meta, chart_type: str | None = None):
    """Evaluate extracted data against meta ground truth.

    Returns (best_rel_err, per_series_errors) where best_rel_err is the
    maximum relative error across matched series (conservative: worst series
    determines pass/fail).
    """
    if not data or not meta or "data" not in meta:
        return 1.0, {}

    gt = meta["data"]
    errors = {}
    max_rel_err = 0.0

    # Match extracted series to ground truth. Color cluster order may not match
    # generation order, especially for multi-series charts with similar ranges.
    extracted_items = list(data.items())
    gt_items = list(gt.items())
    match_mode = "xy" if chart_type == "scatter" else "x"

    if len(extracted_items) <= 5 and len(gt_items) <= 5:
        n_match = min(len(extracted_items), len(gt_items))
        best_score = float("inf")
        best_errors = {}
        for perm in itertools.permutations(range(len(gt_items)), n_match):
            candidate_errors = {}
            candidate_score = 0.0
            for ext_idx, gt_idx in enumerate(perm):
                ext_name, ext_series = extracted_items[ext_idx]
                gt_name, gt_series = gt_items[gt_idx]
                acc = compare_series_accuracy(
                    gt_series["x"], gt_series["y"],
                    ext_series["x"], ext_series["y"],
                    match_mode=match_mode,
                )
                candidate_errors[f"{ext_name}→{gt_name}"] = acc
                candidate_score = max(candidate_score, acc["rel_err"])
            if candidate_score < best_score:
                best_score = candidate_score
                best_errors = candidate_errors
        errors = best_errors
        max_rel_err = best_score if best_score != float("inf") else 1.0
    else:
        # Fallback for unexpectedly large series counts: match by y-range.
        ext_y_ranges = [(name, max(d["y"]) - min(d["y"])) for name, d in extracted_items]
        gt_y_ranges = [(name, max(d["y"]) - min(d["y"])) for name, d in gt_items]
        ext_y_ranges.sort(key=lambda x: x[1], reverse=True)
        gt_y_ranges.sort(key=lambda x: x[1], reverse=True)

        for i, (ext_name, _) in enumerate(ext_y_ranges):
            if i < len(gt_y_ranges):
                gt_name = gt_y_ranges[i][0]
                acc = compare_series_accuracy(
                    gt[gt_name]["x"], gt[gt_name]["y"],
                    data[ext_name]["x"], data[ext_name]["y"],
                    match_mode=match_mode,
                )
                errors[f"{ext_name}→{gt_name}"] = acc
                max_rel_err = max(max_rel_err, acc["rel_err"])

    return max_rel_err, errors


def _load_meta(img_path: Path) -> dict | None:
    meta_path = img_path.parent / f"{img_path.stem}_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def _evaluate_image(img_path: Path, chart_type: str, debug: bool = False, debug_subdir: Path | None = None) -> dict:
    meta = _load_meta(img_path)
    data_threshold = DATA_ACCURACY_THRESHOLDS.get(chart_type, 0.05)

    csv_path = None
    dbg_dir = None
    if debug:
        dbg_dir = debug_subdir or (DEBUG_DIR / chart_type)
        dbg_dir.mkdir(parents=True, exist_ok=True)
        csv_path = dbg_dir / f"{img_path.stem}.csv"

    try:
        result = extract_from_image(img_path, output_csv=csv_path, debug_dir=dbg_dir, meta=meta)
    except Exception as e:
        print(f"    ERROR {img_path.name}: {e}")
        return {"file": img_path.name, "rel_err": 1.0, "ssim": 0.0, "threshold": data_threshold, "passed": False, "error": str(e)}

    diagnostics = result.get("diagnostics") if result else None
    if result and result.get("data"):
        rel_err, _ = evaluate_data_accuracy(result["data"], meta, chart_type=chart_type)
        ssim_full = result.get("ssim") or 0.0
        ssim_crop = result.get("ssim_crop") or 0.0
        ssim = max(ssim_full, ssim_crop)
        passed = rel_err <= data_threshold
        mae_str = "∞" if rel_err == float("inf") else f"{rel_err:.4f}"
    else:
        rel_err = 1.0
        ssim = 0.0
        passed = False
        mae_str = "N/A"

    row = {"file": img_path.name, "rel_err": round(rel_err, 4), "ssim": round(ssim, 4), "threshold": data_threshold, "passed": passed}
    if diagnostics:
        row["diagnostics"] = diagnostics

    status = "PASS" if passed else "FAIL"
    print(f"    {img_path.name}: rel_err={mae_str} SSIM={ssim:.3f} [{status}]")
    if debug and diagnostics:
        axis_summary = ", ".join(
            f"{a['direction']}_{a['side']}:{a['axis_type']}/ticks={a['tick_count']}/res={a['residual']:.2f}"
            for a in diagnostics["axes"]
        )
        series_summary = ", ".join(
            f"{name}:{info['points']}"
            for name, info in diagnostics["series"].items()
        )
        print(f"      diag grid={diagnostics['has_grid']} scatter={diagnostics['is_scatter']} axes=[{axis_summary}] series=[{series_summary}]")

    return row


def classify_v4_scope(meta: dict | None) -> tuple[bool, str, str | None]:
    """Return (in_scope, reason, chart_type) for current extractor capability."""
    if not meta:
        return False, "missing_meta", None

    tags = meta.get("tags") or {}
    chart_types = tags.get("chart_types") or []
    chart_count = int(tags.get("chart_count") or len(chart_types) or 0)

    if tags.get("dataset") != "v4":
        return False, "not_v4", None
    if chart_count != 1 or len(chart_types) != 1:
        return False, "multi_or_combo_chart", chart_types[0] if chart_types else None

    chart_type = chart_types[0]
    if chart_type not in SUPPORTED_V4_SINGLE_TYPES:
        return False, f"unsupported_type:{chart_type}", chart_type

    distortions = set(tags.get("distortions") or [])
    if "partial_crop" in distortions:
        return False, "partial_crop", chart_type

    return True, "supported_single_chart", chart_type


def validate_type(chart_type: str, debug: bool = False, data_dir: Path | None = None) -> dict:
    """Run validation on all images of one type, return aggregate stats."""
    base_dir = data_dir or TEST_DATA_DIR
    type_dir = base_dir / chart_type
    if not type_dir.exists():
        print(f"  SKIP {chart_type}: directory not found")
        return {"type": chart_type, "total": 0, "passed": 0, "pass_rate": 0, "avg_rel_err": 0, "max_rel_err": 0, "results": []}

    data_threshold = DATA_ACCURACY_THRESHOLDS.get(chart_type, 0.05)
    image_files = sorted(type_dir.glob("*.png"))
    results = []

    if debug:
        type_debug = DEBUG_DIR / chart_type
        type_debug.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        row = _evaluate_image(img_path, chart_type, debug=debug, debug_subdir=DEBUG_DIR / chart_type)
        results.append(row)

    if not results:
        return {"type": chart_type, "total": 0, "passed": 0, "pass_rate": 0, "avg_rel_err": 0, "max_rel_err": 0, "results": []}

    passed_count = sum(1 for r in results if r["passed"])
    rel_err_values = [r["rel_err"] for r in results]
    failed = [r for r in results if not r["passed"]]

    summary = {
        "type": chart_type,
        "total": len(results),
        "passed": passed_count,
        "pass_rate": round(passed_count / len(results), 3),
        "avg_rel_err": round(sum(rel_err_values) / len(rel_err_values), 4),
        "max_rel_err": round(max(rel_err_values), 4),
        "results": results,
    }

    print(f"  => {chart_type}: {passed_count}/{len(results)} passed ({summary['pass_rate']:.1%}), "
          f"avg_rel_err={summary['avg_rel_err']:.4f}, max_rel_err={summary['max_rel_err']:.4f}")
    if failed:
        print(f"     Failed: {', '.join(r['file'] for r in failed)}")

    return summary


def validate_v4_special(data_dir: Path, debug: bool = False, types: list[str] | None = None):
    """Validate v4 with explicit supported-domain and out-of-scope accounting."""
    image_files = sorted(data_dir.glob("*/*.png"))
    report_path = data_dir.parent / f"report_{data_dir.name}_special.csv"
    scope_path = data_dir.parent / f"report_{data_dir.name}_scope.csv"
    requested_types = set(types or [])

    in_scope_rows = []
    out_scope_rows = []
    summaries: dict[str, dict] = {}

    print(f"\n--- v4 special validation: {data_dir} ---")
    for img_path in image_files:
        meta = _load_meta(img_path)
        in_scope, reason, chart_type = classify_v4_scope(meta)
        tags = (meta or {}).get("tags") or {}
        chart_types = tags.get("chart_types") or []
        distortions = tags.get("distortions") or []
        rel_path = img_path.relative_to(data_dir).as_posix()

        scope_row = {
            "file": rel_path,
            "chart_type": chart_type or "",
            "chart_types": "|".join(chart_types),
            "chart_count": tags.get("chart_count", ""),
            "distortions": "|".join(distortions),
            "scope": "in" if in_scope else "out",
            "reason": reason,
        }

        if requested_types and chart_type not in requested_types:
            scope_row["scope"] = "out"
            scope_row["reason"] = "filtered_type"
            out_scope_rows.append(scope_row)
            continue

        if not in_scope or not chart_type:
            out_scope_rows.append(scope_row)
            continue

        print(f"\n--- {rel_path} ({chart_type}) ---")
        debug_dir = DEBUG_DIR / "v4_special" / chart_type if debug else None
        row = _evaluate_image(img_path, chart_type, debug=debug, debug_subdir=debug_dir)
        row.update(scope_row)
        row["type"] = chart_type
        in_scope_rows.append(row)

        s = summaries.setdefault(chart_type, {"type": chart_type, "total": 0, "passed": 0, "rel_errs": []})
        s["total"] += 1
        s["passed"] += 1 if row["passed"] else 0
        s["rel_errs"].append(row["rel_err"])

    with open(report_path, "w", newline="") as f:
        fieldnames = [
            "type", "file", "rel_err", "ssim", "threshold", "passed",
            "chart_type", "chart_types", "chart_count", "distortions", "scope", "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(in_scope_rows)

    with open(scope_path, "w", newline="") as f:
        fieldnames = ["file", "chart_type", "chart_types", "chart_count", "distortions", "scope", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(in_scope_rows)
        writer.writerows(out_scope_rows)

    print("\n" + "=" * 86)
    print(f"{'V4 Scope':<22} {'Count':>7}")
    print("-" * 86)
    print(f"{'supported/in-scope':<22} {len(in_scope_rows):>7}")
    print(f"{'out-of-scope':<22} {len(out_scope_rows):>7}")
    print(f"{'total':<22} {len(in_scope_rows) + len(out_scope_rows):>7}")

    print("\n" + "=" * 86)
    print(f"{'Type':<18} {'Pass':>6} {'Rate':>7} {'AvgErr':>8} {'MaxErr':>8}")
    print("-" * 86)
    total_pass = 0
    total_count = 0
    for chart_type in sorted(summaries):
        s = summaries[chart_type]
        rel_errs = s["rel_errs"] or [0]
        pass_rate = s["passed"] / max(s["total"], 1)
        print(f"{chart_type:<18} {s['passed']:>3}/{s['total']:<3} {pass_rate:>6.1%} {sum(rel_errs) / len(rel_errs):>8.4f} {max(rel_errs):>8.4f}")
        total_pass += s["passed"]
        total_count += s["total"]
    print("-" * 86)
    print(f"{'SUPPORTED TOTAL':<18} {total_pass:>3}/{total_count:<3} {total_pass / max(total_count, 1):>6.1%}")
    print(f"\nDetail report saved to {report_path}")
    print(f"Scope report saved to {scope_path}")

    return summaries, in_scope_rows, out_scope_rows


def run_all(types: list[str] | None = None, debug: bool = False, data_dir: Path | None = None):
    """Run validation across specified types (or all)."""
    if types is None:
        types = sorted(TYPE_THRESHOLDS.keys())

    base_dir = data_dir or TEST_DATA_DIR
    report_dir = base_dir
    report_path = report_dir.parent / f"report_{base_dir.name}.csv"

    all_rows = []
    summaries = []

    for chart_type in types:
        print(f"\n--- {chart_type} ---")
        summary = validate_type(chart_type, debug=debug, data_dir=data_dir)
        summaries.append(summary)
        for r in summary["results"]:
            row = {"type": chart_type, **r}
            row.pop("diagnostics", None)
            all_rows.append(row)

    # Write detailed CSV
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "file", "rel_err", "ssim", "threshold", "passed"])
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Type':<18} {'Pass':>6} {'Rate':>7} {'AvgErr':>8} {'MaxErr':>8}")
    print("-" * 70)
    total_pass = 0
    total_count = 0
    for s in summaries:
        print(f"{s['type']:<18} {s['passed']:>3}/{s['total']:<3} {s['pass_rate']:>6.1%} {s['avg_rel_err']:>8.4f} {s['max_rel_err']:>8.4f}")
        total_pass += s["passed"]
        total_count += s["total"]
    print("-" * 70)
    print(f"{'TOTAL':<18} {total_pass:>3}/{total_count:<3} {total_pass / max(total_count, 1):>6.1%}")
    print(f"\nReport saved to {report_path}")

    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="*", default=None, help="Chart types to validate")
    parser.add_argument("--debug", action="store_true", help="Save debug outputs")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: test_data/)")
    parser.add_argument("--v4-special", action="store_true", help="Validate v4 using supported-domain filtering and scope accounting")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    if args.v4_special:
        validate_v4_special(data_dir or (Path(__file__).parent.parent / "test_data_v4"), debug=args.debug, types=args.types)
    else:
        run_all(types=args.types, debug=args.debug, data_dir=data_dir)
