"""Per-type validation: run extraction on all images in test_data/<type>/."""
import csv
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


def compare_series_accuracy(x_gt, y_gt, x_ext, y_ext):
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

    # For each ground truth point, find nearest extracted point by x
    y_pred = []
    for xg in x_gt:
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


def evaluate_data_accuracy(data, meta):
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

    # Match extracted series to ground truth by y-range similarity
    # (color cluster order may not match ground truth series order)
    extracted_items = list(data.items())
    gt_items = list(gt.items())

    # Compute y ranges for both
    ext_y_ranges = [(name, max(d["y"]) - min(d["y"])) for name, d in extracted_items]
    gt_y_ranges = [(name, max(d["y"]) - min(d["y"])) for name, d in gt_items]

    # Sort both by y range (descending)
    ext_y_ranges.sort(key=lambda x: x[1], reverse=True)
    gt_y_ranges.sort(key=lambda x: x[1], reverse=True)

    # Match by position in sorted list
    matched = {}
    for i, (ext_name, ext_range) in enumerate(ext_y_ranges):
        if i < len(gt_y_ranges):
            gt_name = gt_y_ranges[i][0]
            matched[ext_name] = gt_name

    # Compare matched series
    for ext_name, gt_name in matched.items():
        acc = compare_series_accuracy(
            gt[gt_name]["x"], gt[gt_name]["y"],
            data[ext_name]["x"], data[ext_name]["y"],
        )
        errors[f"{ext_name}→{gt_name}"] = acc
        max_rel_err = max(max_rel_err, acc["rel_err"])

    return max_rel_err, errors


def validate_type(chart_type: str, debug: bool = False) -> dict:
    """Run validation on all images of one type, return aggregate stats."""
    type_dir = TEST_DATA_DIR / chart_type
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
        meta_path = img_path.parent / f"{img_path.stem}_meta.json"
        meta = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        csv_path = (DEBUG_DIR / chart_type / f"{img_path.stem}.csv") if debug else None
        dbg_dir = (DEBUG_DIR / chart_type) if debug else None

        try:
            result = extract_from_image(img_path, output_csv=csv_path, debug_dir=dbg_dir, meta=meta)
        except Exception as e:
            print(f"    ERROR {img_path.name}: {e}")
            results.append({"file": img_path.name, "rel_err": 1.0, "threshold": data_threshold, "passed": False, "error": str(e)})
            continue

        diagnostics = result.get("diagnostics") if result else None
        if result and result.get("data"):
            rel_err, per_series = evaluate_data_accuracy(result["data"], meta)
            # Also keep SSIM for reference
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
        results.append(row)
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


def run_all(types: list[str] | None = None, debug: bool = False):
    """Run validation across specified types (or all)."""
    if types is None:
        types = sorted(TYPE_THRESHOLDS.keys())

    all_rows = []
    summaries = []

    for chart_type in types:
        print(f"\n--- {chart_type} ---")
        summary = validate_type(chart_type, debug=debug)
        summaries.append(summary)
        for r in summary["results"]:
            row = {"type": chart_type, **r}
            row.pop("diagnostics", None)
            all_rows.append(row)

    # Write detailed CSV
    with open(REPORT_PATH, "w", newline="") as f:
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
    print(f"\nReport saved to {REPORT_PATH}")

    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="*", default=None, help="Chart types to validate")
    parser.add_argument("--debug", action="store_true", help="Save debug outputs")
    args = parser.parse_args()
    run_all(types=args.types, debug=args.debug)
