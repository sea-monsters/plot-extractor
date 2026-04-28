"""Per-type validation: run extraction on all images in test_data/<type>/.

Scoring discipline:
- Meta JSON is NEVER passed to extract_from_image (no cheating).
- Meta is loaded after extraction for ground-truth comparison only.
- Chart-type guess accuracy and policy routing metrics are collected
  per-image and summarized per-type.
"""
import csv
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from plot_extractor.core.chart_type_guesser import extract_all_features, guess_chart_type
from plot_extractor.core.policy_router import compute_policy
from plot_extractor.core.image_loader import load_image, to_grayscale

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
SUPPORTED_V4_DATASETS = {"v4", "v4a"}

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
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _guess_and_route(img_path: Path):
    """Run feature extraction, chart type guess, and policy computation.

    Returns (type_probs, top1_type, top2_types, policy)."""
    raw = load_image(img_path)
    gray = to_grayscale(raw)
    features = extract_all_features(raw, gray)
    type_probs = guess_chart_type(features)
    policy = compute_policy(features, type_probs)

    sorted_types = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)
    top1 = sorted_types[0][0]
    top2 = [t for t, _ in sorted_types[:2]]

    return type_probs, top1, top2, policy


def _evaluate_image(
    img_path: Path,
    chart_type: str,
    debug: bool = False,
    debug_subdir: Path | None = None,
    use_llm: bool = False,
    use_ocr: bool = False,
    formula_batch_max_crops: int | None = None,
    quiet: bool = False,
) -> dict:
    # --- Load meta for scoring ONLY; never pass to extraction ---
    meta = _load_meta(img_path)
    data_threshold = DATA_ACCURACY_THRESHOLDS.get(chart_type, 0.05)

    # --- Routing metrics (independent of extraction) ---
    type_probs, top1_guess, top2_guesses, policy = _guess_and_route(img_path)
    top1_correct = top1_guess == chart_type
    top2_correct = chart_type in top2_guesses

    csv_path = None
    dbg_dir = None
    if debug:
        dbg_dir = debug_subdir or (DEBUG_DIR / chart_type)
        dbg_dir.mkdir(parents=True, exist_ok=True)
        csv_path = dbg_dir / f"{img_path.stem}.csv"

    # --- Extract WITHOUT meta ---
    try:
        result = extract_from_image(
            img_path, output_csv=csv_path, debug_dir=dbg_dir,
            use_llm=use_llm, use_ocr=use_ocr,
            formula_batch_max_crops=formula_batch_max_crops,
        )
    except (OSError, ValueError, RuntimeError) as e:
        if not quiet:
            print(f"    ERROR {img_path.name}: {e}")
        return {
            "file": img_path.name,
            "rel_err": 1.0,
            "ssim": 0.0,
            "threshold": data_threshold,
            "passed": False,
            "error": str(e),
            "guess_top1": top1_guess,
            "guess_top1_correct": top1_correct,
            "guess_top2_correct": top2_correct,
            "guess_probs": type_probs,
            "policy_density": policy.density_strategy,
            "policy_color": policy.color_strategy,
            "policy_noise": policy.noise_strategy,
        }

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

    row = {
        "file": img_path.name,
        "rel_err": round(rel_err, 4),
        "ssim": round(ssim, 4),
        "threshold": data_threshold,
        "passed": passed,
        "guess_top1": top1_guess,
        "guess_top1_correct": top1_correct,
        "guess_top2_correct": top2_correct,
        "guess_probs": type_probs,
        "policy_density": policy.density_strategy,
        "policy_color": policy.color_strategy,
        "policy_noise": policy.noise_strategy,
    }
    if diagnostics:
        row["diagnostics"] = diagnostics

    status = "PASS" if passed else "FAIL"
    guess_flag = "+" if top1_correct else ("~" if top2_correct else "-")
    if not quiet:
        print(f"    {img_path.name}: rel_err={mae_str} SSIM={ssim:.3f} [{status}] guess={top1_guess}{guess_flag}")
        if debug and diagnostics:
            axis_summary = ", ".join(
                f"{a['direction']}_{a['side']}:{a['axis_type']}/ticks={a['tick_count']}/"
                f"src={a.get('tick_source', 'heuristic')}/res={a['residual']:.2f}/"
                f"sel={a.get('formula_selected_count', 0)}/{a.get('formula_anchor_count', 0)}/"
                f"batch={a.get('formula_batch_requested', 0)}@{a.get('formula_batch_ms', 0.0)}ms/"
                f"keep={a.get('formula_batch_kept_count', 0)}/{a.get('formula_batch_candidate_count', 0)}/"
                f"chunks={a.get('formula_batch_chunks', 0)}"
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

    if tags.get("dataset") not in SUPPORTED_V4_DATASETS:
        return False, "not_v4_or_v4a", None
    if chart_count != 1 or len(chart_types) != 1:
        return False, "multi_or_combo_chart", chart_types[0] if chart_types else None

    chart_type = chart_types[0]
    if chart_type not in SUPPORTED_V4_SINGLE_TYPES:
        return False, f"unsupported_type:{chart_type}", chart_type

    distortions = set(tags.get("distortions") or [])
    if "partial_crop" in distortions:
        return False, "partial_crop", chart_type

    return True, "supported_single_chart", chart_type


def _routing_summary(results: list[dict], chart_type: str) -> dict:
    """Compute routing/guess accuracy metrics for a list of image results."""
    total = len(results)
    if total == 0:
        return {}

    top1_correct = sum(1 for r in results if r.get("guess_top1_correct"))
    top2_correct = sum(1 for r in results if r.get("guess_top2_correct"))

    # Strategy frequency counts
    density_counts = {}
    color_counts = {}
    noise_counts = {}
    for r in results:
        density_counts[r.get("policy_density", "unknown")] = density_counts.get(r.get("policy_density"), 0) + 1
        color_counts[r.get("policy_color", "unknown")] = color_counts.get(r.get("policy_color"), 0) + 1
        noise_counts[r.get("policy_noise", "unknown")] = noise_counts.get(r.get("policy_noise"), 0) + 1

    return {
        "guess_top1_acc": round(top1_correct / total, 3),
        "guess_top2_acc": round(top2_correct / total, 3),
        "density_strategies": density_counts,
        "color_strategies": color_counts,
        "noise_strategies": noise_counts,
    }


def _evaluate_image_worker(args):
    """Process-pool entry point for evaluating a single image."""
    img_path, chart_type, debug, debug_subdir, use_llm, use_ocr, formula_batch_max_crops = args
    return _evaluate_image(
        Path(img_path), chart_type, debug=debug, debug_subdir=Path(debug_subdir) if debug_subdir else None,
        use_llm=use_llm, use_ocr=use_ocr, formula_batch_max_crops=formula_batch_max_crops, quiet=True,
    )


def validate_type(
    chart_type: str,
    debug: bool = False,
    data_dir: Path | None = None,
    use_llm: bool = False,
    use_ocr: bool = False,
    workers: int = 1,
    formula_batch_max_crops: int | None = None,
) -> dict:
    """Run validation on all images of one type, return aggregate stats."""
    base_dir = data_dir or TEST_DATA_DIR
    type_dir = base_dir / chart_type
    if not type_dir.exists():
        print(f"  SKIP {chart_type}: directory not found")
        return {"type": chart_type, "total": 0, "passed": 0, "pass_rate": 0, "avg_rel_err": 0, "max_rel_err": 0, "results": []}

    image_files = sorted(type_dir.glob("*.png"))
    results = []

    if debug:
        type_debug = DEBUG_DIR / chart_type
        type_debug.mkdir(parents=True, exist_ok=True)

    if workers > 1:
        work_items = [
            (str(p), chart_type, debug, str(DEBUG_DIR / chart_type) if debug else None, use_llm, use_ocr, formula_batch_max_crops)
            for p in image_files
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_evaluate_image_worker, item): idx
                for idx, item in enumerate(work_items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    row = future.result()
                except Exception as e:
                    row = {
                        "file": image_files[idx].name,
                        "rel_err": 1.0,
                        "ssim": 0.0,
                        "threshold": DATA_ACCURACY_THRESHOLDS.get(chart_type, 0.05),
                        "passed": False,
                        "error": str(e),
                        "guess_top1": "unknown",
                        "guess_top1_correct": False,
                        "guess_top2_correct": False,
                        "policy_density": "unknown",
                        "policy_color": "unknown",
                        "policy_noise": "unknown",
                    }
                results.append((idx, row))
        results.sort(key=lambda x: x[0])
        results = [r for _, r in results]
    else:
        for img_path in image_files:
            row = _evaluate_image(
                img_path, chart_type, debug=debug, debug_subdir=DEBUG_DIR / chart_type,
                use_llm=use_llm, use_ocr=use_ocr, formula_batch_max_crops=formula_batch_max_crops,
            )
            results.append(row)

    # Print results (needed for parallel mode where workers were quiet)
    for row in results:
        if "error" in row:
            print(f"    ERROR {row['file']}: {row['error']}")
            continue
        mae_str = "∞" if row['rel_err'] == float('inf') else f"{row['rel_err']:.4f}"
        status = "PASS" if row['passed'] else "FAIL"
        guess_flag = "+" if row['guess_top1_correct'] else ("~" if row['guess_top2_correct'] else "-")
        print(f"    {row['file']}: rel_err={mae_str} SSIM={row['ssim']:.3f} [{status}] guess={row['guess_top1']}{guess_flag}")

    if not results:
        return {"type": chart_type, "total": 0, "passed": 0, "pass_rate": 0, "avg_rel_err": 0, "max_rel_err": 0, "results": []}

    passed_count = sum(1 for r in results if r["passed"])
    rel_err_values = [r["rel_err"] for r in results]
    failed = [r for r in results if not r["passed"]]

    routing = _routing_summary(results, chart_type)

    summary = {
        "type": chart_type,
        "total": len(results),
        "passed": passed_count,
        "pass_rate": round(passed_count / len(results), 3),
        "avg_rel_err": round(sum(rel_err_values) / len(rel_err_values), 4),
        "max_rel_err": round(max(rel_err_values), 4),
        "results": results,
        "routing": routing,
    }

    print(f"  => {chart_type}: {passed_count}/{len(results)} passed ({summary['pass_rate']:.1%}), "
          f"avg_rel_err={summary['avg_rel_err']:.4f}, max_rel_err={summary['max_rel_err']:.4f}")
    print(f"     Routing  top1={routing.get('guess_top1_acc', 0):.1%} top2={routing.get('guess_top2_acc', 0):.1%} "
          f"density={routing.get('density_strategies', {})} color={routing.get('color_strategies', {})} noise={routing.get('noise_strategies', {})}")
    if failed:
        print(f"     Failed: {', '.join(r['file'] for r in failed)}")

    return summary


def validate_v4_special(
    data_dir: Path,
    debug: bool = False,
    types: list[str] | None = None,
    use_llm: bool = False,
    use_ocr: bool = False,
    formula_batch_max_crops: int | None = None,
):
    """Validate v4/v4a with explicit supported-domain and out-of-scope accounting."""
    image_files = sorted(data_dir.glob("*/*.png"))
    report_path = data_dir.parent / f"report_{data_dir.name}_special.csv"
    scope_path = data_dir.parent / f"report_{data_dir.name}_scope.csv"
    requested_types = set(types or [])

    in_scope_rows = []
    out_scope_rows = []
    summaries: dict[str, dict] = {}

    print(f"\n--- v4 special validation (v4/v4a): {data_dir} ---")
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
        row = _evaluate_image(
            img_path, chart_type, debug=debug, debug_subdir=debug_dir,
            use_llm=use_llm, use_ocr=use_ocr,
            formula_batch_max_crops=formula_batch_max_crops,
        )
        row.update(scope_row)
        row["type"] = chart_type
        in_scope_rows.append(row)

        s = summaries.setdefault(chart_type, {"type": chart_type, "total": 0, "passed": 0, "rel_errs": [], "routing_rows": []})
        s["total"] += 1
        s["passed"] += 1 if row["passed"] else 0
        s["rel_errs"].append(row["rel_err"])
        s["routing_rows"].append(row)

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "type", "file", "rel_err", "ssim", "threshold", "passed",
            "chart_type", "chart_types", "chart_count", "distortions", "scope", "reason",
            "guess_top1", "guess_top1_correct", "guess_top2_correct",
            "policy_density", "policy_color", "policy_noise",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(in_scope_rows)

    with open(scope_path, "w", newline="", encoding="utf-8") as f:
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
    print(f"{'Type':<18} {'Pass':>6} {'Rate':>7} {'AvgErr':>8} {'MaxErr':>8} {'Top1':>6} {'Top2':>6}")
    print("-" * 86)
    total_pass = 0
    total_count = 0
    for chart_type in sorted(summaries):
        s = summaries[chart_type]
        rel_errs = s["rel_errs"] or [0]
        pass_rate = s["passed"] / max(s["total"], 1)
        routing = _routing_summary(s["routing_rows"], chart_type)
        print(f"{chart_type:<18} {s['passed']:>3}/{s['total']:<3} {pass_rate:>6.1%} {sum(rel_errs) / len(rel_errs):>8.4f} {max(rel_errs):>8.4f} "
              f"{routing.get('guess_top1_acc', 0):>5.1%} {routing.get('guess_top2_acc', 0):>5.1%}")
        total_pass += s["passed"]
        total_count += s["total"]
    print("-" * 86)
    print(f"{'SUPPORTED TOTAL':<18} {total_pass:>3}/{total_count:<3} {total_pass / max(total_count, 1):>6.1%}")
    print(f"\nDetail report saved to {report_path}")
    print(f"Scope report saved to {scope_path}")

    return summaries, in_scope_rows, out_scope_rows


def run_all(
    types: list[str] | None = None,
    debug: bool = False,
    data_dir: Path | None = None,
    use_llm: bool = False,
    use_ocr: bool = False,
    workers: int = 1,
    formula_batch_max_crops: int | None = None,
):
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
        summary = validate_type(
            chart_type, debug=debug, data_dir=data_dir, use_llm=use_llm,
            use_ocr=use_ocr, workers=workers,
            formula_batch_max_crops=formula_batch_max_crops,
        )
        summaries.append(summary)
        for r in summary["results"]:
            row = {"type": chart_type, **r}
            row.pop("diagnostics", None)
            row.pop("guess_probs", None)
            all_rows.append(row)

    # Write detailed CSV
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "type", "file", "rel_err", "ssim", "threshold", "passed",
            "guess_top1", "guess_top1_correct", "guess_top2_correct",
            "policy_density", "policy_color", "policy_noise",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    print("\n" + "=" * 86)
    print(f"{'Type':<18} {'Pass':>6} {'Rate':>7} {'AvgErr':>8} {'MaxErr':>8} {'Top1':>6} {'Top2':>6}")
    print("-" * 86)
    total_pass = 0
    total_count = 0
    total_top1_correct = 0
    total_top2_correct = 0
    for s in summaries:
        routing = s.get("routing", {})
        top1_acc = routing.get("guess_top1_acc", 0)
        top2_acc = routing.get("guess_top2_acc", 0)
        total_top1_correct += int(top1_acc * s["total"])
        total_top2_correct += int(top2_acc * s["total"])
        print(f"{s['type']:<18} {s['passed']:>3}/{s['total']:<3} {s['pass_rate']:>6.1%} {s['avg_rel_err']:>8.4f} {s['max_rel_err']:>8.4f} "
              f"{top1_acc:>5.1%} {top2_acc:>5.1%}")
        total_pass += s["passed"]
        total_count += s["total"]
    print("-" * 86)
    print(f"{'TOTAL':<18} {total_pass:>3}/{total_count:<3} {total_pass / max(total_count, 1):>6.1%} "
          f"{'':>8} {'':>8} {total_top1_correct / max(total_count, 1):>5.1%} {total_top2_correct / max(total_count, 1):>5.1%}")
    print(f"\nReport saved to {report_path}")

    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="*", default=None, help="Chart types to validate")
    parser.add_argument("--debug", action="store_true", help="Save debug outputs")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: test_data/)")
    parser.add_argument("--v4-special", action="store_true", help="Validate v4 using supported-domain filtering and scope accounting")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM vision enhancement for ambiguous charts")
    parser.add_argument("--use-ocr", action="store_true", help="Enable tesseract OCR for tick labels (requires tesseract)")
    parser.add_argument("--formula-batch-max-crops", type=int, default=None, help="Override FormulaOCR batch crop cap for staged validation")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (processes) for image evaluation")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    if args.v4_special:
        validate_v4_special(
            data_dir or (Path(__file__).parent.parent / "test_data_v4"),
            debug=args.debug,
            types=args.types,
            use_llm=args.use_llm,
            use_ocr=args.use_ocr,
            formula_batch_max_crops=args.formula_batch_max_crops,
        )
    else:
        run_all(
            types=args.types,
            debug=args.debug,
            data_dir=data_dir,
            use_llm=args.use_llm,
            use_ocr=args.use_ocr,
            workers=args.workers,
            formula_batch_max_crops=args.formula_batch_max_crops,
        )
