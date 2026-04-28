"""CLI entry point for plot extraction."""
import argparse
import csv
import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from plot_extractor.core.image_loader import load_image, preprocess, to_grayscale, rotate_image
from plot_extractor.core.axis_detector import detect_all_axes, refine_rotation_angle
from plot_extractor.core.axis_calibrator import (
    FormulaLabelContext,
    calibrate_all_axes,
    prepare_formula_label_context,
)
from plot_extractor.core.data_extractor import extract_all_data
from plot_extractor.core.plot_rebuilder import rebuild_plot
from plot_extractor.core.chart_type_guesser import extract_all_features, guess_chart_type
from plot_extractor.core.llm_policy_router import compute_llm_enhanced_policy
from plot_extractor.core.formula_batch_queue import FormulaBatchQueue, FormulaQueueRequest
from plot_extractor.utils.ssim_compare import compare_images
from plot_extractor.layout.panel_split import detect_panel_boundaries, Panel


@dataclass
class _PreparedPanelBatch:
    image_path: Path
    panel: Panel
    raw_image: Any
    image: Any
    policy: Any
    final_type_probs: dict | None
    axes: list
    formula_context: FormulaLabelContext | None


def _build_diagnostics(calibrated_axes, data, plot_bounds, is_scatter, has_grid):
    """Summarize extraction internals without affecting behavior."""
    axes = []
    for ca in calibrated_axes:
        axis = ca.axis
        values = [v for _, v in ca.tick_map if v is not None]
        value_range = None
        if values:
            value_range = [float(min(values)), float(max(values))]
        axes.append({
            "direction": axis.direction,
            "side": axis.side,
            "position": int(axis.position),
            "plot_start": int(axis.plot_start),
            "plot_end": int(axis.plot_end),
            "tick_count": len(axis.ticks or []),
            "labeled_tick_count": len(values),
            "axis_type": ca.axis_type,
            "inverted": bool(ca.inverted),
            "residual": float(ca.residual),
            "tick_source": getattr(ca, "tick_source", "heuristic"),
            "anchor_count": int(getattr(ca, "anchor_count", 0)),
            "formula_anchor_count": int(getattr(ca, "formula_anchor_count", 0)),
            "formula_log_score": round(float(getattr(ca, "formula_log_score", 0.0)), 4),
            "formula_selected_count": int(getattr(ca, "formula_selected_count", 0)),
            "tesseract_anchor_count": int(getattr(ca, "tesseract_anchor_count", 0)),
            "label_anchor_count": int(getattr(ca, "label_anchor_count", 0)),
            "formula_batch_candidate_count": int(getattr(ca, "formula_batch_candidate_count", 0)),
            "formula_batch_kept_count": int(getattr(ca, "formula_batch_kept_count", 0)),
            "formula_batch_chunks": int(getattr(ca, "formula_batch_chunks", 0)),
            "formula_batch_requested": int(getattr(ca, "formula_batch_requested", 0)),
            "formula_batch_returned": int(getattr(ca, "formula_batch_returned", 0)),
            "formula_batch_ms": round(float(getattr(ca, "formula_batch_ms", 0.0)), 2),
            "value_range": value_range,
        })

    series = {}
    for name, series_data in data.items():
        xs = series_data.get("x", [])
        ys = series_data.get("y", [])
        series[name] = {
            "points": len(xs),
            "x_range": [float(min(xs)), float(max(xs))] if xs else None,
            "y_range": [float(min(ys)), float(max(ys))] if ys else None,
        }

    return {
        "axes": axes,
        "plot_bounds": [int(v) for v in plot_bounds],
        "is_scatter": bool(is_scatter),
        "has_grid": bool(has_grid),
        "series": series,
    }


def _extract_from_panel(
    panel: Panel,
    image_path: Path,
    debug_dir: Path = None,
    use_llm: bool = False,
    use_ocr: bool = False,
    formula_batch_max_crops: int | None = None,
    policy=None,
    final_type_probs=None,
):
    """Run extraction pipeline on a single panel."""
    raw_image = panel.image

    # Preprocess with policy-driven noise strategy
    image = preprocess(raw_image, denoise=True, policy=policy)
    gray = to_grayscale(image)

    # Rotation correction from policy (if enabled)
    use_rotated = False
    rot_angle = policy.rotation_angle if policy and policy.rotation_correct else 0.0

    if policy and policy.rotation_correct and abs(rot_angle) >= 0.3:
        # Coarse axis detection for angle refinement
        axes_coarse = detect_all_axes(gray, policy=policy)
        if axes_coarse:
            rot_angle = refine_rotation_angle(gray, axes_coarse, rot_angle)

        if abs(rot_angle) >= 0.3:
            print(
                f"[{image_path.name} Panel {panel.panel_id}] "
                f"Correcting rotation: {rot_angle:.2f} deg"
            )
            image = rotate_image(image, -rot_angle)
            gray = to_grayscale(image)
            raw_image = rotate_image(raw_image, -rot_angle)
            use_rotated = True

    # Detect axes on selected image
    axes = detect_all_axes(gray, policy=policy)
    if not axes:
        print(f"[{image_path.name} Panel {panel.panel_id}] No axes detected.")
        return None

    # Calibrate axes
    calibrated = calibrate_all_axes(
        axes, image, policy=policy, use_ocr=use_ocr, use_llm=use_llm,
        type_probs=final_type_probs, formula_batch_max_crops=formula_batch_max_crops,
    )
    if not calibrated:
        print(f"[{image_path.name} Panel {panel.panel_id}] Axis calibration failed.")
        return None

    # Extract data
    data, is_scatter, has_grid = extract_all_data(
        image, calibrated, image_path=image_path, raw_image=raw_image, policy=policy,
    )
    if not data:
        print(f"[{image_path.name} Panel {panel.panel_id}] No data extracted.")
        return None

    # Compute plot bounds from calibrated axes
    from plot_extractor.core.data_extractor import _get_plot_bounds
    plot_bounds = _get_plot_bounds(calibrated, panel.image.shape)

    return {
        "panel_id": panel.panel_id,
        "data": data,
        "calibrated_axes": calibrated,
        "is_scatter": is_scatter,
        "has_grid": has_grid,
        "plot_bounds": plot_bounds,
    }


def _prepare_panel_batch(
    panel: Panel,
    image_path: Path,
    debug_dir: Path = None,
    use_llm: bool = False,
    use_ocr: bool = False,
    policy=None,
    final_type_probs=None,
    formula_batch_max_crops: int | None = None,
):
    """Prepare a panel for batch calibration without running FormulaOCR."""
    raw_image = panel.image
    image = preprocess(raw_image, denoise=True, policy=policy)
    gray = to_grayscale(image)

    use_rotated = False
    rot_angle = policy.rotation_angle if policy and policy.rotation_correct else 0.0

    if policy and policy.rotation_correct and abs(rot_angle) >= 0.3:
        axes_coarse = detect_all_axes(gray, policy=policy)
        if axes_coarse:
            rot_angle = refine_rotation_angle(gray, axes_coarse, rot_angle)

        if abs(rot_angle) >= 0.3:
            print(
                f"[{image_path.name} Panel {panel.panel_id}] "
                f"Correcting rotation: {rot_angle:.2f} deg"
            )
            image = rotate_image(image, -rot_angle)
            gray = to_grayscale(image)
            raw_image = rotate_image(raw_image, -rot_angle)
            use_rotated = True

    axes = detect_all_axes(gray, policy=policy)
    if not axes:
        print(f"[{image_path.name} Panel {panel.panel_id}] No axes detected.")
        return None

    formula_context = None
    if use_ocr:
        formula_context = prepare_formula_label_context(
            axes,
            image,
            policy=policy,
            use_ocr=use_ocr,
            type_probs=final_type_probs,
            formula_batch_max_crops=formula_batch_max_crops,
        )

    return _PreparedPanelBatch(
        image_path=image_path,
        panel=panel,
        raw_image=raw_image,
        image=image,
        policy=policy,
        final_type_probs=final_type_probs,
        axes=axes,
        formula_context=formula_context,
    )


def _finalize_prepared_panel(
    prepared: _PreparedPanelBatch,
    formula_request_results: dict[tuple[int, int], tuple[str | None, float | None]] | None = None,
    formula_batch_stats=None,
    use_llm: bool = False,
    use_ocr: bool = False,
    formula_batch_max_crops: int | None = None,
):
    """Finish calibration and extraction for a prepared panel."""
    panel = prepared.panel
    image_path = prepared.image_path
    image = prepared.image
    raw_image = prepared.raw_image
    policy = prepared.policy

    calibrated = calibrate_all_axes(
        prepared.axes,
        image,
        policy=policy,
        use_ocr=use_ocr,
        use_llm=use_llm,
        type_probs=prepared.final_type_probs,
        formula_batch_max_crops=formula_batch_max_crops,
        formula_context=prepared.formula_context,
        formula_request_results=formula_request_results,
        formula_batch_stats=formula_batch_stats,
    )
    if not calibrated:
        print(f"[{image_path.name} Panel {panel.panel_id}] Axis calibration failed.")
        return None

    data, is_scatter, has_grid = extract_all_data(
        image,
        calibrated,
        image_path=image_path,
        raw_image=raw_image,
        policy=policy,
    )
    if not data:
        print(f"[{image_path.name} Panel {panel.panel_id}] No data extracted.")
        return None

    from plot_extractor.core.data_extractor import _get_plot_bounds
    plot_bounds = _get_plot_bounds(calibrated, panel.image.shape)

    return {
        "panel_id": panel.panel_id,
        "data": data,
        "calibrated_axes": calibrated,
        "is_scatter": is_scatter,
        "has_grid": has_grid,
        "plot_bounds": plot_bounds,
    }


def extract_from_images_batched(
    image_paths: list[Path],
    output_dir: Path | None = None,
    debug_dir: Path | None = None,
    use_llm: bool = False,
    use_ocr: bool = False,
    formula_batch_max_crops: int | None = None,
    formula_batch_size: int = 16,
):
    """Run extraction across many images while batching FormulaOCR crops."""
    items_by_image: dict[Path, list[_PreparedPanelBatch]] = {}
    queue = FormulaBatchQueue(batch_size=formula_batch_size)

    for image_path in image_paths:
        image_path = Path(image_path)
        raw_image = load_image(image_path)
        gray = to_grayscale(raw_image)
        features = extract_all_features(raw_image, gray)
        type_probs = guess_chart_type(features)
        policy, final_type_probs = compute_llm_enhanced_policy(
            image_path, features, type_probs, use_llm=use_llm,
        )
        panels = detect_panel_boundaries(raw_image)
        print(f"[{image_path.name}] Detected {len(panels)} panel(s)")

        for panel in panels:
            prepared = _prepare_panel_batch(
                panel,
                image_path,
                debug_dir=debug_dir,
                use_llm=use_llm,
                use_ocr=use_ocr,
                policy=policy,
                final_type_probs=final_type_probs,
                formula_batch_max_crops=formula_batch_max_crops,
            )
            if prepared is None:
                continue
            items_by_image.setdefault(image_path, []).append(prepared)

            if prepared.formula_context and prepared.formula_context.formula_plan.requests:
                for idx, req in enumerate(prepared.formula_context.formula_plan.requests):
                    request_id = (
                        f"{image_path.stem}:p{panel.panel_id}:label:{req.axis_id}:{req.anchor_idx}:{idx}"
                    )
                    queue.add(
                        FormulaQueueRequest(
                            request_id=request_id,
                            crop=req.crop,
                            metadata={
                                "image_path": str(image_path),
                                "panel_id": panel.panel_id,
                                "axis_id": req.axis_id,
                                "anchor_idx": req.anchor_idx,
                                "score": req.score,
                                "kind": "label",
                            },
                        )
                    )

    queue_results = queue.run() if len(queue) else {}
    queue_stats = queue.last_stats()

    formula_results_by_panel: dict[tuple[str, int], dict[tuple[int, int], tuple[str | None, float | None]]] = {}
    for result in queue_results.values():
        meta = result.metadata or {}
        image_key = (str(meta.get("image_path", "")), int(meta.get("panel_id", -1)))
        axis_id = meta.get("axis_id")
        anchor_idx = meta.get("anchor_idx")
        if axis_id is None or anchor_idx is None:
            continue
        panel_results = formula_results_by_panel.setdefault(image_key, {})
        panel_results[(int(axis_id), int(anchor_idx))] = (result.latex, result.value)

    results = []
    for image_path, panels in items_by_image.items():
        panel_results = []
        for prepared in panels:
            formula_request_results = formula_results_by_panel.get(
                (str(image_path), int(prepared.panel.panel_id)),
                {},
            )
            finalized = _finalize_prepared_panel(
                prepared,
                formula_request_results=formula_request_results,
                formula_batch_stats=queue_stats,
                use_llm=use_llm,
                use_ocr=use_ocr,
                formula_batch_max_crops=formula_batch_max_crops,
            )
            if finalized:
                panel_results.append(finalized)

        if not panel_results:
            print(f"[{image_path.name}] No data extracted from any panel.")
            continue

        if len(panel_results) == 1:
            single_result = panel_results[0]
            data = single_result["data"]
            calibrated = single_result["calibrated_axes"]
            plot_bounds = single_result["plot_bounds"]
            is_scatter = single_result["is_scatter"]
            has_grid = single_result["has_grid"]
        else:
            data = {}
            calibrated = []
            for pr in panel_results:
                panel_id = pr["panel_id"]
                for series_name, series_data in pr["data"].items():
                    merged_name = f"panel{panel_id}_{series_name}"
                    data[merged_name] = series_data
                calibrated.extend(pr["calibrated_axes"])
            plot_bounds = panel_results[0]["plot_bounds"]
            is_scatter = panel_results[0]["is_scatter"]
            has_grid = panel_results[0]["has_grid"]

        if output_dir is None:
            output_csv = image_path.parent / f"{image_path.stem}.csv"
        else:
            output_csv = Path(output_dir) / f"{image_path.stem}.csv"
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["series", "x", "y"])
            for series_name, series_data in data.items():
                for x, y in zip(series_data["x"], series_data["y"]):
                    writer.writerow([series_name, x, y])
        total_pts = sum(len(s["x"]) for s in data.values())
        print(f"[{image_path.name}] CSV saved: {output_csv} ({total_pts} points)")

        ssim_score = None
        ssim_cropped = None
        if debug_dir:
            img_debug_dir = Path(debug_dir) / image_path.stem
            img_debug_dir.mkdir(parents=True, exist_ok=True)
            rebuilt_path = img_debug_dir / f"{image_path.stem}_rebuilt.png"
            rebuild_plot(
                data,
                calibrated,
                rebuilt_path,
                figsize=(6, 4),
                dpi=100,
                is_scatter=is_scatter,
                has_grid=has_grid,
            )
            ssim_score = compare_images(image_path, rebuilt_path)
            ssim_cropped = compare_images(image_path, rebuilt_path, crop_box=plot_bounds)
            print(f"[{image_path.name}] SSIM full: {ssim_score:.4f}, SSIM crop: {ssim_cropped:.4f}")

        diagnostics = _build_diagnostics(calibrated, data, plot_bounds, is_scatter, has_grid)
        results.append({
            "image_path": image_path,
            "data": data,
            "calibrated_axes": calibrated,
            "ssim": ssim_score,
            "ssim_crop": ssim_cropped,
            "csv": output_csv,
            "plot_bounds": plot_bounds,
            "is_scatter": is_scatter,
            "has_grid": has_grid,
            "diagnostics": diagnostics,
            "panel_count": len(panel_results),
            "formula_batch_requested": queue_stats.requested,
            "formula_batch_returned": queue_stats.returned,
            "formula_batch_chunks": queue_stats.chunks,
            "formula_batch_ms": queue_stats.elapsed_ms,
            "formula_batch_model_calls": queue_stats.model_calls,
        })

    return results


def extract_from_image(
    image_path: Path,
    output_csv: Path = None,
    debug_dir: Path = None,
    use_llm: bool = False,
    use_ocr: bool = False,
    formula_batch_max_crops: int | None = None,
):
    """Run full extraction pipeline on an image with panel-wise workflow."""
    image_path = Path(image_path)
    raw_image = load_image(image_path)

    # --- Policy routing: extract features and compute policy ---
    gray = to_grayscale(raw_image)
    features = extract_all_features(raw_image, gray)
    type_probs = guess_chart_type(features)
    policy, final_type_probs = compute_llm_enhanced_policy(
        image_path, features, type_probs, use_llm=use_llm,
    )

    # --- Panel detection ---
    panels = detect_panel_boundaries(raw_image)
    print(f"[{image_path.name}] Detected {len(panels)} panel(s)")

    panel_results = []
    for panel in panels:
        # --- Extract from this panel ---
        result = _extract_from_panel(
            panel, image_path, debug_dir,
            use_llm, use_ocr, formula_batch_max_crops, policy, final_type_probs,
        )

        if result:
            panel_results.append(result)

    if not panel_results:
        print(f"[{image_path.name}] No data extracted from any panel.")
        return None

    # --- Assemble results from all panels ---
    # For single panel, keep original structure
    if len(panel_results) == 1:
        single_result = panel_results[0]
        data = single_result["data"]
        calibrated = single_result["calibrated_axes"]
        plot_bounds = single_result["plot_bounds"]
        is_scatter = single_result["is_scatter"]
        has_grid = single_result["has_grid"]
    else:
        # Multi-panel: merge series with panel prefix
        data = {}
        calibrated = []
        for pr in panel_results:
            panel_id = pr["panel_id"]
            for series_name, series_data in pr["data"].items():
                merged_name = f"panel{panel_id}_{series_name}"
                data[merged_name] = series_data
            calibrated.extend(pr["calibrated_axes"])
        # Use first panel bounds as overall plot bounds (approximate)
        plot_bounds = panel_results[0]["plot_bounds"]
        is_scatter = panel_results[0]["is_scatter"]
        has_grid = panel_results[0]["has_grid"]

    # --- Write CSV output ---
    if output_csv is None:
        output_csv = image_path.parent / f"{image_path.stem}.csv"
    output_csv = Path(output_csv)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "x", "y"])
        for series_name, series_data in data.items():
            for x, y in zip(series_data["x"], series_data["y"]):
                writer.writerow([series_name, x, y])
    total_pts = sum(len(s["x"]) for s in data.values())
    print(f"[{image_path.name}] CSV saved: {output_csv} ({total_pts} points)")

    # --- Rebuild and compare ---
    ssim_score = None
    ssim_cropped = None
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(exist_ok=True)
        rebuilt_path = debug_dir / f"{image_path.stem}_rebuilt.png"
        rebuild_plot(data, calibrated, rebuilt_path, figsize=(6, 4), dpi=100,
                     is_scatter=is_scatter, has_grid=has_grid)
        ssim_score = compare_images(image_path, rebuilt_path)
        ssim_cropped = compare_images(image_path, rebuilt_path, crop_box=plot_bounds)
        print(f"[{image_path.name}] SSIM full: {ssim_score:.4f}, SSIM crop: {ssim_cropped:.4f}")

    diagnostics = _build_diagnostics(calibrated, data, plot_bounds, is_scatter, has_grid)

    return {
        "data": data,
        "calibrated_axes": calibrated,
        "ssim": ssim_score,
        "ssim_crop": ssim_cropped,
        "csv": output_csv,
        "plot_bounds": plot_bounds,
        "is_scatter": is_scatter,
        "has_grid": has_grid,
        "diagnostics": diagnostics,
        "panel_count": len(panel_results),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract data points from chart images.")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--debug", "-d", type=Path, default=None, help="Debug output directory")
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Enable LLM vision enhancement for ambiguous charts",
    )
    parser.add_argument(
        "--use-ocr", action="store_true",
        help="Enable tesseract OCR for tick labels (requires tesseract)",
    )
    args = parser.parse_args()

    extract_from_image(
        args.input, output_csv=args.output, debug_dir=args.debug,
        use_llm=args.use_llm, use_ocr=args.use_ocr,
    )


if __name__ == "__main__":
    main()
