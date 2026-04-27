"""CLI entry point for plot extraction."""
import argparse
import csv
import json
from pathlib import Path

from plot_extractor.core.image_loader import load_image, preprocess, to_grayscale, rotate_image
from plot_extractor.core.axis_detector import detect_all_axes, estimate_rotation_angle
from plot_extractor.core.axis_calibrator import calibrate_all_axes
from plot_extractor.core.data_extractor import extract_all_data
from plot_extractor.core.plot_rebuilder import rebuild_plot
from plot_extractor.core.ocr_reader import load_meta_labels
from plot_extractor.utils.ssim_compare import compare_images


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


def _score_calibration_quality(calibrated_axes):
    """Score axis calibration quality for rotation-path selection.

    Higher score = better calibration. Uses labeled tick count and residual.
    """
    if not calibrated_axes:
        return -1.0

    score = 0.0
    for ca in calibrated_axes:
        labeled_count = len([v for _, v in ca.tick_map if v is not None])
        score += labeled_count * 10.0  # Each labeled tick contributes significantly
        # Penalize high residuals
        score -= min(ca.residual * 0.01, 50.0)  # Cap penalty to avoid over-penalizing outliers
    return score


def extract_from_image(
    image_path: Path,
    output_csv: Path = None,
    debug_dir: Path = None,
    meta=None,
):
    """Run full extraction pipeline on an image."""
    image_path = Path(image_path)
    image = load_image(image_path)
    image = preprocess(image, denoise=True)
    gray = to_grayscale(image)

    # Detect and correct rotation before axis detection
    rot_angle = estimate_rotation_angle(gray)

    # Rotation quality-gated selection for near-threshold angles
    use_rotated = False
    if abs(rot_angle) >= 0.5:
        # For near-threshold cases, evaluate both paths and choose better calibration
        if abs(rot_angle) < 2.0:  # Borderline zone: try both
            # Path A: no rotation
            gray_a = gray
            axes_a = detect_all_axes(gray_a)
            if axes_a:
                calibrated_a = calibrate_all_axes(axes_a, image, meta=meta)
                score_a = _score_calibration_quality(calibrated_a)
            else:
                score_a = -1.0

            # Path B: rotation
            image_b = rotate_image(image, -rot_angle)
            gray_b = to_grayscale(image_b)
            axes_b = detect_all_axes(gray_b)
            if axes_b:
                calibrated_b = calibrate_all_axes(axes_b, image_b, meta=meta)
                score_b = _score_calibration_quality(calibrated_b)
            else:
                score_b = -1.0

            # Choose better path
            if score_b > score_a and calibrated_b:
                print(f"[{image_path.name}] Rotating by {-rot_angle:.2f}° (detected {rot_angle:.2f}°, quality-gated: B={score_b:.1f} > A={score_a:.1f})")
                image = image_b
                gray = gray_b
                use_rotated = True
            elif calibrated_a:
                print(f"[{image_path.name}] Skipping rotation (detected {rot_angle:.2f}°, quality-gated: A={score_a:.1f} >= B={score_b:.1f})")
                # Keep original image
                use_rotated = False
            else:
                # Neither path has axes, fallback to original behavior
                print(f"[{image_path.name}] Rotating by {-rot_angle:.2f}° (detected {rot_angle:.2f}°, fallback: no axes in either path)")
                image = image_b
                gray = gray_b
                use_rotated = True
        else:
            # Strong rotation: apply directly (not borderline)
            print(f"[{image_path.name}] Rotating by {-rot_angle:.2f}° (detected {rot_angle:.2f}°, strong rotation)")
            image = rotate_image(image, -rot_angle)
            gray = to_grayscale(image)
            use_rotated = True

    # Detect axes on selected image
    axes = detect_all_axes(gray)
    if not axes:
        print(f"[{image_path.name}] No axes detected.")
        return None

    # Calibrate axes
    calibrated = calibrate_all_axes(axes, image, meta=meta)
    if not calibrated:
        print(f"[{image_path.name}] Axis calibration failed.")
        return None

    # Extract data (raw image for grid detection, preprocessed for data extraction)
    raw_image = load_image(image_path)
    if use_rotated:
        raw_image = rotate_image(raw_image, -rot_angle)
    data, is_scatter, has_grid = extract_all_data(
        image, calibrated, image_path=image_path, raw_image=raw_image, meta=meta,
    )
    if not data:
        print(f"[{image_path.name}] No data extracted.")
        return None

    # Write CSV
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

    # Compute plot bounds for crop comparison
    x_axes = [ca for ca in calibrated if ca.axis.direction == "x"]
    y_axes = [ca for ca in calibrated if ca.axis.direction == "y"]
    h, w = image.shape[:2]
    left = max((ca.axis.position for ca in y_axes if ca.axis.side == "left"), default=0)
    right = min((ca.axis.position for ca in y_axes if ca.axis.side == "right"), default=w)
    top = min((ca.axis.position for ca in x_axes if ca.axis.side == "top"), default=0)
    bottom = max((ca.axis.position for ca in x_axes if ca.axis.side == "bottom"), default=h)
    plot_bounds = (left, top, right, bottom)

    # Rebuild and compare
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
    }


def main():
    parser = argparse.ArgumentParser(description="Extract data points from chart images.")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--debug", "-d", type=Path, default=None, help="Debug output directory")
    parser.add_argument(
        "--meta", type=Path, default=None, help="Optional meta JSON with ground truth",
    )
    args = parser.parse_args()

    meta = None
    if args.meta:
        with open(args.meta, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = load_meta_labels(args.input)

    extract_from_image(
        args.input, output_csv=args.output, debug_dir=args.debug, meta=meta,
    )


if __name__ == "__main__":
    main()
