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
    if abs(rot_angle) >= 0.5:
        print(f"[{image_path.name}] Rotating by {-rot_angle:.2f}° (detected {rot_angle:.2f}°)")
        image = rotate_image(image, -rot_angle)
        gray = to_grayscale(image)

    # Detect axes
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
    if abs(rot_angle) >= 0.5:
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
