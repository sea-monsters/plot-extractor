"""Automatic validation loop over all sample charts."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.main import extract_from_image
from plot_extractor.config import (
    SSIM_THRESHOLD_SIMPLE,
    SSIM_THRESHOLD_LOG,
    SSIM_THRESHOLD_SCATTER,
    SSIM_THRESHOLD_MULTI,
)

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
DEBUG_DIR = Path(__file__).parent.parent / "debug"
REPORT_PATH = Path(__file__).parent.parent / "report.csv"


def get_threshold(name):
    if "scatter" in name:
        return SSIM_THRESHOLD_SCATTER
    if "log" in name or "dual" in name:
        return SSIM_THRESHOLD_LOG
    if "multi" in name:
        return SSIM_THRESHOLD_MULTI
    return SSIM_THRESHOLD_SIMPLE


def run_validation():
    DEBUG_DIR.mkdir(exist_ok=True)
    image_files = sorted(SAMPLES_DIR.glob("*.png"))

    rows = []
    for img_path in image_files:
        meta_path = img_path.parent / f"{img_path.stem}_meta.json"
        meta = None
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

        result = extract_from_image(img_path, output_csv=DEBUG_DIR / f"{img_path.stem}.csv",
                                    debug_dir=DEBUG_DIR, meta=meta)

        threshold = get_threshold(img_path.stem)
        if result:
            # Use best of full and cropped SSIM
            ssim_full = result.get("ssim") or 0.0
            ssim_crop = result.get("ssim_crop") or 0.0
            ssim = max(ssim_full, ssim_crop)
            passed = ssim >= threshold
            n_points = sum(len(s["x"]) for s in result["data"].values())
            axis_info = ", ".join(f"{ca.axis.direction}({ca.axis_type})" for ca in result["calibrated_axes"])
        else:
            ssim = 0.0
            passed = False
            n_points = 0
            axis_info = "none"

        rows.append({
            "file": img_path.name,
            "ssim": round(ssim, 4),
            "threshold": threshold,
            "passed": passed,
            "points": n_points,
            "axes": axis_info,
        })
        print(f"  Result: SSIM={ssim:.4f}, threshold={threshold}, passed={passed}\n")

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "ssim", "threshold", "passed", "points", "axes"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    print(f"\nValidation complete: {passed}/{total} passed")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_validation()
