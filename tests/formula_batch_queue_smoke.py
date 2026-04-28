"""Smoke-test cross-image FormulaOCR batching.

This script measures FormulaOCR throughput only. It plans label crops across
many images, queues them, and runs one merged OCR batch. It does not score data
extraction accuracy.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.core.axis_calibrator import _plan_formula_batch_requests  # noqa: E402
from plot_extractor.core.axis_detector import detect_all_axes  # noqa: E402
from plot_extractor.core.formula_batch_queue import (  # noqa: E402
    FormulaBatchQueue,
    FormulaQueueRequest,
)
from plot_extractor.core.image_loader import load_image, preprocess, to_grayscale  # noqa: E402
from plot_extractor.core.ocr_reader import detect_tick_label_anchors  # noqa: E402


def _plan_image_requests(img_path: Path, max_crops: int | None) -> list[FormulaQueueRequest]:
    raw = load_image(img_path)
    image = preprocess(raw, denoise=True)
    gray = to_grayscale(image)
    axes = detect_all_axes(gray)
    axis_anchor_map = {}
    axis_is_log = {}

    for axis in axes:
        tick_pixels = [tick[0] for tick in axis.ticks or []]
        if not tick_pixels:
            continue
        axis_anchor_map[id(axis)] = detect_tick_label_anchors(image, axis, tick_pixels)
        # Throughput smoke keeps routing permissive. Accuracy routing is tested
        # in per-image extraction, not in this queue benchmark.
        axis_is_log[id(axis)] = True

    plan = _plan_formula_batch_requests(
        axes,
        axis_anchor_map,
        axis_is_log,
        max_total_crops=max_crops,
    )
    requests = []
    for idx, req in enumerate(plan.requests):
        requests.append(
            FormulaQueueRequest(
                request_id=f"{img_path.stem}:{idx}",
                crop=req.crop,
                metadata={
                    "file": img_path.name,
                    "axis_id": req.axis_id,
                    "anchor_idx": req.anchor_idx,
                    "score": req.score,
                    "planned_count": plan.requested_count,
                    "kept_count": plan.kept_count,
                },
            )
        )
    return requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("test_data"))
    parser.add_argument("--type", default="loglog", dest="chart_type")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-crops-per-image", type=int, default=2)
    args = parser.parse_args()

    image_dir = args.data_dir / args.chart_type
    image_paths = sorted(image_dir.glob("*.png"))[: args.limit]
    queue = FormulaBatchQueue(batch_size=args.batch_size)
    per_image_counts = {}

    for img_path in image_paths:
        requests = _plan_image_requests(img_path, args.max_crops_per_image)
        per_image_counts[img_path.name] = len(requests)
        queue.extend(requests)

    results = queue.run()
    stats = queue.last_stats()

    print(f"type={args.chart_type} images={len(image_paths)} queued={len(queue)}")
    print(
        "formula_queue "
        f"requested={stats.requested} returned={stats.returned} "
        f"batch_size={stats.batch_size} chunks={stats.chunks} "
        f"model_calls={stats.model_calls} elapsed_ms={stats.elapsed_ms:.2f}"
    )
    print("per_image=" + ", ".join(f"{name}:{count}" for name, count in per_image_counts.items()))
    sample = list(results.values())[:5]
    if sample:
        print("sample=" + ", ".join(f"{item.request_id}:{item.latex}" for item in sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
