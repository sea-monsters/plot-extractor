"""Sweep batched extraction from small to larger batch sizes.

This is a throughput-first validation helper. It runs the real batched
extraction pipeline with increasing image counts so we can watch stability and
queue behavior as the batch grows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.main import extract_from_images_batched  # noqa: E402


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("test_data"))
    parser.add_argument("--type", default="loglog", dest="chart_type")
    parser.add_argument("--limits", default="1,2,4", help="Comma-separated image counts to sweep")
    parser.add_argument("--batch-sizes", default="2,4,8", help="Comma-separated FormulaOCR batch sizes to sweep")
    parser.add_argument("--max-crops-per-image", type=int, default=2)
    parser.add_argument("--use-ocr", action="store_true", default=True)
    args = parser.parse_args()

    image_dir = args.data_dir / args.chart_type
    image_paths = sorted(image_dir.glob("*.png"))
    limits = _parse_int_list(args.limits)
    batch_sizes = _parse_int_list(args.batch_sizes)

    print(f"type={args.chart_type} total_images={len(image_paths)}")
    for batch_size in batch_sizes:
        for limit in limits:
            sample = image_paths[:limit]
            results = extract_from_images_batched(
                sample,
                use_llm=False,
                use_ocr=args.use_ocr,
                formula_batch_max_crops=args.max_crops_per_image,
                formula_batch_size=batch_size,
            )
            if results:
                first = results[0]
                batch_requested = first["formula_batch_requested"]
                batch_returned = first["formula_batch_returned"]
                batch_chunks = first["formula_batch_chunks"]
                batch_calls = first["formula_batch_model_calls"]
                batch_ms = first["formula_batch_ms"]
            else:
                batch_requested = batch_returned = batch_chunks = batch_calls = 0
                batch_ms = 0.0

            print(
                f"limit={limit} batch_size={batch_size} "
                f"results={len(results)} "
                f"requested={batch_requested} returned={batch_returned} "
                f"chunks={batch_chunks} model_calls={batch_calls} elapsed_ms={batch_ms:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
