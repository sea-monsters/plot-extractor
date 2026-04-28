"""Smoke-test the batched extraction pipeline.

This script exercises the cross-image FormulaOCR queue through the real
extraction entrypoint. It focuses on wiring and throughput, not accuracy.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.main import extract_from_images_batched  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("test_data"))
    parser.add_argument("--type", default="loglog", dest="chart_type")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-crops-per-image", type=int, default=2)
    args = parser.parse_args()

    image_dir = args.data_dir / args.chart_type
    image_paths = sorted(image_dir.glob("*.png"))[: args.limit]
    results = extract_from_images_batched(
        image_paths,
        use_llm=False,
        use_ocr=True,
        formula_batch_max_crops=args.max_crops_per_image,
        formula_batch_size=args.batch_size,
    )

    print(f"type={args.chart_type} images={len(image_paths)} results={len(results)}")
    if results:
        first = results[0]
        print(
            "batch="
            f"requested={first['formula_batch_requested']} "
            f"returned={first['formula_batch_returned']} "
            f"chunks={first['formula_batch_chunks']} "
            f"model_calls={first['formula_batch_model_calls']} "
            f"elapsed_ms={first['formula_batch_ms']:.2f}"
        )
        print(
            "first="
            f"csv={first['csv']} "
            f"panel_count={first['panel_count']} "
            f"axis_count={len(first['calibrated_axes'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
