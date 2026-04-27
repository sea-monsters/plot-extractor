"""Generate 500 single-chart pressure images for the v4a test set.

v4a keeps the v4 rendering style and metadata schema, but removes multi-plot
and combo charts entirely. The sampling plan is intentionally biased toward
route-weakness probes for the soft router:

- color_overlap: stress color separation and series clustering
- density_boundary: stress thinning vs. standard extraction
- scatter_clutter: stress scatter routing and component-based extraction
- ocr_axis: stress OCR, calibration, and log/dual-axis routing
- rotation_noise: stress rotation correction and preprocessing robustness

Each image gets route-oriented metadata tags so downstream evaluation can score
both chart-type prediction and strategy activation.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.generate_test_data_v4 import CHART_GENERATORS, DISTORTIONS, _fig_to_array, _save_meta

TEST_DATA_V4A_DIR = Path(__file__).parent.parent / "test_data_v4a"
N_IMAGES = 500
ROUTER_CHART_TYPES = [
    "simple_linear",
    "log_y",
    "log_x",
    "loglog",
    "scatter",
    "multi_series",
    "dense",
    "dual_y",
    "inverted_y",
    "no_grid",
]


def _seed_for(*parts: object) -> int:
    """Derive a stable RNG seed from arbitrary metadata."""
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def _pick_weighted(rng: np.random.Generator, weighted_choices: dict[str, int]) -> str:
    items = list(weighted_choices.items())
    labels = [item[0] for item in items]
    weights = np.array([item[1] for item in items], dtype=float)
    weights = weights / np.sum(weights)
    idx = int(rng.choice(len(labels), p=weights))
    return labels[idx]


def _pick_many(
    rng: np.random.Generator,
    choices: list[str],
    *,
    min_count: int,
    max_count: int,
) -> list[str]:
    """Choose a small distortion subset without replacement."""
    if not choices:
        return []
    count = int(rng.integers(min_count, max_count + 1))
    count = max(0, min(count, len(choices)))
    if count == 0:
        return []

    names = np.array(choices, dtype=object)
    weights = np.array([DISTORTION_WEIGHTS[name] for name in choices], dtype=float)
    weights = weights / np.sum(weights)
    selected = rng.choice(names, size=count, replace=False, p=weights)
    return [str(item) for item in selected.tolist()]


def _apply_profile_distortions(
    img: np.ndarray,
    rng: np.random.Generator,
    allowed: list[str],
    *,
    clean_probability: float,
    min_count: int,
    max_count: int,
) -> tuple[np.ndarray, list[str]]:
    """Apply a profile-constrained distortion subset."""
    if not allowed or rng.random() < clean_probability:
        return img, []

    applied: list[str] = []
    for name in _pick_many(rng, allowed, min_count=min_count, max_count=max_count):
        func = DISTORTION_FUNCS.get(name)
        if func is None:
            continue
        img = func(img, rng)
        applied.append(name)
    return img, applied


def _profile_stats() -> list[dict]:
    """Sampling plan tuned to expose route shortfalls."""
    return [
        {
            "name": "color_overlap",
            "count": 120,
            "chart_weights": {"multi_series": 7, "dense": 3, "simple_linear": 1},
            "distortions": ["jpeg_compression", "grayscale", "gaussian_blur", "moire_pattern", "uneven_lighting"],
            "clean_probability": 0.20,
            "min_count": 0,
            "max_count": 3,
            "targets": ["color_hsv3d", "color_layered", "extract_cc"],
            "confusers": ["extract_scatter", "noise_unsharp"],
        },
        {
            "name": "density_boundary",
            "count": 110,
            "chart_weights": {"dense": 7, "scatter": 2, "simple_linear": 1},
            "distortions": ["resolution_degradation", "gaussian_blur", "line_break", "motion_blur", "sharpening_artifact"],
            "clean_probability": 0.15,
            "min_count": 1,
            "max_count": 3,
            "targets": ["extract_thinning", "extract_cc", "noise_unsharp"],
            "confusers": ["extract_scatter", "color_hsv3d"],
        },
        {
            "name": "scatter_clutter",
            "count": 100,
            "chart_weights": {"scatter": 7, "dense": 2, "multi_series": 1},
            "distortions": ["gaussian_noise", "salt_pepper_noise", "jpeg_compression", "motion_blur", "grayscale"],
            "clean_probability": 0.10,
            "min_count": 1,
            "max_count": 4,
            "targets": ["extract_scatter", "noise_median", "noise_bilateral"],
            "confusers": ["extract_thinning", "color_layered"],
        },
        {
            "name": "ocr_axis",
            "count": 100,
            "chart_weights": {
                "log_y": 3,
                "log_x": 3,
                "loglog": 3,
                "dual_y": 2,
                "inverted_y": 2,
                "no_grid": 1,
                "simple_linear": 1,
            },
            "distortions": ["rotation", "perspective_warp", "uneven_lighting", "edge_shadow", "jpeg_compression", "resolution_degradation"],
            "clean_probability": 0.12,
            "min_count": 1,
            "max_count": 3,
            "targets": ["ocr_adaptive", "ocr_deskew"],
            "confusers": ["rotation_correct", "noise_bilateral"],
        },
        {
            "name": "rotation_noise",
            "count": 70,
            "chart_weights": {"simple_linear": 3, "no_grid": 3, "dual_y": 2, "inverted_y": 2, "log_y": 1},
            "distortions": ["rotation", "perspective_warp", "line_break", "curl_fold", "edge_shadow", "ink_smudge", "moire_pattern", "grayscale"],
            "clean_probability": 0.08,
            "min_count": 1,
            "max_count": 4,
            "targets": ["rotation_correct", "noise_median", "noise_unsharp"],
            "confusers": ["ocr_deskew", "extract_cc"],
        },
    ]


PROFILE_SPECS = _profile_stats()
DISTORTION_FUNCS = {name: func for name, func, _ in DISTORTIONS}
DISTORTION_WEIGHTS = {name: weight for name, _, weight in DISTORTIONS}
SUPPORTED_CHART_TYPES = ROUTER_CHART_TYPES


def _make_schedule() -> list[dict]:
    queue: list[dict] = []
    for profile in PROFILE_SPECS:
        queue.extend([profile] * profile["count"])
    schedule_rng = np.random.default_rng(_seed_for("v4a-schedule"))
    schedule_rng.shuffle(queue)
    return queue


def _choose_chart_type(rng: np.random.Generator, profile: dict) -> str:
    return _pick_weighted(rng, profile["chart_weights"])


def _generate_single_chart(
    rng: np.random.Generator,
    chart_type: str,
) -> tuple[np.ndarray, dict, dict]:
    gen_fn = CHART_GENERATORS.get(chart_type)
    if gen_fn is None:
        gen_fn = CHART_GENERATORS["simple_linear"]
        chart_type = "simple_linear"

    fig, _, data_dict, axes_dict = gen_fn(rng)
    img = _fig_to_array(fig)
    return img, data_dict, axes_dict


def _build_tags(
    profile: dict,
    chart_type: str,
    img: np.ndarray,
    distortions: list[str],
) -> dict:
    return {
        "dataset": "v4a",
        "chart_types": [chart_type],
        "distortions": distortions,
        "chart_count": 1,
        "resolution": list(img.shape[:2]),
        "route_profile": profile["name"],
        "route_targets": profile["targets"],
        "route_confusers": profile["confusers"],
        "route_focus": profile["name"],
    }


def _write_manifest(counts: Counter, profile_counts: Counter) -> None:
    manifest = {
        "dataset": "v4a",
        "total": sum(profile_counts.values()),
        "profiles": {name: profile_counts[name] for name in sorted(profile_counts)},
        "chart_types": {name: counts[name] for name in sorted(counts)},
        "supported_chart_types": SUPPORTED_CHART_TYPES,
        "profile_spec": PROFILE_SPECS,
    }
    with open(TEST_DATA_V4A_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    TEST_DATA_V4A_DIR.mkdir(parents=True, exist_ok=True)

    schedule = _make_schedule()
    generated = 0
    chart_counts: Counter = Counter()
    profile_counts: Counter = Counter()
    route_focus_counts: Counter = Counter()
    by_profile_chart_counts: dict[str, Counter] = defaultdict(Counter)

    for idx, profile in enumerate(schedule):
        seed = _seed_for("v4a", idx, profile["name"])
        rng = np.random.default_rng(seed)
        chart_type = _choose_chart_type(rng, profile)
        img, data_dict, axes_dict = _generate_single_chart(rng, chart_type)
        img, distortions = _apply_profile_distortions(
            img,
            rng,
            profile["distortions"],
            clean_probability=profile["clean_probability"],
            min_count=profile["min_count"],
            max_count=profile["max_count"],
        )

        name = f"{idx + 1:04d}"
        out_dir = TEST_DATA_V4A_DIR / chart_type
        out_dir.mkdir(parents=True, exist_ok=True)
        tags = _build_tags(profile, chart_type, img, distortions)

        Image.fromarray(img).save(out_dir / f"{name}.png")
        _save_meta(out_dir, name, data_dict, axes_dict, tags)

        generated += 1
        chart_counts[chart_type] += 1
        profile_counts[profile["name"]] += 1
        route_focus_counts[profile["name"]] += 1
        by_profile_chart_counts[profile["name"]][chart_type] += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{N_IMAGES}] generated")

    _write_manifest(chart_counts, profile_counts)

    print(f"Done. {generated} images in {TEST_DATA_V4A_DIR}")
    print("Profile counts:")
    for profile_name in sorted(profile_counts):
        print(f"  {profile_name}: {profile_counts[profile_name]}")
    print("Chart counts:")
    for chart_type in sorted(chart_counts):
        print(f"  {chart_type}: {chart_counts[chart_type]}")


if __name__ == "__main__":
    main()
