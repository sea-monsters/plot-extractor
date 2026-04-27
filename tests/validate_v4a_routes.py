"""Route evaluation for the v4a pressure dataset.

This script measures how well the soft router aligns with the route-oriented
v4a sampling plan. It does not run full extraction; instead it evaluates the
lightweight chart-type guesser and policy router so we can inspect:

- top-1 chart-type prediction accuracy
- top-1 / top-3 route target hit rates
- mean activation gaps between intended targets and confusers
- per-profile and per-strategy weaknesses
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.core.chart_type_guesser import extract_all_features, guess_chart_type
from plot_extractor.core.policy_router import POLICY_WEIGHTS, compute_policy, _strategy_activation
from plot_extractor.core.image_loader import load_image, to_grayscale

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data_v4a"
REPORT_PATH = Path(__file__).parent.parent / "report_test_data_v4a_routes.csv"
PROFILE_REPORT_PATH = Path(__file__).parent.parent / "report_test_data_v4a_routes_by_profile.csv"
STRATEGY_REPORT_PATH = Path(__file__).parent.parent / "report_test_data_v4a_routes_by_strategy.csv"


def _load_meta(img_path: Path) -> dict | None:
    meta_path = img_path.parent / f"{img_path.stem}_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _top_items(scores: dict[str, float], n: int = 3) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:n]


def _rank_of(name: str, ordered: list[tuple[str, float]]) -> int | None:
    for idx, (item_name, _) in enumerate(ordered, start=1):
        if item_name == name:
            return idx
    return None


def _format_top_items(items: list[tuple[str, float]]) -> str:
    return "|".join(f"{name}:{score:.4f}" for name, score in items)


def _safe_mean(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return round(total / count, 4)


def _strategy_scores(features, type_probs: dict[str, float]) -> dict[str, float]:
    return {
        strategy_name: float(_strategy_activation(weights, type_probs))
        for strategy_name, weights in POLICY_WEIGHTS.items()
    }


def _policy_state(policy) -> dict[str, str | bool | int]:
    return {
        "noise_strategy": policy.noise_strategy,
        "color_strategy": policy.color_strategy,
        "density_strategy": policy.density_strategy,
        "rotation_correct": bool(policy.rotation_correct),
        "rotation_angle": round(float(policy.rotation_angle), 4),
        "ocr_block_size": int(policy.ocr_block_size),
        "ocr_C": int(policy.ocr_C),
        "ocr_deskew": bool(policy.ocr_deskew),
        "hough_threshold": int(policy.hough_threshold),
    }


def evaluate_image(img_path: Path) -> dict:
    meta = _load_meta(img_path)
    if not meta:
        raise ValueError(f"missing meta for {img_path}")

    tags = meta.get("tags") or {}
    profile = tags.get("route_profile") or "unknown"
    chart_type = (tags.get("chart_types") or [None])[0]
    targets = tags.get("route_targets") or []
    confusers = tags.get("route_confusers") or []

    image = load_image(img_path)
    gray = to_grayscale(image)
    features = extract_all_features(image, gray)
    type_probs = guess_chart_type(features)
    policy = compute_policy(features, type_probs)

    type_top = _top_items(type_probs, 3)
    strategy_scores = _strategy_scores(features, type_probs)
    strategy_order = _top_items(strategy_scores, len(strategy_scores))
    strategy_top = strategy_order[:5]

    best_target = max(((name, strategy_scores.get(name, 0.0)) for name in targets), key=lambda item: item[1], default=(None, 0.0))
    best_confuser = max(((name, strategy_scores.get(name, 0.0)) for name in confusers), key=lambda item: item[1], default=(None, 0.0))
    top1_strategy = strategy_top[0][0] if strategy_top else ""
    top3_strategy_names = [name for name, _ in strategy_order[:3]]

    return {
        "file": img_path.name,
        "profile": profile,
        "chart_type": chart_type or "",
        "top_type": type_top[0][0] if type_top else "",
        "top_type_prob": round(type_top[0][1], 4) if type_top else 0.0,
        "type_top3": _format_top_items(type_top),
        "chart_type_hit": bool(chart_type and type_top and type_top[0][0] == chart_type),
        "top_strategy": top1_strategy,
        "top_strategy_score": round(strategy_top[0][1], 4) if strategy_top else 0.0,
        "strategy_top3": _format_top_items(strategy_top),
        "top1_target_hit": bool(top1_strategy in targets),
        "top3_target_hit": bool(set(top3_strategy_names) & set(targets)),
        "best_target": best_target[0] or "",
        "best_target_score": round(float(best_target[1]), 4),
        "best_confuser": best_confuser[0] or "",
        "best_confuser_score": round(float(best_confuser[1]), 4),
        "target_confuser_gap": round(float(best_target[1]) - float(best_confuser[1]), 4),
        "target_rank": _rank_of(best_target[0], strategy_order),
        "confuser_rank": _rank_of(best_confuser[0], strategy_order),
        "route_targets": "|".join(targets),
        "route_confusers": "|".join(confusers),
        **{f"{name}_activation": round(score, 4) for name, score in strategy_scores.items()},
        **_policy_state(policy),
    }


def _summarize_by_profile(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["profile"]].append(row)

    summaries = []
    for profile, items in sorted(buckets.items()):
        total = len(items)
        summaries.append({
            "profile": profile,
            "total": total,
            "chart_type_hit_rate": round(sum(1 for r in items if r["chart_type_hit"]) / max(total, 1), 3),
            "top1_target_hit_rate": round(sum(1 for r in items if r["top1_target_hit"]) / max(total, 1), 3),
            "top3_target_hit_rate": round(sum(1 for r in items if r["top3_target_hit"]) / max(total, 1), 3),
            "avg_target_score": round(float(np.mean([r["best_target_score"] for r in items])), 4),
            "avg_confuser_score": round(float(np.mean([r["best_confuser_score"] for r in items])), 4),
            "avg_gap": round(float(np.mean([r["target_confuser_gap"] for r in items])), 4),
            "best_target_rate": round(sum(1 for r in items if r["best_target_score"] >= r["best_confuser_score"]) / max(total, 1), 3),
        })
    return summaries


def _summarize_by_strategy(rows: list[dict]) -> list[dict]:
    summary: dict[str, dict] = {}
    strategy_names = list(POLICY_WEIGHTS)

    for name in strategy_names:
        summary[name] = {
            "strategy": name,
            "total": 0,
            "target_count": 0,
            "confuser_count": 0,
            "top1_count": 0,
            "target_activation_sum": 0.0,
            "non_target_activation_sum": 0.0,
            "confuser_activation_sum": 0.0,
        }

    for row in rows:
        top_strategy = row["top_strategy"]
        targets = set(row["route_targets"].split("|")) if row["route_targets"] else set()
        confusers = set(row["route_confusers"].split("|")) if row["route_confusers"] else set()
        for name in strategy_names:
            summary[name]["total"] += 1
            activation = float(row.get(f"{name}_activation", 0.0))
            if name in targets:
                summary[name]["target_count"] += 1
                summary[name]["target_activation_sum"] += activation
            else:
                summary[name]["non_target_activation_sum"] += activation
            if name in confusers:
                summary[name]["confuser_count"] += 1
                summary[name]["confuser_activation_sum"] += activation
            if top_strategy == name:
                summary[name]["top1_count"] += 1

    rows_out = []
    for name in strategy_names:
        item = summary[name]
        target_mean = _safe_mean(item["target_activation_sum"], item["target_count"])
        non_target_mean = _safe_mean(item["non_target_activation_sum"], item["total"] - item["target_count"])
        confuser_mean = _safe_mean(item["confuser_activation_sum"], item["confuser_count"])
        rows_out.append({
            "strategy": name,
            "total": item["total"],
            "target_count": item["target_count"],
            "confuser_count": item["confuser_count"],
            "top1_count": item["top1_count"],
            "top1_rate": round(item["top1_count"] / max(item["total"], 1), 3),
            "mean_activation_when_target": target_mean,
            "mean_activation_when_not_target": non_target_mean,
            "mean_activation_when_confuser": confuser_mean,
            "lift_target_minus_non_target": (
                round(target_mean - non_target_mean, 4)
                if target_mean is not None and non_target_mean is not None
                else None
            ),
        })
    return rows_out


def run(data_dir: Path = TEST_DATA_DIR) -> tuple[list[dict], list[dict], list[dict]]:
    image_files = sorted(data_dir.glob("*/*.png"))
    rows: list[dict] = []

    print(f"--- v4a route evaluation: {data_dir} ---")
    for img_path in image_files:
        row = evaluate_image(img_path)
        rows.append(row)
        print(
            f"  {row['file']}: type={row['chart_type']} top={row['top_type']} "
            f"target_hit={row['top1_target_hit']} gap={row['target_confuser_gap']:.4f}"
        )

    profile_rows = _summarize_by_profile(rows)
    strategy_rows = _summarize_by_strategy(rows)

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file", "profile", "chart_type", "top_type", "top_type_prob", "type_top3",
            "chart_type_hit", "top_strategy", "top_strategy_score", "strategy_top3",
            "top1_target_hit", "top3_target_hit", "best_target", "best_target_score",
            "best_confuser", "best_confuser_score", "target_confuser_gap",
            "target_rank", "confuser_rank", "route_targets", "route_confusers",
            "noise_median_activation", "noise_bilateral_activation", "noise_unsharp_activation",
            "ocr_adaptive_activation", "ocr_deskew_activation", "color_hsv3d_activation",
            "color_layered_activation", "extract_thinning_activation",
            "extract_scatter_activation", "extract_cc_activation",
            "rotation_correct_activation",
            "noise_strategy", "color_strategy", "density_strategy", "rotation_correct",
            "rotation_angle", "ocr_block_size", "ocr_C", "ocr_deskew", "hough_threshold",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(PROFILE_REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "profile", "total", "chart_type_hit_rate", "top1_target_hit_rate",
            "top3_target_hit_rate", "avg_target_score", "avg_confuser_score",
            "avg_gap", "best_target_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(profile_rows)

    with open(STRATEGY_REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "strategy", "total", "target_count", "confuser_count", "top1_count",
            "top1_rate", "mean_activation_when_target",
            "mean_activation_when_not_target", "mean_activation_when_confuser",
            "lift_target_minus_non_target",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(strategy_rows)

    total = len(rows)
    chart_hit = sum(1 for row in rows if row["chart_type_hit"])
    top1_target_hit = sum(1 for row in rows if row["top1_target_hit"])
    top3_target_hit = sum(1 for row in rows if row["top3_target_hit"])

    print("\n" + "=" * 88)
    print(f"{'Metric':<28} {'Value':>8}")
    print("-" * 88)
    print(f"{'total images':<28} {total:>8}")
    print(f"{'chart-type top1 hit rate':<28} {chart_hit / max(total, 1):>8.1%}")
    print(f"{'route top1 target hit rate':<28} {top1_target_hit / max(total, 1):>8.1%}")
    print(f"{'route top3 target hit rate':<28} {top3_target_hit / max(total, 1):>8.1%}")
    print(f"Reports: {REPORT_PATH.name}, {PROFILE_REPORT_PATH.name}, {STRATEGY_REPORT_PATH.name}")

    return rows, profile_rows, strategy_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="v4a data directory (default: test_data_v4a)")
    args = parser.parse_args()
    run(Path(args.data_dir) if args.data_dir else TEST_DATA_DIR)
