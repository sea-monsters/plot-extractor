"""Confidence calculation for extraction results.

Provides scoring functions to quantify extraction reliability,
enabling fallback decisions and warning generation.
"""
from dataclasses import dataclass
from typing import List

from plot_extractor.core.axis_candidates import AxisMappingCandidate
from plot_extractor.core.axis_calibrator import CalibratedAxis


@dataclass
class ExtractionConfidence:
    """Overall extraction confidence metrics."""
    axis_confidence: float  # 0-100
    series_confidence: float  # 0-100
    ocr_confidence: float  # 0-100
    overall_confidence: float  # weighted average
    warnings: List[str]


def compute_axis_confidence(candidate: AxisMappingCandidate) -> float:
    """Compute confidence score for axis mapping.

    Factors:
    - Source reliability (ocr > heuristic)
    - Tick count
    - Sequence regularity
    - Residual magnitude
    """
    # Base score from source
    source_scores = {
        "ocr": candidate.confidence,  # Use precomputed score
        "heuristic": 20.0,
        "default": 0.0,
    }
    base = source_scores.get(candidate.source, 50.0)

    # Residual penalty
    if candidate.residual < 100:
        base += 10
    elif candidate.residual > 1000:
        base -= 20

    # Clamp to 0-100
    return max(0.0, min(100.0, base))


def compute_series_confidence(
    series_points: int,
    coverage_ratio: float,
    continuity_score: float,
) -> float:
    """Compute confidence score for series extraction.

    Factors:
    - Point count
    - Coverage of plot area
    - Continuity (few gaps)
    """
    score = 0.0

    # Point count bonus
    if series_points >= 300:
        score += 30
    elif series_points >= 100:
        score += 20
    elif series_points >= 50:
        score += 10

    # Coverage bonus
    score += coverage_ratio * 40

    # Continuity bonus
    score += continuity_score * 30

    return max(0.0, min(100.0, score))


def compute_ocr_confidence(ocr_tokens: List[tuple]) -> float:
    """Compute confidence for OCR quality.

    Args:
        ocr_tokens: list of (text, confidence) tuples from tesseract

    Returns:
        Average confidence (0-100)
    """
    if not ocr_tokens:
        return 0.0

    # Filter valid tokens
    valid_confidences = [conf for _, conf in ocr_tokens if conf is not None and conf > 0]

    if not valid_confidences:
        return 0.0

    # Average confidence
    avg = sum(valid_confidences) / len(valid_confidences)

    # Tesseract returns 0-100, scale to 0-100
    return max(0.0, min(100.0, avg))


def compute_extraction_confidence(
    calibrated_axes: List[CalibratedAxis],
    series_data: dict,
    ocr_confidence: float = 50.0,
) -> ExtractionConfidence:
    """Compute overall extraction confidence.

    Weights:
    - axis: 40%
    - series: 40%
    - ocr: 20%
    """
    warnings = []

    # Axis confidence: average across axes
    if calibrated_axes:
        axis_scores = []
        for ca in calibrated_axes:
            # Convert CalibratedAxis to candidate-like for scoring
            cand_score = _score_calibrated_axis(ca)
            axis_scores.append(cand_score)

            # Generate warnings for low confidence
            if cand_score < 50:
                warnings.append(f"Low axis confidence ({ca.axis.direction}): {cand_score:.1f}")

        axis_confidence = sum(axis_scores) / len(axis_scores)
    else:
        axis_confidence = 0.0
        warnings.append("No axes calibrated")

    # Series confidence
    if series_data:
        series_scores = []
        for name, points in series_data.items():
            n_points = len(points.get("x", []))
            coverage = 1.0  # TODO: compute actual coverage
            continuity = 1.0  # TODO: compute actual continuity

            score = compute_series_confidence(n_points, coverage, continuity)
            series_scores.append(score)

            if score < 30:
                warnings.append(f"Low series confidence ({name}): {score:.1f}")

        series_confidence = sum(series_scores) / len(series_scores)
    else:
        series_confidence = 0.0
        warnings.append("No series extracted")

    # Overall confidence (weighted average)
    overall = (
        axis_confidence * 0.4 +
        series_confidence * 0.4 +
        ocr_confidence * 0.2
    )

    return ExtractionConfidence(
        axis_confidence=axis_confidence,
        series_confidence=series_confidence,
        ocr_confidence=ocr_confidence,
        overall_confidence=overall,
        warnings=warnings,
    )


def _score_calibrated_axis(ca: CalibratedAxis) -> float:
    """Score a CalibratedAxis (convert to candidate-like scoring)."""
    score = 0.0

    # Tick count
    tick_count = len(ca.tick_map)
    score += tick_count * 10

    # Residual
    if ca.residual < 100:
        score += 50
    elif ca.residual < 1000:
        score += 30
    else:
        score -= 30

    # Type consistency (linear/log)
    # TODO: check if values match axis_type pattern

    return max(0.0, min(100.0, score))
