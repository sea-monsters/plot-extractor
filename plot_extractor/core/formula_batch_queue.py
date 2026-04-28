"""Cross-image FormulaOCR request queue.

This module is intentionally about throughput only: it batches already-selected
label crops across charts and returns OCR results keyed by caller-provided ids.
Crop selection and tesseract-vs-FormulaOCR ownership stay in the calibration
pipeline where accuracy decisions belong.
"""

from dataclasses import dataclass
import time
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class FormulaQueueRequest:
    """One FormulaOCR crop request from any chart/axis/tick."""

    request_id: str
    crop: np.ndarray
    metadata: dict[str, Any] | None = None


@dataclass
class FormulaQueueResult:
    """FormulaOCR result keyed back to the queued request."""

    request_id: str
    latex: Optional[str]
    value: Optional[float]
    log_confidence: float
    metadata: dict[str, Any] | None = None


@dataclass
class FormulaQueueStats:
    """Throughput telemetry for a queue run."""

    requested: int
    returned: int
    batch_size: int
    chunks: int
    elapsed_ms: float
    model_calls: int


class FormulaBatchQueue:
    """Collect FormulaOCR crops across images and execute them as one batch."""

    def __init__(self, batch_size: int = 16):
        self.batch_size = max(1, int(batch_size))
        self._requests: list[FormulaQueueRequest] = []
        self._last_stats = FormulaQueueStats(
            requested=0,
            returned=0,
            batch_size=self.batch_size,
            chunks=0,
            elapsed_ms=0.0,
            model_calls=0,
        )

    def add(self, request: FormulaQueueRequest) -> None:
        """Append one valid crop request to the queue."""
        if request.crop is None or request.crop.size == 0:
            return
        self._requests.append(request)

    def extend(self, requests: list[FormulaQueueRequest]) -> None:
        """Append many crop requests."""
        for request in requests:
            self.add(request)

    def __len__(self) -> int:
        return len(self._requests)

    def clear(self) -> None:
        self._requests.clear()

    def last_stats(self) -> FormulaQueueStats:
        return self._last_stats

    def run(self, formula_ocr=None) -> dict[str, FormulaQueueResult]:
        """Run queued crops through FormulaOCR and return results by id."""
        if not self._requests:
            self._last_stats = FormulaQueueStats(
                requested=0,
                returned=0,
                batch_size=self.batch_size,
                chunks=0,
                elapsed_ms=0.0,
                model_calls=0,
            )
            return {}

        if formula_ocr is None:
            from plot_extractor.core.formula_ocr import get_formula_ocr  # pylint: disable=import-outside-toplevel

            formula_ocr = get_formula_ocr()

        if formula_ocr is None:
            self._last_stats = FormulaQueueStats(
                requested=len(self._requests),
                returned=0,
                batch_size=self.batch_size,
                chunks=0,
                elapsed_ms=0.0,
                model_calls=0,
            )
            return {
                req.request_id: FormulaQueueResult(
                    request_id=req.request_id,
                    latex=None,
                    value=None,
                    log_confidence=0.0,
                    metadata=req.metadata,
                )
                for req in self._requests
            }

        from plot_extractor.core.formula_ocr import parse_latex_value  # pylint: disable=import-outside-toplevel

        started = time.perf_counter()
        crops = [req.crop for req in self._requests]
        raw_results = formula_ocr.read_label_batch(crops, batch_size=self.batch_size)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        batch_stats = formula_ocr.last_batch_stats()

        results: dict[str, FormulaQueueResult] = {}
        returned = 0
        for req, raw in zip(self._requests, raw_results):
            latex = getattr(raw, "latex", None) if raw is not None else None
            value = parse_latex_value(latex) if latex else None
            if value is None and raw is not None and getattr(raw, "values", None):
                value = raw.values[0]
            log_confidence = float(getattr(raw, "log_confidence", 0.0)) if raw is not None else 0.0
            if raw is not None:
                returned += 1
            results[req.request_id] = FormulaQueueResult(
                request_id=req.request_id,
                latex=latex,
                value=value,
                log_confidence=log_confidence,
                metadata=req.metadata,
            )

        self._last_stats = FormulaQueueStats(
            requested=len(self._requests),
            returned=returned,
            batch_size=self.batch_size,
            chunks=int(getattr(batch_stats, "chunks", 0) or 0),
            elapsed_ms=float(getattr(batch_stats, "elapsed_ms", elapsed_ms) or elapsed_ms),
            model_calls=1,
        )
        return results
