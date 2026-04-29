"""PP-FormulaNet integration for superscript/math-notation OCR on axis tick labels.

Provides a singleton ``FormulaOCR`` service that lazy-loads the ~248 MB
PP-FormulaNet_plus-S model once and reuses it across all axis crops in a
thread-safe manner.  This avoids memory blow-up from concurrent model
instances and keeps inference latency predictable.

Usage::

    from plot_extractor.core.formula_ocr import get_formula_ocr
    focr = get_formula_ocr()
    latex, value, confidence = focr.read_label(crop)
    notation_score = focr.detect_log_notation(crop)
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton machinery
# ---------------------------------------------------------------------------

_INSTANCE: Optional["FormulaOCR"] = None
_INSTANCE_LOCK = threading.Lock()


def get_formula_ocr() -> Optional["FormulaOCR"]:
    """Return the process-wide singleton ``FormulaOCR``, or None if unavailable."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        try:
            _INSTANCE = FormulaOCR()
        except Exception:
            _logger.debug("FormulaOCR: not available", exc_info=True)
            _INSTANCE = None
        return _INSTANCE


def unload_formula_ocr():
    """Release the singleton model to free memory."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            _INSTANCE._unload()
            _INSTANCE = None


def formula_ocr_available() -> bool:
    """Check whether FormulaOCR can be created without actually loading the model."""
    try:
        import paddlex  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# LaTeX value parsing
# ---------------------------------------------------------------------------

# Pre-compiled patterns for common log-scale LaTeX outputs
_RE_10_POW = re.compile(r"10\s*\^\s*\{(-?\d+)\}")
_RE_10_POW_SIMPLE = re.compile(r"10\s*\^\s*(-?\d+)")
_RE_SPLIT_10_POW = re.compile(r"1\D{0,24}0\s*\^\s*\{?(-?\d+)\}?")
_RE_TIMES_10_POW = re.compile(r"[×xX]\\?times\s*10\s*\^\s*\{(-?\d+)\}")
_RE_SCI = re.compile(r"(\d+\.?\d*)\s*[eE]\s*([+-]?\d+)")
_RE_PLAIN_NUM = re.compile(r"^\s*(\d+\.?\d*)\s*$")


def parse_latex_value(latex: str) -> Optional[float]:
    """Extract a numeric value from a PP-FormulaNet LaTeX string.

    Handles: ``10^{2}``, ``10^2``, ``1.5\\times10^{3}``, ``1e2``,
    and plain numbers.

    Returns None when no parseable value is found.
    """
    if not latex or not latex.strip():
        return None

    latex = latex.strip()

    # 10^{N} — most common log-scale notation
    m = _RE_10_POW.search(latex)
    if m:
        return float(10 ** int(m.group(1)))

    # 10^N (without braces)
    m = _RE_10_POW_SIMPLE.search(latex)
    if m:
        return float(10 ** int(m.group(1)))

    # FormulaOCR sometimes emits table/aligned fragments where ``10^{N}``
    # is split into ``1 ... 0^{N}``; recover the value without trusting the
    # surrounding layout markup.
    m = _RE_SPLIT_10_POW.search(latex)
    if m:
        return float(10 ** int(m.group(1)))

    # ×10^{N}
    m = _RE_TIMES_10_POW.search(latex)
    if m:
        base_match = re.search(r"(\d+\.?\d*)\s*[×xX]", latex)
        base = float(base_match.group(1)) if base_match else 1.0
        return base * (10 ** int(m.group(1)))

    # Scientific notation: 1e2, 1.5E+3
    m = _RE_SCI.search(latex)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))

    # Plain number fallback
    m = _RE_PLAIN_NUM.match(latex)
    if m:
        return float(m.group(1))

    return None


def score_latex_log_notation(latex: str) -> float:
    """Return 0.0-1.0 indicating how strongly *latex* suggests a log scale.

    High-confidence patterns (>0.7):
      - ``10^{N}``, ``10^N``, ``×10^{N}``

    Moderate patterns (0.3-0.5):
      - Scientific notation, plain powers of 10
    """
    if not latex or not latex.strip():
        return 0.0

    score = 0.0

    if _RE_10_POW.search(latex) or _RE_10_POW_SIMPLE.search(latex) or _RE_SPLIT_10_POW.search(latex):
        score += 0.60
    if _RE_TIMES_10_POW.search(latex):
        score += 0.20
    if _RE_SCI.search(latex):
        score += 0.15

    value = parse_latex_value(latex)
    if value is not None and value > 0:
        log10 = np.log10(value)
        if abs(log10 - round(log10)) < 0.02:
            score += 0.15

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# FormulaOCR singleton class
# ---------------------------------------------------------------------------

@dataclass
class AxisLabelResult:
    """Result from reading all tick labels on one axis in a single pass."""

    latex: str
    exponents: list[int]         # extracted 10^{N} exponents
    values: list[float]          # computed 10^{N} values
    log_confidence: float        # 0-1 aggregate log-scale confidence
    count_10pow: int             # how many 10^{N} patterns were found


@dataclass
class FormulaBatchStats:
    """Lightweight telemetry for the last FormulaOCR batch call."""

    requested: int
    returned: int
    batch_size: int
    chunks: int
    elapsed_ms: float


class FormulaOCR:
    """Singleton wrapper around PP-FormulaNet_plus-S for axis label reading.

    Loads the ~248 MB model once and serialises all inference calls
    through an internal lock.  One crop per axis (not per tick) —
    the LaTeX output is scanned for all ``10^{N}`` patterns.
    """

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._last_batch_stats: Optional[FormulaBatchStats] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_axis_labels(self, axis_crop: np.ndarray) -> Optional[AxisLabelResult]:
        """Send the full axis tick-label strip to PP-FormulaNet in one call.

        *axis_crop* should be a region around the axis containing all
        tick labels (e.g. from ``_crop_axis_region``).

        Returns ``AxisLabelResult`` with all extracted ``10^{N}`` values,
        or None on failure.
        """
        self._ensure_model()
        if self._model is None:
            return None

        with self._lock:
            try:
                latex = self._infer(axis_crop)
            except Exception:
                _logger.debug("FormulaOCR inference failed", exc_info=True)
                return None

        if latex is None:
            return None

        exponents, values = _extract_all_10pow(latex)
        log_conf = _score_axis_latex(latex, len(exponents))
        return AxisLabelResult(
            latex=latex,
            exponents=exponents,
            values=values,
            log_confidence=log_conf,
            count_10pow=len(exponents),
        )

    def read_axes_batch(
        self, axis_crops: list, batch_size: int | None = None,
    ) -> list:
        """Send all axis crops to PP-FormulaNet in a single batch call.

        Returns a list of ``AxisLabelResult`` (or None per crop on failure),
        same length and order as *axis_crops*.
        """
        return self.read_label_batch(axis_crops, batch_size=batch_size)

    def read_label_batch(self, label_crops: list, batch_size: int | None = None) -> list:
        """Send cropped label images to PP-FormulaNet in one batch call."""
        if not label_crops:
            self._last_batch_stats = FormulaBatchStats(
                requested=0,
                returned=0,
                batch_size=0,
                chunks=0,
                elapsed_ms=0.0,
            )
            return []
        self._ensure_model()
        if self._model is None:
            self._last_batch_stats = FormulaBatchStats(
                requested=len(label_crops),
                returned=0,
                batch_size=0,
                chunks=0,
                elapsed_ms=0.0,
            )
            return [None] * len(label_crops)

        started = time.perf_counter()
        target_batch_size = len(label_crops) if not batch_size or batch_size <= 0 else min(batch_size, len(label_crops))
        results = []
        chunks = 0
        with self._lock:
            try:
                for offset in range(0, len(label_crops), target_batch_size):
                    chunk = label_crops[offset:offset + target_batch_size]
                    chunks += 1
                    latex_list = self._infer_batch(chunk)
                    for latex in latex_list:
                        if latex is None:
                            results.append(None)
                        else:
                            exponents, values = _extract_all_10pow(latex)
                            log_conf = _score_axis_latex(latex, len(exponents))
                            results.append(AxisLabelResult(
                                latex=latex,
                                exponents=exponents,
                                values=values,
                                log_confidence=log_conf,
                                count_10pow=len(exponents),
                            ))
            except Exception:
                _logger.debug("FormulaOCR batch inference failed", exc_info=True)
                self._last_batch_stats = FormulaBatchStats(
                    requested=len(label_crops),
                    returned=0,
                    batch_size=target_batch_size,
                    chunks=chunks,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
                return [None] * len(label_crops)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._last_batch_stats = FormulaBatchStats(
            requested=len(label_crops),
            returned=sum(1 for item in results if item is not None),
            batch_size=target_batch_size,
            chunks=chunks,
            elapsed_ms=elapsed_ms,
        )
        return results

    def last_batch_stats(self) -> Optional[FormulaBatchStats]:
        """Return telemetry for the most recent batch call."""
        return self._last_batch_stats

    def detect_log_notation(self, axis_crop: np.ndarray) -> float:
        """Return 0.0-1.0 log-scale confidence for an axis crop.

        One PP-FormulaNet call.  When available, this score takes
        **priority** over tesseract-based notation detection.
        """
        result = self.read_axis_labels(axis_crop)
        if result is None:
            return 0.0
        return result.log_confidence

    def available(self) -> bool:
        """True if the model was loaded successfully."""
        self._ensure_model()
        return self._model is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from paddlex import (  # pylint: disable=import-outside-toplevel
                    create_model,
                )
                self._model = create_model("PP-FormulaNet_plus-S", device="cpu")
            except Exception:
                _logger.debug("FormulaOCR: cannot load model", exc_info=True)
                self._model = None

    def _infer(self, crop: np.ndarray) -> Optional[str]:
        if self._model is None:
            return None
        output = list(self._model.predict(crop, batch_size=1))
        if not output:
            return None
        return output[0].json.get("res", {}).get("rec_formula")

    def _infer_batch(self, crops: list) -> list:
        """Run PP-FormulaNet on a list of crops in one batch call."""
        if self._model is None or not crops:
            return [None] * len(crops)
        output = list(self._model.predict(crops, batch_size=len(crops)))
        results = []
        for item in output:
            try:
                results.append(item.json.get("res", {}).get("rec_formula"))
            except Exception:
                results.append(None)
        # Pad if model returned fewer results than inputs
        while len(results) < len(crops):
            results.append(None)
        return results

    def _unload(self):
        if self._model is not None:
            del self._model
            self._model = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_all_10pow(latex: str) -> Tuple[list, list]:
    """Find every ``10^{N}`` (or ``10^N``) in *latex* and return
    (exponents, computed_values).
    """
    exponents = []
    for m in _RE_10_POW.finditer(latex):
        exponents.append(int(m.group(1)))
    if not exponents:
        for m in _RE_10_POW_SIMPLE.finditer(latex):
            exponents.append(int(m.group(1)))
    values = [float(10 ** e) for e in exponents]
    return exponents, values


def _score_axis_latex(latex: str, count_10pow: int) -> float:
    """Aggregate log-confidence from axis-level LaTeX output.

    Scoring logic (priority-ordered):
      - 3+ ``10^{N}`` patterns → near-certain log scale (0.85+)
      - 2 ``10^{N}`` patterns → strong evidence (0.65)
      - 1 ``10^{N}`` + other superscript cues → moderate (0.50)
      - 1 ``10^{N}`` alone → weak (0.35)
      - No pattern → 0.0
    """
    if count_10pow >= 4:
        return 0.95
    if count_10pow >= 3:
        return 0.85
    if count_10pow >= 2:
        score = 0.65
    elif count_10pow >= 1:
        score = 0.35
        # Boost if other log cues present
        if _RE_TIMES_10_POW.search(latex):
            score += 0.15
        if _RE_SCI.search(latex):
            score += 0.10
    else:
        return 0.0

    # Bonus: check for geometric progression in extracted exponents
    if count_10pow >= 2:
        all_exponents = []
        for m in _RE_10_POW.finditer(latex):
            all_exponents.append(int(m.group(1)))
        if len(all_exponents) >= 3:
            diffs = [all_exponents[i + 1] - all_exponents[i]
                     for i in range(len(all_exponents) - 1)]
            if diffs and max(diffs) - min(diffs) <= 1:
                score += 0.10  # exponents are evenly spaced

    return min(score, 1.0)
