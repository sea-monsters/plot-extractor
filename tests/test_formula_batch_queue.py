"""Tests for cross-image FormulaOCR queue batching."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_extractor.core.formula_batch_queue import (  # noqa: E402
    FormulaBatchQueue,
    FormulaQueueRequest,
)


class FakeFormulaOCR:
    def __init__(self):
        self.calls = []
        self._last_stats = None

    def read_label_batch(self, crops, batch_size=None):
        self.calls.append((len(crops), batch_size))
        self._last_stats = SimpleNamespace(
            requested=len(crops),
            returned=len(crops),
            batch_size=batch_size,
            chunks=1,
            elapsed_ms=12.5,
        )
        return [
            SimpleNamespace(latex=f"10^{{{idx}}}", values=[10.0 ** idx], log_confidence=0.75)
            for idx, _ in enumerate(crops)
        ]

    def last_batch_stats(self):
        return self._last_stats


def test_formula_batch_queue_merges_requests_into_one_model_call():
    queue = FormulaBatchQueue(batch_size=8)
    crop = np.full((12, 12, 3), 255, dtype=np.uint8)
    queue.add(FormulaQueueRequest("chart-a:y:0", crop, {"chart": "a"}))
    queue.add(FormulaQueueRequest("chart-b:y:0", crop, {"chart": "b"}))
    queue.add(FormulaQueueRequest("chart-c:x:0", crop, {"chart": "c"}))

    fake = FakeFormulaOCR()
    results = queue.run(fake)
    stats = queue.last_stats()

    assert fake.calls == [(3, 8)]
    assert stats.requested == 3
    assert stats.returned == 3
    assert stats.model_calls == 1
    assert results["chart-a:y:0"].value == 1.0
    assert results["chart-b:y:0"].value == 10.0
    assert results["chart-c:x:0"].value == 100.0


def test_formula_batch_queue_skips_empty_crops():
    queue = FormulaBatchQueue(batch_size=4)
    queue.add(FormulaQueueRequest("empty", np.empty((0, 0), dtype=np.uint8)))

    fake = FakeFormulaOCR()
    results = queue.run(fake)

    assert results == {}
    assert fake.calls == []
    assert queue.last_stats().requested == 0


if __name__ == "__main__":
    test_formula_batch_queue_merges_requests_into_one_model_call()
    test_formula_batch_queue_skips_empty_crops()
    print("[SUCCESS] Formula batch queue tests passed")
