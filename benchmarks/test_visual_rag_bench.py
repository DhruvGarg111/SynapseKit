from __future__ import annotations

from visual_rag_bench import run_benchmark


def test_visual_doc_qa_benchmark_beats_text_only():
    result = run_benchmark()

    assert result.samples == 8
    assert result.visual_accuracy >= 0.75
    assert result.visual_accuracy > result.text_accuracy
    assert result.uplift >= 0.25


def test_visual_doc_qa_benchmark_is_reproducible():
    first = run_benchmark()
    second = run_benchmark()

    assert first == second
