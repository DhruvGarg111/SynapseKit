"""Tests for OrchestrationEvaluator and OrchestrationEvalReport."""

from __future__ import annotations

import inspect

import pytest
from tests.evaluation.orchestration.fixtures import (
    context_loss_run_graph,
    healthy_run_graph,
    looping_run_graph,
    misrouting_run_graphs,
)

from synapsekit.evaluation.orchestration.detectors import LLMContextLossJudge
from synapsekit.evaluation.orchestration.evaluator import OrchestrationEvaluator
from synapsekit.evaluation.orchestration.report import OrchestrationEvalReport
from synapsekit.llm.base import BaseLLM


class FakeLLM(BaseLLM):
    """Hand-written fake LLM without mocks."""

    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.response

    async def stream(self, prompt: str, **kwargs):
        yield self.response


def test_evaluator_healthy_graph_passes() -> None:
    evaluator = OrchestrationEvaluator()
    graph = healthy_run_graph()
    report = evaluator.evaluate(graph)

    assert isinstance(report, OrchestrationEvalReport)
    assert report.passed is True
    assert len(report.findings) == 0
    assert report.summary["loop"] == 0
    assert report.summary["context_loss"] == 0
    assert report.summary["misrouting"] == 0


def test_evaluator_mixed_batch() -> None:
    evaluator = OrchestrationEvaluator()
    graphs = [
        healthy_run_graph(),
        looping_run_graph(),
        context_loss_run_graph(),
    ]
    report = evaluator.evaluate(graphs)

    assert report.passed is False
    assert len(report.findings) >= 2
    assert report.summary["loop"] >= 1
    assert report.summary["context_loss"] >= 1


def test_evaluator_misrouting_batch() -> None:
    evaluator = OrchestrationEvaluator()
    graphs = misrouting_run_graphs()
    report = evaluator.evaluate(graphs)

    assert report.passed is False
    assert report.summary["misrouting"] >= 1


@pytest.mark.asyncio
async def test_evaluate_async_matches_evaluate_without_llm_judge() -> None:
    assert inspect.iscoroutinefunction(OrchestrationEvaluator.evaluate_async)

    evaluator = OrchestrationEvaluator()
    graphs = [healthy_run_graph(), looping_run_graph()]

    async_report = await evaluator.evaluate_async(graphs)
    sync_report = evaluator.evaluate(graphs)

    assert async_report.summary == sync_report.summary
    assert len(async_report.findings) == len(sync_report.findings)


@pytest.mark.asyncio
async def test_evaluate_async_wires_llm_judge_for_unflagged_transfers() -> None:
    # Regression: LLMContextLossJudge was defined but never invoked anywhere
    # in OrchestrationEvaluator, so passing one had no effect.
    fake_llm = FakeLLM("CONTEXT_LOSS: Order ID dropped during handoff")
    evaluator = OrchestrationEvaluator(llm_judge=LLMContextLossJudge(fake_llm))

    graph = healthy_run_graph()  # heuristic ContextLossDetector finds nothing here
    report = await evaluator.evaluate_async(graph)

    llm_findings = [f for f in report.findings if "LLM judge" in f.message]
    assert len(llm_findings) == len(graph.transfers)
    assert report.summary["context_loss"] == len(graph.transfers)


@pytest.mark.asyncio
async def test_evaluate_async_skips_llm_judge_for_already_flagged_transfers() -> None:
    # The LLM judge should not double-charge a transfer the heuristic detector
    # already flagged.
    fake_llm = FakeLLM("CONTEXT_LOSS: would have flagged this too")
    evaluator = OrchestrationEvaluator(llm_judge=LLMContextLossJudge(fake_llm))

    graph = context_loss_run_graph()  # heuristic detector already flags this
    report = await evaluator.evaluate_async(graph)

    llm_findings = [f for f in report.findings if "LLM judge" in f.message]
    assert len(llm_findings) == 0


def test_report_to_dict_and_to_markdown() -> None:
    evaluator = OrchestrationEvaluator()
    graphs = [looping_run_graph(), context_loss_run_graph()]
    report = evaluator.evaluate(graphs)

    d = report.to_dict()
    assert d["passed"] is False
    assert "summary" in d
    assert "findings" in d
    assert len(d["findings"]) == len(report.findings)

    md = report.to_markdown()
    assert "# Orchestration Evaluation Report" in md
    assert "Infinite Loops" in md
    assert "Context Loss" in md
