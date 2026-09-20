"""Tests for orchestration failure detectors."""

from __future__ import annotations

import pytest
from tests.evaluation.orchestration.fixtures import (
    context_loss_run_graph,
    healthy_run_graph,
    looping_run_graph,
    misrouting_run_graphs,
)

from synapsekit.evaluation.orchestration.detectors import (
    ContextLossDetector,
    LLMContextLossJudge,
    LoopDetector,
    MisroutingDetector,
)
from synapsekit.evaluation.orchestration.run_graph import RunGraph, RunNode, Transfer
from synapsekit.llm.base import BaseLLM


class FakeLLM(BaseLLM):
    """Hand-written fake LLM without mocks."""

    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.response

    async def stream(self, prompt: str, **kwargs):
        yield self.response


def test_loop_detector_fires_on_looping_fixture() -> None:
    detector = LoopDetector(min_cycle_repeats=2, max_cycle_length=4)
    graph = looping_run_graph()
    findings = detector.detect(graph)

    assert len(findings) >= 1
    loop_findings = [f for f in findings if f.mode == "loop"]
    assert len(loop_findings) >= 1

    finding = loop_findings[0]
    assert finding.severity == "critical"
    assert len(finding.node_ids) > 0
    assert "cycle" in finding.evidence or "similarity" in finding.evidence


def test_loop_detector_handles_sparse_step_numbering() -> None:
    # Regression: cycle node_ids used to be looked up via nodes_by_step[step]
    # keyed by raw RunNode.step, but the cycle span is a *positional* index
    # into agent_sequence() (sorted by step). With non-contiguous step values
    # this silently dropped most nodes from the finding's evidence.
    agents = ["triage", "specialist", "triage", "specialist"]
    nodes = [
        RunNode(
            id=f"sparse_node_{i}",
            agent=agent,
            step=i * 5,  # non-contiguous: 0, 5, 10, 15
            input_text=f"input {i}",
            output_text=f"output {i}",
        )
        for i, agent in enumerate(agents)
    ]
    transfers = [
        Transfer(
            id=f"sparse_transfer_{i}",
            source=nodes[i].id,
            target=nodes[i + 1].id,
            from_agent=nodes[i].agent,
            to_agent=nodes[i + 1].agent,
        )
        for i in range(len(nodes) - 1)
    ]
    graph = RunGraph(run_id="sparse_run", goal="", nodes=nodes, transfers=transfers)

    detector = LoopDetector(min_cycle_repeats=2, max_cycle_length=2)
    findings = detector.detect(graph)

    cycle_findings = [f for f in findings if "cycle" in f.evidence]
    assert len(cycle_findings) >= 1
    assert set(cycle_findings[0].node_ids) == {n.id for n in nodes}


def test_loop_detector_silent_on_healthy_fixture() -> None:
    detector = LoopDetector()
    graph = healthy_run_graph()
    findings = detector.detect(graph)
    assert len(findings) == 0


def test_context_loss_detector_fires_on_loss_fixture() -> None:
    detector = ContextLossDetector(retention_threshold=0.7)
    graph = context_loss_run_graph()
    findings = detector.detect(graph)

    assert len(findings) >= 1
    ctx_findings = [f for f in findings if f.mode == "context_loss"]
    assert len(ctx_findings) >= 1

    finding = ctx_findings[0]
    assert finding.severity in ("warning", "critical")
    assert "dropped_facts" in finding.evidence
    dropped = finding.evidence["dropped_facts"]
    assert any("ORD-99823" in f or "John Smith" in f or "ACC-7712" in f for f in dropped)


def test_context_loss_detector_silent_on_healthy_fixture() -> None:
    detector = ContextLossDetector()
    graph = healthy_run_graph()
    findings = detector.detect(graph)
    assert len(findings) == 0


def test_misrouting_detector_fires_on_nondeterminism() -> None:
    detector = MisroutingDetector()
    graphs = misrouting_run_graphs()
    findings = detector.detect_nondeterminism(graphs, consistency_threshold=1.0)

    assert len(findings) >= 1
    finding = findings[0]
    assert finding.mode == "misrouting"
    assert "distribution" in finding.evidence
    assert finding.evidence["modal_share"] < 1.0


def test_misrouting_detector_golden_route() -> None:
    detector = MisroutingDetector()
    graphs = misrouting_run_graphs()
    golden = {"Resolve invoice payment issue": "billing"}

    findings = detector.detect_against_golden(graphs, golden)
    # Since 2 of the 5 runs routed to non-billing agents, we expect findings
    assert len(findings) == 2
    for f in findings:
        assert f.mode == "misrouting"
        assert f.severity == "critical"


def test_misrouting_detector_silent_on_healthy_fixture() -> None:
    detector = MisroutingDetector()
    graph = healthy_run_graph()
    nondet = detector.detect_nondeterminism([graph], consistency_threshold=1.0)
    golden = detector.detect_against_golden([graph], {"Issue order refund": "refund_specialist"})

    assert len(nondet) == 0
    assert len(golden) == 0


@pytest.mark.asyncio
async def test_llm_context_loss_judge_opt_in() -> None:
    fake_llm = FakeLLM("CONTEXT_LOSS: Order ID dropped during handoff")
    judge = LLMContextLossJudge(fake_llm)
    graph = context_loss_run_graph()

    finding = await judge.evaluate(graph.transfers[0], graph.nodes[0], graph.nodes[1])
    assert finding is not None
    assert finding.mode == "context_loss"
    assert "LLM judge" in finding.message
