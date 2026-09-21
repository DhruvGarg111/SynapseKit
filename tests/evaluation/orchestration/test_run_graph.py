"""Tests for RunGraph abstraction module."""

from __future__ import annotations

import pytest

from synapsekit.agents.multi.handoff import Handoff, HandoffChain
from synapsekit.evaluation.orchestration.run_graph import RunGraph, RunNode, Transfer


class FakeAgentExecutor:
    """Hand-written fake executor without mocks."""

    def __init__(self, output: str) -> None:
        self.output = output

    async def run(self, input_text: str) -> str:
        return f"{self.output}: {input_text}"


@pytest.mark.asyncio
async def test_run_graph_from_handoff_result() -> None:
    triage_exec = FakeAgentExecutor("Triage output - need billing")
    billing_exec = FakeAgentExecutor("Billing output - resolved")

    chain = HandoffChain()
    chain.add_agent(
        "triage",
        triage_exec,
        handoffs=[Handoff("billing", condition=lambda r: "billing" in r.lower())],
    )
    chain.add_agent("billing", billing_exec)

    result = await chain.run("triage", "I have an invoice issue")
    graph = RunGraph.from_handoff_result(result, goal="Fix invoice", run_id="test_run_123")

    assert graph.run_id == "test_run_123"
    assert graph.goal == "Fix invoice"
    assert len(graph.nodes) == 2
    assert len(graph.transfers) == 1

    assert graph.nodes[0].agent == "triage"
    assert graph.nodes[1].agent == "billing"

    assert graph.transfers[0].source == graph.nodes[0].id
    assert graph.transfers[0].target == graph.nodes[1].id
    assert graph.transfers[0].from_agent == "triage"
    assert graph.transfers[0].to_agent == "billing"

    assert graph.agent_sequence() == ["triage", "billing"]

    # Regression: HandoffChain.run now stamps each history entry with a
    # timestamp, so RunNode.timestamp should no longer be silently None
    # for graphs built from real HandoffResult objects.
    assert graph.nodes[0].timestamp is not None
    assert graph.nodes[1].timestamp is not None


def test_run_graph_from_events() -> None:
    events = [
        {
            "kind": "run.start",
            "name": "run.start",
            "attributes": {"label": "Process claim ORD-100"},
        },
        {
            "kind": "agent.step",
            "agent": "intake",
            "input": "User claim query",
            "output": "Claim intake verified",
            "ts": 100.0,
        },
        {
            "kind": "handoff",
            "from_agent": "intake",
            "to_agent": "claims_processor",
            "reason": "Escalate to claims",
            "ts": 101.0,
        },
        {
            "kind": "agent.step",
            "agent": "claims_processor",
            "input": "Claim intake verified",
            "output": "Claim processed successfully",
            "ts": 102.0,
        },
    ]

    graph = RunGraph.from_events(events)

    assert graph.goal == "Process claim ORD-100"
    assert len(graph.nodes) == 2
    assert graph.agent_sequence() == ["intake", "claims_processor"]
    assert len(graph.transfers) >= 1
    assert graph.transfers[0].from_agent == "intake"
    assert graph.transfers[0].to_agent == "claims_processor"

    # Regression: source/target must resolve to the actual intake/claims_processor
    # nodes, not collapse into a self-loop on whichever node existed when the
    # handoff event was processed.
    intake_node = next(n for n in graph.nodes if n.agent == "intake")
    claims_node = next(n for n in graph.nodes if n.agent == "claims_processor")
    assert graph.transfers[0].source == intake_node.id
    assert graph.transfers[0].target == claims_node.id
    assert graph.transfers[0].source != graph.transfers[0].target


def test_to_live_graph_formatting() -> None:
    node0 = RunNode(id="n0", agent="triage", step=0, input_text="a", output_text="b")
    node1 = RunNode(id="n1", agent="billing", step=1, input_text="b", output_text="c")
    transfer = Transfer(id="t0", source="n0", target="n1", from_agent="triage", to_agent="billing")
    graph = RunGraph(run_id="r1", goal="Test", nodes=[node0, node1], transfers=[transfer])

    live_nodes, live_edges = graph.to_live_graph()
    assert len(live_nodes) == 2
    assert len(live_edges) == 1
    assert live_nodes[0]["id"] == "n0"
    assert live_nodes[0]["group"] == "triage"
    assert live_edges[0]["source"] == "n0"
    assert live_edges[0]["target"] == "n1"
