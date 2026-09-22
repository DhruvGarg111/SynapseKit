"""Tests for OrchestrationEvaluator live dashboard integration."""

from __future__ import annotations

from tests.evaluation.orchestration.fixtures import (
    context_loss_run_graph,
    healthy_run_graph,
)

import synapsekit.live as live
from synapsekit.evaluation.orchestration.evaluator import OrchestrationEvaluator
from synapsekit.live.bus import bus


def test_evaluate_and_publish_when_live_enabled() -> None:
    # Test using the process-wide bus singleton (real object)
    bus.clear()
    bus.enabled = True
    q = bus.subscribe()

    try:
        evaluator = OrchestrationEvaluator()
        graph = healthy_run_graph()
        report = evaluator.evaluate_and_publish(graph, label="Test Run")

        assert report.passed is True

        published_events = []
        while not q.empty():
            published_events.append(q.get_nowait())

        kinds = [e["kind"] for e in published_events]
        assert "orchestration.eval" in kinds
        assert "graph.snapshot" in kinds

        eval_event = next(e for e in published_events if e["kind"] == "orchestration.eval")
        assert eval_event["status"] == "ok"
        assert eval_event["attributes"]["label"] == "Test Run"
        assert eval_event["attributes"]["passed"] is True
    finally:
        bus.unsubscribe(q)
        bus.enabled = False


def test_evaluate_and_publish_when_live_disabled() -> None:
    bus.clear()
    bus.enabled = False

    evaluator = OrchestrationEvaluator()
    graph = context_loss_run_graph()
    report = evaluator.evaluate_and_publish(graph, label="Disabled Run")

    assert report.passed is False
    assert len(bus.history()) == 0  # Zero overhead when off contract


def test_evaluate_and_publish_snapshots_every_graph_in_batch() -> None:
    # Regression: publish used to snapshot only graph_list[0], silently dropping
    # visualization for every other graph in a multi-run misrouting batch.
    bus.clear()
    bus.enabled = True
    q = bus.subscribe()

    try:
        evaluator = OrchestrationEvaluator()
        graph_a = healthy_run_graph()
        graph_b = context_loss_run_graph()
        evaluator.evaluate_and_publish([graph_a, graph_b], label="Batch Run")

        snapshot = next(e for e in bus.history() if e["kind"] == "graph.snapshot")
        published_ids = {n["id"] for n in snapshot["attributes"]["nodes"]}
        assert {n.id for n in graph_a.nodes} <= published_ids
        assert {n.id for n in graph_b.nodes} <= published_ids
    finally:
        bus.unsubscribe(q)
        bus.enabled = False


def test_evaluate_and_publish_calls_maybe_autostart() -> None:
    # Regression: evaluate_and_publish never called live._maybe_autostart(),
    # so SYNAPSEKIT_LIVE=1 alone (with no prior observe/dream call in-process)
    # left bus.enabled False and every publish() below silently no-op'd.
    # Swap in a real (non-mock) counting function to avoid actually binding
    # the dashboard's fixed port from this unit test.
    calls = []

    def fake_autostart() -> None:
        calls.append(1)
        bus.enabled = True

    bus.clear()
    bus.enabled = False
    prev_autostart = live._maybe_autostart
    live._maybe_autostart = fake_autostart

    try:
        evaluator = OrchestrationEvaluator()
        evaluator.evaluate_and_publish(healthy_run_graph(), label="Autostart Run")
        assert calls == [1]
        assert bus.enabled is True
    finally:
        live._maybe_autostart = prev_autostart
        bus.enabled = False
        bus.clear()
