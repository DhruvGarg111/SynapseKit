"""OrchestrationEvaluator — run-graph evaluation and live dashboard publishing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .detectors import (
    ContextLossDetector,
    DetectorFinding,
    LLMContextLossJudge,
    LoopDetector,
    MisroutingDetector,
)
from .report import OrchestrationEvalReport
from .run_graph import RunGraph


class OrchestrationEvaluator:
    """Evaluates RunGraphs for orchestration failure modes (loops, context loss, misrouting).

    Usage::

        evaluator = OrchestrationEvaluator()
        report = evaluator.evaluate(run_graph)
        if not report.passed:
            print(report.to_markdown())

    Pass ``llm_judge`` to additionally run :class:`LLMContextLossJudge` over
    transfers the heuristic :class:`ContextLossDetector` didn't already flag,
    via the async :meth:`evaluate_async` entry point::

        evaluator = OrchestrationEvaluator(llm_judge=LLMContextLossJudge(my_llm))
        report = await evaluator.evaluate_async(run_graph)
    """

    def __init__(
        self,
        loop_detector: LoopDetector | None = None,
        context_loss_detector: ContextLossDetector | None = None,
        misrouting_detector: MisroutingDetector | None = None,
        thresholds: dict[str, Any] | None = None,
        llm_judge: LLMContextLossJudge | None = None,
    ) -> None:
        self.loop_detector = loop_detector or LoopDetector()
        self.context_loss_detector = context_loss_detector or ContextLossDetector()
        self.misrouting_detector = misrouting_detector or MisroutingDetector()
        self.thresholds = thresholds or {}
        self.llm_judge = llm_judge

    def _detect(self, graph_list: list[RunGraph]) -> list[DetectorFinding]:
        findings: list[DetectorFinding] = []

        # Per-graph detectors
        for g in graph_list:
            findings.extend(self.loop_detector.detect(g))
            findings.extend(self.context_loss_detector.detect(g))

        # Batch misrouting detector across graphs
        if len(graph_list) > 1:
            findings.extend(self.misrouting_detector.detect_nondeterminism(graph_list))

        return findings

    def _build_report(
        self, graph_list: list[RunGraph], findings: list[DetectorFinding]
    ) -> OrchestrationEvalReport:
        summary = {
            "loop": sum(1 for f in findings if f.mode == "loop"),
            "context_loss": sum(1 for f in findings if f.mode == "context_loss"),
            "misrouting": sum(1 for f in findings if f.mode == "misrouting"),
        }
        return OrchestrationEvalReport(
            run_ids=[g.run_id for g in graph_list],
            findings=findings,
            thresholds=self.thresholds,
            summary=summary,
        )

    def evaluate(self, graphs: RunGraph | Sequence[RunGraph]) -> OrchestrationEvalReport:
        """Run orchestration detectors over one or more RunGraph instances."""
        graph_list = [graphs] if isinstance(graphs, RunGraph) else list(graphs)
        findings = self._detect(graph_list)
        return self._build_report(graph_list, findings)

    async def evaluate_async(
        self, graphs: RunGraph | Sequence[RunGraph]
    ) -> OrchestrationEvalReport:
        """Run detectors, then the opt-in LLM context-loss judge if configured.

        Identical to :meth:`evaluate` when no ``llm_judge`` was passed to the
        constructor. When one is configured, it only judges transfers the
        heuristic :class:`ContextLossDetector` didn't already flag, to avoid
        double-charging both a heuristic and an LLM call for the same miss.
        """
        graph_list = [graphs] if isinstance(graphs, RunGraph) else list(graphs)
        findings = self._detect(graph_list)

        if self.llm_judge is not None:
            already_flagged = {
                tid for f in findings if f.mode == "context_loss" for tid in f.transfer_ids
            }
            for g in graph_list:
                nodes_by_id = {n.id: n for n in g.nodes}
                for transfer in g.transfers:
                    if transfer.id in already_flagged:
                        continue
                    src_node = nodes_by_id.get(transfer.source)
                    tgt_node = nodes_by_id.get(transfer.target)
                    if src_node is None or tgt_node is None:
                        continue
                    finding = await self.llm_judge.evaluate(transfer, src_node, tgt_node)
                    if finding is not None:
                        findings.append(finding)

        return self._build_report(graph_list, findings)

    def evaluate_and_publish(
        self,
        graphs: RunGraph | Sequence[RunGraph],
        *,
        label: str = "",
    ) -> OrchestrationEvalReport:
        """Evaluate graphs and publish findings + graph snapshot to SynapseKit Live."""
        report = self.evaluate(graphs)
        graph_list = [graphs] if isinstance(graphs, RunGraph) else list(graphs)

        try:
            from ...live import _maybe_autostart, publish, publish_graph

            # Mirrors observe/runtime.py and dream/core.py: cheap no-op after the
            # first call, but without it SYNAPSEKIT_LIVE=1 alone never flips
            # bus.enabled on and every publish() below silently no-ops.
            _maybe_autostart()

            top_findings = [asdict(f) for f in report.findings[:5]]
            publish(
                "orchestration.eval",
                status="blocked" if not report.passed else "ok",
                attributes={
                    "label": label,
                    "passed": report.passed,
                    "summary": report.summary,
                    "findings_count": len(report.findings),
                    "run_ids": report.run_ids,
                    "top_findings": top_findings,
                },
            )

            if graph_list:
                # Node/edge ids are run_id-prefixed by construction, so combining
                # every evaluated graph into one snapshot is collision-free and
                # avoids silently dropping visualization for all but the first run.
                all_nodes: list[dict[str, Any]] = []
                all_edges: list[dict[str, Any]] = []
                for g in graph_list:
                    g_nodes, g_edges = g.to_live_graph(findings=report.findings)
                    all_nodes.extend(g_nodes)
                    all_edges.extend(g_edges)
                publish_graph(all_nodes, all_edges)
        except ImportError:  # pragma: no cover - live dashboard extra not installed
            pass

        return report
