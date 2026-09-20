"""Orchestration evaluation subpackage — detect handoff loops, context loss, and mis-routing.

Usage::

    from synapsekit.evaluation.orchestration import (
        RunGraph,
        OrchestrationEvaluator,
    )
    from synapsekit.agents.multi import HandoffChain

    # Convert a run into a RunGraph
    graph = RunGraph.from_handoff_result(result, goal="Process customer billing inquiry")

    # Run orchestration evaluation
    evaluator = OrchestrationEvaluator()
    report = evaluator.evaluate(graph)
    if not report.passed:
        print(report.to_markdown())
"""

from .detectors import (
    ContextLossDetector,
    DetectorFinding,
    LLMContextLossJudge,
    LoopDetector,
    MisroutingDetector,
)
from .evaluator import OrchestrationEvaluator
from .report import OrchestrationEvalReport
from .run_graph import RunGraph, RunNode, Transfer

__all__ = [
    "ContextLossDetector",
    "DetectorFinding",
    "LLMContextLossJudge",
    "LoopDetector",
    "MisroutingDetector",
    "OrchestrationEvalReport",
    "OrchestrationEvaluator",
    "RunNode",
    "RunGraph",
    "Transfer",
]
