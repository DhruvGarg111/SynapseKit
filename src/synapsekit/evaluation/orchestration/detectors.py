"""Detectors for orchestration failure modes: loops, context loss, and mis-routing."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ...llm.base import BaseLLM
from .run_graph import RunGraph, RunNode, Transfer


@dataclass(slots=True)
class DetectorFinding:
    """Finding produced by an orchestration failure mode detector."""

    mode: Literal["loop", "context_loss", "misrouting"]
    severity: Literal["info", "warning", "critical"]
    message: str
    node_ids: list[str] = field(default_factory=list)
    transfer_ids: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class LoopDetector:
    """Detects periodic cycles and stalled loops in agent handoff execution."""

    def __init__(
        self,
        min_cycle_repeats: int = 2,
        max_cycle_length: int = 4,
        similarity_threshold: float = 0.92,
    ) -> None:
        self.min_cycle_repeats = min_cycle_repeats
        self.max_cycle_length = max_cycle_length
        self.similarity_threshold = similarity_threshold

    def detect(self, graph: RunGraph) -> list[DetectorFinding]:
        findings: list[DetectorFinding] = []
        seq = graph.agent_sequence()
        # `seq`/cycle spans are positional indices (agent_sequence() sorts by
        # .step), not raw .step values — index into the same sorted list so a
        # sparse/non-contiguous .step numbering doesn't drop nodes from a
        # detected cycle's evidence.
        sorted_nodes = sorted(graph.nodes, key=lambda n: n.step)
        transfers_by_step = {i: t for i, t in enumerate(graph.transfers)}

        # 1. Exact periodic cycle detection
        n = len(seq)
        detected_cycle_spans: set[tuple[int, int]] = set()

        for cycle_len in range(1, min(self.max_cycle_length + 1, n // self.min_cycle_repeats + 1)):
            for i in range(n - cycle_len * self.min_cycle_repeats + 1):
                cycle_pattern = seq[i : i + cycle_len]
                repeats = 1
                while (
                    i + (repeats + 1) * cycle_len <= n
                    and seq[i + repeats * cycle_len : i + (repeats + 1) * cycle_len]
                    == cycle_pattern
                ):
                    repeats += 1

                if repeats >= self.min_cycle_repeats:
                    span = (i, i + repeats * cycle_len)
                    # Check if already covered by an earlier span
                    if any(s <= span[0] and e >= span[1] for s, e in detected_cycle_spans):
                        continue
                    detected_cycle_spans.add(span)

                    cycle_node_ids = [
                        sorted_nodes[pos].id
                        for pos in range(span[0], min(span[1], len(sorted_nodes)))
                    ]
                    cycle_transfer_ids = [
                        transfers_by_step[step].id
                        for step in range(span[0], min(span[1] - 1, len(transfers_by_step)))
                        if step in transfers_by_step
                    ]

                    cycle_str = " -> ".join(cycle_pattern)
                    findings.append(
                        DetectorFinding(
                            mode="loop",
                            severity="critical",
                            message=(
                                f"Detected periodic loop of cycle length {cycle_len} repeating "
                                f"{repeats} times ({cycle_str})"
                            ),
                            node_ids=cycle_node_ids,
                            transfer_ids=cycle_transfer_ids,
                            evidence={
                                "cycle": cycle_pattern,
                                "cycle_length": cycle_len,
                                "repeats": repeats,
                                "start_step": span[0],
                                "end_step": span[1],
                            },
                        )
                    )

        # 2. Stalled loop detection (near-duplicate input for same agent across non-adjacent steps)
        for i in range(len(sorted_nodes)):
            for j in range(i + 2, len(sorted_nodes)):
                node_i = sorted_nodes[i]
                node_j = sorted_nodes[j]
                if node_i.agent == node_j.agent:
                    ratio = difflib.SequenceMatcher(
                        None, node_i.input_text, node_j.input_text
                    ).ratio()
                    if ratio >= self.similarity_threshold:
                        # Find intermediate transfers
                        stalled_transfer_ids = [
                            t.id
                            for t in graph.transfers
                            if any(n.id in (t.source, t.target) for n in (node_i, node_j))
                        ]
                        findings.append(
                            DetectorFinding(
                                mode="loop",
                                severity="critical" if ratio >= 0.98 else "warning",
                                message=(
                                    f"Stalled loop detected: agent '{node_i.agent}' received near-duplicate "
                                    f"input across non-adjacent steps {node_i.step} and {node_j.step} "
                                    f"(similarity {ratio:.2f})"
                                ),
                                node_ids=[node_i.id, node_j.id],
                                transfer_ids=stalled_transfer_ids,
                                evidence={
                                    "agent": node_i.agent,
                                    "step_i": node_i.step,
                                    "step_j": node_j.step,
                                    "similarity": ratio,
                                },
                            )
                        )

        return findings


def _default_fact_extractor(text: str) -> set[str]:
    """Default fact extractor pulling capitalized spans, alnum tokens with digits, and quotes."""
    facts: set[str] = set()

    # Capitalized multi-word spans (e.g. "Order ID", "John Smith")
    cap_spans = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    facts.update(cap_spans)

    # Standalone numbers/IDs (tokens containing at least one digit, e.g., "ORD-12345", "12345")
    id_tokens = re.findall(r"\b[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b", text)
    facts.update(id_tokens)

    # Quoted substrings
    quoted = re.findall(r"""["']([^"']+)["']""", text)
    facts.update(quoted)

    return {f.strip() for f in facts if len(f.strip()) > 1}


class ContextLossDetector:
    """Detects fact loss across agent transfers."""

    def __init__(
        self,
        retention_threshold: float = 0.7,
        fact_extractor: Callable[[str], set[str]] | None = None,
    ) -> None:
        self.retention_threshold = retention_threshold
        self.fact_extractor = fact_extractor or _default_fact_extractor

    def detect(self, graph: RunGraph) -> list[DetectorFinding]:
        findings: list[DetectorFinding] = []
        nodes_by_id = {n.id: n for n in graph.nodes}

        for transfer in graph.transfers:
            src_node = nodes_by_id.get(transfer.source)
            tgt_node = nodes_by_id.get(transfer.target)

            if src_node is None or tgt_node is None:
                continue

            # Source facts from output_text + context values
            src_context_str = " ".join(str(v) for v in src_node.context.values())
            src_text = f"{src_node.output_text}\n{src_context_str}"
            source_facts = self.fact_extractor(src_text)

            if not source_facts:
                continue

            # Target facts from input_text + context_sent values + tgt.context values
            sent_context_str = " ".join(str(v) for v in transfer.context_sent.values())
            tgt_context_str = " ".join(str(v) for v in tgt_node.context.values())
            tgt_text = f"{tgt_node.input_text}\n{sent_context_str}\n{tgt_context_str}"
            target_facts = self.fact_extractor(tgt_text)

            # Substring matching for fuzzy containment
            retained_facts: set[str] = set()
            dropped_facts: set[str] = set()

            tgt_text_lower = tgt_text.lower()
            for fact in source_facts:
                if fact in target_facts or fact.lower() in tgt_text_lower:
                    retained_facts.add(fact)
                else:
                    dropped_facts.add(fact)

            fraction = len(retained_facts) / len(source_facts)
            if fraction < self.retention_threshold:
                findings.append(
                    DetectorFinding(
                        mode="context_loss",
                        severity="critical" if fraction < 0.4 else "warning",
                        message=(
                            f"Context loss in transfer {transfer.from_agent} -> {transfer.to_agent}: "
                            f"retained {len(retained_facts)}/{len(source_facts)} facts ({fraction:.0%})"
                        ),
                        node_ids=[src_node.id, tgt_node.id],
                        transfer_ids=[transfer.id],
                        evidence={
                            "retention_fraction": fraction,
                            "source_facts": sorted(source_facts),
                            "retained_facts": sorted(retained_facts),
                            "dropped_facts": sorted(dropped_facts),
                        },
                    )
                )

        return findings


class LLMContextLossJudge:
    """Opt-in, async semantic judge for transfer context loss using an LLM."""

    def __init__(self, llm: BaseLLM, retention_threshold: float = 0.7) -> None:
        self._llm = llm
        self.retention_threshold = retention_threshold

    async def evaluate(
        self,
        transfer: Transfer,
        source_node: RunNode,
        target_node: RunNode,
    ) -> DetectorFinding | None:
        prompt = (
            f"Analyze this multi-agent transfer for context loss.\n\n"
            f"Source Agent: {source_node.agent}\n"
            f"Source Output: {source_node.output_text}\n"
            f"Source Context: {source_node.context}\n\n"
            f"Target Agent: {target_node.agent}\n"
            f"Target Input: {target_node.input_text}\n"
            f"Transfer Context Sent: {transfer.context_sent}\n\n"
            f"List any critical facts, constraints, or IDs dropped in the handoff. "
            f"If facts were lost, start with 'CONTEXT_LOSS: ' followed by lost facts. "
            f"If all key context was preserved, reply 'NO_LOSS'."
        )
        response = await self._llm.generate(prompt)
        if "CONTEXT_LOSS:" in response.upper():
            return DetectorFinding(
                mode="context_loss",
                severity="warning",
                message=f"LLM judge identified semantic context loss in transfer {transfer.from_agent} -> {transfer.to_agent}",
                node_ids=[source_node.id, target_node.id],
                transfer_ids=[transfer.id],
                evidence={"judge_response": response},
            )
        return None


class MisroutingDetector:
    """Detects non-deterministic mis-routing and deviation from golden routes across run graphs."""

    def detect_nondeterminism(
        self,
        graphs: Sequence[RunGraph],
        consistency_threshold: float = 1.0,
    ) -> list[DetectorFinding]:
        findings: list[DetectorFinding] = []

        # Group graphs by goal
        by_goal: dict[str, list[RunGraph]] = {}
        for g in graphs:
            if g.goal:
                by_goal.setdefault(g.goal, []).append(g)

        for goal, group in by_goal.items():
            if len(group) < 2:
                continue

            first_targets: list[str] = []
            graph_targets: list[tuple[RunGraph, str]] = []

            for g in group:
                if g.transfers:
                    target = g.transfers[0].to_agent
                elif g.nodes:
                    target = g.nodes[0].agent
                else:
                    target = "none"
                first_targets.append(target)
                graph_targets.append((g, target))

            counts = Counter(first_targets)
            modal_target, modal_count = counts.most_common(1)[0]
            modal_share = modal_count / len(first_targets)

            if modal_share < consistency_threshold:
                affected_node_ids = [g.nodes[0].id for g, _ in graph_targets if g.nodes]
                affected_transfer_ids = [g.transfers[0].id for g, _ in graph_targets if g.transfers]

                findings.append(
                    DetectorFinding(
                        mode="misrouting",
                        severity="critical" if modal_share < 0.5 else "warning",
                        message=(
                            f"Non-deterministic misrouting for goal '{goal}': modal target '{modal_target}' "
                            f"has {modal_share:.0%} share across {len(group)} runs (threshold: {consistency_threshold:.0%})"
                        ),
                        node_ids=affected_node_ids,
                        transfer_ids=affected_transfer_ids,
                        evidence={
                            "goal": goal,
                            "distribution": dict(counts),
                            "modal_share": modal_share,
                            "total_runs": len(group),
                        },
                    )
                )

        return findings

    def detect_against_golden(
        self,
        graphs: Sequence[RunGraph],
        expected_route: dict[str, str],
    ) -> list[DetectorFinding]:
        findings: list[DetectorFinding] = []

        for g in graphs:
            if not g.goal or g.goal not in expected_route:
                continue

            expected = expected_route[g.goal]
            actual = (
                g.transfers[0].to_agent if g.transfers else (g.nodes[0].agent if g.nodes else "")
            )

            if actual != expected:
                findings.append(
                    DetectorFinding(
                        mode="misrouting",
                        severity="critical",
                        message=(
                            f"Graph '{g.run_id}' routed to '{actual}' instead of golden target '{expected}' "
                            f"for goal '{g.goal}'"
                        ),
                        node_ids=[g.nodes[0].id] if g.nodes else [],
                        transfer_ids=[g.transfers[0].id] if g.transfers else [],
                        evidence={
                            "run_id": g.run_id,
                            "goal": g.goal,
                            "actual_route": actual,
                            "expected_route": expected,
                        },
                    )
                )

        return findings
