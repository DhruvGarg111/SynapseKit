"""RunGraph abstraction for multi-agent orchestration evaluation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...agents.multi.handoff import HandoffResult
    from .detectors import DetectorFinding


def _nearest_node_for_agent(
    node_records: list[tuple[int, RunNode]],
    agent: str,
    event_idx: int,
    *,
    prefer_before: bool,
) -> RunNode | None:
    """Find the node for ``agent`` closest to ``event_idx`` in event order.

    Used to resolve a handoff event's source/target node by matching agent
    name against the surrounding node events, instead of assuming a fixed
    positional offset (which breaks whenever a handoff isn't immediately
    sandwiched between exactly one node event on each side).
    """
    candidates = [(idx, node) for idx, node in node_records if node.agent == agent]
    if not candidates:
        return None
    if prefer_before:
        before = [c for c in candidates if c[0] <= event_idx]
        if before:
            return before[-1][1]
        return candidates[0][1]
    after = [c for c in candidates if c[0] >= event_idx]
    if after:
        return after[0][1]
    return candidates[-1][1]


@dataclass(slots=True)
class RunNode:
    """A single execution node within a run graph."""

    id: str
    agent: str
    step: int
    input_text: str
    output_text: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Transfer:
    """A transfer (handoff) edge between two nodes in a run graph."""

    id: str
    source: str
    target: str
    from_agent: str
    to_agent: str
    reason: str = ""
    context_sent: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None


@dataclass
class RunGraph:
    """Directed graph representing an agent run execution."""

    run_id: str
    goal: str
    nodes: list[RunNode] = field(default_factory=list)
    transfers: list[Transfer] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def agent_sequence(self) -> list[str]:
        """Return sequence of agent names in step order."""
        sorted_nodes = sorted(self.nodes, key=lambda n: n.step)
        return [n.agent for n in sorted_nodes]

    @classmethod
    def from_handoff_result(
        cls,
        result: HandoffResult,
        goal: str = "",
        run_id: str | None = None,
    ) -> RunGraph:
        """Adapt a HandoffResult into a RunGraph.

        ``HandoffChain.run`` history entries only carry ``agent``/``input``/
        ``output``/``timestamp`` — there is no structured per-step ``context``
        concept in :class:`HandoffChain`, so ``RunNode.context`` stays empty
        unless the caller populates a ``"context"`` key in ``result.history``
        itself. ``ContextLossDetector`` still works in that case via its
        ``input_text``/``output_text`` fallback.
        """
        r_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        nodes: list[RunNode] = []
        transfers: list[Transfer] = []

        history = getattr(result, "history", [])
        for i, item in enumerate(history):
            if not isinstance(item, dict):
                continue
            node_id = f"{r_id}_node_{i}"
            agent_name = str(item.get("agent", f"agent_{i}"))
            input_text = str(item.get("input", ""))
            output_text = str(item.get("output", ""))
            ctx = item.get("context", {})
            context_dict = ctx if isinstance(ctx, dict) else {}

            node = RunNode(
                id=node_id,
                agent=agent_name,
                step=i,
                input_text=input_text,
                output_text=output_text,
                context=context_dict,
                timestamp=item.get("timestamp"),
            )
            nodes.append(node)

        for i in range(len(nodes) - 1):
            src_node = nodes[i]
            tgt_node = nodes[i + 1]
            transfer_id = f"{r_id}_transfer_{i}"
            transfer = Transfer(
                id=transfer_id,
                source=src_node.id,
                target=tgt_node.id,
                from_agent=src_node.agent,
                to_agent=tgt_node.agent,
                reason=f"Handoff from {src_node.agent} to {tgt_node.agent}",
                context_sent=dict(src_node.context),
                timestamp=tgt_node.timestamp,
            )
            transfers.append(transfer)

        return cls(
            run_id=r_id,
            goal=goal,
            nodes=nodes,
            transfers=transfers,
        )

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> RunGraph:
        """Build a RunGraph from SynapseKit Live bus event history."""
        r_id = f"run_{uuid.uuid4().hex[:8]}"
        goal = ""
        nodes: list[RunNode] = []
        transfers: list[Transfer] = []
        # (event_index, node) so transfer endpoints can be resolved by agent name
        # against the events they actually straddle, instead of by position.
        node_records: list[tuple[int, RunNode]] = []
        # Raw transfer descriptors, resolved to node ids after all nodes are known.
        pending_transfers: list[dict[str, Any]] = []

        step_counter = 0

        for event_idx, event in enumerate(events):
            if not isinstance(event, dict):
                continue

            kind = event.get("kind", "")
            attrs = event.get("attributes", {})
            if not isinstance(attrs, dict):
                attrs = {}

            if kind == "run.start" and "label" in attrs:
                goal = str(attrs.get("label", ""))

            # Handle node-like events (agent.step, handoff, agent.call, orchestration.step)
            agent_name = event.get("agent") or attrs.get("agent") or event.get("name")
            input_text = event.get("input") or attrs.get("input") or attrs.get("query", "")
            output_text = event.get("output") or attrs.get("output") or attrs.get("result", "")

            if agent_name and (input_text or output_text or kind.startswith("agent.")):
                node_id = f"{r_id}_node_{step_counter}"
                ctx = event.get("context") or attrs.get("context") or {}
                context_dict = ctx if isinstance(ctx, dict) else {}
                node = RunNode(
                    id=node_id,
                    agent=str(agent_name),
                    step=step_counter,
                    input_text=str(input_text),
                    output_text=str(output_text),
                    context=context_dict,
                    timestamp=event.get("ts"),
                )
                nodes.append(node)
                node_records.append((event_idx, node))
                step_counter += 1

            # Handle transfer-like events
            from_agent = event.get("from_agent") or attrs.get("from_agent")
            to_agent = event.get("to_agent") or attrs.get("to_agent")
            if from_agent and to_agent:
                t_ctx = event.get("context_sent") or attrs.get("context_sent") or {}
                pending_transfers.append(
                    {
                        "event_idx": event_idx,
                        "from_agent": str(from_agent),
                        "to_agent": str(to_agent),
                        "reason": str(event.get("reason") or attrs.get("reason", "")),
                        "context_sent": t_ctx if isinstance(t_ctx, dict) else {},
                        "timestamp": event.get("ts"),
                    }
                )

        for pt in pending_transfers:
            event_idx = pt["event_idx"]
            src_node = _nearest_node_for_agent(
                node_records, pt["from_agent"], event_idx, prefer_before=True
            )
            tgt_node = _nearest_node_for_agent(
                node_records, pt["to_agent"], event_idx, prefer_before=False
            )
            src_id = src_node.id if src_node is not None else "src"
            tgt_id = tgt_node.id if tgt_node is not None else "tgt"
            transfers.append(
                Transfer(
                    id=f"{r_id}_transfer_{len(transfers)}",
                    source=src_id,
                    target=tgt_id,
                    from_agent=pt["from_agent"],
                    to_agent=pt["to_agent"],
                    reason=pt["reason"],
                    context_sent=pt["context_sent"],
                    timestamp=pt["timestamp"],
                )
            )

        # If events created nodes but no transfers were explicitly in events, link adjacent nodes
        if nodes and not transfers:
            for i in range(len(nodes) - 1):
                src_node = nodes[i]
                tgt_node = nodes[i + 1]
                if src_node.agent != tgt_node.agent or len(nodes) > 1:
                    transfers.append(
                        Transfer(
                            id=f"{r_id}_transfer_{i}",
                            source=src_node.id,
                            target=tgt_node.id,
                            from_agent=src_node.agent,
                            to_agent=tgt_node.agent,
                            reason=f"Transfer from {src_node.agent} to {tgt_node.agent}",
                            context_sent=dict(src_node.context),
                            timestamp=tgt_node.timestamp,
                        )
                    )

        return cls(
            run_id=r_id,
            goal=goal,
            nodes=nodes,
            transfers=transfers,
        )

    def to_live_graph(
        self,
        findings: list[DetectorFinding] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Format nodes and edges for live.publish_graph, with optional finding annotations."""
        finding_nodes: set[str] = set()
        finding_transfers: set[str] = set()
        finding_map: dict[str, list[str]] = {}

        if findings:
            for f in findings:
                for nid in f.node_ids:
                    finding_nodes.add(nid)
                    finding_map.setdefault(nid, []).append(f.mode)
                for tid in f.transfer_ids:
                    finding_transfers.add(tid)
                    finding_map.setdefault(tid, []).append(f.mode)

        live_nodes: list[dict[str, Any]] = []
        for n in self.nodes:
            label = f"{n.agent} (step {n.step})"
            group = n.agent
            if n.id in finding_nodes:
                modes = ",".join(sorted(set(finding_map[n.id])))
                label = f"{label} [{modes.upper()}]"
                group = f"{n.agent}_flagged"

            live_nodes.append(
                {
                    "id": n.id,
                    "label": label,
                    "group": group,
                }
            )

        live_edges: list[dict[str, Any]] = []
        for t in self.transfers:
            label = f"{t.from_agent} -> {t.to_agent}"
            group = "normal"
            if t.id in finding_transfers:
                modes = ",".join(sorted(set(finding_map[t.id])))
                label = f"{label} [{modes.upper()}]"
                group = modes

            live_edges.append(
                {
                    "source": t.source,
                    "target": t.target,
                    "label": label,
                    "group": group,
                }
            )

        return live_nodes, live_edges
