"""Hand-built test fixtures for orchestration evaluation (no mocks)."""

from __future__ import annotations

from synapsekit.evaluation.orchestration.run_graph import RunGraph, RunNode, Transfer


def looping_run_graph() -> RunGraph:
    """Fixture producing a 6-node A->B->A->B->A->B periodic loop with near-duplicate inputs."""
    agents = ["triage", "specialist", "triage", "specialist", "triage", "specialist"]
    nodes: list[RunNode] = []
    transfers: list[Transfer] = []

    for i, agent in enumerate(agents):
        nodes.append(
            RunNode(
                id=f"loop_node_{i}",
                agent=agent,
                step=i,
                input_text="Please triage and process order ORD-100",
                output_text=f"Agent {agent} processed step {i} and handing off",
                context={"order_id": "ORD-100"},
            )
        )

    for i in range(len(nodes) - 1):
        transfers.append(
            Transfer(
                id=f"loop_transfer_{i}",
                source=nodes[i].id,
                target=nodes[i + 1].id,
                from_agent=nodes[i].agent,
                to_agent=nodes[i + 1].agent,
                reason="Handoff loop step",
                context_sent=dict(nodes[i].context),
            )
        )

    return RunGraph(
        run_id="run_looping_01",
        goal="Process order ORD-100",
        nodes=nodes,
        transfers=transfers,
    )


def context_loss_run_graph() -> RunGraph:
    """Fixture producing a transfer where critical facts (Order ID, Customer Name) are dropped."""
    node0 = RunNode(
        id="ctx_node_0",
        agent="intake",
        step=0,
        input_text="Customer John Smith inquiring about Order ORD-99823",
        output_text="Verified customer John Smith with Order ORD-99823 and account ACC-7712",
        context={"order_id": "ORD-99823", "customer": "John Smith", "account": "ACC-7712"},
    )
    node1 = RunNode(
        id="ctx_node_1",
        agent="billing",
        step=1,
        input_text="Please handle general billing question",
        output_text="Checked general account status",
        context={},
    )
    transfer = Transfer(
        id="ctx_transfer_0",
        source=node0.id,
        target=node1.id,
        from_agent="intake",
        to_agent="billing",
        reason="Handoff without context details",
        context_sent={},
    )
    return RunGraph(
        run_id="run_ctx_loss_01",
        goal="Resolve billing inquiry",
        nodes=[node0, node1],
        transfers=[transfer],
    )


def misrouting_run_graphs() -> list[RunGraph]:
    """Fixture producing 5 RunGraphs for the identical goal with non-deterministic first-hop target."""
    target_agents = ["billing", "technical", "billing", "general_support", "billing"]
    graphs: list[RunGraph] = []

    for idx, target in enumerate(target_agents):
        node0 = RunNode(
            id=f"mis_node_{idx}_0",
            agent="router",
            step=0,
            input_text="I need help with my monthly invoice payment",
            output_text=f"Routing query to {target}",
        )
        node1 = RunNode(
            id=f"mis_node_{idx}_1",
            agent=target,
            step=1,
            input_text="I need help with my monthly invoice payment",
            output_text="Handled query",
        )
        transfer = Transfer(
            id=f"mis_transfer_{idx}_0",
            source=node0.id,
            target=node1.id,
            from_agent="router",
            to_agent=target,
            reason=f"Routed to {target}",
        )
        graphs.append(
            RunGraph(
                run_id=f"run_misrouting_{idx}",
                goal="Resolve invoice payment issue",
                nodes=[node0, node1],
                transfers=[transfer],
            )
        )

    return graphs


def healthy_run_graph() -> RunGraph:
    """Fixture producing a healthy, single-hop run graph with no loops, full context retention, and clean routing."""
    node0 = RunNode(
        id="healthy_node_0",
        agent="triage",
        step=0,
        input_text="Need refund for Order ORD-44012",
        output_text="Forwarding Order ORD-44012 to refund specialist Sarah Connor",
        context={"order_id": "ORD-44012", "agent_assigned": "Sarah Connor"},
    )
    node1 = RunNode(
        id="healthy_node_1",
        agent="refund_specialist",
        step=1,
        input_text="Forwarding Order ORD-44012 to refund specialist Sarah Connor",
        output_text="Refund issued for Order ORD-44012 for Sarah Connor",
        context={"order_id": "ORD-44012", "status": "refunded"},
    )
    transfer = Transfer(
        id="healthy_transfer_0",
        source=node0.id,
        target=node1.id,
        from_agent="triage",
        to_agent="refund_specialist",
        reason="Order refund transfer",
        context_sent={"order_id": "ORD-44012", "agent_assigned": "Sarah Connor"},
    )
    return RunGraph(
        run_id="run_healthy_01",
        goal="Issue order refund",
        nodes=[node0, node1],
        transfers=[transfer],
    )
