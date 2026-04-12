"""LangGraph state machine — orchestrates the 6-stage application factory.

    Grill → Research → Architect → Dev → QA (↔ self-heal) → Package

QA has a conditional edge that re-runs itself with an internal fix agent when checks fail.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kindle.stages.architect import architect_node
from kindle.stages.dev import dev_node
from kindle.stages.grill import grill_node
from kindle.stages.package import package_node
from kindle.stages.qa import qa_node, qa_router
from kindle.stages.research import research_node
from kindle.state import KindleState
from kindle.ui import UI


def _wrap(
    fn: Callable[[KindleState, UI], Awaitable[dict]],
    ui: UI,
) -> Callable[[KindleState], Awaitable[dict]]:
    """Wrap a node function so it receives the shared UI instance."""

    @functools.wraps(fn)
    async def wrapper(state: KindleState) -> dict:
        return await fn(state, ui)

    return wrapper


ORDERED_STAGES = ["grill", "research", "architect", "dev", "qa", "package"]

_NODE_FACTORIES = {
    "grill": lambda ui: [("grill", grill_node)],
    "research": lambda ui: [("research", research_node)],
    "architect": lambda ui: [("architect", architect_node)],
    "dev": lambda ui: [("dev", dev_node)],
    "qa": lambda ui: [("qa", qa_node)],
    "package": lambda ui: [("package", package_node)],
}

# Simple edges: (source_node, target_node, stages that must be active)
_SIMPLE_EDGES: list[tuple[str, str, set[str]]] = [
    ("grill", "research", {"grill", "research"}),
    ("research", "architect", {"research", "architect"}),
    ("architect", "dev", {"architect", "dev"}),
    ("dev", "qa", {"dev", "qa"}),
    # QA → Package is handled by conditional edge
    ("package", END, {"package"}),
]


def _wire_edges(graph: StateGraph, active: set[str]) -> None:
    """Add edges between active nodes in the graph."""
    for src, dst, required in _SIMPLE_EDGES:
        if required <= active:
            graph.add_edge(src, dst)

    # QA conditional routing: on failure, qa_router maps qa→qa so the QA
    # node re-runs its internal fix agent; on success it routes to package.
    if {"qa", "package"} <= active:
        graph.add_conditional_edges(
            "qa",
            qa_router,
            {"package": "package", "qa": "qa"},
        )


def build_graph(ui: UI, *, entry_stage: str = "grill") -> CompiledStateGraph:
    """Construct and compile the application factory pipeline graph.

    ``entry_stage`` controls where the graph starts (default: grill).
    Used by ``resume`` to skip completed stages.
    """
    if entry_stage not in ORDERED_STAGES:
        entry_stage = "grill"

    entry_idx = ORDERED_STAGES.index(entry_stage)
    active = set(ORDERED_STAGES[entry_idx:])

    graph = StateGraph(KindleState)

    # Add nodes
    for stage in ORDERED_STAGES:
        if stage in active:
            for name, fn in _NODE_FACTORIES[stage](ui):
                graph.add_node(name, _wrap(fn, ui))  # type: ignore[call-overload]

    # Wire edges
    graph.add_edge(START, entry_stage)
    _wire_edges(graph, active)

    return graph.compile()
