"""Tests for kindle.graph — LangGraph state machine construction and routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.graph import ORDERED_STAGES, _wire_edges, _wrap, build_graph
from kindle.stages.qa import qa_router
from kindle.state import KindleState


# ---------------------------------------------------------------------------
# Helper: create a minimal mock UI
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# ORDERED_STAGES constant
# ---------------------------------------------------------------------------


class TestOrderedStages:
    """Verify the canonical stage ordering."""

    def test_contains_all_six_stages(self):
        assert ORDERED_STAGES == ["grill", "research", "architect", "dev", "qa", "package"]

    def test_length(self):
        assert len(ORDERED_STAGES) == 6


# ---------------------------------------------------------------------------
# _wrap helper
# ---------------------------------------------------------------------------


class TestWrap:
    """Tests for the _wrap() UI-injection helper."""

    async def test_wrap_injects_ui_into_node_function(self):
        """Wrapped function should be called with (state, ui)."""
        mock_fn = AsyncMock(return_value={"current_stage": "grill"})
        ui = _make_ui()
        wrapped = _wrap(mock_fn, ui)

        state: KindleState = {"idea": "test"}  # type: ignore[typeddict-item]
        result = await wrapped(state)

        mock_fn.assert_awaited_once_with(state, ui)
        assert result == {"current_stage": "grill"}

    async def test_wrap_preserves_function_name(self):
        """functools.wraps should carry over the original __name__."""

        async def my_custom_node(state: KindleState, ui: MagicMock) -> dict:
            return {}

        wrapped = _wrap(my_custom_node, _make_ui())
        assert wrapped.__name__ == "my_custom_node"


# ---------------------------------------------------------------------------
# build_graph — full pipeline (all stages)
# ---------------------------------------------------------------------------


class TestBuildGraphFullPipeline:
    """Test build_graph() with default entry_stage (all 6 stages)."""

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_returns_compiled_graph(self, *_mocks: AsyncMock):
        """build_graph() should return a compiled LangGraph."""
        ui = _make_ui()
        graph = build_graph(ui)
        # CompiledStateGraph has an invoke method
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_graph_has_all_six_nodes(self, *_mocks: AsyncMock):
        """Compiled graph should contain all 6 stage nodes."""
        ui = _make_ui()
        graph = build_graph(ui)
        # The compiled graph exposes a get_graph() method for inspection
        graph_def = graph.get_graph()
        node_ids = {n.id for n in graph_def.nodes.values()}
        for stage in ORDERED_STAGES:
            assert stage in node_ids, f"Missing node: {stage}"

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_start_edge_points_to_grill(self, *_mocks: AsyncMock):
        """START edge should connect to 'grill' for default entry."""
        ui = _make_ui()
        graph = build_graph(ui)
        graph_def = graph.get_graph()
        # Check edges from __start__
        start_edges = [e for e in graph_def.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "grill"


# ---------------------------------------------------------------------------
# build_graph — custom entry_stage (skip earlier stages)
# ---------------------------------------------------------------------------


class TestBuildGraphEntryStage:
    """Test build_graph() with entry_stage to skip earlier stages."""

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_entry_stage_dev_skips_earlier_nodes(self, *_mocks: AsyncMock):
        """entry_stage='dev' should only include dev, qa, package."""
        ui = _make_ui()
        graph = build_graph(ui, entry_stage="dev")
        graph_def = graph.get_graph()
        node_ids = {n.id for n in graph_def.nodes.values()}

        # Should be present
        assert "dev" in node_ids
        assert "qa" in node_ids
        assert "package" in node_ids

        # Should be absent (skipped stages)
        assert "grill" not in node_ids
        assert "research" not in node_ids
        assert "architect" not in node_ids

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_entry_stage_dev_start_points_to_dev(self, *_mocks: AsyncMock):
        """START edge should connect to 'dev' when entry_stage='dev'."""
        ui = _make_ui()
        graph = build_graph(ui, entry_stage="dev")
        graph_def = graph.get_graph()
        start_edges = [e for e in graph_def.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "dev"

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_entry_stage_qa_skips_to_qa(self, *_mocks: AsyncMock):
        """entry_stage='qa' should only include qa and package."""
        ui = _make_ui()
        graph = build_graph(ui, entry_stage="qa")
        graph_def = graph.get_graph()
        node_ids = {n.id for n in graph_def.nodes.values()}

        assert "qa" in node_ids
        assert "package" in node_ids
        assert "grill" not in node_ids
        assert "research" not in node_ids
        assert "architect" not in node_ids
        assert "dev" not in node_ids

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_entry_stage_package_only(self, *_mocks: AsyncMock):
        """entry_stage='package' should only include the package node."""
        ui = _make_ui()
        graph = build_graph(ui, entry_stage="package")
        graph_def = graph.get_graph()
        node_ids = {n.id for n in graph_def.nodes.values()}

        assert "package" in node_ids
        # All other stages absent
        for stage in ("grill", "research", "architect", "dev", "qa"):
            assert stage not in node_ids

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_entry_stage_research(self, *_mocks: AsyncMock):
        """entry_stage='research' should include research through package."""
        ui = _make_ui()
        graph = build_graph(ui, entry_stage="research")
        graph_def = graph.get_graph()
        node_ids = {n.id for n in graph_def.nodes.values()}

        assert "grill" not in node_ids
        assert "research" in node_ids
        assert "architect" in node_ids
        assert "dev" in node_ids
        assert "qa" in node_ids
        assert "package" in node_ids


# ---------------------------------------------------------------------------
# build_graph — invalid entry_stage fallback
# ---------------------------------------------------------------------------


class TestBuildGraphInvalidEntry:
    """Test build_graph() with an invalid entry_stage falls back to 'grill'."""

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_unknown_entry_stage_falls_back_to_grill(self, *_mocks: AsyncMock):
        """Invalid entry_stage should silently fall back to 'grill'."""
        ui = _make_ui()
        graph = build_graph(ui, entry_stage="nonexistent")
        graph_def = graph.get_graph()

        # All 6 nodes should be present (full pipeline)
        node_ids = {n.id for n in graph_def.nodes.values()}
        for stage in ORDERED_STAGES:
            assert stage in node_ids

        # START points to grill
        start_edges = [e for e in graph_def.edges if e.source == "__start__"]
        assert start_edges[0].target == "grill"

    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    def test_empty_string_entry_stage_falls_back_to_grill(self, *_mocks: AsyncMock):
        """Empty string entry_stage should fall back to 'grill'."""
        ui = _make_ui()
        graph = build_graph(ui, entry_stage="")
        graph_def = graph.get_graph()

        start_edges = [e for e in graph_def.edges if e.source == "__start__"]
        assert start_edges[0].target == "grill"


# ---------------------------------------------------------------------------
# _wire_edges internal helper
# ---------------------------------------------------------------------------


class TestWireEdges:
    """Tests for the _wire_edges() edge-wiring helper."""

    def test_wire_edges_with_all_stages_active(self):
        """All stages active: all simple edges + QA conditional edge added."""
        from langgraph.graph import StateGraph

        graph = StateGraph(KindleState)
        # Add dummy nodes
        for stage in ORDERED_STAGES:
            graph.add_node(stage, lambda s: s)

        active = set(ORDERED_STAGES)
        _wire_edges(graph, active)

        # The graph should compile without errors (edges are valid)
        # We compile after adding START edge
        from langgraph.graph import START

        graph.add_edge(START, "grill")
        compiled = graph.compile()
        assert compiled is not None

    def test_wire_edges_with_partial_stages(self):
        """Only qa + package active: only the conditional edge should exist."""
        from langgraph.graph import START, StateGraph

        graph = StateGraph(KindleState)
        graph.add_node("qa", lambda s: s)
        graph.add_node("package", lambda s: s)

        active = {"qa", "package"}
        _wire_edges(graph, active)

        graph.add_edge(START, "qa")
        compiled = graph.compile()
        assert compiled is not None


# ---------------------------------------------------------------------------
# qa_router — conditional routing logic
# ---------------------------------------------------------------------------


class TestQaRouter:
    """Tests for the QA conditional router function."""

    def test_returns_package_when_both_passed(self):
        """Both qa_passed and cpo_passed True -> 'package'."""
        state: KindleState = {
            "qa_passed": True,
            "cpo_passed": True,
            "qa_retries": 0,
            "cpo_retries": 0,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "package"

    def test_returns_qa_when_qa_failed_retries_remain(self):
        """qa_passed is False, retries below max -> 'qa' (self-heal)."""
        state: KindleState = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 2,
            "cpo_retries": 0,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "qa"

    def test_returns_package_when_qa_retries_exhausted(self):
        """qa_passed is False, retries at max -> 'package' (give up)."""
        state: KindleState = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 10,
            "cpo_retries": 0,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "package"

    def test_returns_qa_when_cpo_failed_retries_remain(self):
        """qa_passed True but cpo_passed False, retries remain -> 'qa'."""
        state: KindleState = {
            "qa_passed": True,
            "cpo_passed": False,
            "qa_retries": 0,
            "cpo_retries": 3,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "qa"

    def test_returns_package_when_cpo_retries_exhausted(self):
        """qa_passed True but cpo_passed False, cpo retries at max -> 'package'."""
        state: KindleState = {
            "qa_passed": True,
            "cpo_passed": False,
            "qa_retries": 0,
            "cpo_retries": 10,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "package"

    def test_returns_package_when_qa_passed_true_via_defaults(self):
        """qa_passed=True, cpo_passed=True with minimal state."""
        state: KindleState = {
            "qa_passed": True,
            "cpo_passed": True,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "package"

    def test_uses_default_retry_limits_when_not_in_state(self):
        """Missing max retries should default to 10."""
        state: KindleState = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 5,
            "cpo_retries": 0,
        }  # type: ignore[typeddict-item]
        # Retries (5) < default max (10), so should self-heal
        assert qa_router(state) == "qa"

    def test_returns_qa_on_empty_state(self):
        """Completely empty state: defaults to qa_passed=False, retries=0 < 10."""
        state: KindleState = {}  # type: ignore[typeddict-item]
        assert qa_router(state) == "qa"

    def test_returns_package_when_qa_retries_exceed_max(self):
        """qa_retries > max_qa_retries should also give up."""
        state: KindleState = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 15,
            "cpo_retries": 0,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "package"

    def test_returns_package_when_cpo_retries_exceed_max(self):
        """cpo_retries > max_cpo_retries should also give up."""
        state: KindleState = {
            "qa_passed": True,
            "cpo_passed": False,
            "qa_retries": 0,
            "cpo_retries": 15,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "package"

    def test_boundary_retries_equal_to_max_gives_up(self):
        """Exactly at the retry limit should give up (>= comparison)."""
        state: KindleState = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 3,
            "cpo_retries": 0,
            "max_qa_retries": 3,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "package"

    def test_boundary_retries_one_below_max_self_heals(self):
        """One retry below the limit should still self-heal."""
        state: KindleState = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 2,
            "cpo_retries": 0,
            "max_qa_retries": 3,
            "max_cpo_retries": 10,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "qa"

    def test_zero_max_retries_immediately_gives_up(self):
        """max_qa_retries=0 means never retry; go straight to package."""
        state: KindleState = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 0,
            "cpo_retries": 0,
            "max_qa_retries": 0,
            "max_cpo_retries": 0,
        }  # type: ignore[typeddict-item]
        assert qa_router(state) == "package"
