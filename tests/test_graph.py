"""Tests for kindle.graph — LangGraph state machine builder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from langgraph.graph import END, START

from kindle.graph import ORDERED_STAGES, _NODE_FACTORIES, _SIMPLE_EDGES, _wire_edges, _wrap, build_graph
from kindle.state import KindleState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods graph wiring actually calls."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.auto_approve = False
    return ui


def _make_state(**overrides: object) -> dict:
    """Build a minimal KindleState dict for testing wrapped functions."""
    state: dict = {
        "idea": "test app",
        "project_dir": "/tmp/kindle-test",
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# ORDERED_STAGES & _NODE_FACTORIES constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify the stage ordering and factory registry are consistent."""

    def test_ordered_stages_has_six_stages(self) -> None:
        assert len(ORDERED_STAGES) == 6

    def test_ordered_stages_correct_order(self) -> None:
        assert ORDERED_STAGES == ["grill", "research", "architect", "dev", "qa", "package"]

    def test_node_factories_keys_match_ordered_stages(self) -> None:
        assert set(_NODE_FACTORIES.keys()) == set(ORDERED_STAGES)

    def test_each_factory_returns_list_of_tuples(self) -> None:
        ui = _make_ui()
        for stage in ORDERED_STAGES:
            result = _NODE_FACTORIES[stage](ui)
            assert isinstance(result, list)
            for name, fn in result:
                assert isinstance(name, str)
                assert callable(fn)

    def test_each_factory_returns_correct_name(self) -> None:
        """Factory for stage 'X' should return a node named 'X'."""
        ui = _make_ui()
        for stage in ORDERED_STAGES:
            pairs = _NODE_FACTORIES[stage](ui)
            names = [name for name, _ in pairs]
            assert stage in names


# ---------------------------------------------------------------------------
# _wrap — wraps async node functions with UI injection
# ---------------------------------------------------------------------------


class TestWrap:
    """_wrap should inject the UI instance into the wrapped node function."""

    @pytest.mark.asyncio
    async def test_wrap_passes_state_and_ui_to_inner_fn(self) -> None:
        """The wrapped function should call the original fn(state, ui)."""
        inner = AsyncMock(return_value={"qa_passed": True})
        ui = _make_ui()
        wrapped = _wrap(inner, ui)

        state = _make_state()
        result = await wrapped(state)

        inner.assert_awaited_once_with(state, ui)
        assert result == {"qa_passed": True}

    @pytest.mark.asyncio
    async def test_wrap_returns_dict_from_inner(self) -> None:
        """The wrapped function returns whatever the inner function returns."""
        expected = {"architecture": "microservices", "dev_tasks": []}
        inner = AsyncMock(return_value=expected)
        ui = _make_ui()
        wrapped = _wrap(inner, ui)

        result = await wrapped(_make_state())
        assert result is expected

    @pytest.mark.asyncio
    async def test_wrap_preserves_function_name(self) -> None:
        """functools.wraps should preserve the original function name."""

        async def my_custom_node(state: KindleState, ui: object) -> dict:
            return {}

        ui = _make_ui()
        wrapped = _wrap(my_custom_node, ui)
        assert wrapped.__name__ == "my_custom_node"

    @pytest.mark.asyncio
    async def test_wrap_propagates_exception(self) -> None:
        """Exceptions from the inner function should propagate through."""
        inner = AsyncMock(side_effect=RuntimeError("boom"))
        ui = _make_ui()
        wrapped = _wrap(inner, ui)

        with pytest.raises(RuntimeError, match="boom"):
            await wrapped(_make_state())

    @pytest.mark.asyncio
    async def test_wrap_different_ui_instances_are_independent(self) -> None:
        """Each _wrap call binds a distinct UI instance."""
        inner = AsyncMock(return_value={})
        ui_a = _make_ui()
        ui_b = _make_ui()

        wrapped_a = _wrap(inner, ui_a)
        wrapped_b = _wrap(inner, ui_b)

        state = _make_state()
        await wrapped_a(state)
        await wrapped_b(state)

        # inner should have been called with both UIs
        calls = inner.await_args_list
        assert calls[0].args[1] is ui_a
        assert calls[1].args[1] is ui_b


# ---------------------------------------------------------------------------
# _wire_edges — edge wiring logic
# ---------------------------------------------------------------------------


class TestWireEdges:
    """_wire_edges should add linear edges and the QA conditional loop."""

    def test_all_stages_active_adds_all_simple_edges(self) -> None:
        """With all stages active, every simple edge should be wired."""
        graph = MagicMock()
        active = set(ORDERED_STAGES)

        _wire_edges(graph, active)

        # Check simple edges
        expected_simple_calls = [
            call("grill", "research"),
            call("research", "architect"),
            call("architect", "dev"),
            call("dev", "qa"),
            call("package", END),
        ]
        graph.add_edge.assert_has_calls(expected_simple_calls, any_order=True)
        assert graph.add_edge.call_count == len(expected_simple_calls)

    def test_all_stages_active_adds_conditional_qa_edge(self) -> None:
        """When qa and package are active, conditional edges should be wired."""
        graph = MagicMock()
        active = set(ORDERED_STAGES)

        _wire_edges(graph, active)

        graph.add_conditional_edges.assert_called_once()
        args = graph.add_conditional_edges.call_args
        assert args[0][0] == "qa"  # source node
        assert args[0][2] == {"package": "package", "qa": "qa"}  # route map

    def test_partial_stages_from_dev_onward(self) -> None:
        """Starting from dev: only dev→qa, qa conditional, and package→END."""
        graph = MagicMock()
        active = {"dev", "qa", "package"}

        _wire_edges(graph, active)

        edge_calls = [c.args for c in graph.add_edge.call_args_list]
        assert ("dev", "qa") in edge_calls
        assert ("package", END) in edge_calls
        # Earlier edges should NOT be present
        assert ("grill", "research") not in edge_calls
        assert ("research", "architect") not in edge_calls
        assert ("architect", "dev") not in edge_calls

        graph.add_conditional_edges.assert_called_once()

    def test_only_package_active(self) -> None:
        """With only 'package' active, only package→END should exist."""
        graph = MagicMock()
        active = {"package"}

        _wire_edges(graph, active)

        assert graph.add_edge.call_count == 1
        graph.add_edge.assert_called_with("package", END)
        graph.add_conditional_edges.assert_not_called()

    def test_qa_without_package_no_conditional_edge(self) -> None:
        """If 'package' is not active, no conditional edge from QA."""
        graph = MagicMock()
        active = {"qa"}

        _wire_edges(graph, active)

        graph.add_conditional_edges.assert_not_called()

    def test_empty_active_set(self) -> None:
        """No active stages → no edges at all."""
        graph = MagicMock()
        _wire_edges(graph, set())

        graph.add_edge.assert_not_called()
        graph.add_conditional_edges.assert_not_called()

    def test_simple_edges_constant_is_well_formed(self) -> None:
        """Each entry in _SIMPLE_EDGES has the expected structure."""
        for src, dst, required in _SIMPLE_EDGES:
            assert isinstance(src, str)
            assert isinstance(dst, str)
            assert isinstance(required, set)
            assert all(isinstance(s, str) for s in required)


# ---------------------------------------------------------------------------
# build_graph — full graph construction
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """build_graph should produce a compiled LangGraph with correct topology."""

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_build_graph_returns_compiled_graph(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """build_graph should return a CompiledStateGraph."""
        ui = _make_ui()
        compiled = build_graph(ui)
        # CompiledStateGraph has an invoke method
        assert hasattr(compiled, "invoke")

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_default_entry_stage_is_grill(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """Default entry should be 'grill' — all 6 nodes present."""
        ui = _make_ui()
        compiled = build_graph(ui)
        # The compiled graph should have all six stage nodes
        node_names = set(compiled.get_graph().nodes.keys())
        for stage in ORDERED_STAGES:
            assert stage in node_names, f"Missing node: {stage}"

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_entry_stage_dev_skips_earlier_stages(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """entry_stage='dev' should only include dev, qa, package."""
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="dev")
        node_names = set(compiled.get_graph().nodes.keys())
        # dev, qa, package should be present
        assert "dev" in node_names
        assert "qa" in node_names
        assert "package" in node_names
        # grill, research, architect should NOT be present
        assert "grill" not in node_names
        assert "research" not in node_names
        assert "architect" not in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_entry_stage_package_only_package_node(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """entry_stage='package' should only include the package node."""
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="package")
        node_names = set(compiled.get_graph().nodes.keys())
        assert "package" in node_names
        for stage in ["grill", "research", "architect", "dev", "qa"]:
            assert stage not in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_invalid_entry_stage_falls_back_to_grill(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """An unrecognized entry_stage should fall back to 'grill'."""
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="nonexistent")
        node_names = set(compiled.get_graph().nodes.keys())
        # Should have all 6 nodes since fallback is grill
        for stage in ORDERED_STAGES:
            assert stage in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_entry_stage_qa_includes_qa_and_package(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """entry_stage='qa' should include qa and package only."""
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="qa")
        node_names = set(compiled.get_graph().nodes.keys())
        assert "qa" in node_names
        assert "package" in node_names
        assert "grill" not in node_names
        assert "research" not in node_names
        assert "architect" not in node_names
        assert "dev" not in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_entry_stage_research_skips_grill(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """entry_stage='research' includes research through package, not grill."""
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="research")
        node_names = set(compiled.get_graph().nodes.keys())
        assert "grill" not in node_names
        for stage in ["research", "architect", "dev", "qa", "package"]:
            assert stage in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_graph_has_start_edge_to_entry(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """The compiled graph should have a START → entry_stage edge."""
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="architect")
        draw = compiled.get_graph()
        # The __start__ node should have an edge to architect
        start_edges = [e for e in draw.edges if e[0] == "__start__"]
        targets = {e[1] for e in start_edges}
        assert "architect" in targets


# ---------------------------------------------------------------------------
# build_graph — edge topology verification
# ---------------------------------------------------------------------------


class TestBuildGraphEdges:
    """Verify the edge structure of the compiled graph."""

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_full_graph_has_linear_edges(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """The full pipeline should have grill→research→...→package→__end__."""
        ui = _make_ui()
        compiled = build_graph(ui)
        draw = compiled.get_graph()
        edges = set(draw.edges)

        # Linear edges
        assert ("grill", "research") in edges
        assert ("research", "architect") in edges
        assert ("architect", "dev") in edges
        assert ("dev", "qa") in edges
        assert ("package", "__end__") in edges

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_full_graph_has_qa_conditional_edges(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """QA should have conditional edges to both package and itself (self-heal)."""
        ui = _make_ui()
        compiled = build_graph(ui)
        draw = compiled.get_graph()
        edges = set(draw.edges)

        # QA conditional: qa→package and qa→qa (self-heal loop)
        assert ("qa", "package") in edges
        assert ("qa", "qa") in edges

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_partial_graph_from_qa_has_correct_edges(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """From qa: start→qa, qa→package, qa→qa, package→end."""
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="qa")
        draw = compiled.get_graph()
        edges = set(draw.edges)

        assert ("__start__", "qa") in edges
        assert ("qa", "package") in edges
        assert ("qa", "qa") in edges
        assert ("package", "__end__") in edges
        # No earlier edges
        assert ("grill", "research") not in edges

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_package_only_graph_has_no_conditional_edges(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """entry_stage='package' means no QA conditional edges."""
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="package")
        draw = compiled.get_graph()
        edges = set(draw.edges)

        assert ("__start__", "package") in edges
        assert ("package", "__end__") in edges
        # No QA edges
        assert ("qa", "package") not in edges
        assert ("qa", "qa") not in edges
