"""Tests for kindle.graph — LangGraph orchestration / state machine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.graph import (
    ORDERED_STAGES,
    _NODE_FACTORIES,
    _SIMPLE_EDGES,
    _wire_edges,
    _wrap,
    build_graph,
)
from kindle.stages.qa import qa_router
from kindle.state import KindleState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    """Return a mock UI instance that satisfies the UI interface."""
    return MagicMock()


def _make_state(**overrides: object) -> KindleState:
    """Return a minimal KindleState dict with optional overrides."""
    base: dict = {
        "idea": "test idea",
        "project_id": "kindle_test",
        "project_dir": "/tmp/test",
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ORDERED_STAGES constant
# ---------------------------------------------------------------------------


class TestOrderedStages:
    """ORDERED_STAGES must define the correct pipeline order."""

    def test_exact_order(self) -> None:
        assert ORDERED_STAGES == [
            "grill",
            "research",
            "architect",
            "dev",
            "qa",
            "package",
        ]

    def test_length(self) -> None:
        assert len(ORDERED_STAGES) == 6

    def test_no_duplicates(self) -> None:
        assert len(ORDERED_STAGES) == len(set(ORDERED_STAGES))

    def test_first_stage_is_grill(self) -> None:
        assert ORDERED_STAGES[0] == "grill"

    def test_last_stage_is_package(self) -> None:
        assert ORDERED_STAGES[-1] == "package"

    def test_qa_precedes_package(self) -> None:
        assert ORDERED_STAGES.index("qa") < ORDERED_STAGES.index("package")


# ---------------------------------------------------------------------------
# _NODE_FACTORIES registry
# ---------------------------------------------------------------------------


class TestNodeFactories:
    """_NODE_FACTORIES should have one entry per ordered stage."""

    def test_keys_match_ordered_stages(self) -> None:
        assert set(_NODE_FACTORIES.keys()) == set(ORDERED_STAGES)

    def test_each_factory_returns_list_of_tuples(self) -> None:
        ui = _make_ui()
        for stage_name, factory in _NODE_FACTORIES.items():
            result = factory(ui)
            assert isinstance(result, list), f"{stage_name} factory did not return list"
            for name, fn in result:
                assert isinstance(name, str)
                assert callable(fn)

    def test_factory_tuple_name_matches_stage(self) -> None:
        ui = _make_ui()
        for stage_name, factory in _NODE_FACTORIES.items():
            pairs = factory(ui)
            names = [name for name, _ in pairs]
            assert stage_name in names


# ---------------------------------------------------------------------------
# _SIMPLE_EDGES
# ---------------------------------------------------------------------------


class TestSimpleEdges:
    """_SIMPLE_EDGES defines the linear pipeline wiring."""

    def test_contains_grill_to_research(self) -> None:
        assert ("grill", "research", {"grill", "research"}) in _SIMPLE_EDGES

    def test_contains_research_to_architect(self) -> None:
        assert ("research", "architect", {"research", "architect"}) in _SIMPLE_EDGES

    def test_contains_architect_to_dev(self) -> None:
        assert ("architect", "dev", {"architect", "dev"}) in _SIMPLE_EDGES

    def test_contains_dev_to_qa(self) -> None:
        assert ("dev", "qa", {"dev", "qa"}) in _SIMPLE_EDGES

    def test_package_to_end(self) -> None:
        from langgraph.graph import END

        assert ("package", END, {"package"}) in _SIMPLE_EDGES

    def test_no_direct_qa_to_package_simple_edge(self) -> None:
        """QA→Package is conditional, not a simple edge."""
        sources = [src for src, _, _ in _SIMPLE_EDGES]
        # QA should not appear as a simple-edge source (it's conditional)
        assert "qa" not in sources


# ---------------------------------------------------------------------------
# _wrap decorator
# ---------------------------------------------------------------------------


class TestWrap:
    """_wrap injects the UI instance and adapts the function signature."""

    async def test_wrapper_calls_original_with_state_and_ui(self) -> None:
        ui = _make_ui()
        inner = AsyncMock(return_value={"key": "value"})
        wrapped = _wrap(inner, ui)

        state: KindleState = _make_state()
        result = await wrapped(state)

        inner.assert_awaited_once_with(state, ui)
        assert result == {"key": "value"}

    async def test_wrapper_accepts_only_state_parameter(self) -> None:
        """The wrapped function should accept (state,) — not (state, ui)."""
        ui = _make_ui()
        inner = AsyncMock(return_value={})
        wrapped = _wrap(inner, ui)

        # Should work with a single positional argument
        await wrapped(_make_state())
        assert inner.await_count == 1

    async def test_wrapper_preserves_function_name(self) -> None:
        """functools.wraps should copy __name__ from the original."""
        ui = _make_ui()

        async def my_stage_node(state: KindleState, ui_arg: object) -> dict:
            return {}

        wrapped = _wrap(my_stage_node, ui)
        assert wrapped.__name__ == "my_stage_node"

    async def test_wrapper_preserves_docstring(self) -> None:
        ui = _make_ui()

        async def documented_node(state: KindleState, ui_arg: object) -> dict:
            """Stage docstring."""
            return {}

        wrapped = _wrap(documented_node, ui)
        assert wrapped.__doc__ == "Stage docstring."

    async def test_wrapper_propagates_exception(self) -> None:
        ui = _make_ui()
        inner = AsyncMock(side_effect=RuntimeError("stage failed"))
        wrapped = _wrap(inner, ui)

        with pytest.raises(RuntimeError, match="stage failed"):
            await wrapped(_make_state())

    async def test_wrapper_returns_whatever_inner_returns(self) -> None:
        ui = _make_ui()
        expected = {"qa_passed": True, "cpo_passed": False, "retries": 3}
        inner = AsyncMock(return_value=expected)
        wrapped = _wrap(inner, ui)

        result = await wrapped(_make_state())
        assert result is expected


# ---------------------------------------------------------------------------
# qa_router — conditional routing from QA stage
# ---------------------------------------------------------------------------


class TestQaRouter:
    """qa_router decides whether to loop back to QA or proceed to package."""

    def test_both_passed_returns_package(self) -> None:
        state = _make_state(qa_passed=True, cpo_passed=True)
        assert qa_router(state) == "package"

    def test_both_passed_ignores_retry_counts(self) -> None:
        state = _make_state(
            qa_passed=True,
            cpo_passed=True,
            qa_retries=0,
            cpo_retries=0,
        )
        assert qa_router(state) == "package"

    def test_qa_failed_retries_exceeded_returns_package(self) -> None:
        state = _make_state(
            qa_passed=False,
            cpo_passed=True,
            qa_retries=10,
            max_qa_retries=10,
        )
        assert qa_router(state) == "package"

    def test_cpo_failed_retries_exceeded_returns_package(self) -> None:
        state = _make_state(
            qa_passed=True,
            cpo_passed=False,
            cpo_retries=10,
            max_cpo_retries=10,
        )
        assert qa_router(state) == "package"

    def test_qa_failed_retries_remaining_returns_qa(self) -> None:
        state = _make_state(
            qa_passed=False,
            cpo_passed=True,
            qa_retries=2,
            max_qa_retries=10,
        )
        assert qa_router(state) == "qa"

    def test_cpo_failed_retries_remaining_returns_qa(self) -> None:
        state = _make_state(
            qa_passed=True,
            cpo_passed=False,
            cpo_retries=2,
            max_cpo_retries=10,
        )
        assert qa_router(state) == "qa"

    def test_both_failed_retries_remaining_returns_qa(self) -> None:
        state = _make_state(
            qa_passed=False,
            cpo_passed=False,
            qa_retries=1,
            cpo_retries=1,
            max_qa_retries=10,
            max_cpo_retries=10,
        )
        assert qa_router(state) == "qa"

    def test_defaults_when_state_keys_missing(self) -> None:
        """When keys are absent, defaults should cause loop-back (qa)."""
        state: KindleState = {}  # type: ignore[typeddict-item]
        # Defaults: qa_passed=False, cpo_passed=False, retries=0, max=10
        assert qa_router(state) == "qa"

    def test_qa_retries_equal_to_max_triggers_package(self) -> None:
        """Boundary: retries == max is the threshold for giving up."""
        state = _make_state(
            qa_passed=False,
            cpo_passed=True,
            qa_retries=5,
            max_qa_retries=5,
        )
        assert qa_router(state) == "package"

    def test_qa_retries_one_below_max_returns_qa(self) -> None:
        state = _make_state(
            qa_passed=False,
            cpo_passed=True,
            qa_retries=4,
            max_qa_retries=5,
        )
        assert qa_router(state) == "qa"

    def test_cpo_retries_equal_to_max_triggers_package(self) -> None:
        state = _make_state(
            qa_passed=True,
            cpo_passed=False,
            cpo_retries=3,
            max_cpo_retries=3,
        )
        assert qa_router(state) == "package"

    def test_both_failed_qa_retries_exceeded_short_circuits(self) -> None:
        """When both fail but qa_retries exceeded, returns package immediately."""
        state = _make_state(
            qa_passed=False,
            cpo_passed=False,
            qa_retries=10,
            cpo_retries=0,
            max_qa_retries=10,
            max_cpo_retries=10,
        )
        assert qa_router(state) == "package"

    def test_both_failed_cpo_retries_exceeded_short_circuits(self) -> None:
        """When both fail but cpo_retries exceeded, returns package."""
        state = _make_state(
            qa_passed=False,
            cpo_passed=False,
            qa_retries=0,
            cpo_retries=10,
            max_qa_retries=10,
            max_cpo_retries=10,
        )
        assert qa_router(state) == "package"


# ---------------------------------------------------------------------------
# _wire_edges
# ---------------------------------------------------------------------------


class TestWireEdges:
    """_wire_edges adds simple and conditional edges based on active stages."""

    def test_full_pipeline_adds_all_simple_edges(self) -> None:
        graph = MagicMock()
        active = set(ORDERED_STAGES)
        _wire_edges(graph, active)

        # Should add all simple edges
        expected_calls = [
            (("grill", "research"),),
            (("research", "architect"),),
            (("architect", "dev"),),
            (("dev", "qa"),),
        ]
        for call_args in expected_calls:
            assert call_args in [c.args for c in graph.add_edge.call_args_list]

    def test_full_pipeline_adds_package_to_end(self) -> None:
        from langgraph.graph import END

        graph = MagicMock()
        active = set(ORDERED_STAGES)
        _wire_edges(graph, active)

        graph.add_edge.assert_any_call("package", END)

    def test_full_pipeline_adds_qa_conditional_edges(self) -> None:
        graph = MagicMock()
        active = set(ORDERED_STAGES)
        _wire_edges(graph, active)

        graph.add_conditional_edges.assert_called_once_with(
            "qa",
            qa_router,
            {"package": "package", "qa": "qa"},
        )

    def test_partial_pipeline_qa_package_only(self) -> None:
        graph = MagicMock()
        active = {"qa", "package"}
        _wire_edges(graph, active)

        # Should NOT have grill→research, etc.
        simple_edge_calls = [c.args for c in graph.add_edge.call_args_list]
        assert ("grill", "research") not in simple_edge_calls
        assert ("research", "architect") not in simple_edge_calls

        # Should have package→END and conditional qa edge
        graph.add_conditional_edges.assert_called_once()

    def test_no_conditional_edge_without_qa_and_package(self) -> None:
        graph = MagicMock()
        active = {"grill", "research"}
        _wire_edges(graph, active)

        graph.add_conditional_edges.assert_not_called()

    def test_only_package_active(self) -> None:
        from langgraph.graph import END

        graph = MagicMock()
        active = {"package"}
        _wire_edges(graph, active)

        graph.add_edge.assert_called_once_with("package", END)
        graph.add_conditional_edges.assert_not_called()

    def test_empty_active_set_adds_no_edges(self) -> None:
        graph = MagicMock()
        _wire_edges(graph, set())

        graph.add_edge.assert_not_called()
        graph.add_conditional_edges.assert_not_called()


# ---------------------------------------------------------------------------
# build_graph — full integration
# ---------------------------------------------------------------------------


class TestBuildGraph:
    """build_graph constructs and compiles the LangGraph pipeline."""

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_default_entry_compiles_successfully(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        ui = _make_ui()
        compiled = build_graph(ui)
        # Should return a compiled graph object
        from langgraph.graph.state import CompiledStateGraph

        assert isinstance(compiled, CompiledStateGraph)

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_default_entry_has_all_six_nodes(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        ui = _make_ui()
        compiled = build_graph(ui)
        node_names = set(compiled.nodes.keys())
        # LangGraph adds __start__ and __end__ automatically
        for stage in ORDERED_STAGES:
            assert stage in node_names, f"Missing node: {stage}"

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_entry_stage_qa_has_only_qa_and_package(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="qa")
        node_names = set(compiled.nodes.keys())
        assert "qa" in node_names
        assert "package" in node_names
        # Earlier stages should NOT be present
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
    def test_entry_stage_dev_has_dev_qa_package(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="dev")
        node_names = set(compiled.nodes.keys())
        assert "dev" in node_names
        assert "qa" in node_names
        assert "package" in node_names
        assert "grill" not in node_names
        assert "research" not in node_names
        assert "architect" not in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_entry_stage_package_has_only_package(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="package")
        node_names = set(compiled.nodes.keys())
        assert "package" in node_names
        assert "qa" not in node_names
        assert "dev" not in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_invalid_entry_stage_falls_back_to_grill(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="nonexistent")
        node_names = set(compiled.nodes.keys())
        # Should fall back to grill, so all 6 stages present
        for stage in ORDERED_STAGES:
            assert stage in node_names, f"Missing node: {stage}"

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_empty_string_entry_stage_falls_back_to_grill(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="")
        node_names = set(compiled.nodes.keys())
        for stage in ORDERED_STAGES:
            assert stage in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_entry_stage_research_skips_grill(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        ui = _make_ui()
        compiled = build_graph(ui, entry_stage="research")
        node_names = set(compiled.nodes.keys())
        assert "grill" not in node_names
        assert "research" in node_names
        assert "architect" in node_names
        assert "dev" in node_names
        assert "qa" in node_names
        assert "package" in node_names

    @patch("kindle.graph.grill_node", new_callable=AsyncMock)
    @patch("kindle.graph.research_node", new_callable=AsyncMock)
    @patch("kindle.graph.architect_node", new_callable=AsyncMock)
    @patch("kindle.graph.dev_node", new_callable=AsyncMock)
    @patch("kindle.graph.qa_node", new_callable=AsyncMock)
    @patch("kindle.graph.package_node", new_callable=AsyncMock)
    def test_each_entry_stage_produces_correct_node_count(
        self, mock_pkg, mock_qa, mock_dev, mock_arch, mock_res, mock_grill
    ) -> None:
        """Parameterised check: entry at index i → (6 - i) active nodes."""
        ui = _make_ui()
        for i, stage in enumerate(ORDERED_STAGES):
            compiled = build_graph(ui, entry_stage=stage)
            node_names = set(compiled.nodes.keys())
            # Filter out LangGraph internal nodes (__start__, __end__)
            active_nodes = {n for n in node_names if not n.startswith("__")}
            expected_count = len(ORDERED_STAGES) - i
            assert len(active_nodes) == expected_count, (
                f"entry_stage={stage!r}: expected {expected_count} active nodes, "
                f"got {len(active_nodes)} ({active_nodes})"
            )
