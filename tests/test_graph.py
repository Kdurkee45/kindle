"""Tests for kindle.graph — LangGraph state machine orchestrating the 6-stage pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph

from kindle.graph import (
    _NODE_FACTORIES,
    _SIMPLE_EDGES,
    ORDERED_STAGES,
    _wire_edges,
    _wrap,
    build_graph,
)
from kindle.stages.qa import qa_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    """Return a mock UI instance with attributes graph.py expects."""
    return MagicMock()


def _mock_all_nodes() -> dict:
    """Return a dict of patches replacing every stage node with an AsyncMock."""
    return {
        "kindle.graph.grill_node": AsyncMock(return_value={}),
        "kindle.graph.research_node": AsyncMock(return_value={}),
        "kindle.graph.architect_node": AsyncMock(return_value={}),
        "kindle.graph.dev_node": AsyncMock(return_value={}),
        "kindle.graph.qa_node": AsyncMock(return_value={}),
        "kindle.graph.package_node": AsyncMock(return_value={}),
        "kindle.graph.qa_router": lambda state: "package",
    }


def _build_with_mocks(entry_stage: str = "grill") -> CompiledStateGraph:
    """Build the graph with all stage nodes mocked out."""
    ui = _make_ui()
    mocks = _mock_all_nodes()
    with (
        patch("kindle.graph.grill_node", mocks["kindle.graph.grill_node"]),
        patch("kindle.graph.research_node", mocks["kindle.graph.research_node"]),
        patch("kindle.graph.architect_node", mocks["kindle.graph.architect_node"]),
        patch("kindle.graph.dev_node", mocks["kindle.graph.dev_node"]),
        patch("kindle.graph.qa_node", mocks["kindle.graph.qa_node"]),
        patch("kindle.graph.package_node", mocks["kindle.graph.package_node"]),
        patch("kindle.graph.qa_router", mocks["kindle.graph.qa_router"]),
    ):
        return build_graph(ui, entry_stage=entry_stage)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestOrderedStages:
    """Verify the canonical stage ordering."""

    def test_contains_all_six_stages(self) -> None:
        assert ORDERED_STAGES == ["grill", "research", "architect", "dev", "qa", "package"]

    def test_node_factories_match_ordered_stages(self) -> None:
        """Every ordered stage has a corresponding node factory."""
        assert set(_NODE_FACTORIES.keys()) == set(ORDERED_STAGES)


# ---------------------------------------------------------------------------
# build_graph() — returns a compiled StateGraph
# ---------------------------------------------------------------------------


class TestBuildGraphReturnType:
    """build_graph() must return a compiled LangGraph."""

    def test_returns_compiled_state_graph(self) -> None:
        graph = _build_with_mocks()
        assert isinstance(graph, CompiledStateGraph)

    def test_returns_compiled_graph_with_qa_entry(self) -> None:
        graph = _build_with_mocks(entry_stage="qa")
        assert isinstance(graph, CompiledStateGraph)


# ---------------------------------------------------------------------------
# build_graph() — default entry_stage='grill' wires all 6 stages
# ---------------------------------------------------------------------------


class TestBuildGraphAllStages:
    """With default entry_stage='grill', all 6 stage nodes are present."""

    def test_all_six_nodes_present(self) -> None:
        """The compiled graph should contain nodes for all 6 stages."""
        graph = _build_with_mocks(entry_stage="grill")
        node_names = set(graph.nodes.keys())
        # LangGraph adds __start__ and __end__ nodes internally
        for stage in ORDERED_STAGES:
            assert stage in node_names, f"Missing node: {stage}"

    def test_grill_is_first_node_after_start(self) -> None:
        """START should connect to 'grill' when entry_stage='grill'."""
        graph = _build_with_mocks(entry_stage="grill")
        # The compiled graph's get_graph() gives the graph structure
        drawable = graph.get_graph()
        start_edges = [e for e in drawable.edges if e.source == "__start__"]
        targets = {e.target for e in start_edges}
        assert "grill" in targets


# ---------------------------------------------------------------------------
# build_graph() — entry_stage skips earlier stages
# ---------------------------------------------------------------------------


class TestBuildGraphEntryStage:
    """entry_stage controls which stages are active."""

    @pytest.mark.parametrize(
        ("entry_stage", "expected_present", "expected_absent"),
        [
            ("qa", {"qa", "package"}, {"grill", "research", "architect", "dev"}),
            ("dev", {"dev", "qa", "package"}, {"grill", "research", "architect"}),
            ("package", {"package"}, {"grill", "research", "architect", "dev", "qa"}),
            ("architect", {"architect", "dev", "qa", "package"}, {"grill", "research"}),
            ("research", {"research", "architect", "dev", "qa", "package"}, {"grill"}),
        ],
    )
    def test_entry_stage_filters_nodes(
        self,
        entry_stage: str,
        expected_present: set[str],
        expected_absent: set[str],
    ) -> None:
        graph = _build_with_mocks(entry_stage=entry_stage)
        node_names = set(graph.nodes.keys())
        for name in expected_present:
            assert name in node_names, f"Expected node '{name}' missing for entry_stage='{entry_stage}'"
        for name in expected_absent:
            assert name not in node_names, f"Unexpected node '{name}' present for entry_stage='{entry_stage}'"

    def test_entry_stage_qa_start_edge(self) -> None:
        """When entry_stage='qa', START should connect directly to 'qa'."""
        graph = _build_with_mocks(entry_stage="qa")
        drawable = graph.get_graph()
        start_edges = [e for e in drawable.edges if e.source == "__start__"]
        targets = {e.target for e in start_edges}
        assert "qa" in targets
        assert "grill" not in targets

    def test_entry_stage_package_has_only_package_node(self) -> None:
        """When entry_stage='package', only the package node is active."""
        graph = _build_with_mocks(entry_stage="package")
        node_names = set(graph.nodes.keys()) - {"__start__", "__end__"}
        assert node_names == {"package"}


# ---------------------------------------------------------------------------
# build_graph() — invalid entry_stage falls back to 'grill'
# ---------------------------------------------------------------------------


class TestBuildGraphInvalidEntryStage:
    """Invalid entry_stage values should fall back to 'grill'."""

    @pytest.mark.parametrize("bad_stage", ["invalid", "", "GRILL", "deploy", "test"])
    def test_invalid_entry_stage_falls_back_to_grill(self, bad_stage: str) -> None:
        graph = _build_with_mocks(entry_stage=bad_stage)
        node_names = set(graph.nodes.keys())
        # All 6 stages should be present (fell back to grill)
        for stage in ORDERED_STAGES:
            assert stage in node_names, f"Expected node '{stage}' after fallback for entry_stage='{bad_stage}'"

    def test_invalid_entry_stage_start_connects_to_grill(self) -> None:
        graph = _build_with_mocks(entry_stage="nonexistent")
        drawable = graph.get_graph()
        start_edges = [e for e in drawable.edges if e.source == "__start__"]
        targets = {e.target for e in start_edges}
        assert "grill" in targets


# ---------------------------------------------------------------------------
# _wrap() — injects UI into node functions
# ---------------------------------------------------------------------------


class TestWrap:
    """_wrap() should produce a wrapper that injects the UI instance."""

    @pytest.mark.asyncio
    async def test_wrapper_passes_ui_to_inner_function(self) -> None:
        """The wrapped function should call the original with (state, ui)."""
        inner = AsyncMock(return_value={"result": "ok"})
        ui = _make_ui()

        wrapped = _wrap(inner, ui)
        state = {"idea": "test"}
        result = await wrapped(state)

        inner.assert_awaited_once_with(state, ui)
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_wrapper_preserves_function_name(self) -> None:
        """functools.wraps should preserve the original function's name."""

        async def my_node(state, ui):
            return {}

        ui = _make_ui()
        wrapped = _wrap(my_node, ui)
        assert wrapped.__name__ == "my_node"

    @pytest.mark.asyncio
    async def test_wrapper_returns_inner_result(self) -> None:
        """The return value should be forwarded from the inner function."""
        inner = AsyncMock(return_value={"qa_passed": True, "qa_report": "all good"})
        ui = _make_ui()

        wrapped = _wrap(inner, ui)
        result = await wrapped({})
        assert result == {"qa_passed": True, "qa_report": "all good"}

    @pytest.mark.asyncio
    async def test_wrapper_propagates_exception(self) -> None:
        """If the inner function raises, the wrapper should propagate it."""
        inner = AsyncMock(side_effect=RuntimeError("stage failed"))
        ui = _make_ui()

        wrapped = _wrap(inner, ui)
        with pytest.raises(RuntimeError, match="stage failed"):
            await wrapped({})

    @pytest.mark.asyncio
    async def test_different_ui_instances_are_isolated(self) -> None:
        """Two wraps with different UIs should inject their respective UI."""
        inner = AsyncMock(return_value={})
        ui1 = MagicMock(name="ui1")
        ui2 = MagicMock(name="ui2")

        wrapped1 = _wrap(inner, ui1)
        wrapped2 = _wrap(inner, ui2)

        await wrapped1({"a": 1})
        await wrapped2({"b": 2})

        calls = inner.await_args_list
        assert calls[0].args == ({"a": 1}, ui1)
        assert calls[1].args == ({"b": 2}, ui2)


# ---------------------------------------------------------------------------
# _wire_edges() — creates correct edges for active stage subsets
# ---------------------------------------------------------------------------


class TestWireEdges:
    """_wire_edges() must add the right edges for the active set of stages."""

    def test_all_stages_active_adds_all_simple_edges(self) -> None:
        """With all stages active, every simple edge should be wired."""
        from langgraph.graph import StateGraph

        from kindle.state import KindleState

        graph = StateGraph(KindleState)
        for stage in ORDERED_STAGES:
            graph.add_node(stage, AsyncMock())

        active = set(ORDERED_STAGES)

        with patch("kindle.graph.qa_router", lambda s: "package"):
            _wire_edges(graph, active)

        # Verify the graph can compile with START edge (needed for a valid graph)
        graph.add_edge("__start__", "grill")
        compiled = graph.compile()
        drawable = compiled.get_graph()
        edge_pairs = {(e.source, e.target) for e in drawable.edges}

        # Check key simple edges exist
        assert ("grill", "research") in edge_pairs
        assert ("research", "architect") in edge_pairs
        assert ("architect", "dev") in edge_pairs
        assert ("dev", "qa") in edge_pairs
        # package -> END
        assert ("package", "__end__") in edge_pairs

    def test_qa_package_active_adds_conditional_edge(self) -> None:
        """When qa and package are active, conditional routing should exist."""
        from langgraph.graph import StateGraph

        from kindle.state import KindleState

        graph = StateGraph(KindleState)
        graph.add_node("qa", AsyncMock())
        graph.add_node("package", AsyncMock())

        with patch("kindle.graph.qa_router", lambda s: "package"):
            _wire_edges(graph, {"qa", "package"})

        graph.add_edge("__start__", "qa")
        compiled = graph.compile()
        drawable = compiled.get_graph()
        edge_pairs = {(e.source, e.target) for e in drawable.edges}

        # QA should connect to package via conditional edge
        assert ("qa", "package") in edge_pairs

    def test_only_package_active_no_qa_conditional(self) -> None:
        """When only 'package' is active, no QA conditional edge is added."""
        from langgraph.graph import StateGraph

        from kindle.state import KindleState

        graph = StateGraph(KindleState)
        graph.add_node("package", AsyncMock())
        _wire_edges(graph, {"package"})

        # If we had a QA edge, compilation would fail because there's no qa node
        graph.add_edge("__start__", "package")
        compiled = graph.compile()
        drawable = compiled.get_graph()
        edge_pairs = {(e.source, e.target) for e in drawable.edges}

        assert ("package", "__end__") in edge_pairs
        assert not any(src == "qa" for src, _ in edge_pairs)

    def test_partial_stages_skips_irrelevant_edges(self) -> None:
        """Edges whose required stages aren't all active get skipped."""
        from langgraph.graph import StateGraph

        from kindle.state import KindleState

        # Only dev, qa, package active
        graph = StateGraph(KindleState)
        for stage in ["dev", "qa", "package"]:
            graph.add_node(stage, AsyncMock())

        with patch("kindle.graph.qa_router", lambda s: "package"):
            _wire_edges(graph, {"dev", "qa", "package"})

        graph.add_edge("__start__", "dev")
        compiled = graph.compile()
        drawable = compiled.get_graph()
        edge_pairs = {(e.source, e.target) for e in drawable.edges}

        assert ("dev", "qa") in edge_pairs
        assert ("package", "__end__") in edge_pairs
        # grill -> research should NOT exist
        assert ("grill", "research") not in edge_pairs
        assert ("research", "architect") not in edge_pairs


# ---------------------------------------------------------------------------
# qa_router() — conditional routing logic
# ---------------------------------------------------------------------------


class TestQaRouter:
    """qa_router returns 'package' on pass and 'qa' (self-heal) on fail."""

    def test_both_passed_returns_package(self) -> None:
        state = {"qa_passed": True, "cpo_passed": True}
        assert qa_router(state) == "package"

    def test_qa_failed_returns_qa(self) -> None:
        state = {"qa_passed": False, "cpo_passed": True, "qa_retries": 0, "max_qa_retries": 10}
        assert qa_router(state) == "qa"

    def test_cpo_failed_returns_qa(self) -> None:
        state = {"qa_passed": True, "cpo_passed": False, "cpo_retries": 0, "max_cpo_retries": 10}
        assert qa_router(state) == "qa"

    def test_both_failed_returns_qa(self) -> None:
        state = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 0,
            "cpo_retries": 0,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }
        assert qa_router(state) == "qa"

    def test_qa_retries_exhausted_returns_package(self) -> None:
        """When QA retries exceed the max, give up and proceed to package."""
        state = {"qa_passed": False, "cpo_passed": True, "qa_retries": 10, "max_qa_retries": 10}
        assert qa_router(state) == "package"

    def test_cpo_retries_exhausted_returns_package(self) -> None:
        """When CPO retries exceed the max, give up and proceed to package."""
        state = {"qa_passed": True, "cpo_passed": False, "cpo_retries": 10, "max_cpo_retries": 10}
        assert qa_router(state) == "package"

    def test_defaults_when_state_is_empty(self) -> None:
        """With an empty state, both default to not-passed and retry=0, triggering self-heal."""
        state = {}
        assert qa_router(state) == "qa"

    def test_qa_passed_false_at_max_retries(self) -> None:
        """Edge: exactly at max retries should stop retrying."""
        state = {"qa_passed": False, "cpo_passed": False, "qa_retries": 5, "max_qa_retries": 5}
        assert qa_router(state) == "package"


# ---------------------------------------------------------------------------
# _SIMPLE_EDGES constant structure
# ---------------------------------------------------------------------------


class TestSimpleEdgesConstant:
    """Verify the _SIMPLE_EDGES constant is well-formed."""

    def test_all_edges_reference_valid_stages_or_end(self) -> None:
        from langgraph.graph import END

        valid_names = set(ORDERED_STAGES) | {END}
        for src, dst, _required in _SIMPLE_EDGES:
            assert src in valid_names, f"Invalid source: {src}"
            assert dst in valid_names, f"Invalid destination: {dst}"

    def test_required_sets_are_subsets_of_ordered_stages(self) -> None:
        all_stages = set(ORDERED_STAGES)
        for _, _, required in _SIMPLE_EDGES:
            assert required <= all_stages, f"Required set {required} has invalid stages"


# ---------------------------------------------------------------------------
# _NODE_FACTORIES — each factory produces (name, fn) tuples
# ---------------------------------------------------------------------------


class TestNodeFactories:
    """Each _NODE_FACTORIES entry should return a list of (name, fn) tuples."""

    @pytest.mark.parametrize("stage", ORDERED_STAGES)
    def test_factory_returns_list_of_name_fn_pairs(self, stage: str) -> None:
        ui = _make_ui()
        pairs = _NODE_FACTORIES[stage](ui)
        assert isinstance(pairs, list)
        assert len(pairs) >= 1
        for name, fn in pairs:
            assert isinstance(name, str)
            assert callable(fn)

    @pytest.mark.parametrize("stage", ORDERED_STAGES)
    def test_factory_name_matches_stage(self, stage: str) -> None:
        """The node name from the factory should match the stage name."""
        ui = _make_ui()
        pairs = _NODE_FACTORIES[stage](ui)
        names = [name for name, _ in pairs]
        assert stage in names


# ---------------------------------------------------------------------------
# Integration: build_graph end-to-end structure validation
# ---------------------------------------------------------------------------


class TestBuildGraphIntegration:
    """Integration tests verifying the full graph structure."""

    def test_graph_has_self_heal_loop_for_qa(self) -> None:
        """The QA node should have an edge back to itself (self-heal loop)."""
        graph = _build_with_mocks(entry_stage="grill")
        drawable = graph.get_graph()
        edge_pairs = {(e.source, e.target) for e in drawable.edges}
        # QA self-heal: qa -> qa
        assert ("qa", "qa") in edge_pairs

    def test_graph_terminates_at_end_after_package(self) -> None:
        """The package node should lead to __end__."""
        graph = _build_with_mocks(entry_stage="grill")
        drawable = graph.get_graph()
        edge_pairs = {(e.source, e.target) for e in drawable.edges}
        assert ("package", "__end__") in edge_pairs

    def test_linear_path_exists_grill_to_package(self) -> None:
        """The happy path grill -> research -> architect -> dev -> qa -> package exists."""
        graph = _build_with_mocks(entry_stage="grill")
        drawable = graph.get_graph()
        edge_pairs = {(e.source, e.target) for e in drawable.edges}

        expected_path = [
            ("__start__", "grill"),
            ("grill", "research"),
            ("research", "architect"),
            ("architect", "dev"),
            ("dev", "qa"),
            ("qa", "package"),
            ("package", "__end__"),
        ]
        for edge in expected_path:
            assert edge in edge_pairs, f"Missing edge in happy path: {edge}"

    def test_entry_stage_dev_has_correct_path(self) -> None:
        """Starting at 'dev' should have dev -> qa -> package -> __end__."""
        graph = _build_with_mocks(entry_stage="dev")
        drawable = graph.get_graph()
        edge_pairs = {(e.source, e.target) for e in drawable.edges}

        assert ("__start__", "dev") in edge_pairs
        assert ("dev", "qa") in edge_pairs
        assert ("qa", "package") in edge_pairs
        assert ("package", "__end__") in edge_pairs
        # Earlier stages should NOT be present
        assert ("grill", "research") not in edge_pairs
