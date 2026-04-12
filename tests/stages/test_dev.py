"""Tests for kindle.stages.dev — topological sort and parallel dev execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from kindle.stages.dev import _topological_sort, dev_node


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    """Create a minimal mock UI matching kindle.ui.UI interface."""
    return MagicMock()


def _make_task(
    task_id: str,
    title: str = "",
    dependencies: list[str] | None = None,
    description: str = "",
    directory_scope: str = ".",
    acceptance_criteria: list[str] | None = None,
) -> dict:
    """Build a dev-task dict matching the schema emitted by architect."""
    return {
        "task_id": task_id,
        "title": title or f"Task {task_id}",
        "dependencies": dependencies or [],
        "description": description,
        "directory_scope": directory_scope,
        "acceptance_criteria": acceptance_criteria or [],
    }


@dataclass
class _FakeAgentResult:
    """Mimics kindle.agent.AgentResult for mocking."""

    text: str = ""
    tool_calls: list = None
    raw_messages: list = None
    elapsed_seconds: float = 1.5
    turns_used: int = 3

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.raw_messages is None:
            self.raw_messages = []


def _make_state(
    dev_tasks: list[dict] | None = None,
    **overrides,
) -> dict:
    """Build a minimal KindleState dict for dev_node tests."""
    state = {
        "project_dir": "/tmp/kindle-test-project",
        "dev_tasks": dev_tasks or [],
        "max_concurrent_agents": 4,
        "feature_spec": {"name": "test-app"},
        "architecture": "simple arch doc",
    }
    state.update(overrides)
    return state


# ===========================================================================
# _topological_sort — pure algorithm tests
# ===========================================================================


class TestTopologicalSortSingleTask:
    """Single task with no dependencies produces exactly one layer."""

    def test_single_task_one_tier(self):
        tasks = [_make_task("a")]
        layers = _topological_sort(tasks)
        assert len(layers) == 1
        assert layers[0] == [tasks[0]]

    def test_single_task_preserves_task_dict(self):
        task = _make_task("x", title="Only task", description="do stuff")
        layers = _topological_sort([task])
        assert layers[0][0] is task  # same object, not a copy


class TestTopologicalSortLinearChain:
    """Linear dependency chain A→B→C should yield 3 separate tiers."""

    def test_three_tiers(self):
        a = _make_task("a")
        b = _make_task("b", dependencies=["a"])
        c = _make_task("c", dependencies=["b"])
        layers = _topological_sort([a, b, c])

        assert len(layers) == 3
        assert layers[0] == [a]
        assert layers[1] == [b]
        assert layers[2] == [c]

    def test_order_independent_of_input_order(self):
        """Even if tasks arrive in reverse, the layers respect deps."""
        c = _make_task("c", dependencies=["b"])
        b = _make_task("b", dependencies=["a"])
        a = _make_task("a")
        layers = _topological_sort([c, b, a])

        assert len(layers) == 3
        # Layer 0 must contain 'a' (no deps)
        assert layers[0] == [a]
        assert layers[1] == [b]
        assert layers[2] == [c]


class TestTopologicalSortIndependentTasks:
    """Tasks with no dependencies all land in a single tier."""

    def test_all_independent(self):
        tasks = [_make_task("x"), _make_task("y"), _make_task("z")]
        layers = _topological_sort(tasks)
        assert len(layers) == 1
        assert len(layers[0]) == 3

    def test_preserves_input_order_within_tier(self):
        tasks = [_make_task("c"), _make_task("a"), _make_task("b")]
        layers = _topological_sort(tasks)
        ids = [t["task_id"] for t in layers[0]]
        assert ids == ["c", "a", "b"]


class TestTopologicalSortDiamond:
    """Diamond: A → (B, C) → D. Two tasks in the middle tier."""

    def test_diamond_produces_three_layers(self):
        a = _make_task("a")
        b = _make_task("b", dependencies=["a"])
        c = _make_task("c", dependencies=["a"])
        d = _make_task("d", dependencies=["b", "c"])
        layers = _topological_sort([a, b, c, d])

        assert len(layers) == 3
        assert layers[0] == [a]
        # b and c should both be in layer 1 (order matches input)
        layer1_ids = {t["task_id"] for t in layers[1]}
        assert layer1_ids == {"b", "c"}
        assert layers[2] == [d]

    def test_diamond_middle_layer_count(self):
        a = _make_task("a")
        b = _make_task("b", dependencies=["a"])
        c = _make_task("c", dependencies=["a"])
        d = _make_task("d", dependencies=["b", "c"])
        layers = _topological_sort([a, b, c, d])
        assert len(layers[1]) == 2


class TestTopologicalSortMissingDeps:
    """Tasks referencing unknown dependencies — treated as unresolvable."""

    def test_missing_dep_causes_fallback(self):
        """If a dep is never satisfied, the algo forces the remaining tasks
        into a single layer (the circular-dependency fallback)."""
        a = _make_task("a")
        b = _make_task("b", dependencies=["nonexistent"])
        layers = _topological_sort([a, b])

        # Layer 0 = [a] (no deps).  Layer 1 = [b] via the fallback.
        assert len(layers) == 2
        assert layers[0] == [a]
        assert layers[1] == [b]

    def test_all_missing_deps_single_fallback_layer(self):
        """Multiple tasks all pointing to missing deps get forced together."""
        x = _make_task("x", dependencies=["ghost_1"])
        y = _make_task("y", dependencies=["ghost_2"])
        layers = _topological_sort([x, y])

        # Neither can resolve, so the first iter produces empty layer →
        # fallback dumps them all into one layer.
        assert len(layers) == 1
        assert len(layers[0]) == 2

    def test_partial_missing_dep(self):
        """One valid dep + one missing: task blocked until fallback."""
        a = _make_task("a")
        b = _make_task("b", dependencies=["a", "missing"])
        layers = _topological_sort([a, b])
        # a goes first, then b is forced via fallback
        assert len(layers) == 2


class TestTopologicalSortCircularDeps:
    """Circular dependencies trigger the fallback (force remaining tasks)."""

    def test_two_node_cycle(self):
        a = _make_task("a", dependencies=["b"])
        b = _make_task("b", dependencies=["a"])
        layers = _topological_sort([a, b])
        # Both are stuck → fallback layer
        assert len(layers) == 1
        assert len(layers[0]) == 2

    def test_cycle_with_independent_root(self):
        root = _make_task("root")
        a = _make_task("a", dependencies=["b"])
        b = _make_task("b", dependencies=["a"])
        layers = _topological_sort([root, a, b])
        assert len(layers) == 2
        assert layers[0] == [root]
        # a and b stuck in cycle → fallback
        assert len(layers[1]) == 2


class TestTopologicalSortEmptyInput:
    """Empty task list produces no layers."""

    def test_empty_list(self):
        assert _topological_sort([]) == []


class TestTopologicalSortDefaultTaskIds:
    """Tasks missing explicit task_id get auto-generated IDs (task_00, task_01)."""

    def test_auto_ids_respected_in_dependencies(self):
        """A task can depend on the auto-generated ID of another task."""
        t0 = {"title": "First"}  # auto-id = task_00
        t1 = {"title": "Second", "dependencies": ["task_00"]}  # auto-id = task_01
        layers = _topological_sort([t0, t1])
        assert len(layers) == 2
        assert layers[0] == [t0]
        assert layers[1] == [t1]


# ===========================================================================
# dev_node — async integration tests (run_agent is mocked)
# ===========================================================================


class TestDevNodeNoTasks:
    """dev_node with an empty task list skips gracefully."""

    @patch("kindle.stages.dev.mark_stage_complete")
    async def test_returns_current_stage(self, mock_mark):
        ui = _make_ui()
        state = _make_state(dev_tasks=[])
        result = await dev_node(state, ui)

        assert result == {"current_stage": "dev"}

    @patch("kindle.stages.dev.mark_stage_complete")
    async def test_calls_error_and_stage_lifecycle(self, mock_mark):
        ui = _make_ui()
        state = _make_state(dev_tasks=[])
        await dev_node(state, ui)

        ui.stage_start.assert_called_once_with("dev")
        ui.error.assert_called_once()
        ui.stage_done.assert_called_once_with("dev")
        mock_mark.assert_called_once_with("/tmp/kindle-test-project", "dev")


class TestDevNodeSingleTask:
    """dev_node with a single task runs it through run_agent."""

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_runs_one_task(self, mock_agent, mock_ws, mock_mark, mock_save):
        mock_agent.return_value = _FakeAgentResult(elapsed_seconds=2.0, turns_used=5)
        ui = _make_ui()
        task = _make_task("t1", title="Build API")
        state = _make_state(dev_tasks=[task])

        result = await dev_node(state, ui)

        assert result == {"current_stage": "dev"}
        mock_agent.assert_awaited_once()

        # Verify the agent was called with correct kwargs
        agent_call = mock_agent.call_args
        assert agent_call.kwargs["stage"] == "dev_t1"
        assert agent_call.kwargs["cwd"] == "/tmp/ws"
        assert "Build API" in agent_call.kwargs["user_prompt"]

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_task_result_saved_as_artifact(self, mock_agent, mock_ws, mock_mark, mock_save):
        mock_agent.return_value = _FakeAgentResult(elapsed_seconds=2.0, turns_used=5)
        ui = _make_ui()
        task = _make_task("t1", title="Build API")
        state = _make_state(dev_tasks=[task])

        await dev_node(state, ui)

        mock_save.assert_called_once()
        saved_name = mock_save.call_args[0][1]
        assert saved_name == "dev_results.json"

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_ui_lifecycle_calls(self, mock_agent, mock_ws, mock_mark, mock_save):
        mock_agent.return_value = _FakeAgentResult()
        ui = _make_ui()
        task = _make_task("t1")
        state = _make_state(dev_tasks=[task])

        await dev_node(state, ui)

        ui.stage_start.assert_called_once_with("dev")
        ui.task_start.assert_called_once_with("t1", "Task t1", 1, 1)
        ui.task_done.assert_called_once_with("t1")
        ui.stage_done.assert_called_once_with("dev")


class TestDevNodeParallelExecution:
    """dev_node runs independent tasks concurrently, bounded by semaphore."""

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_independent_tasks_same_layer(self, mock_agent, mock_ws, mock_mark, mock_save):
        """Three independent tasks should run in a single layer."""
        mock_agent.return_value = _FakeAgentResult()
        ui = _make_ui()
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        state = _make_state(dev_tasks=tasks)

        await dev_node(state, ui)

        # All 3 tasks should run
        assert mock_agent.await_count == 3
        # Single layer announced
        ui.info.assert_called_once()
        assert "1/1" in ui.info.call_args[0][0]

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_dependent_tasks_run_in_order(self, mock_agent, mock_ws, mock_mark, mock_save):
        """Chain A→B should produce 2 layers with separate info calls."""
        mock_agent.return_value = _FakeAgentResult()
        ui = _make_ui()
        tasks = [_make_task("a"), _make_task("b", dependencies=["a"])]
        state = _make_state(dev_tasks=tasks)

        await dev_node(state, ui)

        assert mock_agent.await_count == 2
        # Two layer announcements
        assert ui.info.call_count == 2

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_semaphore_limits_concurrency(self, mock_agent, mock_ws, mock_mark, mock_save):
        """Verify semaphore is respected via max_concurrent_agents setting."""
        concurrency_high_water = 0
        current_running = 0
        lock = asyncio.Lock()

        original_return = _FakeAgentResult()

        async def _tracked_agent(**kwargs):
            nonlocal concurrency_high_water, current_running
            async with lock:
                current_running += 1
                concurrency_high_water = max(concurrency_high_water, current_running)
            await asyncio.sleep(0.01)  # simulate work
            async with lock:
                current_running -= 1
            return original_return

        mock_agent.side_effect = _tracked_agent
        ui = _make_ui()

        # 5 independent tasks, but max_concurrent_agents=2
        tasks = [_make_task(f"t{i}") for i in range(5)]
        state = _make_state(dev_tasks=tasks, max_concurrent_agents=2)

        await dev_node(state, ui)

        assert mock_agent.await_count == 5
        assert concurrency_high_water <= 2


class TestDevNodeErrorHandling:
    """dev_node gracefully handles task failures."""

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_failed_task_recorded_as_error(self, mock_agent, mock_ws, mock_mark, mock_save):
        """An exception from run_agent is caught and recorded as a failed result."""
        mock_agent.side_effect = RuntimeError("agent crashed")
        ui = _make_ui()
        task = _make_task("t1")
        state = _make_state(dev_tasks=[task])

        result = await dev_node(state, ui)

        # Pipeline still completes
        assert result == {"current_stage": "dev"}
        ui.error.assert_called_once()
        assert "agent crashed" in ui.error.call_args[0][0]

        # dev_results.json is still saved with the failure info
        mock_save.assert_called_once()
        saved_json = mock_save.call_args[0][2]
        assert "failed" in saved_json

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_partial_failure_records_both(self, mock_agent, mock_ws, mock_mark, mock_save):
        """Mix of successful and failing tasks — both are recorded."""
        mock_agent.side_effect = [
            _FakeAgentResult(elapsed_seconds=1.0, turns_used=2),
            RuntimeError("boom"),
        ]
        ui = _make_ui()
        tasks = [_make_task("ok"), _make_task("fail")]
        state = _make_state(dev_tasks=tasks)

        result = await dev_node(state, ui)
        assert result == {"current_stage": "dev"}

        # One error UI call for the failed task
        ui.error.assert_called_once()

        # Artifact saved with both results
        saved_json = mock_save.call_args[0][2]
        assert "completed" in saved_json
        assert "failed" in saved_json

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_all_tasks_fail(self, mock_agent, mock_ws, mock_mark, mock_save):
        """Even if every task fails, the stage completes and marks done."""
        mock_agent.side_effect = RuntimeError("total failure")
        ui = _make_ui()
        tasks = [_make_task("a"), _make_task("b")]
        state = _make_state(dev_tasks=tasks)

        result = await dev_node(state, ui)

        assert result == {"current_stage": "dev"}
        mock_mark.assert_called_once_with("/tmp/kindle-test-project", "dev")
        mock_save.assert_called_once()
        assert ui.error.call_count == 2


class TestDevNodeResultAggregation:
    """Verify shape of result dicts passed to save_artifact."""

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_successful_result_shape(self, mock_agent, mock_ws, mock_mark, mock_save):
        mock_agent.return_value = _FakeAgentResult(elapsed_seconds=3.0, turns_used=10)
        ui = _make_ui()
        task = _make_task("api", title="Build REST API")
        state = _make_state(dev_tasks=[task])

        await dev_node(state, ui)

        import json

        saved_json = mock_save.call_args[0][2]
        results = json.loads(saved_json)
        assert len(results) == 1
        r = results[0]
        assert r["task_id"] == "api"
        assert r["title"] == "Build REST API"
        assert r["status"] == "completed"
        assert r["elapsed_seconds"] == 3.0
        assert r["turns_used"] == 10

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_failed_result_shape(self, mock_agent, mock_ws, mock_mark, mock_save):
        mock_agent.side_effect = ValueError("bad input")
        ui = _make_ui()
        task = _make_task("broken")
        state = _make_state(dev_tasks=[task])

        await dev_node(state, ui)

        import json

        saved_json = mock_save.call_args[0][2]
        results = json.loads(saved_json)
        assert len(results) == 1
        r = results[0]
        assert r["status"] == "failed"
        assert r["error"] == "bad input"


class TestDevNodeModelPassthrough:
    """Verify model and max_turns settings are forwarded to run_agent."""

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_model_forwarded(self, mock_agent, mock_ws, mock_mark, mock_save):
        mock_agent.return_value = _FakeAgentResult()
        ui = _make_ui()
        task = _make_task("t1")
        state = _make_state(dev_tasks=[task], model="claude-sonnet", max_agent_turns=25)

        await dev_node(state, ui)

        kwargs = mock_agent.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet"
        assert kwargs["max_turns"] == 25

    @patch("kindle.stages.dev.save_artifact")
    @patch("kindle.stages.dev.mark_stage_complete")
    @patch("kindle.stages.dev.workspace_path", return_value="/tmp/ws")
    @patch("kindle.stages.dev.run_agent", new_callable=AsyncMock)
    async def test_default_max_turns(self, mock_agent, mock_ws, mock_mark, mock_save):
        mock_agent.return_value = _FakeAgentResult()
        ui = _make_ui()
        task = _make_task("t1")
        # No max_agent_turns in state → defaults to 50
        state = _make_state(dev_tasks=[task])

        await dev_node(state, ui)

        kwargs = mock_agent.call_args.kwargs
        assert kwargs["max_turns"] == 50
