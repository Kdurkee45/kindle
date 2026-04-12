"""Tests for kindle.stages.dev — parallel build with multiple agents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.stages.dev import SYSTEM_PROMPT, _run_task, _topological_sort, dev_node

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_FEATURE_SPEC = {
    "app_name": "TaskFlow",
    "idea": "a task management app",
    "core_features": ["task CRUD", "auth"],
    "tech_constraints": ["React frontend"],
}

SAMPLE_ARCHITECTURE = """\
# Architecture

## Tech Stack
- React 18.x frontend
- Express 4.x backend
"""

SAMPLE_DEV_TASKS: list[dict] = [
    {
        "task_id": "task_01",
        "title": "Set up project structure",
        "description": "Scaffold the project.",
        "directory_scope": ".",
        "dependencies": [],
        "acceptance_criteria": ["pyproject.toml exists"],
    },
    {
        "task_id": "task_02",
        "title": "Build data models",
        "description": "Create models.",
        "directory_scope": "src/models/",
        "dependencies": ["task_01"],
        "acceptance_criteria": ["All models defined", "Tests pass"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal KindleState dict pointing at *tmp_path* as project_dir."""
    project_dir = tmp_path / "project"
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)
    (project_dir / "workspace").mkdir(parents=True, exist_ok=True)
    # metadata.json needed by mark_stage_complete
    meta = {"project_id": "kindle_test1234", "stages_completed": []}
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    state: dict = {
        "project_dir": str(project_dir),
        "idea": "a task management app",
        "feature_spec": SAMPLE_FEATURE_SPEC,
        "architecture": SAMPLE_ARCHITECTURE,
        "dev_tasks": SAMPLE_DEV_TASKS,
        "max_concurrent_agents": 4,
    }
    state.update(overrides)
    return state


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods dev_node actually calls."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.task_start = MagicMock()
    ui.task_done = MagicMock()
    return ui


def _make_agent_result(elapsed: float = 1.5, turns: int = 3) -> MagicMock:
    """Return a mock AgentResult."""
    result = MagicMock()
    result.elapsed_seconds = elapsed
    result.turns_used = turns
    return result


# ---------------------------------------------------------------------------
# _topological_sort — pure function tests
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Tests for the dependency-based topological sort."""

    def test_single_task_no_deps(self) -> None:
        """A single task with no dependencies produces one layer."""
        tasks = [{"task_id": "t1", "dependencies": []}]
        layers = _topological_sort(tasks)
        assert len(layers) == 1
        assert layers[0] == [tasks[0]]

    def test_independent_tasks_same_layer(self) -> None:
        """Multiple tasks with no dependencies all go in layer 0."""
        tasks = [
            {"task_id": "a", "dependencies": []},
            {"task_id": "b", "dependencies": []},
            {"task_id": "c", "dependencies": []},
        ]
        layers = _topological_sort(tasks)
        assert len(layers) == 1
        assert len(layers[0]) == 3
        ids = {t["task_id"] for t in layers[0]}
        assert ids == {"a", "b", "c"}

    def test_linear_chain(self) -> None:
        """A -> B -> C produces three sequential layers."""
        tasks = [
            {"task_id": "A", "dependencies": []},
            {"task_id": "B", "dependencies": ["A"]},
            {"task_id": "C", "dependencies": ["B"]},
        ]
        layers = _topological_sort(tasks)
        assert len(layers) == 3
        assert layers[0][0]["task_id"] == "A"
        assert layers[1][0]["task_id"] == "B"
        assert layers[2][0]["task_id"] == "C"

    def test_diamond_dependency(self) -> None:
        """Diamond: A -> B, A -> C, B+C -> D produces three layers."""
        tasks = [
            {"task_id": "A", "dependencies": []},
            {"task_id": "B", "dependencies": ["A"]},
            {"task_id": "C", "dependencies": ["A"]},
            {"task_id": "D", "dependencies": ["B", "C"]},
        ]
        layers = _topological_sort(tasks)
        assert len(layers) == 3
        # Layer 0: A
        assert [t["task_id"] for t in layers[0]] == ["A"]
        # Layer 1: B and C (parallel)
        layer1_ids = {t["task_id"] for t in layers[1]}
        assert layer1_ids == {"B", "C"}
        # Layer 2: D
        assert [t["task_id"] for t in layers[2]] == ["D"]

    def test_circular_dependency_fallback(self) -> None:
        """Circular deps should not hang — all stuck tasks dumped into one layer."""
        tasks = [
            {"task_id": "X", "dependencies": ["Y"]},
            {"task_id": "Y", "dependencies": ["X"]},
        ]
        layers = _topological_sort(tasks)
        # Both tasks should end up in a single fallback layer
        assert len(layers) == 1
        ids = {t["task_id"] for t in layers[0]}
        assert ids == {"X", "Y"}

    def test_partial_circular_with_independent_root(self) -> None:
        """A has no deps (resolves first), then B<->C are circular."""
        tasks = [
            {"task_id": "A", "dependencies": []},
            {"task_id": "B", "dependencies": ["A", "C"]},
            {"task_id": "C", "dependencies": ["A", "B"]},
        ]
        layers = _topological_sort(tasks)
        # Layer 0: A (no deps)
        assert layers[0][0]["task_id"] == "A"
        # Layer 1: B and C fall through circular fallback
        assert len(layers) == 2
        layer1_ids = {t["task_id"] for t in layers[1]}
        assert layer1_ids == {"B", "C"}

    def test_empty_task_list(self) -> None:
        """Empty input produces no layers."""
        layers = _topological_sort([])
        assert layers == []

    def test_missing_task_id_uses_index_fallback(self) -> None:
        """Tasks without explicit task_id get an auto-generated ID."""
        tasks = [
            {"title": "First", "dependencies": []},
            {"title": "Second", "dependencies": ["task_00"]},
        ]
        layers = _topological_sort(tasks)
        # First task (task_00) has no deps -> layer 0
        # Second task depends on task_00 -> layer 1
        assert len(layers) == 2

    def test_wide_fan_out(self) -> None:
        """One root task with many dependents fans out to two layers."""
        root = {"task_id": "root", "dependencies": []}
        children = [{"task_id": f"child_{i}", "dependencies": ["root"]} for i in range(10)]
        layers = _topological_sort([root, *children])
        assert len(layers) == 2
        assert layers[0][0]["task_id"] == "root"
        assert len(layers[1]) == 10


# ---------------------------------------------------------------------------
# _run_task — single task execution
# ---------------------------------------------------------------------------


class TestRunTask:
    """Tests for the _run_task coroutine."""

    @pytest.mark.asyncio
    async def test_run_task_returns_correct_result(self, tmp_path: Path) -> None:
        """_run_task should return a dict with task metadata and agent stats."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        task = SAMPLE_DEV_TASKS[0]
        semaphore = asyncio.Semaphore(4)

        mock_result = _make_agent_result(elapsed=2.5, turns=7)
        mock_agent = AsyncMock(return_value=mock_result)

        with patch("kindle.stages.dev.run_agent", mock_agent):
            result = await _run_task(task, state, ui, semaphore, 1, 2)

        assert result["task_id"] == "task_01"
        assert result["title"] == "Set up project structure"
        assert result["status"] == "completed"
        assert result["elapsed_seconds"] == 2.5
        assert result["turns_used"] == 7

    @pytest.mark.asyncio
    async def test_run_task_calls_agent_with_correct_args(self, tmp_path: Path) -> None:
        """Verify run_agent receives proper persona, system prompt, and user prompt."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        task = SAMPLE_DEV_TASKS[0]
        semaphore = asyncio.Semaphore(2)

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await _run_task(task, state, ui, semaphore, 1, 1)

        mock_agent.assert_called_once()
        kwargs = mock_agent.call_args.kwargs
        assert kwargs["system_prompt"] == SYSTEM_PROMPT
        assert kwargs["persona"] == "Principal Software Engineer (task_01)"
        assert kwargs["stage"] == "dev_task_01"
        assert "task_01" in kwargs["user_prompt"]
        assert "Set up project structure" in kwargs["user_prompt"]
        assert "Scaffold the project." in kwargs["user_prompt"]
        assert "pyproject.toml exists" in kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_run_task_includes_feature_spec_and_architecture(self, tmp_path: Path) -> None:
        """The user prompt should contain the feature spec and architecture from state."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        task = SAMPLE_DEV_TASKS[0]
        semaphore = asyncio.Semaphore(1)

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await _run_task(task, state, ui, semaphore, 1, 1)

        prompt = mock_agent.call_args.kwargs["user_prompt"]
        assert "TaskFlow" in prompt  # from feature_spec
        assert "React 18.x frontend" in prompt  # from architecture

    @pytest.mark.asyncio
    async def test_run_task_ui_lifecycle(self, tmp_path: Path) -> None:
        """_run_task should call task_start and task_done on the UI."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        task = SAMPLE_DEV_TASKS[0]
        semaphore = asyncio.Semaphore(4)

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await _run_task(task, state, ui, semaphore, 1, 2)

        ui.task_start.assert_called_once_with("task_01", "Set up project structure", 1, 2)
        ui.task_done.assert_called_once_with("task_01")

    @pytest.mark.asyncio
    async def test_run_task_uses_default_task_id(self, tmp_path: Path) -> None:
        """Tasks without task_id fall back to index-based ID."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        task = {"title": "No ID task", "description": "Test", "dependencies": []}
        semaphore = asyncio.Semaphore(4)

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            result = await _run_task(task, state, ui, semaphore, 3, 5)

        assert result["task_id"] == "task_03"

    @pytest.mark.asyncio
    async def test_run_task_passes_model_and_max_turns(self, tmp_path: Path) -> None:
        """State model and max_agent_turns flow through to run_agent."""
        state = _make_state(tmp_path, model="opus", max_agent_turns=25)
        ui = _make_ui()
        task = SAMPLE_DEV_TASKS[0]
        semaphore = asyncio.Semaphore(4)

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await _run_task(task, state, ui, semaphore, 1, 1)

        kwargs = mock_agent.call_args.kwargs
        assert kwargs["model"] == "opus"
        assert kwargs["max_turns"] == 25


# ---------------------------------------------------------------------------
# dev_node — integration tests
# ---------------------------------------------------------------------------


class TestDevNodeHappyPath:
    """Tests for the dev_node LangGraph entry point."""

    @pytest.mark.asyncio
    async def test_dev_node_returns_current_stage(self, tmp_path: Path) -> None:
        """dev_node should return {'current_stage': 'dev'}."""
        state = _make_state(tmp_path)
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            result = await dev_node(state, ui)

        assert result == {"current_stage": "dev"}

    @pytest.mark.asyncio
    async def test_dev_node_saves_results_artifact(self, tmp_path: Path) -> None:
        """dev_results.json should be written with one entry per task."""
        state = _make_state(tmp_path)
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result(elapsed=1.0, turns=2))

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        artifact_path = Path(state["project_dir"]) / "artifacts" / "dev_results.json"
        assert artifact_path.exists()
        results = json.loads(artifact_path.read_text())
        assert len(results) == 2
        assert results[0]["task_id"] == "task_01"
        assert results[0]["status"] == "completed"
        assert results[1]["task_id"] == "task_02"
        assert results[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_dev_node_marks_stage_complete(self, tmp_path: Path) -> None:
        """dev should appear in stages_completed after successful run."""
        state = _make_state(tmp_path)
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "dev" in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_dev_node_ui_lifecycle(self, tmp_path: Path) -> None:
        """stage_start and stage_done should be called."""
        state = _make_state(tmp_path)
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        ui.stage_start.assert_called_once_with("dev")
        ui.stage_done.assert_called_once_with("dev")

    @pytest.mark.asyncio
    async def test_dev_node_respects_dependency_ordering(self, tmp_path: Path) -> None:
        """task_02 depends on task_01; verify agent is called for task_01 first."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        call_order: list[str] = []

        async def fake_run_agent(**kwargs):
            # Extract task_id from the stage parameter (dev_task_01, dev_task_02)
            stage = kwargs.get("stage", "")
            call_order.append(stage)
            return _make_agent_result()

        mock_agent = AsyncMock(side_effect=fake_run_agent)

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        # task_01 must be called before task_02
        assert call_order.index("dev_task_01") < call_order.index("dev_task_02")

    @pytest.mark.asyncio
    async def test_dev_node_layer_info_messages(self, tmp_path: Path) -> None:
        """UI should report layer progress messages."""
        state = _make_state(tmp_path)
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        # With SAMPLE_DEV_TASKS: layer 1 has task_01, layer 2 has task_02
        ui.info.assert_any_call("Layer 1/2 — 1 task(s) in parallel")
        ui.info.assert_any_call("Layer 2/2 — 1 task(s) in parallel")


class TestDevNodeNoTasks:
    """Tests for when dev_tasks is empty."""

    @pytest.mark.asyncio
    async def test_empty_tasks_skips_gracefully(self, tmp_path: Path) -> None:
        """No dev tasks should skip agent calls and log an error."""
        state = _make_state(tmp_path, dev_tasks=[])
        ui = _make_ui()

        result = await dev_node(state, ui)

        assert result == {"current_stage": "dev"}
        ui.error.assert_called_once_with("No dev tasks found — skipping dev stage.")

    @pytest.mark.asyncio
    async def test_empty_tasks_still_marks_complete(self, tmp_path: Path) -> None:
        """Even with no tasks, stage should be marked complete."""
        state = _make_state(tmp_path, dev_tasks=[])
        ui = _make_ui()

        await dev_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "dev" in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_empty_tasks_calls_stage_lifecycle(self, tmp_path: Path) -> None:
        """stage_start and stage_done are still called when skipping."""
        state = _make_state(tmp_path, dev_tasks=[])
        ui = _make_ui()

        await dev_node(state, ui)

        ui.stage_start.assert_called_once_with("dev")
        ui.stage_done.assert_called_once_with("dev")


class TestDevNodeErrorHandling:
    """Tests for error handling in dev_node."""

    @pytest.mark.asyncio
    async def test_exception_captured_in_results(self, tmp_path: Path) -> None:
        """If run_agent throws, the error is captured — not re-raised."""
        single_task = [
            {"task_id": "fail_task", "title": "Boom", "dependencies": [], "description": "explodes"},
        ]
        state = _make_state(tmp_path, dev_tasks=single_task)
        ui = _make_ui()

        mock_agent = AsyncMock(side_effect=RuntimeError("agent crashed"))

        with patch("kindle.stages.dev.run_agent", mock_agent):
            result = await dev_node(state, ui)

        # Should not raise, should return normally
        assert result == {"current_stage": "dev"}
        ui.error.assert_called_once()
        assert "agent crashed" in ui.error.call_args[0][0]

        # The artifact should contain the failed result
        artifact_path = Path(state["project_dir"]) / "artifacts" / "dev_results.json"
        results = json.loads(artifact_path.read_text())
        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert results[0]["error"] == "agent crashed"

    @pytest.mark.asyncio
    async def test_partial_failure_other_tasks_still_complete(self, tmp_path: Path) -> None:
        """If one task fails in a layer, others in the same layer still succeed."""
        tasks = [
            {"task_id": "ok_task", "title": "Works", "dependencies": [], "description": "fine"},
            {"task_id": "bad_task", "title": "Fails", "dependencies": [], "description": "broken"},
        ]
        state = _make_state(tmp_path, dev_tasks=tasks)
        ui = _make_ui()

        async def selective_agent(**kwargs):
            if "bad_task" in kwargs.get("stage", ""):
                raise ValueError("selective failure")
            return _make_agent_result()

        mock_agent = AsyncMock(side_effect=selective_agent)

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        artifact_path = Path(state["project_dir"]) / "artifacts" / "dev_results.json"
        results = json.loads(artifact_path.read_text())
        assert len(results) == 2

        statuses = {r.get("task_id", r.get("error", "")): r["status"] for r in results}
        # One completed, one failed (order may vary since they run in parallel)
        assert "completed" in statuses.values()
        assert "failed" in statuses.values()

    @pytest.mark.asyncio
    async def test_failure_in_later_layer_after_success(self, tmp_path: Path) -> None:
        """First layer succeeds, second layer fails — both captured."""
        tasks = [
            {"task_id": "t1", "title": "OK", "dependencies": [], "description": "fine"},
            {"task_id": "t2", "title": "Boom", "dependencies": ["t1"], "description": "fails"},
        ]
        state = _make_state(tmp_path, dev_tasks=tasks)
        ui = _make_ui()

        call_count = 0

        async def agent_fails_on_second(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("layer 2 failure")
            return _make_agent_result()

        mock_agent = AsyncMock(side_effect=agent_fails_on_second)

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        artifact_path = Path(state["project_dir"]) / "artifacts" / "dev_results.json"
        results = json.loads(artifact_path.read_text())
        assert len(results) == 2
        assert results[0]["status"] == "completed"
        assert results[1]["status"] == "failed"


# ---------------------------------------------------------------------------
# Concurrency — semaphore limiting
# ---------------------------------------------------------------------------


class TestDevNodeConcurrency:
    """Verify that asyncio.Semaphore properly limits concurrent agent calls."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, tmp_path: Path) -> None:
        """With max_concurrent_agents=2 and 4 independent tasks, at most 2 run at once."""
        tasks = [
            {"task_id": f"t{i}", "title": f"Task {i}", "dependencies": [], "description": f"desc {i}"} for i in range(4)
        ]
        state = _make_state(tmp_path, dev_tasks=tasks, max_concurrent_agents=2)
        ui = _make_ui()

        max_concurrent_seen = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def counting_agent(**kwargs):
            nonlocal max_concurrent_seen, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent_seen:
                    max_concurrent_seen = current_concurrent
            # Simulate work so tasks overlap
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return _make_agent_result()

        mock_agent = AsyncMock(side_effect=counting_agent)

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        assert max_concurrent_seen <= 2
        assert mock_agent.call_count == 4

    @pytest.mark.asyncio
    async def test_default_concurrency_is_four(self, tmp_path: Path) -> None:
        """When max_concurrent_agents is not set in state, default to 4."""
        tasks = [
            {"task_id": f"t{i}", "title": f"Task {i}", "dependencies": [], "description": f"desc {i}"} for i in range(6)
        ]
        # Explicitly omit max_concurrent_agents — it should default
        state = _make_state(tmp_path, dev_tasks=tasks)
        state.pop("max_concurrent_agents", None)
        ui = _make_ui()

        max_concurrent_seen = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def counting_agent(**kwargs):
            nonlocal max_concurrent_seen, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent_seen:
                    max_concurrent_seen = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return _make_agent_result()

        mock_agent = AsyncMock(side_effect=counting_agent)

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        assert max_concurrent_seen <= 4
        assert mock_agent.call_count == 6

    @pytest.mark.asyncio
    async def test_semaphore_of_one_serializes_execution(self, tmp_path: Path) -> None:
        """With max_concurrent_agents=1, tasks run strictly one at a time."""
        tasks = [
            {"task_id": f"t{i}", "title": f"Task {i}", "dependencies": [], "description": f"d{i}"} for i in range(3)
        ]
        state = _make_state(tmp_path, dev_tasks=tasks, max_concurrent_agents=1)
        ui = _make_ui()

        max_concurrent_seen = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def counting_agent(**kwargs):
            nonlocal max_concurrent_seen, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent_seen:
                    max_concurrent_seen = current_concurrent
            await asyncio.sleep(0.02)
            async with lock:
                current_concurrent -= 1
            return _make_agent_result()

        mock_agent = AsyncMock(side_effect=counting_agent)

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        assert max_concurrent_seen == 1
        assert mock_agent.call_count == 3


# ---------------------------------------------------------------------------
# Prompt construction details
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify that the prompt assembled for the agent contains all required pieces."""

    @pytest.mark.asyncio
    async def test_acceptance_criteria_in_prompt(self, tmp_path: Path) -> None:
        """Each acceptance criterion should appear in the prompt as a bullet."""
        task = {
            "task_id": "ac_test",
            "title": "AC Task",
            "description": "desc",
            "dependencies": [],
            "acceptance_criteria": ["criterion alpha", "criterion beta"],
        }
        state = _make_state(tmp_path, dev_tasks=[task])
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        prompt = mock_agent.call_args.kwargs["user_prompt"]
        assert "- criterion alpha" in prompt
        assert "- criterion beta" in prompt

    @pytest.mark.asyncio
    async def test_directory_scope_in_prompt(self, tmp_path: Path) -> None:
        """Directory scope should appear in the prompt."""
        task = {
            "task_id": "scope_test",
            "title": "Scoped",
            "description": "desc",
            "directory_scope": "src/api/",
            "dependencies": [],
        }
        state = _make_state(tmp_path, dev_tasks=[task])
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        prompt = mock_agent.call_args.kwargs["user_prompt"]
        assert "src/api/" in prompt

    @pytest.mark.asyncio
    async def test_missing_optional_fields_handled(self, tmp_path: Path) -> None:
        """Tasks missing description, directory_scope, or acceptance_criteria don't crash."""
        task = {"task_id": "minimal", "title": "Minimal", "dependencies": []}
        state = _make_state(tmp_path, dev_tasks=[task])
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            result = await dev_node(state, ui)

        assert result == {"current_stage": "dev"}
        mock_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_cwd_is_workspace_path(self, tmp_path: Path) -> None:
        """The agent cwd should be the workspace directory."""
        state = _make_state(tmp_path)
        ui = _make_ui()

        mock_agent = AsyncMock(return_value=_make_agent_result())

        with patch("kindle.stages.dev.run_agent", mock_agent):
            await dev_node(state, ui)

        expected_ws = str(Path(state["project_dir"]) / "workspace")
        first_call_kwargs = mock_agent.call_args_list[0].kwargs
        assert first_call_kwargs["cwd"] == expected_ws
