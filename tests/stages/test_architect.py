"""Tests for kindle.stages.architect — architecture design and dev task creation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.stages.architect import SYSTEM_PROMPT, architect_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_FEATURE_SPEC = {
    "app_name": "TaskFlow",
    "idea": "a task management app",
    "core_features": ["task CRUD", "auth"],
    "tech_constraints": ["React frontend"],
}

SAMPLE_RESEARCH_REPORT = """\
# Technology Landscape Research

## Frameworks
- React 18.x for the frontend SPA
- Express 4.x for the REST API
"""

SAMPLE_ARCHITECTURE = """\
# Architecture

## Tech Stack
- React 18.x frontend
- Express 4.x backend
- PostgreSQL database

## Directory Structure
```
src/
  components/
  api/
  models/
```
"""

SAMPLE_DEV_TASKS: list[dict] = [
    {
        "task_id": "task_01",
        "title": "Set up project structure and dependencies",
        "description": "Scaffold the project with pyproject.toml, directory structure, and config.",
        "directory_scope": ".",
        "dependencies": [],
        "acceptance_criteria": [
            "pyproject.toml exists with all dependencies",
            "Directory structure matches architecture",
        ],
    },
    {
        "task_id": "task_02",
        "title": "Build data models and database layer",
        "description": "Create SQLAlchemy models and migration scripts.",
        "directory_scope": "src/models/",
        "dependencies": ["task_01"],
        "acceptance_criteria": ["All models defined", "Tests pass"],
    },
]


def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal KindleState dict pointing at *tmp_path* as project_dir."""
    project_dir = tmp_path / "project"
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)
    # metadata.json needed by mark_stage_complete
    meta = {"project_id": "kindle_test1234", "stages_completed": []}
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    state: dict = {
        "project_dir": str(project_dir),
        "idea": "a task management app",
        "feature_spec": SAMPLE_FEATURE_SPEC,
        "research_report": SAMPLE_RESEARCH_REPORT,
        "stack_preference": "React + Node.js",
    }
    state.update(overrides)
    return state


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods architect_node actually calls."""
    ui = MagicMock()
    ui.auto_approve = False
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.show_artifact = MagicMock()
    ui.prompt_arch_review = MagicMock(return_value=(True, ""))
    return ui


def _write_workspace_files(
    ws: Path,
    architecture: str = SAMPLE_ARCHITECTURE,
    dev_tasks: list[dict] | None = None,
) -> None:
    """Write architecture.md and dev_tasks.json into the workspace directory."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "architecture.md").write_text(architecture)
    if dev_tasks is not None:
        (ws / "dev_tasks.json").write_text(json.dumps(dev_tasks, indent=2))
    else:
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_DEV_TASKS, indent=2))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestArchitectHappyPath:
    """Agent generates architecture.md and dev_tasks.json, both are read and saved."""

    @pytest.mark.asyncio
    async def test_architecture_read_from_workspace_and_saved_as_artifact(self, tmp_path: Path) -> None:
        """When agent writes architecture.md the file contents become the artifact."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="fallback text")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["architecture"] == SAMPLE_ARCHITECTURE
        # Artifact should be persisted on disk
        artifact_path = Path(state["project_dir"]) / "artifacts" / "architecture.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == SAMPLE_ARCHITECTURE

    @pytest.mark.asyncio
    async def test_dev_tasks_read_from_workspace_and_saved_as_artifact(self, tmp_path: Path) -> None:
        """dev_tasks.json contents are parsed and saved as artifact."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="fallback text")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["dev_tasks"] == SAMPLE_DEV_TASKS
        artifact_path = Path(state["project_dir"]) / "artifacts" / "dev_tasks.json"
        assert artifact_path.exists()
        saved = json.loads(artifact_path.read_text())
        assert saved == SAMPLE_DEV_TASKS

    @pytest.mark.asyncio
    async def test_workspace_files_cleaned_up_after_read(self, tmp_path: Path) -> None:
        """architecture.md and dev_tasks.json are deleted from workspace after reading."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="fallback")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        assert not (ws / "architecture.md").exists()
        assert not (ws / "dev_tasks.json").exists()

    @pytest.mark.asyncio
    async def test_fallback_to_agent_text_when_no_architecture_file(self, tmp_path: Path) -> None:
        """If agent never writes architecture.md, we fall back to result.text."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        fallback = "Agent returned this text instead of a file."

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            # Write dev_tasks.json but NOT architecture.md
            (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_DEV_TASKS))
            return MagicMock(text=fallback)

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["architecture"] == fallback
        artifact_path = Path(state["project_dir"]) / "artifacts" / "architecture.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == fallback

    @pytest.mark.asyncio
    async def test_info_message_reports_task_count(self, tmp_path: Path) -> None:
        """ui.info is called with the number of dev tasks created."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        ui.info.assert_any_call("Architecture designed — 2 dev task(s) created.")


# ---------------------------------------------------------------------------
# Dev tasks JSON parsing
# ---------------------------------------------------------------------------


class TestDevTasksParsing:
    """Verify list[dict] parsing from agent output file."""

    @pytest.mark.asyncio
    async def test_plain_array_format(self, tmp_path: Path) -> None:
        """dev_tasks.json containing a plain JSON array is parsed correctly."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        tasks = [{"task_id": "task_01", "title": "Only task"}]

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws, dev_tasks=tasks)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["dev_tasks"] == tasks

    @pytest.mark.asyncio
    async def test_object_wrapper_with_tasks_key(self, tmp_path: Path) -> None:
        """dev_tasks.json with {"tasks": [...]} wrapper extracts the inner list."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        tasks = [{"task_id": "task_01", "title": "Wrapped task"}]
        wrapped = {"tasks": tasks}

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
            (ws / "dev_tasks.json").write_text(json.dumps(wrapped))
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["dev_tasks"] == tasks

    @pytest.mark.asyncio
    async def test_invalid_json_produces_empty_tasks_and_error(self, tmp_path: Path) -> None:
        """Malformed JSON in dev_tasks.json results in empty list and ui.error."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
            (ws / "dev_tasks.json").write_text("{invalid json!!!")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["dev_tasks"] == []
        ui.error.assert_called_once_with("Failed to parse dev_tasks.json — no tasks generated.")

    @pytest.mark.asyncio
    async def test_missing_dev_tasks_file_produces_empty_list_and_error(self, tmp_path: Path) -> None:
        """When dev_tasks.json is missing, returns empty list and calls ui.error."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
            # Intentionally do NOT write dev_tasks.json
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["dev_tasks"] == []
        ui.error.assert_called_once_with("Agent did not produce dev_tasks.json.")

    @pytest.mark.asyncio
    async def test_empty_array_produces_zero_tasks(self, tmp_path: Path) -> None:
        """dev_tasks.json containing [] results in an empty task list."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws, dev_tasks=[])
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["dev_tasks"] == []
        ui.info.assert_any_call("Architecture designed — 0 dev task(s) created.")

    @pytest.mark.asyncio
    async def test_object_without_tasks_key_produces_empty_list(self, tmp_path: Path) -> None:
        """dev_tasks.json with an object lacking a 'tasks' key gives empty list."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
            (ws / "dev_tasks.json").write_text(json.dumps({"other_key": "value"}))
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["dev_tasks"] == []


# ---------------------------------------------------------------------------
# Architecture review gate
# ---------------------------------------------------------------------------


class TestArchitectureReviewGate:
    """Verify the human-review gate for architecture approval."""

    @pytest.mark.asyncio
    async def test_review_gate_triggered_when_review_arch_true_and_auto_approve_false(self, tmp_path: Path) -> None:
        """When review_arch=True and auto_approve=False, prompt_arch_review is called."""
        state = _make_state(tmp_path, review_arch=True, auto_approve=False)
        ui = _make_ui()
        ui.prompt_arch_review.return_value = (True, "")
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        ui.show_artifact.assert_called_once_with("Architecture", SAMPLE_ARCHITECTURE)
        ui.prompt_arch_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_gate_shows_architecture_content(self, tmp_path: Path) -> None:
        """show_artifact is called with the actual architecture content."""
        custom_arch = "# Custom Architecture\nSpecial design."
        state = _make_state(tmp_path, review_arch=True, auto_approve=False)
        ui = _make_ui()
        ui.prompt_arch_review.return_value = (True, "")
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws, architecture=custom_arch)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        ui.show_artifact.assert_called_once_with("Architecture", custom_arch)

    @pytest.mark.asyncio
    async def test_review_gate_skipped_when_review_arch_false(self, tmp_path: Path) -> None:
        """When review_arch=False, prompt_arch_review is NOT called."""
        state = _make_state(tmp_path, review_arch=False, auto_approve=False)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        ui.prompt_arch_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_review_gate_skipped_when_auto_approve_true(self, tmp_path: Path) -> None:
        """When auto_approve=True, prompt_arch_review is NOT called even with review_arch=True."""
        state = _make_state(tmp_path, review_arch=True, auto_approve=True)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        ui.prompt_arch_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_review_gate_skipped_when_review_arch_absent(self, tmp_path: Path) -> None:
        """When review_arch is absent from state, prompt_arch_review is NOT called."""
        state = _make_state(tmp_path)
        # Ensure no review_arch key
        state.pop("review_arch", None)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        ui.prompt_arch_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_review_disapproved_logs_feedback(self, tmp_path: Path) -> None:
        """When reviewer disapproves, feedback is logged via ui.info."""
        state = _make_state(tmp_path, review_arch=True, auto_approve=False)
        ui = _make_ui()
        ui.prompt_arch_review.return_value = (False, "Needs more detail on auth.")
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        ui.info.assert_any_call("Architecture feedback: Needs more detail on auth.")
        ui.info.assert_any_call("Architecture revision not yet implemented — proceeding with current design.")
        # Should still return the architecture even if disapproved
        assert result["architecture"] == SAMPLE_ARCHITECTURE

    @pytest.mark.asyncio
    async def test_review_approved_does_not_log_revision_message(self, tmp_path: Path) -> None:
        """When reviewer approves, the revision-not-implemented message is NOT logged."""
        state = _make_state(tmp_path, review_arch=True, auto_approve=False)
        ui = _make_ui()
        ui.prompt_arch_review.return_value = (True, "")
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        # Check that revision message was NOT logged
        for call in ui.info.call_args_list:
            assert "revision not yet implemented" not in str(call)


# ---------------------------------------------------------------------------
# State return shape
# ---------------------------------------------------------------------------


class TestStateReturn:
    """Verify the dict returned by architect_node has the correct keys/values."""

    @pytest.mark.asyncio
    async def test_return_keys(self, tmp_path: Path) -> None:
        """Returned dict contains architecture, dev_tasks, and current_stage."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert set(result.keys()) == {"architecture", "dev_tasks", "current_stage"}

    @pytest.mark.asyncio
    async def test_current_stage_is_architect(self, tmp_path: Path) -> None:
        """current_stage is always 'architect'."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["current_stage"] == "architect"

    @pytest.mark.asyncio
    async def test_architecture_is_string(self, tmp_path: Path) -> None:
        """The architecture value is a string."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert isinstance(result["architecture"], str)
        assert result["architecture"] == SAMPLE_ARCHITECTURE

    @pytest.mark.asyncio
    async def test_dev_tasks_is_list_of_dicts(self, tmp_path: Path) -> None:
        """The dev_tasks value is a list of dicts."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert isinstance(result["dev_tasks"], list)
        for task in result["dev_tasks"]:
            assert isinstance(task, dict)
        assert len(result["dev_tasks"]) == 2


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify the user prompt sent to run_agent contains all relevant state."""

    @pytest.mark.asyncio
    async def test_prompt_includes_idea(self, tmp_path: Path) -> None:
        """The user's original idea appears in the prompt."""
        state = _make_state(tmp_path, idea="a social media dashboard")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "a social media dashboard" in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_prompt_includes_feature_spec_json(self, tmp_path: Path) -> None:
        """The full feature spec is serialized as JSON in the prompt."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        prompt = call_kwargs["user_prompt"]
        assert json.dumps(SAMPLE_FEATURE_SPEC, indent=2) in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_research_report(self, tmp_path: Path) -> None:
        """The research report appears in the prompt."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert SAMPLE_RESEARCH_REPORT in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_prompt_includes_stack_preference(self, tmp_path: Path) -> None:
        """When stack_preference is set, it appears in the prompt."""
        state = _make_state(tmp_path, stack_preference="Python + FastAPI")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "STACK PREFERENCE: Python + FastAPI" in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_prompt_omits_stack_preference_when_empty(self, tmp_path: Path) -> None:
        """When stack_preference is empty string, STACK PREFERENCE is not in prompt."""
        state = _make_state(tmp_path, stack_preference="")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "STACK PREFERENCE" not in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_prompt_ends_with_file_write_instruction(self, tmp_path: Path) -> None:
        """The prompt ends with instruction to write architecture.md and dev_tasks.json."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        prompt = call_kwargs["user_prompt"]
        assert "architecture.md" in prompt
        assert "dev_tasks.json" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_is_architect_constant(self, tmp_path: Path) -> None:
        """The system prompt sent to run_agent is the SYSTEM_PROMPT constant."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["system_prompt"] == SYSTEM_PROMPT
        assert call_kwargs["persona"] == "Principal Solution Architect"


# ---------------------------------------------------------------------------
# Missing optional state
# ---------------------------------------------------------------------------


class TestMissingOptionalState:
    """Graceful handling when optional keys are absent from state."""

    @pytest.mark.asyncio
    async def test_missing_stack_preference(self, tmp_path: Path) -> None:
        """No KeyError when stack_preference is completely absent from state."""
        state = _make_state(tmp_path)
        del state["stack_preference"]
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        assert result["architecture"] == SAMPLE_ARCHITECTURE
        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "STACK PREFERENCE" not in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_missing_feature_spec_defaults_to_empty_dict(self, tmp_path: Path) -> None:
        """feature_spec defaults to {} when absent."""
        state = _make_state(tmp_path)
        del state["feature_spec"]
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert json.dumps({}, indent=2) in call_kwargs["user_prompt"]
        assert result["architecture"] == SAMPLE_ARCHITECTURE

    @pytest.mark.asyncio
    async def test_missing_research_report_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """research_report defaults to '' when absent."""
        state = _make_state(tmp_path)
        del state["research_report"]
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "RESEARCH REPORT:\n" in call_kwargs["user_prompt"]
        assert result["architecture"] == SAMPLE_ARCHITECTURE

    @pytest.mark.asyncio
    async def test_missing_idea_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """idea defaults to '' when absent."""
        state = _make_state(tmp_path)
        del state["idea"]
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            result = await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "IDEA: " in call_kwargs["user_prompt"]
        assert result["architecture"] == SAMPLE_ARCHITECTURE


# ---------------------------------------------------------------------------
# Stage lifecycle (UI + artifact calls)
# ---------------------------------------------------------------------------


class TestStageLifecycle:
    """Verify ui.stage_start, mark_stage_complete, ui.stage_done are called in order."""

    @pytest.mark.asyncio
    async def test_ui_stage_start_called(self, tmp_path: Path) -> None:
        """ui.stage_start('architect') is called before the agent runs."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        ui.stage_start.assert_called_once_with("architect")

    @pytest.mark.asyncio
    async def test_ui_stage_done_called(self, tmp_path: Path) -> None:
        """ui.stage_done('architect') is called at the end."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        ui.stage_done.assert_called_once_with("architect")

    @pytest.mark.asyncio
    async def test_mark_stage_complete_records_in_metadata(self, tmp_path: Path) -> None:
        """mark_stage_complete writes 'architect' to metadata.json stages_completed."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "architect" in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_lifecycle_ordering(self, tmp_path: Path) -> None:
        """stage_start is called before stage_done, with mark_stage_complete between."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        call_order: list[str] = []

        def track_stage_start(stage):
            call_order.append("stage_start")

        def track_stage_done(stage):
            call_order.append("stage_done")

        ui.stage_start = MagicMock(side_effect=track_stage_start)
        ui.stage_done = MagicMock(side_effect=track_stage_done)

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with (
            patch("kindle.stages.architect.run_agent", mock_agent),
            patch(
                "kindle.stages.architect.mark_stage_complete",
                side_effect=lambda *a: call_order.append("mark_stage_complete"),
            ),
        ):
            await architect_node(state, ui)

        assert call_order == ["stage_start", "mark_stage_complete", "stage_done"]


# ---------------------------------------------------------------------------
# Agent call arguments
# ---------------------------------------------------------------------------


class TestAgentCallArgs:
    """Verify the keyword arguments passed to run_agent."""

    @pytest.mark.asyncio
    async def test_agent_called_with_correct_stage(self, tmp_path: Path) -> None:
        """run_agent receives stage='architect'."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["stage"] == "architect"

    @pytest.mark.asyncio
    async def test_agent_called_with_cwd_as_workspace(self, tmp_path: Path) -> None:
        """run_agent cwd is the workspace directory."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["cwd"] == str(ws)

    @pytest.mark.asyncio
    async def test_agent_called_with_allowed_tools(self, tmp_path: Path) -> None:
        """run_agent receives the expected allowed_tools list."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["allowed_tools"] == [
            "Read",
            "Write",
            "Bash",
            "Glob",
            "Grep",
        ]

    @pytest.mark.asyncio
    async def test_agent_called_with_max_turns(self, tmp_path: Path) -> None:
        """run_agent receives max_turns=30."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["max_turns"] == 30

    @pytest.mark.asyncio
    async def test_model_passed_through_from_state(self, tmp_path: Path) -> None:
        """run_agent receives model from state."""
        state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_model_none_when_absent_from_state(self, tmp_path: Path) -> None:
        """run_agent receives model=None when state has no model key."""
        state = _make_state(tmp_path)
        state.pop("model", None)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["model"] is None

    @pytest.mark.asyncio
    async def test_agent_called_exactly_once(self, tmp_path: Path) -> None:
        """run_agent is invoked exactly once per architect_node call."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            _write_workspace_files(ws)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.architect.run_agent", mock_agent):
            await architect_node(state, ui)

        assert mock_agent.call_count == 1
