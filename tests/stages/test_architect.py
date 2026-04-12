"""Tests for kindle.stages.architect — architect_node() stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from kindle.state import KindleState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeAgentResult:
    """Stand-in for kindle.agent.AgentResult."""

    text: str = ""
    tool_calls: list = None  # type: ignore[assignment]
    raw_messages: list = None  # type: ignore[assignment]
    elapsed_seconds: float = 1.0
    turns_used: int = 5

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.raw_messages is None:
            self.raw_messages = []


def _make_ui() -> MagicMock:
    """Create a minimal mock UI with the methods architect_node calls."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.show_artifact = MagicMock()
    ui.prompt_arch_review = MagicMock(return_value=(True, ""))
    return ui


SAMPLE_TASKS = [
    {
        "task_id": "task_01",
        "title": "Project scaffolding",
        "description": "Set up initial structure",
        "directory_scope": ".",
        "dependencies": [],
        "acceptance_criteria": ["pyproject.toml exists"],
    },
    {
        "task_id": "task_02",
        "title": "Data models",
        "description": "Build data layer",
        "directory_scope": "src/models/",
        "dependencies": ["task_01"],
        "acceptance_criteria": ["Models defined"],
    },
]

SAMPLE_ARCHITECTURE = "# Architecture\n\nA great system design."


def _minimal_state(project_dir: str, **overrides) -> KindleState:
    """Return a minimal KindleState dict for architect_node."""
    base: dict = {
        "project_dir": project_dir,
        "idea": "Build a CLI tool",
        "feature_spec": {"name": "CLI tool", "features": ["fast"]},
        "research_report": "Use Python + Click.",
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def _setup_project_dir(tmp_path: Path) -> Path:
    """Create the project directory structure expected by artifacts module."""
    project_dir = tmp_path / "kindle_test1234"
    (project_dir / "artifacts").mkdir(parents=True)
    (project_dir / "logs").mkdir(parents=True)
    (project_dir / "workspace").mkdir(parents=True)
    # metadata.json is required by mark_stage_complete
    metadata = {
        "project_id": "kindle_test1234",
        "idea": "test",
        "created_at": "2026-01-01T00:00:00",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(metadata))
    return project_dir


# ---------------------------------------------------------------------------
# Happy path — architecture.md and dev_tasks.json both exist on disk
# ---------------------------------------------------------------------------


class TestArchitectNodeHappyPath:
    """run_agent produces both architecture.md and dev_tasks.json in workspace."""

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_returns_architecture_and_dev_tasks(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """architect_node returns dict with architecture text and parsed tasks."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        ws.mkdir(exist_ok=True)

        # Simulate agent writing files to workspace
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult(text="fallback text")

        state = _minimal_state(str(project_dir))
        ui = _make_ui()
        result = await architect_node(state, ui)

        assert result["architecture"] == SAMPLE_ARCHITECTURE
        assert result["dev_tasks"] == SAMPLE_TASKS
        assert result["current_stage"] == "architect"

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_saves_architecture_artifact(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """architecture.md is persisted to the artifacts directory."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult()

        result = await architect_node(_minimal_state(str(project_dir)), _make_ui())

        artifact = project_dir / "artifacts" / "architecture.md"
        assert artifact.exists()
        assert artifact.read_text() == SAMPLE_ARCHITECTURE

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_saves_dev_tasks_artifact(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """dev_tasks.json is persisted (pretty-printed) to the artifacts directory."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        artifact = project_dir / "artifacts" / "dev_tasks.json"
        assert artifact.exists()
        saved = json.loads(artifact.read_text())
        assert saved == SAMPLE_TASKS

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_marks_stage_complete(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """metadata.json is updated with 'architect' in stages_completed."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "architect" in meta["stages_completed"]

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_cleans_up_workspace_files(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """architecture.md and dev_tasks.json are removed from workspace after saving."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        assert not (ws / "architecture.md").exists()
        assert not (ws / "dev_tasks.json").exists()

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_ui_stage_lifecycle(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """stage_start and stage_done are called with 'architect'."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()

        await architect_node(_minimal_state(str(project_dir)), ui)

        ui.stage_start.assert_called_once_with("architect")
        ui.stage_done.assert_called_once_with("architect")

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_ui_info_reports_task_count(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """ui.info is called with the number of dev tasks created."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()

        await architect_node(_minimal_state(str(project_dir)), ui)

        # Find the info call that reports task count
        info_calls = [str(c) for c in ui.info.call_args_list]
        assert any("2 dev task(s)" in c for c in info_calls)


# ---------------------------------------------------------------------------
# run_agent call verification
# ---------------------------------------------------------------------------


class TestRunAgentInvocation:
    """Verify the arguments passed to run_agent."""

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_run_agent_called_with_correct_persona(self, mock_run_agent: AsyncMock, tmp_path: Path):
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        mock_run_agent.assert_awaited_once()
        kwargs = mock_run_agent.call_args.kwargs
        assert kwargs["persona"] == "Principal Solution Architect"
        assert kwargs["stage"] == "architect"
        assert kwargs["max_turns"] == 30

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_run_agent_receives_allowed_tools(self, mock_run_agent: AsyncMock, tmp_path: Path):
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        kwargs = mock_run_agent.call_args.kwargs
        assert kwargs["allowed_tools"] == ["Read", "Write", "Bash", "Glob", "Grep"]

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_prompt_includes_idea_and_spec(self, mock_run_agent: AsyncMock, tmp_path: Path):
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        user_prompt = mock_run_agent.call_args.kwargs["user_prompt"]
        assert "Build a CLI tool" in user_prompt
        assert "CLI tool" in user_prompt  # from feature_spec
        assert "Use Python + Click." in user_prompt  # research_report

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_stack_preference_included_when_set(self, mock_run_agent: AsyncMock, tmp_path: Path):
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        state = _minimal_state(str(project_dir), stack_preference="Rust + Tokio")
        await architect_node(state, _make_ui())

        user_prompt = mock_run_agent.call_args.kwargs["user_prompt"]
        assert "STACK PREFERENCE: Rust + Tokio" in user_prompt

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_stack_preference_omitted_when_empty(self, mock_run_agent: AsyncMock, tmp_path: Path):
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        state = _minimal_state(str(project_dir), stack_preference="")
        await architect_node(state, _make_ui())

        user_prompt = mock_run_agent.call_args.kwargs["user_prompt"]
        assert "STACK PREFERENCE" not in user_prompt

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_model_forwarded_from_state(self, mock_run_agent: AsyncMock, tmp_path: Path):
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        state = _minimal_state(str(project_dir), model="claude-sonnet-4-20250514")
        await architect_node(state, _make_ui())

        assert mock_run_agent.call_args.kwargs["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Fallback paths — missing or malformed workspace files
# ---------------------------------------------------------------------------


class TestArchitectNodeFallbacks:
    """Edge cases when agent doesn't produce expected files."""

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_architecture_falls_back_to_result_text(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """When architecture.md doesn't exist, uses result.text as fallback."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        # No architecture.md written — only dev_tasks.json
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult(text="Fallback architecture")

        result = await architect_node(_minimal_state(str(project_dir)), _make_ui())

        assert result["architecture"] == "Fallback architecture"
        # Artifact should still be saved with fallback content
        artifact = project_dir / "artifacts" / "architecture.md"
        assert artifact.read_text() == "Fallback architecture"

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_missing_dev_tasks_produces_empty_list(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """When dev_tasks.json doesn't exist, dev_tasks is empty and error shown."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        # No dev_tasks.json

        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()

        result = await architect_node(_minimal_state(str(project_dir)), ui)

        assert result["dev_tasks"] == []
        ui.error.assert_called_once_with("Agent did not produce dev_tasks.json.")

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_malformed_json_produces_empty_list(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """Invalid JSON in dev_tasks.json results in empty task list and error."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text("{not valid json!!")

        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()

        result = await architect_node(_minimal_state(str(project_dir)), ui)

        assert result["dev_tasks"] == []
        ui.error.assert_called_once_with("Failed to parse dev_tasks.json \u2014 no tasks generated.")

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_dev_tasks_wrapped_in_object_extracted(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """dev_tasks.json with {"tasks": [...]} wrapper is handled correctly."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        wrapped = {"tasks": SAMPLE_TASKS, "metadata": "extra"}
        (ws / "dev_tasks.json").write_text(json.dumps(wrapped))

        mock_run_agent.return_value = _FakeAgentResult()

        result = await architect_node(_minimal_state(str(project_dir)), _make_ui())

        assert result["dev_tasks"] == SAMPLE_TASKS

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_dev_tasks_as_direct_list(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """dev_tasks.json that is a plain array is used directly."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))

        mock_run_agent.return_value = _FakeAgentResult()

        result = await architect_node(_minimal_state(str(project_dir)), _make_ui())

        assert result["dev_tasks"] == SAMPLE_TASKS

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_empty_task_list(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """dev_tasks.json with an empty array produces zero tasks."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text("[]")

        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()

        result = await architect_node(_minimal_state(str(project_dir)), ui)

        assert result["dev_tasks"] == []
        # No error should be shown — file exists and is valid, just empty
        ui.error.assert_not_called()

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_both_files_missing(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """Neither file produced — architecture from result.text, tasks empty."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        mock_run_agent.return_value = _FakeAgentResult(text="Agent raw output")
        ui = _make_ui()

        result = await architect_node(_minimal_state(str(project_dir)), ui)

        assert result["architecture"] == "Agent raw output"
        assert result["dev_tasks"] == []
        ui.error.assert_called_once_with("Agent did not produce dev_tasks.json.")

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_object_without_tasks_key_produces_empty_list(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """dev_tasks.json is an object without 'tasks' key -> empty list."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text('{"metadata": "no tasks key"}')

        mock_run_agent.return_value = _FakeAgentResult()

        result = await architect_node(_minimal_state(str(project_dir)), _make_ui())

        assert result["dev_tasks"] == []


# ---------------------------------------------------------------------------
# State defaults / missing keys
# ---------------------------------------------------------------------------


class TestArchitectNodeStateDefaults:
    """Verify graceful handling of missing optional state keys."""

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_missing_feature_spec_defaults_to_empty_dict(self, mock_run_agent: AsyncMock, tmp_path: Path):
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        state: KindleState = {  # type: ignore[typeddict-item]
            "project_dir": str(project_dir),
            "idea": "test",
            # no feature_spec, no research_report
        }
        result = await architect_node(state, _make_ui())

        # Should not crash; prompt uses defaults
        assert result["current_stage"] == "architect"
        user_prompt = mock_run_agent.call_args.kwargs["user_prompt"]
        assert "{}" in user_prompt  # empty dict stringified

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_missing_model_passes_none(self, mock_run_agent: AsyncMock, tmp_path: Path):
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        state = _minimal_state(str(project_dir))  # no model key
        await architect_node(state, _make_ui())

        assert mock_run_agent.call_args.kwargs["model"] is None


# ---------------------------------------------------------------------------
# Architecture review path (review_arch=True)
# ---------------------------------------------------------------------------


class TestArchitectureReview:
    """Tests for the optional human-review-of-architecture flow."""

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_review_not_triggered_by_default(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """Without review_arch in state, no review prompt occurs."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()

        await architect_node(_minimal_state(str(project_dir)), ui)

        ui.prompt_arch_review.assert_not_called()
        ui.show_artifact.assert_not_called()

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_review_not_triggered_when_auto_approve(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """review_arch=True + auto_approve=True skips review."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()

        state = _minimal_state(str(project_dir), review_arch=True, auto_approve=True)
        await architect_node(state, ui)

        ui.prompt_arch_review.assert_not_called()

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_review_approved_proceeds_normally(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """review_arch=True, user approves — proceeds without feedback message."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()
        ui.prompt_arch_review.return_value = (True, "")

        state = _minimal_state(str(project_dir), review_arch=True)
        result = await architect_node(state, ui)

        # prompt_arch_review is called with no arguments (this is the actual
        # behavior in architect.py line 147 — see signature bug note below)
        ui.prompt_arch_review.assert_called_once_with()
        ui.show_artifact.assert_called_once_with("Architecture", SAMPLE_ARCHITECTURE)
        # Stage still completes
        assert result["current_stage"] == "architect"

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_review_rejected_logs_feedback(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """review_arch=True, user rejects — feedback is logged via ui.info."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()
        ui.prompt_arch_review.return_value = (False, "Needs more error handling")

        state = _minimal_state(str(project_dir), review_arch=True)
        result = await architect_node(state, ui)

        # Verify feedback message is logged
        info_calls = [str(c) for c in ui.info.call_args_list]
        assert any("Needs more error handling" in c for c in info_calls)
        assert any("revision not yet implemented" in c for c in info_calls)
        # Stage still completes despite rejection (revision not implemented)
        assert result["current_stage"] == "architect"

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_review_called_without_arch_summary_arg(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """Document the signature mismatch: architect.py calls
        ui.prompt_arch_review() with NO arguments, but the real UI method
        expects (self, arch_summary: str). With a MagicMock this works fine,
        but calling the real UI.prompt_arch_review would raise TypeError.

        This test documents the bug at architect.py:147.
        """
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()
        ui = _make_ui()
        ui.prompt_arch_review.return_value = (True, "")

        state = _minimal_state(str(project_dir), review_arch=True)
        await architect_node(state, ui)

        # The call was made with ZERO positional arguments (excluding self).
        # The real method signature is: prompt_arch_review(self, arch_summary: str)
        # This means the real call would fail with:
        #   TypeError: prompt_arch_review() missing 1 required positional argument: 'arch_summary'
        ui.prompt_arch_review.assert_called_once_with()
        assert ui.prompt_arch_review.call_args == call()


# ---------------------------------------------------------------------------
# Cleanup behavior
# ---------------------------------------------------------------------------


class TestWorkspaceCleanup:
    """Verify cleanup of workspace files even when one is missing."""

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_only_existing_files_are_unlinked(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """If only architecture.md exists, only it is removed."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        # dev_tasks.json does NOT exist

        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        assert not (ws / "architecture.md").exists()
        assert not (ws / "dev_tasks.json").exists()  # was never there

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_no_crash_when_neither_file_exists(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """Cleanup loop handles both files missing without error."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        mock_run_agent.return_value = _FakeAgentResult()

        # Should not raise
        await architect_node(_minimal_state(str(project_dir)), _make_ui())


# ---------------------------------------------------------------------------
# Artifact content fidelity
# ---------------------------------------------------------------------------


class TestArtifactContentFidelity:
    """Ensure saved artifacts match original content exactly."""

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_dev_tasks_artifact_is_pretty_printed(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """The saved dev_tasks.json artifact uses indent=2 formatting."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        (ws / "dev_tasks.json").write_text(json.dumps(SAMPLE_TASKS))
        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        artifact_text = (project_dir / "artifacts" / "dev_tasks.json").read_text()
        assert artifact_text == json.dumps(SAMPLE_TASKS, indent=2)

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_empty_tasks_saved_as_empty_array(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """When no tasks produced, artifact is saved as '[]'."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        (ws / "architecture.md").write_text(SAMPLE_ARCHITECTURE)
        # No dev_tasks.json

        mock_run_agent.return_value = _FakeAgentResult()

        await architect_node(_minimal_state(str(project_dir)), _make_ui())

        artifact_text = (project_dir / "artifacts" / "dev_tasks.json").read_text()
        assert json.loads(artifact_text) == []

    @patch("kindle.stages.architect.run_agent", new_callable=AsyncMock)
    async def test_architecture_with_special_characters_preserved(self, mock_run_agent: AsyncMock, tmp_path: Path):
        """Architecture text with unicode and special chars is saved faithfully."""
        from kindle.stages.architect import architect_node

        project_dir = _setup_project_dir(tmp_path)
        ws = project_dir / "workspace"
        special_arch = "# Architecture \u2014 \u00e9l\u00e8ve design\n\n```\n+---+\n| A |\n+---+\n```"
        (ws / "architecture.md").write_text(special_arch)
        (ws / "dev_tasks.json").write_text("[]")
        mock_run_agent.return_value = _FakeAgentResult()

        result = await architect_node(_minimal_state(str(project_dir)), _make_ui())

        assert result["architecture"] == special_arch
        assert (project_dir / "artifacts" / "architecture.md").read_text() == special_arch
