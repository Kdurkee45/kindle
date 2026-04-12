"""Tests for kindle.stages.research — technology landscape research."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.stages.research import SYSTEM_PROMPT, research_node

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

## Libraries
- Prisma 5.x as the ORM
- NextAuth.js for authentication

## Pitfalls
- N+1 query patterns with Prisma
- JWT token storage in localStorage is insecure
"""


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
        "stack_preference": "React + Node.js",
    }
    state.update(overrides)
    return state


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods research_node actually calls."""
    ui = MagicMock()
    ui.auto_approve = False
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    return ui


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestResearchHappyPath:
    """Agent generates research_report.md, it is read, saved as artifact, and returned."""

    @pytest.mark.asyncio
    async def test_report_read_from_workspace_and_saved_as_artifact(self, tmp_path: Path) -> None:
        """When agent writes research_report.md the file contents become the artifact."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text(SAMPLE_RESEARCH_REPORT)
            return MagicMock(text="fallback text")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            result = await research_node(state, ui)

        assert result["research_report"] == SAMPLE_RESEARCH_REPORT
        # Artifact should be persisted on disk
        artifact_path = Path(state["project_dir"]) / "artifacts" / "research_report.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == SAMPLE_RESEARCH_REPORT

    @pytest.mark.asyncio
    async def test_workspace_file_cleaned_up_after_read(self, tmp_path: Path) -> None:
        """The workspace research_report.md is deleted after it's been read."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text(SAMPLE_RESEARCH_REPORT)
            return MagicMock(text="fallback")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        assert not (ws / "research_report.md").exists()

    @pytest.mark.asyncio
    async def test_fallback_to_agent_text_when_no_file(self, tmp_path: Path) -> None:
        """If the agent never writes a file, we fall back to result.text."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        fallback = "Agent returned this text instead of a file."

        async def fake_run_agent(**kwargs):
            # Intentionally do NOT write research_report.md
            return MagicMock(text=fallback)

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            result = await research_node(state, ui)

        assert result["research_report"] == fallback
        # Artifact should still be saved (with the fallback text)
        artifact_path = Path(state["project_dir"]) / "artifacts" / "research_report.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == fallback


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
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "a social media dashboard" in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_prompt_includes_feature_spec_json(self, tmp_path: Path) -> None:
        """The full feature spec is serialized as JSON in the prompt."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        prompt = call_kwargs["user_prompt"]
        # The JSON should be embedded with indent=2
        assert json.dumps(SAMPLE_FEATURE_SPEC, indent=2) in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_stack_preference(self, tmp_path: Path) -> None:
        """When stack_preference is set, it appears in the prompt."""
        state = _make_state(tmp_path, stack_preference="Python + FastAPI")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "STACK PREFERENCE: Python + FastAPI" in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_prompt_omits_stack_preference_when_empty(self, tmp_path: Path) -> None:
        """When stack_preference is empty string, STACK PREFERENCE is not in prompt."""
        state = _make_state(tmp_path, stack_preference="")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "STACK PREFERENCE" not in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_system_prompt_is_research_engineer(self, tmp_path: Path) -> None:
        """The system prompt sent to run_agent is the SYSTEM_PROMPT constant."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["system_prompt"] == SYSTEM_PROMPT
        assert call_kwargs["persona"] == "Principal Research Engineer"


# ---------------------------------------------------------------------------
# Missing optional state
# ---------------------------------------------------------------------------


class TestMissingOptionalState:
    """Graceful handling when optional keys are absent from state."""

    @pytest.mark.asyncio
    async def test_missing_stack_preference(self, tmp_path: Path) -> None:
        """No KeyError when stack_preference is completely absent from state."""
        state = _make_state(tmp_path)
        del state["stack_preference"]  # remove entirely
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            result = await research_node(state, ui)

        assert result["research_report"] == "report"
        # STACK PREFERENCE should NOT appear in prompt
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
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            result = await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert json.dumps({}, indent=2) in call_kwargs["user_prompt"]
        assert result["research_report"] == "report"

    @pytest.mark.asyncio
    async def test_missing_idea_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """idea defaults to '' when absent."""
        state = _make_state(tmp_path)
        del state["idea"]
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            result = await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "IDEA: " in call_kwargs["user_prompt"]
        assert result["research_report"] == "report"


# ---------------------------------------------------------------------------
# State return shape
# ---------------------------------------------------------------------------


class TestStateReturn:
    """Verify the dict returned by research_node has the correct keys/values."""

    @pytest.mark.asyncio
    async def test_return_keys(self, tmp_path: Path) -> None:
        """Returned dict contains exactly research_report and current_stage."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report content")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            result = await research_node(state, ui)

        assert set(result.keys()) == {"research_report", "current_stage"}

    @pytest.mark.asyncio
    async def test_current_stage_is_research(self, tmp_path: Path) -> None:
        """current_stage is always 'research'."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report content")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            result = await research_node(state, ui)

        assert result["current_stage"] == "research"

    @pytest.mark.asyncio
    async def test_research_report_content_matches_file(self, tmp_path: Path) -> None:
        """The research_report value is the exact content from the workspace file."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        expected = "# Detailed Research\n\nSome thorough analysis here.\n"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text(expected)
            return MagicMock(text="ignored fallback")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            result = await research_node(state, ui)

        assert result["research_report"] == expected


# ---------------------------------------------------------------------------
# Stage lifecycle (UI + artifact calls)
# ---------------------------------------------------------------------------


class TestStageLifecycle:
    """Verify ui.stage_start, mark_stage_complete, ui.stage_done are called in order."""

    @pytest.mark.asyncio
    async def test_ui_stage_start_called(self, tmp_path: Path) -> None:
        """ui.stage_start('research') is called before the agent runs."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        ui.stage_start.assert_called_once_with("research")

    @pytest.mark.asyncio
    async def test_ui_stage_done_called(self, tmp_path: Path) -> None:
        """ui.stage_done('research') is called at the end."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        ui.stage_done.assert_called_once_with("research")

    @pytest.mark.asyncio
    async def test_mark_stage_complete_records_in_metadata(self, tmp_path: Path) -> None:
        """mark_stage_complete writes 'research' to metadata.json stages_completed."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "research" in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_lifecycle_ordering(self, tmp_path: Path) -> None:
        """stage_start is called before stage_done, with mark_stage_complete between."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        call_order: list[str] = []

        original_stage_start = ui.stage_start
        original_stage_done = ui.stage_done

        def track_stage_start(stage):
            call_order.append("stage_start")

        def track_stage_done(stage):
            call_order.append("stage_done")

        ui.stage_start = MagicMock(side_effect=track_stage_start)
        ui.stage_done = MagicMock(side_effect=track_stage_done)

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with (
            patch("kindle.stages.research.run_agent", mock_agent),
            patch(
                "kindle.stages.research.mark_stage_complete",
                side_effect=lambda *a: call_order.append("mark_stage_complete"),
            ),
        ):
            await research_node(state, ui)

        assert call_order == ["stage_start", "mark_stage_complete", "stage_done"]


# ---------------------------------------------------------------------------
# Agent call arguments
# ---------------------------------------------------------------------------


class TestAgentCallArgs:
    """Verify the keyword arguments passed to run_agent."""

    @pytest.mark.asyncio
    async def test_agent_called_with_correct_stage(self, tmp_path: Path) -> None:
        """run_agent receives stage='research'."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["stage"] == "research"

    @pytest.mark.asyncio
    async def test_agent_called_with_cwd_as_workspace(self, tmp_path: Path) -> None:
        """run_agent cwd is the workspace directory."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["cwd"] == str(ws)

    @pytest.mark.asyncio
    async def test_agent_called_with_allowed_tools(self, tmp_path: Path) -> None:
        """run_agent receives the expected allowed_tools list."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["allowed_tools"] == ["Read", "Write", "Bash", "Glob", "Grep"]

    @pytest.mark.asyncio
    async def test_agent_called_with_max_turns(self, tmp_path: Path) -> None:
        """run_agent receives max_turns=30."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["max_turns"] == 30

    @pytest.mark.asyncio
    async def test_model_passed_through_from_state(self, tmp_path: Path) -> None:
        """run_agent receives model from state."""
        state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_model_none_when_absent_from_state(self, tmp_path: Path) -> None:
        """run_agent receives model=None when state has no model key."""
        state = _make_state(tmp_path)
        # Ensure no model key
        state.pop("model", None)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["model"] is None

    @pytest.mark.asyncio
    async def test_agent_called_exactly_once(self, tmp_path: Path) -> None:
        """run_agent is invoked exactly once per research_node call."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        assert mock_agent.call_count == 1
