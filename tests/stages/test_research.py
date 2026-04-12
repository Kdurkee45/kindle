"""Tests for kindle.stages.research — technology landscape research node."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.agent import AgentResult
from kindle.stages.research import SYSTEM_PROMPT, research_node
from kindle.state import KindleState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods research_node calls."""
    return MagicMock()


def _make_agent_result(text: str = "fallback research text") -> AgentResult:
    return AgentResult(
        text=text,
        tool_calls=[],
        raw_messages=[],
        elapsed_seconds=4.2,
        turns_used=5,
    )


def _base_state(project_dir: str) -> KindleState:
    """Minimal valid state for research_node."""
    return {
        "project_dir": project_dir,
        "idea": "Build a todo app with real-time sync",
        "feature_spec": {"features": ["real-time sync", "offline mode"]},
    }  # type: ignore[typeddict-item]


# ---------------------------------------------------------------------------
# research_node — happy path
# ---------------------------------------------------------------------------


class TestResearchNodeHappyPath:
    """Tests for the main success path of research_node."""

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_returns_research_report_from_file(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """When the agent writes research_report.md, that content is returned."""
        project_dir = str(tmp_path)
        ws = tmp_path / "workspace"
        ws.mkdir(parents=True)
        report_content = "# Research Report\n\nDetailed findings here."
        (ws / "research_report.md").write_text(report_content)

        mock_run_agent.return_value = _make_agent_result("agent fallback text")
        ui = _make_ui()

        result = await research_node(_base_state(project_dir), ui)

        assert result["research_report"] == report_content
        assert result["current_stage"] == "research"

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_falls_back_to_result_text_when_no_file(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """When the agent does NOT write research_report.md, result.text is used."""
        project_dir = str(tmp_path)
        # workspace exists but no report file
        (tmp_path / "workspace").mkdir(parents=True)

        fallback = "Inline research from agent"
        mock_run_agent.return_value = _make_agent_result(fallback)
        ui = _make_ui()

        result = await research_node(_base_state(project_dir), ui)

        assert result["research_report"] == fallback

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_return_dict_has_expected_keys(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """Return value must contain exactly research_report and current_stage."""
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        result = await research_node(_base_state(project_dir), ui)

        assert set(result.keys()) == {"research_report", "current_stage"}


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


class TestResearchArtifactSaving:
    """Verify that research_report is persisted via save_artifact."""

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_save_artifact_called_with_report_from_file(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """save_artifact receives the file content when the report file exists."""
        project_dir = str(tmp_path)
        ws = tmp_path / "workspace"
        ws.mkdir(parents=True)
        report = "# File-based report"
        (ws / "research_report.md").write_text(report)

        mock_run_agent.return_value = _make_agent_result("ignored fallback")
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        mock_save_artifact.assert_called_once_with(project_dir, "research_report.md", report)

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_save_artifact_called_with_fallback_text(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """save_artifact receives result.text when no report file exists."""
        project_dir = str(tmp_path)
        (tmp_path / "workspace").mkdir(parents=True)

        fallback = "Agent text fallback"
        mock_run_agent.return_value = _make_agent_result(fallback)
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        mock_save_artifact.assert_called_once_with(project_dir, "research_report.md", fallback)


# ---------------------------------------------------------------------------
# Stage completion
# ---------------------------------------------------------------------------


class TestResearchStageCompletion:
    """Verify mark_stage_complete is called correctly."""

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_mark_stage_complete_called(self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path):
        """research stage is marked complete after saving the artifact."""
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        mock_mark_complete.assert_called_once_with(project_dir, "research")


# ---------------------------------------------------------------------------
# UI lifecycle calls
# ---------------------------------------------------------------------------


class TestResearchUILifecycle:
    """Verify UI stage_start/stage_done are called in the right order."""

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_ui_stage_start_called(self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path):
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        ui.stage_start.assert_called_once_with("research")

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_ui_stage_done_called(self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path):
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        ui.stage_done.assert_called_once_with("research")

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_stage_start_before_stage_done(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """stage_start must be called before stage_done."""
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()
        call_order: list[str] = []
        ui.stage_start.side_effect = lambda s: call_order.append("start")
        ui.stage_done.side_effect = lambda s: call_order.append("done")

        await research_node(_base_state(project_dir), ui)

        assert call_order == ["start", "done"]


# ---------------------------------------------------------------------------
# Agent prompt construction
# ---------------------------------------------------------------------------


class TestResearchPromptConstruction:
    """Verify the prompt sent to run_agent includes required context."""

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_prompt_includes_feature_spec(self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path):
        """The user_prompt must contain the JSON-serialized feature_spec."""
        project_dir = str(tmp_path)
        feature_spec = {"features": ["auth", "dashboard"], "priority": "high"}
        state = _base_state(project_dir)
        state["feature_spec"] = feature_spec
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(state, ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        user_prompt = call_kwargs["user_prompt"]
        # The feature spec is JSON-dumped into the prompt
        assert json.dumps(feature_spec, indent=2) in user_prompt

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_prompt_includes_idea(self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path):
        """The user_prompt must contain the idea text."""
        project_dir = str(tmp_path)
        state = _base_state(project_dir)
        state["idea"] = "A recipe sharing platform"
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(state, ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        user_prompt = call_kwargs["user_prompt"]
        assert "A recipe sharing platform" in user_prompt

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_prompt_includes_stack_preference_when_provided(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """When stack_preference is set, it must appear in the prompt."""
        project_dir = str(tmp_path)
        state = _base_state(project_dir)
        state["stack_preference"] = "nextjs with tailwind"
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(state, ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        user_prompt = call_kwargs["user_prompt"]
        assert "STACK PREFERENCE: nextjs with tailwind" in user_prompt

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_prompt_excludes_stack_preference_when_empty(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """When stack_preference is empty string, STACK PREFERENCE line is omitted."""
        project_dir = str(tmp_path)
        state = _base_state(project_dir)
        state["stack_preference"] = ""
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(state, ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        user_prompt = call_kwargs["user_prompt"]
        assert "STACK PREFERENCE" not in user_prompt

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_prompt_excludes_stack_preference_when_missing(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """When stack_preference is not in state at all, STACK PREFERENCE is omitted."""
        project_dir = str(tmp_path)
        state = _base_state(project_dir)
        # stack_preference not set (relies on .get default "")
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(state, ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        user_prompt = call_kwargs["user_prompt"]
        assert "STACK PREFERENCE" not in user_prompt

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_prompt_includes_empty_feature_spec_when_missing(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """When feature_spec is absent from state, default {} is serialized."""
        project_dir = str(tmp_path)
        state: KindleState = {
            "project_dir": project_dir,
            "idea": "test idea",
        }  # type: ignore[typeddict-item]
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(state, ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        user_prompt = call_kwargs["user_prompt"]
        assert "{}" in user_prompt


# ---------------------------------------------------------------------------
# run_agent call arguments
# ---------------------------------------------------------------------------


class TestResearchRunAgentArgs:
    """Verify the keyword arguments passed to run_agent."""

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_called_with_correct_persona(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["persona"] == "Principal Research Engineer"

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_called_with_system_prompt(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["system_prompt"] == SYSTEM_PROMPT

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_called_with_correct_stage(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["stage"] == "research"

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_called_with_max_turns_30(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["max_turns"] == 30

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_called_with_correct_allowed_tools(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["allowed_tools"] == ["Read", "Write", "Bash", "Glob", "Grep"]

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_passes_model_from_state(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """model from state should be forwarded to run_agent."""
        project_dir = str(tmp_path)
        state = _base_state(project_dir)
        state["model"] = "claude-sonnet-4-20250514"
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(state, ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_passes_none_model_when_absent(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """When model is not in state, None is passed to run_agent."""
        project_dir = str(tmp_path)
        state = _base_state(project_dir)
        # model not in state
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(state, ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["model"] is None

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_cwd_is_workspace(self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path):
        """cwd passed to run_agent should be the workspace directory."""
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        expected_cwd = str(tmp_path / "workspace")
        assert call_kwargs["cwd"] == expected_cwd

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_run_agent_project_dir_forwarded(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        project_dir = str(tmp_path)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["project_dir"] == project_dir


# ---------------------------------------------------------------------------
# Report file cleanup
# ---------------------------------------------------------------------------


class TestResearchFileCleanup:
    """Verify that the workspace report file is deleted after being read."""

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_report_file_deleted_after_save(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """research_report.md in workspace should be unlinked after saving."""
        project_dir = str(tmp_path)
        ws = tmp_path / "workspace"
        ws.mkdir(parents=True)
        report_file = ws / "research_report.md"
        report_file.write_text("# Report")

        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        await research_node(_base_state(project_dir), ui)

        assert not report_file.exists(), "Report file should be deleted after saving"

    @patch("kindle.stages.research.mark_stage_complete")
    @patch("kindle.stages.research.save_artifact")
    @patch("kindle.stages.research.run_agent", new_callable=AsyncMock)
    async def test_no_error_when_report_file_missing(
        self, mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
    ):
        """No crash when the report file does not exist (nothing to clean up)."""
        project_dir = str(tmp_path)
        (tmp_path / "workspace").mkdir(parents=True)
        mock_run_agent.return_value = _make_agent_result()
        ui = _make_ui()

        # Should not raise
        result = await research_node(_base_state(project_dir), ui)
        assert result["research_report"] is not None


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT constant
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """Validate the SYSTEM_PROMPT content."""

    def test_system_prompt_mentions_research_engineer(self):
        assert "Principal Research Engineer" in SYSTEM_PROMPT

    def test_system_prompt_mentions_research_report_output(self):
        assert "research_report.md" in SYSTEM_PROMPT

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100
