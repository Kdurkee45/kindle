"""Tests for kindle.stages.research — technology landscape research."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.stages.research import SYSTEM_PROMPT, research_node
from tests.constants import SAMPLE_FEATURE_SPEC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


@pytest.fixture
def research_state(make_state):
    """Factory with research-stage defaults pre-applied."""

    def _factory(**overrides):
        defaults = {
            "feature_spec": SAMPLE_FEATURE_SPEC,
            "stack_preference": "React + Node.js",
        }
        defaults.update(overrides)
        return make_state(**defaults)

    return _factory


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestResearchHappyPath:
    """Agent generates research_report.md, it is read, saved as artifact, and returned."""

    @pytest.mark.asyncio
    async def test_report_read_from_workspace_and_saved_as_artifact(
        self, tmp_path: Path, research_state, make_ui
    ) -> None:
        """When agent writes research_report.md the file contents become the artifact."""
        state = research_state()
        ui = make_ui()
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
    async def test_workspace_file_cleaned_up_after_read(self, tmp_path: Path, research_state, make_ui) -> None:
        """The workspace research_report.md is deleted after it's been read."""
        state = research_state()
        ui = make_ui()
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
    async def test_fallback_to_agent_text_when_no_file(self, tmp_path: Path, research_state, make_ui) -> None:
        """If the agent never writes a file, we fall back to result.text."""
        state = research_state()
        ui = make_ui()
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
    async def test_prompt_includes_idea(self, tmp_path: Path, research_state, make_ui) -> None:
        """The user's original idea appears in the prompt."""
        state = research_state(idea="a social media dashboard")
        ui = make_ui()
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
    async def test_prompt_includes_feature_spec_json(self, tmp_path: Path, research_state, make_ui) -> None:
        """The full feature spec is serialized as JSON in the prompt."""
        state = research_state()
        ui = make_ui()
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
    async def test_prompt_includes_stack_preference(self, tmp_path: Path, research_state, make_ui) -> None:
        """When stack_preference is set, it appears in the prompt."""
        state = research_state(stack_preference="Python + FastAPI")
        ui = make_ui()
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
    async def test_prompt_omits_stack_preference_when_empty(self, tmp_path: Path, research_state, make_ui) -> None:
        """When stack_preference is empty string, STACK PREFERENCE is not in prompt."""
        state = research_state(stack_preference="")
        ui = make_ui()
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
    async def test_system_prompt_is_research_engineer(self, tmp_path: Path, research_state, make_ui) -> None:
        """The system prompt sent to run_agent is the SYSTEM_PROMPT constant."""
        state = research_state()
        ui = make_ui()
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
    async def test_missing_stack_preference(self, tmp_path: Path, research_state, make_ui) -> None:
        """No KeyError when stack_preference is completely absent from state."""
        state = research_state()
        del state["stack_preference"]  # remove entirely
        ui = make_ui()
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
    async def test_missing_feature_spec_defaults_to_empty_dict(self, tmp_path: Path, research_state, make_ui) -> None:
        """feature_spec defaults to {} when absent."""
        state = research_state()
        del state["feature_spec"]
        ui = make_ui()
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
    async def test_missing_idea_defaults_to_empty_string(self, tmp_path: Path, research_state, make_ui) -> None:
        """idea defaults to '' when absent."""
        state = research_state()
        del state["idea"]
        ui = make_ui()
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
    async def test_return_keys(self, tmp_path: Path, research_state, make_ui) -> None:
        """Returned dict contains exactly research_report and current_stage."""
        state = research_state()
        ui = make_ui()
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
    async def test_current_stage_is_research(self, tmp_path: Path, research_state, make_ui) -> None:
        """current_stage is always 'research'."""
        state = research_state()
        ui = make_ui()
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
    async def test_research_report_content_matches_file(self, tmp_path: Path, research_state, make_ui) -> None:
        """The research_report value is the exact content from the workspace file."""
        state = research_state()
        ui = make_ui()
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
    async def test_ui_stage_start_called(self, tmp_path: Path, research_state, make_ui) -> None:
        """ui.stage_start('research') is called before the agent runs."""
        state = research_state()
        ui = make_ui()
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
    async def test_ui_stage_done_called(self, tmp_path: Path, research_state, make_ui) -> None:
        """ui.stage_done('research') is called at the end."""
        state = research_state()
        ui = make_ui()
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
    async def test_mark_stage_complete_records_in_metadata(self, tmp_path: Path, research_state, make_ui) -> None:
        """mark_stage_complete writes 'research' to metadata.json stages_completed."""
        state = research_state()
        ui = make_ui()
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
    async def test_lifecycle_ordering(self, tmp_path: Path, research_state, make_ui) -> None:
        """stage_start is called before stage_done, with mark_stage_complete between."""
        state = research_state()
        ui = make_ui()
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
    async def test_agent_called_with_correct_stage(self, tmp_path: Path, research_state, make_ui) -> None:
        """run_agent receives stage='research'."""
        state = research_state()
        ui = make_ui()
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
    async def test_agent_called_with_cwd_as_workspace(self, tmp_path: Path, research_state, make_ui) -> None:
        """run_agent cwd is the workspace directory."""
        state = research_state()
        ui = make_ui()
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
    async def test_agent_called_with_allowed_tools(self, tmp_path: Path, research_state, make_ui) -> None:
        """run_agent receives the expected allowed_tools list."""
        state = research_state()
        ui = make_ui()
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
    async def test_agent_called_with_max_turns(self, tmp_path: Path, research_state, make_ui) -> None:
        """run_agent receives max_turns=30."""
        state = research_state()
        ui = make_ui()
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
    async def test_model_passed_through_from_state(self, tmp_path: Path, research_state, make_ui) -> None:
        """run_agent receives model from state."""
        state = research_state(model="claude-sonnet-4-20250514")
        ui = make_ui()
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
    async def test_model_none_when_absent_from_state(self, tmp_path: Path, research_state, make_ui) -> None:
        """run_agent receives model=None when state has no model key."""
        state = research_state()
        # Ensure no model key
        state.pop("model", None)
        ui = make_ui()
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
    async def test_agent_called_exactly_once(self, tmp_path: Path, research_state, make_ui) -> None:
        """run_agent is invoked exactly once per research_node call."""
        state = research_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "research_report.md").write_text("report")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.research.run_agent", mock_agent):
            await research_node(state, ui)

        assert mock_agent.call_count == 1
