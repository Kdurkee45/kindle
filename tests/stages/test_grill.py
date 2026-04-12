"""Tests for kindle.stages.grill — the structured interrogation stage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from kindle.stages.grill import (
    COMPILE_SYSTEM_PROMPT,
    INTERROGATION_SYSTEM_PROMPT,
    grill_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    {
        "question": "What are the must-have features for the MVP?",
        "category": "core_functionality",
        "recommended_answer": "User auth, dashboard, and basic CRUD.",
    },
    {
        "question": "Should the app require authentication?",
        "category": "user_model",
        "recommended_answer": "Yes, email/password auth with OAuth optional.",
    },
    {
        "question": "What is the primary platform?",
        "category": "platform",
        "recommended_answer": "Web application (SPA).",
    },
]

SAMPLE_FEATURE_SPEC = {
    "app_name": "TestApp",
    "idea": "A task manager",
    "decisions": [],
    "core_features": ["task CRUD", "user auth"],
    "user_stories": [],
    "data_model": {"entities": []},
    "tech_constraints": ["React frontend"],
    "scope": {"mvp": ["tasks"], "out_of_scope": ["mobile"]},
    "design_direction": "minimal",
    "platform": "web",
}


def _make_state(project_dir: str, **overrides) -> dict:
    """Build a minimal KindleState dict for testing."""
    base = {
        "project_dir": project_dir,
        "idea": "A task manager for remote teams",
    }
    base.update(overrides)
    return base


def _make_auto_ui() -> MagicMock:
    """Return a UI mock that behaves like auto_approve=True.

    grill_question returns the recommended answer (2nd arg) by default.
    """
    ui = MagicMock()
    ui.auto_approve = True
    # Return the recommended answer (second positional arg)
    ui.grill_question = MagicMock(side_effect=lambda q, rec, cat, num: rec)
    return ui


def _setup_workspace(project_dir: Path) -> None:
    """Create the directory skeleton that artifacts.py expects."""
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)
    # metadata.json is required by mark_stage_complete
    meta = {
        "project_id": "kindle_test1234",
        "idea": "A task manager",
        "created_at": "2025-01-01T00:00:00+00:00",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def _make_agent_side_effect(workspace: Path, questions: list[dict], spec: dict):
    """Create an async side-effect that writes expected files, simulating run_agent."""

    call_count = 0

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        # First call: write open_questions.json
        if call_count == 1:
            (workspace / "open_questions.json").write_text(json.dumps(questions))
        # Second call: write feature_spec.json
        elif call_count == 2:
            (workspace / "feature_spec.json").write_text(json.dumps(spec))

    return _side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGrillNodeQuestionsPresented:
    """Verify that generated questions are loaded and shown to the user."""

    @pytest.mark.asyncio
    async def test_each_question_is_presented_via_ui(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        # Each question should have been presented
        assert ui.grill_question.call_count == len(SAMPLE_QUESTIONS)
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            ui.grill_question.assert_any_call(q["question"], q["recommended_answer"], q["category"], i)

    @pytest.mark.asyncio
    async def test_questions_presented_in_order(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        calls = ui.grill_question.call_args_list
        for i, (args, _kwargs) in enumerate(calls):
            # 4th arg (number) should be sequential starting at 1
            assert args[3] == i + 1
            assert args[0] == SAMPLE_QUESTIONS[i]["question"]


class TestGrillNodeEarlyExit:
    """Verify early-exit when the user answers 'done'."""

    @pytest.mark.asyncio
    async def test_done_stops_asking_remaining_questions(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        # First question: normal answer, second: "done"
        ui.grill_question = MagicMock(side_effect=["My custom answer", "done"])
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        # Only 2 questions should have been asked (1st answered, 2nd = "done")
        assert ui.grill_question.call_count == 2
        # The info message about early exit should be shown
        ui.info.assert_any_call("User requested early exit — using defaults for remaining questions.")

    @pytest.mark.asyncio
    async def test_done_fills_remaining_with_defaults(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        # Answer "done" on the very first question
        ui.grill_question = MagicMock(return_value="done")
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        # All 3 questions' recommended answers should appear in transcript
        for q in SAMPLE_QUESTIONS:
            assert q["recommended_answer"] in transcript
        # "auto-default" markers should appear for remaining questions
        assert "auto-default, user said done" in transcript
        assert "auto-default)" in transcript

    @pytest.mark.asyncio
    async def test_done_case_insensitive(self, tmp_path):
        """'DONE' or 'Done' should also trigger early exit."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        ui.grill_question = MagicMock(return_value="DONE")
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        # Only 1 question asked, then early exit
        assert ui.grill_question.call_count == 1
        ui.info.assert_any_call("User requested early exit — using defaults for remaining questions.")


class TestGrillNodeTranscript:
    """Verify that the Q&A transcript is built correctly."""

    @pytest.mark.asyncio
    async def test_transcript_contains_all_qa_pairs(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            assert f"Q{i} [{q['category']}]: {q['question']}" in transcript
            assert f"Recommended: {q['recommended_answer']}" in transcript
            assert f"Answer: {q['recommended_answer']}" in transcript

    @pytest.mark.asyncio
    async def test_transcript_reflects_custom_answers(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        custom_answers = [
            "Only task CRUD for now",
            "No auth for MVP",
            "Desktop app with Electron",
        ]
        ui = _make_auto_ui()
        ui.grill_question = MagicMock(side_effect=custom_answers)
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        for answer in custom_answers:
            assert f"Answer: {answer}" in transcript

    @pytest.mark.asyncio
    async def test_transcript_saved_as_artifact(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        artifact_path = project_dir / "artifacts" / "grill_transcript.md"
        assert artifact_path.exists()
        saved_transcript = artifact_path.read_text()
        assert saved_transcript == result["grill_transcript"]


class TestGrillNodeFeatureSpec:
    """Verify the compiled feature spec is saved and returned."""

    @pytest.mark.asyncio
    async def test_feature_spec_returned_in_state(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        assert result["feature_spec"] == SAMPLE_FEATURE_SPEC

    @pytest.mark.asyncio
    async def test_feature_spec_saved_as_artifact(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        artifact_path = project_dir / "artifacts" / "feature_spec.json"
        assert artifact_path.exists()
        saved_spec = json.loads(artifact_path.read_text())
        assert saved_spec == SAMPLE_FEATURE_SPEC

    @pytest.mark.asyncio
    async def test_empty_spec_when_compile_produces_invalid_json(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        call_count = 0

        async def _bad_compile(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            elif call_count == 2:
                (ws / "feature_spec.json").write_text("NOT VALID JSON{{{")

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _bad_compile
            result = await grill_node(state, ui)

        assert result["feature_spec"] == {}
        ui.error.assert_any_call("Failed to parse feature_spec.json.")


class TestGrillNodeStageLifecycle:
    """Verify stage start/done markers and completion tracking."""

    @pytest.mark.asyncio
    async def test_stage_marked_complete(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_stage_start_and_done_called(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        ui.stage_start.assert_called_once_with("grill")
        ui.stage_done.assert_called_once_with("grill")

    @pytest.mark.asyncio
    async def test_returns_current_stage(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        assert result["current_stage"] == "grill"


class TestGrillNodeAgentCalls:
    """Verify that run_agent is called with correct arguments."""

    @pytest.mark.asyncio
    async def test_interrogation_agent_called_with_idea(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        idea = "A task manager for remote teams"
        state = _make_state(str(project_dir), idea=idea)

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        # First call is the interrogation agent
        first_call_kwargs = mock_agent.call_args_list[0].kwargs
        assert first_call_kwargs["persona"] == "Principal Product Interrogator"
        assert first_call_kwargs["system_prompt"] == INTERROGATION_SYSTEM_PROMPT
        assert idea in first_call_kwargs["user_prompt"]
        assert first_call_kwargs["cwd"] == str(ws)
        assert first_call_kwargs["stage"] == "grill"

    @pytest.mark.asyncio
    async def test_compile_agent_called_with_transcript(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        # Second call is the compile agent
        assert mock_agent.call_count == 2
        second_call_kwargs = mock_agent.call_args_list[1].kwargs
        assert second_call_kwargs["persona"] == "Principal Product Architect"
        assert second_call_kwargs["system_prompt"] == COMPILE_SYSTEM_PROMPT
        assert second_call_kwargs["stage"] == "grill_compile"
        # The compile prompt must contain the Q&A transcript
        for q in SAMPLE_QUESTIONS:
            assert q["question"] in second_call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_stack_preference_included_in_prompts(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir), stack_preference="nextjs")

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        # Stack preference in interrogation prompt
        first_call = mock_agent.call_args_list[0].kwargs
        assert "nextjs" in first_call["user_prompt"]
        # Stack preference in compile prompt
        second_call = mock_agent.call_args_list[1].kwargs
        assert "nextjs" in second_call["user_prompt"]

    @pytest.mark.asyncio
    async def test_model_forwarded_to_agent(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir), model="claude-sonnet-4-20250514")

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        for call_args in mock_agent.call_args_list:
            assert call_args.kwargs["model"] == "claude-sonnet-4-20250514"


class TestGrillNodeEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_no_questions_generated(self, tmp_path):
        """When agent produces no questions, node proceeds without asking."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            # Agent produces no questions file at all
            mock_agent.side_effect = _make_agent_side_effect(ws, [], SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        ui.grill_question.assert_not_called()
        ui.info.assert_any_call("No questions generated — proceeding with idea as-is.")
        assert result["grill_transcript"] == ""

    @pytest.mark.asyncio
    async def test_questions_file_missing(self, tmp_path):
        """When agent doesn't write open_questions.json at all."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        call_count = 0

        async def _no_questions(**kwargs):
            nonlocal call_count
            call_count += 1
            # First call: don't write anything
            if call_count == 2:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _no_questions
            result = await grill_node(state, ui)

        ui.grill_question.assert_not_called()
        ui.info.assert_any_call("No questions generated — proceeding with idea as-is.")

    @pytest.mark.asyncio
    async def test_invalid_questions_json(self, tmp_path):
        """When open_questions.json is not valid JSON."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        call_count = 0

        async def _bad_questions(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                (ws / "open_questions.json").write_text("{invalid json!!")
            elif call_count == 2:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _bad_questions
            result = await grill_node(state, ui)

        ui.error.assert_any_call("Failed to parse open_questions.json")
        ui.grill_question.assert_not_called()

    @pytest.mark.asyncio
    async def test_questions_json_is_dict_not_list(self, tmp_path):
        """When open_questions.json contains a dict instead of a list."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        call_count = 0

        async def _dict_questions(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                (ws / "open_questions.json").write_text(json.dumps({"not": "a list"}))
            elif call_count == 2:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _dict_questions
            result = await grill_node(state, ui)

        # Dict is not a list, so open_questions should be empty
        ui.grill_question.assert_not_called()
        ui.info.assert_any_call("No questions generated — proceeding with idea as-is.")

    @pytest.mark.asyncio
    async def test_feature_spec_file_missing(self, tmp_path):
        """When the compile agent doesn't write feature_spec.json."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        call_count = 0

        async def _no_spec(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            # Second call: don't write feature_spec.json

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _no_spec
            result = await grill_node(state, ui)

        assert result["feature_spec"] == {}

    @pytest.mark.asyncio
    async def test_empty_idea_in_state(self, tmp_path):
        """Node should still work when idea is empty."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir), idea="")

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, [], SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        # Should not crash; first agent call should have idea=""
        first_call = mock_agent.call_args_list[0].kwargs
        assert "IDEA: " in first_call["user_prompt"]

    @pytest.mark.asyncio
    async def test_questions_with_missing_fields(self, tmp_path):
        """Questions without recommended_answer or category should use defaults."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        sparse_questions = [
            {"question": "What platform?"},  # no recommended_answer, no category
            "Just a string question",  # not even a dict
        ]

        ui = _make_auto_ui()
        # For string question, recommended will be "No recommendation"
        ui.grill_question = MagicMock(side_effect=["Web", "CLI"])
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, sparse_questions, SAMPLE_FEATURE_SPEC)
            result = await grill_node(state, ui)

        assert ui.grill_question.call_count == 2
        # First question: dict without recommended_answer
        first_call = ui.grill_question.call_args_list[0]
        assert first_call[0][0] == "What platform?"
        assert first_call[0][1] == "No recommendation"
        assert first_call[0][2] == "general"
        # Second question: bare string
        second_call = ui.grill_question.call_args_list[1]
        assert second_call[0][0] == "Just a string question"
        assert second_call[0][1] == "No recommendation"
        assert second_call[0][2] == "general"


class TestGrillNodeCleanup:
    """Verify that temporary workspace files are cleaned up."""

    @pytest.mark.asyncio
    async def test_workspace_files_removed_after_completion(self, tmp_path):
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _make_agent_side_effect(ws, SAMPLE_QUESTIONS, SAMPLE_FEATURE_SPEC)
            await grill_node(state, ui)

        # Temp files should be cleaned up
        assert not (ws / "open_questions.json").exists()
        assert not (ws / "feature_spec.json").exists()

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_files_gracefully(self, tmp_path):
        """Cleanup should not error if files were never created."""
        project_dir = tmp_path / "project"
        _setup_workspace(project_dir)
        ws = project_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        ui = _make_auto_ui()
        state = _make_state(str(project_dir))

        async def _noop(**kwargs):
            pass  # Agent writes nothing

        with (
            patch("kindle.stages.grill.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("kindle.stages.grill.workspace_path", return_value=ws),
        ):
            mock_agent.side_effect = _noop
            # Should not raise
            result = await grill_node(state, ui)

        assert result["feature_spec"] == {}
