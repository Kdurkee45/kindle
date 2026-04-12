"""Tests for kindle.stages.grill — structured interrogation to build a feature spec."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.stages.grill import grill_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    {
        "question": "What are the must-have features for the MVP?",
        "category": "core_functionality",
        "recommended_answer": "User login, dashboard, and CRUD for tasks",
    },
    {
        "question": "Is authentication required?",
        "category": "user_model",
        "recommended_answer": "Yes, email/password auth",
    },
    {
        "question": "What is the target platform?",
        "category": "platform",
        "recommended_answer": "Web application (SPA)",
    },
]

SAMPLE_FEATURE_SPEC = {
    "app_name": "TaskFlow",
    "idea": "a task management app",
    "decisions": [],
    "core_features": ["task CRUD", "auth"],
    "user_stories": [],
    "data_model": {"entities": []},
    "tech_constraints": ["React frontend"],
    "scope": {"mvp": ["tasks", "auth"], "out_of_scope": ["mobile"]},
    "design_direction": "minimal",
    "platform": "web",
}


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
        "stack_preference": "",
    }
    state.update(overrides)
    return state


def _make_ui(*, auto_approve: bool = False) -> MagicMock:
    """Return a mock UI with the methods grill_node actually calls."""
    ui = MagicMock()
    ui.auto_approve = auto_approve
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.grill_question = MagicMock()
    return ui


# ---------------------------------------------------------------------------
# Question generation phase
# ---------------------------------------------------------------------------


class TestQuestionGeneration:
    """Tests for the first phase — generating open_questions.json via the agent."""

    @pytest.mark.asyncio
    async def test_agent_called_with_idea(self, tmp_path: Path) -> None:
        """run_agent receives a prompt containing the user's idea."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_gen_agent(**kwargs):
            # Write empty questions so the rest of the node proceeds
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "open_questions.json").write_text("[]")
            return MagicMock()

        async def fake_compile_agent(**kwargs):
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        with patch("kindle.stages.grill.run_agent", new=AsyncMock(side_effect=[fake_gen_agent, fake_compile_agent])):
            # run_agent is called, but side_effect returns coroutines
            # We need run_agent to be awaitable, so fix approach
            pass

        # Re-approach: use a single AsyncMock with side_effect list
        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1
            return MagicMock()

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        # First call is question generation
        first_call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "a task management app" in first_call_kwargs["user_prompt"]
        assert first_call_kwargs["stage"] == "grill"

    @pytest.mark.asyncio
    async def test_agent_prompt_includes_stack_preference(self, tmp_path: Path) -> None:
        """When stack_preference is set, it appears in the generation prompt."""
        state = _make_state(tmp_path, stack_preference="Python + React")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        first_call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "Python + React" in first_call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_questions_loaded_from_json(self, tmp_path: Path) -> None:
        """Questions generated by agent are loaded and presented to user."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "accepted"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        # Each question should be presented exactly once
        assert ui.grill_question.call_count == len(SAMPLE_QUESTIONS)

    @pytest.mark.asyncio
    async def test_no_questions_file_skips_grill(self, tmp_path: Path) -> None:
        """If the agent fails to create open_questions.json, proceed gracefully."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                pass  # Don't create open_questions.json
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        ui.info.assert_any_call("No questions generated — proceeding with idea as-is.")
        ui.grill_question.assert_not_called()
        assert result["grill_transcript"] == ""

    @pytest.mark.asyncio
    async def test_malformed_json_shows_error(self, tmp_path: Path) -> None:
        """Invalid JSON in open_questions.json triggers error and empty question list."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("{not valid json!!")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        ui.error.assert_any_call("Failed to parse open_questions.json")
        ui.grill_question.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_list_json_treated_as_empty(self, tmp_path: Path) -> None:
        """If open_questions.json contains a dict instead of a list, treat as empty."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text('{"not": "a list"}')
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        ui.info.assert_any_call("No questions generated — proceeding with idea as-is.")
        ui.grill_question.assert_not_called()


# ---------------------------------------------------------------------------
# Question walk-through
# ---------------------------------------------------------------------------


class TestQuestionWalkThrough:
    """Tests for the interactive Q&A loop that builds the transcript."""

    @pytest.mark.asyncio
    async def test_transcript_records_all_answers(self, tmp_path: Path) -> None:
        """Every Q&A pair is captured in the transcript with correct formatting."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.side_effect = ["my custom answer", "another answer", "third answer"]
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        # Verify each question and answer appears
        assert "Q1 [core_functionality]:" in transcript
        assert "my custom answer" in transcript
        assert "Q2 [user_model]:" in transcript
        assert "another answer" in transcript
        assert "Q3 [platform]:" in transcript
        assert "third answer" in transcript

    @pytest.mark.asyncio
    async def test_grill_question_called_with_correct_args(self, tmp_path: Path) -> None:
        """grill_question receives question text, recommended answer, category, and number."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS[:1]))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.grill_question.assert_called_once_with(
            "What are the must-have features for the MVP?",
            "User login, dashboard, and CRUD for tasks",
            "core_functionality",
            1,
        )

    @pytest.mark.asyncio
    async def test_transcript_includes_recommended_answers(self, tmp_path: Path) -> None:
        """The transcript records both the recommended answer and the actual answer."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "custom answer"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS[:1]))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        assert "Recommended: User login, dashboard, and CRUD for tasks" in transcript
        assert "Answer: custom answer" in transcript

    @pytest.mark.asyncio
    async def test_decisions_list_passed_to_compile(self, tmp_path: Path) -> None:
        """The compile prompt includes the full grill transcript."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.side_effect = ["answer1", "answer2", "answer3"]
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        # Second call is the compile phase
        compile_kwargs = mock_agent.call_args_list[1].kwargs
        assert "GRILL TRANSCRIPT:" in compile_kwargs["user_prompt"]
        assert "answer1" in compile_kwargs["user_prompt"]
        assert "answer2" in compile_kwargs["user_prompt"]
        assert "answer3" in compile_kwargs["user_prompt"]
        assert compile_kwargs["stage"] == "grill_compile"


# ---------------------------------------------------------------------------
# Auto-approve mode
# ---------------------------------------------------------------------------


class TestAutoApprove:
    """When auto_approve is True, recommended answers are used without prompting."""

    @pytest.mark.asyncio
    async def test_auto_approve_uses_recommended_answers(self, tmp_path: Path) -> None:
        """In auto-approve, UI.grill_question still gets called (UI returns recommended)."""
        state = _make_state(tmp_path)
        ui = _make_ui(auto_approve=True)
        # Simulate the real UI behavior: auto_approve returns the recommended answer
        ui.grill_question.side_effect = lambda q, rec, cat, num: rec
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        # All recommended answers should be in the transcript
        for q in SAMPLE_QUESTIONS:
            assert f"Answer: {q['recommended_answer']}" in transcript

    @pytest.mark.asyncio
    async def test_auto_approve_all_questions_answered(self, tmp_path: Path) -> None:
        """All questions should be answered when in auto-approve mode."""
        state = _make_state(tmp_path)
        ui = _make_ui(auto_approve=True)
        ui.grill_question.side_effect = lambda q, rec, cat, num: rec
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        assert ui.grill_question.call_count == len(SAMPLE_QUESTIONS)


# ---------------------------------------------------------------------------
# 'done' early exit
# ---------------------------------------------------------------------------


class TestDoneEarlyExit:
    """When the user types 'done', remaining questions get their default answers."""

    @pytest.mark.asyncio
    async def test_done_fills_remaining_with_defaults(self, tmp_path: Path) -> None:
        """Typing 'done' on Q1 auto-fills Q2 and Q3 with recommended answers."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        # User answers 'done' on the first question
        ui.grill_question.return_value = "done"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]

        # Q1 should be answered with its recommended (user said done)
        assert "(auto-default, user said done)" in transcript

        # Q2, Q3 should get auto-default treatment
        assert "Q2 [user_model]: Is authentication required?" in transcript
        assert "Answer: Yes, email/password auth (auto-default)" in transcript
        assert "Q3 [platform]: What is the target platform?" in transcript
        assert "Answer: Web application (SPA) (auto-default)" in transcript

    @pytest.mark.asyncio
    async def test_done_only_asks_one_question(self, tmp_path: Path) -> None:
        """After 'done', no further grill_question calls are made."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "done"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        # Only called once (for Q1), remaining filled automatically
        assert ui.grill_question.call_count == 1

    @pytest.mark.asyncio
    async def test_done_on_second_question(self, tmp_path: Path) -> None:
        """Typing 'done' on Q2 keeps Q1's answer and auto-fills Q3."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.side_effect = ["my custom answer", "done"]
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]

        # Q1 has custom answer
        assert "Answer: my custom answer" in transcript
        # Q2 was where user said done — gets auto-default, user said done
        assert "(auto-default, user said done)" in transcript
        # Q3 gets plain auto-default
        assert "Answer: Web application (SPA) (auto-default)" in transcript
        # Only 2 grill_question calls
        assert ui.grill_question.call_count == 2

    @pytest.mark.asyncio
    async def test_done_case_insensitive(self, tmp_path: Path) -> None:
        """'Done', 'DONE', 'done' all trigger early exit."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "Done"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        # Should have triggered early exit info message
        ui.info.assert_any_call("User requested early exit — using defaults for remaining questions.")
        assert ui.grill_question.call_count == 1

    @pytest.mark.asyncio
    async def test_done_on_last_question(self, tmp_path: Path) -> None:
        """Typing 'done' on the very last question still records it with auto-default."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        # Answer first two normally, then 'done' on last
        ui.grill_question.side_effect = ["ans1", "ans2", "done"]
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        assert "Answer: ans1" in transcript
        assert "Answer: ans2" in transcript
        # Q3 answered with recommended (auto-default, user said done)
        assert "(auto-default, user said done)" in transcript
        # No remaining questions after Q3
        assert "Q4" not in transcript


# ---------------------------------------------------------------------------
# Spec compilation phase
# ---------------------------------------------------------------------------


class TestSpecCompilation:
    """Tests for the second phase — compiling decisions into feature_spec.json."""

    @pytest.mark.asyncio
    async def test_feature_spec_parsed_and_returned(self, tmp_path: Path) -> None:
        """Feature spec JSON is parsed and returned in state."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "accepted"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS[:1]))
            else:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert result["feature_spec"] == SAMPLE_FEATURE_SPEC
        assert result["feature_spec"]["app_name"] == "TaskFlow"
        assert "task CRUD" in result["feature_spec"]["core_features"]

    @pytest.mark.asyncio
    async def test_compile_prompt_includes_idea_and_stack(self, tmp_path: Path) -> None:
        """The compilation prompt includes the idea and stack preference."""
        state = _make_state(tmp_path, stack_preference="Django + Vue")
        ui = _make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS[:1]))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        compile_kwargs = mock_agent.call_args_list[1].kwargs
        assert "a task management app" in compile_kwargs["user_prompt"]
        assert "Django + Vue" in compile_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_malformed_feature_spec_returns_empty_dict(self, tmp_path: Path) -> None:
        """If feature_spec.json is invalid JSON, return an empty dict."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{{broken json}}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        ui.error.assert_any_call("Failed to parse feature_spec.json.")
        assert result["feature_spec"] == {}

    @pytest.mark.asyncio
    async def test_missing_feature_spec_returns_empty_dict(self, tmp_path: Path) -> None:
        """If agent fails to create feature_spec.json, return an empty dict."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                pass  # Don't create feature_spec.json
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert result["feature_spec"] == {}

    @pytest.mark.asyncio
    async def test_no_stack_preference_shows_choose_best_fit(self, tmp_path: Path) -> None:
        """When stack_preference is empty, compile prompt says 'choose the best fit'."""
        state = _make_state(tmp_path, stack_preference="")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        compile_kwargs = mock_agent.call_args_list[1].kwargs
        assert "None — choose the best fit" in compile_kwargs["user_prompt"]


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


class TestArtifactSaving:
    """Tests for save_artifact calls — grill_transcript.md and feature_spec.json."""

    @pytest.mark.asyncio
    async def test_grill_transcript_saved(self, tmp_path: Path) -> None:
        """grill_transcript.md is saved as an artifact."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "answer"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(SAMPLE_QUESTIONS[:1]))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        # Check artifact was actually written to disk
        transcript_path = Path(state["project_dir"]) / "artifacts" / "grill_transcript.md"
        assert transcript_path.exists()
        content = transcript_path.read_text()
        assert "Q1 [core_functionality]:" in content
        assert "Answer: answer" in content

    @pytest.mark.asyncio
    async def test_feature_spec_json_saved(self, tmp_path: Path) -> None:
        """feature_spec.json is saved as a formatted artifact."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text(json.dumps(SAMPLE_FEATURE_SPEC))
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        spec_path = Path(state["project_dir"]) / "artifacts" / "feature_spec.json"
        assert spec_path.exists()
        saved_spec = json.loads(spec_path.read_text())
        assert saved_spec == SAMPLE_FEATURE_SPEC

    @pytest.mark.asyncio
    async def test_empty_spec_saved_as_empty_dict(self, tmp_path: Path) -> None:
        """When feature_spec.json is missing, an empty dict is saved."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                pass  # No feature_spec.json
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        spec_path = Path(state["project_dir"]) / "artifacts" / "feature_spec.json"
        assert spec_path.exists()
        assert json.loads(spec_path.read_text()) == {}

    @pytest.mark.asyncio
    async def test_temp_files_cleaned_up(self, tmp_path: Path) -> None:
        """open_questions.json and feature_spec.json are removed from workspace."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        assert not (ws / "open_questions.json").exists()
        assert not (ws / "feature_spec.json").exists()


# ---------------------------------------------------------------------------
# State return shape
# ---------------------------------------------------------------------------


class TestStateReturn:
    """Tests for the returned state dictionary."""

    @pytest.mark.asyncio
    async def test_returns_required_keys(self, tmp_path: Path) -> None:
        """Return dict must contain feature_spec, grill_transcript, current_stage."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert "feature_spec" in result
        assert "grill_transcript" in result
        assert "current_stage" in result

    @pytest.mark.asyncio
    async def test_current_stage_is_grill(self, tmp_path: Path) -> None:
        """current_stage is always 'grill' after this node runs."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert result["current_stage"] == "grill"

    @pytest.mark.asyncio
    async def test_feature_spec_is_dict(self, tmp_path: Path) -> None:
        """feature_spec is always a dict, even on parse failure."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("not json")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert isinstance(result["feature_spec"], dict)

    @pytest.mark.asyncio
    async def test_grill_transcript_is_string(self, tmp_path: Path) -> None:
        """grill_transcript is always a string."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert isinstance(result["grill_transcript"], str)

    @pytest.mark.asyncio
    async def test_only_three_keys_returned(self, tmp_path: Path) -> None:
        """Return dict has exactly three keys — nothing extra leaking."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert set(result.keys()) == {"feature_spec", "grill_transcript", "current_stage"}


# ---------------------------------------------------------------------------
# Stage lifecycle
# ---------------------------------------------------------------------------


class TestStageLifecycle:
    """Tests for UI stage lifecycle calls and mark_stage_complete."""

    @pytest.mark.asyncio
    async def test_stage_start_and_done_called(self, tmp_path: Path) -> None:
        """ui.stage_start('grill') and ui.stage_done('grill') bracket the node."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.stage_start.assert_called_once_with("grill")
        ui.stage_done.assert_called_once_with("grill")

    @pytest.mark.asyncio
    async def test_mark_stage_complete_called(self, tmp_path: Path) -> None:
        """mark_stage_complete updates metadata.json with 'grill'."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        meta = json.loads(Path(state["project_dir"], "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_run_agent_called_twice(self, tmp_path: Path) -> None:
        """run_agent is called exactly twice: once for generation, once for compilation."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        assert mock_agent.call_count == 2

    @pytest.mark.asyncio
    async def test_compile_info_message_shown(self, tmp_path: Path) -> None:
        """User sees an info message about spec compilation."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.info.assert_any_call("Compiling feature specification from decisions...")


# ---------------------------------------------------------------------------
# Edge cases: non-dict question items
# ---------------------------------------------------------------------------


class TestEdgeCaseQuestionFormats:
    """Questions might be strings instead of dicts; verify graceful handling."""

    @pytest.mark.asyncio
    async def test_plain_string_questions(self, tmp_path: Path) -> None:
        """If questions are plain strings (not dicts), they still work."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "my answer"
        ws = Path(state["project_dir"]) / "workspace"

        plain_questions = ["What features?", "What platform?"]

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(plain_questions))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert ui.grill_question.call_count == 2
        # For plain strings, recommended is "No recommendation" and category is "general"
        first_call = ui.grill_question.call_args_list[0]
        assert first_call[0] == ("What features?", "No recommendation", "general", 1)

    @pytest.mark.asyncio
    async def test_dict_without_recommended_answer(self, tmp_path: Path) -> None:
        """Questions with missing recommended_answer default to 'No recommendation'."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "sure"
        ws = Path(state["project_dir"]) / "workspace"

        questions = [{"question": "Auth needed?", "category": "user_model"}]

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(questions))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        ui.grill_question.assert_called_once_with("Auth needed?", "No recommendation", "user_model", 1)

    @pytest.mark.asyncio
    async def test_dict_without_category(self, tmp_path: Path) -> None:
        """Questions with missing category default to 'general'."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        questions = [{"question": "Platform?", "recommended_answer": "Web"}]

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(questions))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.grill_question.assert_called_once_with("Platform?", "Web", "general", 1)

    @pytest.mark.asyncio
    async def test_done_with_plain_string_remaining(self, tmp_path: Path) -> None:
        """'done' early exit works even when remaining questions are plain strings."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        mixed_questions = [
            {"question": "Features?", "category": "core", "recommended_answer": "CRUD"},
            "What database?",
        ]
        ui.grill_question.return_value = "done"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text(json.dumps(mixed_questions))
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        # Q1 done -> auto-default
        assert "(auto-default, user said done)" in transcript
        # Q2 is plain string -> auto-default with "No recommendation"
        assert "What database?" in transcript
        assert "No recommendation (auto-default)" in transcript
        assert ui.grill_question.call_count == 1

    @pytest.mark.asyncio
    async def test_model_passed_through_to_agent(self, tmp_path: Path) -> None:
        """state['model'] is forwarded to both run_agent calls."""
        state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def fake_run_agent(**kwargs):
            nonlocal call_count
            ws.mkdir(parents=True, exist_ok=True)
            if call_count == 0:
                (ws / "open_questions.json").write_text("[]")
            else:
                (ws / "feature_spec.json").write_text("{}")
            call_count += 1

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        for c in mock_agent.call_args_list:
            assert c.kwargs["model"] == "claude-sonnet-4-20250514"
