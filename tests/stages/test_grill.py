"""Tests for kindle.stages.grill — adaptive conversational interrogation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.stages.grill import (
    _build_history_prompt,
    _parse_agent_response,
    grill_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRILL_FEATURE_SPEC = {
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

# Standard question responses the mock agent returns
QUESTION_RESPONSES = [
    {
        "status": "question",
        "question": "What are the must-have features for the MVP?",
        "category": "core_functionality",
        "recommended_answer": "User login, dashboard, and CRUD for tasks",
        "why_asking": "Need to scope the core build",
    },
    {
        "status": "question",
        "question": "Is authentication required?",
        "category": "user_model",
        "recommended_answer": "Yes, email/password auth",
        "why_asking": "Determines user model complexity",
    },
    {
        "status": "question",
        "question": "What is the target platform?",
        "category": "platform",
        "recommended_answer": "Web application (SPA)",
        "why_asking": "Affects tech stack decisions",
    },
]

DONE_RESPONSE = {
    "status": "done",
    "summary": "Building a task management web app with auth and CRUD",
    "assumptions": ["Email/password auth", "PostgreSQL database"],
    "confidence": "high",
}


def _make_agent_result(response: dict) -> MagicMock:
    """Create a MagicMock that mimics an agent result with a .text attribute."""
    result = MagicMock()
    result.text = json.dumps(response)
    return result


def _make_conversation_side_effect(
    questions: list[dict],
    done: dict,
    feature_spec: dict | None = None,
    ws: Path | None = None,
):
    """Build a side_effect function for run_agent that simulates a conversation.

    Returns question responses for the first N calls, then a done response,
    then handles the compile call by writing feature_spec.json.
    """
    call_count = 0
    total_questions = len(questions)

    async def side_effect(**kwargs):
        nonlocal call_count
        idx = call_count
        call_count += 1

        # Conversation turns: questions, then done
        if idx < total_questions:
            return _make_agent_result(questions[idx])
        if idx == total_questions:
            return _make_agent_result(done)

        # Compile phase — write feature_spec.json
        if ws is not None:
            ws.mkdir(parents=True, exist_ok=True)
            spec = feature_spec if feature_spec is not None else {}
            (ws / "feature_spec.json").write_text(json.dumps(spec))
        return MagicMock()

    return side_effect


@pytest.fixture
def grill_state(make_state):
    """Factory with grill-stage defaults pre-applied."""

    def _factory(**overrides):
        defaults = {"stack_preference": ""}
        defaults.update(overrides)
        return make_state(**defaults)

    return _factory


# ---------------------------------------------------------------------------
# _parse_agent_response unit tests
# ---------------------------------------------------------------------------


class TestParseAgentResponse:
    """Tests for the JSON extraction helper."""

    def test_parses_clean_json(self) -> None:
        result = _parse_agent_response('{"status": "done", "summary": "ok"}')
        assert result["status"] == "done"

    def test_parses_json_in_code_fence(self) -> None:
        text = '```json\n{"status": "question", "question": "What?"}\n```'
        result = _parse_agent_response(text)
        assert result["status"] == "question"

    def test_parses_json_with_preamble(self) -> None:
        text = 'Here is my response:\n{"status": "done", "summary": "all good"}'
        result = _parse_agent_response(text)
        assert result["status"] == "done"

    def test_returns_error_for_garbage(self) -> None:
        result = _parse_agent_response("not json at all")
        assert result["status"] == "error"

    def test_handles_whitespace(self) -> None:
        result = _parse_agent_response('  \n  {"status": "done"}  \n  ')
        assert result["status"] == "done"


# ---------------------------------------------------------------------------
# _build_history_prompt unit tests
# ---------------------------------------------------------------------------


class TestBuildHistoryPrompt:
    """Tests for conversation prompt construction."""

    def test_includes_idea(self) -> None:
        prompt = _build_history_prompt("a todo app", "", [])
        assert "a todo app" in prompt

    def test_includes_stack_preference(self) -> None:
        prompt = _build_history_prompt("app", "Python + React", [])
        assert "Python + React" in prompt

    def test_omits_stack_when_empty(self) -> None:
        prompt = _build_history_prompt("app", "", [])
        assert "STACK PREFERENCE" not in prompt

    def test_includes_history(self) -> None:
        history = [
            {
                "role": "agent",
                "data": {
                    "question": "What features?",
                    "category": "core",
                    "recommended_answer": "CRUD",
                    "why_asking": "Need scope",
                },
                "turn": 1,
            },
            {"role": "user", "answer": "Just CRUD and auth"},
        ]
        prompt = _build_history_prompt("app", "", history)
        assert "What features?" in prompt
        assert "Just CRUD and auth" in prompt


# ---------------------------------------------------------------------------
# Conversation loop — agent question flow
# ---------------------------------------------------------------------------


class TestConversationLoop:
    """Tests for the adaptive conversation loop in grill_node."""

    @pytest.mark.asyncio
    async def test_agent_called_with_idea_in_prompt(self, tmp_path: Path, grill_state, make_ui) -> None:
        """The first run_agent call receives the user's idea in the prompt."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "accepted"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        first_call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "a task management app" in first_call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_agent_prompt_includes_stack_preference(self, tmp_path: Path, grill_state, make_ui) -> None:
        """When stack_preference is set, it appears in the conversation prompt."""
        state = grill_state(stack_preference="Python + React")
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        first_call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "Python + React" in first_call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_questions_presented_to_user(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Each question from the agent is presented via ui.grill_question."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "accepted"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES, DONE_RESPONSE, GRILL_FEATURE_SPEC, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        assert ui.grill_question.call_count == len(QUESTION_RESPONSES)

    @pytest.mark.asyncio
    async def test_grill_question_called_with_correct_kwargs(self, tmp_path: Path, grill_state, make_ui) -> None:
        """grill_question receives question, recommended, category, number, and why_asking."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        q = QUESTION_RESPONSES[0]
        ui.grill_question.assert_called_once_with(
            question=q["question"],
            recommended=q["recommended_answer"],
            category=q["category"],
            number=1,
            why_asking=q["why_asking"],
        )

    @pytest.mark.asyncio
    async def test_agent_stops_on_done_status(self, tmp_path: Path, grill_state, make_ui) -> None:
        """When agent returns status=done, no more questions are asked."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "yes"
        ws = Path(state["project_dir"]) / "workspace"

        # 2 questions then done
        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:2], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        assert ui.grill_question.call_count == 2

    @pytest.mark.asyncio
    async def test_error_response_breaks_loop(self, tmp_path: Path, grill_state, make_ui) -> None:
        """If agent returns unparseable response, loop exits gracefully."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                result = MagicMock()
                result.text = "totally not json {{{{"
                return result
            # compile call
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        ui.error.assert_any_call("Grill agent returned unparseable response on turn 1. Wrapping up.")
        ui.grill_question.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_question_breaks_loop(self, tmp_path: Path, grill_state, make_ui) -> None:
        """If agent returns a question with empty text, loop exits."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        empty_q = {
            "status": "question",
            "question": "",
            "category": "general",
            "recommended_answer": "something",
            "why_asking": "",
        }

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(empty_q)
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.error.assert_any_call("Grill agent returned empty question on turn 1. Wrapping up.")
        ui.grill_question.assert_not_called()


# ---------------------------------------------------------------------------
# Transcript recording
# ---------------------------------------------------------------------------


class TestTranscriptRecording:
    """Tests for the grill transcript output."""

    @pytest.mark.asyncio
    async def test_transcript_records_all_answers(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Every Q&A pair is captured in the transcript."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.side_effect = [
            "my custom answer",
            "another answer",
            "third answer",
        ]
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES, DONE_RESPONSE, GRILL_FEATURE_SPEC, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        assert "Q1 [core_functionality]" in transcript
        assert "my custom answer" in transcript
        assert "Q2 [user_model]" in transcript
        assert "another answer" in transcript
        assert "Q3 [platform]" in transcript
        assert "third answer" in transcript

    @pytest.mark.asyncio
    async def test_transcript_includes_recommended_answers(self, tmp_path: Path, grill_state, make_ui) -> None:
        """The transcript records the recommended answer for each question."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "custom answer"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        assert "Recommended: User login, dashboard, and CRUD for tasks" in transcript
        assert "Answer:** custom answer" in transcript

    @pytest.mark.asyncio
    async def test_transcript_includes_why_asking(self, tmp_path: Path, grill_state, make_ui) -> None:
        """The transcript records why the agent asked each question."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        assert "Need to scope the core build" in transcript

    @pytest.mark.asyncio
    async def test_transcript_records_done_summary(self, tmp_path: Path, grill_state, make_ui) -> None:
        """When agent says done, the summary and confidence are recorded."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        assert "Agent concluded after 1 questions" in transcript
        assert DONE_RESPONSE["summary"] in transcript
        assert "high" in transcript

    @pytest.mark.asyncio
    async def test_transcript_records_done_assumptions(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Done assumptions appear in the transcript."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        for assumption in DONE_RESPONSE["assumptions"]:
            assert assumption in transcript


# ---------------------------------------------------------------------------
# Compile phase
# ---------------------------------------------------------------------------


class TestCompilePhase:
    """Tests for the compilation of decisions into feature_spec.json."""

    @pytest.mark.asyncio
    async def test_compile_prompt_includes_transcript(self, tmp_path: Path, grill_state, make_ui) -> None:
        """The compile agent call includes the full conversation transcript."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.side_effect = ["answer1", "answer2", "answer3"]
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES, DONE_RESPONSE, GRILL_FEATURE_SPEC, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        # Last call is the compile phase
        compile_kwargs = mock_agent.call_args_list[-1].kwargs
        assert "CONVERSATION TRANSCRIPT" in compile_kwargs["user_prompt"]
        assert "answer1" in compile_kwargs["user_prompt"]
        assert compile_kwargs["stage"] == "grill_compile"

    @pytest.mark.asyncio
    async def test_compile_prompt_includes_idea_and_stack(self, tmp_path: Path, grill_state, make_ui) -> None:
        """The compilation prompt includes the idea and stack preference."""
        state = grill_state(stack_preference="Django + Vue")
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        compile_kwargs = mock_agent.call_args_list[-1].kwargs
        assert "a task management app" in compile_kwargs["user_prompt"]
        assert "Django + Vue" in compile_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_feature_spec_parsed_and_returned(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Feature spec JSON is parsed and returned in state."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "accepted"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, GRILL_FEATURE_SPEC, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert result["feature_spec"] == GRILL_FEATURE_SPEC
        assert result["feature_spec"]["app_name"] == "TaskFlow"
        assert "task CRUD" in result["feature_spec"]["core_features"]

    @pytest.mark.asyncio
    async def test_malformed_feature_spec_returns_empty_dict(self, tmp_path: Path, grill_state, make_ui) -> None:
        """If feature_spec.json is invalid JSON, return an empty dict."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(DONE_RESPONSE)
            # Compile — write broken JSON
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{{broken json}}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        ui.error.assert_any_call("Failed to parse feature_spec.json.")
        assert result["feature_spec"] == {}

    @pytest.mark.asyncio
    async def test_missing_feature_spec_returns_empty_dict(self, tmp_path: Path, grill_state, make_ui) -> None:
        """If agent fails to create feature_spec.json, return an empty dict."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(DONE_RESPONSE)
            # Compile — don't write anything
            ws.mkdir(parents=True, exist_ok=True)
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert result["feature_spec"] == {}

    @pytest.mark.asyncio
    async def test_no_stack_preference_shows_choose_best_fit(self, tmp_path: Path, grill_state, make_ui) -> None:
        """When stack_preference is empty, compile prompt says 'choose the best fit'."""
        state = grill_state(stack_preference="")
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        compile_kwargs = mock_agent.call_args_list[-1].kwargs
        assert "None — choose the best fit" in compile_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_compile_info_message_shown(self, tmp_path: Path, grill_state, make_ui) -> None:
        """User sees an info message about spec compilation."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.info.assert_any_call("Compiling feature specification from conversation...")


# ---------------------------------------------------------------------------
# Auto-approve mode
# ---------------------------------------------------------------------------


class TestAutoApprove:
    """When auto_approve is True, recommended answers are used without prompting."""

    @pytest.mark.asyncio
    async def test_auto_approve_uses_recommended_answers(self, tmp_path: Path, grill_state, make_ui) -> None:
        """In auto-approve, UI.grill_question still gets called (UI returns recommended)."""
        state = grill_state()
        ui = make_ui(auto_approve=True)
        # Simulate the real UI behavior: auto_approve returns the recommended answer
        ui.grill_question.side_effect = lambda **kwargs: kwargs["recommended"]
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES, DONE_RESPONSE, GRILL_FEATURE_SPEC, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        # All recommended answers should be in the transcript
        for q in QUESTION_RESPONSES:
            assert f"Answer:** {q['recommended_answer']}" in transcript

    @pytest.mark.asyncio
    async def test_auto_approve_all_questions_answered(self, tmp_path: Path, grill_state, make_ui) -> None:
        """All questions should be answered when in auto-approve mode."""
        state = grill_state()
        ui = make_ui(auto_approve=True)
        ui.grill_question.side_effect = lambda **kwargs: kwargs["recommended"]
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES, DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        assert ui.grill_question.call_count == len(QUESTION_RESPONSES)


# ---------------------------------------------------------------------------
# 'done' early exit
# ---------------------------------------------------------------------------


class TestDoneEarlyExit:
    """When the user types 'done', agent wraps up with assumptions."""

    @pytest.mark.asyncio
    async def test_done_triggers_wrap_up(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Typing 'done' on Q1 triggers a wrap-up agent call."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "done"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                # First question
                return _make_agent_result(QUESTION_RESPONSES[0])
            if idx == 1:
                # Wrap-up call after user says "done"
                return _make_agent_result(DONE_RESPONSE)
            # Compile
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text(json.dumps(GRILL_FEATURE_SPEC))
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        ui.info.assert_any_call("User requested early exit — agent will fill remaining gaps with assumptions.")
        # Only one grill_question call (Q1)
        assert ui.grill_question.call_count == 1

    @pytest.mark.asyncio
    async def test_done_records_assumptions_in_transcript(self, tmp_path: Path, grill_state, make_ui) -> None:
        """After user says 'done', assumptions from the wrap-up are in transcript."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "done"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(QUESTION_RESPONSES[0])
            if idx == 1:
                return _make_agent_result(DONE_RESPONSE)
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        assert "User ended conversation at Q1" in transcript
        for assumption in DONE_RESPONSE["assumptions"]:
            assert assumption in transcript

    @pytest.mark.asyncio
    async def test_done_on_second_question(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Typing 'done' on Q2 keeps Q1's answer and triggers wrap-up."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.side_effect = ["my custom answer", "done"]
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(QUESTION_RESPONSES[0])
            if idx == 1:
                return _make_agent_result(QUESTION_RESPONSES[1])
            if idx == 2:
                # Wrap-up after done on Q2
                return _make_agent_result(DONE_RESPONSE)
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        assert "my custom answer" in transcript
        assert "User ended conversation at Q2" in transcript
        assert ui.grill_question.call_count == 2

    @pytest.mark.asyncio
    async def test_done_case_insensitive(self, tmp_path: Path, grill_state, make_ui) -> None:
        """'Done', 'DONE', 'done' all trigger early exit."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "Done"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(QUESTION_RESPONSES[0])
            if idx == 1:
                return _make_agent_result(DONE_RESPONSE)
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.info.assert_any_call("User requested early exit — agent will fill remaining gaps with assumptions.")
        assert ui.grill_question.call_count == 1

    @pytest.mark.asyncio
    async def test_done_wrap_up_returns_non_done_status(self, tmp_path: Path, grill_state, make_ui) -> None:
        """When wrap-up agent returns a question instead of 'done', no crash occurs."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "done"
        ws = Path(state["project_dir"]) / "workspace"

        non_done_wrap_up = {
            "status": "question",
            "question": "One more thing...",
            "category": "scope",
            "recommended_answer": "Yes",
            "why_asking": "Just checking",
        }

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(QUESTION_RESPONSES[0])
            if idx == 1:
                # Wrap-up returns a question instead of done
                return _make_agent_result(non_done_wrap_up)
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        # Should not crash and should still produce a result
        assert "feature_spec" in result
        assert "grill_transcript" in result
        transcript = result["grill_transcript"]
        assert "User ended conversation at Q1" in transcript

    @pytest.mark.asyncio
    async def test_done_records_triggering_question_in_transcript(self, tmp_path: Path, grill_state, make_ui) -> None:
        """When user says 'done', the triggering question is recorded in transcript."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "done"
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(QUESTION_RESPONSES[0])
            if idx == 1:
                return _make_agent_result(DONE_RESPONSE)
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        transcript = result["grill_transcript"]
        # The triggering question should appear in transcript
        assert QUESTION_RESPONSES[0]["question"] in transcript
        assert "(user ended early)" in transcript


# ---------------------------------------------------------------------------
# MAX_QUESTIONS boundary
# ---------------------------------------------------------------------------


class TestMaxQuestionsBoundary:
    """Tests for behavior when the agent asks all MAX_QUESTIONS without saying done."""

    @pytest.mark.asyncio
    async def test_max_questions_exhausted(self, tmp_path: Path, grill_state, make_ui) -> None:
        """When agent never says 'done', loop exits after MAX_QUESTIONS."""
        from kindle.stages.grill import MAX_QUESTIONS

        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        # Agent always returns a question, never "done"
        endless_question = {
            "status": "question",
            "question": "Tell me more?",
            "category": "scope",
            "recommended_answer": "More details",
            "why_asking": "Need to understand better",
        }

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx < MAX_QUESTIONS:
                return _make_agent_result(endless_question)
            # Compile phase
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("{}")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        # Should complete without error
        assert "feature_spec" in result
        # Should have asked MAX_QUESTIONS questions
        assert ui.grill_question.call_count == MAX_QUESTIONS
        # Transcript should note the boundary
        assert f"Reached maximum {MAX_QUESTIONS} questions" in result["grill_transcript"]
        # Info message should be shown
        ui.info.assert_any_call(f"Reached maximum {MAX_QUESTIONS} questions. Compiling spec.")


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


class TestArtifactSaving:
    """Tests for save_artifact calls — grill_transcript.md and feature_spec.json."""

    @pytest.mark.asyncio
    async def test_grill_transcript_saved(self, tmp_path: Path, grill_state, make_ui) -> None:
        """grill_transcript.md is saved as an artifact."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "answer"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        transcript_path = Path(state["project_dir"]) / "artifacts" / "grill_transcript.md"
        assert transcript_path.exists()
        content = transcript_path.read_text()
        assert "Q1 [core_functionality]" in content
        assert "answer" in content

    @pytest.mark.asyncio
    async def test_feature_spec_json_saved(self, tmp_path: Path, grill_state, make_ui) -> None:
        """feature_spec.json is saved as a formatted artifact."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, GRILL_FEATURE_SPEC, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        spec_path = Path(state["project_dir"]) / "artifacts" / "feature_spec.json"
        assert spec_path.exists()
        saved_spec = json.loads(spec_path.read_text())
        assert saved_spec == GRILL_FEATURE_SPEC

    @pytest.mark.asyncio
    async def test_empty_spec_saved_as_empty_dict(self, tmp_path: Path, grill_state, make_ui) -> None:
        """When feature_spec.json is missing, an empty dict is saved."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(DONE_RESPONSE)
            ws.mkdir(parents=True, exist_ok=True)
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        spec_path = Path(state["project_dir"]) / "artifacts" / "feature_spec.json"
        assert spec_path.exists()
        assert json.loads(spec_path.read_text()) == {}

    @pytest.mark.asyncio
    async def test_temp_files_cleaned_up(self, tmp_path: Path, grill_state, make_ui) -> None:
        """open_questions.json and feature_spec.json are removed from workspace."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
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
    async def test_returns_required_keys(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Return dict must contain feature_spec, grill_transcript, current_stage."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert "feature_spec" in result
        assert "grill_transcript" in result
        assert "current_stage" in result

    @pytest.mark.asyncio
    async def test_current_stage_is_grill(self, tmp_path: Path, grill_state, make_ui) -> None:
        """current_stage is always 'grill' after this node runs."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert result["current_stage"] == "grill"

    @pytest.mark.asyncio
    async def test_feature_spec_is_dict(self, tmp_path: Path, grill_state, make_ui) -> None:
        """feature_spec is always a dict, even on parse failure."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 0:
                return _make_agent_result(DONE_RESPONSE)
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "feature_spec.json").write_text("not json")
            return MagicMock()

        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert isinstance(result["feature_spec"], dict)

    @pytest.mark.asyncio
    async def test_grill_transcript_is_string(self, tmp_path: Path, grill_state, make_ui) -> None:
        """grill_transcript is always a string."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert isinstance(result["grill_transcript"], str)

    @pytest.mark.asyncio
    async def test_only_three_keys_returned(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Return dict has exactly three keys — nothing extra leaking."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            result = await grill_node(state, ui)

        assert set(result.keys()) == {
            "feature_spec",
            "grill_transcript",
            "current_stage",
        }


# ---------------------------------------------------------------------------
# Stage lifecycle
# ---------------------------------------------------------------------------


class TestStageLifecycle:
    """Tests for UI stage lifecycle calls and mark_stage_complete."""

    @pytest.mark.asyncio
    async def test_stage_start_and_done_called(self, tmp_path: Path, grill_state, make_ui) -> None:
        """ui.stage_start('grill') and ui.stage_done('grill') bracket the node."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.stage_start.assert_called_once_with("grill")
        ui.stage_done.assert_called_once_with("grill")

    @pytest.mark.asyncio
    async def test_mark_stage_complete_called(self, tmp_path: Path, grill_state, make_ui) -> None:
        """mark_stage_complete updates metadata.json with 'grill'."""
        state = grill_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect([], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        meta = json.loads(Path(state["project_dir"], "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_conversation_calls_plus_compile(self, tmp_path: Path, grill_state, make_ui) -> None:
        """run_agent is called N+1 times: N conversation turns + done + compile."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        n_questions = 2
        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:n_questions], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        # n_questions conversation turns + 1 done + 1 compile = n_questions + 2
        assert mock_agent.call_count == n_questions + 2

    @pytest.mark.asyncio
    async def test_model_passed_through_to_agent(self, tmp_path: Path, grill_state, make_ui) -> None:
        """state['model'] is forwarded to all run_agent calls."""
        state = grill_state(model="claude-sonnet-4-20250514")
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        for c in mock_agent.call_args_list:
            assert c.kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_grill_complete_info_message(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Info message shown when grill completes with confidence."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:1], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        ui.info.assert_any_call("Grill complete after 1 questions (confidence: high).")

    @pytest.mark.asyncio
    async def test_stage_tags_increment(self, tmp_path: Path, grill_state, make_ui) -> None:
        """Each conversation turn uses an incrementing stage tag like grill_q1, grill_q2."""
        state = grill_state()
        ui = make_ui()
        ui.grill_question.return_value = "ok"
        ws = Path(state["project_dir"]) / "workspace"

        side_effect = _make_conversation_side_effect(QUESTION_RESPONSES[:2], DONE_RESPONSE, {}, ws)
        mock_agent = AsyncMock(side_effect=side_effect)
        with patch("kindle.stages.grill.run_agent", mock_agent):
            await grill_node(state, ui)

        stages = [c.kwargs["stage"] for c in mock_agent.call_args_list]
        assert stages[0] == "grill_q1"
        assert stages[1] == "grill_q2"
        # Third is done response (grill_q3), fourth is compile
        assert stages[-1] == "grill_compile"
