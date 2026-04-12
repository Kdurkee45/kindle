"""Tests for kindle.agent — Claude Agent SDK wrapper with retry logic."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from kindle.agent import (
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    AgentResult,
    _process_message,
    run_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_block(text: str) -> SimpleNamespace:
    """Create a mock content block that has a .text attribute."""
    return SimpleNamespace(text=text)


def _make_tool_block(name: str, input_data: dict | None = None) -> SimpleNamespace:
    """Create a mock content block that has a .name and .input attribute."""
    return SimpleNamespace(name=name, input=input_data or {})


def _make_message(*blocks: SimpleNamespace) -> SimpleNamespace:
    """Create a mock SDK message with a .content list."""
    return SimpleNamespace(content=list(blocks))


def _make_ui() -> MagicMock:
    """Create a mock UI instance with a stage_log method."""
    ui = MagicMock()
    ui.stage_log = MagicMock()
    return ui


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


class TestAgentResult:
    """Tests for the AgentResult dataclass fields and defaults."""

    def test_required_fields_only(self) -> None:
        result = AgentResult(text="hello", tool_calls=[], raw_messages=[])
        assert result.text == "hello"
        assert result.tool_calls == []
        assert result.raw_messages == []

    def test_default_elapsed_seconds(self) -> None:
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.elapsed_seconds == 0.0

    def test_default_turns_used(self) -> None:
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.turns_used == 0

    def test_all_fields_set(self) -> None:
        msgs = [{"role": "assistant"}]
        tools = [{"tool": "Read", "input": {}}]
        result = AgentResult(
            text="output",
            tool_calls=tools,
            raw_messages=msgs,
            elapsed_seconds=1.5,
            turns_used=3,
        )
        assert result.text == "output"
        assert result.tool_calls == tools
        assert result.raw_messages == msgs
        assert result.elapsed_seconds == 1.5
        assert result.turns_used == 3

    def test_field_names(self) -> None:
        """Verify the exact set of field names on the dataclass."""
        names = {f.name for f in fields(AgentResult)}
        assert names == {"text", "tool_calls", "raw_messages", "elapsed_seconds", "turns_used"}

    def test_equality(self) -> None:
        a = AgentResult(text="x", tool_calls=[], raw_messages=[], elapsed_seconds=1.0, turns_used=1)
        b = AgentResult(text="x", tool_calls=[], raw_messages=[], elapsed_seconds=1.0, turns_used=1)
        assert a == b

    def test_inequality(self) -> None:
        a = AgentResult(text="x", tool_calls=[], raw_messages=[])
        b = AgentResult(text="y", tool_calls=[], raw_messages=[])
        assert a != b


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    """Tests for _process_message — extracting text and tool calls from SDK messages."""

    def test_text_block_appended(self) -> None:
        msg = _make_message(_make_text_block("Hello world"))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp/project")
        assert text_parts == ["Hello world"]
        assert tool_calls == []

    def test_tool_block_captured(self) -> None:
        msg = _make_message(_make_tool_block("Read", {"file_path": "/a.py"}))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp/project")
        assert text_parts == []
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Read"
        assert tool_calls[0]["input"] == {"file_path": "/a.py"}

    def test_mixed_blocks(self) -> None:
        msg = _make_message(
            _make_text_block("Analyzing..."),
            _make_tool_block("Bash", {"command": "ls"}),
            _make_text_block("Done."),
        )
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp/project")
        assert text_parts == ["Analyzing...", "Done."]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Bash"

    def test_message_without_content_attribute(self) -> None:
        """Messages without .content should be silently skipped."""
        msg = SimpleNamespace(role="system")  # no .content
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp/project")
        assert text_parts == []
        assert tool_calls == []

    def test_empty_text_block_skipped(self) -> None:
        """A block with text='' should not be appended."""
        msg = _make_message(SimpleNamespace(text=""))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp/project")
        assert text_parts == []

    def test_none_text_block_skipped(self) -> None:
        """A block with text=None should not be appended."""
        msg = _make_message(SimpleNamespace(text=None))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp/project")
        assert text_parts == []

    def test_tool_block_without_input_defaults_to_empty_dict(self) -> None:
        """If a tool block has no .input, getattr should default to {}."""
        block = SimpleNamespace(name="Glob")  # no .input attribute
        msg = _make_message(block)
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp/project")
        assert tool_calls == [{"tool": "Glob", "input": {}}]

    def test_block_with_neither_text_nor_name(self) -> None:
        """A block that has no .text and no .name should not cause errors."""
        block = SimpleNamespace(type="image")  # neither text nor name
        msg = _make_message(block)
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp/project")
        assert text_parts == []
        assert tool_calls == []

    def test_ui_stage_log_called_for_text(self) -> None:
        ui = _make_ui()
        msg = _make_message(_make_text_block("Some output"))
        _process_message(msg, [], [], "qa", ui, "/tmp/project")
        ui.stage_log.assert_called()
        # The preview should be the first 200 chars with newlines replaced
        logged_text = ui.stage_log.call_args_list[-1][0][1]
        assert "Some output" in logged_text

    def test_ui_stage_log_called_for_tool(self) -> None:
        ui = _make_ui()
        msg = _make_message(_make_tool_block("Write"))
        _process_message(msg, [], [], "build", ui, "/tmp/project")
        ui.stage_log.assert_called()
        logged_text = ui.stage_log.call_args_list[-1][0][1]
        assert "Write" in logged_text

    def test_text_preview_truncated_at_200_chars(self) -> None:
        ui = _make_ui()
        long_text = "A" * 300
        msg = _make_message(_make_text_block(long_text))
        _process_message(msg, [], [], "dev", ui, "/tmp/project")
        logged_text = ui.stage_log.call_args_list[-1][0][1]
        assert len(logged_text) == 200

    def test_text_preview_newlines_replaced(self) -> None:
        ui = _make_ui()
        msg = _make_message(_make_text_block("line1\nline2\nline3"))
        _process_message(msg, [], [], "dev", ui, "/tmp/project")
        logged_text = ui.stage_log.call_args_list[-1][0][1]
        assert "\n" not in logged_text
        assert "line1 line2 line3" in logged_text


# ---------------------------------------------------------------------------
# run_agent — happy path
# ---------------------------------------------------------------------------


async def _mock_query_yielding(messages: list[Any]) -> Any:
    """Helper: create an async generator that yields the given messages."""
    for msg in messages:
        yield msg


class TestRunAgentHappyPath:
    """Tests for run_agent when the SDK query succeeds on the first attempt."""

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_returns_agent_result(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        msg = _make_message(_make_text_block("Hello from agent"))
        mock_query.return_value = _mock_query_yielding([msg])

        result = await run_agent(
            persona="architect",
            system_prompt="You are an architect.",
            user_prompt="Design a system.",
            cwd="/tmp/work",
            project_dir="/tmp/project",
            stage="architect",
            ui=_make_ui(),
        )

        assert isinstance(result, AgentResult)
        assert result.text == "Hello from agent"
        assert result.tool_calls == []
        assert result.raw_messages == [msg]
        assert result.turns_used == 1
        assert result.elapsed_seconds > 0

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_options_constructed_with_correct_defaults(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])

        await run_agent(
            persona="dev",
            system_prompt="System prompt",
            user_prompt="User prompt",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        mock_options_cls.assert_called_once_with(
            system_prompt="System prompt",
            cwd="/work",
            max_turns=50,
            allowed_tools=["Read", "Write", "Edit", "MultiEdit", "Bash", "Glob", "Grep"],
            permission_mode="bypassPermissions",
        )

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_custom_allowed_tools(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])

        await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
            allowed_tools=["Read", "Bash"],
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["allowed_tools"] == ["Read", "Bash"]

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_custom_max_turns(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])

        await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
            max_turns=100,
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["max_turns"] == 100

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_model_passed_when_specified(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])

        await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
            model="claude-sonnet-4-20250514",
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_model_not_passed_when_none(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])

        await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        _, kwargs = mock_options_cls.call_args
        assert "model" not in kwargs

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_query_called_with_prompt_and_options(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])
        options_instance = mock_options_cls.return_value

        await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="Do something",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        mock_query.assert_called_once_with(prompt="Do something", options=options_instance)

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_save_log_called_with_full_text(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        msg1 = _make_message(_make_text_block("Part 1"))
        msg2 = _make_message(_make_text_block("Part 2"))
        mock_query.return_value = _mock_query_yielding([msg1, msg2])

        await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/tmp/project",
            stage="build",
            ui=_make_ui(),
        )

        mock_save_log.assert_called_once_with("/tmp/project", "build", "Part 1\nPart 2")

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_multiple_messages_concatenated(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        msgs = [
            _make_message(_make_text_block("Alpha")),
            _make_message(_make_tool_block("Bash", {"command": "ls"})),
            _make_message(_make_text_block("Beta")),
        ]
        mock_query.return_value = _mock_query_yielding(msgs)

        result = await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "Alpha\nBeta"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "Bash"
        assert result.raw_messages == msgs
        assert result.turns_used == 3

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_no_messages_returns_empty(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])

        result = await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == ""
        assert result.tool_calls == []
        assert result.raw_messages == []
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# run_agent — retry logic
# ---------------------------------------------------------------------------


class TestRunAgentRetry:
    """Tests for transient failure retry with exponential backoff."""

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_retries_on_connection_error(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """First attempt raises ConnectionError, second succeeds."""

        call_count = 0

        async def _query_side_effect(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection refused")
            async def gen() -> Any:
                yield _make_message(_make_text_block("recovered"))
            return gen()

        mock_query.side_effect = _query_side_effect

        result = await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "recovered"
        assert call_count == 2
        # First retry: backoff = 2^1 = 2.0 seconds
        mock_sleep.assert_called_once_with(RETRY_BACKOFF_BASE**1)

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_retries_on_timeout_error(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        call_count = 0

        async def _query_side_effect(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            async def gen() -> Any:
                yield _make_message(_make_text_block("ok"))
            return gen()

        mock_query.side_effect = _query_side_effect

        result = await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "ok"

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_retries_on_os_error(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        call_count = 0

        async def _query_side_effect(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("network unreachable")
            async def gen() -> Any:
                yield _make_message(_make_text_block("ok"))
            return gen()

        mock_query.side_effect = _query_side_effect

        result = await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "ok"

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_exponential_backoff_delays(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """Two failures before success — verify both backoff delays."""
        call_count = 0

        async def _query_side_effect(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("fail")
            async def gen() -> Any:
                yield _make_message(_make_text_block("success"))
            return gen()

        mock_query.side_effect = _query_side_effect

        result = await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "success"
        assert call_count == 3
        # Backoff: attempt 1 → 2^1=2.0s, attempt 2 → 2^2=4.0s
        assert mock_sleep.call_args_list == [
            call(RETRY_BACKOFF_BASE**1),
            call(RETRY_BACKOFF_BASE**2),
        ]

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_error_during_iteration_triggers_retry(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """Error raised during async iteration (not at call time) should retry."""
        call_count = 0

        async def _query_side_effect(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1

            async def failing_gen() -> Any:
                raise ConnectionError("mid-stream disconnect")
                yield  # noqa: RUF059 — make this an async generator

            async def success_gen() -> Any:
                yield _make_message(_make_text_block("recovered"))

            if call_count == 1:
                return failing_gen()
            return success_gen()

        mock_query.side_effect = _query_side_effect

        result = await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "recovered"
        assert call_count == 2


# ---------------------------------------------------------------------------
# run_agent — MAX_RETRIES exhaustion
# ---------------------------------------------------------------------------


class TestRunAgentMaxRetries:
    """Tests for when all retry attempts are exhausted."""

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_raises_runtime_error_after_max_retries(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        async def _always_fail(**kwargs: Any) -> Any:
            raise ConnectionError("persistent failure")

        mock_query.side_effect = _always_fail

        with pytest.raises(RuntimeError, match=r"failed after 3 attempts"):
            await run_agent(
                persona="dev",
                system_prompt="sp",
                user_prompt="up",
                cwd="/work",
                project_dir="/project",
                stage="dev",
                ui=_make_ui(),
            )

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_runtime_error_includes_persona(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        async def _always_fail(**kwargs: Any) -> Any:
            raise TimeoutError("timeout")

        mock_query.side_effect = _always_fail

        with pytest.raises(RuntimeError, match=r"Agent \(architect\)"):
            await run_agent(
                persona="architect",
                system_prompt="sp",
                user_prompt="up",
                cwd="/work",
                project_dir="/project",
                stage="architect",
                ui=_make_ui(),
            )

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_runtime_error_wraps_last_exception(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        async def _always_fail(**kwargs: Any) -> Any:
            raise OSError("disk full")

        mock_query.side_effect = _always_fail

        with pytest.raises(RuntimeError, match="disk full"):
            await run_agent(
                persona="dev",
                system_prompt="sp",
                user_prompt="up",
                cwd="/work",
                project_dir="/project",
                stage="dev",
                ui=_make_ui(),
            )

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_sleep_called_max_retries_times(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        async def _always_fail(**kwargs: Any) -> Any:
            raise ConnectionError("fail")

        mock_query.side_effect = _always_fail

        with pytest.raises(RuntimeError):
            await run_agent(
                persona="dev",
                system_prompt="sp",
                user_prompt="up",
                cwd="/work",
                project_dir="/project",
                stage="dev",
                ui=_make_ui(),
            )

        assert mock_sleep.call_count == MAX_RETRIES
        expected_calls = [call(RETRY_BACKOFF_BASE**i) for i in range(1, MAX_RETRIES + 1)]
        assert mock_sleep.call_args_list == expected_calls

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_non_retryable_error_propagates_immediately(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """Errors that are NOT ConnectionError/TimeoutError/OSError should propagate without retry."""
        async def _fail(**kwargs: Any) -> Any:
            raise ValueError("unexpected error")

        mock_query.side_effect = _fail

        with pytest.raises(ValueError, match="unexpected error"):
            await run_agent(
                persona="dev",
                system_prompt="sp",
                user_prompt="up",
                cwd="/work",
                project_dir="/project",
                stage="dev",
                ui=_make_ui(),
            )

        mock_sleep.assert_not_called()

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_save_log_not_called_when_all_retries_fail(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """save_log should only be called on success, not on failure."""
        async def _always_fail(**kwargs: Any) -> Any:
            raise ConnectionError("fail")

        mock_query.side_effect = _always_fail

        with pytest.raises(RuntimeError):
            await run_agent(
                persona="dev",
                system_prompt="sp",
                user_prompt="up",
                cwd="/work",
                project_dir="/project",
                stage="dev",
                ui=_make_ui(),
            )

        mock_save_log.assert_not_called()


# ---------------------------------------------------------------------------
# run_agent — UI logging
# ---------------------------------------------------------------------------


class TestRunAgentUILogging:
    """Tests for UI interaction during run_agent execution."""

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_starting_message_logged(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])
        ui = _make_ui()

        await run_agent(
            persona="architect",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="architect",
            ui=ui,
        )

        # First call should be the "starting" log
        first_call_args = ui.stage_log.call_args_list[0]
        assert first_call_args[0][0] == "architect"
        assert "architect" in first_call_args[0][1].lower()

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_finished_message_logged(
        self,
        mock_query: MagicMock,
        mock_options_cls: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _mock_query_yielding([])
        ui = _make_ui()

        await run_agent(
            persona="dev",
            system_prompt="sp",
            user_prompt="up",
            cwd="/work",
            project_dir="/project",
            stage="dev",
            ui=ui,
        )

        # Last call should be the "finished" log
        last_call_args = ui.stage_log.call_args_list[-1]
        assert last_call_args[0][0] == "dev"
        assert "finished" in last_call_args[0][1].lower()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are sensible."""

    def test_max_retries_is_three(self) -> None:
        assert MAX_RETRIES == 3

    def test_retry_backoff_base(self) -> None:
        assert RETRY_BACKOFF_BASE == 2.0
