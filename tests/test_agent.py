"""Tests for kindle.agent — Claude Agent SDK wrapper with retry logic."""

from __future__ import annotations

import asyncio
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
    """Create a mock content block with a .text attribute."""
    return SimpleNamespace(text=text)


def _make_tool_block(name: str, input_data: dict | None = None) -> SimpleNamespace:
    """Create a mock content block with .name and .input attributes."""
    return SimpleNamespace(name=name, input=input_data or {})


def _make_message(*blocks: SimpleNamespace) -> SimpleNamespace:
    """Create a mock SDK message with a .content list of blocks."""
    return SimpleNamespace(content=list(blocks))


def _make_ui() -> MagicMock:
    """Create a mock UI instance."""
    ui = MagicMock()
    ui.stage_log = MagicMock()
    return ui


async def _async_iter(items: list[Any]):
    """Convert a list into an async iterator (simulates ``query()``)."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


class TestAgentResult:
    """Tests for the AgentResult dataclass structure and defaults."""

    def test_has_expected_fields(self) -> None:
        field_names = {f.name for f in fields(AgentResult)}
        assert field_names == {"text", "tool_calls", "raw_messages", "elapsed_seconds", "turns_used"}

    def test_fields_populated_correctly(self) -> None:
        result = AgentResult(
            text="hello",
            tool_calls=[{"tool": "Read", "input": {}}],
            raw_messages=["msg1", "msg2"],
            elapsed_seconds=1.5,
            turns_used=2,
        )
        assert result.text == "hello"
        assert result.tool_calls == [{"tool": "Read", "input": {}}]
        assert result.raw_messages == ["msg1", "msg2"]
        assert result.elapsed_seconds == 1.5
        assert result.turns_used == 2

    def test_elapsed_seconds_defaults_to_zero(self) -> None:
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.elapsed_seconds == 0.0

    def test_turns_used_defaults_to_zero(self) -> None:
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.turns_used == 0

    def test_empty_text_and_collections(self) -> None:
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.text == ""
        assert result.tool_calls == []
        assert result.raw_messages == []


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify retry constants are set to expected values."""

    def test_max_retries_is_three(self) -> None:
        assert MAX_RETRIES == 3

    def test_retry_backoff_base_is_two(self) -> None:
        assert RETRY_BACKOFF_BASE == 2.0


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    """Tests for _process_message — extracting text and tool-use info from SDK messages."""

    def test_extracts_text_from_text_block(self) -> None:
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        message = _make_message(_make_text_block("Hello world"))

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == ["Hello world"]
        assert tool_calls == []

    def test_extracts_tool_call_from_tool_block(self) -> None:
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        message = _make_message(_make_tool_block("Read", {"file_path": "/foo.py"}))

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == [{"tool": "Read", "input": {"file_path": "/foo.py"}}]

    def test_extracts_both_text_and_tool_blocks(self) -> None:
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        message = _make_message(
            _make_text_block("Analyzing code..."),
            _make_tool_block("Bash", {"command": "ls"}),
            _make_text_block("Done."),
        )

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == ["Analyzing code...", "Done."]
        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]

    def test_skips_message_without_content_attr(self) -> None:
        """Messages that lack a .content attribute should be silently skipped."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        message = SimpleNamespace(role="system")  # no .content

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == []

    def test_skips_block_with_empty_text(self) -> None:
        """A block with text='' should not be appended to text_parts."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        message = _make_message(SimpleNamespace(text=""))

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []

    def test_skips_block_with_no_text_or_name(self) -> None:
        """A block that has neither .text nor .name should be skipped."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        # A block with an unrelated attribute
        message = _make_message(SimpleNamespace(type="image"))

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == []

    def test_tool_block_without_input_attr(self) -> None:
        """A tool block missing .input should default to empty dict."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        block = SimpleNamespace(name="Glob")  # no .input attribute
        message = _make_message(block)

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert tool_calls == [{"tool": "Glob", "input": {}}]

    def test_logs_text_preview_to_ui(self) -> None:
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        message = _make_message(_make_text_block("Short text"))

        _process_message(message, text_parts, tool_calls, "build", ui, "/tmp/proj")

        ui.stage_log.assert_any_call("build", "Short text")

    def test_logs_tool_name_to_ui(self) -> None:
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        message = _make_message(_make_tool_block("Edit"))

        _process_message(message, text_parts, tool_calls, "qa", ui, "/tmp/proj")

        ui.stage_log.assert_any_call("qa", "  \u21b3 tool: Edit")

    def test_text_preview_truncated_to_200_chars(self) -> None:
        """The UI preview of text blocks is limited to 200 characters."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        long_text = "A" * 500
        message = _make_message(_make_text_block(long_text))

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        # The full text should be in text_parts
        assert text_parts == [long_text]
        # The UI preview should be truncated
        logged_preview = ui.stage_log.call_args_list[0][0][1]
        assert len(logged_preview) <= 200

    def test_text_preview_newlines_replaced_with_spaces(self) -> None:
        """Newlines in the preview should be replaced with spaces for cleaner output."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        message = _make_message(_make_text_block("line1\nline2\nline3"))

        _process_message(message, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        logged_preview = ui.stage_log.call_args_list[0][0][1]
        assert "\n" not in logged_preview
        assert "line1 line2 line3" == logged_preview

    def test_multiple_messages_accumulate(self) -> None:
        """Processing multiple messages should accumulate into the same lists."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        msg1 = _make_message(_make_text_block("First"))
        msg2 = _make_message(_make_tool_block("Read"), _make_text_block("Second"))

        _process_message(msg1, text_parts, tool_calls, "dev", ui, "/tmp/proj")
        _process_message(msg2, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == ["First", "Second"]
        assert tool_calls == [{"tool": "Read", "input": {}}]


# ---------------------------------------------------------------------------
# run_agent — happy path
# ---------------------------------------------------------------------------


class TestRunAgentHappyPath:
    """Tests for run_agent successful execution."""

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_returns_agent_result(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        messages = [_make_message(_make_text_block("output text"))]
        mock_query.return_value = _async_iter(messages)

        result = await run_agent(
            persona="coder",
            system_prompt="You are a coder.",
            user_prompt="Write code.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        assert isinstance(result, AgentResult)
        assert result.text == "output text"
        assert result.turns_used == 1
        assert result.elapsed_seconds > 0

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_collects_raw_messages(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        msg1 = _make_message(_make_text_block("part1"))
        msg2 = _make_message(_make_text_block("part2"))
        mock_query.return_value = _async_iter([msg1, msg2])

        result = await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        assert len(result.raw_messages) == 2
        assert result.turns_used == 2
        assert result.text == "part1\npart2"

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_collects_tool_calls(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        msg = _make_message(
            _make_text_block("Reading file"),
            _make_tool_block("Read", {"file_path": "/foo.py"}),
        )
        mock_query.return_value = _async_iter([msg])

        result = await run_agent(
            persona="reviewer",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="qa",
            ui=_make_ui(),
        )

        assert result.tool_calls == [{"tool": "Read", "input": {"file_path": "/foo.py"}}]

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_saves_log_artifact(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([_make_message(_make_text_block("log content"))])

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        mock_save_log.assert_called_once_with("/tmp/proj", "dev", "log content")

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_logs_start_and_finish_to_ui(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([_make_message(_make_text_block("x"))])
        ui = _make_ui()

        await run_agent(
            persona="architect",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="architect",
            ui=ui,
        )

        stage_log_calls = [c[0] for c in ui.stage_log.call_args_list]
        # First call should be the "starting" log
        assert "starting" in stage_log_calls[0][1].lower()
        # Last call should be the "finished" log
        assert "finished" in stage_log_calls[-1][1].lower()


# ---------------------------------------------------------------------------
# run_agent — options / tool_allowlist
# ---------------------------------------------------------------------------


class TestRunAgentOptions:
    """Tests for ClaudeAgentOptions construction inside run_agent."""

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_default_allowed_tools(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        """When allowed_tools is not provided, uses the 7 default tools."""
        mock_query.return_value = _async_iter([])

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["allowed_tools"] == [
            "Read",
            "Write",
            "Edit",
            "MultiEdit",
            "Bash",
            "Glob",
            "Grep",
        ]

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_custom_allowed_tools_passed_through(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([])
        custom_tools = ["Read", "Bash"]

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
            allowed_tools=custom_tools,
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["allowed_tools"] == ["Read", "Bash"]

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_permission_mode_is_bypass(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([])

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["permission_mode"] == "bypassPermissions"

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_model_passed_when_provided(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([])

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
            model="claude-sonnet-4-20250514",
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_model_omitted_when_none(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        """When model is not provided, it should NOT be in the options kwargs."""
        mock_query.return_value = _async_iter([])

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        _, kwargs = mock_options_cls.call_args
        assert "model" not in kwargs

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_system_prompt_and_cwd_passed(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([])

        await run_agent(
            persona="coder",
            system_prompt="You are a coder.",
            user_prompt="Do something.",
            cwd="/work/dir",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["system_prompt"] == "You are a coder."
        assert kwargs["cwd"] == "/work/dir"

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_max_turns_passed(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([])

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
            max_turns=100,
        )

        _, kwargs = mock_options_cls.call_args
        assert kwargs["max_turns"] == 100

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_user_prompt_passed_to_query(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([])

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="Build a REST API",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        _, kwargs = mock_query.call_args
        assert kwargs["prompt"] == "Build a REST API"


# ---------------------------------------------------------------------------
# run_agent — retry logic
# ---------------------------------------------------------------------------


class TestRunAgentRetry:
    """Tests for transient error retry with exponential backoff."""

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_retries_on_connection_error(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """ConnectionError on first attempt, success on second."""
        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("server down")
            return _async_iter([_make_message(_make_text_block("recovered"))])

        mock_query.side_effect = _side_effect

        result = await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "recovered"
        assert call_count == 2
        # First retry: backoff = 2^1 = 2
        mock_sleep.assert_called_once_with(RETRY_BACKOFF_BASE**1)

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_retries_on_timeout_error(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """TimeoutError is a retryable transient error."""
        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            return _async_iter([_make_message(_make_text_block("ok"))])

        mock_query.side_effect = _side_effect

        result = await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "ok"

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_retries_on_os_error(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """OSError is a retryable transient error."""
        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("network unreachable")
            return _async_iter([_make_message(_make_text_block("ok"))])

        mock_query.side_effect = _side_effect

        result = await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "ok"

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_exponential_backoff_delays(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """Each retry should use RETRY_BACKOFF_BASE ** attempt as delay."""
        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("still down")
            return _async_iter([_make_message(_make_text_block("ok"))])

        mock_query.side_effect = _side_effect

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        assert mock_sleep.call_count == 2
        # attempt 1: 2^1 = 2.0, attempt 2: 2^2 = 4.0
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_raises_runtime_error_after_max_retries(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """After MAX_RETRIES failures, raises RuntimeError with the last error info."""

        async def _always_fail(**kwargs):
            raise ConnectionError("permanently down")

        mock_query.side_effect = _always_fail

        with pytest.raises(RuntimeError, match=r"failed after 3 attempts"):
            await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="usr",
                cwd="/tmp",
                project_dir="/tmp/proj",
                stage="dev",
                ui=_make_ui(),
            )

        assert mock_sleep.call_count == MAX_RETRIES

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_runtime_error_includes_persona(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        async def _always_fail(**kwargs):
            raise TimeoutError("timeout")

        mock_query.side_effect = _always_fail

        with pytest.raises(RuntimeError, match=r"Agent \(architect\)"):
            await run_agent(
                persona="architect",
                system_prompt="sys",
                user_prompt="usr",
                cwd="/tmp",
                project_dir="/tmp/proj",
                stage="architect",
                ui=_make_ui(),
            )

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_runtime_error_includes_last_exception(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        async def _always_fail(**kwargs):
            raise ConnectionError("specific network error")

        mock_query.side_effect = _always_fail

        with pytest.raises(RuntimeError, match="specific network error"):
            await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="usr",
                cwd="/tmp",
                project_dir="/tmp/proj",
                stage="dev",
                ui=_make_ui(),
            )

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_non_transient_error_not_retried(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """Errors that are NOT ConnectionError/TimeoutError/OSError should propagate immediately."""

        async def _raise_value_error(**kwargs):
            raise ValueError("bad input")

        mock_query.side_effect = _raise_value_error

        with pytest.raises(ValueError, match="bad input"):
            await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="usr",
                cwd="/tmp",
                project_dir="/tmp/proj",
                stage="dev",
                ui=_make_ui(),
            )

        # Should NOT have retried
        mock_sleep.assert_not_called()

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_retry_logs_warning_to_ui(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """Each retry attempt should log a yellow warning to the UI."""
        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("down")
            return _async_iter([_make_message(_make_text_block("ok"))])

        mock_query.side_effect = _side_effect
        ui = _make_ui()

        await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=ui,
        )

        # Find the retry warning log call
        warning_calls = [
            c for c in ui.stage_log.call_args_list if "failed" in str(c).lower() and "attempt" in str(c).lower()
        ]
        assert len(warning_calls) == 1
        assert "1/3" in str(warning_calls[0])


# ---------------------------------------------------------------------------
# run_agent — error during async iteration (mid-stream failure)
# ---------------------------------------------------------------------------


class TestRunAgentMidStreamError:
    """Tests for errors that occur during message iteration, not at query() call."""

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_retries_on_mid_stream_connection_error(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
        mock_sleep: AsyncMock,
    ) -> None:
        """A ConnectionError raised during async iteration should trigger retry."""
        call_count = 0

        async def _failing_iter():
            yield _make_message(_make_text_block("partial"))
            raise ConnectionError("stream dropped")

        async def _good_iter():
            yield _make_message(_make_text_block("complete"))

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _failing_iter()
            return _good_iter()

        mock_query.side_effect = _side_effect

        result = await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == "complete"
        assert call_count == 2


# ---------------------------------------------------------------------------
# run_agent — empty output
# ---------------------------------------------------------------------------


class TestRunAgentEmptyOutput:
    """Edge case: agent produces no messages at all."""

    @patch("kindle.agent.artifacts.save_log")
    @patch("kindle.agent.query")
    @patch("kindle.agent.ClaudeAgentOptions")
    async def test_empty_message_stream(
        self,
        mock_options_cls: MagicMock,
        mock_query: MagicMock,
        mock_save_log: MagicMock,
    ) -> None:
        mock_query.return_value = _async_iter([])

        result = await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="usr",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="dev",
            ui=_make_ui(),
        )

        assert result.text == ""
        assert result.tool_calls == []
        assert result.raw_messages == []
        assert result.turns_used == 0
