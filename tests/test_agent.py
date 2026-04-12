"""Tests for kindle.agent — Claude Agent SDK wrapper with retry logic."""

from __future__ import annotations

from types import SimpleNamespace
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


def _make_ui() -> MagicMock:
    return MagicMock()


def _text_block(text: str) -> SimpleNamespace:
    """Create a mock content block that has a .text attribute."""
    return SimpleNamespace(text=text)


def _tool_block(name: str, input_data: dict | None = None) -> SimpleNamespace:
    """Create a mock content block that has a .name (tool use) attribute."""
    return SimpleNamespace(name=name, input=input_data or {})


def _message(*blocks: SimpleNamespace) -> SimpleNamespace:
    """Create a mock SDK message with a .content list of blocks."""
    return SimpleNamespace(content=list(blocks))


def _run_agent_kwargs(ui: MagicMock | None = None, **overrides) -> dict:
    """Return a default set of keyword arguments for run_agent()."""
    defaults = dict(
        persona="tester",
        system_prompt="You are helpful.",
        user_prompt="Say hello",
        cwd="/tmp",
        project_dir="/tmp/proj",
        stage="dev",
        ui=ui or _make_ui(),
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


class TestAgentResult:
    """Verify AgentResult fields and defaults."""

    def test_required_fields(self):
        result = AgentResult(text="hi", tool_calls=[], raw_messages=[])
        assert result.text == "hi"
        assert result.tool_calls == []
        assert result.raw_messages == []

    def test_default_elapsed_seconds(self):
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.elapsed_seconds == 0.0

    def test_default_turns_used(self):
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.turns_used == 0

    def test_custom_elapsed_and_turns(self):
        result = AgentResult(
            text="done",
            tool_calls=[{"tool": "Read"}],
            raw_messages=["m1", "m2"],
            elapsed_seconds=4.2,
            turns_used=2,
        )
        assert result.elapsed_seconds == 4.2
        assert result.turns_used == 2
        assert len(result.raw_messages) == 2


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    """Tests for the internal _process_message() helper."""

    def test_extracts_text_content(self):
        msg = _message(_text_block("Hello world"))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == ["Hello world"]
        assert tool_calls == []

    def test_extracts_tool_use_content(self):
        msg = _message(_tool_block("Bash", {"command": "ls"}))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]

    def test_extracts_mixed_content(self):
        msg = _message(
            _text_block("Analyzing…"),
            _tool_block("Read", {"path": "/foo"}),
            _text_block("Done."),
        )
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == ["Analyzing…", "Done."]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Read"

    def test_skips_message_without_content_attribute(self):
        """Messages without .content (e.g. heartbeats) should be ignored."""
        msg = SimpleNamespace(type="heartbeat")
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == []

    def test_skips_empty_text_blocks(self):
        """Blocks with falsy .text (empty string) should not be appended."""
        msg = _message(SimpleNamespace(text=""))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []

    def test_logs_text_preview_to_ui(self):
        msg = _message(_text_block("Short text"))
        ui = _make_ui()

        _process_message(msg, [], [], "qa", ui, "/tmp/proj")

        ui.stage_log.assert_called_once_with("qa", "Short text")

    def test_logs_tool_name_to_ui(self):
        msg = _message(_tool_block("Grep"))
        ui = _make_ui()

        _process_message(msg, [], [], "dev", ui, "/tmp/proj")

        ui.stage_log.assert_called_once_with("dev", "  ↳ tool: Grep")

    def test_text_preview_truncated_to_200_chars(self):
        long_text = "x" * 400
        msg = _message(_text_block(long_text))
        ui = _make_ui()

        _process_message(msg, [], [], "dev", ui, "/tmp/proj")

        logged_preview = ui.stage_log.call_args[0][1]
        assert len(logged_preview) == 200

    def test_text_preview_replaces_newlines(self):
        msg = _message(_text_block("line1\nline2\nline3"))
        ui = _make_ui()

        _process_message(msg, [], [], "dev", ui, "/tmp/proj")

        logged_preview = ui.stage_log.call_args[0][1]
        assert "\n" not in logged_preview
        assert "line1 line2 line3" == logged_preview

    def test_tool_block_without_input_attribute(self):
        """Tool blocks missing .input should default to empty dict."""
        block = SimpleNamespace(name="Write")  # no .input attr
        msg = SimpleNamespace(content=[block])
        tool_calls: list[dict] = []

        _process_message(msg, [], tool_calls, "dev", _make_ui(), "/tmp/proj")

        assert tool_calls == [{"tool": "Write", "input": {}}]


# ---------------------------------------------------------------------------
# run_agent — successful invocation
# ---------------------------------------------------------------------------


class TestRunAgentSuccess:
    """Test run_agent() when the SDK query returns normally."""

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_returns_agent_result_with_text(self, mock_query, mock_opts_cls, mock_artifacts):
        """Single-turn agent returns AgentResult containing extracted text."""
        msg = _message(_text_block("Hello from agent"))
        mock_query.return_value = _async_iter([msg])

        result = await run_agent(**_run_agent_kwargs())

        assert isinstance(result, AgentResult)
        assert result.text == "Hello from agent"
        assert result.turns_used == 1

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_returns_tool_calls(self, mock_query, mock_opts_cls, mock_artifacts):
        msg = _message(_tool_block("Bash", {"command": "echo hi"}))
        mock_query.return_value = _async_iter([msg])

        result = await run_agent(**_run_agent_kwargs())

        assert result.tool_calls == [{"tool": "Bash", "input": {"command": "echo hi"}}]

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_elapsed_seconds_is_positive(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([_message(_text_block("ok"))])

        result = await run_agent(**_run_agent_kwargs())

        assert result.elapsed_seconds > 0

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_raw_messages_captured(self, mock_query, mock_opts_cls, mock_artifacts):
        msgs = [_message(_text_block("a")), _message(_text_block("b"))]
        mock_query.return_value = _async_iter(msgs)

        result = await run_agent(**_run_agent_kwargs())

        assert len(result.raw_messages) == 2
        assert result.turns_used == 2

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_multiple_text_blocks_joined_with_newline(self, mock_query, mock_opts_cls, mock_artifacts):
        msgs = [_message(_text_block("line1")), _message(_text_block("line2"))]
        mock_query.return_value = _async_iter(msgs)

        result = await run_agent(**_run_agent_kwargs())

        assert result.text == "line1\nline2"

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_calls_save_log(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([_message(_text_block("log me"))])

        await run_agent(**_run_agent_kwargs(project_dir="/my/proj", stage="qa"))

        mock_artifacts.save_log.assert_called_once_with("/my/proj", "qa", "log me")

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_ui_stage_log_called_on_start_and_finish(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([_message(_text_block("hi"))])
        ui = _make_ui()

        await run_agent(**_run_agent_kwargs(ui=ui, persona="coder", stage="dev"))

        stage_log_calls = [c[0] for c in ui.stage_log.call_args_list]
        # First call: starting message
        assert stage_log_calls[0][0] == "dev"
        assert "coder" in stage_log_calls[0][1]
        assert "starting" in stage_log_calls[0][1]
        # Last call: finished message
        assert stage_log_calls[-1][0] == "dev"
        assert "finished" in stage_log_calls[-1][1]


# ---------------------------------------------------------------------------
# run_agent — options construction
# ---------------------------------------------------------------------------


class TestRunAgentOptions:
    """Verify how ClaudeAgentOptions is constructed."""

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_default_allowed_tools(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([])

        await run_agent(**_run_agent_kwargs())

        _, kwargs = mock_opts_cls.call_args
        assert kwargs["allowed_tools"] == [
            "Read",
            "Write",
            "Edit",
            "MultiEdit",
            "Bash",
            "Glob",
            "Grep",
        ]

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_custom_allowed_tools(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([])

        await run_agent(**_run_agent_kwargs(allowed_tools=["Read", "Grep"]))

        _, kwargs = mock_opts_cls.call_args
        assert kwargs["allowed_tools"] == ["Read", "Grep"]

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_model_forwarded_when_set(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([])

        await run_agent(**_run_agent_kwargs(model="claude-sonnet-4-20250514"))

        _, kwargs = mock_opts_cls.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_model_omitted_when_none(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([])

        await run_agent(**_run_agent_kwargs(model=None))

        _, kwargs = mock_opts_cls.call_args
        assert "model" not in kwargs

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_bypass_permissions_mode(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([])

        await run_agent(**_run_agent_kwargs())

        _, kwargs = mock_opts_cls.call_args
        assert kwargs["permission_mode"] == "bypassPermissions"

    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_max_turns_forwarded(self, mock_query, mock_opts_cls, mock_artifacts):
        mock_query.return_value = _async_iter([])

        await run_agent(**_run_agent_kwargs(max_turns=10))

        _, kwargs = mock_opts_cls.call_args
        assert kwargs["max_turns"] == 10


# ---------------------------------------------------------------------------
# run_agent — retry logic
# ---------------------------------------------------------------------------


class TestRunAgentRetry:
    """Tests for transient-failure retry with exponential backoff."""

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_retries_on_connection_error(self, mock_query, mock_opts_cls, mock_artifacts, mock_sleep):
        """First call raises ConnectionError, second succeeds — 2 attempts total."""
        mock_query.side_effect = [
            _raise_async_iter(ConnectionError("network down")),
            _async_iter([_message(_text_block("recovered"))]),
        ]

        result = await run_agent(**_run_agent_kwargs())

        assert result.text == "recovered"
        assert mock_query.call_count == 2
        # Should have slept once between attempts (backoff^1 = 2s)
        mock_sleep.assert_awaited_once_with(RETRY_BACKOFF_BASE**1)

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_retries_on_timeout_error(self, mock_query, mock_opts_cls, mock_artifacts, mock_sleep):
        mock_query.side_effect = [
            _raise_async_iter(TimeoutError("timed out")),
            _async_iter([_message(_text_block("ok"))]),
        ]

        result = await run_agent(**_run_agent_kwargs())

        assert result.text == "ok"
        assert mock_query.call_count == 2

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_retries_on_os_error(self, mock_query, mock_opts_cls, mock_artifacts, mock_sleep):
        mock_query.side_effect = [
            _raise_async_iter(OSError("io failure")),
            _async_iter([_message(_text_block("ok"))]),
        ]

        result = await run_agent(**_run_agent_kwargs())

        assert result.text == "ok"

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_max_retries_exhausted_raises_runtime_error(
        self, mock_query, mock_opts_cls, mock_artifacts, mock_sleep
    ):
        """All 3 attempts fail — RuntimeError should be raised."""
        mock_query.side_effect = [
            _raise_async_iter(ConnectionError("fail 1")),
            _raise_async_iter(ConnectionError("fail 2")),
            _raise_async_iter(ConnectionError("fail 3")),
        ]

        with pytest.raises(RuntimeError, match=r"failed after 3 attempts"):
            await run_agent(**_run_agent_kwargs(persona="builder"))

        assert mock_query.call_count == MAX_RETRIES
        assert mock_sleep.await_count == MAX_RETRIES

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_backoff_delays_are_exponential(self, mock_query, mock_opts_cls, mock_artifacts, mock_sleep):
        """Verify the sleep durations follow RETRY_BACKOFF_BASE ** attempt."""
        mock_query.side_effect = [
            _raise_async_iter(ConnectionError("e1")),
            _raise_async_iter(ConnectionError("e2")),
            _raise_async_iter(ConnectionError("e3")),
        ]

        with pytest.raises(RuntimeError):
            await run_agent(**_run_agent_kwargs())

        expected_delays = [RETRY_BACKOFF_BASE**i for i in range(1, MAX_RETRIES + 1)]
        actual_delays = [c.args[0] for c in mock_sleep.await_args_list]
        assert actual_delays == expected_delays

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_retry_ui_log_includes_attempt_count(self, mock_query, mock_opts_cls, mock_artifacts, mock_sleep):
        mock_query.side_effect = [
            _raise_async_iter(ConnectionError("oops")),
            _async_iter([_message(_text_block("ok"))]),
        ]
        ui = _make_ui()

        await run_agent(**_run_agent_kwargs(ui=ui))

        # Find the yellow warning log call
        warning_calls = [c for c in ui.stage_log.call_args_list if "attempt 1/" in str(c)]
        assert len(warning_calls) == 1

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_non_retryable_error_propagates_immediately(
        self, mock_query, mock_opts_cls, mock_artifacts, mock_sleep
    ):
        """Errors not in the retry set (e.g. ValueError) should not be caught."""
        mock_query.side_effect = [
            _raise_async_iter(ValueError("bad input")),
        ]

        with pytest.raises(ValueError, match="bad input"):
            await run_agent(**_run_agent_kwargs())

        assert mock_query.call_count == 1
        mock_sleep.assert_not_awaited()

    @patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("kindle.agent.artifacts")
    @patch("kindle.agent.ClaudeAgentOptions")
    @patch("kindle.agent.query")
    async def test_runtime_error_includes_persona_and_last_error(
        self, mock_query, mock_opts_cls, mock_artifacts, mock_sleep
    ):
        mock_query.side_effect = [
            _raise_async_iter(ConnectionError("net err")),
            _raise_async_iter(ConnectionError("net err")),
            _raise_async_iter(ConnectionError("net err")),
        ]

        with pytest.raises(RuntimeError, match=r"Agent \(mybot\)") as exc_info:
            await run_agent(**_run_agent_kwargs(persona="mybot"))

        assert "net err" in str(exc_info.value)


# ---------------------------------------------------------------------------
# run_agent — constants
# ---------------------------------------------------------------------------


class TestAgentConstants:
    """Verify module-level constants are sensible."""

    def test_max_retries_is_three(self):
        assert MAX_RETRIES == 3

    def test_backoff_base_is_two(self):
        assert RETRY_BACKOFF_BASE == 2.0


# ---------------------------------------------------------------------------
# Async iterator helpers for mocking query()
# ---------------------------------------------------------------------------


async def _async_iter(items):
    """Return an async iterable that yields the given items."""
    for item in items:
        yield item


async def _raise_async_iter(exc: Exception):
    """Return an async iterable that immediately raises *exc*."""
    raise exc
    # The yield makes this an async generator (unreachable but needed for syntax).
    yield  # noqa: RET503  # pragma: no cover
