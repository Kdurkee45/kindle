"""Tests for kindle.agent — Claude Agent SDK wrapper."""

from __future__ import annotations

import asyncio
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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

DEFAULT_TOOLS = ["Read", "Write", "Edit", "MultiEdit", "Bash", "Glob", "Grep"]


def _make_ui() -> MagicMock:
    """Return a lightweight mock UI with the methods agent.py calls."""
    ui = MagicMock()
    ui.stage_log = MagicMock()
    return ui


def _text_block(text: str) -> SimpleNamespace:
    """Create a fake text content block."""
    return SimpleNamespace(text=text)


def _tool_block(name: str, tool_input: dict | None = None) -> SimpleNamespace:
    """Create a fake tool-use content block."""
    return SimpleNamespace(name=name, input=tool_input or {})


def _message(blocks: list) -> SimpleNamespace:
    """Create a fake SDK message with the given content blocks."""
    return SimpleNamespace(content=blocks)


def _message_no_content() -> SimpleNamespace:
    """Create a fake SDK message *without* a content attribute."""
    return SimpleNamespace(role="system")


def _base_kwargs() -> dict:
    """Minimal keyword arguments for run_agent."""
    return dict(
        persona="tester",
        system_prompt="You are a test agent.",
        user_prompt="Do the thing.",
        cwd="/tmp",
        project_dir="/tmp/project",
        stage="dev",
        ui=_make_ui(),
    )


# ------------------------------------------------------------------
# AgentResult dataclass
# ------------------------------------------------------------------


class TestAgentResult:
    """Basic sanity checks on the result dataclass."""

    def test_defaults(self) -> None:
        result = AgentResult(text="hi", tool_calls=[], raw_messages=[])
        assert result.elapsed_seconds == 0.0
        assert result.turns_used == 0

    def test_all_fields(self) -> None:
        result = AgentResult(
            text="output",
            tool_calls=[{"tool": "Bash", "input": {}}],
            raw_messages=["m1"],
            elapsed_seconds=1.5,
            turns_used=3,
        )
        assert result.text == "output"
        assert len(result.tool_calls) == 1
        assert result.raw_messages == ["m1"]
        assert result.elapsed_seconds == 1.5
        assert result.turns_used == 3


# ------------------------------------------------------------------
# _process_message
# ------------------------------------------------------------------


class TestProcessMessage:
    """Unit tests for _process_message helper."""

    def test_message_without_content_attr_is_noop(self) -> None:
        """Messages lacking a content attribute are silently skipped."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _message_no_content()
        _process_message(msg, text_parts, tool_calls, "dev", _make_ui(), "/tmp")
        assert text_parts == []
        assert tool_calls == []

    def test_text_block_appended_and_previewed(self) -> None:
        """Text blocks are collected and their preview is sent to the UI."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([_text_block("Hello world")])

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp")

        assert text_parts == ["Hello world"]
        assert tool_calls == []
        ui.stage_log.assert_called_once_with("dev", "Hello world")

    def test_text_preview_truncated_to_200_chars(self) -> None:
        """Long text blocks are previewed at 200 characters max."""
        long_text = "A" * 500
        text_parts: list[str] = []
        ui = _make_ui()
        msg = _message([_text_block(long_text)])

        _process_message(msg, text_parts, [], "qa", ui, "/tmp")

        # The full text is stored
        assert text_parts == [long_text]
        # The preview sent to UI is truncated
        preview_arg = ui.stage_log.call_args[0][1]
        assert len(preview_arg) == 200

    def test_text_preview_newlines_replaced(self) -> None:
        """Newlines in the preview are collapsed to spaces."""
        text = "line1\nline2\nline3"
        ui = _make_ui()
        msg = _message([_text_block(text)])

        _process_message(msg, [], [], "dev", ui, "/tmp")

        preview = ui.stage_log.call_args[0][1]
        assert "\n" not in preview
        assert "line1 line2 line3" == preview

    def test_tool_use_block_recorded(self) -> None:
        """Tool-use blocks are captured with name and input."""
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([_tool_block("Bash", {"command": "ls"})])

        _process_message(msg, [], tool_calls, "dev", ui, "/tmp")

        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]
        ui.stage_log.assert_called_once_with("dev", "  ↳ tool: Bash")

    def test_tool_use_block_missing_input_defaults_to_empty_dict(self) -> None:
        """Tool blocks without an input attribute default to {}."""
        tool_calls: list[dict] = []
        block = SimpleNamespace(name="Read")  # no 'input' attr
        msg = _message([block])

        _process_message(msg, [], tool_calls, "dev", _make_ui(), "/tmp")

        assert tool_calls == [{"tool": "Read", "input": {}}]

    def test_mixed_blocks_in_one_message(self) -> None:
        """A message can contain both text and tool-use blocks."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([
            _text_block("thinking…"),
            _tool_block("Edit", {"file": "a.py"}),
            _text_block("done"),
        ])

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp")

        assert text_parts == ["thinking…", "done"]
        assert tool_calls == [{"tool": "Edit", "input": {"file": "a.py"}}]
        assert ui.stage_log.call_count == 3

    def test_empty_text_block_ignored(self) -> None:
        """Text blocks with empty string are skipped (falsy check)."""
        text_parts: list[str] = []
        ui = _make_ui()
        msg = _message([_text_block("")])

        _process_message(msg, text_parts, [], "dev", ui, "/tmp")

        assert text_parts == []
        ui.stage_log.assert_not_called()


# ------------------------------------------------------------------
# run_agent — happy path
# ------------------------------------------------------------------


class TestRunAgentHappyPath:
    """run_agent returns a correctly populated AgentResult on success."""

    @pytest.mark.asyncio
    async def test_basic_text_response(self) -> None:
        """A single text message yields correct text and metadata."""
        msg = _message([_text_block("Hello from agent")])
        kwargs = _base_kwargs()

        async def fake_query(**_kw):
            yield msg

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts") as mock_artifacts,
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            result = await run_agent(**kwargs)

        assert isinstance(result, AgentResult)
        assert result.text == "Hello from agent"
        assert result.tool_calls == []
        assert result.raw_messages == [msg]
        assert result.turns_used == 1
        assert result.elapsed_seconds > 0
        mock_artifacts.save_log.assert_called_once_with(
            "/tmp/project", "dev", "Hello from agent"
        )

    @pytest.mark.asyncio
    async def test_multiple_messages(self) -> None:
        """Multiple messages are concatenated and all raw messages collected."""
        msg1 = _message([_text_block("Part one")])
        msg2 = _message([_tool_block("Bash", {"command": "ls"})])
        msg3 = _message([_text_block("Part two")])
        kwargs = _base_kwargs()

        async def fake_query(**_kw):
            for m in [msg1, msg2, msg3]:
                yield m

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            result = await run_agent(**kwargs)

        assert result.text == "Part one\nPart two"
        assert result.tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]
        assert len(result.raw_messages) == 3
        assert result.turns_used == 3

    @pytest.mark.asyncio
    async def test_message_without_content_included_in_raw(self) -> None:
        """Messages lacking .content are still appended to raw_messages."""
        msg_no_content = _message_no_content()
        msg_text = _message([_text_block("ok")])
        kwargs = _base_kwargs()

        async def fake_query(**_kw):
            yield msg_no_content
            yield msg_text

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            result = await run_agent(**kwargs)

        assert len(result.raw_messages) == 2
        assert result.text == "ok"


# ------------------------------------------------------------------
# run_agent — default and custom parameters
# ------------------------------------------------------------------


class TestRunAgentParameters:
    """Verify default allowed_tools and passthrough of model / allowed_tools."""

    @pytest.mark.asyncio
    async def test_default_allowed_tools(self) -> None:
        """When allowed_tools is None, the seven defaults are used."""
        captured_options: list = []

        async def fake_query(**kw):
            yield _message([_text_block("ok")])

        def capture_options(**kw):
            captured_options.append(kw)
            return MagicMock()

        kwargs = _base_kwargs()

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions", side_effect=capture_options),
        ):
            await run_agent(**kwargs)

        assert len(captured_options) == 1
        assert captured_options[0]["allowed_tools"] == DEFAULT_TOOLS

    @pytest.mark.asyncio
    async def test_custom_allowed_tools_passthrough(self) -> None:
        """Custom allowed_tools are forwarded to ClaudeAgentOptions."""
        captured_options: list = []

        async def fake_query(**kw):
            yield _message([_text_block("ok")])

        def capture_options(**kw):
            captured_options.append(kw)
            return MagicMock()

        kwargs = _base_kwargs()
        kwargs["allowed_tools"] = ["Read", "Bash"]

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions", side_effect=capture_options),
        ):
            await run_agent(**kwargs)

        assert captured_options[0]["allowed_tools"] == ["Read", "Bash"]

    @pytest.mark.asyncio
    async def test_custom_model_passthrough(self) -> None:
        """When model is provided, it appears in ClaudeAgentOptions kwargs."""
        captured_options: list = []

        async def fake_query(**kw):
            yield _message([_text_block("ok")])

        def capture_options(**kw):
            captured_options.append(kw)
            return MagicMock()

        kwargs = _base_kwargs()
        kwargs["model"] = "claude-sonnet-4-20250514"

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions", side_effect=capture_options),
        ):
            await run_agent(**kwargs)

        assert captured_options[0]["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_no_model_by_default(self) -> None:
        """When model is None, it is NOT passed to ClaudeAgentOptions."""
        captured_options: list = []

        async def fake_query(**kw):
            yield _message([_text_block("ok")])

        def capture_options(**kw):
            captured_options.append(kw)
            return MagicMock()

        kwargs = _base_kwargs()
        # model defaults to None

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions", side_effect=capture_options),
        ):
            await run_agent(**kwargs)

        assert "model" not in captured_options[0]


# ------------------------------------------------------------------
# run_agent — retry logic
# ------------------------------------------------------------------


class TestRunAgentRetry:
    """Retry with exponential backoff for transient errors."""

    @pytest.mark.asyncio
    async def test_retry_on_connection_error_then_succeed(self) -> None:
        """ConnectionError on first attempt triggers retry; success on second."""
        attempt_count = 0

        async def flaky_query(**_kw):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ConnectionError("connection reset")
            yield _message([_text_block("recovered")])

        kwargs = _base_kwargs()

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await run_agent(**kwargs)

        assert result.text == "recovered"
        assert attempt_count == 2
        # First failure → backoff = RETRY_BACKOFF_BASE ** 1
        mock_sleep.assert_called_once_with(RETRY_BACKOFF_BASE**1)

    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self) -> None:
        """TimeoutError also triggers retry."""
        attempt_count = 0

        async def flaky_query(**_kw):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise TimeoutError("timed out")
            yield _message([_text_block("ok")])

        kwargs = _base_kwargs()

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_agent(**kwargs)

        assert result.text == "ok"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_os_error(self) -> None:
        """OSError also triggers retry."""
        attempt_count = 0

        async def flaky_query(**_kw):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise OSError("network unreachable")
            yield _message([_text_block("ok")])

        kwargs = _base_kwargs()

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_agent(**kwargs)

        assert result.text == "ok"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        """Each retry uses RETRY_BACKOFF_BASE ** attempt as delay."""
        attempt_count = 0

        async def flaky_query(**_kw):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < MAX_RETRIES:
                raise ConnectionError("fail")
            yield _message([_text_block("finally")])

        kwargs = _base_kwargs()

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await run_agent(**kwargs)

        assert result.text == "finally"
        assert attempt_count == MAX_RETRIES
        # Verify each sleep call had the correct backoff
        expected_calls = [
            call(RETRY_BACKOFF_BASE**i) for i in range(1, MAX_RETRIES)
        ]
        assert mock_sleep.call_args_list == expected_calls

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_runtime_error(self) -> None:
        """When all MAX_RETRIES fail, RuntimeError is raised."""

        async def always_fail(**_kw):
            raise ConnectionError("persistent failure")

        kwargs = _base_kwargs()

        with (
            patch("kindle.agent.query", side_effect=always_fail),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(RuntimeError, match=r"failed after 3 attempts"):
                await run_agent(**kwargs)

        # All MAX_RETRIES attempts were made, each followed by a sleep
        assert mock_sleep.call_count == MAX_RETRIES

    @pytest.mark.asyncio
    async def test_exhausted_retries_includes_last_error_message(self) -> None:
        """The RuntimeError message includes the persona and last exception."""

        async def always_fail(**_kw):
            raise TimeoutError("specific timeout message")

        kwargs = _base_kwargs()

        with (
            patch("kindle.agent.query", side_effect=always_fail),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match=r"tester.*specific timeout message"):
                await run_agent(**kwargs)

    @pytest.mark.asyncio
    async def test_non_transient_error_not_retried(self) -> None:
        """Errors not in the retry set (e.g. ValueError) propagate immediately."""

        async def bad_query(**_kw):
            raise ValueError("unexpected")

        kwargs = _base_kwargs()

        with (
            patch("kindle.agent.query", side_effect=bad_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(ValueError, match="unexpected"):
                await run_agent(**kwargs)

        mock_sleep.assert_not_called()


# ------------------------------------------------------------------
# run_agent — logging
# ------------------------------------------------------------------


class TestRunAgentLogging:
    """Verify that artifacts.save_log is called with the full text."""

    @pytest.mark.asyncio
    async def test_save_log_called_with_joined_text(self) -> None:
        """save_log receives project_dir, stage, and newline-joined text."""
        msg1 = _message([_text_block("alpha")])
        msg2 = _message([_text_block("bravo")])
        kwargs = _base_kwargs()

        async def fake_query(**_kw):
            yield msg1
            yield msg2

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts") as mock_artifacts,
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            await run_agent(**kwargs)

        mock_artifacts.save_log.assert_called_once_with(
            "/tmp/project", "dev", "alpha\nbravo"
        )

    @pytest.mark.asyncio
    async def test_ui_stage_log_called_on_start_and_finish(self) -> None:
        """UI receives starting and finished messages."""
        kwargs = _base_kwargs()
        ui = kwargs["ui"]

        async def fake_query(**_kw):
            yield _message([_text_block("ok")])

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            await run_agent(**kwargs)

        # First call is the "starting" message
        first_call_msg = ui.stage_log.call_args_list[0][0][1]
        assert "starting" in first_call_msg

        # Last call is the "finished" message
        last_call_msg = ui.stage_log.call_args_list[-1][0][1]
        assert "finished" in last_call_msg

    @pytest.mark.asyncio
    async def test_ui_stage_log_on_retry_failure(self) -> None:
        """UI receives a warning when an attempt fails and retries."""
        attempt_count = 0

        async def flaky_query(**_kw):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ConnectionError("boom")
            yield _message([_text_block("ok")])

        kwargs = _base_kwargs()
        ui = kwargs["ui"]

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.artifacts"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            await run_agent(**kwargs)

        # Find the retry warning call
        retry_calls = [
            c for c in ui.stage_log.call_args_list
            if "failed" in str(c) and "Retrying" in str(c)
        ]
        assert len(retry_calls) == 1
        retry_msg = retry_calls[0][0][1]
        assert "attempt 1/3" in retry_msg
        assert "boom" in retry_msg
