"""Tests for kindle.agent — Claude Agent SDK wrapper with retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
    """Return a mock UI with the methods run_agent calls."""
    ui = MagicMock()
    ui.stage_log = MagicMock()
    return ui


def _text_block(text: str) -> MagicMock:
    """Create a mock content block that has a .text attribute but no .name."""
    block = MagicMock(spec=[])  # empty spec — no attrs by default
    block.text = text
    return block


def _tool_block(name: str, tool_input: dict | None = None) -> MagicMock:
    """Create a mock content block that has a .name attribute (tool use)."""
    block = MagicMock(spec=[])
    block.name = name
    block.input = tool_input or {}
    return block


def _message(*blocks: MagicMock) -> MagicMock:
    """Wrap content blocks in a message-like object."""
    msg = MagicMock()
    msg.content = list(blocks)
    return msg


async def _run_agent_defaults(ui: MagicMock | None = None, **overrides) -> AgentResult:
    """Call run_agent with sensible defaults, patching SDK dependencies."""
    kw: dict = dict(
        persona="tester",
        system_prompt="You are a tester.",
        user_prompt="Do the thing.",
        cwd="/tmp",
        project_dir="/tmp/proj",
        stage="test_stage",
        ui=ui or _make_ui(),
    )
    kw.update(overrides)
    return await run_agent(**kw)


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


class TestAgentResult:
    """Basic dataclass behaviour."""

    def test_default_elapsed_seconds(self) -> None:
        r = AgentResult(text="hi", tool_calls=[], raw_messages=[])
        assert r.elapsed_seconds == 0.0

    def test_default_turns_used(self) -> None:
        r = AgentResult(text="hi", tool_calls=[], raw_messages=[])
        assert r.turns_used == 0

    def test_all_fields(self) -> None:
        r = AgentResult(
            text="done",
            tool_calls=[{"tool": "Read"}],
            raw_messages=["m1", "m2"],
            elapsed_seconds=3.5,
            turns_used=2,
        )
        assert r.text == "done"
        assert r.tool_calls == [{"tool": "Read"}]
        assert r.raw_messages == ["m1", "m2"]
        assert r.elapsed_seconds == 3.5
        assert r.turns_used == 2


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    """Unit tests for the internal _process_message helper."""

    def test_text_block_appended(self) -> None:
        """Text blocks are added to text_parts."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _message(_text_block("hello world"))
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "stg", ui, "/tmp")

        assert text_parts == ["hello world"]
        assert tool_calls == []

    def test_tool_block_appended(self) -> None:
        """Tool-use blocks are added to tool_calls."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _message(_tool_block("Bash", {"command": "ls"}))
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "stg", ui, "/tmp")

        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]
        assert text_parts == []

    def test_mixed_blocks(self) -> None:
        """A message with both text and tool blocks populates both lists."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _message(_text_block("thinking…"), _tool_block("Grep"))
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "stg", ui, "/tmp")

        assert text_parts == ["thinking…"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Grep"

    def test_message_without_content_attr(self) -> None:
        """Messages lacking a .content attribute are silently skipped."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = MagicMock(spec=[])  # no .content
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "stg", ui, "/tmp")

        assert text_parts == []
        assert tool_calls == []

    def test_empty_text_not_appended(self) -> None:
        """A text block with empty string is not appended."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        block = MagicMock(spec=[])
        block.text = ""
        msg = _message(block)
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "stg", ui, "/tmp")

        assert text_parts == []

    def test_text_preview_logged(self) -> None:
        """Stage log receives a truncated preview of text blocks."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        long_text = "A" * 300
        msg = _message(_text_block(long_text))
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "stg", ui, "/tmp")

        # The preview should be at most 200 chars
        logged_preview = ui.stage_log.call_args_list[-1].args[1]
        assert len(logged_preview) <= 200

    def test_tool_use_logged(self) -> None:
        """Tool-use blocks produce a stage_log entry with the tool name."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _message(_tool_block("Write"))
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "stg", ui, "/tmp")

        ui.stage_log.assert_called()
        logged = ui.stage_log.call_args_list[-1].args[1]
        assert "Write" in logged


# ---------------------------------------------------------------------------
# run_agent — happy path
# ---------------------------------------------------------------------------


class TestRunAgentHappyPath:
    """Successful single-attempt execution."""

    @pytest.mark.asyncio
    async def test_returns_agent_result(self) -> None:
        msg = _message(_text_block("Hello from agent"))
        ui = _make_ui()

        async def fake_query(**kwargs):
            yield msg

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert isinstance(result, AgentResult)
        assert result.text == "Hello from agent"

    @pytest.mark.asyncio
    async def test_turns_used_equals_message_count(self) -> None:
        msg1 = _message(_text_block("first"))
        msg2 = _message(_text_block("second"))
        ui = _make_ui()

        async def fake_query(**kwargs):
            yield msg1
            yield msg2

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert result.turns_used == 2
        assert result.text == "first\nsecond"

    @pytest.mark.asyncio
    async def test_raw_messages_collected(self) -> None:
        msg = _message(_text_block("hi"))
        ui = _make_ui()

        async def fake_query(**kwargs):
            yield msg

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert result.raw_messages == [msg]

    @pytest.mark.asyncio
    async def test_tool_calls_collected(self) -> None:
        msg = _message(_tool_block("Bash", {"command": "echo hi"}))
        ui = _make_ui()

        async def fake_query(**kwargs):
            yield msg

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert result.tool_calls == [{"tool": "Bash", "input": {"command": "echo hi"}}]

    @pytest.mark.asyncio
    async def test_elapsed_seconds_is_positive(self) -> None:
        msg = _message(_text_block("ok"))
        ui = _make_ui()

        async def fake_query(**kwargs):
            yield msg

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert result.elapsed_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_save_log_called(self) -> None:
        msg = _message(_text_block("output"))
        ui = _make_ui()

        async def fake_query(**kwargs):
            yield msg

        mock_save = MagicMock()
        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log", mock_save),
        ):
            await _run_agent_defaults(ui=ui, project_dir="/proj", stage="dev")

        mock_save.assert_called_once_with("/proj", "dev", "output")

    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty_text(self) -> None:
        """An agent that produces no messages returns empty text."""
        ui = _make_ui()

        async def fake_query(**kwargs):
            return
            yield

        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert result.text == ""
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# run_agent — options / arguments
# ---------------------------------------------------------------------------


class TestRunAgentOptions:
    """Verify ClaudeAgentOptions and query receive the correct arguments."""

    @pytest.mark.asyncio
    async def test_default_allowed_tools(self) -> None:
        """When allowed_tools is None, the seven default tools are used."""
        ui = _make_ui()

        async def fake_query(**kwargs):
            return
            yield

        mock_opts = MagicMock()
        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions", mock_opts),
            patch("kindle.agent.artifacts.save_log"),
        ):
            await _run_agent_defaults(ui=ui, allowed_tools=None)

        call_kwargs = mock_opts.call_args.kwargs
        assert call_kwargs["allowed_tools"] == [
            "Read",
            "Write",
            "Edit",
            "MultiEdit",
            "Bash",
            "Glob",
            "Grep",
        ]

    @pytest.mark.asyncio
    async def test_custom_allowed_tools(self) -> None:
        """Explicit allowed_tools are passed through."""
        ui = _make_ui()

        async def fake_query(**kwargs):
            return
            yield

        mock_opts = MagicMock()
        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions", mock_opts),
            patch("kindle.agent.artifacts.save_log"),
        ):
            await _run_agent_defaults(ui=ui, allowed_tools=["Read", "Grep"])

        call_kwargs = mock_opts.call_args.kwargs
        assert call_kwargs["allowed_tools"] == ["Read", "Grep"]

    @pytest.mark.asyncio
    async def test_model_included_when_set(self) -> None:
        """When model is provided, it is passed to ClaudeAgentOptions."""
        ui = _make_ui()

        async def fake_query(**kwargs):
            return
            yield

        mock_opts = MagicMock()
        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions", mock_opts),
            patch("kindle.agent.artifacts.save_log"),
        ):
            await _run_agent_defaults(ui=ui, model="claude-sonnet-4-20250514")

        call_kwargs = mock_opts.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_model_omitted_when_none(self) -> None:
        """When model is None, it is NOT passed to ClaudeAgentOptions."""
        ui = _make_ui()

        async def fake_query(**kwargs):
            return
            yield

        mock_opts = MagicMock()
        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions", mock_opts),
            patch("kindle.agent.artifacts.save_log"),
        ):
            await _run_agent_defaults(ui=ui, model=None)

        call_kwargs = mock_opts.call_args.kwargs
        assert "model" not in call_kwargs

    @pytest.mark.asyncio
    async def test_max_turns_passed(self) -> None:
        """max_turns is forwarded to ClaudeAgentOptions."""
        ui = _make_ui()

        async def fake_query(**kwargs):
            return
            yield

        mock_opts = MagicMock()
        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions", mock_opts),
            patch("kindle.agent.artifacts.save_log"),
        ):
            await _run_agent_defaults(ui=ui, max_turns=100)

        call_kwargs = mock_opts.call_args.kwargs
        assert call_kwargs["max_turns"] == 100

    @pytest.mark.asyncio
    async def test_permission_mode_is_bypass(self) -> None:
        """Permission mode is always bypassPermissions."""
        ui = _make_ui()

        async def fake_query(**kwargs):
            return
            yield

        mock_opts = MagicMock()
        with (
            patch("kindle.agent.query", side_effect=fake_query),
            patch("kindle.agent.ClaudeAgentOptions", mock_opts),
            patch("kindle.agent.artifacts.save_log"),
        ):
            await _run_agent_defaults(ui=ui)

        call_kwargs = mock_opts.call_args.kwargs
        assert call_kwargs["permission_mode"] == "bypassPermissions"

    @pytest.mark.asyncio
    async def test_query_receives_user_prompt(self) -> None:
        """query() is called with prompt=user_prompt."""
        ui = _make_ui()

        async def fake_query(**kwargs):
            return
            yield

        mock_query = MagicMock(side_effect=fake_query)
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
        ):
            await _run_agent_defaults(ui=ui, user_prompt="Build an app")

        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["prompt"] == "Build an app"


# ---------------------------------------------------------------------------
# run_agent — retry on transient errors
# ---------------------------------------------------------------------------


class TestRunAgentRetry:
    """Retry logic for transient failures (ConnectionError, TimeoutError, OSError).

    IMPORTANT: Error-raising side effects must be *sync* functions (not async
    coroutines without ``yield``).  ``query()`` is consumed via ``async for``,
    so the mock's return value must be an async-iterable or the call must raise
    synchronously.  A bare ``async def`` that raises (without ``yield``) returns
    a *coroutine*, which is **not** an async-iterable and causes ``TypeError``.
    """

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self) -> None:
        """ConnectionError triggers retry; success on second attempt returns result."""
        ui = _make_ui()
        msg = _message(_text_block("recovered"))
        call_count = 0

        def flaky_query(**kwargs):
            """Sync function: raises on first call, returns async gen on second."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("refused")

            async def gen():
                yield msg

            return gen()

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert result.text == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_timeout_error(self) -> None:
        """TimeoutError triggers retry."""
        ui = _make_ui()
        msg = _message(_text_block("ok"))
        call_count = 0

        def flaky_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")

            async def gen():
                yield msg

            return gen()

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_os_error(self) -> None:
        """OSError triggers retry."""
        ui = _make_ui()
        msg = _message(_text_block("ok"))
        call_count = 0

        def flaky_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("network unreachable")

            async def gen():
                yield msg

            return gen()

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _run_agent_defaults(ui=ui)

        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        """Each retry sleeps with exponential backoff: base^attempt."""
        ui = _make_ui()

        def always_fail(**kwargs):
            raise ConnectionError("down")

        mock_sleep = AsyncMock()
        with (
            patch("kindle.agent.query", side_effect=always_fail),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", mock_sleep),
        ):
            with pytest.raises(RuntimeError):
                await _run_agent_defaults(ui=ui)

        # Attempts 1, 2, 3 → delays base^1, base^2, base^3
        expected_delays = [RETRY_BACKOFF_BASE**i for i in range(1, MAX_RETRIES + 1)]
        actual_delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_runtime_error(self) -> None:
        """After MAX_RETRIES transient failures, RuntimeError is raised."""
        ui = _make_ui()

        def always_fail(**kwargs):
            raise ConnectionError("connection refused")

        mock_sleep = AsyncMock()
        with (
            patch("kindle.agent.query", side_effect=always_fail),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", mock_sleep),
        ):
            with pytest.raises(RuntimeError, match=f"failed after {MAX_RETRIES} attempts"):
                await _run_agent_defaults(ui=ui, persona="researcher")

    @pytest.mark.asyncio
    async def test_exhausted_retries_includes_last_error_message(self) -> None:
        """The RuntimeError message includes the text of the last exception."""
        ui = _make_ui()
        call_count = 0

        def always_fail(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError(f"failure #{call_count}")

        mock_sleep = AsyncMock()
        with (
            patch("kindle.agent.query", side_effect=always_fail),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", mock_sleep),
        ):
            with pytest.raises(RuntimeError, match=f"failure #{MAX_RETRIES}"):
                await _run_agent_defaults(ui=ui)

    @pytest.mark.asyncio
    async def test_non_transient_error_not_retried(self) -> None:
        """Errors not in (ConnectionError, TimeoutError, OSError) propagate immediately."""
        ui = _make_ui()
        call_count = 0

        def bad_query(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid input")

        mock_sleep = AsyncMock()
        with (
            patch("kindle.agent.query", side_effect=bad_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", mock_sleep),
        ):
            with pytest.raises(ValueError, match="invalid input"):
                await _run_agent_defaults(ui=ui)

        assert call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_logs_warning(self) -> None:
        """Each retry attempt logs a warning via ui.stage_log."""
        ui = _make_ui()
        msg = _message(_text_block("ok"))
        call_count = 0

        def flaky_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("down")

            async def gen():
                yield msg

            return gen()

        with (
            patch("kindle.agent.query", side_effect=flaky_query),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            await _run_agent_defaults(ui=ui, stage="dev")

        # Check that warning-style log entries were written for retries
        log_calls = [str(c) for c in ui.stage_log.call_args_list]
        retry_logs = [c for c in log_calls if "Retrying" in c]
        assert len(retry_logs) == 2

    @pytest.mark.asyncio
    async def test_query_called_max_retries_times_on_total_failure(self) -> None:
        """query() is invoked exactly MAX_RETRIES times when every attempt fails."""
        ui = _make_ui()
        call_count = 0

        def always_fail(**kwargs):
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timeout")

        mock_sleep = AsyncMock()
        with (
            patch("kindle.agent.query", side_effect=always_fail),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", mock_sleep),
        ):
            with pytest.raises(RuntimeError):
                await _run_agent_defaults(ui=ui)

        assert call_count == MAX_RETRIES


# ---------------------------------------------------------------------------
# run_agent — error during iteration (mid-stream)
# ---------------------------------------------------------------------------


class TestRunAgentMidStreamError:
    """Transient errors raised during async iteration (not at call time)."""

    @pytest.mark.asyncio
    async def test_mid_stream_connection_error_retried(self) -> None:
        """ConnectionError raised mid-iteration triggers retry."""
        ui = _make_ui()
        call_count = 0

        def flaky_stream(**kwargs):
            nonlocal call_count
            call_count += 1

            async def gen_fail():
                yield _message(_text_block("partial"))
                raise ConnectionError("dropped")

            async def gen_ok():
                yield _message(_text_block("complete"))

            if call_count == 1:
                return gen_fail()
            return gen_ok()

        with (
            patch("kindle.agent.query", side_effect=flaky_stream),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await _run_agent_defaults(ui=ui)

        # Second attempt succeeds cleanly
        assert result.text == "complete"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants."""

    def test_max_retries(self) -> None:
        assert MAX_RETRIES == 3

    def test_retry_backoff_base(self) -> None:
        assert RETRY_BACKOFF_BASE == 2.0
