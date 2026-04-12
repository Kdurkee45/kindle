"""Tests for kindle.agent — Claude Agent SDK wrapper with retry and streaming."""

from __future__ import annotations

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


def _make_ui() -> MagicMock:
    """Return a mock UI instance with the stage_log method."""
    return MagicMock()


def _text_block(text: str) -> SimpleNamespace:
    """Simulate a content block with a .text attribute."""
    return SimpleNamespace(text=text)


def _tool_block(name: str, tool_input: dict[str, Any] | None = None) -> SimpleNamespace:
    """Simulate a content block with a .name (tool-use) attribute."""
    return SimpleNamespace(name=name, input=tool_input or {})


def _sdk_message(*blocks: SimpleNamespace) -> SimpleNamespace:
    """Build a fake SDK message with a .content list."""
    return SimpleNamespace(content=list(blocks))


async def _async_iter(items: list[Any]):
    """Helper that yields items as an async iterator."""
    for item in items:
        yield item


def _base_kwargs() -> dict[str, Any]:
    """Minimal keyword arguments for run_agent()."""
    return dict(
        persona="tester",
        system_prompt="You are a tester.",
        user_prompt="Write tests.",
        cwd="/tmp",
        project_dir="/tmp/proj",
        stage="qa",
        ui=_make_ui(),
    )


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


class TestAgentResult:
    """Verify the AgentResult dataclass fields and defaults."""

    def test_required_fields(self) -> None:
        result = AgentResult(text="hi", tool_calls=[], raw_messages=[])
        assert result.text == "hi"
        assert result.tool_calls == []
        assert result.raw_messages == []

    def test_default_elapsed_seconds(self) -> None:
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.elapsed_seconds == 0.0

    def test_default_turns_used(self) -> None:
        result = AgentResult(text="", tool_calls=[], raw_messages=[])
        assert result.turns_used == 0

    def test_custom_elapsed_and_turns(self) -> None:
        result = AgentResult(
            text="done",
            tool_calls=[{"tool": "Read", "input": {}}],
            raw_messages=["m1", "m2"],
            elapsed_seconds=5.2,
            turns_used=2,
        )
        assert result.elapsed_seconds == 5.2
        assert result.turns_used == 2
        assert len(result.tool_calls) == 1
        assert len(result.raw_messages) == 2

    def test_equality(self) -> None:
        a = AgentResult(text="x", tool_calls=[], raw_messages=[], elapsed_seconds=1.0, turns_used=1)
        b = AgentResult(text="x", tool_calls=[], raw_messages=[], elapsed_seconds=1.0, turns_used=1)
        assert a == b


# ---------------------------------------------------------------------------
# _process_message()
# ---------------------------------------------------------------------------


class TestProcessMessage:
    """Verify text/tool extraction from individual SDK messages."""

    def test_extracts_text_block(self) -> None:
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _sdk_message(_text_block("hello world"))

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == ["hello world"]
        assert tool_calls == []
        ui.stage_log.assert_called()

    def test_extracts_tool_use_block(self) -> None:
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _sdk_message(_tool_block("Bash", {"command": "ls"}))

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]

    def test_extracts_mixed_blocks(self) -> None:
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _sdk_message(
            _text_block("thinking…"),
            _tool_block("Read", {"path": "/tmp"}),
            _text_block("done"),
        )

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == ["thinking…", "done"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Read"

    def test_skips_message_without_content(self) -> None:
        """Messages lacking a .content attribute are silently ignored."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = SimpleNamespace(role="system")  # no .content

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == []
        ui.stage_log.assert_not_called()

    def test_skips_empty_text_block(self) -> None:
        """Blocks with text='' are not appended."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _sdk_message(SimpleNamespace(text=""))

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert text_parts == []

    def test_tool_block_without_input(self) -> None:
        """Tool blocks missing .input default to empty dict."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        # Create a block with .name but no .input
        block = SimpleNamespace(name="Glob")
        msg = _sdk_message(block)

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        assert tool_calls == [{"tool": "Glob", "input": {}}]

    def test_text_preview_truncates_long_text(self) -> None:
        """stage_log preview should receive at most 200 chars, newlines replaced."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        long_text = "A" * 300 + "\nmore"
        msg = _sdk_message(_text_block(long_text))

        _process_message(msg, text_parts, tool_calls, "dev", ui, "/tmp/proj")

        # The full text is still captured
        assert text_parts[0] == long_text
        # The preview passed to stage_log is truncated
        logged_preview = ui.stage_log.call_args[0][1]
        assert len(logged_preview) <= 200
        assert "\n" not in logged_preview


# ---------------------------------------------------------------------------
# run_agent() — happy path
# ---------------------------------------------------------------------------


class TestRunAgentHappyPath:
    """Verify run_agent completes successfully with mocked SDK."""

    @pytest.fixture
    def _patch_deps(self):
        """Patch claude_agent_sdk.query and artifacts.save_log."""
        messages = [
            _sdk_message(_text_block("Hello from agent")),
            _sdk_message(_tool_block("Bash", {"command": "echo hi"})),
            _sdk_message(_text_block("All done")),
        ]
        mock_query = MagicMock(return_value=_async_iter(messages))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log") as mock_save_log,
            patch("kindle.agent.ClaudeAgentOptions") as mock_opts_cls,
        ):
            yield {
                "query": mock_query,
                "save_log": mock_save_log,
                "opts_cls": mock_opts_cls,
            }

    async def test_returns_agent_result(self, _patch_deps: dict) -> None:
        result = await run_agent(**_base_kwargs())
        assert isinstance(result, AgentResult)

    async def test_captures_text(self, _patch_deps: dict) -> None:
        result = await run_agent(**_base_kwargs())
        assert "Hello from agent" in result.text
        assert "All done" in result.text

    async def test_captures_tool_calls(self, _patch_deps: dict) -> None:
        result = await run_agent(**_base_kwargs())
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "Bash"

    async def test_raw_messages_count(self, _patch_deps: dict) -> None:
        result = await run_agent(**_base_kwargs())
        assert len(result.raw_messages) == 3

    async def test_turns_used_equals_message_count(self, _patch_deps: dict) -> None:
        result = await run_agent(**_base_kwargs())
        assert result.turns_used == 3

    async def test_elapsed_seconds_positive(self, _patch_deps: dict) -> None:
        result = await run_agent(**_base_kwargs())
        assert result.elapsed_seconds >= 0.0

    async def test_save_log_called(self, _patch_deps: dict) -> None:
        kwargs = _base_kwargs()
        await run_agent(**kwargs)
        _patch_deps["save_log"].assert_called_once_with(
            kwargs["project_dir"],
            kwargs["stage"],
            "Hello from agent\nAll done",
        )

    async def test_stage_log_messages_include_start_and_finish(self, _patch_deps: dict) -> None:
        kwargs = _base_kwargs()
        await run_agent(**kwargs)
        ui = kwargs["ui"]
        log_messages = [c.args[1] for c in ui.stage_log.call_args_list]
        assert any("starting" in m for m in log_messages)
        assert any("finished" in m for m in log_messages)


# ---------------------------------------------------------------------------
# run_agent() — ClaudeAgentOptions construction
# ---------------------------------------------------------------------------


class TestRunAgentOptions:
    """Verify that ClaudeAgentOptions receives the right kwargs."""

    async def test_default_allowed_tools(self) -> None:
        """When allowed_tools is None, the default 7-tool list is used."""
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions") as mock_opts_cls,
        ):
            await run_agent(**_base_kwargs())
            opts_call = mock_opts_cls.call_args
            assert opts_call.kwargs["allowed_tools"] == [
                "Read",
                "Write",
                "Edit",
                "MultiEdit",
                "Bash",
                "Glob",
                "Grep",
            ]

    async def test_custom_allowed_tools(self) -> None:
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions") as mock_opts_cls,
        ):
            kwargs = _base_kwargs()
            kwargs["allowed_tools"] = ["Read", "Bash"]
            await run_agent(**kwargs)
            opts_call = mock_opts_cls.call_args
            assert opts_call.kwargs["allowed_tools"] == ["Read", "Bash"]

    async def test_system_prompt_passed(self) -> None:
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions") as mock_opts_cls,
        ):
            kwargs = _base_kwargs()
            kwargs["system_prompt"] = "Custom system prompt"
            await run_agent(**kwargs)
            opts_call = mock_opts_cls.call_args
            assert opts_call.kwargs["system_prompt"] == "Custom system prompt"

    async def test_model_passed_when_provided(self) -> None:
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions") as mock_opts_cls,
        ):
            kwargs = _base_kwargs()
            kwargs["model"] = "claude-sonnet-4-20250514"
            await run_agent(**kwargs)
            opts_call = mock_opts_cls.call_args
            assert opts_call.kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_model_omitted_when_none(self) -> None:
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions") as mock_opts_cls,
        ):
            await run_agent(**_base_kwargs())
            opts_call = mock_opts_cls.call_args
            assert "model" not in opts_call.kwargs

    async def test_max_turns_forwarded(self) -> None:
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions") as mock_opts_cls,
        ):
            kwargs = _base_kwargs()
            kwargs["max_turns"] = 10
            await run_agent(**kwargs)
            opts_call = mock_opts_cls.call_args
            assert opts_call.kwargs["max_turns"] == 10

    async def test_permission_mode_bypass(self) -> None:
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions") as mock_opts_cls,
        ):
            await run_agent(**_base_kwargs())
            opts_call = mock_opts_cls.call_args
            assert opts_call.kwargs["permission_mode"] == "bypassPermissions"

    async def test_user_prompt_passed_to_query(self) -> None:
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            kwargs = _base_kwargs()
            kwargs["user_prompt"] = "Build me a CLI"
            await run_agent(**kwargs)
            mock_query.assert_called_once()
            assert mock_query.call_args.kwargs["prompt"] == "Build me a CLI"


# ---------------------------------------------------------------------------
# run_agent() — retry logic
# ---------------------------------------------------------------------------


class TestRunAgentRetries:
    """Verify exponential backoff and retry behaviour on transient failures."""

    @pytest.mark.parametrize("exc_type", [ConnectionError, TimeoutError, OSError])
    async def test_retries_on_transient_error(self, exc_type: type) -> None:
        """Transient errors trigger retry; success on second attempt works."""
        attempt = 0

        async def _flaky_query(**kwargs):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise exc_type("transient")
            return _async_iter([_sdk_message(_text_block("ok"))])

        # _flaky_query raises on first call, then returns an async generator
        # We need to handle the fact that query() should return an async iterable
        # but the first call raises before returning
        call_count = 0

        async def _flaky_iter(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise exc_type("transient")
            async for item in _async_iter([_sdk_message(_text_block("ok"))]):
                yield item

        mock_query = MagicMock(side_effect=lambda **kw: _flaky_iter(**kw))

        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await run_agent(**_base_kwargs())
            assert result.text == "ok"
            mock_sleep.assert_called_once_with(RETRY_BACKOFF_BASE**1)

    async def test_raises_runtime_error_after_max_retries(self) -> None:
        """All attempts exhausted → RuntimeError with persona info."""

        async def _always_fail(**kwargs):
            raise ConnectionError("down")
            yield

        mock_query = MagicMock(side_effect=lambda **kw: _always_fail(**kw))

        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match=r"tester.*failed after 3 attempts"):
                await run_agent(**_base_kwargs())

    async def test_exponential_backoff_delays(self) -> None:
        """Each retry sleeps for RETRY_BACKOFF_BASE ** attempt."""
        call_count = 0

        async def _fail_then_succeed(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("slow")
            async for item in _async_iter([_sdk_message(_text_block("ok"))]):
                yield item

        mock_query = MagicMock(side_effect=lambda **kw: _fail_then_succeed(**kw))

        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await run_agent(**_base_kwargs())
            assert result.text == "ok"
            assert mock_sleep.call_args_list == [
                call(RETRY_BACKOFF_BASE**1),
                call(RETRY_BACKOFF_BASE**2),
            ]

    async def test_retry_logs_warnings(self) -> None:
        """Each retry emits a yellow warning via ui.stage_log."""
        call_count = 0

        async def _fail_once(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("oops")
            async for item in _async_iter([_sdk_message(_text_block("ok"))]):
                yield item

        mock_query = MagicMock(side_effect=lambda **kw: _fail_once(**kw))

        kwargs = _base_kwargs()
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            await run_agent(**kwargs)
            ui = kwargs["ui"]
            log_messages = [c.args[1] for c in ui.stage_log.call_args_list]
            retry_warnings = [m for m in log_messages if "failed" in m and "attempt" in m]
            assert len(retry_warnings) == 1
            assert "1/3" in retry_warnings[0]

    async def test_non_transient_error_propagates_immediately(self) -> None:
        """Errors outside the retry set are not retried."""

        async def _value_error(**kwargs):
            raise ValueError("bad input")
            yield

        mock_query = MagicMock(side_effect=lambda **kw: _value_error(**kw))

        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
            patch("kindle.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(ValueError, match="bad input"):
                await run_agent(**_base_kwargs())
            mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# run_agent() — streaming and message capture
# ---------------------------------------------------------------------------


class TestRunAgentStreaming:
    """Verify streaming message capture produces correct AgentResult."""

    async def test_multiple_text_blocks_joined(self) -> None:
        messages = [
            _sdk_message(_text_block("line one")),
            _sdk_message(_text_block("line two")),
        ]
        mock_query = MagicMock(return_value=_async_iter(messages))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            result = await run_agent(**_base_kwargs())
            assert result.text == "line one\nline two"

    async def test_empty_stream_returns_empty_text(self) -> None:
        mock_query = MagicMock(return_value=_async_iter([]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            result = await run_agent(**_base_kwargs())
            assert result.text == ""
            assert result.tool_calls == []
            assert result.raw_messages == []
            assert result.turns_used == 0

    async def test_tool_only_messages_have_empty_text(self) -> None:
        messages = [_sdk_message(_tool_block("Grep", {"pattern": "foo"}))]
        mock_query = MagicMock(return_value=_async_iter(messages))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            result = await run_agent(**_base_kwargs())
            assert result.text == ""
            assert len(result.tool_calls) == 1

    async def test_messages_without_content_are_skipped(self) -> None:
        """SDK messages lacking .content are collected in raw but produce no text."""
        no_content_msg = SimpleNamespace(role="system")
        text_msg = _sdk_message(_text_block("real"))
        mock_query = MagicMock(return_value=_async_iter([no_content_msg, text_msg]))
        with (
            patch("kindle.agent.query", mock_query),
            patch("kindle.agent.artifacts.save_log"),
            patch("kindle.agent.ClaudeAgentOptions"),
        ):
            result = await run_agent(**_base_kwargs())
            assert result.text == "real"
            assert result.turns_used == 2  # both messages counted


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify module-level retry configuration."""

    def test_max_retries_is_three(self) -> None:
        assert MAX_RETRIES == 3

    def test_backoff_base_is_two(self) -> None:
        assert RETRY_BACKOFF_BASE == 2.0
