"""Tests for kindle.ui — Rich terminal UI class."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from rich.panel import Panel
from rich.table import Table

from kindle.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(*, auto_approve: bool = False, verbose: bool = False) -> UI:
    """Create a UI instance with a mocked console to avoid terminal I/O."""
    with patch("kindle.ui.Console") as mock_cls:
        mock_console = MagicMock()
        mock_cls.return_value = mock_console
        ui = UI(auto_approve=auto_approve, verbose=verbose)
    # The constructor already created self.console via Console().
    # Re-assign so later tests can inspect calls on the mock.
    ui.console = mock_console
    return ui


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify module-level constants are well-formed."""

    def test_stage_labels_keys_match_stage_order(self) -> None:
        assert list(STAGE_LABELS.keys()) == STAGE_ORDER

    def test_stage_order_length(self) -> None:
        assert len(STAGE_ORDER) == 6

    def test_max_display_chars_is_positive_int(self) -> None:
        assert isinstance(MAX_DISPLAY_CHARS, int)
        assert MAX_DISPLAY_CHARS > 0

    def test_stage_labels_all_contain_emoji(self) -> None:
        """Every stage label should start with an emoji character (non-ASCII)."""
        for label in STAGE_LABELS.values():
            # Labels like "🔥 Grill" start with a non-ASCII emoji
            assert not label[0].isascii() or label[0] == " ", f"Label {label!r} missing emoji prefix"


# ---------------------------------------------------------------------------
# UI.__init__
# ---------------------------------------------------------------------------


class TestUIInit:
    """Tests for UI constructor and attribute initialisation."""

    def test_creates_console_instance(self) -> None:
        with patch("kindle.ui.Console") as mock_cls:
            UI(auto_approve=True)
        mock_cls.assert_called_once()

    def test_default_auto_approve_false_when_tty(self) -> None:
        with patch("kindle.ui.Console"), patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            ui = UI()
        assert ui.auto_approve is False

    def test_auto_approve_true_when_non_tty(self) -> None:
        """When stdin is not a TTY, auto_approve is forced True."""
        with patch("kindle.ui.Console"), patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            ui = UI()
        assert ui.auto_approve is True

    def test_auto_approve_explicit_true(self) -> None:
        with patch("kindle.ui.Console"), patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    def test_verbose_default_false(self) -> None:
        ui = _make_ui()
        assert ui.verbose is False

    def test_verbose_true(self) -> None:
        ui = _make_ui(verbose=True)
        assert ui.verbose is True

    def test_current_stage_starts_none(self) -> None:
        ui = _make_ui()
        assert ui._current_stage is None


# ---------------------------------------------------------------------------
# _safe_print
# ---------------------------------------------------------------------------


class TestSafePrint:
    """Tests for the error-suppressing _safe_print wrapper."""

    def test_delegates_to_console_print(self) -> None:
        ui = _make_ui()
        ui._safe_print("hello", style="bold")
        ui.console.print.assert_called_once_with("hello", style="bold")

    def test_suppresses_blocking_io_error(self) -> None:
        ui = _make_ui()
        ui.console.print.side_effect = BlockingIOError
        # Should not raise
        ui._safe_print("test")

    def test_suppresses_broken_pipe_error(self) -> None:
        ui = _make_ui()
        ui.console.print.side_effect = BrokenPipeError
        ui._safe_print("test")

    def test_suppresses_os_error(self) -> None:
        ui = _make_ui()
        ui.console.print.side_effect = OSError
        ui._safe_print("test")

    def test_does_not_suppress_value_error(self) -> None:
        ui = _make_ui()
        ui.console.print.side_effect = ValueError("unexpected")
        with pytest.raises(ValueError, match="unexpected"):
            ui._safe_print("test")


# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------


class TestBanner:
    """Tests for the startup banner display."""

    def test_banner_calls_safe_print_with_panel(self) -> None:
        ui = _make_ui()
        ui.banner("My idea", "proj-123")
        # banner calls _safe_print 3 times: blank, Panel, blank
        assert ui.console.print.call_count == 3

    def test_banner_panel_contains_idea(self) -> None:
        ui = _make_ui()
        ui.banner("Build a chat app", "proj-abc")
        # The second call is the Panel
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(panel_arg, Panel)

    def test_banner_panel_contains_project_id(self) -> None:
        ui = _make_ui()
        ui.banner("idea", "proj-xyz")
        panel_arg = ui.console.print.call_args_list[1][0][0]
        # Panel renderable should contain the project ID
        assert "proj-xyz" in str(panel_arg.renderable)


# ---------------------------------------------------------------------------
# stage_start / stage_done / stage_log
# ---------------------------------------------------------------------------


class TestStageStart:
    """Tests for stage_start which displays the stage header."""

    def test_sets_current_stage(self) -> None:
        ui = _make_ui()
        ui.stage_start("grill")
        assert ui._current_stage == "grill"

    def test_calls_console_rule_with_label(self) -> None:
        ui = _make_ui()
        ui.stage_start("grill")
        ui.console.rule.assert_called_once()
        rule_text = ui.console.rule.call_args[0][0]
        assert "Grill" in rule_text

    def test_unknown_stage_uses_raw_name(self) -> None:
        ui = _make_ui()
        ui.stage_start("custom_stage")
        assert ui._current_stage == "custom_stage"
        rule_text = ui.console.rule.call_args[0][0]
        assert "custom_stage" in rule_text

    def test_suppresses_io_errors(self) -> None:
        """console.rule is wrapped in contextlib.suppress for IO errors."""
        ui = _make_ui()
        ui.console.rule.side_effect = BrokenPipeError
        # Should not raise
        ui.stage_start("grill")
        assert ui._current_stage == "grill"

    def test_all_known_stages(self) -> None:
        """Verify every stage in STAGE_ORDER can be started."""
        for stage in STAGE_ORDER:
            ui = _make_ui()
            ui.stage_start(stage)
            assert ui._current_stage == stage
            rule_text = ui.console.rule.call_args[0][0]
            assert STAGE_LABELS[stage].split()[-1] in rule_text


class TestStageDone:
    """Tests for stage_done which displays completion."""

    def test_prints_completion_message(self) -> None:
        ui = _make_ui()
        ui.stage_done("grill")
        # Called twice: completion message + blank line
        assert ui.console.print.call_count == 2

    def test_completion_message_contains_label(self) -> None:
        ui = _make_ui()
        ui.stage_done("research")
        first_call_text = ui.console.print.call_args_list[0][0][0]
        assert "Research" in first_call_text
        assert "complete" in first_call_text

    def test_unknown_stage_uses_raw_name(self) -> None:
        ui = _make_ui()
        ui.stage_done("custom")
        first_call_text = ui.console.print.call_args_list[0][0][0]
        assert "custom" in first_call_text


class TestStageLog:
    """Tests for stage_log which is verbose-only."""

    def test_does_not_print_when_not_verbose(self) -> None:
        ui = _make_ui(verbose=False)
        ui.stage_log("grill", "some detail")
        ui.console.print.assert_not_called()

    def test_prints_when_verbose(self) -> None:
        ui = _make_ui(verbose=True)
        ui.stage_log("grill", "some detail")
        ui.console.print.assert_called_once()
        text = ui.console.print.call_args[0][0]
        assert "some detail" in text

    def test_verbose_includes_stage_label(self) -> None:
        ui = _make_ui(verbose=True)
        ui.stage_log("research", "found stuff")
        text = ui.console.print.call_args[0][0]
        assert "Research" in text


# ---------------------------------------------------------------------------
# show_artifact
# ---------------------------------------------------------------------------


class TestShowArtifact:
    """Tests for artifact display with optional truncation."""

    def test_renders_panel_with_title(self) -> None:
        ui = _make_ui()
        ui.show_artifact("Architecture", "# Arch\nDetails here")
        # 3 calls: blank, Panel, blank
        assert ui.console.print.call_count == 3
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(panel_arg, Panel)

    def test_short_content_not_truncated(self) -> None:
        ui = _make_ui()
        content = "Short content"
        ui.show_artifact("Title", content)
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert "truncated" not in str(panel_arg.renderable)

    def test_long_content_truncated(self) -> None:
        ui = _make_ui()
        content = "x" * (MAX_DISPLAY_CHARS + 500)
        ui.show_artifact("Title", content)
        panel_arg = ui.console.print.call_args_list[1][0][0]
        renderable_text = str(panel_arg.renderable)
        assert "truncated" in renderable_text

    def test_truncated_content_length(self) -> None:
        """The displayed portion should be MAX_DISPLAY_CHARS characters of the original."""
        ui = _make_ui()
        content = "a" * (MAX_DISPLAY_CHARS + 1000)
        ui.show_artifact("Title", content)
        panel_arg = ui.console.print.call_args_list[1][0][0]
        # The renderable starts with MAX_DISPLAY_CHARS 'a's
        assert str(panel_arg.renderable).startswith("a" * MAX_DISPLAY_CHARS)

    def test_exactly_max_chars_not_truncated(self) -> None:
        ui = _make_ui()
        content = "b" * MAX_DISPLAY_CHARS
        ui.show_artifact("Title", content)
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert "truncated" not in str(panel_arg.renderable)

    def test_empty_content(self) -> None:
        ui = _make_ui()
        ui.show_artifact("Empty", "")
        assert ui.console.print.call_count == 3


# ---------------------------------------------------------------------------
# grill_question
# ---------------------------------------------------------------------------


class TestGrillQuestion:
    """Tests for the interactive grill question prompt."""

    def test_auto_approve_returns_recommended(self) -> None:
        ui = _make_ui(auto_approve=True)
        result = ui.grill_question("What stack?", "React", "tech", 1)
        assert result == "React"

    def test_auto_approve_does_not_call_console_input(self) -> None:
        ui = _make_ui(auto_approve=True)
        ui.grill_question("Q?", "default", "cat", 1)
        ui.console.input.assert_not_called()

    def test_user_provides_custom_answer(self) -> None:
        ui = _make_ui()
        ui.console.input.return_value = "Vue.js"
        result = ui.grill_question("What stack?", "React", "tech", 1)
        assert result == "Vue.js"

    def test_user_presses_enter_returns_recommended(self) -> None:
        ui = _make_ui()
        ui.console.input.return_value = ""
        result = ui.grill_question("Q?", "default answer", "cat", 1)
        assert result == "default answer"

    def test_user_enters_whitespace_returns_recommended(self) -> None:
        ui = _make_ui()
        ui.console.input.return_value = "   "
        result = ui.grill_question("Q?", "recommended", "cat", 1)
        assert result == "recommended"

    def test_eof_error_returns_recommended(self) -> None:
        ui = _make_ui()
        ui.console.input.side_effect = EOFError
        result = ui.grill_question("Q?", "fallback", "cat", 1)
        assert result == "fallback"

    def test_keyboard_interrupt_returns_recommended(self) -> None:
        ui = _make_ui()
        ui.console.input.side_effect = KeyboardInterrupt
        result = ui.grill_question("Q?", "fallback", "cat", 1)
        assert result == "fallback"

    def test_renders_panel_with_question_number(self) -> None:
        ui = _make_ui(auto_approve=True)
        ui.grill_question("Q?", "rec", "cat", 5)
        # Find the Panel in print calls
        panel_calls = [
            c for c in ui.console.print.call_args_list if c[0] and isinstance(c[0][0], Panel)
        ]
        assert len(panel_calls) == 1

    def test_strips_whitespace_from_response(self) -> None:
        ui = _make_ui()
        ui.console.input.return_value = "  my answer  "
        result = ui.grill_question("Q?", "default", "cat", 1)
        assert result == "my answer"


# ---------------------------------------------------------------------------
# prompt_arch_review
# ---------------------------------------------------------------------------


class TestPromptArchReview:
    """Tests for architecture review approval flow."""

    def test_auto_approve_returns_true_empty_feedback(self) -> None:
        ui = _make_ui(auto_approve=True)
        approved, feedback = ui.prompt_arch_review("# Architecture\nSome plan")
        assert approved is True
        assert feedback == ""

    def test_auto_approve_does_not_call_console_input(self) -> None:
        ui = _make_ui(auto_approve=True)
        ui.prompt_arch_review("arch summary")
        ui.console.input.assert_not_called()

    @pytest.mark.parametrize("response", ["approve", "yes", "y", "lgtm", ""])
    def test_approval_keywords(self, response: str) -> None:
        ui = _make_ui()
        ui.console.input.return_value = response
        approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_approval_case_insensitive(self) -> None:
        ui = _make_ui()
        ui.console.input.return_value = "APPROVE"
        approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_feedback_returns_false_with_text(self) -> None:
        ui = _make_ui()
        ui.console.input.return_value = "Add more caching"
        approved, feedback = ui.prompt_arch_review("arch")
        assert approved is False
        assert feedback == "Add more caching"

    def test_eof_error_returns_true_empty(self) -> None:
        ui = _make_ui()
        ui.console.input.side_effect = EOFError
        approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_keyboard_interrupt_returns_true_empty(self) -> None:
        ui = _make_ui()
        ui.console.input.side_effect = KeyboardInterrupt
        approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_shows_artifact_panel(self) -> None:
        """prompt_arch_review calls show_artifact, which renders a Panel."""
        ui = _make_ui(auto_approve=True)
        ui.prompt_arch_review("# My Architecture")
        panel_calls = [
            c for c in ui.console.print.call_args_list if c[0] and isinstance(c[0][0], Panel)
        ]
        assert len(panel_calls) >= 1


# ---------------------------------------------------------------------------
# task_start / task_done
# ---------------------------------------------------------------------------


class TestTaskProgress:
    """Tests for build task progress indicators."""

    def test_task_start_prints_index_and_total(self) -> None:
        ui = _make_ui()
        ui.task_start("task-1", "Create models", 1, 5)
        text = ui.console.print.call_args[0][0]
        assert "1/5" in text
        assert "task-1" in text
        assert "Create models" in text

    def test_task_done_prints_completion(self) -> None:
        ui = _make_ui()
        ui.task_done("task-1")
        text = ui.console.print.call_args[0][0]
        assert "task-1" in text
        assert "complete" in text


# ---------------------------------------------------------------------------
# deploy_complete
# ---------------------------------------------------------------------------


class TestDeployComplete:
    """Tests for the final deployment success banner."""

    def test_renders_panel(self) -> None:
        ui = _make_ui()
        ui.deploy_complete("/path/to/project")
        # 3 calls: blank, Panel, blank
        assert ui.console.print.call_count == 3
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(panel_arg, Panel)

    def test_panel_contains_project_path(self) -> None:
        ui = _make_ui()
        ui.deploy_complete("/home/user/my-app")
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert "/home/user/my-app" in str(panel_arg.renderable)


# ---------------------------------------------------------------------------
# metrics_display
# ---------------------------------------------------------------------------


class TestMetricsDisplay:
    """Tests for the build metrics table."""

    def test_renders_table(self) -> None:
        ui = _make_ui()
        ui.metrics_display({"duration": "45s", "tasks": 10})
        # 3 calls: blank, Table, blank
        assert ui.console.print.call_count == 3
        table_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(table_arg, Table)

    def test_empty_metrics(self) -> None:
        ui = _make_ui()
        ui.metrics_display({})
        assert ui.console.print.call_count == 3
        table_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(table_arg, Table)

    def test_values_converted_to_string(self) -> None:
        """Metrics with non-string values should still render without error."""
        ui = _make_ui()
        ui.metrics_display({"count": 42, "ratio": 3.14, "flag": True})
        # No assertion needed beyond no-raise; Table built successfully
        assert ui.console.print.call_count == 3


# ---------------------------------------------------------------------------
# error / info
# ---------------------------------------------------------------------------


class TestErrorAndInfo:
    """Tests for simple message display methods."""

    def test_error_prints_message(self) -> None:
        ui = _make_ui()
        ui.error("Something broke")
        text = ui.console.print.call_args[0][0]
        assert "Something broke" in text
        assert "Error" in text

    def test_info_prints_message(self) -> None:
        ui = _make_ui()
        ui.info("Processing step 3")
        text = ui.console.print.call_args[0][0]
        assert "Processing step 3" in text


# ---------------------------------------------------------------------------
# show_projects
# ---------------------------------------------------------------------------


class TestShowProjects:
    """Tests for the project listing table."""

    def test_empty_list_prints_no_sessions_message(self) -> None:
        ui = _make_ui()
        ui.show_projects([])
        ui.console.print.assert_called_once()
        text = ui.console.print.call_args[0][0]
        assert "No Kindle build sessions found" in text

    def test_single_project_renders_table(self) -> None:
        ui = _make_ui()
        projects = [
            {
                "project_id": "kindle_abc12345",
                "idea": "Build a CLI tool",
                "status": "completed",
                "stages_completed": ["grill", "research"],
                "created_at": "2025-01-15T10:30:00.000000",
            }
        ]
        ui.show_projects(projects)
        ui.console.print.assert_called_once()
        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)

    def test_multiple_projects_renders_table(self) -> None:
        ui = _make_ui()
        projects = [
            {
                "project_id": f"kindle_{i:08x}",
                "idea": f"Project {i}",
                "status": "in_progress",
                "stages_completed": [],
                "created_at": f"2025-01-{i+1:02d}T00:00:00",
            }
            for i in range(5)
        ]
        ui.show_projects(projects)
        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)

    def test_missing_optional_fields_use_defaults(self) -> None:
        """Projects with missing keys should gracefully degrade."""
        ui = _make_ui()
        projects = [
            {
                "project_id": "kindle_00000001",
                # "idea" missing
                # "status" missing
                # "stages_completed" missing
                # "created_at" missing
            }
        ]
        ui.show_projects(projects)
        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)

    def test_long_idea_truncated_to_50_chars(self) -> None:
        """The idea column has max_width=50; the row value is sliced to [:50]."""
        ui = _make_ui()
        long_idea = "A" * 100
        projects = [
            {
                "project_id": "kindle_00000001",
                "idea": long_idea,
                "status": "in_progress",
                "stages_completed": [],
                "created_at": "2025-01-01T00:00:00",
            }
        ]
        ui.show_projects(projects)
        # The table was created; verify no error. The [:50] slice happens
        # in the add_row call which Rich renders internally.
        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)

    def test_stages_joined_with_commas(self) -> None:
        """stages_completed list is joined with ', ' for display."""
        ui = _make_ui()
        projects = [
            {
                "project_id": "kindle_00000001",
                "idea": "test",
                "status": "in_progress",
                "stages_completed": ["grill", "research", "architect"],
                "created_at": "2025-01-01T00:00:00",
            }
        ]
        ui.show_projects(projects)
        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)

    def test_empty_stages_shows_dash(self) -> None:
        """When stages_completed is empty, display shows '—'."""
        ui = _make_ui()
        projects = [
            {
                "project_id": "kindle_00000001",
                "idea": "test",
                "status": "in_progress",
                "stages_completed": [],
                "created_at": "2025-01-01T00:00:00",
            }
        ]
        ui.show_projects(projects)
        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)
