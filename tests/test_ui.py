"""Tests for kindle.ui — Rich terminal UI class."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from kindle.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ---------------------------------------------------------------------------
# Constructor defaults and TTY detection
# ---------------------------------------------------------------------------


class TestUIConstructor:
    """Tests for UI.__init__ defaults and TTY-based auto_approve logic."""

    def test_default_auto_approve_is_false_when_tty(self) -> None:
        with patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            ui = UI()
        assert ui.auto_approve is False

    def test_default_verbose_is_false(self) -> None:
        with patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            ui = UI()
        assert ui.verbose is False

    def test_auto_approve_true_when_explicitly_set(self) -> None:
        with patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    def test_auto_approve_forced_true_when_not_tty(self) -> None:
        """When stdin is not a TTY (piped/CI), auto_approve must be True."""
        with patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            ui = UI(auto_approve=False)
        assert ui.auto_approve is True

    def test_auto_approve_stays_true_when_set_and_not_tty(self) -> None:
        with patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    def test_verbose_true_when_set(self) -> None:
        with patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            ui = UI(verbose=True)
        assert ui.verbose is True

    def test_current_stage_initially_none(self) -> None:
        with patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            ui = UI()
        assert ui._current_stage is None

    def test_console_is_created(self) -> None:
        with patch("kindle.ui.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            ui = UI()
        assert ui.console is not None


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are well-formed."""

    def test_stage_labels_keys_match_order(self) -> None:
        assert list(STAGE_LABELS.keys()) == STAGE_ORDER

    def test_stage_order_has_expected_stages(self) -> None:
        expected = ["grill", "research", "architect", "dev", "qa", "package"]
        assert STAGE_ORDER == expected

    def test_max_display_chars_is_3000(self) -> None:
        assert MAX_DISPLAY_CHARS == 3000


# ---------------------------------------------------------------------------
# Helper to build a UI with mocked console
# ---------------------------------------------------------------------------


@pytest.fixture
def ui() -> UI:
    """Return a UI instance with a mocked Console and TTY stdin."""
    with patch("kindle.ui.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = True
        instance = UI()
    instance.console = MagicMock()
    return instance


@pytest.fixture
def ui_auto() -> UI:
    """Return a UI instance with auto_approve=True and mocked Console."""
    with patch("kindle.ui.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = True
        instance = UI(auto_approve=True)
    instance.console = MagicMock()
    return instance


@pytest.fixture
def ui_verbose() -> UI:
    """Return a UI instance with verbose=True and mocked Console."""
    with patch("kindle.ui.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = True
        instance = UI(verbose=True)
    instance.console = MagicMock()
    return instance


# ---------------------------------------------------------------------------
# _safe_print — error suppression
# ---------------------------------------------------------------------------


class TestSafePrint:
    """_safe_print should suppress IO-related exceptions."""

    def test_safe_print_calls_console_print(self, ui: UI) -> None:
        ui._safe_print("hello")
        ui.console.print.assert_called_once_with("hello")

    def test_safe_print_suppresses_broken_pipe(self, ui: UI) -> None:
        ui.console.print.side_effect = BrokenPipeError
        ui._safe_print("hello")  # should not raise

    def test_safe_print_suppresses_blocking_io(self, ui: UI) -> None:
        ui.console.print.side_effect = BlockingIOError
        ui._safe_print("hello")  # should not raise

    def test_safe_print_suppresses_os_error(self, ui: UI) -> None:
        ui.console.print.side_effect = OSError
        ui._safe_print("hello")  # should not raise

    def test_safe_print_passes_kwargs(self, ui: UI) -> None:
        ui._safe_print("hello", style="bold")
        ui.console.print.assert_called_once_with("hello", style="bold")


# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------


class TestBanner:
    """Tests for banner() — displays the project header panel."""

    def test_banner_calls_console_print(self, ui: UI) -> None:
        ui.banner("Build a chat app", "proj-123")
        assert ui.console.print.call_count >= 1

    def test_banner_does_not_raise(self, ui: UI) -> None:
        ui.banner("My Idea", "session-abc")

    def test_banner_with_empty_strings(self, ui: UI) -> None:
        ui.banner("", "")
        assert ui.console.print.called

    def test_banner_suppresses_io_errors(self, ui: UI) -> None:
        ui.console.print.side_effect = BrokenPipeError
        ui.banner("idea", "id")  # should not raise


# ---------------------------------------------------------------------------
# stage_start / stage_done
# ---------------------------------------------------------------------------


class TestStageStart:
    """Tests for stage_start() — sets current stage and prints a rule."""

    def test_stage_start_sets_current_stage(self, ui: UI) -> None:
        ui.stage_start("grill")
        assert ui._current_stage == "grill"

    def test_stage_start_uses_label(self, ui: UI) -> None:
        ui.stage_start("grill")
        ui.console.rule.assert_called_once()
        rule_arg = ui.console.rule.call_args[0][0]
        assert "Grill" in rule_arg

    def test_stage_start_unknown_stage_uses_raw_name(self, ui: UI) -> None:
        ui.stage_start("custom_stage")
        assert ui._current_stage == "custom_stage"
        ui.console.rule.assert_called_once()
        rule_arg = ui.console.rule.call_args[0][0]
        assert "custom_stage" in rule_arg

    def test_stage_start_suppresses_broken_pipe(self, ui: UI) -> None:
        ui.console.rule.side_effect = BrokenPipeError
        ui.stage_start("dev")  # should not raise

    def test_stage_start_suppresses_os_error(self, ui: UI) -> None:
        ui.console.rule.side_effect = OSError
        ui.stage_start("dev")  # should not raise


class TestStageDone:
    """Tests for stage_done() — prints stage completion message."""

    def test_stage_done_prints_completion(self, ui: UI) -> None:
        ui.stage_done("grill")
        assert ui.console.print.called
        printed = ui.console.print.call_args_list[0][0][0]
        assert "Grill" in printed
        assert "complete" in printed

    def test_stage_done_unknown_stage_uses_raw_name(self, ui: UI) -> None:
        ui.stage_done("unknown")
        printed = ui.console.print.call_args_list[0][0][0]
        assert "unknown" in printed

    def test_stage_done_calls_print_twice(self, ui: UI) -> None:
        """stage_done prints the message then an empty line."""
        ui.stage_done("qa")
        assert ui.console.print.call_count == 2


# ---------------------------------------------------------------------------
# stage_log — verbose-only output
# ---------------------------------------------------------------------------


class TestStageLog:
    """stage_log should only print when verbose=True."""

    def test_stage_log_silent_when_not_verbose(self, ui: UI) -> None:
        assert ui.verbose is False
        ui.stage_log("grill", "Processing question 1")
        ui.console.print.assert_not_called()

    def test_stage_log_prints_when_verbose(self, ui_verbose: UI) -> None:
        ui_verbose.stage_log("grill", "Processing question 1")
        ui_verbose.console.print.assert_called_once()
        printed = ui_verbose.console.print.call_args[0][0]
        assert "Processing question 1" in printed

    def test_stage_log_uses_label_when_verbose(self, ui_verbose: UI) -> None:
        ui_verbose.stage_log("research", "Searching web")
        printed = ui_verbose.console.print.call_args[0][0]
        assert "Research" in printed

    def test_stage_log_unknown_stage_when_verbose(self, ui_verbose: UI) -> None:
        ui_verbose.stage_log("custom", "msg")
        printed = ui_verbose.console.print.call_args[0][0]
        assert "custom" in printed


# ---------------------------------------------------------------------------
# show_artifact — content display with truncation
# ---------------------------------------------------------------------------


class TestShowArtifact:
    """Tests for show_artifact() — content panel with truncation at MAX_DISPLAY_CHARS."""

    def test_short_content_displayed_in_full(self, ui: UI) -> None:
        ui.show_artifact("Title", "short content")
        assert ui.console.print.called
        # Find the Panel call (not the empty-line calls)
        panel_calls = [
            c for c in ui.console.print.call_args_list if c[0] and hasattr(c[0][0], "renderable")
        ]
        assert len(panel_calls) >= 1

    def test_content_at_exactly_max_chars_not_truncated(self, ui: UI) -> None:
        content = "x" * MAX_DISPLAY_CHARS
        ui.show_artifact("Title", content)
        # Verify no truncation marker was added
        for c in ui.console.print.call_args_list:
            if c[0]:
                arg = c[0][0]
                if hasattr(arg, "renderable"):
                    assert "truncated" not in str(arg.renderable)

    def test_content_over_max_chars_is_truncated(self, ui: UI) -> None:
        content = "x" * (MAX_DISPLAY_CHARS + 500)
        ui.show_artifact("Title", content)
        # Find the Panel and check its renderable contains truncation notice
        for c in ui.console.print.call_args_list:
            if c[0]:
                arg = c[0][0]
                if hasattr(arg, "renderable"):
                    assert "truncated" in str(arg.renderable)
                    # Verify we don't include the full content
                    assert len(str(arg.renderable)) < len(content)

    def test_truncation_preserves_first_max_chars(self, ui: UI) -> None:
        prefix = "A" * MAX_DISPLAY_CHARS
        content = prefix + "B" * 1000
        ui.show_artifact("Title", content)
        for c in ui.console.print.call_args_list:
            if c[0]:
                arg = c[0][0]
                if hasattr(arg, "renderable"):
                    renderable_str = str(arg.renderable)
                    assert renderable_str.startswith(prefix)
                    assert "B" not in renderable_str.replace("truncated", "").replace("[dim]", "").replace("[/dim]", "")

    def test_show_artifact_does_not_raise(self, ui: UI) -> None:
        ui.show_artifact("Title", "content")

    def test_show_artifact_prints_three_times(self, ui: UI) -> None:
        """Empty line, Panel, empty line."""
        ui.show_artifact("Title", "content")
        assert ui.console.print.call_count == 3


# ---------------------------------------------------------------------------
# grill_question
# ---------------------------------------------------------------------------


class TestGrillQuestion:
    """Tests for grill_question() — interactive Q&A with auto-approve support."""

    def test_auto_approve_returns_recommended(self, ui_auto: UI) -> None:
        result = ui_auto.grill_question(
            question="What language?",
            recommended="Python",
            category="tech",
            number=1,
        )
        assert result == "Python"

    def test_auto_approve_does_not_call_console_input(self, ui_auto: UI) -> None:
        ui_auto.grill_question("Q?", "A", "cat", 1)
        ui_auto.console.input.assert_not_called()

    def test_manual_input_custom_answer(self, ui: UI) -> None:
        ui.console.input.return_value = "TypeScript"
        result = ui.grill_question("What language?", "Python", "tech", 1)
        assert result == "TypeScript"

    def test_manual_input_empty_returns_recommended(self, ui: UI) -> None:
        ui.console.input.return_value = ""
        result = ui.grill_question("What language?", "Python", "tech", 1)
        assert result == "Python"

    def test_manual_input_whitespace_only_returns_recommended(self, ui: UI) -> None:
        ui.console.input.return_value = "   "
        result = ui.grill_question("What language?", "Python", "tech", 1)
        assert result == "Python"

    def test_eof_error_returns_recommended(self, ui: UI) -> None:
        ui.console.input.side_effect = EOFError
        result = ui.grill_question("Q?", "default", "cat", 1)
        assert result == "default"

    def test_keyboard_interrupt_returns_recommended(self, ui: UI) -> None:
        ui.console.input.side_effect = KeyboardInterrupt
        result = ui.grill_question("Q?", "default", "cat", 1)
        assert result == "default"

    def test_input_is_stripped(self, ui: UI) -> None:
        ui.console.input.return_value = "  Go  "
        result = ui.grill_question("Q?", "Python", "tech", 1)
        assert result == "Go"

    def test_question_number_displayed(self, ui: UI) -> None:
        ui.console.input.return_value = ""
        ui.grill_question("Q?", "A", "cat", 42)
        # Verify the Panel was printed with the question number
        panel_printed = False
        for c in ui.console.print.call_args_list:
            if c[0]:
                arg_str = str(c[0][0])
                if "42" in arg_str:
                    panel_printed = True
        assert panel_printed


# ---------------------------------------------------------------------------
# prompt_arch_review
# ---------------------------------------------------------------------------


class TestPromptArchReview:
    """Tests for prompt_arch_review() — architecture approval gate."""

    def test_auto_approve_returns_true_empty_feedback(self, ui_auto: UI) -> None:
        result = ui_auto.prompt_arch_review("Architecture summary")
        assert result == (True, "")

    def test_auto_approve_does_not_call_input(self, ui_auto: UI) -> None:
        ui_auto.prompt_arch_review("Summary")
        ui_auto.console.input.assert_not_called()

    def test_manual_approve_with_approve_keyword(self, ui: UI) -> None:
        ui.console.input.return_value = "approve"
        result = ui.prompt_arch_review("Summary")
        assert result == (True, "")

    def test_manual_approve_with_yes(self, ui: UI) -> None:
        ui.console.input.return_value = "yes"
        result = ui.prompt_arch_review("Summary")
        assert result == (True, "")

    def test_manual_approve_with_y(self, ui: UI) -> None:
        ui.console.input.return_value = "y"
        result = ui.prompt_arch_review("Summary")
        assert result == (True, "")

    def test_manual_approve_with_lgtm(self, ui: UI) -> None:
        ui.console.input.return_value = "lgtm"
        result = ui.prompt_arch_review("Summary")
        assert result == (True, "")

    def test_manual_approve_with_empty_input(self, ui: UI) -> None:
        ui.console.input.return_value = ""
        result = ui.prompt_arch_review("Summary")
        assert result == (True, "")

    def test_manual_approve_case_insensitive(self, ui: UI) -> None:
        ui.console.input.return_value = "APPROVE"
        result = ui.prompt_arch_review("Summary")
        assert result == (True, "")

    def test_manual_reject_with_feedback(self, ui: UI) -> None:
        ui.console.input.return_value = "Needs more error handling"
        result = ui.prompt_arch_review("Summary")
        assert result == (False, "Needs more error handling")

    def test_eof_error_auto_approves(self, ui: UI) -> None:
        ui.console.input.side_effect = EOFError
        result = ui.prompt_arch_review("Summary")
        assert result == (True, "")

    def test_keyboard_interrupt_auto_approves(self, ui: UI) -> None:
        ui.console.input.side_effect = KeyboardInterrupt
        result = ui.prompt_arch_review("Summary")
        assert result == (True, "")

    def test_input_is_stripped(self, ui: UI) -> None:
        ui.console.input.return_value = "  fix the DB layer  "
        approved, feedback = ui.prompt_arch_review("Summary")
        assert approved is False
        assert feedback == "fix the DB layer"

    def test_shows_architecture_artifact(self, ui: UI) -> None:
        ui.console.input.return_value = "approve"
        ui.prompt_arch_review("My architecture plan")
        # show_artifact calls _safe_print which calls console.print
        # Verify at least one call was made (the panel)
        assert ui.console.print.called


# ---------------------------------------------------------------------------
# task_start / task_done
# ---------------------------------------------------------------------------


class TestTaskStartDone:
    """Tests for task_start() and task_done()."""

    def test_task_start_prints_id_and_title(self, ui: UI) -> None:
        ui.task_start("TASK-01", "Create user model", 1, 5)
        ui.console.print.assert_called_once()
        printed = ui.console.print.call_args[0][0]
        assert "TASK-01" in printed
        assert "Create user model" in printed
        assert "1" in printed
        assert "5" in printed

    def test_task_done_prints_completion(self, ui: UI) -> None:
        ui.task_done("TASK-01")
        ui.console.print.assert_called_once()
        printed = ui.console.print.call_args[0][0]
        assert "TASK-01" in printed
        assert "complete" in printed


# ---------------------------------------------------------------------------
# deploy_complete
# ---------------------------------------------------------------------------


class TestDeployComplete:
    """Tests for deploy_complete() — final success panel."""

    def test_deploy_complete_prints_path(self, ui: UI) -> None:
        ui.deploy_complete("/home/user/project")
        assert ui.console.print.called

    def test_deploy_complete_does_not_raise(self, ui: UI) -> None:
        ui.deploy_complete("/path/to/project")


# ---------------------------------------------------------------------------
# metrics_display
# ---------------------------------------------------------------------------


class TestMetricsDisplay:
    """Tests for metrics_display() — build metrics table."""

    def test_metrics_display_prints_table(self, ui: UI) -> None:
        metrics = {"total_time": "120s", "tasks": 5}
        ui.metrics_display(metrics)
        assert ui.console.print.called

    def test_metrics_display_empty_dict(self, ui: UI) -> None:
        ui.metrics_display({})
        assert ui.console.print.called

    def test_metrics_display_does_not_raise(self, ui: UI) -> None:
        ui.metrics_display({"key": "value"})


# ---------------------------------------------------------------------------
# error / info
# ---------------------------------------------------------------------------


class TestErrorInfo:
    """Tests for error() and info() output formatting."""

    def test_error_includes_message(self, ui: UI) -> None:
        ui.error("something went wrong")
        ui.console.print.assert_called_once()
        printed = ui.console.print.call_args[0][0]
        assert "something went wrong" in printed
        assert "Error" in printed

    def test_error_contains_red_markup(self, ui: UI) -> None:
        ui.error("fail")
        printed = ui.console.print.call_args[0][0]
        assert "red" in printed

    def test_info_includes_message(self, ui: UI) -> None:
        ui.info("all good")
        ui.console.print.assert_called_once()
        printed = ui.console.print.call_args[0][0]
        assert "all good" in printed

    def test_info_contains_cyan_markup(self, ui: UI) -> None:
        ui.info("status")
        printed = ui.console.print.call_args[0][0]
        assert "cyan" in printed

    def test_error_with_special_characters(self, ui: UI) -> None:
        ui.error("File not found: /path/to/[file]")
        printed = ui.console.print.call_args[0][0]
        assert "File not found" in printed

    def test_info_with_empty_message(self, ui: UI) -> None:
        ui.info("")
        ui.console.print.assert_called_once()


# ---------------------------------------------------------------------------
# show_projects
# ---------------------------------------------------------------------------


class TestShowProjects:
    """Tests for show_projects() — project listing table."""

    def test_empty_projects_prints_no_sessions_message(self, ui: UI) -> None:
        ui.show_projects([])
        ui.console.print.assert_called_once()
        printed = ui.console.print.call_args[0][0]
        assert "No Kindle build sessions found" in printed

    def test_projects_displays_table(self, ui: UI) -> None:
        projects = [
            {
                "project_id": "proj-001",
                "idea": "Build a todo app",
                "status": "complete",
                "stages_completed": ["grill", "research"],
                "created_at": "2025-01-15T10:30:00Z",
            }
        ]
        ui.show_projects(projects)
        ui.console.print.assert_called_once()

    def test_projects_with_missing_optional_fields(self, ui: UI) -> None:
        projects = [{"project_id": "proj-002"}]
        ui.show_projects(projects)
        ui.console.print.assert_called_once()

    def test_projects_truncates_long_idea(self, ui: UI) -> None:
        """Ideas longer than 50 chars should be truncated in the table."""
        projects = [
            {
                "project_id": "proj-003",
                "idea": "A" * 100,
                "status": "running",
                "stages_completed": [],
                "created_at": "2025-01-15T10:30:00Z",
            }
        ]
        ui.show_projects(projects)
        ui.console.print.assert_called_once()

    def test_multiple_projects(self, ui: UI) -> None:
        projects = [
            {
                "project_id": f"proj-{i:03d}",
                "idea": f"Idea {i}",
                "status": "complete",
                "stages_completed": [],
                "created_at": "2025-01-15T10:30:00Z",
            }
            for i in range(5)
        ]
        ui.show_projects(projects)
        ui.console.print.assert_called_once()
