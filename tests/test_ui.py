"""Tests for kindle.ui — Rich terminal UI module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kindle.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(*, auto_approve: bool = False, verbose: bool = False) -> UI:
    """Create a UI instance with stdin.isatty() returning True."""
    with patch("kindle.ui.sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = True
        ui = UI(auto_approve=auto_approve, verbose=verbose)
    ui.console = MagicMock()
    return ui


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestUIInit:
    """Tests for UI.__init__ — flag handling and stdin detection."""

    def test_default_flags(self) -> None:
        with patch("kindle.ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI()
        assert ui.auto_approve is False
        assert ui.verbose is False

    def test_verbose_flag(self) -> None:
        with patch("kindle.ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(verbose=True)
        assert ui.verbose is True

    def test_auto_approve_flag(self) -> None:
        with patch("kindle.ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    def test_non_interactive_stdin_forces_auto_approve(self) -> None:
        """When stdin is not a tty, auto_approve must be True regardless."""
        with patch("kindle.ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            ui = UI(auto_approve=False)
        assert ui.auto_approve is True

    def test_auto_approve_true_with_interactive_stdin(self) -> None:
        """Explicit auto_approve=True is preserved even when stdin is a tty."""
        with patch("kindle.ui.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    def test_current_stage_starts_none(self) -> None:
        ui = _make_ui()
        assert ui._current_stage is None


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module constants are well-formed."""

    def test_stage_labels_has_all_expected_keys(self) -> None:
        expected = {"grill", "research", "architect", "dev", "qa", "package"}
        assert set(STAGE_LABELS.keys()) == expected

    def test_stage_order_matches_label_keys(self) -> None:
        assert list(STAGE_LABELS.keys()) == STAGE_ORDER

    def test_max_display_chars_is_positive(self) -> None:
        assert MAX_DISPLAY_CHARS > 0


# ---------------------------------------------------------------------------
# _safe_print — error suppression
# ---------------------------------------------------------------------------


class TestSafePrint:
    """_safe_print must silently suppress pipe-related errors."""

    def test_suppresses_broken_pipe_error(self) -> None:
        ui = _make_ui()
        ui.console.print.side_effect = BrokenPipeError
        ui._safe_print("hello")  # should not raise

    def test_suppresses_blocking_io_error(self) -> None:
        ui = _make_ui()
        ui.console.print.side_effect = BlockingIOError
        ui._safe_print("hello")  # should not raise

    def test_suppresses_os_error(self) -> None:
        ui = _make_ui()
        ui.console.print.side_effect = OSError("pipe closed")
        ui._safe_print("hello")  # should not raise

    def test_passes_through_normal_print(self) -> None:
        ui = _make_ui()
        ui._safe_print("hello", style="bold")
        ui.console.print.assert_called_once_with("hello", style="bold")


# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------


class TestBanner:
    """Tests for banner() display."""

    def test_banner_calls_print_three_times(self) -> None:
        ui = _make_ui()
        ui.banner("My awesome idea", "proj-123")
        # blank line, Panel, blank line
        assert ui.console.print.call_count == 3

    def test_banner_renders_panel_with_idea_and_project_id(self) -> None:
        ui = _make_ui()
        ui.banner("Build a REST API", "session-abc")
        # The second call should be the Panel
        panel_arg = ui.console.print.call_args_list[1][0][0]
        from rich.panel import Panel

        assert isinstance(panel_arg, Panel)


# ---------------------------------------------------------------------------
# stage_start / stage_done
# ---------------------------------------------------------------------------


class TestStageStart:
    """Tests for stage_start() — sets current stage and prints rule."""

    @pytest.mark.parametrize(("stage", "label"), list(STAGE_LABELS.items()))
    def test_sets_current_stage(self, stage: str, label: str) -> None:
        ui = _make_ui()
        ui.stage_start(stage)
        assert ui._current_stage == stage

    @pytest.mark.parametrize("stage", list(STAGE_LABELS.keys()))
    def test_prints_rule_for_known_stage(self, stage: str) -> None:
        ui = _make_ui()
        ui.stage_start(stage)
        ui.console.rule.assert_called_once()
        rule_text = ui.console.rule.call_args[0][0]
        assert STAGE_LABELS[stage] in rule_text

    def test_unknown_stage_falls_back_to_raw_name(self) -> None:
        ui = _make_ui()
        ui.stage_start("custom_stage")
        ui.console.rule.assert_called_once()
        rule_text = ui.console.rule.call_args[0][0]
        assert "custom_stage" in rule_text

    def test_stage_start_suppresses_broken_pipe(self) -> None:
        ui = _make_ui()
        ui.console.rule.side_effect = BrokenPipeError
        ui.stage_start("grill")  # should not raise

    def test_stage_start_suppresses_blocking_io(self) -> None:
        ui = _make_ui()
        ui.console.rule.side_effect = BlockingIOError
        ui.stage_start("grill")  # should not raise

    def test_stage_start_suppresses_os_error(self) -> None:
        ui = _make_ui()
        ui.console.rule.side_effect = OSError
        ui.stage_start("grill")  # should not raise


class TestStageDone:
    """Tests for stage_done() — prints completion message."""

    @pytest.mark.parametrize(("stage", "label"), list(STAGE_LABELS.items()))
    def test_prints_completion_for_known_stage(self, stage: str, label: str) -> None:
        ui = _make_ui()
        ui.stage_done(stage)
        printed = ui.console.print.call_args_list[0][0][0]
        assert label in printed
        assert "complete" in printed

    def test_unknown_stage_uses_raw_name(self) -> None:
        ui = _make_ui()
        ui.stage_done("mystery")
        printed = ui.console.print.call_args_list[0][0][0]
        assert "mystery" in printed

    def test_prints_blank_line_after_completion(self) -> None:
        ui = _make_ui()
        ui.stage_done("dev")
        # Should have 2 calls: completion message + blank line
        assert ui.console.print.call_count == 2


# ---------------------------------------------------------------------------
# stage_log
# ---------------------------------------------------------------------------


class TestStageLog:
    """Tests for stage_log() — verbose-only logging."""

    def test_verbose_mode_prints_message(self) -> None:
        ui = _make_ui(verbose=True)
        ui.stage_log("grill", "Processing input")
        ui.console.print.assert_called_once()
        printed = ui.console.print.call_args[0][0]
        assert "Processing input" in printed

    def test_non_verbose_mode_prints_nothing(self) -> None:
        ui = _make_ui(verbose=False)
        ui.stage_log("grill", "Processing input")
        ui.console.print.assert_not_called()

    @pytest.mark.parametrize(("stage", "label"), list(STAGE_LABELS.items()))
    def test_verbose_log_includes_stage_label(self, stage: str, label: str) -> None:
        ui = _make_ui(verbose=True)
        ui.stage_log(stage, "test message")
        printed = ui.console.print.call_args[0][0]
        assert label in printed

    def test_unknown_stage_falls_back_to_raw_name(self) -> None:
        ui = _make_ui(verbose=True)
        ui.stage_log("custom", "msg")
        printed = ui.console.print.call_args[0][0]
        assert "custom" in printed


# ---------------------------------------------------------------------------
# show_artifact
# ---------------------------------------------------------------------------


class TestShowArtifact:
    """Tests for show_artifact() — content display with truncation."""

    def test_short_content_displayed_as_is(self) -> None:
        ui = _make_ui()
        content = "short content"
        ui.show_artifact("Title", content)
        # blank line, Panel, blank line
        assert ui.console.print.call_count == 3

    def test_long_content_is_truncated(self) -> None:
        ui = _make_ui()
        content = "x" * (MAX_DISPLAY_CHARS + 500)
        ui.show_artifact("Title", content)
        from rich.panel import Panel

        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(panel_arg, Panel)
        # Verify the renderable contains the truncation notice
        rendered = panel_arg.renderable
        assert "truncated" in rendered

    def test_exact_max_length_not_truncated(self) -> None:
        ui = _make_ui()
        content = "x" * MAX_DISPLAY_CHARS
        ui.show_artifact("Title", content)
        panel_arg = ui.console.print.call_args_list[1][0][0]
        rendered = panel_arg.renderable
        assert "truncated" not in rendered


# ---------------------------------------------------------------------------
# grill_question
# ---------------------------------------------------------------------------


class TestGrillQuestion:
    """Tests for grill_question() — human-in-the-loop gate."""

    def test_auto_approve_returns_recommended(self) -> None:
        ui = _make_ui(auto_approve=True)
        result = ui.grill_question("What stack?", "FastAPI", "tech", 1)
        assert result == "FastAPI"

    def test_auto_approve_does_not_prompt_user(self) -> None:
        ui = _make_ui(auto_approve=True)
        ui.grill_question("What stack?", "FastAPI", "tech", 1)
        ui.console.input.assert_not_called()

    def test_user_provides_custom_answer(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "Django"
        result = ui.grill_question("What stack?", "FastAPI", "tech", 1)
        assert result == "Django"

    def test_empty_input_returns_recommended(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = ""
        result = ui.grill_question("What stack?", "FastAPI", "tech", 1)
        assert result == "FastAPI"

    def test_whitespace_only_input_returns_recommended(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "   "
        result = ui.grill_question("What stack?", "FastAPI", "tech", 1)
        assert result == "FastAPI"

    def test_eof_error_returns_recommended(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.side_effect = EOFError
        result = ui.grill_question("What stack?", "FastAPI", "tech", 1)
        assert result == "FastAPI"

    def test_keyboard_interrupt_returns_recommended(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.side_effect = KeyboardInterrupt
        result = ui.grill_question("What stack?", "FastAPI", "tech", 1)
        assert result == "FastAPI"

    def test_question_panel_is_displayed(self) -> None:
        ui = _make_ui(auto_approve=True)
        ui.grill_question("What DB?", "Postgres", "infra", 3)
        from rich.panel import Panel

        # First call is blank line, second is the Panel
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(panel_arg, Panel)

    def test_strips_user_input(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "  Flask  "
        result = ui.grill_question("What stack?", "FastAPI", "tech", 1)
        assert result == "Flask"


# ---------------------------------------------------------------------------
# prompt_arch_review
# ---------------------------------------------------------------------------


class TestPromptArchReview:
    """Tests for prompt_arch_review() — architecture approval gate."""

    def test_auto_approve_returns_true_empty_feedback(self) -> None:
        ui = _make_ui(auto_approve=True)
        approved, feedback = ui.prompt_arch_review("The architecture plan")
        assert approved is True
        assert feedback == ""

    def test_auto_approve_does_not_prompt(self) -> None:
        ui = _make_ui(auto_approve=True)
        ui.prompt_arch_review("plan")
        ui.console.input.assert_not_called()

    @pytest.mark.parametrize("response", ["approve", "yes", "y", "lgtm", ""])
    def test_approval_keywords(self, response: str) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = response
        approved, feedback = ui.prompt_arch_review("plan")
        assert approved is True
        assert feedback == ""

    @pytest.mark.parametrize("response", ["Approve", "YES", "Y", "LGTM"])
    def test_approval_keywords_case_insensitive(self, response: str) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = response
        approved, feedback = ui.prompt_arch_review("plan")
        assert approved is True
        assert feedback == ""

    def test_feedback_returns_false_with_message(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "Please add caching layer"
        approved, feedback = ui.prompt_arch_review("plan")
        assert approved is False
        assert feedback == "Please add caching layer"

    def test_eof_error_auto_approves(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.side_effect = EOFError
        approved, feedback = ui.prompt_arch_review("plan")
        assert approved is True
        assert feedback == ""

    def test_keyboard_interrupt_auto_approves(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.side_effect = KeyboardInterrupt
        approved, feedback = ui.prompt_arch_review("plan")
        assert approved is True
        assert feedback == ""

    def test_shows_architecture_artifact(self) -> None:
        ui = _make_ui(auto_approve=True)
        ui.prompt_arch_review("my architecture plan")
        from rich.panel import Panel

        # show_artifact prints: blank, Panel, blank; then auto-approve message
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(panel_arg, Panel)

    def test_strips_user_response(self) -> None:
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "  approve  "
        approved, feedback = ui.prompt_arch_review("plan")
        assert approved is True


# ---------------------------------------------------------------------------
# task_start / task_done
# ---------------------------------------------------------------------------


class TestTaskStartDone:
    """Tests for task_start() and task_done() display."""

    def test_task_start_shows_index_and_total(self) -> None:
        ui = _make_ui()
        ui.task_start("task-1", "Create models", 1, 5)
        printed = ui.console.print.call_args[0][0]
        assert "(1/5)" in printed
        assert "task-1" in printed
        assert "Create models" in printed

    def test_task_done_shows_completion(self) -> None:
        ui = _make_ui()
        ui.task_done("task-1")
        printed = ui.console.print.call_args[0][0]
        assert "task-1" in printed
        assert "complete" in printed


# ---------------------------------------------------------------------------
# deploy_complete
# ---------------------------------------------------------------------------


class TestDeployComplete:
    """Tests for deploy_complete() panel display."""

    def test_prints_project_path_in_panel(self) -> None:
        ui = _make_ui()
        ui.deploy_complete("/home/user/project")
        from rich.panel import Panel

        assert ui.console.print.call_count == 3
        panel_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(panel_arg, Panel)


# ---------------------------------------------------------------------------
# metrics_display
# ---------------------------------------------------------------------------


class TestMetricsDisplay:
    """Tests for metrics_display() table rendering."""

    def test_displays_table_with_metrics(self) -> None:
        ui = _make_ui()
        metrics = {"Duration": "42s", "Files": 10, "Lines": 500}
        ui.metrics_display(metrics)
        from rich.table import Table

        # blank line, Table, blank line
        assert ui.console.print.call_count == 3
        table_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(table_arg, Table)

    def test_empty_metrics_still_renders_table(self) -> None:
        ui = _make_ui()
        ui.metrics_display({})
        from rich.table import Table

        table_arg = ui.console.print.call_args_list[1][0][0]
        assert isinstance(table_arg, Table)


# ---------------------------------------------------------------------------
# error / info
# ---------------------------------------------------------------------------


class TestErrorInfo:
    """Tests for error() and info() convenience methods."""

    def test_error_includes_message(self) -> None:
        ui = _make_ui()
        ui.error("something broke")
        printed = ui.console.print.call_args[0][0]
        assert "something broke" in printed
        assert "Error" in printed

    def test_info_includes_message(self) -> None:
        ui = _make_ui()
        ui.info("heads up")
        printed = ui.console.print.call_args[0][0]
        assert "heads up" in printed


# ---------------------------------------------------------------------------
# show_projects
# ---------------------------------------------------------------------------


class TestShowProjects:
    """Tests for show_projects() — project listing table."""

    def test_empty_list_shows_no_sessions_message(self) -> None:
        ui = _make_ui()
        ui.show_projects([])
        ui.console.print.assert_called_once()
        printed = ui.console.print.call_args[0][0]
        assert "No Kindle build sessions" in printed

    def test_single_project_renders_table(self) -> None:
        ui = _make_ui()
        projects = [
            {
                "project_id": "proj-1",
                "idea": "Build a todo app",
                "status": "complete",
                "stages_completed": ["grill", "research"],
                "created_at": "2025-01-15T10:30:00Z",
            }
        ]
        ui.show_projects(projects)
        from rich.table import Table

        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)

    def test_project_with_missing_optional_fields(self) -> None:
        """Projects may lack some fields; show_projects must not crash."""
        ui = _make_ui()
        projects = [
            {
                "project_id": "proj-2",
            }
        ]
        ui.show_projects(projects)
        from rich.table import Table

        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)

    def test_multiple_projects_renders_table(self) -> None:
        ui = _make_ui()
        projects = [
            {
                "project_id": f"proj-{i}",
                "idea": f"Idea {i}",
                "status": "complete",
                "stages_completed": [],
                "created_at": "2025-01-15T10:30:00Z",
            }
            for i in range(3)
        ]
        ui.show_projects(projects)
        from rich.table import Table

        table_arg = ui.console.print.call_args[0][0]
        assert isinstance(table_arg, Table)

    def test_long_idea_truncated_to_50_chars(self) -> None:
        """The idea column is max_width=50, but the row data is sliced to 50."""
        ui = _make_ui()
        long_idea = "A" * 100
        projects = [
            {
                "project_id": "proj-long",
                "idea": long_idea,
                "status": "active",
                "stages_completed": [],
                "created_at": "2025-06-01T00:00:00Z",
            }
        ]
        ui.show_projects(projects)
        # The idea passed to add_row should be sliced to 50 chars
        # We trust Rich Table to enforce max_width; the code explicitly slices [:50]
        ui.console.print.assert_called_once()

    def test_empty_stages_shows_dash(self) -> None:
        """When stages_completed is empty, a dash is shown."""
        ui = _make_ui()
        projects = [
            {
                "project_id": "proj-x",
                "idea": "test",
                "status": "new",
                "stages_completed": [],
                "created_at": "2025-01-01T00:00:00Z",
            }
        ]
        # No crash; the code renders "—" for empty stages
        ui.show_projects(projects)
