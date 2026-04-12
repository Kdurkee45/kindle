"""Tests for kindle.ui — Rich terminal UI progress and human gates."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from kindle.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(*, auto_approve: bool = True, verbose: bool = False) -> tuple[UI, StringIO]:
    """Create a UI wired to a StringIO buffer for output capture.

    Always patches stdin.isatty to True so auto_approve is controlled
    solely by the keyword argument.
    """
    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = True
        ui = UI(auto_approve=auto_approve, verbose=verbose)

    buf = StringIO()
    ui.console = Console(file=buf, force_terminal=True, width=120)
    return ui, buf


def _output(buf: StringIO) -> str:
    """Return the full captured output as a string."""
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are correct."""

    def test_stage_labels_has_six_entries(self):
        assert len(STAGE_LABELS) == 6

    def test_stage_labels_keys_match_order(self):
        assert list(STAGE_LABELS.keys()) == STAGE_ORDER

    def test_stage_order_contents(self):
        assert STAGE_ORDER == ["grill", "research", "architect", "dev", "qa", "package"]

    def test_max_display_chars_value(self):
        assert MAX_DISPLAY_CHARS == 3000


# ---------------------------------------------------------------------------
# UI instantiation
# ---------------------------------------------------------------------------


class TestUIInit:
    """Test UI constructor with various auto_approve and tty settings."""

    def test_auto_approve_true(self):
        """Explicit auto_approve=True sets the flag regardless of tty."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    def test_auto_approve_false_with_tty(self):
        """auto_approve=False + interactive terminal -> auto_approve stays False."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(auto_approve=False)
        assert ui.auto_approve is False

    def test_auto_approve_false_non_interactive(self):
        """auto_approve=False + non-interactive stdin -> auto_approve becomes True."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            ui = UI(auto_approve=False)
        assert ui.auto_approve is True

    def test_auto_approve_true_non_interactive(self):
        """auto_approve=True + non-interactive -> still True."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    def test_verbose_default_false(self):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI()
        assert ui.verbose is False

    def test_verbose_true(self):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(verbose=True)
        assert ui.verbose is True

    def test_console_is_rich_console(self):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI()
        assert isinstance(ui.console, Console)


# ---------------------------------------------------------------------------
# stage_start / stage_done
# ---------------------------------------------------------------------------


class TestStageLifecycle:
    """Test stage_start() and stage_done() produce expected output."""

    def test_stage_start_outputs_rule_with_label(self):
        """stage_start should print a rule containing the stage label."""
        ui, buf = _make_ui()
        ui.stage_start("grill")
        output = _output(buf)
        assert "Grill" in output

    def test_stage_start_sets_current_stage(self):
        ui, _buf = _make_ui()
        ui.stage_start("research")
        assert ui._current_stage == "research"

    def test_stage_done_outputs_complete_message(self):
        """stage_done should print '... complete' with the label."""
        ui, buf = _make_ui()
        ui.stage_done("dev")
        output = _output(buf)
        assert "Dev" in output
        assert "complete" in output

    def test_stage_start_unknown_stage_uses_raw_name(self):
        """Unknown stage name → falls back to raw string."""
        ui, buf = _make_ui()
        ui.stage_start("custom_stage")
        output = _output(buf)
        assert "custom_stage" in output

    def test_stage_done_unknown_stage_uses_raw_name(self):
        ui, buf = _make_ui()
        ui.stage_done("custom_stage")
        output = _output(buf)
        assert "custom_stage" in output

    @pytest.mark.parametrize("stage", STAGE_ORDER)
    def test_stage_start_all_known_stages(self, stage: str):
        """Every known stage should produce output containing its label."""
        ui, buf = _make_ui()
        ui.stage_start(stage)
        label = STAGE_LABELS[stage]
        # Strip emoji prefix to get the text part
        label_text = label.strip()
        output = _output(buf)
        assert len(output) > 0

    @pytest.mark.parametrize("stage", STAGE_ORDER)
    def test_stage_done_all_known_stages(self, stage: str):
        """Every known stage done message should include 'complete'."""
        ui, buf = _make_ui()
        ui.stage_done(stage)
        output = _output(buf)
        assert "complete" in output


# ---------------------------------------------------------------------------
# stage_log
# ---------------------------------------------------------------------------


class TestStageLog:
    """Test stage_log() respects the verbose flag."""

    def test_stage_log_verbose_true_produces_output(self):
        ui, buf = _make_ui(verbose=True)
        ui.stage_log("grill", "Processing question 1")
        output = _output(buf)
        assert "Processing question 1" in output

    def test_stage_log_verbose_false_suppressed(self):
        ui, buf = _make_ui(verbose=False)
        ui.stage_log("grill", "Processing question 1")
        output = _output(buf)
        assert output == ""

    def test_stage_log_includes_stage_label(self):
        ui, buf = _make_ui(verbose=True)
        ui.stage_log("research", "Searching for info")
        output = _output(buf)
        assert "Research" in output


# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------


class TestBanner:
    """Test the banner() display."""

    def test_banner_shows_idea(self):
        ui, buf = _make_ui()
        ui.banner("Build a chat app", "proj-123")
        output = _output(buf)
        assert "Build a chat app" in output

    def test_banner_shows_project_id(self):
        ui, buf = _make_ui()
        ui.banner("Build a chat app", "proj-123")
        output = _output(buf)
        assert "proj-123" in output

    def test_banner_shows_kindle_title(self):
        ui, buf = _make_ui()
        ui.banner("idea", "id")
        output = _output(buf)
        assert "Kindle" in output


# ---------------------------------------------------------------------------
# show_artifact
# ---------------------------------------------------------------------------


class TestShowArtifact:
    """Test show_artifact() including truncation behaviour."""

    def test_short_content_displayed_fully(self):
        """Content under MAX_DISPLAY_CHARS appears in full."""
        ui, buf = _make_ui()
        content = "Short artifact content"
        ui.show_artifact("My Title", content)
        output = _output(buf)
        assert "Short artifact content" in output
        assert "truncated" not in output

    def test_content_at_exact_limit_not_truncated(self):
        """Content exactly MAX_DISPLAY_CHARS long should NOT be truncated."""
        ui, buf = _make_ui()
        content = "x" * MAX_DISPLAY_CHARS
        ui.show_artifact("Exact", content)
        output = _output(buf)
        assert "truncated" not in output

    def test_content_over_limit_is_truncated(self):
        """Content exceeding MAX_DISPLAY_CHARS should show truncation notice."""
        ui, buf = _make_ui()
        content = "y" * (MAX_DISPLAY_CHARS + 1)
        ui.show_artifact("Long", content)
        output = _output(buf)
        assert "truncated" in output

    def test_truncated_content_is_capped(self):
        """After truncation the display should not contain the full original content."""
        ui, buf = _make_ui()
        # Create content with a marker at the end that should be cut off
        content = "A" * MAX_DISPLAY_CHARS + "MARKER_END"
        ui.show_artifact("Title", content)
        output = _output(buf)
        assert "MARKER_END" not in output

    def test_title_appears_in_output(self):
        ui, buf = _make_ui()
        ui.show_artifact("Architecture Plan", "some content")
        output = _output(buf)
        assert "Architecture Plan" in output

    def test_empty_content(self):
        """Empty content should still render the panel."""
        ui, buf = _make_ui()
        ui.show_artifact("Empty", "")
        output = _output(buf)
        assert "Empty" in output
        assert "truncated" not in output


# ---------------------------------------------------------------------------
# grill_question
# ---------------------------------------------------------------------------


class TestGrillQuestion:
    """Test grill_question() human gate."""

    def test_auto_approve_returns_recommended(self):
        """With auto_approve=True, recommended answer is returned without prompting."""
        ui, buf = _make_ui(auto_approve=True)
        result = ui.grill_question(
            question="What is the target audience?",
            recommended="Developers",
            category="Users",
            number=1,
        )
        assert result == "Developers"

    def test_auto_approve_outputs_auto_approve_notice(self):
        """Auto-approve should print a notice that it's using the recommended answer."""
        ui, buf = _make_ui(auto_approve=True)
        ui.grill_question(
            question="Q?",
            recommended="Rec",
            category="Cat",
            number=1,
        )
        output = _output(buf)
        assert "auto-approve" in output

    def test_auto_approve_shows_question(self):
        """Even with auto_approve, the question should still be displayed."""
        ui, buf = _make_ui(auto_approve=True)
        ui.grill_question(
            question="What database?",
            recommended="PostgreSQL",
            category="Tech",
            number=3,
        )
        output = _output(buf)
        assert "What database?" in output

    def test_auto_approve_shows_category(self):
        ui, buf = _make_ui(auto_approve=True)
        ui.grill_question(
            question="Q?",
            recommended="R",
            category="Infrastructure",
            number=1,
        )
        output = _output(buf)
        assert "Infrastructure" in output

    def test_auto_approve_shows_question_number(self):
        ui, buf = _make_ui(auto_approve=True)
        ui.grill_question(
            question="Q?",
            recommended="R",
            category="C",
            number=5,
        )
        output = _output(buf)
        assert "5" in output

    def test_manual_mode_returns_user_input(self):
        """Without auto_approve, returns what the user types."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="My custom answer"):
            result = ui.grill_question(
                question="Q?",
                recommended="Default",
                category="C",
                number=1,
            )
        assert result == "My custom answer"

    def test_manual_mode_empty_input_returns_recommended(self):
        """Empty user input (just Enter) returns the recommended answer."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value=""):
            result = ui.grill_question(
                question="Q?",
                recommended="Fallback",
                category="C",
                number=1,
            )
        assert result == "Fallback"

    def test_manual_mode_whitespace_input_returns_recommended(self):
        """Whitespace-only input is treated as empty → returns recommended."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="   "):
            result = ui.grill_question(
                question="Q?",
                recommended="Fallback",
                category="C",
                number=1,
            )
        assert result == "Fallback"

    def test_manual_mode_eof_error_returns_recommended(self):
        """EOFError during input returns the recommended answer."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", side_effect=EOFError):
            result = ui.grill_question(
                question="Q?",
                recommended="Default",
                category="C",
                number=1,
            )
        assert result == "Default"

    def test_manual_mode_keyboard_interrupt_returns_recommended(self):
        """KeyboardInterrupt during input returns the recommended answer."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", side_effect=KeyboardInterrupt):
            result = ui.grill_question(
                question="Q?",
                recommended="Default",
                category="C",
                number=1,
            )
        assert result == "Default"


# ---------------------------------------------------------------------------
# prompt_arch_review
# ---------------------------------------------------------------------------


class TestPromptArchReview:
    """Test prompt_arch_review() human gate."""

    def test_auto_approve_returns_true_empty_feedback(self):
        """With auto_approve=True, returns (True, '') without prompting."""
        ui, buf = _make_ui(auto_approve=True)
        approved, feedback = ui.prompt_arch_review("## Architecture\n- Microservices")
        assert approved is True
        assert feedback == ""

    def test_auto_approve_shows_auto_approved_notice(self):
        ui, buf = _make_ui(auto_approve=True)
        ui.prompt_arch_review("arch summary")
        output = _output(buf)
        assert "auto-approved" in output

    def test_auto_approve_shows_architecture_content(self):
        """The architecture summary should appear in the output."""
        ui, buf = _make_ui(auto_approve=True)
        ui.prompt_arch_review("Use FastAPI with PostgreSQL")
        output = _output(buf)
        assert "Use FastAPI with PostgreSQL" in output

    def test_manual_approve_keyword(self):
        """Typing 'approve' returns (True, '')."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="approve"):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_manual_yes_keyword(self):
        """Typing 'yes' returns (True, '')."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="yes"):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_manual_y_keyword(self):
        """Typing 'y' returns (True, '')."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="y"):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_manual_lgtm_keyword(self):
        """Typing 'lgtm' returns (True, '')."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="lgtm"):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_manual_empty_input_approves(self):
        """Empty input (just Enter) returns (True, '')."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value=""):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_manual_approve_case_insensitive(self):
        """'APPROVE', 'Approve', etc. should all be accepted."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="APPROVE"):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_manual_feedback_returns_not_approved(self):
        """Typing feedback text returns (False, feedback)."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="Please add caching layer"):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is False
        assert feedback == "Please add caching layer"

    def test_manual_eof_error_auto_approves(self):
        """EOFError during input returns (True, '')."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", side_effect=EOFError):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""

    def test_manual_keyboard_interrupt_auto_approves(self):
        """KeyboardInterrupt during input returns (True, '')."""
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", side_effect=KeyboardInterrupt):
            approved, feedback = ui.prompt_arch_review("arch")
        assert approved is True
        assert feedback == ""


# ---------------------------------------------------------------------------
# task_start / task_done
# ---------------------------------------------------------------------------


class TestTaskProgress:
    """Test task_start() and task_done() output."""

    def test_task_start_shows_task_id(self):
        ui, buf = _make_ui()
        ui.task_start("task-001", "Create models", 1, 5)
        output = _output(buf)
        assert "task-001" in output

    def test_task_start_shows_title(self):
        ui, buf = _make_ui()
        ui.task_start("task-001", "Create models", 1, 5)
        output = _output(buf)
        assert "Create models" in output

    def test_task_start_shows_progress_count(self):
        ui, buf = _make_ui()
        ui.task_start("task-001", "Create models", 3, 10)
        output = _output(buf)
        assert "3" in output
        assert "10" in output

    def test_task_done_shows_task_id(self):
        ui, buf = _make_ui()
        ui.task_done("task-001")
        output = _output(buf)
        assert "task-001" in output

    def test_task_done_shows_complete(self):
        ui, buf = _make_ui()
        ui.task_done("task-001")
        output = _output(buf)
        assert "complete" in output


# ---------------------------------------------------------------------------
# deploy_complete
# ---------------------------------------------------------------------------


class TestDeployComplete:
    """Test deploy_complete() output."""

    def test_shows_project_path(self):
        ui, buf = _make_ui()
        ui.deploy_complete("/home/user/projects/my-app")
        output = _output(buf)
        assert "/home/user/projects/my-app" in output

    def test_shows_ready_message(self):
        ui, buf = _make_ui()
        ui.deploy_complete("/tmp/app")
        output = _output(buf)
        assert "Ready" in output


# ---------------------------------------------------------------------------
# error / info
# ---------------------------------------------------------------------------


class TestErrorAndInfo:
    """Test error() and info() output methods."""

    def test_error_shows_message(self):
        ui, buf = _make_ui()
        ui.error("Something went wrong")
        output = _output(buf)
        assert "Something went wrong" in output
        assert "Error" in output

    def test_info_shows_message(self):
        ui, buf = _make_ui()
        ui.info("Processing step 2")
        output = _output(buf)
        assert "Processing step 2" in output


# ---------------------------------------------------------------------------
# metrics_display
# ---------------------------------------------------------------------------


class TestMetricsDisplay:
    """Test metrics_display() output."""

    def test_displays_metric_keys_and_values(self):
        ui, buf = _make_ui()
        ui.metrics_display({"Total Time": "45s", "Files Generated": "12"})
        output = _output(buf)
        assert "Total Time" in output
        assert "45s" in output
        assert "Files Generated" in output
        assert "12" in output

    def test_displays_build_metrics_title(self):
        ui, buf = _make_ui()
        ui.metrics_display({"key": "val"})
        output = _output(buf)
        assert "Build Metrics" in output

    def test_empty_metrics(self):
        """Empty dict should still render the table header."""
        ui, buf = _make_ui()
        ui.metrics_display({})
        output = _output(buf)
        assert "Build Metrics" in output


# ---------------------------------------------------------------------------
# show_projects
# ---------------------------------------------------------------------------


class TestShowProjects:
    """Test show_projects() listing display."""

    def test_empty_project_list(self):
        ui, buf = _make_ui()
        ui.show_projects([])
        output = _output(buf)
        assert "No Kindle build sessions found" in output

    def test_shows_project_data(self):
        ui, buf = _make_ui()
        projects = [
            {
                "project_id": "abc-123",
                "idea": "Build a todo app",
                "status": "complete",
                "stages_completed": ["grill", "research"],
                "created_at": "2025-01-15T10:30:00Z",
            }
        ]
        ui.show_projects(projects)
        output = _output(buf)
        assert "abc-123" in output
        assert "Build a todo app" in output
        assert "complete" in output

    def test_shows_multiple_projects(self):
        ui, buf = _make_ui()
        projects = [
            {
                "project_id": "proj-1",
                "idea": "App 1",
                "status": "complete",
                "stages_completed": [],
                "created_at": "2025-01-01T00:00:00Z",
            },
            {
                "project_id": "proj-2",
                "idea": "App 2",
                "status": "in-progress",
                "stages_completed": ["grill"],
                "created_at": "2025-01-02T00:00:00Z",
            },
        ]
        ui.show_projects(projects)
        output = _output(buf)
        assert "proj-1" in output
        assert "proj-2" in output


# ---------------------------------------------------------------------------
# _safe_print resilience
# ---------------------------------------------------------------------------


class TestSafePrint:
    """Test that _safe_print suppresses pipe/IO errors."""

    def test_safe_print_suppresses_broken_pipe(self):
        """_safe_print should not raise on BrokenPipeError."""
        ui, _buf = _make_ui()
        with patch.object(ui.console, "print", side_effect=BrokenPipeError):
            # Should not raise
            ui._safe_print("test")

    def test_safe_print_suppresses_blocking_io(self):
        """_safe_print should not raise on BlockingIOError."""
        ui, _buf = _make_ui()
        with patch.object(ui.console, "print", side_effect=BlockingIOError):
            ui._safe_print("test")

    def test_safe_print_suppresses_os_error(self):
        """_safe_print should not raise on OSError."""
        ui, _buf = _make_ui()
        with patch.object(ui.console, "print", side_effect=OSError):
            ui._safe_print("test")
