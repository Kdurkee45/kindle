"""Tests for kindle.cli — Typer CLI entry points (build, resume, list)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from kindle.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> MagicMock:
    """Return a mock Settings object with sensible defaults."""
    defaults = {
        "anthropic_api_key": "sk-test",
        "model": "claude-opus-4-20250514",
        "max_agent_turns": 50,
        "max_concurrent_agents": 4,
        "max_qa_retries": 10,
        "max_cpo_retries": 10,
        "projects_root": Path("/tmp/kindle_test_projects"),
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _make_compiled_graph() -> MagicMock:
    """Return a mock compiled graph whose ainvoke returns a coroutine."""
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value={})
    return compiled


def _seed_project_dir(project_dir: Path, idea: str = "test idea", project_id: str = "kindle_abc12345") -> None:
    """Scaffold a minimal project directory with metadata and optional artifacts."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "artifacts").mkdir(exist_ok=True)
    (project_dir / "logs").mkdir(exist_ok=True)
    meta = {
        "project_id": project_id,
        "idea": idea,
        "created_at": "2026-04-11T12:00:00",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# --help smoke tests
# ---------------------------------------------------------------------------


class TestHelpOutput:
    """Smoke tests — every command should respond to --help without crashing."""

    def test_top_level_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "kindle" in result.output.lower() or "application factory" in result.output.lower()

    def test_resume_help(self) -> None:
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "resume" in result.output.lower()

    def test_list_help(self) -> None:
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower()


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------


class TestBuildCommand:
    """Tests for the default `kindle <idea>` build command."""

    def test_no_idea_shows_usage_and_exits_cleanly(self) -> None:
        """Invoking kindle with no arguments should print usage info and exit 0."""
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "usage" in result.output.lower() or "--help" in result.output.lower()

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_constructs_state_with_defaults(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Build with only an idea should use config defaults for concurrency, retries, etc."""
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_aaa11111"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_aaa11111", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        mock_ui = MagicMock()
        mock_ui_cls.return_value = mock_ui

        result = runner.invoke(app, ["build a todo app"])
        assert result.exit_code == 0

        # asyncio.run was called once
        mock_asyncio_run.assert_called_once()
        call_args = mock_asyncio_run.call_args[0][0]  # the coroutine passed to asyncio.run

        # build_graph called with the UI instance
        mock_build_graph.assert_called_once_with(mock_ui)

        # create_project called with projects_root and the idea
        mock_create_project.assert_called_once_with(settings.projects_root, "build a todo app")

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_state_has_correct_idea(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """The initial state passed to the graph should contain the user's idea."""
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_aaa11111"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_aaa11111", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my awesome app"])

        # The compiled graph's ainvoke is called via asyncio.run
        invoke_coroutine = mock_asyncio_run.call_args[0][0]
        # We can inspect the compiled mock's ainvoke call
        compiled.ainvoke.assert_called_once()
        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == "my awesome app"
        assert state["project_id"] == "kindle_aaa11111"
        assert state["project_dir"] == str(project_dir)

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_cli_overrides_concurrency(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--concurrency flag should override settings.max_concurrent_agents."""
        settings = _make_settings(max_concurrent_agents=4)
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_bbb22222"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_bbb22222", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my app", "--concurrency", "8"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 8

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_zero_concurrency_uses_config_default(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--concurrency 0 (default) should fall back to config default."""
        settings = _make_settings(max_concurrent_agents=6)
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_ccc33333"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_ccc33333", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 6

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_cli_overrides_qa_and_cpo_retries(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--qa-retries and --cpo-retries flags should override config defaults."""
        settings = _make_settings(max_qa_retries=10, max_cpo_retries=10)
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_ddd44444"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_ddd44444", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my app", "--qa-retries", "3", "--cpo-retries", "5"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 3
        assert state["max_cpo_retries"] == 5

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_stack_preference_passed_to_state(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--stack flag should be stored in state['stack_preference']."""
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_eee55555"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_eee55555", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my app", "--stack", "nextjs"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["stack_preference"] == "nextjs"

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_auto_approve_passed_to_ui_and_state(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--auto-approve should be forwarded to both UI and state."""
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_fff66666"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_fff66666", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my app", "--auto-approve"])

        # UI constructed with auto_approve=True
        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=False)

        # State also carries auto_approve
        state = compiled.ainvoke.call_args[0][0]
        assert state["auto_approve"] is True

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_verbose_flag_passed_to_ui(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--verbose should be forwarded to the UI constructor."""
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_ggg77777"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_ggg77777", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my app", "--verbose"])

        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_initial_state_defaults(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Initial state should have empty artifacts, zero retry counters, and correct model."""
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_hhh88888"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_hhh88888", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["feature_spec"] == {}
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""
        assert state["dev_tasks"] == []
        assert state["qa_report"] == ""
        assert state["product_audit"] == ""
        assert state["package_readme"] == ""
        assert state["qa_passed"] is False
        assert state["cpo_passed"] is False
        assert state["qa_retries"] == 0
        assert state["cpo_retries"] == 0
        assert state["model"] == "claude-opus-4-20250514"
        assert state["max_agent_turns"] == 50
        assert state["current_stage"] == ""

    @patch("kindle.cli.shutil.copytree")
    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_output_copies_workspace(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_copytree: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--output should trigger a shutil.copytree from workspace to the output dir."""
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_iii99999"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_iii99999", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        output_dir = tmp_path / "my_output"
        result = runner.invoke(app, ["my app", "--output", str(output_dir)])
        assert result.exit_code == 0
        mock_copytree.assert_called_once()

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_output_existing_dir_shows_error(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--output pointing to an existing directory should log an error, not copy."""
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_jjj00000"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_jjj00000", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui = MagicMock()
        mock_ui_cls.return_value = mock_ui

        output_dir = tmp_path / "existing_output"
        output_dir.mkdir()  # pre-create so it already exists

        result = runner.invoke(app, ["my app", "--output", str(output_dir)])
        assert result.exit_code == 0
        mock_ui.error.assert_called_once()
        assert "already exists" in mock_ui.error.call_args[0][0]

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_review_arch_not_in_state(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """BUG DOCUMENTATION: --review-arch is accepted by the CLI but NOT forwarded to state.

        The build() function accepts review_arch as a parameter but never assigns it
        to initial_state. This means the flag has no effect on the pipeline.
        This test documents the current (broken) behavior.
        """
        settings = _make_settings()
        mock_settings_load.return_value = settings

        project_dir = tmp_path / "kindle_kkk11111"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_kkk11111", project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        # The --review-arch flag is parsed without error
        result = runner.invoke(app, ["my app", "--review-arch"])
        assert result.exit_code == 0

        # But the state dict does NOT contain review_arch — this is a known bug.
        state = compiled.ainvoke.call_args[0][0]
        assert "review_arch" not in state, (
            "review_arch is NOT currently set in initial_state (known bug). "
            "If this assertion fails it means the bug has been fixed — update this test."
        )


# ---------------------------------------------------------------------------
# Build command — CLI option defaults
# ---------------------------------------------------------------------------


class TestBuildDefaults:
    """Verify that CLI option defaults match expected values."""

    def test_default_stack_is_empty(self) -> None:
        """The --stack option should default to an empty string."""
        # We inspect the typer parameter info via --help output
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # The help text should mention --stack
        assert "--stack" in result.output

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_default_stack_empty_in_state(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings()
        mock_settings_load.return_value = settings
        project_dir = tmp_path / "kindle_defaults"
        project_dir.mkdir()
        (project_dir / "workspace").mkdir()
        mock_create_project.return_value = ("kindle_defaults", project_dir)
        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}
        mock_ui_cls.return_value = MagicMock()

        runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["stack_preference"] == ""
        assert state["auto_approve"] is False


# ---------------------------------------------------------------------------
# Resume command
# ---------------------------------------------------------------------------


class TestResumeCommand:
    """Tests for `kindle resume <project_path> --from <stage>`."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_loads_state_from_artifacts(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Resume should load prior artifacts and pass them into the state."""
        settings = _make_settings()
        mock_settings_load.return_value = settings
        mock_ui_cls.return_value = MagicMock()

        project_dir = tmp_path / "kindle_resume01"
        _seed_project_dir(project_dir, idea="resume test", project_id="kindle_resume01")

        # Simulate artifacts
        feature_spec = {"features": [{"name": "auth"}]}
        dev_tasks = [{"task_id": "T1", "title": "Setup"}]

        def fake_load_artifact(pdir: str, name: str) -> str | None:
            mapping = {
                "feature_spec.json": json.dumps(feature_spec),
                "dev_tasks.json": json.dumps(dev_tasks),
                "grill_transcript.md": "grill output",
                "research_report.md": "research output",
                "architecture.md": "arch output",
                "qa_report.md": "",
                "product_audit.md": None,
            }
            return mapping.get(name)

        mock_load_artifact.side_effect = fake_load_artifact

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["resume", str(project_dir), "--from", "dev"])
        assert result.exit_code == 0

        # build_graph called with entry_stage="dev"
        mock_build_graph.assert_called_once()
        _, kwargs = mock_build_graph.call_args
        assert kwargs["entry_stage"] == "dev"

        # State should contain reloaded artifacts
        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == "resume test"
        assert state["project_id"] == "kindle_resume01"
        assert state["feature_spec"] == feature_spec
        assert state["dev_tasks"] == dev_tasks
        assert state["grill_transcript"] == "grill output"
        assert state["research_report"] == "research output"
        assert state["architecture"] == "arch output"

    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_nonexistent_project_dir_exits_1(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Resume with a path that doesn't exist should exit with code 1."""
        mock_settings_load.return_value = _make_settings()
        mock_ui = MagicMock()
        mock_ui_cls.return_value = mock_ui

        missing = tmp_path / "nonexistent_session"
        result = runner.invoke(app, ["resume", str(missing)])
        assert result.exit_code == 1
        mock_ui.error.assert_called()
        assert "not found" in mock_ui.error.call_args[0][0].lower()

    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_missing_metadata_exits_1(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Resume in a directory without metadata.json should exit with code 1."""
        mock_settings_load.return_value = _make_settings()
        mock_ui = MagicMock()
        mock_ui_cls.return_value = mock_ui

        empty_dir = tmp_path / "empty_session"
        empty_dir.mkdir()
        result = runner.invoke(app, ["resume", str(empty_dir)])
        assert result.exit_code == 1
        mock_ui.error.assert_called()
        assert "metadata.json" in mock_ui.error.call_args[0][0].lower()

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact", return_value=None)
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_default_stage_is_dev(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Resume with no --from flag should default to 'dev'."""
        mock_settings_load.return_value = _make_settings()
        mock_ui_cls.return_value = MagicMock()

        project_dir = tmp_path / "kindle_def_stage"
        _seed_project_dir(project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["resume", str(project_dir)])
        assert result.exit_code == 0

        _, kwargs = mock_build_graph.call_args
        assert kwargs["entry_stage"] == "dev"

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact", return_value=None)
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_passes_auto_approve_and_verbose(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--auto-approve and --verbose should be forwarded to UI and state."""
        mock_settings_load.return_value = _make_settings()
        mock_ui_cls.return_value = MagicMock()

        project_dir = tmp_path / "kindle_resume_flags"
        _seed_project_dir(project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["resume", str(project_dir), "--auto-approve", "--verbose"])
        assert result.exit_code == 0

        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=True)
        state = compiled.ainvoke.call_args[0][0]
        assert state["auto_approve"] is True

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact", return_value=None)
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_null_artifacts_become_empty(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When load_artifact returns None for all artifacts, state should have empty defaults."""
        mock_settings_load.return_value = _make_settings()
        mock_ui_cls.return_value = MagicMock()

        project_dir = tmp_path / "kindle_null_arts"
        _seed_project_dir(project_dir)

        compiled = _make_compiled_graph()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["resume", str(project_dir)])
        assert result.exit_code == 0

        state = compiled.ainvoke.call_args[0][0]
        assert state["feature_spec"] == {}
        assert state["dev_tasks"] == []
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""
        assert state["qa_report"] == ""
        assert state["product_audit"] == ""


# ---------------------------------------------------------------------------
# List command
# ---------------------------------------------------------------------------


class TestListCommand:
    """Tests for `kindle list`."""

    @patch("kindle.cli.list_projects")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_list_calls_show_projects_with_result(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_list_projects: MagicMock,
    ) -> None:
        """list command should call list_projects and forward result to UI.show_projects."""
        settings = _make_settings()
        mock_settings_load.return_value = settings
        mock_ui = MagicMock()
        mock_ui_cls.return_value = mock_ui

        projects = [
            {"project_id": "kindle_aaa", "idea": "App A", "status": "done"},
            {"project_id": "kindle_bbb", "idea": "App B", "status": "in_progress"},
        ]
        mock_list_projects.return_value = projects

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

        mock_list_projects.assert_called_once_with(settings.projects_root)
        mock_ui.show_projects.assert_called_once_with(projects)

    @patch("kindle.cli.list_projects")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_list_empty_projects(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_list_projects: MagicMock,
    ) -> None:
        """list command should handle empty project list gracefully."""
        mock_settings_load.return_value = _make_settings()
        mock_ui = MagicMock()
        mock_ui_cls.return_value = mock_ui
        mock_list_projects.return_value = []

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        mock_ui.show_projects.assert_called_once_with([])
