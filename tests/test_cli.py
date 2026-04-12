"""Tests for kindle.cli — the user-facing Typer CLI."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from kindle.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings(**overrides):
    """Build a mock Settings with realistic defaults."""
    s = MagicMock()
    s.model = overrides.get("model", "claude-opus-4-20250514")
    s.max_agent_turns = overrides.get("max_agent_turns", 50)
    s.max_concurrent_agents = overrides.get("max_concurrent_agents", 4)
    s.max_qa_retries = overrides.get("max_qa_retries", 10)
    s.max_cpo_retries = overrides.get("max_cpo_retries", 10)
    s.projects_root = overrides.get("projects_root", Path("/tmp/kindle/projects"))
    return s


@contextmanager
def _build_env(settings=None, project_id="kindle_abc12345", project_dir=None):
    """Context manager patching all dependencies for the ``build`` command."""
    if settings is None:
        settings = _fake_settings()
    if project_dir is None:
        project_dir = Path("/tmp/kindle/projects") / project_id

    ui_instance = MagicMock()
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value={})
    workspace = project_dir / "workspace"

    with (
        patch("kindle.cli.Settings.load", return_value=settings) as mock_settings_load,
        patch("kindle.cli.UI", return_value=ui_instance) as mock_ui_cls,
        patch("kindle.cli.create_project", return_value=(project_id, project_dir)) as mock_create,
        patch("kindle.cli.build_graph", return_value=compiled) as mock_build_graph,
        patch("kindle.cli.workspace_path", return_value=workspace) as mock_ws,
        patch("kindle.cli.asyncio.run") as mock_async_run,
    ):
        yield SimpleNamespace(
            settings=settings,
            ui=ui_instance,
            compiled=compiled,
            workspace=workspace,
            project_id=project_id,
            project_dir=project_dir,
            UI_cls=mock_ui_cls,
            Settings_load=mock_settings_load,
            create_project=mock_create,
            build_graph=mock_build_graph,
            workspace_path=mock_ws,
            async_run=mock_async_run,
        )


def _make_resume_dir(tmp_path, project_id="kindle_test123", idea="A test app"):
    """Create a minimal project directory for resume tests."""
    project_dir = tmp_path / project_id
    (project_dir / "artifacts").mkdir(parents=True)
    meta = {"project_id": project_id, "idea": idea}
    (project_dir / "metadata.json").write_text(json.dumps(meta))
    return project_dir


@contextmanager
def _resume_env(project_dir, settings=None):
    """Context manager patching all dependencies for the ``resume`` command."""
    if settings is None:
        settings = _fake_settings()

    ui_instance = MagicMock()
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value={})
    workspace = project_dir / "workspace"

    with (
        patch("kindle.cli.Settings.load", return_value=settings),
        patch("kindle.cli.UI", return_value=ui_instance) as mock_ui_cls,
        patch("kindle.cli.build_graph", return_value=compiled) as mock_build_graph,
        patch("kindle.cli.workspace_path", return_value=workspace),
        patch("kindle.cli.asyncio.run") as mock_async_run,
        patch("kindle.cli.load_artifact", return_value=None) as mock_load,
    ):
        yield SimpleNamespace(
            settings=settings,
            ui=ui_instance,
            compiled=compiled,
            workspace=workspace,
            project_dir=project_dir,
            UI_cls=mock_ui_cls,
            build_graph=mock_build_graph,
            async_run=mock_async_run,
            load_artifact=mock_load,
        )


# ---------------------------------------------------------------------------
# build — no idea provided
# ---------------------------------------------------------------------------


class TestBuildNoIdea:
    """When no idea argument is provided, show usage and exit zero."""

    def test_no_idea_prints_usage(self) -> None:
        # Mock Settings.load defensively even though it isn't reached
        with patch("kindle.cli.Settings.load", return_value=_fake_settings()):
            result = runner.invoke(app, [])
        assert "Usage:" in result.output

    def test_no_idea_exits_zero(self) -> None:
        with patch("kindle.cli.Settings.load", return_value=_fake_settings()):
            result = runner.invoke(app, [])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# build — happy path
# ---------------------------------------------------------------------------


class TestBuildHappyPath:
    """Normal build execution with all dependencies mocked."""

    def test_exit_code_zero(self) -> None:
        with _build_env():
            result = runner.invoke(app, ["Build a todo app"])
        assert result.exit_code == 0

    def test_invokes_create_project_with_idea(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build a todo app"])
        env.create_project.assert_called_once_with(
            env.settings.projects_root, "Build a todo app"
        )

    def test_calls_build_graph_with_ui(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build a todo app"])
        env.build_graph.assert_called_once_with(env.ui)

    def test_calls_asyncio_run(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build a todo app"])
        env.async_run.assert_called_once()

    def test_banner_called_with_idea_and_project_id(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build a todo app"])
        env.ui.banner.assert_called_once_with("Build a todo app", env.project_id)

    def test_workspace_info_printed(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build a todo app"])
        info_messages = [c[0][0] for c in env.ui.info.call_args_list]
        assert any("Project built at:" in m for m in info_messages)


# ---------------------------------------------------------------------------
# build — settings overrides
# ---------------------------------------------------------------------------


class TestBuildSettingsOverrides:
    """CLI flags override Settings defaults when non-zero."""

    def test_concurrency_overrides_settings(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "--concurrency", "8"])
        state = env.compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 8

    def test_qa_retries_overrides_settings(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "--qa-retries", "5"])
        state = env.compiled.ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 5

    def test_cpo_retries_overrides_settings(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "--cpo-retries", "3"])
        state = env.compiled.ainvoke.call_args[0][0]
        assert state["max_cpo_retries"] == 3

    def test_zero_concurrency_uses_config_default(self) -> None:
        settings = _fake_settings(max_concurrent_agents=6)
        with _build_env(settings=settings) as env:
            runner.invoke(app, ["Build an app", "--concurrency", "0"])
        state = env.compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 6

    def test_zero_qa_retries_uses_config_default(self) -> None:
        settings = _fake_settings(max_qa_retries=7)
        with _build_env(settings=settings) as env:
            runner.invoke(app, ["Build an app", "--qa-retries", "0"])
        state = env.compiled.ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 7


# ---------------------------------------------------------------------------
# build — flag propagation to UI
# ---------------------------------------------------------------------------


class TestBuildFlagPropagation:
    """auto_approve and verbose must be forwarded to the UI constructor."""

    def test_auto_approve_passed_to_ui(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "--auto-approve"])
        env.UI_cls.assert_called_once_with(auto_approve=True, verbose=False)

    def test_verbose_passed_to_ui(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "--verbose"])
        env.UI_cls.assert_called_once_with(auto_approve=False, verbose=True)

    def test_both_flags_together(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "--auto-approve", "--verbose"])
        env.UI_cls.assert_called_once_with(auto_approve=True, verbose=True)


# ---------------------------------------------------------------------------
# build — output directory
# ---------------------------------------------------------------------------


class TestBuildOutput:
    """--output flag copies workspace to a target directory."""

    def test_output_copies_workspace(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        output_dir = tmp_path / "my_output"  # Does NOT exist

        with _build_env(project_dir=project_dir) as env:
            with patch("shutil.copytree") as mock_copy:
                result = runner.invoke(
                    app, ["Build an app", "--output", str(output_dir)]
                )

        assert result.exit_code == 0
        # The code resolves the output path: Path(output).expanduser().resolve()
        expected_output = str(Path(str(output_dir)).expanduser().resolve())
        mock_copy.assert_called_once_with(str(env.workspace), expected_output)

    def test_output_already_exists_shows_error(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        output_dir = tmp_path / "existing_output"
        output_dir.mkdir()  # Create it so .exists() returns True

        with _build_env(project_dir=project_dir) as env:
            result = runner.invoke(
                app, ["Build an app", "--output", str(output_dir)]
            )

        assert result.exit_code == 0
        env.ui.error.assert_called_once()
        assert "already exists" in env.ui.error.call_args[0][0]


# ---------------------------------------------------------------------------
# build — review_arch flag
# ---------------------------------------------------------------------------


class TestBuildReviewArch:
    """--review-arch flag is accepted by the CLI without error."""

    def test_review_arch_flag_accepted(self) -> None:
        with _build_env():
            result = runner.invoke(app, ["Build an app", "--review-arch"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# build — short flag aliases
# ---------------------------------------------------------------------------


class TestBuildShortFlags:
    """Short flag aliases (-s, -c, -v, -o) work correctly."""

    def test_short_stack_flag(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "-s", "react"])
        state = env.compiled.ainvoke.call_args[0][0]
        assert state["stack_preference"] == "react"

    def test_short_concurrency_flag(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "-c", "6"])
        state = env.compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 6

    def test_short_verbose_flag(self) -> None:
        with _build_env() as env:
            runner.invoke(app, ["Build an app", "-v"])
        env.UI_cls.assert_called_once_with(auto_approve=False, verbose=True)


# ---------------------------------------------------------------------------
# resume — happy path
# ---------------------------------------------------------------------------


class TestResumeHappyPath:
    """Normal resume execution with mocked dependencies."""

    def test_resume_loads_artifacts_and_runs_graph(self, tmp_path: Path) -> None:
        project_dir = _make_resume_dir(tmp_path)
        with _resume_env(project_dir) as env:
            result = runner.invoke(app, ["resume", str(project_dir)])
        assert result.exit_code == 0
        env.build_graph.assert_called_once_with(env.ui, entry_stage="dev")
        env.async_run.assert_called_once()

    def test_resume_from_stage_forwarded_to_build_graph(self, tmp_path: Path) -> None:
        project_dir = _make_resume_dir(tmp_path)
        with _resume_env(project_dir) as env:
            result = runner.invoke(
                app, ["resume", str(project_dir), "--from", "qa"]
            )
        assert result.exit_code == 0
        env.build_graph.assert_called_once_with(env.ui, entry_stage="qa")

    def test_resume_reloads_feature_spec_artifact(self, tmp_path: Path) -> None:
        project_dir = _make_resume_dir(tmp_path)
        feature_spec = {"app_name": "TestApp", "core_features": ["auth"]}

        def fake_load(pd, name):
            if name == "feature_spec.json":
                return json.dumps(feature_spec)
            return None

        settings = _fake_settings()
        mock_ui = MagicMock()
        mock_compiled = MagicMock()
        mock_compiled.ainvoke = AsyncMock(return_value={})

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.workspace_path", return_value=project_dir / "workspace"),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.load_artifact", side_effect=fake_load),
        ):
            result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 0
        state = mock_compiled.ainvoke.call_args[0][0]
        assert state["feature_spec"] == feature_spec

    def test_resume_auto_approve_and_verbose_passed_to_ui(
        self, tmp_path: Path
    ) -> None:
        project_dir = _make_resume_dir(tmp_path)
        with _resume_env(project_dir) as env:
            result = runner.invoke(
                app,
                ["resume", str(project_dir), "--auto-approve", "--verbose"],
            )
        assert result.exit_code == 0
        env.UI_cls.assert_called_once_with(auto_approve=True, verbose=True)

    def test_resume_null_artifacts_produce_empty_defaults(
        self, tmp_path: Path
    ) -> None:
        project_dir = _make_resume_dir(tmp_path)
        with _resume_env(project_dir) as env:
            result = runner.invoke(app, ["resume", str(project_dir)])
        assert result.exit_code == 0
        state = env.compiled.ainvoke.call_args[0][0]
        assert state["feature_spec"] == {}
        assert state["dev_tasks"] == []
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""
        assert state["qa_report"] == ""
        assert state["product_audit"] == ""


# ---------------------------------------------------------------------------
# resume — error paths
# ---------------------------------------------------------------------------


class TestResumeErrors:
    """Resume must exit 1 for missing directories or metadata."""

    def test_missing_directory_exits_with_error(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        mock_ui = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
        ):
            result = runner.invoke(app, ["resume", str(nonexistent)])

        assert result.exit_code == 1
        mock_ui.error.assert_called_once()
        assert "not found" in mock_ui.error.call_args[0][0]

    def test_missing_metadata_json_exits_with_error(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "kindle_nometa"
        project_dir.mkdir()
        # No metadata.json!
        mock_ui = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
        ):
            result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 1
        mock_ui.error.assert_called_once()
        assert "metadata.json" in mock_ui.error.call_args[0][0]


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListCommand:
    """The ``list`` subcommand delegates to list_projects + UI.show_projects."""

    def test_list_delegates_to_list_projects_and_show(self) -> None:
        settings = _fake_settings()
        mock_ui = MagicMock()
        projects = [{"project_id": "p1", "idea": "test"}]

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.list_projects", return_value=projects) as mock_lp,
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_lp.assert_called_once_with(settings.projects_root)
        mock_ui.show_projects.assert_called_once_with(projects)

    def test_list_empty_projects(self) -> None:
        settings = _fake_settings()
        mock_ui = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_ui.show_projects.assert_called_once_with([])
