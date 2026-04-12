"""Tests for kindle.cli — Typer CLI commands (build, resume, list)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from kindle.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers: shared mock factories
# ---------------------------------------------------------------------------


def _default_settings(**overrides: object) -> MagicMock:
    """Return a mock Settings object with sensible defaults."""
    s = MagicMock()
    s.model = overrides.get("model", "claude-opus-4-20250514")
    s.max_agent_turns = overrides.get("max_agent_turns", 50)
    s.max_concurrent_agents = overrides.get("max_concurrent_agents", 4)
    s.max_qa_retries = overrides.get("max_qa_retries", 10)
    s.max_cpo_retries = overrides.get("max_cpo_retries", 10)
    s.projects_root = overrides.get("projects_root", Path("/tmp/projects"))
    return s


def _patch_build_deps(
    *,
    settings: MagicMock | None = None,
    project_id: str = "kindle_abc12345",
    project_dir: Path = Path("/tmp/projects/kindle_abc12345"),
    workspace: Path = Path("/tmp/projects/kindle_abc12345/workspace"),
    ainvoke_return: dict | None = None,
) -> dict:
    """Return a dict of mock patches suitable for the build command.

    Usage::
        deps = _patch_build_deps()
        with patch(..., deps["settings"]), ...:
    """
    if settings is None:
        settings = _default_settings()

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=ainvoke_return or {})

    return {
        "settings": settings,
        "create_project": MagicMock(return_value=(project_id, project_dir)),
        "workspace_path": MagicMock(return_value=workspace),
        "build_graph": MagicMock(return_value=mock_graph),
        "ui_cls": MagicMock(),
        "mock_graph": mock_graph,
    }


# ---------------------------------------------------------------------------
# build command — no idea provided
# ---------------------------------------------------------------------------


class TestBuildNoIdea:
    """When no idea is given the CLI should print usage and exit cleanly."""

    @patch("kindle.cli.Settings.load", return_value=_default_settings())
    def test_no_args_prints_usage_and_exits_zero(self, _mock_settings: MagicMock) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output or "kindle" in result.output.lower()

    @patch("kindle.cli.Settings.load", return_value=_default_settings())
    def test_empty_string_idea_prints_usage(self, _mock_settings: MagicMock) -> None:
        result = runner.invoke(app, [""])
        assert result.exit_code == 0
        assert "Usage:" in result.output


# ---------------------------------------------------------------------------
# build command — happy path with required --idea
# ---------------------------------------------------------------------------


class TestBuildHappyPath:
    """The build command should create a project and invoke the graph."""

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_creates_project_and_invokes_graph(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        settings = _default_settings()
        mock_settings_load.return_value = settings
        mock_create_project.return_value = ("kindle_abc12345", Path("/tmp/p/kindle_abc12345"))
        mock_workspace_path.return_value = Path("/tmp/p/kindle_abc12345/workspace")

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        result = runner.invoke(app, ["Build me a todo app"])
        assert result.exit_code == 0

        # Project was created with correct arguments
        mock_create_project.assert_called_once_with(settings.projects_root, "Build me a todo app")

        # Graph was built and invoked
        mock_build_graph.assert_called_once()
        compiled.ainvoke.assert_awaited_once()

        # Verify the state passed to ainvoke
        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == "Build me a todo app"
        assert state["project_id"] == "kindle_abc12345"
        assert state["project_dir"] == str(Path("/tmp/p/kindle_abc12345"))

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_build_passes_default_state_fields(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        settings = _default_settings()
        mock_settings_load.return_value = settings
        mock_create_project.return_value = ("kindle_00000000", Path("/tmp/p/kindle_00000000"))
        mock_workspace_path.return_value = Path("/tmp/p/kindle_00000000/workspace")

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        result = runner.invoke(app, ["some idea"])
        assert result.exit_code == 0

        state = compiled.ainvoke.call_args[0][0]
        # Verify default state shape
        assert state["stack_preference"] == ""
        assert state["auto_approve"] is False
        assert state["max_concurrent_agents"] == 4
        assert state["max_qa_retries"] == 10
        assert state["max_cpo_retries"] == 10
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


# ---------------------------------------------------------------------------
# build command — optional CLI flags
# ---------------------------------------------------------------------------


class TestBuildOptionalFlags:
    """CLI options should be passed through to the initial state."""

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_stack_flag_passes_through(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--stack", "react"])
        state = compiled.ainvoke.call_args[0][0]
        assert state["stack_preference"] == "react"

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_auto_approve_flag(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--auto-approve"])
        state = compiled.ainvoke.call_args[0][0]
        assert state["auto_approve"] is True

        # UI should also be created with auto_approve=True
        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=False)

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_concurrency_flag_overrides_config(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings(max_concurrent_agents=4)
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--concurrency", "8"])
        state = compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 8

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_concurrency_zero_uses_config_default(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings(max_concurrent_agents=4)
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--concurrency", "0"])
        state = compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 4  # falls back to config

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_qa_retries_flag_overrides_config(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings(max_qa_retries=10)
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--qa-retries", "5"])
        state = compiled.ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 5

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_qa_retries_zero_uses_config_default(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings(max_qa_retries=10)
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--qa-retries", "0"])
        state = compiled.ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 10

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_cpo_retries_flag_overrides_config(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings(max_cpo_retries=10)
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--cpo-retries", "3"])
        state = compiled.ainvoke.call_args[0][0]
        assert state["max_cpo_retries"] == 3

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_cpo_retries_zero_uses_config_default(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings(max_cpo_retries=10)
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--cpo-retries", "0"])
        state = compiled.ainvoke.call_args[0][0]
        assert state["max_cpo_retries"] == 10

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_verbose_flag(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["my idea", "--verbose"])
        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_all_flags_combined(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        result = runner.invoke(
            app,
            [
                "Build a chat app",
                "--stack", "nextjs",
                "--auto-approve",
                "--concurrency", "16",
                "--qa-retries", "3",
                "--cpo-retries", "2",
                "--verbose",
            ],
        )
        assert result.exit_code == 0

        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == "Build a chat app"
        assert state["stack_preference"] == "nextjs"
        assert state["auto_approve"] is True
        assert state["max_concurrent_agents"] == 16
        assert state["max_qa_retries"] == 3
        assert state["max_cpo_retries"] == 2

        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=True)


# ---------------------------------------------------------------------------
# build command — output directory copy
# ---------------------------------------------------------------------------


class TestBuildOutputFlag:
    """The --output flag should copy the workspace to the target directory."""

    @patch("kindle.cli.shutil.copytree")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_output_flag_copies_workspace(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        mock_copytree: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        ws = Path("/tmp/kid/workspace")
        mock_workspace_path.return_value = ws
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        # Target does not exist
        output_dir = tmp_path / "my_output"
        result = runner.invoke(app, ["my idea", "--output", str(output_dir)])
        assert result.exit_code == 0
        mock_copytree.assert_called_once_with(str(ws), str(output_dir))

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_output_flag_errors_if_dir_exists(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        # Target already exists
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        result = runner.invoke(app, ["my idea", "--output", str(existing_dir)])
        assert result.exit_code == 0  # doesn't crash, just prints error

        # UI.error was called
        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        error_msg = ui_instance.error.call_args[0][0]
        assert "already exists" in error_msg

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_no_output_flag_skips_copy(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_create_project.return_value = ("kid", Path("/tmp/kid"))
        mock_workspace_path.return_value = Path("/tmp/kid/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        with patch("kindle.cli.shutil.copytree") as mock_copytree:
            runner.invoke(app, ["my idea"])
            mock_copytree.assert_not_called()


# ---------------------------------------------------------------------------
# build command — UI lifecycle
# ---------------------------------------------------------------------------


class TestBuildUILifecycle:
    """Verify that banner and info are called during a build."""

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_banner_called_with_idea_and_project_id(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_create_project.return_value = ("kindle_banner01", Path("/tmp/p/kindle_banner01"))
        mock_workspace_path.return_value = Path("/tmp/p/kindle_banner01/workspace")
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["A weather app"])

        ui_instance = mock_ui_cls.return_value
        ui_instance.banner.assert_called_once_with("A weather app", "kindle_banner01")

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_info_called_with_project_paths(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_create_project: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        project_dir = Path("/tmp/p/kindle_info01")
        mock_create_project.return_value = ("kindle_info01", project_dir)
        ws = Path("/tmp/p/kindle_info01/workspace")
        mock_workspace_path.return_value = ws

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["An app"])

        ui_instance = mock_ui_cls.return_value
        info_calls = [c[0][0] for c in ui_instance.info.call_args_list]
        assert any(str(ws) in msg for msg in info_calls)
        assert any(str(project_dir) in msg for msg in info_calls)


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListCommand:
    """The 'list' command should load projects and render them."""

    @patch("kindle.cli.list_projects")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_list_calls_show_projects(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_list_projects: MagicMock,
    ) -> None:
        settings = _default_settings()
        mock_settings_load.return_value = settings
        mock_list_projects.return_value = [
            {"project_id": "kindle_aaa", "idea": "A todo app", "status": "completed"},
        ]

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

        mock_list_projects.assert_called_once_with(settings.projects_root)
        ui_instance = mock_ui_cls.return_value
        ui_instance.show_projects.assert_called_once()
        shown = ui_instance.show_projects.call_args[0][0]
        assert len(shown) == 1
        assert shown[0]["project_id"] == "kindle_aaa"

    @patch("kindle.cli.list_projects")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_list_with_no_projects(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_list_projects: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        mock_list_projects.return_value = []

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

        ui_instance = mock_ui_cls.return_value
        ui_instance.show_projects.assert_called_once_with([])

    @patch("kindle.cli.list_projects")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_list_with_multiple_projects(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_list_projects: MagicMock,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        projects = [
            {"project_id": "kindle_001", "idea": "App 1", "status": "completed"},
            {"project_id": "kindle_002", "idea": "App 2", "status": "in_progress"},
            {"project_id": "kindle_003", "idea": "App 3", "status": "completed"},
        ]
        mock_list_projects.return_value = projects

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

        ui_instance = mock_ui_cls.return_value
        shown = ui_instance.show_projects.call_args[0][0]
        assert len(shown) == 3


# ---------------------------------------------------------------------------
# resume command — error cases
# ---------------------------------------------------------------------------


class TestResumeErrors:
    """The resume command should fail gracefully for invalid sessions."""

    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_nonexistent_dir_exits_1(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        missing = tmp_path / "does_not_exist"

        result = runner.invoke(app, ["resume", str(missing)])
        assert result.exit_code == 1

        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        assert "not found" in ui_instance.error.call_args[0][0]

    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_missing_metadata_exits_1(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()
        # Directory exists but no metadata.json
        session_dir = tmp_path / "kindle_no_meta"
        session_dir.mkdir()

        result = runner.invoke(app, ["resume", str(session_dir)])
        assert result.exit_code == 1

        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        assert "metadata.json" in ui_instance.error.call_args[0][0]


# ---------------------------------------------------------------------------
# resume command — happy path
# ---------------------------------------------------------------------------


class TestResumeHappyPath:
    """The resume command should load state from disk and invoke the graph."""

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_loads_metadata_and_invokes_graph(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()

        # Create a valid session directory with metadata
        session_dir = tmp_path / "kindle_resume01"
        session_dir.mkdir()
        metadata = {
            "project_id": "kindle_resume01",
            "idea": "A resume test app",
            "status": "in_progress",
        }
        (session_dir / "metadata.json").write_text(json.dumps(metadata))

        # load_artifact returns different values based on name
        def artifact_side_effect(project_dir: str, name: str) -> str | None:
            artifacts = {
                "feature_spec.json": '{"feature": "login"}',
                "grill_transcript.md": "Q: What? A: This",
                "research_report.md": "Research findings",
                "architecture.md": "## Architecture",
                "dev_tasks.json": '[{"id": "task1"}]',
                "qa_report.md": "QA passed",
                "product_audit.md": "Looks good",
            }
            return artifacts.get(name)

        mock_load_artifact.side_effect = artifact_side_effect
        mock_workspace_path.return_value = session_dir / "workspace"

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        result = runner.invoke(app, ["resume", str(session_dir), "--from", "dev"])
        assert result.exit_code == 0

        # Graph built with entry_stage
        mock_build_graph.assert_called_once()
        _, kwargs = mock_build_graph.call_args
        assert kwargs.get("entry_stage") == "dev" or mock_build_graph.call_args[0][-1] == "dev"

        # Graph was invoked
        compiled.ainvoke.assert_awaited_once()

        # State loaded correctly from artifacts
        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == "A resume test app"
        assert state["project_id"] == "kindle_resume01"
        assert state["project_dir"] == str(session_dir)
        assert state["feature_spec"] == {"feature": "login"}
        assert state["grill_transcript"] == "Q: What? A: This"
        assert state["research_report"] == "Research findings"
        assert state["architecture"] == "## Architecture"
        assert state["dev_tasks"] == [{"id": "task1"}]
        assert state["qa_report"] == "QA passed"
        assert state["product_audit"] == "Looks good"

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_with_missing_artifacts_uses_defaults(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()

        session_dir = tmp_path / "kindle_empty_arts"
        session_dir.mkdir()
        metadata = {"project_id": "kindle_empty_arts", "idea": "Minimal session"}
        (session_dir / "metadata.json").write_text(json.dumps(metadata))

        # All artifacts missing
        mock_load_artifact.return_value = None
        mock_workspace_path.return_value = session_dir / "workspace"

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        result = runner.invoke(app, ["resume", str(session_dir)])
        assert result.exit_code == 0

        state = compiled.ainvoke.call_args[0][0]
        assert state["feature_spec"] == {}
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""
        assert state["dev_tasks"] == []
        assert state["qa_report"] == ""
        assert state["product_audit"] == ""

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_default_from_stage_is_dev(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()

        session_dir = tmp_path / "kindle_default_stage"
        session_dir.mkdir()
        metadata = {"project_id": "kindle_default_stage", "idea": "test"}
        (session_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_load_artifact.return_value = None
        mock_workspace_path.return_value = session_dir / "workspace"

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        # No --from flag → defaults to "dev"
        result = runner.invoke(app, ["resume", str(session_dir)])
        assert result.exit_code == 0

        ui_instance = mock_ui_cls.return_value
        mock_build_graph.assert_called_once()
        call_args = mock_build_graph.call_args
        # entry_stage should be "dev"
        assert call_args == ((ui_instance,), {"entry_stage": "dev"}) or call_args[1].get("entry_stage") == "dev"

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_auto_approve_and_verbose_flags(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()

        session_dir = tmp_path / "kindle_flags"
        session_dir.mkdir()
        metadata = {"project_id": "kindle_flags", "idea": "test"}
        (session_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_load_artifact.return_value = None
        mock_workspace_path.return_value = session_dir / "workspace"

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        result = runner.invoke(
            app, ["resume", str(session_dir), "--auto-approve", "--verbose"]
        )
        assert result.exit_code == 0

        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=True)

        state = compiled.ainvoke.call_args[0][0]
        assert state["auto_approve"] is True

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_banner_and_info_called(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_settings_load.return_value = _default_settings()

        session_dir = tmp_path / "kindle_ui_check"
        session_dir.mkdir()
        metadata = {"project_id": "kindle_ui_check", "idea": "Banner test"}
        (session_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_load_artifact.return_value = None
        ws = session_dir / "workspace"
        mock_workspace_path.return_value = ws

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["resume", str(session_dir), "--from", "qa"])

        ui_instance = mock_ui_cls.return_value
        ui_instance.banner.assert_called_once_with("Banner test", "kindle_ui_check")

        # info called with stage and project path
        info_calls = [c[0][0] for c in ui_instance.info.call_args_list]
        assert any("qa" in msg for msg in info_calls)
        assert any(str(ws) in msg for msg in info_calls)

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_uses_settings_for_state_fields(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _default_settings(
            model="claude-sonnet-4-20250514",
            max_agent_turns=25,
            max_concurrent_agents=8,
            max_qa_retries=5,
            max_cpo_retries=3,
        )
        mock_settings_load.return_value = settings

        session_dir = tmp_path / "kindle_settings"
        session_dir.mkdir()
        metadata = {"project_id": "kindle_settings", "idea": "test settings"}
        (session_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_load_artifact.return_value = None
        mock_workspace_path.return_value = session_dir / "workspace"

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        runner.invoke(app, ["resume", str(session_dir)])

        state = compiled.ainvoke.call_args[0][0]
        assert state["model"] == "claude-sonnet-4-20250514"
        assert state["max_agent_turns"] == 25
        assert state["max_concurrent_agents"] == 8
        assert state["max_qa_retries"] == 5
        assert state["max_cpo_retries"] == 3


# ---------------------------------------------------------------------------
# resume command — metadata edge cases
# ---------------------------------------------------------------------------


class TestResumeMetadataEdgeCases:
    """Edge cases in metadata handling during resume."""

    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.Settings.load")
    @patch("kindle.cli.UI")
    def test_resume_metadata_missing_idea_key(
        self,
        mock_ui_cls: MagicMock,
        mock_settings_load: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_workspace_path: MagicMock,
        tmp_path: Path,
    ) -> None:
        """metadata.json missing 'idea' key should default to empty string."""
        mock_settings_load.return_value = _default_settings()

        session_dir = tmp_path / "kindle_no_idea"
        session_dir.mkdir()
        metadata = {"project_id": "kindle_no_idea"}  # no "idea" key
        (session_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_load_artifact.return_value = None
        mock_workspace_path.return_value = session_dir / "workspace"

        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        mock_build_graph.return_value = compiled

        result = runner.invoke(app, ["resume", str(session_dir)])
        assert result.exit_code == 0

        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == ""
