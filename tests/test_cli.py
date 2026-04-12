"""Tests for kindle.cli — Typer CLI entry point, command routing, and argument parsing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import typer
from typer.testing import CliRunner

from kindle.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> MagicMock:
    """Return a mock Settings object with sensible defaults."""
    defaults = {
        "anthropic_api_key": "sk-test-key",
        "model": "claude-opus-4-20250514",
        "max_agent_turns": 50,
        "max_concurrent_agents": 4,
        "max_qa_retries": 10,
        "max_cpo_retries": 10,
        "projects_root": Path("/tmp/kindle-test/projects"),
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _patch_build_stack(settings: MagicMock | None = None):
    """Return a dict of mock patches common to the build command."""
    if settings is None:
        settings = _make_settings()
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value={})
    return {
        "settings": settings,
        "compiled": compiled,
    }


# ---------------------------------------------------------------------------
# build command (default / callback)
# ---------------------------------------------------------------------------


class TestBuildCommand:
    """Tests for the default build command invoked as `kindle 'some idea'`."""

    def test_no_idea_shows_usage_and_exits_cleanly(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output or "kindle" in result.output.lower()

    def test_settings_load_is_called(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_abc123", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            result = runner.invoke(app, ["Build a todo app"])
        assert result.exit_code == 0

    def test_create_project_receives_settings_root_and_idea(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_abc", Path("/tmp/p"))) as mock_create,
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["Build a chat app"])
        mock_create.assert_called_once_with(mocks["settings"].projects_root, "Build a chat app")

    def test_initial_state_built_from_cli_args_and_settings(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_abc", Path("/tmp/proj"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}) as mock_arun,
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/proj/workspace")),
        ):
            runner.invoke(app, ["My idea", "--stack", "react", "--auto-approve"])

        # asyncio.run receives compiled.ainvoke(state) — grab the state dict
        mock_arun.assert_called_once()
        # The argument to asyncio.run is compiled.ainvoke(initial_state)
        # We verify build_graph was called and the compiled graph was used
        mocks["compiled"].ainvoke.assert_called_once()
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["idea"] == "My idea"
        assert state["stack_preference"] == "react"
        assert state["auto_approve"] is True
        assert state["project_id"] == "kindle_abc"
        assert state["project_dir"] == str(Path("/tmp/proj"))
        assert state["feature_spec"] == {}
        assert state["grill_transcript"] == ""

    def test_build_graph_called_without_entry_stage(self) -> None:
        """Default build invokes build_graph with no entry_stage (defaults to 'grill')."""
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.create_project", return_value=("kindle_abc", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]) as mock_bg,
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["An idea"])
        # build_graph receives ui instance only (no entry_stage keyword)
        mock_bg.assert_called_once_with(mock_ui_cls.return_value)

    def test_cli_concurrency_overrides_settings(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea", "--concurrency", "8"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 8

    def test_zero_concurrency_falls_back_to_settings(self) -> None:
        settings = _make_settings(max_concurrent_agents=6)
        mocks = _patch_build_stack(settings=settings)
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 6

    def test_qa_retries_override(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea", "--qa-retries", "5"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 5

    def test_cpo_retries_override(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea", "--cpo-retries", "3"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_cpo_retries"] == 3

    def test_zero_retries_fall_back_to_settings(self) -> None:
        settings = _make_settings(max_qa_retries=7, max_cpo_retries=9)
        mocks = _patch_build_stack(settings=settings)
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 7
        assert state["max_cpo_retries"] == 9

    def test_model_and_max_turns_from_settings(self) -> None:
        settings = _make_settings(model="claude-sonnet-4-20250514", max_agent_turns=30)
        mocks = _patch_build_stack(settings=settings)
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["model"] == "claude-sonnet-4-20250514"
        assert state["max_agent_turns"] == 30

    def test_ui_banner_called_with_idea_and_project_id(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.create_project", return_value=("kindle_xyz", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["My cool app"])
        mock_ui_cls.return_value.banner.assert_called_once_with("My cool app", "kindle_xyz")

    def test_ui_created_with_auto_approve_and_verbose(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea", "--auto-approve", "--verbose"])
        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=True)

    def test_workspace_path_info_printed(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/proj"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/proj/workspace")),
        ):
            runner.invoke(app, ["idea"])
        ui = mock_ui_cls.return_value
        info_messages = [c.args[0] for c in ui.info.call_args_list]
        assert any("/tmp/proj/workspace" in m for m in info_messages)
        assert any("/tmp/proj" in m for m in info_messages)

    def test_initial_state_has_all_required_keys(self) -> None:
        """Verify that KindleState receives every expected key with correct defaults."""
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_abc", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["Build something"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        expected_keys = {
            "idea", "project_id", "project_dir", "stack_preference",
            "auto_approve", "max_concurrent_agents", "max_qa_retries",
            "max_cpo_retries", "feature_spec", "grill_transcript",
            "research_report", "architecture", "dev_tasks", "qa_report",
            "product_audit", "package_readme", "qa_passed", "cpo_passed",
            "qa_retries", "cpo_retries", "model", "max_agent_turns",
            "current_stage",
        }
        assert set(state.keys()) == expected_keys
        # Verify blank/default artifact values
        assert state["dev_tasks"] == []
        assert state["qa_passed"] is False
        assert state["cpo_passed"] is False
        assert state["qa_retries"] == 0
        assert state["cpo_retries"] == 0
        assert state["current_stage"] == ""
        assert state["package_readme"] == ""


# ---------------------------------------------------------------------------
# --output flag (copy workspace to custom directory)
# ---------------------------------------------------------------------------


class TestBuildOutputCopy:
    """Tests for the --output flag that copies the workspace to a user directory."""

    def test_copytree_called_when_output_provided(self, tmp_path: Path) -> None:
        mocks = _patch_build_stack()
        output_dir = tmp_path / "my_output"
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
            patch("shutil.copytree") as mock_copy,
        ):
            runner.invoke(app, ["idea", "--output", str(output_dir)])
        mock_copy.assert_called_once_with(str(Path("/tmp/p/workspace")), str(output_dir.resolve()))

    def test_no_copytree_when_output_not_provided(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
            patch("shutil.copytree") as mock_copy,
        ):
            runner.invoke(app, ["idea"])
        mock_copy.assert_not_called()

    def test_output_dir_already_exists_shows_error(self, tmp_path: Path) -> None:
        mocks = _patch_build_stack()
        existing = tmp_path / "already_here"
        existing.mkdir()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
            patch("shutil.copytree") as mock_copy,
        ):
            runner.invoke(app, ["idea", "--output", str(existing)])
        mock_copy.assert_not_called()
        mock_ui_cls.return_value.error.assert_called_once()
        error_msg = mock_ui_cls.return_value.error.call_args[0][0]
        assert "already exists" in error_msg


# ---------------------------------------------------------------------------
# build command — error handling
# ---------------------------------------------------------------------------


class TestBuildErrorHandling:
    """Tests for graceful error handling during build."""

    def test_settings_load_failure_exits(self) -> None:
        """When Settings.load() raises SystemExit, CLI propagates it."""
        with patch(
            "kindle.cli.Settings.load",
            side_effect=SystemExit("ANTHROPIC_API_KEY not set"),
        ):
            result = runner.invoke(app, ["Build an app"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# resume command
# ---------------------------------------------------------------------------


class TestResumeCommand:
    """Tests for `kindle resume <project_path> [--from stage]`."""

    def _setup_session(self, tmp_path: Path) -> Path:
        """Create a minimal session directory with metadata and artifacts."""
        session = tmp_path / "kindle_abc12345"
        session.mkdir()
        artifacts = session / "artifacts"
        artifacts.mkdir()

        meta = {"project_id": "kindle_abc12345", "idea": "A chat app"}
        (session / "metadata.json").write_text(json.dumps(meta))

        # Write some artifacts
        (artifacts / "feature_spec.json").write_text(json.dumps({"feature": "chat"}))
        (artifacts / "grill_transcript.md").write_text("Q&A transcript")
        (artifacts / "research_report.md").write_text("Research findings")
        (artifacts / "architecture.md").write_text("System design")
        (artifacts / "dev_tasks.json").write_text(json.dumps([{"task": "build"}]))
        (artifacts / "qa_report.md").write_text("QA passed")
        (artifacts / "product_audit.md").write_text("Audit OK")

        return session

    def test_resume_loads_metadata_and_artifacts(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            result = runner.invoke(app, ["resume", str(session)])
        assert result.exit_code == 0

        compiled.ainvoke.assert_called_once()
        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == "A chat app"
        assert state["project_id"] == "kindle_abc12345"
        assert state["feature_spec"] == {"feature": "chat"}
        assert state["grill_transcript"] == "Q&A transcript"
        assert state["research_report"] == "Research findings"
        assert state["architecture"] == "System design"
        assert state["dev_tasks"] == [{"task": "build"}]
        assert state["qa_report"] == "QA passed"
        assert state["product_audit"] == "Audit OK"

    def test_resume_build_graph_called_with_entry_stage(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.build_graph", return_value=compiled) as mock_bg,
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session), "--from", "qa"])
        mock_bg.assert_called_once_with(mock_ui_cls.return_value, entry_stage="qa")

    def test_resume_default_entry_stage_is_dev(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.build_graph", return_value=compiled) as mock_bg,
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session)])
        mock_bg.assert_called_once_with(mock_ui_cls.return_value, entry_stage="dev")

    def test_resume_nonexistent_directory_exits_with_error(self, tmp_path: Path) -> None:
        fake_path = tmp_path / "does_not_exist"
        settings = _make_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
        ):
            result = runner.invoke(app, ["resume", str(fake_path)])
        assert result.exit_code == 1
        mock_ui_cls.return_value.error.assert_called_once()
        error_msg = mock_ui_cls.return_value.error.call_args[0][0]
        assert "not found" in error_msg

    def test_resume_missing_metadata_exits_with_error(self, tmp_path: Path) -> None:
        session = tmp_path / "kindle_no_meta"
        session.mkdir()
        # No metadata.json created
        settings = _make_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
        ):
            result = runner.invoke(app, ["resume", str(session)])
        assert result.exit_code == 1
        error_msg = mock_ui_cls.return_value.error.call_args[0][0]
        assert "metadata.json" in error_msg

    def test_resume_missing_artifacts_treated_as_empty(self, tmp_path: Path) -> None:
        """When artifact files don't exist, state fields default to empty."""
        session = tmp_path / "kindle_sparse"
        session.mkdir()
        (session / "artifacts").mkdir()
        meta = {"project_id": "kindle_sparse", "idea": "Minimal"}
        (session / "metadata.json").write_text(json.dumps(meta))

        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            result = runner.invoke(app, ["resume", str(session)])
        assert result.exit_code == 0
        state = compiled.ainvoke.call_args[0][0]
        assert state["feature_spec"] == {}
        assert state["dev_tasks"] == []
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""
        assert state["qa_report"] == ""
        assert state["product_audit"] == ""

    def test_resume_ui_banner_and_stage_info(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session), "--from", "architect"])
        ui = mock_ui_cls.return_value
        ui.banner.assert_called_once_with("A chat app", "kindle_abc12345")
        info_messages = [c.args[0] for c in ui.info.call_args_list]
        assert any("architect" in m for m in info_messages)

    def test_resume_settings_values_propagated_to_state(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        settings = _make_settings(
            max_concurrent_agents=12,
            max_qa_retries=20,
            max_cpo_retries=15,
            model="claude-haiku-2025",
            max_agent_turns=100,
        )
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session)])
        state = compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 12
        assert state["max_qa_retries"] == 20
        assert state["max_cpo_retries"] == 15
        assert state["model"] == "claude-haiku-2025"
        assert state["max_agent_turns"] == 100

    def test_resume_auto_approve_and_verbose_forwarded(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session), "--auto-approve", "--verbose"])
        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=True)
        state = compiled.ainvoke.call_args[0][0]
        assert state["auto_approve"] is True


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListCommand:
    """Tests for `kindle list` — displays all build sessions."""

    def test_list_calls_list_projects_with_settings_root(self) -> None:
        settings = _make_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.list_projects", return_value=[]) as mock_lp,
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        mock_lp.assert_called_once_with(settings.projects_root)

    def test_list_passes_projects_to_ui_show_projects(self) -> None:
        settings = _make_settings()
        fake_projects = [
            {"project_id": "kindle_aaa", "idea": "App A"},
            {"project_id": "kindle_bbb", "idea": "App B"},
        ]
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.list_projects", return_value=fake_projects),
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        mock_ui_cls.return_value.show_projects.assert_called_once_with(fake_projects)

    def test_list_with_empty_projects(self) -> None:
        settings = _make_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        mock_ui_cls.return_value.show_projects.assert_called_once_with([])

    def test_list_creates_ui_with_defaults(self) -> None:
        settings = _make_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            runner.invoke(app, ["list"])
        mock_ui_cls.assert_called_once_with()

    def test_list_settings_load_failure_exits(self) -> None:
        with patch(
            "kindle.cli.Settings.load",
            side_effect=SystemExit("ANTHROPIC_API_KEY not set"),
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI argument parsing edge cases
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    """Edge cases in argument parsing and option handling."""

    def test_short_stack_flag(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea", "-s", "nextjs"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["stack_preference"] == "nextjs"

    def test_short_concurrency_flag(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea", "-c", "16"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 16

    def test_short_output_flag(self, tmp_path: Path) -> None:
        mocks = _patch_build_stack()
        output_dir = tmp_path / "out"
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
            patch("shutil.copytree") as mock_copy,
        ):
            runner.invoke(app, ["idea", "-o", str(output_dir)])
        mock_copy.assert_called_once()

    def test_short_verbose_flag(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea", "-v"])
        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)

    def test_help_flag_shows_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "kindle" in result.output.lower() or "application factory" in result.output.lower()

    def test_resume_help_flag(self) -> None:
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "resume" in result.output.lower()

    def test_list_help_flag(self) -> None:
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower()

    def test_empty_stack_defaults_to_empty_string(self) -> None:
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            runner.invoke(app, ["idea"])
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["stack_preference"] == ""

    def test_review_arch_flag_parsed(self) -> None:
        """--review-arch is accepted by the CLI without error."""
        mocks = _patch_build_stack()
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", Path("/tmp/p"))),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=Path("/tmp/p/workspace")),
        ):
            result = runner.invoke(app, ["idea", "--review-arch"])
        assert result.exit_code == 0
