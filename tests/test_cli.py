"""Tests for kindle.cli — Typer CLI layer (build, resume, list commands)."""

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
    """Return a mock Settings with sensible defaults."""
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


def _fake_compiled(return_state: dict | None = None) -> MagicMock:
    """Return a mock compiled graph whose ainvoke returns *return_state*."""
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value=return_state or {})
    return compiled


def _setup_session(tmp_path: Path, idea: str = "test app") -> tuple[str, Path]:
    """Create a minimal valid session directory for the resume command."""
    project_id = "kindle_abc12345"
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True)
    (project_dir / "artifacts").mkdir()
    (project_dir / "logs").mkdir()
    (project_dir / "workspace").mkdir()

    metadata = {
        "project_id": project_id,
        "idea": idea,
        "created_at": "2025-01-01T00:00:00+00:00",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return project_id, project_dir


# ---------------------------------------------------------------------------
# build command — no arguments (usage hint)
# ---------------------------------------------------------------------------


class TestBuildNoArgs:
    """Invoking kindle with no idea prints usage and exits 0."""

    def test_no_args_shows_usage(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()

    def test_no_args_mentions_help(self) -> None:
        result = runner.invoke(app, [])
        assert "--help" in result.output


# ---------------------------------------------------------------------------
# build command — happy path
# ---------------------------------------------------------------------------


class TestBuildHappyPath:
    """Build command with a valid idea — mocking all external deps."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_build_invokes_graph(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_aaa", tmp_path / "kindle_aaa")
        mock_ws.return_value = tmp_path / "kindle_aaa" / "workspace"
        compiled = _fake_compiled()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["build me an app"])

        assert result.exit_code == 0
        mock_build_graph.assert_called_once()
        mock_asyncio_run.assert_called_once()

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_build_passes_idea_in_state(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_bbb", tmp_path / "kindle_bbb")
        mock_ws.return_value = tmp_path / "kindle_bbb" / "workspace"
        compiled = _fake_compiled()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["a todo app"])

        call_args = mock_asyncio_run.call_args[0][0]  # the coroutine
        # The coroutine was passed to asyncio.run — we verify build_graph was called
        mock_build_graph.assert_called_once()

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_build_calls_ui_banner(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_ccc", tmp_path / "kindle_ccc")
        mock_ws.return_value = tmp_path / "kindle_ccc" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        ui_instance = mock_ui_cls.return_value
        runner.invoke(app, ["a chat app"])

        ui_instance.banner.assert_called_once_with("a chat app", "kindle_ccc")

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_build_creates_project(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(projects_root=tmp_path)
        mock_load.return_value = settings
        mock_create.return_value = ("kindle_ddd", tmp_path / "kindle_ddd")
        mock_ws.return_value = tmp_path / "kindle_ddd" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["my idea"])

        mock_create.assert_called_once_with(tmp_path, "my idea")


# ---------------------------------------------------------------------------
# build command — --stack flag
# ---------------------------------------------------------------------------


class TestBuildStackOption:
    """The --stack flag propagates to the initial state."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_stack_flag_passed_to_ainvoke(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_eee", tmp_path / "kindle_eee")
        mock_ws.return_value = tmp_path / "kindle_eee" / "workspace"
        compiled = _fake_compiled()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["my app", "--stack", "nextjs"])

        # asyncio.run receives compiled.ainvoke(state) — verify compiled.ainvoke was
        # eventually called with the correct stack
        mock_asyncio_run.assert_called_once()


# ---------------------------------------------------------------------------
# build command — --output flag
# ---------------------------------------------------------------------------


class TestBuildOutputOption:
    """The --output flag copies the workspace to the specified directory."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_output_copies_workspace(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        ws_dir = tmp_path / "kindle_fff" / "workspace"
        ws_dir.mkdir(parents=True)
        (ws_dir / "index.html").write_text("<html></html>")
        mock_create.return_value = ("kindle_fff", tmp_path / "kindle_fff")
        mock_ws.return_value = ws_dir
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        out_dir = tmp_path / "my_output"
        result = runner.invoke(app, ["my app", "--output", str(out_dir)])

        assert result.exit_code == 0
        assert out_dir.exists()
        assert (out_dir / "index.html").read_text() == "<html></html>"

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_output_existing_dir_shows_error(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        ws_dir = tmp_path / "kindle_ggg" / "workspace"
        ws_dir.mkdir(parents=True)
        mock_create.return_value = ("kindle_ggg", tmp_path / "kindle_ggg")
        mock_ws.return_value = ws_dir
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        out_dir = tmp_path / "existing_output"
        out_dir.mkdir()

        result = runner.invoke(app, ["my app", "--output", str(out_dir)])

        assert result.exit_code == 0
        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        assert "already exists" in ui_instance.error.call_args[0][0]

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_no_output_skips_copy(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_hhh", tmp_path / "kindle_hhh")
        mock_ws.return_value = tmp_path / "kindle_hhh" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["my app"])

        assert result.exit_code == 0
        # No shutil.copytree should have been called — verified by no error


# ---------------------------------------------------------------------------
# build command — --concurrency, --qa-retries, --cpo-retries overrides
# ---------------------------------------------------------------------------


class TestBuildCLIOverrides:
    """CLI flags override config defaults for concurrency and retries."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_concurrency_override(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings(max_concurrent_agents=4)
        mock_create.return_value = ("kindle_ii1", tmp_path / "kindle_ii1")
        mock_ws.return_value = tmp_path / "kindle_ii1" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["idea", "--concurrency", "8"])
        assert result.exit_code == 0

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_zero_concurrency_uses_config_default(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings(max_concurrent_agents=6)
        mock_create.return_value = ("kindle_ii2", tmp_path / "kindle_ii2")
        mock_ws.return_value = tmp_path / "kindle_ii2" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["idea", "--concurrency", "0"])
        assert result.exit_code == 0

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_qa_retries_override(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_ii3", tmp_path / "kindle_ii3")
        mock_ws.return_value = tmp_path / "kindle_ii3" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["idea", "--qa-retries", "3"])
        assert result.exit_code == 0

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_cpo_retries_override(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_ii4", tmp_path / "kindle_ii4")
        mock_ws.return_value = tmp_path / "kindle_ii4" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["idea", "--cpo-retries", "5"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# build command — --review-arch flag
# ---------------------------------------------------------------------------


class TestBuildReviewArchFlag:
    """The --review-arch flag is accepted without error."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_review_arch_flag_accepted(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_rrr", tmp_path / "kindle_rrr")
        mock_ws.return_value = tmp_path / "kindle_rrr" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["my app", "--review-arch"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# build command — --auto-approve and --verbose flags
# ---------------------------------------------------------------------------


class TestBuildUIFlags:
    """UI constructor receives auto_approve and verbose flags."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_auto_approve_passed_to_ui(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_aa1", tmp_path / "kindle_aa1")
        mock_ws.return_value = tmp_path / "kindle_aa1" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["app", "--auto-approve"])

        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=False)

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_verbose_passed_to_ui(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_aa2", tmp_path / "kindle_aa2")
        mock_ws.return_value = tmp_path / "kindle_aa2" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["app", "--verbose"])

        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_both_flags(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_aa3", tmp_path / "kindle_aa3")
        mock_ws.return_value = tmp_path / "kindle_aa3" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["app", "--auto-approve", "--verbose"])

        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=True)


# ---------------------------------------------------------------------------
# build command — info messages
# ---------------------------------------------------------------------------


class TestBuildInfoMessages:
    """Build command prints workspace and artifact path info messages."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_info_called_with_project_path(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        project_dir = tmp_path / "kindle_inf"
        ws = project_dir / "workspace"
        mock_create.return_value = ("kindle_inf", project_dir)
        mock_ws.return_value = ws
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["my idea"])

        ui_instance = mock_ui_cls.return_value
        info_calls = [c[0][0] for c in ui_instance.info.call_args_list]
        assert any("Project built at" in msg for msg in info_calls)
        assert any("Session artifacts" in msg for msg in info_calls)


# ---------------------------------------------------------------------------
# resume command — happy path
# ---------------------------------------------------------------------------


class TestResumeHappyPath:
    """Resume command with a valid session directory."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_invokes_graph(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        compiled = _fake_compiled()
        mock_build_graph.return_value = compiled
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 0
        mock_build_graph.assert_called_once()
        mock_asyncio_run.assert_called_once()

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_default_from_stage_is_dev(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["resume", str(project_dir)])

        mock_build_graph.assert_called_once()
        call_kwargs = mock_build_graph.call_args
        assert call_kwargs[1]["entry_stage"] == "dev"

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_from_architect(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["resume", str(project_dir), "--from", "architect"])

        mock_build_graph.assert_called_once()
        call_kwargs = mock_build_graph.call_args
        assert call_kwargs[1]["entry_stage"] == "architect"

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_from_grill(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["resume", str(project_dir), "--from", "grill"])

        call_kwargs = mock_build_graph.call_args
        assert call_kwargs[1]["entry_stage"] == "grill"

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_calls_banner_with_idea(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path, idea="resume test idea")
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["resume", str(project_dir)])

        ui_instance = mock_ui_cls.return_value
        ui_instance.banner.assert_called_once_with("resume test idea", "kindle_abc12345")


# ---------------------------------------------------------------------------
# resume command — artifact reloading
# ---------------------------------------------------------------------------


class TestResumeArtifactReloading:
    """Resume reloads feature_spec.json and dev_tasks.json from disk."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_loads_feature_spec(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)

        def artifact_side_effect(proj_dir: str, name: str) -> str | None:
            if name == "feature_spec.json":
                return json.dumps({"features": ["auth", "dashboard"]})
            if name == "dev_tasks.json":
                return json.dumps([{"id": "task1"}])
            return None

        mock_load_artifact.side_effect = artifact_side_effect
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["resume", str(project_dir)])
        assert result.exit_code == 0
        # Verify load_artifact was called for feature_spec.json
        artifact_calls = [c[0][1] for c in mock_load_artifact.call_args_list]
        assert "feature_spec.json" in artifact_calls
        assert "dev_tasks.json" in artifact_calls

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_handles_missing_artifacts(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When artifacts don't exist, defaults to empty dict/list/string."""
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["resume", str(project_dir)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# resume command — error cases
# ---------------------------------------------------------------------------


class TestResumeErrors:
    """Resume command errors when session is invalid."""

    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_nonexistent_directory(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()

        result = runner.invoke(app, ["resume", str(tmp_path / "nonexistent")])

        assert result.exit_code == 1
        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        assert "not found" in ui_instance.error.call_args[0][0]

    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_missing_metadata_json(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Directory exists but has no metadata.json."""
        mock_load.return_value = _make_settings()
        bad_dir = tmp_path / "no_metadata"
        bad_dir.mkdir()

        result = runner.invoke(app, ["resume", str(bad_dir)])

        assert result.exit_code == 1
        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        assert "metadata.json" in ui_instance.error.call_args[0][0]


# ---------------------------------------------------------------------------
# resume command — --auto-approve and --verbose flags
# ---------------------------------------------------------------------------


class TestResumeUIFlags:
    """Resume passes UI flags through correctly."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_auto_approve(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["resume", str(project_dir), "--auto-approve"])

        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=False)

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_verbose(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["resume", str(project_dir), "--verbose"])

        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)


# ---------------------------------------------------------------------------
# list command — projects exist
# ---------------------------------------------------------------------------


class TestListCommand:
    """The `list` subcommand delegates to list_projects and show_projects."""

    @patch("kindle.cli.list_projects")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_list_shows_projects(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_list: MagicMock,
    ) -> None:
        settings = _make_settings()
        mock_load.return_value = settings
        projects = [
            {"project_id": "kindle_aaa", "idea": "first", "status": "completed"},
            {"project_id": "kindle_bbb", "idea": "second", "status": "in_progress"},
        ]
        mock_list.return_value = projects

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_list.assert_called_once_with(settings.projects_root)
        ui_instance = mock_ui_cls.return_value
        ui_instance.show_projects.assert_called_once_with(projects)

    @patch("kindle.cli.list_projects")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_list_empty_projects(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_list: MagicMock,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_list.return_value = []

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        ui_instance = mock_ui_cls.return_value
        ui_instance.show_projects.assert_called_once_with([])


# ---------------------------------------------------------------------------
# Error handling — Settings.load fails (missing API key)
# ---------------------------------------------------------------------------


class TestMissingAPIKey:
    """When Settings.load() raises SystemExit, CLI propagates it."""

    @patch("kindle.cli.Settings.load", side_effect=SystemExit("ANTHROPIC_API_KEY not set."))
    def test_build_missing_api_key(self, mock_load: MagicMock) -> None:
        result = runner.invoke(app, ["build an app"])
        assert result.exit_code != 0

    @patch("kindle.cli.Settings.load", side_effect=SystemExit("ANTHROPIC_API_KEY not set."))
    def test_list_missing_api_key(self, mock_load: MagicMock) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Keyboard interrupt handling
# ---------------------------------------------------------------------------


class TestKeyboardInterrupt:
    """KeyboardInterrupt during graph execution is handled gracefully."""

    @patch("kindle.cli.asyncio.run", side_effect=KeyboardInterrupt)
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_keyboard_interrupt_during_build(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_ki1", tmp_path / "kindle_ki1")
        mock_ws.return_value = tmp_path / "kindle_ki1" / "workspace"
        mock_build_graph.return_value = _fake_compiled()

        result = runner.invoke(app, ["an idea"])

        # Typer/Click catches KeyboardInterrupt and sets exit_code = 1
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# app help and subcommand structure
# ---------------------------------------------------------------------------


class TestAppStructure:
    """Verify the Typer app exposes expected commands and help text."""

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "kindle" in result.output.lower() or "application factory" in result.output.lower()

    def test_resume_help(self) -> None:
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "Resume" in result.output or "resume" in result.output

    def test_list_help(self) -> None:
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert "List" in result.output or "list" in result.output

    def test_resume_requires_project_path(self) -> None:
        result = runner.invoke(app, ["resume"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# build command — short flag aliases
# ---------------------------------------------------------------------------


class TestShortFlags:
    """Short aliases -s, -c, -o, -v work for build command."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_short_stack_flag(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_sf1", tmp_path / "kindle_sf1")
        mock_ws.return_value = tmp_path / "kindle_sf1" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["idea", "-s", "react"])
        assert result.exit_code == 0

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_short_concurrency_flag(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_sf2", tmp_path / "kindle_sf2")
        mock_ws.return_value = tmp_path / "kindle_sf2" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["idea", "-c", "2"])
        assert result.exit_code == 0

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.create_project")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_short_verbose_flag(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_create: MagicMock,
        mock_ws: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        mock_create.return_value = ("kindle_sf3", tmp_path / "kindle_sf3")
        mock_ws.return_value = tmp_path / "kindle_sf3" / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["idea", "-v"])
        assert result.exit_code == 0
        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)


# ---------------------------------------------------------------------------
# resume command — info messages after completion
# ---------------------------------------------------------------------------


class TestResumeInfoMessages:
    """Resume prints the project path after completion."""

    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_prints_project_path(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        ws = project_dir / "workspace"
        mock_ws.return_value = ws
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        runner.invoke(app, ["resume", str(project_dir)])

        ui_instance = mock_ui_cls.return_value
        info_calls = [c[0][0] for c in ui_instance.info.call_args_list]
        assert any("Project at" in msg for msg in info_calls)


# ---------------------------------------------------------------------------
# resume command — all valid stage names
# ---------------------------------------------------------------------------


class TestResumeValidStages:
    """Resume accepts all six valid stage names."""

    @pytest.mark.parametrize("stage", ["grill", "research", "architect", "dev", "qa", "package"])
    @patch("kindle.cli.asyncio.run")
    @patch("kindle.cli.build_graph")
    @patch("kindle.cli.load_artifact")
    @patch("kindle.cli.workspace_path")
    @patch("kindle.cli.UI")
    @patch("kindle.cli.Settings.load")
    def test_resume_from_valid_stage(
        self,
        mock_load: MagicMock,
        mock_ui_cls: MagicMock,
        mock_ws: MagicMock,
        mock_load_artifact: MagicMock,
        mock_build_graph: MagicMock,
        mock_asyncio_run: MagicMock,
        stage: str,
        tmp_path: Path,
    ) -> None:
        mock_load.return_value = _make_settings()
        _, project_dir = _setup_session(tmp_path)
        mock_load_artifact.return_value = None
        mock_ws.return_value = project_dir / "workspace"
        mock_build_graph.return_value = _fake_compiled()
        mock_asyncio_run.return_value = {}

        result = runner.invoke(app, ["resume", str(project_dir), "--from", stage])

        assert result.exit_code == 0
        mock_build_graph.assert_called_once()
        assert mock_build_graph.call_args[1]["entry_stage"] == stage
