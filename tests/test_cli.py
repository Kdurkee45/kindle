"""Tests for kindle.cli — Typer CLI commands: build, resume, list."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import typer
from typer.testing import CliRunner

from kindle.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings(**overrides):
    """Return a mock Settings object with sensible defaults."""
    defaults = {
        "anthropic_api_key": "sk-test-key",
        "model": "claude-opus-4-20250514",
        "max_agent_turns": 50,
        "max_concurrent_agents": 4,
        "max_qa_retries": 10,
        "max_cpo_retries": 10,
        "projects_root": Path("/tmp/kindle-test-projects"),
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _setup_build_mocks(
    tmp_path: Path,
    idea: str = "Build a todo app",
    project_id: str = "kindle_abc12345",
):
    """Return a dict of common patches for the build command."""
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True)
    ws = project_dir / "workspace"
    ws.mkdir()

    settings = _fake_settings(projects_root=tmp_path)
    mock_compiled = MagicMock()
    # ainvoke is a coroutine; asyncio.run will be mocked, so we don't need AsyncMock
    mock_compiled.ainvoke.return_value = {"current_stage": "package"}

    return {
        "settings": settings,
        "project_id": project_id,
        "project_dir": project_dir,
        "workspace": ws,
        "mock_compiled": mock_compiled,
    }


# ---------------------------------------------------------------------------
# build command — no idea provided
# ---------------------------------------------------------------------------


class TestBuildNoIdea:
    """When no idea argument is supplied, the CLI should print usage and exit."""

    @patch("kindle.cli.Settings.load", return_value=_fake_settings())
    def test_no_idea_shows_usage(self, _mock_settings: MagicMock) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output or "kindle" in result.output.lower()

    @patch("kindle.cli.Settings.load", return_value=_fake_settings())
    def test_no_idea_does_not_create_project(self, _mock_settings: MagicMock) -> None:
        with patch("kindle.cli.create_project") as mock_create:
            result = runner.invoke(app, [])
            mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# build command — happy path
# ---------------------------------------------------------------------------


class TestBuildHappyPath:
    """The build command should wire up Settings, create a project, build the
    graph, invoke it, and report the output path."""

    def test_build_creates_project_and_runs_graph(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={"current_stage": "package"}) as mock_arun,
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["Build a todo app"])

        assert result.exit_code == 0
        mock_arun.assert_called_once()

    def test_build_passes_idea_to_create_project(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])) as mock_create,
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["Build a todo app"])

        mock_create.assert_called_once_with(ctx["settings"].projects_root, "Build a todo app")

    def test_build_calls_build_graph_with_ui(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        mock_ui_instance = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI", return_value=mock_ui_instance),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]) as mock_bg,
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["Build a todo app"])

        mock_bg.assert_called_once_with(mock_ui_instance)

    def test_build_calls_ui_banner(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        mock_ui_instance = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI", return_value=mock_ui_instance),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["Build a todo app"])

        mock_ui_instance.banner.assert_called_once_with("Build a todo app", ctx["project_id"])

    def test_build_reports_project_paths(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        mock_ui_instance = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI", return_value=mock_ui_instance),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["Build a todo app"])

        info_calls = [c.args[0] for c in mock_ui_instance.info.call_args_list]
        assert any("Project built at:" in c for c in info_calls)
        assert any("Session artifacts:" in c for c in info_calls)


# ---------------------------------------------------------------------------
# build command — initial state construction
# ---------------------------------------------------------------------------


class TestBuildInitialState:
    """Verify the initial_state dict passed to graph.ainvoke has correct fields."""

    def _capture_state(self, tmp_path: Path, cli_args: list[str]) -> dict:
        """Run the build command and return the KindleState passed to ainvoke."""
        ctx = _setup_build_mocks(tmp_path)
        captured = {}

        def fake_run(coro):
            # The coroutine is compiled.ainvoke(initial_state); we can't inspect
            # the coroutine args directly. Instead capture from asyncio.run's arg.
            return {}

        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]) as mock_bg,
            patch("kindle.cli.asyncio.run", side_effect=fake_run),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            runner.invoke(app, cli_args)

        # The compiled graph's ainvoke was called with initial_state
        if ctx["mock_compiled"].ainvoke.call_args:
            captured = ctx["mock_compiled"].ainvoke.call_args.args[0]
        return captured

    def test_state_contains_idea(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["My cool app"])
        assert state["idea"] == "My cool app"

    def test_state_has_empty_stack_by_default(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["stack_preference"] == ""

    def test_state_uses_stack_from_cli(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app", "--stack", "react"])
        assert state["stack_preference"] == "react"

    def test_state_auto_approve_default_false(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["auto_approve"] is False

    def test_state_auto_approve_from_cli(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app", "--auto-approve"])
        assert state["auto_approve"] is True

    def test_state_uses_settings_concurrency_when_cli_zero(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        # Default concurrency CLI arg is 0 → should use settings value (4)
        assert state["max_concurrent_agents"] == 4

    def test_state_uses_cli_concurrency_when_nonzero(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app", "--concurrency", "8"])
        assert state["max_concurrent_agents"] == 8

    def test_state_uses_settings_qa_retries_when_cli_zero(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["max_qa_retries"] == 10

    def test_state_uses_cli_qa_retries_when_nonzero(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app", "--qa-retries", "5"])
        assert state["max_qa_retries"] == 5

    def test_state_uses_settings_cpo_retries_when_cli_zero(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["max_cpo_retries"] == 10

    def test_state_uses_cli_cpo_retries_when_nonzero(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app", "--cpo-retries", "3"])
        assert state["max_cpo_retries"] == 3

    def test_state_model_from_settings(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["model"] == "claude-opus-4-20250514"

    def test_state_max_agent_turns_from_settings(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["max_agent_turns"] == 50

    def test_state_empty_artifacts_on_fresh_build(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["feature_spec"] == {}
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""
        assert state["dev_tasks"] == []
        assert state["qa_report"] == ""
        assert state["product_audit"] == ""
        assert state["package_readme"] == ""

    def test_state_quality_flags_initially_false(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["qa_passed"] is False
        assert state["cpo_passed"] is False
        assert state["qa_retries"] == 0
        assert state["cpo_retries"] == 0

    def test_state_current_stage_initially_empty(self, tmp_path: Path) -> None:
        state = self._capture_state(tmp_path, ["An app"])
        assert state["current_stage"] == ""


# ---------------------------------------------------------------------------
# build command — output directory handling
# ---------------------------------------------------------------------------


class TestBuildOutputDir:
    """Tests for the --output flag that copies the workspace to a target dir."""

    def test_output_copies_workspace_to_target(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        # Create a file in workspace so copytree has something to copy
        (ctx["workspace"] / "index.html").write_text("<h1>Hello</h1>")

        output_dir = tmp_path / "my-output"

        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["An app", "--output", str(output_dir)])

        assert result.exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "index.html").read_text() == "<h1>Hello</h1>"

    def test_output_errors_when_dir_already_exists(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)

        output_dir = tmp_path / "existing-output"
        output_dir.mkdir()

        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["An app", "--output", str(output_dir)])

        mock_ui.error.assert_called_once()
        error_msg = mock_ui.error.call_args.args[0]
        assert "already exists" in error_msg

    def test_no_output_flag_does_not_copy(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
            patch("kindle.cli.shutil", create=True) as mock_shutil,
        ):
            result = runner.invoke(app, ["An app"])

        # shutil should never be imported/called when no --output
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# build command — UI construction flags
# ---------------------------------------------------------------------------


class TestBuildUIFlags:
    """Verify that --auto-approve and --verbose are forwarded to the UI."""

    def test_auto_approve_forwarded_to_ui(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            runner.invoke(app, ["An app", "--auto-approve"])

        MockUI.assert_called_once_with(auto_approve=True, verbose=False)

    def test_verbose_forwarded_to_ui(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            runner.invoke(app, ["An app", "--verbose"])

        MockUI.assert_called_once_with(auto_approve=False, verbose=True)

    def test_both_flags_forwarded(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            runner.invoke(app, ["An app", "--auto-approve", "--verbose"])

        MockUI.assert_called_once_with(auto_approve=True, verbose=True)

    def test_defaults_no_flags(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            runner.invoke(app, ["An app"])

        MockUI.assert_called_once_with(auto_approve=False, verbose=False)


# ---------------------------------------------------------------------------
# resume command — error cases
# ---------------------------------------------------------------------------


class TestResumeErrors:
    """The resume command should fail gracefully for missing paths and metadata."""

    @patch("kindle.cli.Settings.load", return_value=_fake_settings())
    @patch("kindle.cli.UI")
    def test_nonexistent_project_dir_exits_1(self, MockUI, _mock_settings) -> None:
        mock_ui = MagicMock()
        MockUI.return_value = mock_ui
        result = runner.invoke(app, ["resume", "/no/such/path"])
        assert result.exit_code == 1
        mock_ui.error.assert_called_once()
        assert "not found" in mock_ui.error.call_args.args[0]

    @patch("kindle.cli.Settings.load", return_value=_fake_settings())
    @patch("kindle.cli.UI")
    def test_missing_metadata_json_exits_1(self, MockUI, _mock_settings, tmp_path: Path) -> None:
        # Directory exists, but no metadata.json
        session_dir = tmp_path / "kindle_abc"
        session_dir.mkdir()

        mock_ui = MagicMock()
        MockUI.return_value = mock_ui

        result = runner.invoke(app, ["resume", str(session_dir)])
        assert result.exit_code == 1
        mock_ui.error.assert_called_once()
        assert "metadata.json" in mock_ui.error.call_args.args[0]


# ---------------------------------------------------------------------------
# resume command — happy path
# ---------------------------------------------------------------------------


class TestResumeHappyPath:
    """The resume command should reload artifacts, rebuild graph from entry stage."""

    def _setup_session(self, tmp_path: Path) -> Path:
        """Create a minimal session directory with metadata and artifacts."""
        session = tmp_path / "kindle_abc12345"
        (session / "artifacts").mkdir(parents=True)
        (session / "logs").mkdir(parents=True)
        (session / "workspace").mkdir()

        meta = {
            "project_id": "kindle_abc12345",
            "idea": "A todo app",
            "created_at": "2025-01-01T00:00:00",
            "status": "in_progress",
            "stages_completed": ["grill", "research", "architect"],
        }
        (session / "metadata.json").write_text(json.dumps(meta))

        # Write some artifacts
        (session / "artifacts" / "feature_spec.json").write_text(json.dumps({"features": ["auth"]}))
        (session / "artifacts" / "grill_transcript.md").write_text("Q: What auth?\nA: OAuth")
        (session / "artifacts" / "research_report.md").write_text("Use React")
        (session / "artifacts" / "architecture.md").write_text("## Architecture\nMicroservices")
        (session / "artifacts" / "dev_tasks.json").write_text(json.dumps([{"id": "T1"}]))

        return session

    def test_resume_loads_metadata_and_runs_graph(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.build_graph", return_value=mock_compiled) as mock_bg,
            patch("kindle.cli.asyncio.run", return_value={}) as mock_arun,
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            result = runner.invoke(app, ["resume", str(session)])

        assert result.exit_code == 0
        mock_bg.assert_called_once()
        mock_arun.assert_called_once()

    def test_resume_builds_graph_with_from_stage(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=mock_compiled) as mock_bg,
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            result = runner.invoke(app, ["resume", str(session), "--from", "qa"])

        mock_bg.assert_called_once()
        _, kwargs = mock_bg.call_args
        assert kwargs["entry_stage"] == "qa"

    def test_resume_default_from_stage_is_dev(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=mock_compiled) as mock_bg,
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            result = runner.invoke(app, ["resume", str(session)])

        _, kwargs = mock_bg.call_args
        assert kwargs["entry_stage"] == "dev"

    def test_resume_reloads_feature_spec_artifact(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session)])

        # Extract state passed to ainvoke
        state = mock_compiled.ainvoke.call_args.args[0]
        assert state["feature_spec"] == {"features": ["auth"]}

    def test_resume_reloads_text_artifacts(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session)])

        state = mock_compiled.ainvoke.call_args.args[0]
        assert state["grill_transcript"] == "Q: What auth?\nA: OAuth"
        assert state["research_report"] == "Use React"
        assert state["architecture"] == "## Architecture\nMicroservices"

    def test_resume_reloads_dev_tasks(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session)])

        state = mock_compiled.ainvoke.call_args.args[0]
        assert state["dev_tasks"] == [{"id": "T1"}]

    def test_resume_missing_artifacts_default_to_empty(self, tmp_path: Path) -> None:
        """If artifact files don't exist, state fields should be empty defaults."""
        session = tmp_path / "kindle_noarts"
        (session / "artifacts").mkdir(parents=True)
        (session / "logs").mkdir()
        meta = {
            "project_id": "kindle_noarts",
            "idea": "Minimal app",
            "status": "in_progress",
            "stages_completed": [],
        }
        (session / "metadata.json").write_text(json.dumps(meta))

        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session)])

        state = mock_compiled.ainvoke.call_args.args[0]
        assert state["feature_spec"] == {}
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""
        assert state["dev_tasks"] == []
        assert state["qa_report"] == ""
        assert state["product_audit"] == ""

    def test_resume_calls_ui_banner_with_idea_and_id(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()
        mock_ui = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session)])

        mock_ui.banner.assert_called_once_with("A todo app", "kindle_abc12345")

    def test_resume_calls_ui_info_with_stage(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()
        mock_ui = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session), "--from", "qa"])

        info_calls = [c.args[0] for c in mock_ui.info.call_args_list]
        assert any("qa" in c for c in info_calls)

    def test_resume_auto_approve_forwarded(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session), "--auto-approve"])

        MockUI.assert_called_once_with(auto_approve=True, verbose=False)

    def test_resume_verbose_forwarded(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session), "--verbose"])

        MockUI.assert_called_once_with(auto_approve=False, verbose=True)

    def test_resume_state_uses_settings_defaults(self, tmp_path: Path) -> None:
        """Resume state should pull config values from Settings, not CLI."""
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()

        settings = _fake_settings(
            max_concurrent_agents=6,
            max_qa_retries=8,
            max_cpo_retries=7,
            model="claude-sonnet-4-20250514",
            max_agent_turns=100,
        )

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=session / "workspace"),
        ):
            runner.invoke(app, ["resume", str(session)])

        state = mock_compiled.ainvoke.call_args.args[0]
        assert state["max_concurrent_agents"] == 6
        assert state["max_qa_retries"] == 8
        assert state["max_cpo_retries"] == 7
        assert state["model"] == "claude-sonnet-4-20250514"
        assert state["max_agent_turns"] == 100

    def test_resume_reports_workspace_path(self, tmp_path: Path) -> None:
        session = self._setup_session(tmp_path)
        mock_compiled = MagicMock()
        mock_ui = MagicMock()
        ws = session / "workspace"

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.build_graph", return_value=mock_compiled),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ws),
        ):
            runner.invoke(app, ["resume", str(session)])

        info_calls = [c.args[0] for c in mock_ui.info.call_args_list]
        assert any("Project at:" in c for c in info_calls)


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListCommand:
    """Tests for the 'list' subcommand."""

    def test_list_calls_list_projects_with_settings_root(self) -> None:
        settings = _fake_settings(projects_root=Path("/fake/root"))
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.list_projects", return_value=[]) as mock_lp,
        ):
            mock_ui = MagicMock()
            MockUI.return_value = mock_ui
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_lp.assert_called_once_with(Path("/fake/root"))

    def test_list_passes_projects_to_ui_show_projects(self) -> None:
        settings = _fake_settings()
        projects = [
            {"project_id": "kindle_aaa", "idea": "App 1", "status": "completed"},
            {"project_id": "kindle_bbb", "idea": "App 2", "status": "in_progress"},
        ]
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.list_projects", return_value=projects),
        ):
            mock_ui = MagicMock()
            MockUI.return_value = mock_ui
            result = runner.invoke(app, ["list"])

        mock_ui.show_projects.assert_called_once_with(projects)

    def test_list_with_no_projects(self) -> None:
        settings = _fake_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            mock_ui = MagicMock()
            MockUI.return_value = mock_ui
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_ui.show_projects.assert_called_once_with([])


# ---------------------------------------------------------------------------
# build command — CLI option parsing
# ---------------------------------------------------------------------------


class TestBuildCLIOptionParsing:
    """Verify that CLI option short forms and combinations are parsed correctly."""

    def test_short_stack_flag(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["An app", "-s", "nextjs"])

        state = ctx["mock_compiled"].ainvoke.call_args.args[0]
        assert state["stack_preference"] == "nextjs"

    def test_short_concurrency_flag(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["An app", "-c", "12"])

        state = ctx["mock_compiled"].ainvoke.call_args.args[0]
        assert state["max_concurrent_agents"] == 12

    def test_short_output_flag(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        (ctx["workspace"] / "app.py").write_text("print('hi')")
        output_dir = tmp_path / "out"

        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["An app", "-o", str(output_dir)])

        assert output_dir.exists()

    def test_short_verbose_flag(self, tmp_path: Path) -> None:
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["An app", "-v"])

        MockUI.assert_called_once_with(auto_approve=False, verbose=True)


# ---------------------------------------------------------------------------
# build command — settings override logic
# ---------------------------------------------------------------------------


class TestBuildSettingsOverrides:
    """When CLI provides nonzero values, they should override Settings defaults.
    When CLI args are 0 (the default), Settings values should be used."""

    def test_all_overrides_applied(self, tmp_path: Path) -> None:
        """All three override-able settings set from CLI."""
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=ctx["settings"]),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, [
                "An app",
                "--concurrency", "16",
                "--qa-retries", "20",
                "--cpo-retries", "15",
            ])

        state = ctx["mock_compiled"].ainvoke.call_args.args[0]
        assert state["max_concurrent_agents"] == 16
        assert state["max_qa_retries"] == 20
        assert state["max_cpo_retries"] == 15

    def test_no_overrides_uses_settings(self, tmp_path: Path) -> None:
        """When all CLI args are default (0), Settings values are used."""
        settings = _fake_settings(
            max_concurrent_agents=12,
            max_qa_retries=7,
            max_cpo_retries=9,
        )
        ctx = _setup_build_mocks(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=(ctx["project_id"], ctx["project_dir"])),
            patch("kindle.cli.build_graph", return_value=ctx["mock_compiled"]),
            patch("kindle.cli.asyncio.run", return_value={}),
            patch("kindle.cli.workspace_path", return_value=ctx["workspace"]),
        ):
            result = runner.invoke(app, ["An app"])

        state = ctx["mock_compiled"].ainvoke.call_args.args[0]
        assert state["max_concurrent_agents"] == 12
        assert state["max_qa_retries"] == 7
        assert state["max_cpo_retries"] == 9


# ---------------------------------------------------------------------------
# app entrypoint
# ---------------------------------------------------------------------------


class TestAppEntrypoint:
    """The typer app object should be properly configured."""

    def test_app_name(self) -> None:
        assert app.info.name == "kindle"

    def test_app_has_help_text(self) -> None:
        assert app.info.help is not None
        assert "application factory" in app.info.help.lower()

    def test_help_flag_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "kindle" in result.output.lower()

    def test_list_subcommand_in_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "list" in result.output

    def test_resume_subcommand_in_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "resume" in result.output
