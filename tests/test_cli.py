"""Tests for kindle.cli — Typer CLI entry points for build, resume, and list."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from kindle.cli import app

# ---------------------------------------------------------------------------
# Shared helpers and fixtures
# ---------------------------------------------------------------------------

runner = CliRunner()


def _fake_settings(**overrides: object) -> MagicMock:
    """Return a mock Settings with sensible defaults."""
    s = MagicMock()
    s.max_concurrent_agents = overrides.get("max_concurrent_agents", 4)
    s.max_qa_retries = overrides.get("max_qa_retries", 10)
    s.max_cpo_retries = overrides.get("max_cpo_retries", 10)
    s.model = overrides.get("model", "claude-opus-4-20250514")
    s.max_agent_turns = overrides.get("max_agent_turns", 50)
    s.projects_root = overrides.get("projects_root", Path("/tmp/kindle/projects"))
    return s


def _fake_compiled_graph() -> MagicMock:
    """Return a mock compiled graph whose ainvoke is an AsyncMock."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={})
    return graph


@pytest.fixture
def settings():
    return _fake_settings()


@pytest.fixture
def compiled_graph():
    return _fake_compiled_graph()


@pytest.fixture
def _patch_build_deps(settings, compiled_graph, tmp_path):
    """Patch all heavy dependencies used by the ``build`` command."""
    project_dir = tmp_path / "kindle_abc12345"
    project_dir.mkdir(parents=True)

    with (
        patch("kindle.cli.Settings.load", return_value=settings),
        patch("kindle.cli.UI", return_value=MagicMock()),
        patch("kindle.cli.create_project", return_value=("kindle_abc12345", project_dir)),
        patch("kindle.cli.build_graph", return_value=compiled_graph),
        patch("kindle.cli.workspace_path", return_value=tmp_path / "workspace"),
        patch("kindle.cli.asyncio.run") as mock_arun,
    ):
        mock_arun.return_value = None
        yield {
            "settings": settings,
            "compiled_graph": compiled_graph,
            "project_dir": project_dir,
            "mock_arun": mock_arun,
        }


# ---------------------------------------------------------------------------
# build — no idea provided (usage hint)
# ---------------------------------------------------------------------------


class TestBuildNoIdea:
    """When no idea argument is given, the CLI prints usage and exits 0."""

    def test_prints_usage_hint(self):
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
        ):
            result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage" in result.output or "kindle" in result.output

    def test_does_not_invoke_graph(self):
        with (
            patch("kindle.cli.Settings.load") as mock_load,
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph") as mock_bg,
        ):
            runner.invoke(app, [])
        mock_bg.assert_not_called()


# ---------------------------------------------------------------------------
# build — happy path
# ---------------------------------------------------------------------------


class TestBuildHappyPath:
    """The ``build`` command wires up settings, creates a project, and runs the graph."""

    @pytest.mark.usefixtures("_patch_build_deps")
    def test_exit_code_is_zero(self, _patch_build_deps):
        result = runner.invoke(app, ["build a todo app"])
        assert result.exit_code == 0

    def test_calls_settings_load(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        with (
            patch("kindle.cli.Settings.load", return_value=s) as mock_load,
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["build a todo app"])
        mock_load.assert_called_once()

    def test_creates_project_with_idea(self, tmp_path):
        s = _fake_settings(projects_root=tmp_path)
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)) as mock_cp,
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["build a todo app"])
        mock_cp.assert_called_once_with(tmp_path, "build a todo app")

    def test_passes_initial_state_to_graph(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run") as mock_arun,
        ):
            runner.invoke(app, ["build a todo app"])

        # asyncio.run should be called with the coroutine from ainvoke
        mock_arun.assert_called_once()
        # The coroutine passed to asyncio.run comes from g.ainvoke(state)
        g.ainvoke.assert_called_once()
        state = g.ainvoke.call_args[0][0]
        assert state["idea"] == "build a todo app"
        assert state["project_id"] == "kindle_x"
        assert state["project_dir"] == str(pd)

    def test_calls_build_graph_with_ui(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g) as mock_bg,
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["an idea"])
        mock_bg.assert_called_once_with(mock_ui)

    def test_ui_banner_called_with_idea_and_project_id(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["my app idea"])
        mock_ui.banner.assert_called_once_with("my app idea", "kindle_x")

    def test_ui_info_reports_workspace_path(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        ws = tmp_path / "workspace"
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=ws),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["idea"])
        info_calls = [str(c) for c in mock_ui.info.call_args_list]
        assert any(str(ws) in c for c in info_calls)


# ---------------------------------------------------------------------------
# build — CLI option overrides
# ---------------------------------------------------------------------------


class TestBuildCLIOverrides:
    """CLI flags like --concurrency override config defaults in initial_state."""

    def _run_build_with_args(self, args: list[str], tmp_path: Path, settings=None):
        """Invoke the build command and return the state passed to ainvoke."""
        s = settings or _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir(exist_ok=True)
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            result = runner.invoke(app, args)
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        g.ainvoke.assert_called_once()
        return g.ainvoke.call_args[0][0]

    def test_concurrency_flag_overrides_default(self, tmp_path):
        state = self._run_build_with_args(["my app", "--concurrency", "8"], tmp_path)
        assert state["max_concurrent_agents"] == 8

    def test_concurrency_zero_uses_config(self, tmp_path):
        s = _fake_settings(max_concurrent_agents=6)
        state = self._run_build_with_args(["my app", "--concurrency", "0"], tmp_path, settings=s)
        assert state["max_concurrent_agents"] == 6

    def test_qa_retries_flag(self, tmp_path):
        state = self._run_build_with_args(["my app", "--qa-retries", "5"], tmp_path)
        assert state["max_qa_retries"] == 5

    def test_qa_retries_zero_uses_config(self, tmp_path):
        s = _fake_settings(max_qa_retries=7)
        state = self._run_build_with_args(["my app", "--qa-retries", "0"], tmp_path, settings=s)
        assert state["max_qa_retries"] == 7

    def test_cpo_retries_flag(self, tmp_path):
        state = self._run_build_with_args(["my app", "--cpo-retries", "3"], tmp_path)
        assert state["max_cpo_retries"] == 3

    def test_cpo_retries_zero_uses_config(self, tmp_path):
        s = _fake_settings(max_cpo_retries=12)
        state = self._run_build_with_args(["my app", "--cpo-retries", "0"], tmp_path, settings=s)
        assert state["max_cpo_retries"] == 12

    def test_stack_flag(self, tmp_path):
        state = self._run_build_with_args(["my app", "--stack", "react"], tmp_path)
        assert state["stack_preference"] == "react"

    def test_stack_short_flag(self, tmp_path):
        state = self._run_build_with_args(["my app", "-s", "fastapi"], tmp_path)
        assert state["stack_preference"] == "fastapi"

    def test_auto_approve_flag(self, tmp_path):
        state = self._run_build_with_args(["my app", "--auto-approve"], tmp_path)
        assert state["auto_approve"] is True

    def test_auto_approve_default_false(self, tmp_path):
        state = self._run_build_with_args(["my app"], tmp_path)
        assert state["auto_approve"] is False

    def test_model_from_settings(self, tmp_path):
        s = _fake_settings(model="claude-sonnet-4-20250514")
        state = self._run_build_with_args(["my app"], tmp_path, settings=s)
        assert state["model"] == "claude-sonnet-4-20250514"

    def test_max_agent_turns_from_settings(self, tmp_path):
        s = _fake_settings(max_agent_turns=100)
        state = self._run_build_with_args(["my app"], tmp_path, settings=s)
        assert state["max_agent_turns"] == 100


# ---------------------------------------------------------------------------
# build — initial state structure
# ---------------------------------------------------------------------------


class TestBuildInitialState:
    """Verify the shape and defaults of the initial KindleState passed to the graph."""

    def _get_state(self, tmp_path: Path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir(exist_ok=True)
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["a cool idea"])
        return g.ainvoke.call_args[0][0]

    def test_feature_spec_is_empty_dict(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["feature_spec"] == {}

    def test_grill_transcript_is_empty(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["grill_transcript"] == ""

    def test_research_report_is_empty(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["research_report"] == ""

    def test_architecture_is_empty(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["architecture"] == ""

    def test_dev_tasks_is_empty_list(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["dev_tasks"] == []

    def test_qa_passed_is_false(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["qa_passed"] is False

    def test_cpo_passed_is_false(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["cpo_passed"] is False

    def test_qa_retries_is_zero(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["qa_retries"] == 0

    def test_cpo_retries_is_zero(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["cpo_retries"] == 0

    def test_current_stage_is_empty(self, tmp_path):
        state = self._get_state(tmp_path)
        assert state["current_stage"] == ""


# ---------------------------------------------------------------------------
# build — output directory copy
# ---------------------------------------------------------------------------


class TestBuildOutputCopy:
    """When --output is given, the workspace is copied to the target path."""

    def test_copies_workspace_when_output_not_exists(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        ws = tmp_path / "workspace"
        ws.mkdir()
        output_dir = tmp_path / "my_output"

        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=ws),
            patch("kindle.cli.asyncio.run"),
            patch("shutil.copytree") as mock_copytree,
        ):
            result = runner.invoke(app, ["idea", "--output", str(output_dir)])
        assert result.exit_code == 0
        mock_copytree.assert_called_once_with(str(ws), str(output_dir))

    def test_error_when_output_exists(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        ws = tmp_path / "workspace"
        ws.mkdir()
        output_dir = tmp_path / "existing_output"
        output_dir.mkdir()

        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=ws),
            patch("kindle.cli.asyncio.run"),
            patch("shutil.copytree") as mock_copytree,
        ):
            result = runner.invoke(app, ["idea", "--output", str(output_dir)])
        mock_copytree.assert_not_called()
        mock_ui.error.assert_called_once()
        assert "already exists" in str(mock_ui.error.call_args)

    def test_no_copy_when_output_not_specified(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI"),
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
            patch("shutil.copytree") as mock_copytree,
        ):
            result = runner.invoke(app, ["idea"])
        mock_copytree.assert_not_called()


# ---------------------------------------------------------------------------
# build — verbose and auto_approve forwarded to UI
# ---------------------------------------------------------------------------


class TestBuildUIFlags:
    """Verify that --verbose and --auto-approve are passed to the UI constructor."""

    def test_verbose_flag_forwarded(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["idea", "--verbose"])
        MockUI.assert_called_once_with(auto_approve=False, verbose=True)

    def test_auto_approve_flag_forwarded(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["idea", "--auto-approve"])
        MockUI.assert_called_once_with(auto_approve=True, verbose=False)

    def test_both_flags_forwarded(self, tmp_path):
        s = _fake_settings()
        g = _fake_compiled_graph()
        pd = tmp_path / "kindle_x"
        pd.mkdir()
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.create_project", return_value=("kindle_x", pd)),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["idea", "--auto-approve", "--verbose"])
        MockUI.assert_called_once_with(auto_approve=True, verbose=True)


# ---------------------------------------------------------------------------
# resume — happy path
# ---------------------------------------------------------------------------


class TestResumeHappyPath:
    """The ``resume`` command reloads artifacts and resumes the graph."""

    def _setup_project_dir(self, tmp_path: Path, *, idea: str = "my app") -> Path:
        """Create a fake project directory with metadata."""
        pd = tmp_path / "kindle_abc12345"
        pd.mkdir(parents=True)
        meta = {"project_id": "kindle_abc12345", "idea": idea}
        (pd / "metadata.json").write_text(json.dumps(meta))
        return pd

    def test_exit_code_is_zero(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            result = runner.invoke(app, ["resume", str(pd)])
        assert result.exit_code == 0

    def test_loads_settings(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()) as mock_load,
            patch("kindle.cli.UI"),
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)])
        mock_load.assert_called_once()

    def test_builds_graph_with_entry_stage(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g) as mock_bg,
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd), "--from", "qa"])
        mock_bg.assert_called_once_with(mock_ui, entry_stage="qa")

    def test_default_from_stage_is_dev(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g) as mock_bg,
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)])
        mock_bg.assert_called_once_with(mock_ui, entry_stage="dev")

    def test_state_contains_idea_and_project_id(self, tmp_path):
        pd = self._setup_project_dir(tmp_path, idea="my cool app")
        g = _fake_compiled_graph()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)])
        state = g.ainvoke.call_args[0][0]
        assert state["idea"] == "my cool app"
        assert state["project_id"] == "kindle_abc12345"
        assert state["project_dir"] == str(pd)

    def test_reloads_feature_spec_artifact(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()
        spec = {"features": ["auth", "dashboard"]}

        def fake_load_artifact(project_dir, name):
            if name == "feature_spec.json":
                return json.dumps(spec)
            return None

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.load_artifact", side_effect=fake_load_artifact),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)])
        state = g.ainvoke.call_args[0][0]
        assert state["feature_spec"] == spec

    def test_reloads_dev_tasks_artifact(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()
        tasks = [{"name": "implement auth"}, {"name": "add tests"}]

        def fake_load_artifact(project_dir, name):
            if name == "dev_tasks.json":
                return json.dumps(tasks)
            return None

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.load_artifact", side_effect=fake_load_artifact),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)])
        state = g.ainvoke.call_args[0][0]
        assert state["dev_tasks"] == tasks

    def test_reloads_text_artifacts(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()

        artifact_map = {
            "grill_transcript.md": "# Grill output",
            "research_report.md": "# Research output",
            "architecture.md": "# Architecture",
            "qa_report.md": "# QA report",
            "product_audit.md": "# Product audit",
        }

        def fake_load_artifact(project_dir, name):
            return artifact_map.get(name)

        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.load_artifact", side_effect=fake_load_artifact),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)])
        state = g.ainvoke.call_args[0][0]
        assert state["grill_transcript"] == "# Grill output"
        assert state["research_report"] == "# Research output"
        assert state["architecture"] == "# Architecture"
        assert state["qa_report"] == "# QA report"
        assert state["product_audit"] == "# Product audit"

    def test_missing_artifacts_default_to_empty(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)])
        state = g.ainvoke.call_args[0][0]
        assert state["feature_spec"] == {}
        assert state["dev_tasks"] == []
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""

    def test_ui_banner_called(self, tmp_path):
        pd = self._setup_project_dir(tmp_path, idea="my app")
        g = _fake_compiled_graph()
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)])
        mock_ui.banner.assert_called_once_with("my app", "kindle_abc12345")

    def test_ui_info_resume_stage(self, tmp_path):
        pd = self._setup_project_dir(tmp_path)
        g = _fake_compiled_graph()
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd), "--from", "architect"])
        info_calls = [str(c) for c in mock_ui.info.call_args_list]
        assert any("architect" in c for c in info_calls)


# ---------------------------------------------------------------------------
# resume — error cases
# ---------------------------------------------------------------------------


class TestResumeErrors:
    """Error paths when resuming a build session."""

    def test_nonexistent_project_dir_exits_1(self, tmp_path):
        fake_dir = tmp_path / "nonexistent"
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
        ):
            result = runner.invoke(app, ["resume", str(fake_dir)])
        assert result.exit_code == 1

    def test_nonexistent_project_dir_shows_error(self, tmp_path):
        fake_dir = tmp_path / "nonexistent"
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
        ):
            runner.invoke(app, ["resume", str(fake_dir)])
        mock_ui.error.assert_called_once()
        assert "not found" in str(mock_ui.error.call_args)

    def test_missing_metadata_json_exits_1(self, tmp_path):
        pd = tmp_path / "kindle_x"
        pd.mkdir()  # No metadata.json
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
        ):
            result = runner.invoke(app, ["resume", str(pd)])
        assert result.exit_code == 1

    def test_missing_metadata_shows_helpful_error(self, tmp_path):
        pd = tmp_path / "kindle_x"
        pd.mkdir()  # No metadata.json
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
        ):
            runner.invoke(app, ["resume", str(pd)])
        mock_ui.error.assert_called_once()
        error_msg = str(mock_ui.error.call_args)
        assert "metadata.json" in error_msg or "valid Kindle session" in error_msg


# ---------------------------------------------------------------------------
# resume — UI flags forwarding
# ---------------------------------------------------------------------------


class TestResumeUIFlags:
    """--auto-approve and --verbose are forwarded to the UI constructor in resume."""

    def _setup_and_run(self, tmp_path: Path, extra_args: list[str]):
        pd = tmp_path / "kindle_x"
        pd.mkdir(parents=True)
        meta = {"project_id": "kindle_x", "idea": "test"}
        (pd / "metadata.json").write_text(json.dumps(meta))
        g = _fake_compiled_graph()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.load_artifact", return_value=None),
            patch("kindle.cli.build_graph", return_value=g),
            patch("kindle.cli.workspace_path", return_value=tmp_path / "ws"),
            patch("kindle.cli.asyncio.run"),
        ):
            runner.invoke(app, ["resume", str(pd)] + extra_args)
        return MockUI

    def test_verbose_forwarded(self, tmp_path):
        MockUI = self._setup_and_run(tmp_path, ["--verbose"])
        MockUI.assert_called_once_with(auto_approve=False, verbose=True)

    def test_auto_approve_forwarded(self, tmp_path):
        MockUI = self._setup_and_run(tmp_path, ["--auto-approve"])
        MockUI.assert_called_once_with(auto_approve=True, verbose=False)

    def test_both_flags_forwarded(self, tmp_path):
        MockUI = self._setup_and_run(tmp_path, ["--auto-approve", "--verbose"])
        MockUI.assert_called_once_with(auto_approve=True, verbose=True)


# ---------------------------------------------------------------------------
# list — happy path
# ---------------------------------------------------------------------------


class TestListCommand:
    """The ``list`` command loads settings, fetches projects, and displays them."""

    def test_exit_code_is_zero(self):
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI"),
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

    def test_calls_list_projects_with_projects_root(self):
        root = Path("/custom/root")
        s = _fake_settings(projects_root=root)
        with (
            patch("kindle.cli.Settings.load", return_value=s),
            patch("kindle.cli.UI"),
            patch("kindle.cli.list_projects", return_value=[]) as mock_lp,
        ):
            runner.invoke(app, ["list"])
        mock_lp.assert_called_once_with(root)

    def test_passes_projects_to_ui_show_projects(self):
        projects = [
            {"project_id": "kindle_abc", "idea": "a todo app"},
            {"project_id": "kindle_def", "idea": "a chat app"},
        ]
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.list_projects", return_value=projects),
        ):
            runner.invoke(app, ["list"])
        mock_ui.show_projects.assert_called_once_with(projects)

    def test_empty_project_list_still_calls_show(self):
        mock_ui = MagicMock()
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI", return_value=mock_ui),
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            runner.invoke(app, ["list"])
        mock_ui.show_projects.assert_called_once_with([])

    def test_ui_created_with_defaults(self):
        with (
            patch("kindle.cli.Settings.load", return_value=_fake_settings()),
            patch("kindle.cli.UI") as MockUI,
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            runner.invoke(app, ["list"])
        MockUI.assert_called_once_with()


# ---------------------------------------------------------------------------
# help output
# ---------------------------------------------------------------------------


class TestHelpOutput:
    """Verify the CLI --help text is accessible and descriptive."""

    def test_help_exit_code_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_contains_app_description(self):
        result = runner.invoke(app, ["--help"])
        assert "application factory" in result.output.lower() or "kindle" in result.output.lower()

    def test_resume_help_exit_code_zero(self):
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0

    def test_list_help_exit_code_zero(self):
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
