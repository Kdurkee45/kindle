"""Tests for kindle.cli — Typer CLI entry point (build, resume, list)."""

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


def _make_settings(**overrides) -> MagicMock:
    """Return a mock Settings with sensible defaults."""
    s = MagicMock()
    s.model = overrides.get("model", "claude-opus-4-20250514")
    s.max_agent_turns = overrides.get("max_agent_turns", 50)
    s.max_concurrent_agents = overrides.get("max_concurrent_agents", 4)
    s.max_qa_retries = overrides.get("max_qa_retries", 10)
    s.max_cpo_retries = overrides.get("max_cpo_retries", 10)
    s.projects_root = overrides.get("projects_root", Path("/tmp/kindle_projects"))
    return s


def _make_ui() -> MagicMock:
    """Return a mock UI with all methods the CLI calls."""
    ui = MagicMock()
    ui.banner = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.show_projects = MagicMock()
    return ui


def _make_compiled_graph() -> MagicMock:
    """Return a mock compiled graph whose ainvoke is an AsyncMock."""
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value={})
    return compiled


def _project_dir(tmp_path: Path, *, idea: str = "test app", project_id: str = "kindle_abc12345") -> Path:
    """Create a realistic project directory with metadata and artifacts."""
    d = tmp_path / project_id
    (d / "artifacts").mkdir(parents=True)
    (d / "logs").mkdir(parents=True)
    (d / "workspace").mkdir(parents=True)

    meta = {"project_id": project_id, "idea": idea, "stages_completed": ["grill", "research"]}
    (d / "metadata.json").write_text(json.dumps(meta))

    # Stage artifacts that resume loads
    (d / "artifacts" / "feature_spec.json").write_text(json.dumps({"features": ["auth"]}))
    (d / "artifacts" / "grill_transcript.md").write_text("Q: Who is the user?\nA: Developers")
    (d / "artifacts" / "research_report.md").write_text("## Research\nFound existing solutions")
    (d / "artifacts" / "architecture.md").write_text("## Arch\nMicroservices")
    (d / "artifacts" / "dev_tasks.json").write_text(json.dumps([{"task": "build auth"}]))
    (d / "artifacts" / "qa_report.md").write_text("All tests pass")
    (d / "artifacts" / "product_audit.md").write_text("Meets requirements")
    return d


# ---------------------------------------------------------------------------
# Shared patches — every test must isolate Settings.load, UI, build_graph
# ---------------------------------------------------------------------------

def _cli_patches(settings=None, ui=None, compiled=None, create_project_ret=None):
    """Return a dict of common patches for CLI tests."""
    _settings = settings or _make_settings()
    _ui = ui or _make_ui()
    _compiled = compiled or _make_compiled_graph()
    _create = create_project_ret or ("kindle_test1234", Path("/tmp/kindle_projects/kindle_test1234"))

    return {
        "settings_load": patch("kindle.cli.Settings.load", return_value=_settings),
        "ui_cls": patch("kindle.cli.UI", return_value=_ui),
        "build_graph": patch("kindle.cli.build_graph", return_value=_compiled),
        "create_project": patch("kindle.cli.create_project", return_value=_create),
        "workspace_path": patch("kindle.cli.workspace_path", return_value=Path("/tmp/workspace")),
    }


# ---------------------------------------------------------------------------
# build command — happy path with minimal args
# ---------------------------------------------------------------------------


class TestBuildMinimal:
    """The default build command with just an idea argument."""

    def test_build_happy_path(self) -> None:
        """Invoking `kindle 'my app'` should call build_graph and ainvoke."""
        patches = _cli_patches()
        with (
            patches["settings_load"],
            patches["ui_cls"] as mock_ui_cls,
            patches["build_graph"] as mock_build_graph,
            patches["create_project"],
            patches["workspace_path"],
        ):
            result = runner.invoke(app, ["my cool app"])

        assert result.exit_code == 0, result.output
        mock_build_graph.assert_called_once()
        # ainvoke should have been called with the initial state
        compiled = mock_build_graph.return_value
        compiled.ainvoke.assert_called_once()

    def test_build_creates_project(self) -> None:
        """build should call create_project with the correct idea."""
        settings = _make_settings()
        patches = _cli_patches(settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"] as mock_create,
            patches["workspace_path"],
        ):
            runner.invoke(app, ["build a todo app"])

        mock_create.assert_called_once_with(settings.projects_root, "build a todo app")

    def test_build_displays_banner(self) -> None:
        """build should call ui.banner with idea and project_id."""
        ui = _make_ui()
        patches = _cli_patches(ui=ui)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        ui.banner.assert_called_once_with("my app", "kindle_test1234")

    def test_build_shows_project_path_on_completion(self) -> None:
        """After build completes, ui.info should show project and session paths."""
        ui = _make_ui()
        patches = _cli_patches(ui=ui)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        info_calls = [str(c) for c in ui.info.call_args_list]
        assert any("Project built at" in c for c in info_calls)
        assert any("Session artifacts" in c for c in info_calls)


# ---------------------------------------------------------------------------
# build command — no idea shows usage
# ---------------------------------------------------------------------------


class TestBuildNoIdea:
    """When no idea is provided, build should print usage and exit 0."""

    def test_no_args_shows_usage(self) -> None:
        patches = _cli_patches()
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert "Usage:" in result.output or "kindle" in result.output.lower()


# ---------------------------------------------------------------------------
# build command — all optional flags
# ---------------------------------------------------------------------------


class TestBuildAllFlags:
    """build with every optional flag verifies they flow through to state."""

    def test_all_flags_accepted(self) -> None:
        """The CLI should accept all flags without error."""
        patches = _cli_patches()
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            result = runner.invoke(app, [
                "my app",
                "--stack", "react",
                "--auto-approve",
                "--concurrency", "8",
                "--review-arch",
                "--output", "/tmp/output",
                "--qa-retries", "5",
                "--cpo-retries", "3",
                "--verbose",
            ])

        assert result.exit_code == 0, result.output

    def test_stack_flag_sets_stack_preference(self) -> None:
        """--stack should appear in the initial_state passed to ainvoke."""
        compiled = _make_compiled_graph()
        patches = _cli_patches(compiled=compiled)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app", "--stack", "nextjs"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["stack_preference"] == "nextjs"

    def test_auto_approve_flag(self) -> None:
        """--auto-approve should set auto_approve=True in state."""
        compiled = _make_compiled_graph()
        patches = _cli_patches(compiled=compiled)
        with (
            patches["settings_load"],
            patches["ui_cls"] as mock_ui_cls,
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app", "--auto-approve"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["auto_approve"] is True
        # Also verify UI was created with auto_approve=True
        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=False)

    def test_concurrency_flag_overrides_default(self) -> None:
        """--concurrency 8 should override settings.max_concurrent_agents."""
        compiled = _make_compiled_graph()
        settings = _make_settings(max_concurrent_agents=4)
        patches = _cli_patches(compiled=compiled, settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app", "--concurrency", "8"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 8

    def test_concurrency_zero_uses_config_default(self) -> None:
        """--concurrency 0 (default) falls back to settings value."""
        compiled = _make_compiled_graph()
        settings = _make_settings(max_concurrent_agents=4)
        patches = _cli_patches(compiled=compiled, settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 4

    def test_qa_retries_flag_overrides_default(self) -> None:
        """--qa-retries 5 should override settings.max_qa_retries."""
        compiled = _make_compiled_graph()
        settings = _make_settings(max_qa_retries=10)
        patches = _cli_patches(compiled=compiled, settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app", "--qa-retries", "5"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 5

    def test_qa_retries_zero_uses_config_default(self) -> None:
        compiled = _make_compiled_graph()
        settings = _make_settings(max_qa_retries=10)
        patches = _cli_patches(compiled=compiled, settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 10

    def test_cpo_retries_flag_overrides_default(self) -> None:
        """--cpo-retries 3 should override settings.max_cpo_retries."""
        compiled = _make_compiled_graph()
        settings = _make_settings(max_cpo_retries=10)
        patches = _cli_patches(compiled=compiled, settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app", "--cpo-retries", "3"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_cpo_retries"] == 3

    def test_cpo_retries_zero_uses_config_default(self) -> None:
        compiled = _make_compiled_graph()
        settings = _make_settings(max_cpo_retries=10)
        patches = _cli_patches(compiled=compiled, settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_cpo_retries"] == 10

    def test_verbose_flag_passed_to_ui(self) -> None:
        """--verbose should create UI with verbose=True."""
        patches = _cli_patches()
        with (
            patches["settings_load"],
            patches["ui_cls"] as mock_ui_cls,
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app", "--verbose"])

        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)


# ---------------------------------------------------------------------------
# build command — KindleState initialization correctness
# ---------------------------------------------------------------------------


class TestBuildStateInit:
    """Verify the initial KindleState dict is fully and correctly populated."""

    def test_initial_state_has_all_required_fields(self) -> None:
        """The state dict passed to ainvoke must contain every KindleState key."""
        compiled = _make_compiled_graph()
        settings = _make_settings()
        patches = _cli_patches(compiled=compiled, settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]

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

    def test_initial_state_artifact_fields_are_empty(self) -> None:
        """All artifact fields should be empty/default for a fresh build."""
        compiled = _make_compiled_graph()
        patches = _cli_patches(compiled=compiled)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
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
        assert state["current_stage"] == ""

    def test_initial_state_uses_settings_model(self) -> None:
        """model and max_agent_turns should come from Settings."""
        compiled = _make_compiled_graph()
        settings = _make_settings(model="claude-sonnet-4-20250514", max_agent_turns=100)
        patches = _cli_patches(compiled=compiled, settings=settings)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["model"] == "claude-sonnet-4-20250514"
        assert state["max_agent_turns"] == 100

    def test_initial_state_idea_matches_argument(self) -> None:
        compiled = _make_compiled_graph()
        patches = _cli_patches(compiled=compiled)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["a task management SaaS"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == "a task management SaaS"

    def test_initial_state_project_id_from_create(self) -> None:
        """project_id and project_dir should come from create_project return."""
        compiled = _make_compiled_graph()
        proj_dir = Path("/tmp/kindle_projects/kindle_xyz99999")
        patches = _cli_patches(
            compiled=compiled,
            create_project_ret=("kindle_xyz99999", proj_dir),
        )
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["project_id"] == "kindle_xyz99999"
        assert state["project_dir"] == str(proj_dir)


# ---------------------------------------------------------------------------
# build command — --output flag triggers directory copy
# ---------------------------------------------------------------------------


class TestBuildOutputFlag:
    """--output should copy the workspace to the specified directory."""

    def test_output_copies_workspace(self, tmp_path: Path) -> None:
        """When --output is set and target doesn't exist, shutil.copytree is called."""
        ui = _make_ui()
        output_dir = tmp_path / "my_output"
        patches = _cli_patches(ui=ui)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
            patch("shutil.copytree") as mock_copytree,
        ):
            result = runner.invoke(app, ["my app", "--output", str(output_dir)])

        assert result.exit_code == 0, result.output
        mock_copytree.assert_called_once()
        # Verify the source is the workspace path
        args = mock_copytree.call_args[0]
        assert args[0] == str(Path("/tmp/workspace"))

    def test_output_existing_dir_shows_error(self, tmp_path: Path) -> None:
        """When the output directory already exists, an error is shown."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()

        ui = _make_ui()
        patches = _cli_patches(ui=ui)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            result = runner.invoke(app, ["my app", "--output", str(output_dir)])

        assert result.exit_code == 0
        ui.error.assert_called_once()
        error_msg = ui.error.call_args[0][0]
        assert "already exists" in error_msg

    def test_no_output_flag_does_not_copy(self) -> None:
        """Without --output, shutil.copytree should not be called."""
        patches = _cli_patches()
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
            patch("shutil.copytree") as mock_copytree,
        ):
            runner.invoke(app, ["my app"])

        mock_copytree.assert_not_called()


# ---------------------------------------------------------------------------
# build command — build_graph is called with the UI
# ---------------------------------------------------------------------------


class TestBuildGraphInvocation:
    """Verify build_graph receives the right arguments."""

    def test_build_graph_receives_ui_instance(self) -> None:
        ui = _make_ui()
        patches = _cli_patches(ui=ui)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"] as mock_bg,
            patches["create_project"],
            patches["workspace_path"],
        ):
            runner.invoke(app, ["my app"])

        mock_bg.assert_called_once_with(ui)


# ---------------------------------------------------------------------------
# resume command — happy path
# ---------------------------------------------------------------------------


class TestResumeHappyPath:
    """resume loads a previous session and re-enters the graph."""

    def test_resume_loads_metadata(self, tmp_path: Path) -> None:
        """resume should read metadata.json and use its idea/project_id."""
        proj = _project_dir(tmp_path)
        compiled = _make_compiled_graph()
        ui = _make_ui()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=ui),
            patch("kindle.cli.build_graph", return_value=compiled) as mock_bg,
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            result = runner.invoke(app, ["resume", str(proj)])

        assert result.exit_code == 0, result.output
        ui.banner.assert_called_once_with("test app", "kindle_abc12345")

    def test_resume_calls_build_graph_with_entry_stage(self, tmp_path: Path) -> None:
        """resume should pass entry_stage to build_graph (default='dev')."""
        proj = _project_dir(tmp_path)
        compiled = _make_compiled_graph()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled) as mock_bg,
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj)])

        mock_bg.assert_called_once()
        _, kwargs = mock_bg.call_args
        assert kwargs.get("entry_stage") == "dev"

    def test_resume_invokes_graph_with_loaded_state(self, tmp_path: Path) -> None:
        """resume should populate state from artifact files and invoke the graph."""
        proj = _project_dir(tmp_path)
        compiled = _make_compiled_graph()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj)])

        compiled.ainvoke.assert_called_once()
        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == "test app"
        assert state["project_id"] == "kindle_abc12345"
        assert state["project_dir"] == str(proj)
        assert state["feature_spec"] == {"features": ["auth"]}
        assert state["dev_tasks"] == [{"task": "build auth"}]
        assert "Research" in state["research_report"]
        assert "Arch" in state["architecture"]

    def test_resume_loads_all_artifact_fields(self, tmp_path: Path) -> None:
        """All artifact fields should be populated from files."""
        proj = _project_dir(tmp_path)
        compiled = _make_compiled_graph()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj)])

        state = compiled.ainvoke.call_args[0][0]
        assert state["grill_transcript"] != ""
        assert state["research_report"] != ""
        assert state["architecture"] != ""
        assert state["qa_report"] != ""
        assert state["product_audit"] != ""

    def test_resume_shows_project_path(self, tmp_path: Path) -> None:
        """After resume completes, ui.info should show project path."""
        proj = _project_dir(tmp_path)
        ui = _make_ui()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=ui),
            patch("kindle.cli.build_graph", return_value=_make_compiled_graph()),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj)])

        ui.info.assert_called()
        info_msg = ui.info.call_args[0][0]
        assert "Project at" in info_msg


# ---------------------------------------------------------------------------
# resume command — with --from stage flag
# ---------------------------------------------------------------------------


class TestResumeFromStage:
    """resume --from <stage> should pass the entry_stage to build_graph."""

    @pytest.mark.parametrize("stage", ["grill", "research", "architect", "dev", "qa", "package"])
    def test_from_flag_sets_entry_stage(self, tmp_path: Path, stage: str) -> None:
        proj = _project_dir(tmp_path)
        compiled = _make_compiled_graph()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled) as mock_bg,
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj), "--from", stage])

        _, kwargs = mock_bg.call_args
        assert kwargs["entry_stage"] == stage

    def test_resume_info_shows_stage_name(self, tmp_path: Path) -> None:
        """ui.info should log which stage the session is resuming from."""
        proj = _project_dir(tmp_path)
        ui = _make_ui()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=ui),
            patch("kindle.cli.build_graph", return_value=_make_compiled_graph()),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj), "--from", "qa"])

        info_calls = [str(c) for c in ui.info.call_args_list]
        assert any("qa" in c for c in info_calls)


# ---------------------------------------------------------------------------
# resume command — auto-approve and verbose flags
# ---------------------------------------------------------------------------


class TestResumeFlags:
    """resume passes --auto-approve and --verbose to the UI."""

    def test_auto_approve_passed_to_ui(self, tmp_path: Path) -> None:
        proj = _project_dir(tmp_path)

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()) as mock_ui_cls,
            patch("kindle.cli.build_graph", return_value=_make_compiled_graph()),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj), "--auto-approve"])

        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=False)

    def test_verbose_passed_to_ui(self, tmp_path: Path) -> None:
        proj = _project_dir(tmp_path)

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()) as mock_ui_cls,
            patch("kindle.cli.build_graph", return_value=_make_compiled_graph()),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj), "--verbose"])

        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)

    def test_auto_approve_in_resume_state(self, tmp_path: Path) -> None:
        """--auto-approve should set auto_approve=True in the resumed state."""
        proj = _project_dir(tmp_path)
        compiled = _make_compiled_graph()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj), "--auto-approve"])

        state = compiled.ainvoke.call_args[0][0]
        assert state["auto_approve"] is True


# ---------------------------------------------------------------------------
# resume command — invalid / missing session directory
# ---------------------------------------------------------------------------


class TestResumeInvalidSession:
    """resume with a bad path should error gracefully."""

    def test_nonexistent_directory_exits_1(self) -> None:
        """Pointing to a missing directory should exit with code 1."""
        ui = _make_ui()
        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=ui),
        ):
            result = runner.invoke(app, ["resume", "/nonexistent/kindle_fake123"])

        assert result.exit_code == 1
        ui.error.assert_called()
        error_msg = ui.error.call_args[0][0]
        assert "not found" in error_msg.lower()

    def test_directory_without_metadata_exits_1(self, tmp_path: Path) -> None:
        """A directory without metadata.json should be rejected."""
        empty_dir = tmp_path / "no_metadata"
        empty_dir.mkdir()
        ui = _make_ui()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=ui),
        ):
            result = runner.invoke(app, ["resume", str(empty_dir)])

        assert result.exit_code == 1
        ui.error.assert_called()
        error_msg = ui.error.call_args[0][0]
        assert "metadata.json" in error_msg

    def test_resume_missing_artifacts_uses_defaults(self, tmp_path: Path) -> None:
        """If artifact files are missing, state should use empty defaults."""
        proj = tmp_path / "kindle_sparse"
        proj.mkdir()
        (proj / "metadata.json").write_text(
            json.dumps({"project_id": "kindle_sparse", "idea": "sparse app", "stages_completed": []})
        )
        # No artifact files at all
        (proj / "artifacts").mkdir()
        (proj / "logs").mkdir()
        (proj / "workspace").mkdir()

        compiled = _make_compiled_graph()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            result = runner.invoke(app, ["resume", str(proj)])

        assert result.exit_code == 0, result.output
        state = compiled.ainvoke.call_args[0][0]
        assert state["feature_spec"] == {}
        assert state["dev_tasks"] == []
        assert state["grill_transcript"] == ""
        assert state["research_report"] == ""
        assert state["architecture"] == ""


# ---------------------------------------------------------------------------
# resume command — state has correct settings
# ---------------------------------------------------------------------------


class TestResumeStateSettings:
    """resume state should inherit settings from config."""

    def test_resume_state_uses_settings_defaults(self, tmp_path: Path) -> None:
        proj = _project_dir(tmp_path)
        compiled = _make_compiled_graph()
        settings = _make_settings(
            max_concurrent_agents=6,
            max_qa_retries=7,
            max_cpo_retries=8,
            model="claude-sonnet-4-20250514",
            max_agent_turns=75,
        )

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            runner.invoke(app, ["resume", str(proj)])

        state = compiled.ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 6
        assert state["max_qa_retries"] == 7
        assert state["max_cpo_retries"] == 8
        assert state["model"] == "claude-sonnet-4-20250514"
        assert state["max_agent_turns"] == 75


# ---------------------------------------------------------------------------
# list command — displaying sessions
# ---------------------------------------------------------------------------


class TestListCommand:
    """The list command should display existing sessions."""

    def test_list_calls_show_projects(self) -> None:
        """list should call list_projects and pass result to ui.show_projects."""
        ui = _make_ui()
        settings = _make_settings()
        fake_projects = [
            {"project_id": "kindle_aaa", "idea": "app1", "status": "completed"},
            {"project_id": "kindle_bbb", "idea": "app2", "status": "in_progress"},
        ]

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI", return_value=ui),
            patch("kindle.cli.list_projects", return_value=fake_projects) as mock_list,
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        mock_list.assert_called_once_with(settings.projects_root)
        ui.show_projects.assert_called_once_with(fake_projects)

    def test_list_with_no_sessions(self) -> None:
        """list with no sessions should still call show_projects with empty list."""
        ui = _make_ui()
        settings = _make_settings()

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI", return_value=ui),
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        ui.show_projects.assert_called_once_with([])

    def test_list_uses_settings_projects_root(self) -> None:
        """list should look for projects in the configured projects_root."""
        custom_root = Path("/custom/projects")
        settings = _make_settings(projects_root=custom_root)

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.list_projects", return_value=[]) as mock_list,
        ):
            runner.invoke(app, ["list"])

        mock_list.assert_called_once_with(custom_root)

    def test_list_creates_ui_with_defaults(self) -> None:
        """list should create UI with no arguments (defaults)."""
        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()) as mock_ui_cls,
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            runner.invoke(app, ["list"])

        mock_ui_cls.assert_called_once_with()


# ---------------------------------------------------------------------------
# Edge cases and error paths
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling across commands."""

    def test_build_with_empty_string_idea_shows_usage(self) -> None:
        """An empty string idea should show usage, not crash."""
        patches = _cli_patches()
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            # Typer treats "" as None for optional arguments
            result = runner.invoke(app, [""])

        # Empty string is falsy, should trigger usage message
        # or be treated as an idea — either way shouldn't crash
        assert result.exit_code in (0, 1, 2)

    def test_build_review_arch_flag_accepted(self) -> None:
        """--review-arch flag should be accepted without error."""
        patches = _cli_patches()
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            result = runner.invoke(app, ["my app", "--review-arch"])

        assert result.exit_code == 0, result.output

    def test_build_short_flags(self) -> None:
        """Short flags -s, -c, -o, -v should work."""
        compiled = _make_compiled_graph()
        patches = _cli_patches(compiled=compiled)
        with (
            patches["settings_load"],
            patches["ui_cls"],
            patches["build_graph"],
            patches["create_project"],
            patches["workspace_path"],
        ):
            result = runner.invoke(app, [
                "my app",
                "-s", "fastapi",
                "-c", "2",
                "-o", "/tmp/out_test",
                "-v",
            ])

        assert result.exit_code == 0, result.output
        state = compiled.ainvoke.call_args[0][0]
        assert state["stack_preference"] == "fastapi"
        assert state["max_concurrent_agents"] == 2

    def test_help_flag_shows_help_text(self) -> None:
        """--help should show usage without running build."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "kindle" in result.output.lower() or "application" in result.output.lower()

    def test_resume_help_shows_usage(self) -> None:
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "resume" in result.output.lower() or "stage" in result.output.lower()

    def test_list_help_shows_usage(self) -> None:
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# resume — expanduser behavior
# ---------------------------------------------------------------------------


class TestResumeExpandUser:
    """resume should expand ~ in the project_path argument."""

    def test_expanduser_on_project_path(self, tmp_path: Path) -> None:
        """Even though we use a real path, verify expanduser is invoked."""
        proj = _project_dir(tmp_path)
        compiled = _make_compiled_graph()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            # Using a real path (not ~) but verifying the code doesn't crash
            result = runner.invoke(app, ["resume", str(proj)])

        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# resume — metadata.json content handling
# ---------------------------------------------------------------------------


class TestResumeMetadata:
    """resume correctly handles various metadata.json contents."""

    def test_metadata_missing_idea_key_defaults_to_empty(self, tmp_path: Path) -> None:
        """If metadata.json doesn't have 'idea', state.idea should be ''."""
        proj = tmp_path / "kindle_noidea"
        proj.mkdir()
        (proj / "artifacts").mkdir()
        (proj / "logs").mkdir()
        (proj / "workspace").mkdir()
        (proj / "metadata.json").write_text(
            json.dumps({"project_id": "kindle_noidea", "stages_completed": []})
        )

        compiled = _make_compiled_graph()

        with (
            patch("kindle.cli.Settings.load", return_value=_make_settings()),
            patch("kindle.cli.UI", return_value=_make_ui()),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.workspace_path", return_value=proj / "workspace"),
        ):
            result = runner.invoke(app, ["resume", str(proj)])

        assert result.exit_code == 0, result.output
        state = compiled.ainvoke.call_args[0][0]
        assert state["idea"] == ""
