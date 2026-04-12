"""Tests for kindle.cli — Typer CLI entry point (build, resume, list)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from kindle.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> MagicMock:
    """Return a mock Settings with sensible defaults."""
    s = MagicMock()
    s.model = overrides.get("model", "claude-opus-4-20250514")
    s.max_agent_turns = overrides.get("max_agent_turns", 50)
    s.max_concurrent_agents = overrides.get("max_concurrent_agents", 4)
    s.max_qa_retries = overrides.get("max_qa_retries", 10)
    s.max_cpo_retries = overrides.get("max_cpo_retries", 10)
    s.projects_root = overrides.get("projects_root", Path("/fake/projects"))
    return s


def _patch_build_deps(
    tmp_path: Path,
    *,
    idea: str = "build a todo app",
    settings_overrides: dict | None = None,
) -> dict[str, MagicMock]:
    """Return a dict of mocks suitable for patching build()'s dependencies.

    Callers should unpack into ``patch()`` context managers.
    """
    settings = _make_settings(**(settings_overrides or {}))
    project_id = "kindle_abc12345"
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    ws = project_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)

    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value={})

    return {
        "settings": settings,
        "project_id": project_id,
        "project_dir": project_dir,
        "workspace": ws,
        "compiled": compiled,
    }


# ---------------------------------------------------------------------------
# build callback — happy path
# ---------------------------------------------------------------------------


class TestBuildHappyPath:
    """Tests for the default `kindle <idea>` invocation."""

    def test_happy_path_runs_graph(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run") as mock_run,
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["build a todo app"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        # The compiled graph's ainvoke coroutine was passed to asyncio.run
        mocks["compiled"].ainvoke.assert_called_once()

    def test_initial_state_contains_idea_and_defaults(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        captured_state: list[dict] = []

        def _capture_run(coro: object) -> None:
            """No-op asyncio.run that lets us inspect ainvoke's argument."""

        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run", side_effect=_capture_run),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["build a todo app"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["idea"] == "build a todo app"
        assert state["project_id"] == mocks["project_id"]
        assert state["project_dir"] == str(mocks["project_dir"])
        assert state["max_concurrent_agents"] == 4
        assert state["max_qa_retries"] == 10
        assert state["max_cpo_retries"] == 10

    def test_banner_called_with_idea_and_project_id(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["build a todo app"])

        assert result.exit_code == 0
        ui_instance = mock_ui_cls.return_value
        ui_instance.banner.assert_called_once_with("build a todo app", mocks["project_id"])


# ---------------------------------------------------------------------------
# build callback — no idea (early exit)
# ---------------------------------------------------------------------------


class TestBuildNoIdea:
    """When no idea argument is provided, the CLI prints usage and exits 0."""

    def test_no_args_prints_usage(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output or "kindle" in result.output.lower()

    def test_empty_string_idea_prints_usage(self) -> None:
        result = runner.invoke(app, [""])
        assert result.exit_code == 0
        assert "Usage:" in result.output


# ---------------------------------------------------------------------------
# build callback — CLI option overrides
# ---------------------------------------------------------------------------


class TestBuildOptionOverrides:
    """CLI flags --stack, --concurrency, --qa-retries, --cpo-retries override config."""

    def test_stack_preference_propagated(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--stack", "react"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["stack_preference"] == "react"

    def test_concurrency_override(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--concurrency", "8"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 8

    def test_concurrency_zero_falls_back_to_config(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path, settings_overrides={"max_concurrent_agents": 6})
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--concurrency", "0"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 6

    def test_qa_retries_override(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--qa-retries", "3"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 3

    def test_qa_retries_zero_falls_back_to_config(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path, settings_overrides={"max_qa_retries": 7})
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--qa-retries", "0"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_qa_retries"] == 7

    def test_cpo_retries_override(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--cpo-retries", "5"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_cpo_retries"] == 5

    def test_cpo_retries_zero_falls_back_to_config(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path, settings_overrides={"max_cpo_retries": 12})
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--cpo-retries", "0"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_cpo_retries"] == 12


# ---------------------------------------------------------------------------
# build callback — --auto-approve / --verbose flags
# ---------------------------------------------------------------------------


class TestBuildFlagPropagation:
    """--auto-approve and --verbose are forwarded to the UI constructor."""

    def test_auto_approve_passed_to_ui(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--auto-approve"])

        assert result.exit_code == 0
        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=False)

    def test_verbose_passed_to_ui(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--verbose"])

        assert result.exit_code == 0
        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)

    def test_both_flags_together(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--auto-approve", "--verbose"])

        assert result.exit_code == 0
        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=True)


# ---------------------------------------------------------------------------
# build callback — --output copy-to-directory
# ---------------------------------------------------------------------------


class TestBuildOutput:
    """Tests for the --output flag: copy workspace to a user-specified directory."""

    def test_output_copies_workspace(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        dest = tmp_path / "my_output"  # does not exist yet
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
            patch("shutil.copytree") as mock_copy,
        ):
            result = runner.invoke(app, ["my app", "--output", str(dest)])

        assert result.exit_code == 0
        mock_copy.assert_called_once_with(str(mocks["workspace"]), str(dest))

    def test_output_already_exists_shows_error(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        dest = tmp_path / "existing_output"
        dest.mkdir()  # pre-create so it "already exists"
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--output", str(dest)])

        assert result.exit_code == 0
        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        error_msg = ui_instance.error.call_args[0][0]
        assert "already exists" in error_msg

    def test_no_output_flag_skips_copy(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app"])

        assert result.exit_code == 0
        ui_instance = mock_ui_cls.return_value
        # error should not be called since no output flag
        ui_instance.error.assert_not_called()


# ---------------------------------------------------------------------------
# build callback — review-arch flag in initial state
# ---------------------------------------------------------------------------


class TestBuildReviewArch:
    """The --review-arch flag is a CLI option (currently influences UI flow)."""

    def test_review_arch_flag_accepted(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "--review-arch"])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# resume command — happy path
# ---------------------------------------------------------------------------


class TestResumeHappyPath:
    """Tests for `kindle resume <project_path>`."""

    def test_resume_loads_artifacts_and_runs_graph(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "kindle_abc12345"
        project_dir.mkdir()
        meta = {"project_id": "kindle_abc12345", "idea": "my idea"}
        (project_dir / "metadata.json").write_text(json.dumps(meta))

        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        ws = project_dir / "workspace"
        ws.mkdir()

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=ws),
            patch("kindle.cli.load_artifact", return_value=None),
        ):
            result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 0
        ui_instance = mock_ui_cls.return_value
        ui_instance.banner.assert_called_once_with("my idea", "kindle_abc12345")

    def test_resume_from_stage_forwarded_to_build_graph(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "kindle_abc12345"
        project_dir.mkdir()
        meta = {"project_id": "kindle_abc12345", "idea": "my idea"}
        (project_dir / "metadata.json").write_text(json.dumps(meta))

        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        ws = project_dir / "workspace"
        ws.mkdir()

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=compiled) as mock_bg,
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=ws),
            patch("kindle.cli.load_artifact", return_value=None),
        ):
            result = runner.invoke(
                app, ["resume", str(project_dir), "--from", "architect"]
            )

        assert result.exit_code == 0
        mock_bg.assert_called_once()
        _, kwargs = mock_bg.call_args
        assert kwargs["entry_stage"] == "architect"

    def test_resume_reloads_feature_spec_artifact(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "kindle_abc12345"
        project_dir.mkdir()
        meta = {"project_id": "kindle_abc12345", "idea": "my idea"}
        (project_dir / "metadata.json").write_text(json.dumps(meta))

        feature_spec = {"features": ["auth", "dashboard"]}

        def _fake_load(proj_dir: str, name: str) -> str | None:
            if name == "feature_spec.json":
                return json.dumps(feature_spec)
            if name == "dev_tasks.json":
                return json.dumps([{"task": "build auth"}])
            if name == "grill_transcript.md":
                return "transcript"
            if name == "research_report.md":
                return "research"
            if name == "architecture.md":
                return "arch doc"
            if name == "qa_report.md":
                return "qa ok"
            if name == "product_audit.md":
                return "audit ok"
            return None

        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        ws = project_dir / "workspace"
        ws.mkdir()

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=ws),
            patch("kindle.cli.load_artifact", side_effect=_fake_load),
        ):
            result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 0
        state = compiled.ainvoke.call_args[0][0]
        assert state["feature_spec"] == feature_spec
        assert state["dev_tasks"] == [{"task": "build auth"}]
        assert state["grill_transcript"] == "transcript"
        assert state["research_report"] == "research"
        assert state["architecture"] == "arch doc"
        assert state["qa_report"] == "qa ok"
        assert state["product_audit"] == "audit ok"

    def test_resume_auto_approve_and_verbose_passed_to_ui(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "kindle_abc12345"
        project_dir.mkdir()
        meta = {"project_id": "kindle_abc12345", "idea": "my idea"}
        (project_dir / "metadata.json").write_text(json.dumps(meta))

        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        ws = project_dir / "workspace"
        ws.mkdir()

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=ws),
            patch("kindle.cli.load_artifact", return_value=None),
        ):
            result = runner.invoke(
                app, ["resume", str(project_dir), "--auto-approve", "--verbose"]
            )

        assert result.exit_code == 0
        mock_ui_cls.assert_called_once_with(auto_approve=True, verbose=True)

    def test_resume_null_artifacts_produce_empty_defaults(self, tmp_path: Path) -> None:
        """When load_artifact returns None for every file, state uses empty defaults."""
        project_dir = tmp_path / "kindle_abc12345"
        project_dir.mkdir()
        meta = {"project_id": "kindle_abc12345", "idea": "my idea"}
        (project_dir / "metadata.json").write_text(json.dumps(meta))

        settings = _make_settings()
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(return_value={})
        ws = project_dir / "workspace"
        ws.mkdir()

        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI"),
            patch("kindle.cli.build_graph", return_value=compiled),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=ws),
            patch("kindle.cli.load_artifact", return_value=None),
        ):
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
# resume command — error paths
# ---------------------------------------------------------------------------


class TestResumeErrors:
    """Error handling for the resume command."""

    def test_missing_directory_exits_with_error(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        settings = _make_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
        ):
            result = runner.invoke(app, ["resume", str(nonexistent)])

        assert result.exit_code == 1
        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        error_msg = ui_instance.error.call_args[0][0]
        assert "not found" in error_msg

    def test_missing_metadata_json_exits_with_error(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "kindle_no_meta"
        project_dir.mkdir()
        # No metadata.json created
        settings = _make_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
        ):
            result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 1
        ui_instance = mock_ui_cls.return_value
        ui_instance.error.assert_called_once()
        error_msg = ui_instance.error.call_args[0][0]
        assert "metadata.json" in error_msg


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListCommand:
    """Tests for `kindle list`."""

    def test_list_delegates_to_list_projects_and_show(self) -> None:
        settings = _make_settings()
        fake_projects = [{"project_id": "kindle_aaa", "idea": "app 1"}]
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.list_projects", return_value=fake_projects) as mock_lp,
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        mock_lp.assert_called_once_with(settings.projects_root)
        ui_instance = mock_ui_cls.return_value
        ui_instance.show_projects.assert_called_once_with(fake_projects)

    def test_list_empty_projects(self) -> None:
        settings = _make_settings()
        with (
            patch("kindle.cli.Settings.load", return_value=settings),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch("kindle.cli.list_projects", return_value=[]),
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        ui_instance = mock_ui_cls.return_value
        ui_instance.show_projects.assert_called_once_with([])


# ---------------------------------------------------------------------------
# build — model / max_agent_turns propagated from settings
# ---------------------------------------------------------------------------


class TestBuildSettingsPropagation:
    """Settings fields (model, max_agent_turns) are forwarded into initial state."""

    def test_model_and_max_turns_from_settings(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(
            tmp_path,
            settings_overrides={"model": "claude-sonnet", "max_agent_turns": 25},
        )
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["model"] == "claude-sonnet"
        assert state["max_agent_turns"] == 25


# ---------------------------------------------------------------------------
# build — workspace / session info messages
# ---------------------------------------------------------------------------


class TestBuildInfoMessages:
    """After build completes, UI.info is called with workspace and session paths."""

    def test_info_messages_printed(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app"])

        assert result.exit_code == 0
        ui_instance = mock_ui_cls.return_value
        info_calls = [c[0][0] for c in ui_instance.info.call_args_list]
        assert any("built at" in msg.lower() or str(mocks["workspace"]) in msg for msg in info_calls)
        assert any("artifacts" in msg.lower() or str(mocks["project_dir"]) in msg for msg in info_calls)


# ---------------------------------------------------------------------------
# build — short flag aliases
# ---------------------------------------------------------------------------


class TestBuildShortFlags:
    """Short flag aliases -s, -c, -o, -v work correctly."""

    def test_short_stack_flag(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "-s", "fastapi"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["stack_preference"] == "fastapi"

    def test_short_concurrency_flag(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI"),
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "-c", "2"])

        assert result.exit_code == 0
        state = mocks["compiled"].ainvoke.call_args[0][0]
        assert state["max_concurrent_agents"] == 2

    def test_short_verbose_flag(self, tmp_path: Path) -> None:
        mocks = _patch_build_deps(tmp_path)
        with (
            patch("kindle.cli.Settings.load", return_value=mocks["settings"]),
            patch("kindle.cli.UI") as mock_ui_cls,
            patch(
                "kindle.cli.create_project",
                return_value=(mocks["project_id"], mocks["project_dir"]),
            ),
            patch("kindle.cli.build_graph", return_value=mocks["compiled"]),
            patch("kindle.cli.asyncio.run"),
            patch("kindle.cli.workspace_path", return_value=mocks["workspace"]),
        ):
            result = runner.invoke(app, ["my app", "-v"])

        assert result.exit_code == 0
        mock_ui_cls.assert_called_once_with(auto_approve=False, verbose=True)
