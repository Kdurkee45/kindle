"""Tests for kindle.stages.package — packaging and delivery stage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.stages.package import SYSTEM_PROMPT, package_node
from tests.constants import SAMPLE_FEATURE_SPEC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_ARCHITECTURE = """\
# Architecture

## Stack
- React 18.x frontend
- Express 4.x backend

## Structure
src/
  components/
  api/
"""

SAMPLE_DEV_TASKS = [
    {"id": 1, "title": "Setup project scaffolding", "status": "done"},
    {"id": 2, "title": "Implement auth module", "status": "done"},
    {"id": 3, "title": "Build task CRUD API", "status": "done"},
]

SAMPLE_README = """\
# TaskFlow

A modern task management app built with React and Express.

## Getting Started

```bash
npm install
npm start
```

## Tech Stack
- React 18.x
- Express 4.x
- Prisma ORM

## Project Structure
```
src/
  components/
  api/
```

## License
MIT
"""


@pytest.fixture
def package_state(make_state):
    """Factory with package-stage defaults pre-applied."""

    def _factory(**overrides):
        defaults = {
            "feature_spec": SAMPLE_FEATURE_SPEC,
            "architecture": SAMPLE_ARCHITECTURE,
            "dev_tasks": SAMPLE_DEV_TASKS,
        }
        defaults.update(overrides)
        return make_state(metadata_extra={"status": "in_progress"}, **defaults)

    return _factory


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestPackageHappyPath:
    """Agent generates README.md in workspace, it is read and saved as artifact."""

    @pytest.mark.asyncio
    async def test_readme_read_from_workspace_and_saved_as_artifact(
        self, tmp_path: Path, package_state, make_ui
    ) -> None:
        """When agent writes README.md the file contents become the artifact."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text(SAMPLE_README)
            return MagicMock(text="agent output")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        assert result["package_readme"] == SAMPLE_README
        # Artifact should be persisted on disk
        artifact_path = Path(state["project_dir"]) / "artifacts" / "package_readme.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == SAMPLE_README

    @pytest.mark.asyncio
    async def test_readme_is_string_type(self, tmp_path: Path, package_state, make_ui) -> None:
        """package_readme in result is always a string."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text(SAMPLE_README)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        assert isinstance(result["package_readme"], str)

    @pytest.mark.asyncio
    async def test_artifact_content_matches_workspace_file(self, tmp_path: Path, package_state, make_ui) -> None:
        """The saved artifact exactly matches what the agent wrote to workspace."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        custom_readme = "# Custom\n\nSpecial content with unicode: café ñ\n"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text(custom_readme)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        artifact_path = Path(state["project_dir"]) / "artifacts" / "package_readme.md"
        assert artifact_path.read_text() == custom_readme
        assert result["package_readme"] == custom_readme


# ---------------------------------------------------------------------------
# Missing README fallback
# ---------------------------------------------------------------------------


class TestMissingReadmeFallback:
    """Graceful handling when agent does not generate README.md."""

    @pytest.mark.asyncio
    async def test_empty_string_when_no_readme(self, tmp_path: Path, package_state, make_ui) -> None:
        """If the agent never writes README.md, package_readme is empty string."""
        state = package_state()
        ui = make_ui()

        async def fake_run_agent(**kwargs):
            # Intentionally do NOT write README.md
            return MagicMock(text="agent output without file")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        assert result["package_readme"] == ""

    @pytest.mark.asyncio
    async def test_no_artifact_saved_when_readme_missing(self, tmp_path: Path, package_state, make_ui) -> None:
        """No artifact is written when README.md is not found."""
        state = package_state()
        ui = make_ui()

        async def fake_run_agent(**kwargs):
            return MagicMock(text="no file")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        artifact_path = Path(state["project_dir"]) / "artifacts" / "package_readme.md"
        assert not artifact_path.exists()

    @pytest.mark.asyncio
    async def test_stage_still_completes_without_readme(self, tmp_path: Path, package_state, make_ui) -> None:
        """Even without README.md, the stage lifecycle completes normally."""
        state = package_state()
        ui = make_ui()

        async def fake_run_agent(**kwargs):
            return MagicMock(text="no file")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        assert result["current_stage"] == "package"
        ui.stage_done.assert_called_once_with("package")
        ui.deploy_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_project_marked_done_without_readme(self, tmp_path: Path, package_state, make_ui) -> None:
        """mark_project_done is still called even if README.md was not generated."""
        state = package_state()
        ui = make_ui()

        async def fake_run_agent(**kwargs):
            return MagicMock(text="no file")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["status"] == "completed"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify the user prompt sent to run_agent contains all upstream artifacts."""

    @pytest.mark.asyncio
    async def test_prompt_includes_idea(self, tmp_path: Path, package_state, make_ui) -> None:
        """The user's original idea appears in the prompt."""
        state = package_state(idea="a social media dashboard")
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "a social media dashboard" in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_prompt_includes_feature_spec_json(self, tmp_path: Path, package_state, make_ui) -> None:
        """The full feature spec is serialized as JSON in the prompt."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        prompt = call_kwargs["user_prompt"]
        assert json.dumps(SAMPLE_FEATURE_SPEC, indent=2) in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_architecture(self, tmp_path: Path, package_state, make_ui) -> None:
        """The architecture text appears in the prompt."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        prompt = call_kwargs["user_prompt"]
        assert "ARCHITECTURE:" in prompt
        assert SAMPLE_ARCHITECTURE in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_dev_tasks_json(self, tmp_path: Path, package_state, make_ui) -> None:
        """The dev tasks are serialized as JSON in the prompt."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        prompt = call_kwargs["user_prompt"]
        assert "DEV TASKS:" in prompt
        assert json.dumps(SAMPLE_DEV_TASKS, indent=2) in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_packaging_instructions(self, tmp_path: Path, package_state, make_ui) -> None:
        """The prompt instructs the agent to generate README, git init, etc."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        prompt = call_kwargs["user_prompt"]
        assert "README.md" in prompt
        assert "atomic commits" in prompt
        assert "Dockerfile" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_is_devops_engineer(self, tmp_path: Path, package_state, make_ui) -> None:
        """The system prompt sent to run_agent is the SYSTEM_PROMPT constant."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["system_prompt"] == SYSTEM_PROMPT
        assert call_kwargs["persona"] == "Principal DevOps Engineer"


# ---------------------------------------------------------------------------
# Missing optional state
# ---------------------------------------------------------------------------


class TestMissingOptionalState:
    """Graceful handling when optional keys are absent from state."""

    @pytest.mark.asyncio
    async def test_missing_feature_spec_defaults_to_empty_dict(self, tmp_path: Path, package_state, make_ui) -> None:
        """feature_spec defaults to {} when absent."""
        state = package_state()
        del state["feature_spec"]
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert json.dumps({}, indent=2) in call_kwargs["user_prompt"]
        assert result["package_readme"] == "readme"

    @pytest.mark.asyncio
    async def test_missing_architecture_defaults_to_empty_string(self, tmp_path: Path, package_state, make_ui) -> None:
        """architecture defaults to '' when absent."""
        state = package_state()
        del state["architecture"]
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "ARCHITECTURE:\n" in call_kwargs["user_prompt"]
        assert result["package_readme"] == "readme"

    @pytest.mark.asyncio
    async def test_missing_dev_tasks_defaults_to_empty_list(self, tmp_path: Path, package_state, make_ui) -> None:
        """dev_tasks defaults to [] when absent."""
        state = package_state()
        del state["dev_tasks"]
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert json.dumps([], indent=2) in call_kwargs["user_prompt"]
        assert result["package_readme"] == "readme"

    @pytest.mark.asyncio
    async def test_missing_idea_defaults_to_empty_string(self, tmp_path: Path, package_state, make_ui) -> None:
        """idea defaults to '' when absent."""
        state = package_state()
        del state["idea"]
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert "IDEA: " in call_kwargs["user_prompt"]
        assert result["package_readme"] == "readme"


# ---------------------------------------------------------------------------
# State return shape
# ---------------------------------------------------------------------------


class TestStateReturn:
    """Verify the dict returned by package_node has the correct keys/values."""

    @pytest.mark.asyncio
    async def test_return_keys(self, tmp_path: Path, package_state, make_ui) -> None:
        """Returned dict contains exactly package_readme and current_stage."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text(SAMPLE_README)
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        assert set(result.keys()) == {"package_readme", "current_stage"}

    @pytest.mark.asyncio
    async def test_current_stage_is_package(self, tmp_path: Path, package_state, make_ui) -> None:
        """current_stage is always 'package'."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        assert result["current_stage"] == "package"

    @pytest.mark.asyncio
    async def test_package_readme_content_matches_file(self, tmp_path: Path, package_state, make_ui) -> None:
        """The package_readme value is the exact content from the workspace file."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        expected = "# Detailed README\n\nSome thorough docs here.\n"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text(expected)
            return MagicMock(text="ignored")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            result = await package_node(state, ui)

        assert result["package_readme"] == expected


# ---------------------------------------------------------------------------
# Project completion lifecycle (mark_project_done)
# ---------------------------------------------------------------------------


class TestProjectCompletion:
    """Verify mark_project_done is called and metadata is updated."""

    @pytest.mark.asyncio
    async def test_mark_project_done_sets_completed_status(self, tmp_path: Path, package_state, make_ui) -> None:
        """mark_project_done writes status='completed' to metadata.json."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["status"] == "completed"

    @pytest.mark.asyncio
    async def test_mark_project_done_sets_completed_at(self, tmp_path: Path, package_state, make_ui) -> None:
        """mark_project_done writes a completed_at timestamp to metadata.json."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "completed_at" in meta

    @pytest.mark.asyncio
    async def test_mark_stage_complete_records_package(self, tmp_path: Path, package_state, make_ui) -> None:
        """mark_stage_complete writes 'package' to metadata.json stages_completed."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "package" in meta["stages_completed"]


# ---------------------------------------------------------------------------
# deploy_complete UI call
# ---------------------------------------------------------------------------


class TestDeployComplete:
    """Verify ui.deploy_complete is called with the correct workspace path."""

    @pytest.mark.asyncio
    async def test_deploy_complete_called_with_workspace_path(self, tmp_path: Path, package_state, make_ui) -> None:
        """ui.deploy_complete receives the workspace path as a string."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        ui.deploy_complete.assert_called_once_with(str(ws))

    @pytest.mark.asyncio
    async def test_deploy_complete_called_once(self, tmp_path: Path, package_state, make_ui) -> None:
        """ui.deploy_complete is called exactly once."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        assert ui.deploy_complete.call_count == 1


# ---------------------------------------------------------------------------
# Stage lifecycle (UI + artifact calls ordering)
# ---------------------------------------------------------------------------


class TestStageLifecycle:
    """Verify ui.stage_start, mark_stage_complete, mark_project_done, ui.stage_done ordering."""

    @pytest.mark.asyncio
    async def test_ui_stage_start_called(self, tmp_path: Path, package_state, make_ui) -> None:
        """ui.stage_start('package') is called before the agent runs."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        ui.stage_start.assert_called_once_with("package")

    @pytest.mark.asyncio
    async def test_ui_stage_done_called(self, tmp_path: Path, package_state, make_ui) -> None:
        """ui.stage_done('package') is called at the end."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        ui.stage_done.assert_called_once_with("package")

    @pytest.mark.asyncio
    async def test_lifecycle_ordering(self, tmp_path: Path, package_state, make_ui) -> None:
        """stage_start → mark_stage_complete → mark_project_done → deploy_complete → stage_done."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"
        call_order: list[str] = []

        ui.stage_start = MagicMock(side_effect=lambda s: call_order.append("stage_start"))
        ui.stage_done = MagicMock(side_effect=lambda s: call_order.append("stage_done"))
        ui.deploy_complete = MagicMock(side_effect=lambda p: call_order.append("deploy_complete"))

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with (
            patch("kindle.stages.package.run_agent", mock_agent),
            patch(
                "kindle.stages.package.mark_stage_complete",
                side_effect=lambda *a: call_order.append("mark_stage_complete"),
            ),
            patch(
                "kindle.stages.package.mark_project_done",
                side_effect=lambda *a: call_order.append("mark_project_done"),
            ),
        ):
            await package_node(state, ui)

        assert call_order == [
            "stage_start",
            "mark_stage_complete",
            "mark_project_done",
            "deploy_complete",
            "stage_done",
        ]


# ---------------------------------------------------------------------------
# Agent call arguments
# ---------------------------------------------------------------------------


class TestAgentCallArgs:
    """Verify the keyword arguments passed to run_agent."""

    @pytest.mark.asyncio
    async def test_agent_called_with_correct_stage(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent receives stage='package'."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["stage"] == "package"

    @pytest.mark.asyncio
    async def test_agent_called_with_cwd_as_workspace(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent cwd is the workspace directory."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["cwd"] == str(ws)

    @pytest.mark.asyncio
    async def test_model_passed_through_from_state(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent receives model from state."""
        state = package_state(model="claude-sonnet-4-20250514")
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_model_none_when_absent_from_state(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent receives model=None when state has no model key."""
        state = package_state()
        state.pop("model", None)
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["model"] is None

    @pytest.mark.asyncio
    async def test_max_turns_from_state(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent receives max_turns from state's max_agent_turns."""
        state = package_state(max_agent_turns=75)
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["max_turns"] == 75

    @pytest.mark.asyncio
    async def test_max_turns_defaults_to_50(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent receives max_turns=50 when max_agent_turns is absent from state."""
        state = package_state()
        state.pop("max_agent_turns", None)
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["max_turns"] == 50

    @pytest.mark.asyncio
    async def test_agent_called_exactly_once(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent is invoked exactly once per package_node call."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        assert mock_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_agent_receives_ui(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent receives the UI instance."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["ui"] is ui

    @pytest.mark.asyncio
    async def test_agent_receives_project_dir(self, tmp_path: Path, package_state, make_ui) -> None:
        """run_agent receives the project_dir from state."""
        state = package_state()
        ui = make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text("readme")
            return MagicMock(text="")

        mock_agent = AsyncMock(side_effect=fake_run_agent)
        with patch("kindle.stages.package.run_agent", mock_agent):
            await package_node(state, ui)

        call_kwargs = mock_agent.call_args_list[0].kwargs
        assert call_kwargs["project_dir"] == state["project_dir"]
