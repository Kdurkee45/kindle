"""Comprehensive tests for kindle.artifacts — the persistent artifact layer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from kindle.artifacts import (
    create_project,
    list_projects,
    load_artifact,
    mark_project_done,
    mark_stage_complete,
    save_artifact,
    save_log,
    workspace_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def projects_root(tmp_path: Path) -> Path:
    """Return an isolated projects root directory."""
    root = tmp_path / "projects"
    root.mkdir()
    return root


@pytest.fixture()
def project(projects_root: Path) -> tuple[str, Path]:
    """Create a single project and return (project_id, project_dir)."""
    return create_project(projects_root, "Build a weather app")


# ---------------------------------------------------------------------------
# create_project
# ---------------------------------------------------------------------------


class TestCreateProject:
    def test_returns_id_and_dir(self, projects_root: Path) -> None:
        project_id, project_dir = create_project(projects_root, "test idea")
        assert isinstance(project_id, str)
        assert isinstance(project_dir, Path)

    def test_id_format(self, projects_root: Path) -> None:
        project_id, _ = create_project(projects_root, "test idea")
        assert project_id.startswith("kindle_")
        # 8 hex chars after the prefix
        hex_part = project_id.removeprefix("kindle_")
        assert len(hex_part) == 8
        int(hex_part, 16)  # should not raise

    def test_unique_ids(self, projects_root: Path) -> None:
        ids = {create_project(projects_root, "idea")[0] for _ in range(20)}
        assert len(ids) == 20

    def test_creates_artifacts_dir(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        assert (project_dir / "artifacts").is_dir()

    def test_creates_logs_dir(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        assert (project_dir / "logs").is_dir()

    def test_creates_metadata_json(self, project: tuple[str, Path]) -> None:
        project_id, project_dir = project
        meta_path = project_dir / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["project_id"] == project_id
        assert meta["idea"] == "Build a weather app"
        assert meta["status"] == "in_progress"
        assert meta["stages_completed"] == []
        assert "created_at" in meta

    def test_metadata_created_at_is_iso(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        meta = json.loads((project_dir / "metadata.json").read_text())
        # Should parse without error — ISO 8601
        from datetime import datetime

        datetime.fromisoformat(meta["created_at"])

    def test_project_dir_lives_under_root(self, projects_root: Path) -> None:
        _, project_dir = create_project(projects_root, "idea")
        assert project_dir.parent == projects_root

    def test_creates_root_parents_if_needed(self, tmp_path: Path) -> None:
        deep_root = tmp_path / "a" / "b" / "c"
        # deep_root doesn't exist yet — create_project must create it via mkdir(parents=True)
        _pid, pdir = create_project(deep_root, "deep idea")
        assert pdir.exists()
        assert (pdir / "artifacts").is_dir()


# ---------------------------------------------------------------------------
# workspace_path
# ---------------------------------------------------------------------------


class TestWorkspacePath:
    def test_creates_workspace_dir(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        ws = workspace_path(project_dir)
        assert ws.is_dir()
        assert ws == project_dir / "workspace"

    def test_idempotent(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        ws1 = workspace_path(project_dir)
        ws2 = workspace_path(project_dir)
        assert ws1 == ws2
        assert ws1.is_dir()

    def test_accepts_string_path(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        ws = workspace_path(str(project_dir))
        assert isinstance(ws, Path)
        assert ws.is_dir()


# ---------------------------------------------------------------------------
# save_artifact / load_artifact round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadArtifact:
    def test_round_trip(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        save_artifact(project_dir, "spec.md", "# Spec\nHello")
        content = load_artifact(project_dir, "spec.md")
        assert content == "# Spec\nHello"

    def test_returns_path(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        p = save_artifact(project_dir, "plan.txt", "the plan")
        assert isinstance(p, Path)
        assert p.exists()
        assert p.name == "plan.txt"
        assert p.parent.name == "artifacts"

    def test_overwrite(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        save_artifact(project_dir, "data.json", '{"v":1}')
        save_artifact(project_dir, "data.json", '{"v":2}')
        assert load_artifact(project_dir, "data.json") == '{"v":2}'

    def test_load_missing_returns_none(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        assert load_artifact(project_dir, "nonexistent.txt") is None

    def test_nested_artifact_name(self, project: tuple[str, Path]) -> None:
        """Artifact name with slashes should create subdirectories."""
        _, project_dir = project
        save_artifact(project_dir, "sub/deep/file.txt", "nested content")
        assert load_artifact(project_dir, "sub/deep/file.txt") == "nested content"

    def test_empty_content(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        save_artifact(project_dir, "empty.txt", "")
        assert load_artifact(project_dir, "empty.txt") == ""

    def test_unicode_content(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        text = "こんにちは 🌍 café"
        save_artifact(project_dir, "unicode.txt", text)
        assert load_artifact(project_dir, "unicode.txt") == text

    def test_accepts_string_path(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        save_artifact(str(project_dir), "via_str.txt", "hello")
        assert load_artifact(str(project_dir), "via_str.txt") == "hello"

    def test_large_content(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        big = "x" * 100_000
        save_artifact(project_dir, "big.txt", big)
        assert load_artifact(project_dir, "big.txt") == big


# ---------------------------------------------------------------------------
# save_log
# ---------------------------------------------------------------------------


class TestSaveLog:
    def test_creates_log_file(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        p = save_log(project_dir, "build", "compiled successfully")
        assert p.exists()
        assert p.name == "build.log"
        assert p.parent.name == "logs"

    def test_log_contains_content(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        save_log(project_dir, "test", "all 42 tests passed")
        text = (project_dir / "logs" / "test.log").read_text()
        assert "all 42 tests passed" in text

    def test_log_contains_timestamp_header(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        save_log(project_dir, "deploy", "deployed v1")
        text = (project_dir / "logs" / "deploy.log").read_text()
        # Timestamp format: --- [YYYY-MM-DD HH:MM:SS] ---
        assert "--- [" in text
        assert "] ---" in text

    def test_append_behavior(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        save_log(project_dir, "multi", "first entry")
        save_log(project_dir, "multi", "second entry")
        save_log(project_dir, "multi", "third entry")
        text = (project_dir / "logs" / "multi.log").read_text()
        assert "first entry" in text
        assert "second entry" in text
        assert "third entry" in text
        # All three timestamp headers should be present
        assert text.count("--- [") == 3

    def test_different_stages_separate_files(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        save_log(project_dir, "build", "build output")
        save_log(project_dir, "test", "test output")
        assert (project_dir / "logs" / "build.log").exists()
        assert (project_dir / "logs" / "test.log").exists()
        assert "test output" not in (project_dir / "logs" / "build.log").read_text()

    def test_accepts_string_path(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        p = save_log(str(project_dir), "str_stage", "content")
        assert p.exists()


# ---------------------------------------------------------------------------
# mark_stage_complete
# ---------------------------------------------------------------------------


class TestMarkStageComplete:
    def test_appends_stage(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_stage_complete(project_dir, "ideation")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "ideation" in meta["stages_completed"]

    def test_multiple_stages(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_stage_complete(project_dir, "ideation")
        mark_stage_complete(project_dir, "design")
        mark_stage_complete(project_dir, "build")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["stages_completed"] == ["ideation", "design", "build"]

    def test_idempotent_no_duplicates(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_stage_complete(project_dir, "ideation")
        mark_stage_complete(project_dir, "ideation")
        mark_stage_complete(project_dir, "ideation")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["stages_completed"].count("ideation") == 1

    def test_sets_last_updated(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        meta_before = json.loads((project_dir / "metadata.json").read_text())
        assert "last_updated" not in meta_before

        mark_stage_complete(project_dir, "ideation")
        meta_after = json.loads((project_dir / "metadata.json").read_text())
        assert "last_updated" in meta_after
        # Should parse as ISO datetime
        from datetime import datetime

        datetime.fromisoformat(meta_after["last_updated"])

    def test_preserves_other_metadata(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        meta_before = json.loads((project_dir / "metadata.json").read_text())
        original_idea = meta_before["idea"]
        original_id = meta_before["project_id"]

        mark_stage_complete(project_dir, "build")
        meta_after = json.loads((project_dir / "metadata.json").read_text())
        assert meta_after["idea"] == original_idea
        assert meta_after["project_id"] == original_id
        assert meta_after["status"] == "in_progress"

    def test_accepts_string_path(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_stage_complete(str(project_dir), "test_stage")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "test_stage" in meta["stages_completed"]


# ---------------------------------------------------------------------------
# mark_project_done
# ---------------------------------------------------------------------------


class TestMarkProjectDone:
    def test_sets_status_completed(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_project_done(project_dir)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "completed"

    def test_sets_completed_at(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_project_done(project_dir)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "completed_at" in meta
        from datetime import datetime

        datetime.fromisoformat(meta["completed_at"])

    def test_preserves_stages_completed(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_stage_complete(project_dir, "ideation")
        mark_stage_complete(project_dir, "design")
        mark_project_done(project_dir)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["stages_completed"] == ["ideation", "design"]
        assert meta["status"] == "completed"

    def test_preserves_idea_and_id(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        meta_before = json.loads((project_dir / "metadata.json").read_text())
        mark_project_done(project_dir)
        meta_after = json.loads((project_dir / "metadata.json").read_text())
        assert meta_after["project_id"] == meta_before["project_id"]
        assert meta_after["idea"] == meta_before["idea"]

    def test_idempotent(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_project_done(project_dir)
        mark_project_done(project_dir)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "completed"

    def test_accepts_string_path(self, project: tuple[str, Path]) -> None:
        _, project_dir = project
        mark_project_done(str(project_dir))
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "completed"


# ---------------------------------------------------------------------------
# list_projects
# ---------------------------------------------------------------------------


class TestListProjects:
    def test_empty_root(self, projects_root: Path) -> None:
        assert list_projects(projects_root) == []

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        assert list_projects(missing) == []

    def test_single_project(self, projects_root: Path) -> None:
        create_project(projects_root, "solo project")
        result = list_projects(projects_root)
        assert len(result) == 1
        assert result[0]["idea"] == "solo project"

    def test_multiple_projects(self, projects_root: Path) -> None:
        create_project(projects_root, "first")
        create_project(projects_root, "second")
        create_project(projects_root, "third")
        result = list_projects(projects_root)
        assert len(result) == 3

    def test_returns_metadata_dicts(self, projects_root: Path) -> None:
        create_project(projects_root, "check fields")
        result = list_projects(projects_root)
        meta = result[0]
        assert "project_id" in meta
        assert "idea" in meta
        assert "created_at" in meta
        assert "status" in meta
        assert "stages_completed" in meta

    def test_ignores_dirs_without_metadata(self, projects_root: Path) -> None:
        """Directories that aren't projects (no metadata.json) should be skipped."""
        create_project(projects_root, "real project")
        (projects_root / "random_dir").mkdir()
        (projects_root / "another_dir").mkdir()
        result = list_projects(projects_root)
        assert len(result) == 1
        assert result[0]["idea"] == "real project"

    def test_sorted_reverse_by_dir_name(self, projects_root: Path) -> None:
        """Projects are sorted by directory name in reverse (most recent first).

        Since uuid-based names are not ordered by time, we create projects
        with known directory names to verify sort order.
        """
        # Manually create project dirs with lexicographically ordered names
        for name in ["kindle_aaaaaaaa", "kindle_mmmmmmmm", "kindle_zzzzzzzz"]:
            d = projects_root / name
            (d / "artifacts").mkdir(parents=True)
            (d / "logs").mkdir(parents=True)
            meta = {
                "project_id": name,
                "idea": f"idea for {name}",
                "created_at": "2025-01-01T00:00:00+00:00",
                "status": "in_progress",
                "stages_completed": [],
            }
            (d / "metadata.json").write_text(json.dumps(meta))

        result = list_projects(projects_root)
        ids = [m["project_id"] for m in result]
        # reverse sorted: z > m > a
        assert ids == ["kindle_zzzzzzzz", "kindle_mmmmmmmm", "kindle_aaaaaaaa"]

    def test_includes_completed_projects(self, projects_root: Path) -> None:
        _, pdir = create_project(projects_root, "will complete")
        mark_project_done(pdir)
        result = list_projects(projects_root)
        assert len(result) == 1
        assert result[0]["status"] == "completed"


# ---------------------------------------------------------------------------
# Integration / end-to-end scenario
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_project_lifecycle(self, projects_root: Path) -> None:
        """Simulate a complete project lifecycle through all artifact functions."""
        # 1. Create project
        pid, pdir = create_project(projects_root, "Build a CLI tool")
        assert pdir.is_dir()

        # 2. Get workspace
        ws = workspace_path(pdir)
        assert ws.is_dir()

        # 3. Save artifacts from different stages
        save_artifact(pdir, "spec.md", "# Specification\nBuild a CLI")
        save_artifact(pdir, "design.json", '{"components": ["parser", "runner"]}')

        # 4. Log stage outputs
        save_log(pdir, "ideation", "Generated project specification")
        save_log(pdir, "design", "Designed component architecture")

        # 5. Mark stages complete
        mark_stage_complete(pdir, "ideation")
        mark_stage_complete(pdir, "design")
        mark_stage_complete(pdir, "build")

        # 6. Verify intermediate state
        meta = json.loads((pdir / "metadata.json").read_text())
        assert meta["status"] == "in_progress"
        assert meta["stages_completed"] == ["ideation", "design", "build"]

        # 7. Load artifacts back
        assert load_artifact(pdir, "spec.md") == "# Specification\nBuild a CLI"
        assert load_artifact(pdir, "nonexistent") is None

        # 8. Mark project done
        mark_project_done(pdir)
        meta = json.loads((pdir / "metadata.json").read_text())
        assert meta["status"] == "completed"
        assert "completed_at" in meta

        # 9. List projects
        projects = list_projects(projects_root)
        assert len(projects) == 1
        assert projects[0]["project_id"] == pid
        assert projects[0]["status"] == "completed"

    def test_multiple_projects_isolation(self, projects_root: Path) -> None:
        """Artifacts from one project should not leak into another."""
        _, dir_a = create_project(projects_root, "Project A")
        _, dir_b = create_project(projects_root, "Project B")

        save_artifact(dir_a, "secret.txt", "A's secret")
        save_artifact(dir_b, "secret.txt", "B's secret")

        assert load_artifact(dir_a, "secret.txt") == "A's secret"
        assert load_artifact(dir_b, "secret.txt") == "B's secret"

        mark_stage_complete(dir_a, "build")
        meta_a = json.loads((dir_a / "metadata.json").read_text())
        meta_b = json.loads((dir_b / "metadata.json").read_text())
        assert "build" in meta_a["stages_completed"]
        assert "build" not in meta_b["stages_completed"]
