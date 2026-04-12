"""Tests for kindle.artifacts — Persistent artifact layer for build projects."""

from __future__ import annotations

import json
from pathlib import Path

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
# create_project
# ---------------------------------------------------------------------------


class TestCreateProject:
    """Tests for project directory scaffolding and metadata initialisation."""

    def test_returns_project_id_and_dir(self, tmp_path: Path) -> None:
        project_id, project_dir = create_project(tmp_path, "my idea")
        assert isinstance(project_id, str)
        assert isinstance(project_dir, Path)

    def test_project_id_has_kindle_prefix(self, tmp_path: Path) -> None:
        project_id, _ = create_project(tmp_path, "idea")
        assert project_id.startswith("kindle_")

    def test_project_id_is_correct_length(self, tmp_path: Path) -> None:
        """Format is kindle_ + 8 hex chars → 15 chars total."""
        project_id, _ = create_project(tmp_path, "idea")
        hex_part = project_id.removeprefix("kindle_")
        assert len(hex_part) == 8
        # Verify it's valid hex
        int(hex_part, 16)

    def test_unique_project_ids(self, tmp_path: Path) -> None:
        """Two calls must produce distinct project IDs."""
        id1, _ = create_project(tmp_path, "idea")
        id2, _ = create_project(tmp_path, "idea")
        assert id1 != id2

    def test_creates_artifacts_directory(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "idea")
        assert (project_dir / "artifacts").is_dir()

    def test_creates_logs_directory(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "idea")
        assert (project_dir / "logs").is_dir()

    def test_project_dir_lives_under_projects_root(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "idea")
        assert project_dir.parent == tmp_path

    def test_metadata_json_exists(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "idea")
        assert (project_dir / "metadata.json").is_file()

    def test_metadata_has_project_id(self, tmp_path: Path) -> None:
        project_id, project_dir = create_project(tmp_path, "idea")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["project_id"] == project_id

    def test_metadata_has_idea(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "a cool app")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["idea"] == "a cool app"

    def test_metadata_has_created_at_iso(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "idea")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "created_at" in meta
        # Should be parseable as ISO 8601
        assert "T" in meta["created_at"]

    def test_metadata_status_is_in_progress(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "idea")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "in_progress"

    def test_metadata_stages_completed_empty(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "idea")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["stages_completed"] == []

    def test_metadata_is_valid_json(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "idea")
        content = (project_dir / "metadata.json").read_text()
        meta = json.loads(content)
        assert isinstance(meta, dict)

    def test_idea_with_special_characters(self, tmp_path: Path) -> None:
        """Ideas with quotes, newlines, and unicode should persist correctly."""
        idea = 'A "fancy"\napp with émojis 🎉'
        _, project_dir = create_project(tmp_path, idea)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["idea"] == idea

    def test_empty_idea(self, tmp_path: Path) -> None:
        _, project_dir = create_project(tmp_path, "")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["idea"] == ""


# ---------------------------------------------------------------------------
# workspace_path
# ---------------------------------------------------------------------------


class TestWorkspacePath:
    """Tests for workspace directory creation and retrieval."""

    def test_returns_path_under_project_dir(self, tmp_path: Path) -> None:
        ws = workspace_path(tmp_path)
        assert ws == tmp_path / "workspace"

    def test_creates_directory(self, tmp_path: Path) -> None:
        ws = workspace_path(tmp_path)
        assert ws.is_dir()

    def test_idempotent_creation(self, tmp_path: Path) -> None:
        """Calling twice should not raise — exist_ok semantics."""
        ws1 = workspace_path(tmp_path)
        ws2 = workspace_path(tmp_path)
        assert ws1 == ws2
        assert ws1.is_dir()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        ws = workspace_path(str(tmp_path))
        assert isinstance(ws, Path)
        assert ws.is_dir()

    def test_creates_nested_parents(self, tmp_path: Path) -> None:
        """workspace_path should work even if project_dir doesn't exist yet."""
        deep = tmp_path / "a" / "b" / "c"
        ws = workspace_path(deep)
        assert ws.is_dir()
        assert ws == deep / "workspace"


# ---------------------------------------------------------------------------
# save_artifact / load_artifact
# ---------------------------------------------------------------------------


class TestSaveArtifact:
    """Tests for writing stage artifacts to disk."""

    def test_returns_artifact_path(self, tmp_path: Path) -> None:
        p = save_artifact(tmp_path, "spec.md", "# Spec")
        assert p == tmp_path / "artifacts" / "spec.md"

    def test_writes_content(self, tmp_path: Path) -> None:
        save_artifact(tmp_path, "spec.md", "# Spec\nDetails here")
        content = (tmp_path / "artifacts" / "spec.md").read_text()
        assert content == "# Spec\nDetails here"

    def test_creates_artifacts_dir_if_missing(self, tmp_path: Path) -> None:
        """save_artifact should work even if artifacts/ doesn't exist."""
        project = tmp_path / "new_project"
        save_artifact(project, "data.txt", "hello")
        assert (project / "artifacts" / "data.txt").is_file()

    def test_overwrites_existing_artifact(self, tmp_path: Path) -> None:
        save_artifact(tmp_path, "draft.md", "v1")
        save_artifact(tmp_path, "draft.md", "v2")
        content = (tmp_path / "artifacts" / "draft.md").read_text()
        assert content == "v2"

    def test_empty_content(self, tmp_path: Path) -> None:
        save_artifact(tmp_path, "empty.txt", "")
        content = (tmp_path / "artifacts" / "empty.txt").read_text()
        assert content == ""

    def test_large_content(self, tmp_path: Path) -> None:
        big = "x" * 100_000
        save_artifact(tmp_path, "big.txt", big)
        content = (tmp_path / "artifacts" / "big.txt").read_text()
        assert content == big

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        p = save_artifact(str(tmp_path), "test.txt", "data")
        assert isinstance(p, Path)
        assert p.read_text() == "data"

    def test_unicode_content(self, tmp_path: Path) -> None:
        save_artifact(tmp_path, "i18n.txt", "こんにちは世界 🌍")
        content = (tmp_path / "artifacts" / "i18n.txt").read_text()
        assert content == "こんにちは世界 🌍"


class TestLoadArtifact:
    """Tests for reading stage artifacts from disk."""

    def test_round_trip(self, tmp_path: Path) -> None:
        save_artifact(tmp_path, "spec.md", "# Spec content")
        result = load_artifact(tmp_path, "spec.md")
        assert result == "# Spec content"

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        result = load_artifact(tmp_path, "nonexistent.md")
        assert result is None

    def test_returns_none_when_artifacts_dir_missing(self, tmp_path: Path) -> None:
        """If the artifacts/ subdirectory doesn't exist, returns None."""
        project = tmp_path / "empty_project"
        project.mkdir()
        result = load_artifact(project, "anything.txt")
        assert result is None

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        save_artifact(tmp_path, "s.txt", "content")
        result = load_artifact(str(tmp_path), "s.txt")
        assert result == "content"

    def test_preserves_whitespace(self, tmp_path: Path) -> None:
        original = "  leading\n\n  trailing  \n"
        save_artifact(tmp_path, "ws.txt", original)
        assert load_artifact(tmp_path, "ws.txt") == original

    def test_multiple_artifacts_independent(self, tmp_path: Path) -> None:
        save_artifact(tmp_path, "a.txt", "alpha")
        save_artifact(tmp_path, "b.txt", "bravo")
        assert load_artifact(tmp_path, "a.txt") == "alpha"
        assert load_artifact(tmp_path, "b.txt") == "bravo"


# ---------------------------------------------------------------------------
# save_log
# ---------------------------------------------------------------------------


class TestSaveLog:
    """Tests for append-mode log writing with timestamps."""

    def test_returns_log_path(self, tmp_path: Path) -> None:
        p = save_log(tmp_path, "build", "started")
        assert p == tmp_path / "logs" / "build.log"

    def test_creates_log_file(self, tmp_path: Path) -> None:
        save_log(tmp_path, "build", "started")
        assert (tmp_path / "logs" / "build.log").is_file()

    def test_log_contains_content(self, tmp_path: Path) -> None:
        save_log(tmp_path, "build", "Build started successfully")
        content = (tmp_path / "logs" / "build.log").read_text()
        assert "Build started successfully" in content

    def test_log_contains_timestamp(self, tmp_path: Path) -> None:
        save_log(tmp_path, "build", "msg")
        content = (tmp_path / "logs" / "build.log").read_text()
        # Timestamp format: YYYY-MM-DD HH:MM:SS
        assert "---" in content
        # Should contain date-like pattern
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)

    def test_append_behavior(self, tmp_path: Path) -> None:
        """Multiple writes should append, not overwrite."""
        save_log(tmp_path, "build", "first entry")
        save_log(tmp_path, "build", "second entry")
        content = (tmp_path / "logs" / "build.log").read_text()
        assert "first entry" in content
        assert "second entry" in content

    def test_separate_stages_separate_files(self, tmp_path: Path) -> None:
        save_log(tmp_path, "build", "building")
        save_log(tmp_path, "test", "testing")
        assert (tmp_path / "logs" / "build.log").is_file()
        assert (tmp_path / "logs" / "test.log").is_file()
        build_content = (tmp_path / "logs" / "build.log").read_text()
        test_content = (tmp_path / "logs" / "test.log").read_text()
        assert "building" in build_content
        assert "testing" in test_content
        assert "testing" not in build_content

    def test_creates_logs_dir_if_missing(self, tmp_path: Path) -> None:
        project = tmp_path / "new_project"
        save_log(project, "stage", "content")
        assert (project / "logs" / "stage.log").is_file()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        p = save_log(str(tmp_path), "stage", "msg")
        assert isinstance(p, Path)
        assert p.is_file()

    def test_multiline_content(self, tmp_path: Path) -> None:
        save_log(tmp_path, "qa", "line1\nline2\nline3")
        content = (tmp_path / "logs" / "qa.log").read_text()
        assert "line1\nline2\nline3" in content

    def test_timestamp_bracket_format(self, tmp_path: Path) -> None:
        """Timestamp should appear in [YYYY-MM-DD HH:MM:SS] bracket format."""
        save_log(tmp_path, "build", "msg")
        content = (tmp_path / "logs" / "build.log").read_text()
        import re

        assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", content)


# ---------------------------------------------------------------------------
# mark_stage_complete
# ---------------------------------------------------------------------------


class TestMarkStageComplete:
    """Tests for recording stage completion in project metadata."""

    def _setup_project(self, tmp_path: Path) -> Path:
        """Helper: create a project and return its directory."""
        _, project_dir = create_project(tmp_path, "test idea")
        return project_dir

    def test_adds_stage_to_stages_completed(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        mark_stage_complete(project_dir, "build")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "build" in meta["stages_completed"]

    def test_multiple_stages(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        mark_stage_complete(project_dir, "research")
        mark_stage_complete(project_dir, "build")
        mark_stage_complete(project_dir, "test")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["stages_completed"] == ["research", "build", "test"]

    def test_idempotent_no_duplicates(self, tmp_path: Path) -> None:
        """Marking the same stage twice should not create a duplicate entry."""
        project_dir = self._setup_project(tmp_path)
        mark_stage_complete(project_dir, "build")
        mark_stage_complete(project_dir, "build")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["stages_completed"].count("build") == 1

    def test_sets_last_updated(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        mark_stage_complete(project_dir, "build")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "last_updated" in meta
        assert "T" in meta["last_updated"]  # ISO 8601

    def test_preserves_existing_metadata_fields(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        original_meta = json.loads((project_dir / "metadata.json").read_text())
        mark_stage_complete(project_dir, "build")
        updated_meta = json.loads((project_dir / "metadata.json").read_text())
        assert updated_meta["project_id"] == original_meta["project_id"]
        assert updated_meta["idea"] == original_meta["idea"]
        assert updated_meta["created_at"] == original_meta["created_at"]
        assert updated_meta["status"] == "in_progress"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        mark_stage_complete(str(project_dir), "build")
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "build" in meta["stages_completed"]

    def test_order_preserved(self, tmp_path: Path) -> None:
        """Stages should appear in the order they were completed."""
        project_dir = self._setup_project(tmp_path)
        stages = ["spec", "research", "architecture", "build", "qa"]
        for stage in stages:
            mark_stage_complete(project_dir, stage)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["stages_completed"] == stages


# ---------------------------------------------------------------------------
# mark_project_done
# ---------------------------------------------------------------------------


class TestMarkProjectDone:
    """Tests for marking a project as completed."""

    def _setup_project(self, tmp_path: Path) -> Path:
        _, project_dir = create_project(tmp_path, "test idea")
        return project_dir

    def test_sets_status_completed(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        mark_project_done(project_dir)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "completed"

    def test_sets_completed_at(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        mark_project_done(project_dir)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "completed_at" in meta
        assert "T" in meta["completed_at"]  # ISO 8601

    def test_preserves_existing_fields(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        original_meta = json.loads((project_dir / "metadata.json").read_text())
        mark_project_done(project_dir)
        updated_meta = json.loads((project_dir / "metadata.json").read_text())
        assert updated_meta["project_id"] == original_meta["project_id"]
        assert updated_meta["idea"] == original_meta["idea"]
        assert updated_meta["created_at"] == original_meta["created_at"]

    def test_preserves_stages_completed(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        mark_stage_complete(project_dir, "build")
        mark_stage_complete(project_dir, "qa")
        mark_project_done(project_dir)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["stages_completed"] == ["build", "qa"]
        assert meta["status"] == "completed"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        project_dir = self._setup_project(tmp_path)
        mark_project_done(str(project_dir))
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "completed"

    def test_calling_done_twice(self, tmp_path: Path) -> None:
        """Marking done twice should not error; completed_at is updated."""
        project_dir = self._setup_project(tmp_path)
        mark_project_done(project_dir)
        mark_project_done(project_dir)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "completed"


# ---------------------------------------------------------------------------
# list_projects
# ---------------------------------------------------------------------------


class TestListProjects:
    """Tests for listing all projects with metadata, sorted by recency."""

    def test_empty_when_root_missing(self, tmp_path: Path) -> None:
        """Non-existent root directory should return an empty list."""
        result = list_projects(tmp_path / "nonexistent")
        assert result == []

    def test_empty_when_root_exists_but_empty(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        result = list_projects(root)
        assert result == []

    def test_returns_single_project_metadata(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        project_id, _ = create_project(root, "only project")
        results = list_projects(root)
        assert len(results) == 1
        assert results[0]["project_id"] == project_id
        assert results[0]["idea"] == "only project"

    def test_returns_multiple_projects(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        create_project(root, "first")
        create_project(root, "second")
        create_project(root, "third")
        results = list_projects(root)
        assert len(results) == 3

    def test_sorted_by_directory_name_reverse(self, tmp_path: Path) -> None:
        """list_projects uses sorted(iterdir(), reverse=True) on directory names."""
        root = tmp_path / "projects"
        root.mkdir()
        # Create projects with known names to verify sort order
        ids = []
        for idea in ["a", "b", "c"]:
            pid, _ = create_project(root, idea)
            ids.append(pid)
        results = list_projects(root)
        result_ids = [r["project_id"] for r in results]
        # reverse-sorted by directory name (which equals project_id)
        assert result_ids == sorted(ids, reverse=True)

    def test_skips_directories_without_metadata(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        create_project(root, "real project")
        # Create a rogue directory with no metadata.json
        (root / "not_a_project").mkdir()
        results = list_projects(root)
        assert len(results) == 1
        assert results[0]["idea"] == "real project"

    def test_returns_list_of_dicts(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        root.mkdir()
        create_project(root, "idea")
        results = list_projects(root)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_metadata_reflects_completion_status(self, tmp_path: Path) -> None:
        """list_projects should return the latest metadata, including done status."""
        root = tmp_path / "projects"
        root.mkdir()
        _, project_dir = create_project(root, "done project")
        mark_stage_complete(project_dir, "build")
        mark_project_done(project_dir)
        results = list_projects(root)
        assert len(results) == 1
        assert results[0]["status"] == "completed"
        assert "completed_at" in results[0]
        assert results[0]["stages_completed"] == ["build"]


# ---------------------------------------------------------------------------
# Integration: full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end test combining multiple artifact functions."""

    def test_full_project_lifecycle(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        root.mkdir()

        # 1. Create project
        project_id, project_dir = create_project(root, "Build a CLI tool")
        assert project_id.startswith("kindle_")

        # 2. Get workspace
        ws = workspace_path(project_dir)
        assert ws.is_dir()

        # 3. Save artifacts from different stages
        save_artifact(project_dir, "spec.md", "# Feature Spec")
        save_artifact(project_dir, "architecture.md", "# Architecture")

        # 4. Load artifacts back
        assert load_artifact(project_dir, "spec.md") == "# Feature Spec"
        assert load_artifact(project_dir, "architecture.md") == "# Architecture"
        assert load_artifact(project_dir, "missing.md") is None

        # 5. Log stage activity
        save_log(project_dir, "build", "Compiling project")
        save_log(project_dir, "build", "Build complete")

        # 6. Mark stages complete
        mark_stage_complete(project_dir, "spec")
        mark_stage_complete(project_dir, "architecture")
        mark_stage_complete(project_dir, "build")

        # 7. Finish project
        mark_project_done(project_dir)

        # 8. Verify final state via list_projects
        projects = list_projects(root)
        assert len(projects) == 1
        meta = projects[0]
        assert meta["project_id"] == project_id
        assert meta["idea"] == "Build a CLI tool"
        assert meta["status"] == "completed"
        assert meta["stages_completed"] == ["spec", "architecture", "build"]
        assert "completed_at" in meta
        assert "created_at" in meta
