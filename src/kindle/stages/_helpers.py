"""Shared helpers for pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

from kindle.artifacts import mark_stage_complete, workspace_path
from kindle.state import KindleState
from kindle.ui import UI


def stage_setup(state: KindleState, ui: UI, stage_name: str) -> tuple[str, Path]:
    """Signal stage start and return ``(project_dir, workspace)``.

    Every stage node begins with the same three-step preamble:

    1. ``ui.stage_start(stage_name)``
    2. Extract ``project_dir`` from *state*
    3. Resolve (and create) the workspace directory

    Returns:
        A ``(project_dir, workspace_path)`` pair ready for agent invocations.
    """
    ui.stage_start(stage_name)
    project_dir: str = state["project_dir"]
    ws = workspace_path(project_dir)
    return project_dir, ws


def stage_teardown(project_dir: str, stage_name: str, ui: UI) -> None:
    """Mark a stage as complete and signal the UI."""
    mark_stage_complete(project_dir, stage_name)
    ui.stage_done(stage_name)


def load_json_artifact(path: Path) -> dict | list | None:
    """Load a JSON artifact, returning *None* on missing or malformed input."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_text_artifact(path: Path) -> str:
    """Load a text artifact, returning an empty string when the file is absent."""
    if path.exists():
        return path.read_text()
    return ""


def cleanup_workspace_files(*paths: Path) -> None:
    """Remove transient workspace files, silently skipping any that are missing."""
    for p in paths:
        p.unlink(missing_ok=True)
