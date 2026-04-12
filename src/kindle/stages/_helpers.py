"""Shared helpers for pipeline stages."""

from __future__ import annotations

from pathlib import Path

from kindle.artifacts import workspace_path
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
