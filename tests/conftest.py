"""Shared pytest fixtures for the kindle test suite."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def make_state(tmp_path: Path):
    """Factory fixture that builds a minimal KindleState dict.

    Creates the standard project directory scaffold (artifacts/, logs/,
    and metadata.json) under *tmp_path* and returns a state dict.

    Parameters
    ----------
    create_workspace : bool
        Also create a ``workspace/`` directory (default ``False``).
    metadata_extra : dict | None
        Additional keys merged into ``metadata.json``.
    **overrides
        Merged into the returned state dict, overriding defaults.
    """

    def _factory(
        *,
        create_workspace: bool = False,
        metadata_extra: dict | None = None,
        **overrides: object,
    ) -> dict:
        project_dir = tmp_path / "project"
        (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (project_dir / "logs").mkdir(parents=True, exist_ok=True)
        if create_workspace:
            (project_dir / "workspace").mkdir(parents=True, exist_ok=True)
        meta: dict = {"project_id": "kindle_test1234", "stages_completed": []}
        if metadata_extra:
            meta.update(metadata_extra)
        (project_dir / "metadata.json").write_text(json.dumps(meta))

        state: dict = {
            "project_dir": str(project_dir),
            "idea": "a task management app",
        }
        state.update(overrides)
        return state

    return _factory


@pytest.fixture
def make_ui():
    """Factory fixture returning a properly configured MagicMock UI.

    Sets ``auto_approve`` to the given value (default ``False``).
    All other MagicMock attributes are auto-created on access.
    """

    def _factory(*, auto_approve: bool = False) -> MagicMock:
        ui = MagicMock()
        ui.auto_approve = auto_approve
        return ui

    return _factory
