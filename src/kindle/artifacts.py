"""Persistent artifact layer — every stage reads from and writes to disk."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def create_project(projects_root: Path, idea: str) -> tuple[str, Path]:
    """Create a new build project directory and return (project_id, project_dir)."""
    project_id = f"kindle_{uuid4().hex[:8]}"
    project_dir = projects_root / project_id
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)

    metadata = {
        "project_id": project_id,
        "idea": idea,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return project_id, project_dir


def workspace_path(project_dir: str | Path) -> Path:
    """Return the workspace directory where the built project lives."""
    ws = Path(project_dir) / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def save_artifact(project_dir: str | Path, name: str, content: str) -> Path:
    """Write a stage artifact to disk."""
    p = Path(project_dir) / "artifacts" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def load_artifact(project_dir: str | Path, name: str) -> str | None:
    """Read a stage artifact from disk, or None if it doesn't exist."""
    p = Path(project_dir) / "artifacts" / name
    return p.read_text() if p.exists() else None


def save_log(project_dir: str | Path, stage: str, content: str) -> Path:
    """Append to a stage log file."""
    p = Path(project_dir) / "logs" / f"{stage}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    with p.open("a") as f:
        f.write(f"\n--- [{timestamp}] ---\n{content}\n")
    return p


def mark_stage_complete(project_dir: str | Path, stage: str) -> None:
    """Record that a stage has finished successfully."""
    meta_path = Path(project_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text())
    if stage not in meta["stages_completed"]:
        meta["stages_completed"].append(stage)
    meta["last_updated"] = datetime.now(UTC).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2))


def mark_project_done(project_dir: str | Path) -> None:
    """Mark the build project as complete."""
    meta_path = Path(project_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["status"] = "completed"
    meta["completed_at"] = datetime.now(UTC).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2))


def list_projects(projects_root: Path) -> list[dict]:
    """Return metadata for all build projects, most recent first."""
    projects: list[dict] = []
    if not projects_root.exists():
        return projects
    for d in sorted(projects_root.iterdir(), reverse=True):
        meta_path = d / "metadata.json"
        if meta_path.exists():
            projects.append(json.loads(meta_path.read_text()))
    return projects
