"""Kindle CLI — the user-facing interface to the application factory."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import typer

from kindle.artifacts import create_project, list_projects, load_artifact, workspace_path
from kindle.config import Settings
from kindle.graph import build_graph
from kindle.state import KindleState
from kindle.ui import UI

app = typer.Typer(
    name="kindle",
    help="AI-powered application factory — describe what you want, the machine builds it.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def build(
    ctx: typer.Context,
    idea: str = typer.Argument(None, help="Describe the application you want to build"),
    stack: str = typer.Option("", "--stack", "-s", help="Tech stack preference (e.g. react, fastapi, nextjs)"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Auto-answer all Grill questions with defaults"),
    concurrency: int = typer.Option(0, "--concurrency", "-c", help="Max concurrent dev agents (0 = use config default)"),
    review_arch: bool = typer.Option(False, "--review-arch", help="Pause for human review of architecture"),
    output: str = typer.Option("", "--output", "-o", help="Output directory for the built project"),
    qa_retries: int = typer.Option(0, "--qa-retries", help="Max QA self-healing retries (0 = use config default)"),
    cpo_retries: int = typer.Option(0, "--cpo-retries", help="Max product audit retries (0 = use config default)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show agent message stream"),
) -> None:
    """Build an application from a description."""
    # If a subcommand was invoked, skip the default
    if ctx.invoked_subcommand is not None:
        return
    if not idea:
        typer.echo("Usage: kindle 'describe your application' [OPTIONS]")
        typer.echo("Try: kindle --help")
        raise typer.Exit(0)

    settings = Settings.load()
    ui = UI(auto_approve=auto_approve, verbose=verbose)

    # Resolve settings with CLI overrides
    max_concurrent = concurrency if concurrency > 0 else settings.max_concurrent_agents
    max_qa = qa_retries if qa_retries > 0 else settings.max_qa_retries
    max_cpo = cpo_retries if cpo_retries > 0 else settings.max_cpo_retries

    project_id, project_dir = create_project(settings.projects_root, idea)
    ui.banner(idea, project_id)

    initial_state: KindleState = {
        "idea": idea,
        "project_id": project_id,
        "project_dir": str(project_dir),
        "stack_preference": stack,
        "auto_approve": auto_approve,
        "max_concurrent_agents": max_concurrent,
        "max_qa_retries": max_qa,
        "max_cpo_retries": max_cpo,
        "feature_spec": {},
        "grill_transcript": "",
        "research_report": "",
        "architecture": "",
        "dev_tasks": [],
        "qa_report": "",
        "product_audit": "",
        "package_readme": "",
        "qa_passed": False,
        "cpo_passed": False,
        "qa_retries": 0,
        "cpo_retries": 0,
        "model": settings.model,
        "max_agent_turns": settings.max_agent_turns,
        "current_stage": "",
    }

    compiled = build_graph(ui)
    result = asyncio.run(compiled.ainvoke(initial_state))

    ws = workspace_path(str(project_dir))
    ui.info(f"Project built at: {ws}")
    ui.info(f"Session artifacts: {project_dir}")

    # Copy to output directory if specified
    if output:
        import shutil

        output_path = Path(output).expanduser().resolve()
        if output_path.exists():
            ui.error(f"Output directory already exists: {output_path}")
        else:
            shutil.copytree(str(ws), str(output_path))
            ui.info(f"Copied to: {output_path}")


@app.command()
def resume(
    project_path: str = typer.Argument(
        ..., help="Path to the Kindle session directory (e.g. ~/.kindle/projects/kindle_XXXXX)"
    ),
    from_stage: str = typer.Option(
        "dev", "--from", help="Stage to resume from: grill, research, architect, dev, qa, package"
    ),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Auto-answer all Grill questions with defaults"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show agent message stream"),
) -> None:
    """Resume a Kindle build session from a specific stage."""
    settings = Settings.load()
    ui = UI(auto_approve=auto_approve, verbose=verbose)

    project_dir = Path(project_path).expanduser()
    if not project_dir.exists():
        ui.error(f"Session directory not found: {project_dir}")
        raise typer.Exit(1)

    meta_path = project_dir / "metadata.json"
    if not meta_path.exists():
        ui.error("No metadata.json found — is this a valid Kindle session?")
        raise typer.Exit(1)

    meta = json.loads(meta_path.read_text())
    idea = meta.get("idea", "")
    project_id = meta["project_id"]

    ui.banner(idea, project_id)
    ui.info(f"Resuming from stage: {from_stage}")

    # Reload artifacts from previous stages
    feature_spec_raw = load_artifact(str(project_dir), "feature_spec.json")
    feature_spec = json.loads(feature_spec_raw) if feature_spec_raw else {}

    dev_tasks_raw = load_artifact(str(project_dir), "dev_tasks.json")
    dev_tasks = json.loads(dev_tasks_raw) if dev_tasks_raw else []

    state: KindleState = {
        "idea": idea,
        "project_id": project_id,
        "project_dir": str(project_dir),
        "stack_preference": "",
        "auto_approve": auto_approve,
        "max_concurrent_agents": settings.max_concurrent_agents,
        "max_qa_retries": settings.max_qa_retries,
        "max_cpo_retries": settings.max_cpo_retries,
        "feature_spec": feature_spec,
        "grill_transcript": load_artifact(str(project_dir), "grill_transcript.md") or "",
        "research_report": load_artifact(str(project_dir), "research_report.md") or "",
        "architecture": load_artifact(str(project_dir), "architecture.md") or "",
        "dev_tasks": dev_tasks,
        "qa_report": load_artifact(str(project_dir), "qa_report.md") or "",
        "product_audit": load_artifact(str(project_dir), "product_audit.md") or "",
        "package_readme": "",
        "qa_passed": False,
        "cpo_passed": False,
        "qa_retries": 0,
        "cpo_retries": 0,
        "model": settings.model,
        "max_agent_turns": settings.max_agent_turns,
        "current_stage": "",
    }

    compiled = build_graph(ui, entry_stage=from_stage)
    result = asyncio.run(compiled.ainvoke(state))

    ws = workspace_path(str(project_dir))
    ui.info(f"Project at: {ws}")


@app.command(name="list")
def list_cmd() -> None:
    """List all Kindle build sessions."""
    settings = Settings.load()
    ui = UI()
    projects = list_projects(settings.projects_root)
    ui.show_projects(projects)


if __name__ == "__main__":
    app()
