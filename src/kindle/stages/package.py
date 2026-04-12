"""Package stage — ensure the project is runnable and well-documented.

The Principal DevOps Engineer generates README, initializes git with atomic
commits per dev task, and optionally creates a Dockerfile.
"""

from __future__ import annotations

import json

from kindle.agent import run_agent
from kindle.artifacts import mark_project_done, mark_stage_complete, save_artifact, workspace_path
from kindle.state import KindleState
from kindle.ui import UI

SYSTEM_PROMPT = """\
You are a Principal DevOps Engineer. Your job is to ensure the project is
runnable, well-documented, and ready to hand off.

## What You Must Do

1. **Verify Dependencies**
   - Check that all dependencies are declared in the manifest
   - Run the install command to verify they resolve
   - Lock dependencies if not already locked

2. **Verify the Project Runs**
   - Run the appropriate start command (npm start, python -m app, etc.)
   - If it crashes, fix the issue
   - For CLI tools, verify the help command works
   - For APIs, verify the health endpoint responds
   - For web apps, verify the build succeeds

3. **Generate README.md**
   Write a comprehensive README with:
   - Project name and description (from the feature spec)
   - How to install and run
   - Tech stack overview
   - Project structure (directory tree)
   - Environment variables needed
   - API endpoints (if applicable)
   - Testing instructions
   - License placeholder

4. **Initialize Git Repository**
   - Run `git init` in the project directory
   - Create a .gitignore appropriate for the tech stack
   - Stage all files and create an initial commit:
     "Initial commit: <project name>"
   - Then, for each dev task, make an atomic commit with the message:
     "feat: <task title>" (use `git add -A && git commit --allow-empty -m "..."`
     if the files are already committed)

5. **Optional: Dockerfile**
   If the project is a web app or API, create a basic Dockerfile with:
   - Multi-stage build if applicable
   - Production-ready configuration
   - Health check

## Output

Write all files directly in the project directory. The project should be
ready to `git clone` and run with the README instructions.
"""


async def package_node(state: KindleState, ui: UI) -> dict:
    """LangGraph node: package the project for delivery."""
    ui.stage_start("package")
    project_dir = state["project_dir"]
    feature_spec = state.get("feature_spec", {})
    architecture = state.get("architecture", "")
    dev_tasks = state.get("dev_tasks", [])
    idea = state.get("idea", "")
    ws = workspace_path(project_dir)

    prompt_parts = [
        "Package this project for delivery.",
        f"\nIDEA: {idea}",
        f"\nFEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}",
        f"\nARCHITECTURE:\n{architecture}",
        f"\nDEV TASKS:\n{json.dumps(dev_tasks, indent=2)}",
        "\nEnsure the project runs, generate README.md, initialize git with "
        "atomic commits per dev task, and optionally add a Dockerfile.",
    ]

    await run_agent(
        persona="Principal DevOps Engineer",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="\n".join(prompt_parts),
        cwd=str(ws),
        project_dir=project_dir,
        stage="package",
        ui=ui,
        model=state.get("model"),
        max_turns=state.get("max_agent_turns", 50),
    )

    # Read the README if generated
    readme_path = ws / "README.md"
    package_readme = readme_path.read_text() if readme_path.exists() else ""
    if package_readme:
        save_artifact(project_dir, "package_readme.md", package_readme)

    mark_stage_complete(project_dir, "package")
    mark_project_done(project_dir)

    ui.deploy_complete(str(ws))
    ui.stage_done("package")

    return {
        "package_readme": package_readme,
        "current_stage": "package",
    }
