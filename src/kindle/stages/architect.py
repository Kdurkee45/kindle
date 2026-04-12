"""Architect stage — design the system and split work into parallel dev tasks.

The Principal Solution Architect reads the spec and research, then designs
the architecture and creates non-overlapping dev tasks for parallel execution.
"""

from __future__ import annotations

import json
from kindle.agent import run_agent
from kindle.artifacts import mark_stage_complete, save_artifact, workspace_path
from kindle.state import KindleState
from kindle.ui import UI

SYSTEM_PROMPT = """\
You are a Principal Solution Architect. You have a complete feature specification
and a technical research report. Your job: design the system and split the work
into parallel dev tasks.

## What You Must Produce

### 1. architecture.md
A comprehensive architecture document covering:
- Tech stack choices (with justification from the research)
- High-level architecture (diagrams in ASCII art)
- Directory structure (every file that needs to be created)
- Data model (entities, relationships, schemas)
- API surface (endpoints, request/response shapes)
- Authentication/authorization approach
- Error handling strategy
- Testing strategy
- Build and run instructions

### 2. dev_tasks.json
An array of dev tasks for parallel execution:

```json
[
  {
    "task_id": "task_01",
    "title": "Set up project structure and dependencies",
    "description": "Detailed description of what to build...",
    "directory_scope": ".",
    "dependencies": [],
    "acceptance_criteria": [
      "pyproject.toml exists with all dependencies",
      "Directory structure matches architecture"
    ]
  },
  {
    "task_id": "task_02",
    "title": "Build data models and database layer",
    "description": "...",
    "directory_scope": "src/models/",
    "dependencies": ["task_01"],
    "acceptance_criteria": ["All models defined", "Tests pass"]
  }
]
```

## Critical Rules

1. **Non-overlapping directory scopes.** Each task owns specific directories.
   No two tasks write to the same directory. This enables parallel execution.

2. **Dependencies are task IDs.** If task_02 depends on task_01's output,
   declare it. Tasks without dependencies can run in parallel.

3. **Tests are mandatory.** Every task that writes code must also write tests.
   Include this in the description and acceptance criteria.

4. **No placeholders.** The description must be detailed enough that a developer
   can build it without asking questions.

5. **Shared code gets its own task.** Cross-cutting concerns (types, utilities,
   config) should be a separate task that others depend on.

6. **Task 01 should always be project scaffolding** — pyproject.toml / package.json,
   directory structure, config files, etc.

Write both files to the working directory.
"""


async def architect_node(state: KindleState, ui: UI) -> dict:
    """LangGraph node: design architecture and create dev tasks."""
    ui.stage_start("architect")
    project_dir = state["project_dir"]
    feature_spec = state.get("feature_spec", {})
    research_report = state.get("research_report", "")
    idea = state.get("idea", "")
    stack_pref = state.get("stack_preference", "")
    ws = workspace_path(project_dir)

    prompt_parts = [
        f"Design the architecture and create dev tasks for this application.",
        f"\nIDEA: {idea}",
        f"\nFEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}",
        f"\nRESEARCH REPORT:\n{research_report}",
    ]
    if stack_pref:
        prompt_parts.append(f"\nSTACK PREFERENCE: {stack_pref}")
    prompt_parts.append(
        "\nWrite architecture.md and dev_tasks.json to the working directory."
    )

    result = await run_agent(
        persona="Principal Solution Architect",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="\n".join(prompt_parts),
        cwd=str(ws),
        project_dir=project_dir,
        stage="architect",
        ui=ui,
        model=state.get("model"),
        max_turns=30,
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
    )

    # Read architecture
    arch_path = ws / "architecture.md"
    architecture = arch_path.read_text() if arch_path.exists() else result.text
    save_artifact(project_dir, "architecture.md", architecture)

    # Read dev tasks
    tasks_path = ws / "dev_tasks.json"
    dev_tasks: list[dict] = []
    if tasks_path.exists():
        try:
            data = json.loads(tasks_path.read_text())
            dev_tasks = data if isinstance(data, list) else data.get("tasks", [])
        except json.JSONDecodeError:
            ui.error("Failed to parse dev_tasks.json — no tasks generated.")
    else:
        ui.error("Agent did not produce dev_tasks.json.")

    save_artifact(project_dir, "dev_tasks.json", json.dumps(dev_tasks, indent=2))

    # Clean up
    for p in [arch_path, tasks_path]:
        if p.exists():
            p.unlink()

    ui.info(f"Architecture designed — {len(dev_tasks)} dev task(s) created.")

    # Optional human review of architecture before building
    if state.get("review_arch") and not state.get("auto_approve"):
        ui.show_artifact("Architecture", architecture)
        approved, feedback = ui.prompt_arch_review()
        if not approved:
            ui.info(f"Architecture feedback: {feedback}")
            ui.info("Architecture revision not yet implemented — proceeding with current design.")

    mark_stage_complete(project_dir, "architect")
    ui.stage_done("architect")

    return {
        "architecture": architecture,
        "dev_tasks": dev_tasks,
        "current_stage": "architect",
    }
