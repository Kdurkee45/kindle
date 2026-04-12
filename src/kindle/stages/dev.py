"""Dev stage — parallel build with multiple agents.

Each dev task runs in its own agent session scoped to its directory.
Uses asyncio.Semaphore for concurrency control.
"""

from __future__ import annotations

import asyncio
import json

from kindle.agent import run_agent
from kindle.artifacts import mark_stage_complete, save_artifact, workspace_path
from kindle.state import KindleState
from kindle.ui import UI

SYSTEM_PROMPT = """\
You are a Principal Software Engineer. You are building one part of a larger
application. Follow the architecture exactly — do not improvise.

## Your Rules

1. **Write complete, working code.** No placeholders, no TODOs, no "implement later".
   Every file must be functional.

2. **Write tests for everything you build.** Co-locate tests or put them in a
   tests/ directory. Every function/component needs at least one test.

3. **Stay in your directory scope.** Only write files in the directories assigned
   to your task. Do not modify files outside your scope.

4. **Follow the architecture.** The architecture document is the source of truth.
   Use the specified tech stack, patterns, and conventions.

5. **Handle errors properly.** No bare except, no swallowed errors. Proper
   validation, proper error messages.

6. **Clean code.** Meaningful names, docstrings, type hints, consistent style.

When finished, ensure all files are written and the code compiles/lints clean.
"""


async def _run_task(
    task: dict,
    state: KindleState,
    ui: UI,
    semaphore: asyncio.Semaphore,
    task_index: int,
    total_tasks: int,
) -> dict:
    """Run a single dev task within the semaphore."""
    task_id = task.get("task_id", f"task_{task_index:02d}")
    title = task.get("title", "Untitled")
    project_dir = state["project_dir"]
    ws = workspace_path(project_dir)

    async with semaphore:
        ui.task_start(task_id, title, task_index, total_tasks)

        feature_spec = state.get("feature_spec", {})
        architecture = state.get("architecture", "")

        prompt_parts = [
            "Build this dev task for the application.\n",
            f"TASK ID: {task_id}",
            f"TASK TITLE: {title}",
            f"TASK DESCRIPTION:\n{task.get('description', '')}",
            f"\nDIRECTORY SCOPE: {task.get('directory_scope', '.')}",
            "\nACCEPTANCE CRITERIA:\n" + "\n".join(f"  - {c}" for c in task.get("acceptance_criteria", [])),
            f"\nFEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}",
            f"\nARCHITECTURE:\n{architecture}",
            f"\nWrite all code and tests to the working directory. "
            f"Stay within your directory scope: {task.get('directory_scope', '.')}",
        ]

        result = await run_agent(
            persona=f"Principal Software Engineer ({task_id})",
            system_prompt=SYSTEM_PROMPT,
            user_prompt="\n".join(prompt_parts),
            cwd=str(ws),
            project_dir=project_dir,
            stage=f"dev_{task_id}",
            ui=ui,
            model=state.get("model"),
            max_turns=state.get("max_agent_turns", 50),
        )

        ui.task_done(task_id)

        return {
            "task_id": task_id,
            "title": title,
            "status": "completed",
            "elapsed_seconds": result.elapsed_seconds,
            "turns_used": result.turns_used,
        }


def _topological_sort(tasks: list[dict]) -> list[list[dict]]:
    """Sort tasks into layers by dependencies. Tasks in the same layer can run in parallel."""
    completed: set[str] = set()
    layers: list[list[dict]] = []

    remaining = list(tasks)
    while remaining:
        layer: list[dict] = []
        still_remaining: list[dict] = []

        for task in remaining:
            deps = task.get("dependencies", [])
            if all(d in completed for d in deps):
                layer.append(task)
            else:
                still_remaining.append(task)

        if not layer:
            # Circular dependency or unresolvable — just run everything
            layer = still_remaining
            still_remaining = []

        layers.append(layer)
        for task in layer:
            completed.add(task.get("task_id", ""))
        remaining = still_remaining

    return layers


async def dev_node(state: KindleState, ui: UI) -> dict:
    """LangGraph node: build all dev tasks in parallel with dependency ordering."""
    ui.stage_start("dev")
    project_dir = state["project_dir"]
    dev_tasks = state.get("dev_tasks", [])
    max_concurrent = state.get("max_concurrent_agents", 4)

    if not dev_tasks:
        ui.error("No dev tasks found — skipping dev stage.")
        mark_stage_complete(project_dir, "dev")
        ui.stage_done("dev")
        return {"current_stage": "dev"}

    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(dev_tasks)
    layers = _topological_sort(dev_tasks)

    all_results: list[dict] = []
    task_counter = 0

    for layer_idx, layer in enumerate(layers):
        ui.info(f"Layer {layer_idx + 1}/{len(layers)} — {len(layer)} task(s) in parallel")

        coros = []
        for task in layer:
            task_counter += 1
            coros.append(_run_task(task, state, ui, semaphore, task_counter, total))

        results = await asyncio.gather(*coros, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                ui.error(f"Dev task failed: {r}")
                all_results.append({"status": "failed", "error": str(r)})
            else:
                all_results.append(r)

    save_artifact(project_dir, "dev_results.json", json.dumps(all_results, indent=2))

    mark_stage_complete(project_dir, "dev")
    ui.stage_done("dev")

    return {"current_stage": "dev"}
