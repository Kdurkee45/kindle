"""Research stage — gather technical context for architecture decisions.

The Principal Research Engineer examines the technology landscape, libraries,
patterns, prior art, and potential pitfalls. Pure research — no opinions.
"""

from __future__ import annotations

import json

from kindle.agent import run_agent
from kindle.artifacts import save_artifact
from kindle.stages._helpers import (
    cleanup_workspace_files,
    load_text_artifact,
    stage_setup,
    stage_teardown,
)
from kindle.state import KindleState
from kindle.ui import UI

SYSTEM_PROMPT = """\
You are a Principal Research Engineer. Your job is to research the technical
landscape for building an application — gathering facts, not making decisions.

You have a complete feature specification from the Grill phase. Research:

1. **Technology Landscape**
   - Best frameworks and libraries for this type of application
   - Pros/cons of different approaches
   - Current ecosystem maturity and community support

2. **Patterns & Best Practices**
   - Architecture patterns for this app type (MVC, microservices, serverless, etc.)
   - Data modeling patterns
   - API design patterns
   - Authentication/authorization patterns

3. **Prior Art**
   - Similar applications and how they were built
   - Common pitfalls when building this type of app
   - Known scaling challenges

4. **Libraries & Dependencies**
   - Recommended packages with version numbers
   - Database options and ORMs
   - Testing frameworks
   - Build tools and dev dependencies

5. **Potential Pitfalls**
   - Common mistakes when building this type of app
   - Security considerations
   - Performance gotchas
   - Cross-browser/platform issues

## Output

Write `research_report.md` to the working directory — a comprehensive technical
research document. This is a factual reference, NOT an architecture proposal.
The Architect stage will read this and make decisions.

Be thorough but factual. Cite specific library names and versions where possible.
"""


async def research_node(state: KindleState, ui: UI) -> dict:
    """LangGraph node: research the technology landscape."""
    project_dir, ws = stage_setup(state, ui, "research")
    feature_spec = state.get("feature_spec", {})
    idea = state.get("idea", "")
    stack_pref = state.get("stack_preference", "")

    prompt_parts = [
        "Research the technology landscape for building this application.",
        f"\nIDEA: {idea}",
        f"\nFEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}",
    ]
    if stack_pref:
        prompt_parts.append(f"\nSTACK PREFERENCE: {stack_pref}")
    prompt_parts.append(
        "\nWrite research_report.md to the working directory. "
        "Be thorough — the Architect will use this to make all technical decisions."
    )

    result = await run_agent(
        persona="Principal Research Engineer",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="\n".join(prompt_parts),
        cwd=str(ws),
        project_dir=project_dir,
        stage="research",
        ui=ui,
        model=state.get("model"),
        max_turns=30,
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
    )

    # Read the report
    report_path = ws / "research_report.md"
    research_report = load_text_artifact(report_path) or result.text
    save_artifact(project_dir, "research_report.md", research_report)

    # Clean up
    cleanup_workspace_files(report_path)

    stage_teardown(project_dir, "research", ui)

    return {
        "research_report": research_report,
        "current_stage": "research",
    }
