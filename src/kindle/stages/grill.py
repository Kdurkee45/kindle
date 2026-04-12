"""Grill stage — structured interrogation to turn a vague idea into a precise spec.

The Principal Product Interrogator asks focused questions one at a time with
recommended answers. The human can accept, override, or type 'done' to proceed.

In auto-approve mode, all recommended answers are used automatically.
"""

from __future__ import annotations

import json

from kindle.agent import run_agent
from kindle.artifacts import mark_stage_complete, save_artifact, workspace_path
from kindle.state import KindleState
from kindle.ui import UI

INTERROGATION_SYSTEM_PROMPT = """\
You are a Principal Product Interrogator. Your job is to turn a vague app idea
into a precise, buildable specification through focused questioning.

You have the user's initial idea. Generate a list of focused questions to fill
in the gaps. Ask about:

1. Core functionality — what are the must-have features?
2. User model — who uses this? Auth required?
3. Data model — key entities and relationships?
4. Tech preferences — stack constraints? (React, Python, etc.)
5. Scope boundaries — what's explicitly out of scope?
6. Design preferences — minimal? polished? dashboard-heavy?
7. Platform — web, mobile, CLI, API?

## Output

Write `open_questions.json` to the working directory with an array of objects:

```json
[
  {
    "question": "What are the must-have features for the MVP?",
    "category": "core_functionality",
    "recommended_answer": "Based on the idea, I'd recommend: ..."
  }
]
```

Each question MUST have a recommended_answer — a sensible default based on the
idea and industry norms. The human should be able to say "yes" to most of them.

Generate 5-15 focused questions. Quality over quantity. Write the file to the
current working directory.
"""

COMPILE_SYSTEM_PROMPT = """\
You are a Principal Product Architect. You have completed a structured
interrogation about an application the human wants to build.

You have:
1. The original idea
2. A complete transcript of the Q&A session with all decisions
3. Optional stack preference

Your job: compile everything into a definitive feature specification.

## Output

Write `feature_spec.json` to the working directory with this shape:

```json
{
  "app_name": "TaskFlow",
  "idea": "original idea...",
  "decisions": [
    {
      "question": "...",
      "recommended": "...",
      "answer": "...",
      "category": "core_functionality|user_model|data_model|tech_preferences|scope|design|platform",
      "implications": ["needs auth system", "needs real-time sync"]
    }
  ],
  "core_features": ["feature1", "feature2"],
  "user_stories": [
    {"as_a": "user", "i_want": "...", "so_that": "...", "acceptance_criteria": ["..."]}
  ],
  "data_model": {
    "entities": [{"name": "Task", "fields": ["id", "title", "status"], "relationships": ["belongs_to User"]}]
  },
  "tech_constraints": ["Must use React", "Python backend preferred"],
  "scope": {
    "mvp": ["item1", "item2"],
    "out_of_scope": ["item3", "item4"]
  },
  "design_direction": "minimal and clean",
  "platform": "web"
}
```

Be precise. Every decision must be documented. Every implication must be noted.
This spec drives the entire pipeline — ambiguity here means bugs later.

Write the file to the current working directory.
"""


def _fill_remaining_defaults(questions: list[dict], start_index: int) -> tuple[list[str], list[dict]]:
    """Fill remaining questions with their recommended answers after early exit."""
    transcript_lines: list[str] = []
    decisions: list[dict] = []
    for j, remaining in enumerate(questions[start_index - 1 :], start_index):
        rq = remaining.get("question", str(remaining)) if isinstance(remaining, dict) else str(remaining)
        rr = (
            remaining.get("recommended_answer", "No recommendation")
            if isinstance(remaining, dict)
            else "No recommendation"
        )
        rc = remaining.get("category", "general") if isinstance(remaining, dict) else "general"
        transcript_lines.append(f"Q{j} [{rc}]: {rq}")
        transcript_lines.append(f"  Recommended: {rr}")
        transcript_lines.append(f"  Answer: {rr} (auto-default)")
        transcript_lines.append("")
        decisions.append({"question": rq, "recommended": rr, "answer": rr, "category": rc})
    return transcript_lines, decisions


def _walk_questions(questions: list[dict], ui: UI) -> tuple[list[str], list[dict]]:
    """Walk through questions interactively, handling 'done' early-exit."""
    transcript_lines: list[str] = []
    decisions: list[dict] = []

    for i, q in enumerate(questions, 1):
        question = q.get("question", str(q)) if isinstance(q, dict) else str(q)
        recommended = q.get("recommended_answer", "No recommendation") if isinstance(q, dict) else "No recommendation"
        category = q.get("category", "general") if isinstance(q, dict) else "general"

        answer = ui.grill_question(question, recommended, category, i)

        # Check for early exit
        if answer.lower() == "done":
            ui.info("User requested early exit — using defaults for remaining questions.")
            # Record the current question with its recommended answer
            transcript_lines.append(f"Q{i} [{category}]: {question}")
            transcript_lines.append(f"  Recommended: {recommended}")
            transcript_lines.append(f"  Answer: {recommended} (auto-default, user said done)")
            transcript_lines.append("")
            decisions.append(
                {"question": question, "recommended": recommended, "answer": recommended, "category": category}
            )
            # Fill in remaining questions with defaults
            remaining_lines, remaining_decisions = _fill_remaining_defaults(questions, i + 1)
            transcript_lines.extend(remaining_lines)
            decisions.extend(remaining_decisions)
            break

        transcript_lines.append(f"Q{i} [{category}]: {question}")
        transcript_lines.append(f"  Recommended: {recommended}")
        transcript_lines.append(f"  Answer: {answer}")
        transcript_lines.append("")

        decisions.append({"question": question, "recommended": recommended, "answer": answer, "category": category})

    return transcript_lines, decisions


async def grill_node(state: KindleState, ui: UI) -> dict:
    """LangGraph node: interrogate the human to build a complete feature spec."""
    ui.stage_start("grill")
    project_dir = state["project_dir"]
    idea = state.get("idea", "")
    stack_pref = state.get("stack_preference", "")
    ws = workspace_path(project_dir)

    # Generate questions via agent
    gen_prompt = f"Generate focused questions for building this application.\n\nIDEA: {idea}"
    if stack_pref:
        gen_prompt += f"\nSTACK PREFERENCE: {stack_pref}"
    gen_prompt += "\n\nWrite open_questions.json to the working directory."

    await run_agent(
        persona="Principal Product Interrogator",
        system_prompt=INTERROGATION_SYSTEM_PROMPT,
        user_prompt=gen_prompt,
        cwd=str(ws),
        project_dir=project_dir,
        stage="grill",
        ui=ui,
        model=state.get("model"),
        max_turns=10,
        allowed_tools=["Read", "Write", "Bash"],
    )

    # Load generated questions
    questions_path = ws / "open_questions.json"
    open_questions: list[dict] = []
    if questions_path.exists():
        try:
            data = json.loads(questions_path.read_text())
            open_questions = data if isinstance(data, list) else []
        except json.JSONDecodeError:
            ui.error("Failed to parse open_questions.json")

    if not open_questions:
        ui.info("No questions generated — proceeding with idea as-is.")

    # Walk through questions one at a time
    transcript_lines, _decisions = _walk_questions(open_questions, ui)

    grill_transcript = "\n".join(transcript_lines)
    save_artifact(project_dir, "grill_transcript.md", grill_transcript)

    # Compile decisions into feature spec using an agent
    ui.info("Compiling feature specification from decisions...")

    compile_prompt = (
        f"Compile the feature specification.\n\n"
        f"IDEA: {idea}\n\n"
        f"STACK PREFERENCE: {stack_pref or 'None — choose the best fit'}\n\n"
        f"GRILL TRANSCRIPT:\n{grill_transcript}\n\n"
        f"Write feature_spec.json to the working directory."
    )

    await run_agent(
        persona="Principal Product Architect",
        system_prompt=COMPILE_SYSTEM_PROMPT,
        user_prompt=compile_prompt,
        cwd=str(ws),
        project_dir=project_dir,
        stage="grill_compile",
        ui=ui,
        model=state.get("model"),
        max_turns=10,
        allowed_tools=["Read", "Write", "Bash"],
    )

    # Read compiled spec
    spec_path = ws / "feature_spec.json"
    feature_spec: dict = {}
    if spec_path.exists():
        try:
            feature_spec = json.loads(spec_path.read_text())
        except json.JSONDecodeError:
            ui.error("Failed to parse feature_spec.json.")

    save_artifact(project_dir, "feature_spec.json", json.dumps(feature_spec, indent=2))

    # Clean up temp files
    for p in [questions_path, spec_path]:
        if p.exists():
            p.unlink()

    mark_stage_complete(project_dir, "grill")
    ui.stage_done("grill")

    return {
        "feature_spec": feature_spec,
        "grill_transcript": grill_transcript,
        "current_stage": "grill",
    }
