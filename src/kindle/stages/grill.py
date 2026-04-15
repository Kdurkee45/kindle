"""Grill stage — adaptive conversational interrogation.

A real-time conversation loop where the agent drives the interrogation,
adapting each question based on everything it's learned so far. The agent
asks one question at a time, digs deeper on vague answers, skips irrelevant
questions, and stops when it has enough context to build a spec.

In auto-approve mode, the agent's recommended answer is used for every
question. The conversation still adapts — recommendations for later
questions are informed by earlier ones.
"""

from __future__ import annotations

import json

from kindle.agent import run_agent
from kindle.artifacts import mark_stage_complete, save_artifact
from kindle.stages._helpers import stage_setup
from kindle.state import KindleState
from kindle.ui import UI

MAX_QUESTIONS = 25

CONVERSATION_SYSTEM_PROMPT = """\
You are a Principal Product Interrogator conducting a structured discovery
conversation. Your goal: turn a vague app idea into a precise, buildable
specification through adaptive questioning.

RULES:
1. Ask ONE question at a time. Wait for the answer before asking the next.
2. Every question MUST include a recommended_answer — your best guess
   based on everything you've heard so far.
3. Adapt your questions to the user's answers. If they say "mobile app,"
   don't ask about server-side rendering. If they say "simple MVP," don't
   ask about enterprise features.
4. Dig deeper when answers are vague. "It should handle payments" → ask
   about payment providers, subscription vs one-time, refund policy.
5. Skip questions when prior answers already cover the topic.
6. Track what you know and what you don't. Stop when you have enough
   to build the app confidently.
7. Think in layers:
   - Layer 1: Platform, core features, user model (broad strokes)
   - Layer 2: Data model, key workflows, integrations (structure)
   - Layer 3: Edge cases, error handling, design specifics (detail)
   You don't need to reach Layer 3 for every topic — only for the
   critical paths.
8. Be conversational, not robotic. Acknowledge what the user said before
   asking the next question. Reference their previous answers.

RESPONSE FORMAT:
You MUST respond with a single JSON object and nothing else.

When you need to ask a question:
{
  "status": "question",
  "question": "Your question here",
  "category": "core_functionality|user_model|data_model|tech|scope|
              design|platform|integration|workflow|auth|api|deployment",
  "recommended_answer": "Your recommendation based on what you've heard so far",
  "why_asking": "Brief explanation of why this matters for the build"
}

When you have enough context to build the app:
{
  "status": "done",
  "summary": "Here's what I understand you want to build: ...",
  "assumptions": ["Assumption 1 — filling a gap the user didn't address", "..."],
  "confidence": "high|medium|low"
}

Do NOT ask more than 25 questions. If you've asked 20 and still have
gaps, wrap up with assumptions and set confidence accordingly.

Do NOT include any text outside the JSON object. No preamble, no
explanation — just the JSON.
"""

COMPILE_SYSTEM_PROMPT = """\
You are a Principal Product Architect. You have the complete transcript
of a structured discovery conversation about an application the human
wants to build.

Your job: compile everything into a definitive feature specification.

## Output

Write `feature_spec.json` to the working directory with this shape:

```json
{
  "app_name": "<name>",
  "idea": "<original idea>",
  "summary": "<one-paragraph description of what we're building>",
  "decisions": [
    {
      "question": "...",
      "answer": "...",
      "category": "...",
      "implications": ["needs auth system", "needs real-time sync"]
    }
  ],
  "assumptions": [
    "Assumed email/password auth since no preference stated",
    "..."
  ],
  "confidence": "high|medium|low",
  "core_features": ["feature1", "feature2"],
  "user_stories": [
    {"as_a": "user", "i_want": "...", "so_that": "...", "acceptance_criteria": ["..."]}
  ],
  "data_model": {
    "entities": [
      {"name": "Task", "fields": ["id", "title", "status"], "relationships": ["belongs_to User"]}
    ]
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

Be precise. Every decision must be documented. Every assumption must be
noted. This spec drives the entire pipeline — ambiguity here means bugs later.

Write the file to the current working directory.
"""


def _parse_agent_response(text: str) -> dict:
    """Extract JSON from the agent's response text.

    The agent should respond with pure JSON, but sometimes wraps it in
    markdown code fences or adds preamble text. This handles those cases.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try finding first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return {"status": "error", "message": "Failed to parse agent response"}


def _build_history_prompt(idea: str, stack_pref: str, history: list[dict]) -> str:
    """Build the prompt for the next conversation turn.

    Includes the full Q&A history so the agent has complete context.
    """
    parts = [
        f"APP IDEA: {idea}",
    ]
    if stack_pref:
        parts.append(f"STACK PREFERENCE: {stack_pref}")

    if history:
        parts.append("\nCONVERSATION SO FAR:")
        for entry in history:
            if entry["role"] == "agent":
                data = entry["data"]
                parts.append(f"\nQ{entry['turn']} [{data.get('category', '?')}]: {data.get('question', '?')}")
                parts.append(f"  Recommended: {data.get('recommended_answer', '?')}")
                parts.append(f"  Why asked: {data.get('why_asking', '?')}")
            elif entry["role"] == "user":
                parts.append(f"  User answered: {entry['answer']}")

    parts.append("\nAsk your next question, or respond with status 'done' if you have enough context.")
    return "\n".join(parts)


async def _ask_one_question(
    idea: str,
    stack_pref: str,
    history: list[dict],
    ws_path: str,
    project_dir: str,
    ui: UI,
    model: str | None,
    turn: int,
) -> dict:
    """Run one conversation turn — agent produces the next question or done signal."""
    prompt = _build_history_prompt(idea, stack_pref, history)

    result = await run_agent(
        persona="Principal Product Interrogator",
        system_prompt=CONVERSATION_SYSTEM_PROMPT,
        user_prompt=prompt,
        cwd=ws_path,
        project_dir=project_dir,
        stage=f"grill_q{turn}",
        ui=ui,
        model=model,
        max_turns=3,
        allowed_tools=["Write"],
    )

    return _parse_agent_response(result.text)


def _append_assumptions(transcript_lines: list[str], assumptions: list[str]) -> None:
    """Append assumption lines to transcript."""
    if assumptions:
        transcript_lines.append("")
        transcript_lines.append("**Assumptions:**")
        for a in assumptions:
            transcript_lines.append(f"- {a}")


def _handle_done_response(
    response: dict,
    turn: int,
    transcript_lines: list[str],
    ui: UI,
) -> tuple[str, list[str]]:
    """Handle agent 'done' status — returns (summary, assumptions)."""
    done_summary = response.get("summary", "")
    assumptions = response.get("assumptions", [])
    confidence = response.get("confidence", "medium")
    transcript_lines.append(f"**Agent concluded after {turn - 1} questions.**")
    transcript_lines.append("")
    transcript_lines.append(f"**Summary:** {done_summary}")
    transcript_lines.append(f"**Confidence:** {confidence}")
    _append_assumptions(transcript_lines, assumptions)
    ui.info(f"Grill complete after {turn - 1} questions (confidence: {confidence}).")
    return done_summary, assumptions


def _record_question_in_transcript(
    transcript_lines: list[str],
    turn: int,
    category: str,
    question: str,
    why_asking: str,
    recommended: str,
    answer: str,
) -> None:
    """Append a Q&A exchange to the transcript."""
    transcript_lines.append(f"### Q{turn} [{category}]")
    transcript_lines.append("")
    transcript_lines.append(f"**{question}**")
    transcript_lines.append("")
    transcript_lines.append(f"*Why I'm asking: {why_asking}*")
    transcript_lines.append("")
    transcript_lines.append(f"Recommended: {recommended}")
    transcript_lines.append("")
    transcript_lines.append(f"**Answer:** {answer}")
    transcript_lines.append("")


async def _handle_early_exit(
    history: list[dict],
    response: dict,
    turn: int,
    idea: str,
    stack_pref: str,
    ws: object,
    project_dir: str,
    ui: UI,
    state: KindleState,
    transcript_lines: list[str],
) -> tuple[str, list[str]]:
    """Handle user typing 'done' — wrap up with assumptions."""
    ui.info("User requested early exit — agent will fill remaining gaps with assumptions.")
    history.append({"role": "agent", "data": response, "turn": turn})
    early_exit_msg = "I'm done answering questions. Fill in any remaining gaps with your best judgment and wrap up."
    history.append({"role": "user", "answer": early_exit_msg})
    wrap_up = await _ask_one_question(
        idea,
        stack_pref,
        history,
        str(ws),
        project_dir,
        ui,
        state.get("model"),
        turn + 1,
    )
    _done_summary = ""
    assumptions: list[str] = []
    if wrap_up.get("status") == "done":
        done_summary = wrap_up.get("summary", "")
        assumptions = wrap_up.get("assumptions", [])

    transcript_lines.append(f"**User ended conversation at Q{turn}. Agent filled gaps.**")
    _append_assumptions(transcript_lines, assumptions)
    return done_summary, assumptions


async def grill_node(state: KindleState, ui: UI) -> dict:
    """LangGraph node: adaptive conversational interrogation."""
    project_dir, ws = stage_setup(state, ui, "grill")
    idea = state.get("idea", "")
    stack_pref = state.get("stack_preference", "")
    auto_approve = state.get("auto_approve", False)

    history: list[dict] = []
    transcript_lines: list[str] = [
        "# Grill Transcript",
        "",
        f"**Idea:** {idea}",
        f"**Stack preference:** {stack_pref or 'None'}",
        f"**Mode:** {'auto-approve' if auto_approve else 'interactive'}",
        "",
        "---",
        "",
    ]
    decisions: list[dict] = []
    assumptions: list[str] = []
    _done_summary = ""

    for turn in range(1, MAX_QUESTIONS + 1):
        # Agent decides what to ask next (or that it's done)
        response = await _ask_one_question(
            idea,
            stack_pref,
            history,
            str(ws),
            project_dir,
            ui,
            state.get("model"),
            turn,
        )

        if response.get("status") == "error":
            ui.error(f"Grill agent returned unparseable response on turn {turn}. Wrapping up.")
            break

        if response.get("status") == "done":
            _done_summary, assumptions = _handle_done_response(
                response,
                turn,
                transcript_lines,
                ui,
            )
            break

        # Extract question data
        question = response.get("question", "")
        recommended = response.get("recommended_answer", "")
        category = response.get("category", "general")
        why_asking = response.get("why_asking", "")

        if not question:
            ui.error(f"Grill agent returned empty question on turn {turn}. Wrapping up.")
            break

        # Get user's answer
        answer = ui.grill_question(
            question=question,
            recommended=recommended,
            category=category,
            number=turn,
            why_asking=why_asking,
        )

        # Check for early exit
        if answer.lower() == "done":
            _done_summary, assumptions = await _handle_early_exit(
                history,
                response,
                turn,
                idea,
                stack_pref,
                ws,
                project_dir,
                ui,
                state,
                transcript_lines,
            )
            break

        # Record in history
        history.append({"role": "agent", "data": response, "turn": turn})
        history.append({"role": "user", "answer": answer})

        # Record in transcript
        _record_question_in_transcript(
            transcript_lines,
            turn,
            category,
            question,
            why_asking,
            recommended,
            answer,
        )

        decisions.append(
            {
                "question": question,
                "recommended": recommended,
                "answer": answer,
                "category": category,
                "why_asking": why_asking,
            }
        )

    grill_transcript = "\n".join(transcript_lines)
    save_artifact(project_dir, "grill_transcript.md", grill_transcript)

    # Compile decisions into feature spec
    ui.info("Compiling feature specification from conversation...")

    compile_prompt = (
        f"Compile the feature specification.\n\n"
        f"IDEA: {idea}\n\n"
        f"STACK PREFERENCE: {stack_pref or 'None — choose the best fit'}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{grill_transcript}\n\n"
        f"ASSUMPTIONS FROM INTERROGATOR:\n"
        + json.dumps(assumptions, indent=2)
        + "\n\nWrite feature_spec.json to the working directory."
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
    for name in ["feature_spec.json", "open_questions.json"]:
        p = ws / name
        p.unlink(missing_ok=True)

    mark_stage_complete(project_dir, "grill")
    ui.stage_done("grill")

    return {
        "feature_spec": feature_spec,
        "grill_transcript": grill_transcript,
        "current_stage": "grill",
    }
