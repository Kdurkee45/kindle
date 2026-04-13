# Grill Stage Redesign — Conversational Interrogation

## Problem

The current Grill generates a static list of 5-15 questions upfront, before
seeing any user answers. This produces a form-filling experience, not a
conversation. The agent can't:

- Adapt questions based on previous answers
- Dig deeper on vague answers
- Skip irrelevant questions when earlier answers make them unnecessary
- Follow up on surprising or complex answers
- Recognize when it has enough context to stop

## Goal

Replace the static question list with a **real-time conversational loop**
where the agent drives the interrogation, adapting each question based on
everything it's learned so far.

The user experience should feel like talking to a sharp product manager
who's trying to understand exactly what you want to build — not filling
out a form.

---

## Architecture

### Current (broken)

```
Agent generates 15 questions → User answers each → Agent compiles spec
     (one shot)                  (no adaptation)      (one shot)
```

### Proposed

```
┌─────────────────────────────────────────────┐
│           CONVERSATION LOOP                  │
│                                              │
│  Agent sees: idea + full Q&A history         │
│  Agent decides: ask next question OR done    │
│  User answers (or accepts recommendation)    │
│  History updated                             │
│  Loop                                        │
│                                              │
│  Exit when: agent says DONE, user says       │
│  "done", or max questions reached            │
│                                              │
└─────────────────────────────────────────────┘
                    │
                    ▼
         Agent compiles spec from
         full conversation history
```

### Key Design Decisions

**1. Single agent session, maintained across the entire conversation.**

Not separate calls per question. One agent session that accumulates context
through the conversation. Each turn, the agent sees the full history and
decides what to explore next.

**2. The agent controls the flow, not the code.**

The code's job is simple: relay the agent's question to the user, relay
the user's answer back to the agent. The agent decides:
- What to ask next
- Whether to dig deeper on the last answer
- Whether it has enough context to stop
- What to recommend as a default answer

**3. Structured output per turn.**

Each agent turn produces a JSON response:

```json
{
  "status": "question",
  "question": "What are the must-have features for the MVP?",
  "category": "core_functionality",
  "recommended_answer": "Based on a task manager, I'd recommend: task CRUD, assignment, due dates, status tracking",
  "reasoning": "Starting broad to understand core scope before diving into specifics",
  "depth": 1
}
```

Or when done:

```json
{
  "status": "done",
  "summary": "I have a clear picture of the app. Here's what I understand: ...",
  "confidence": "high",
  "gaps": []
}
```

**4. Adaptive depth via the agent's judgment.**

The agent is prompted to:
- Start broad (platform, core features, user model)
- Go deeper on areas that are ambiguous or complex
- Skip areas that are already well-defined
- Follow threads — if the user mentions "real-time collaboration," ask
  about conflict resolution, presence indicators, WebSocket vs SSE
- Cap itself at ~20 questions to avoid fatigue

**5. User can still type "done" at any point.**

Remaining context gaps get filled with the agent's best recommendations,
documented as assumptions in the spec.

---

## Conversation Protocol

### Agent System Prompt (core instructions)

```
You are a Principal Product Interrogator conducting a structured
discovery conversation. Your goal: turn a vague idea into a precise,
buildable specification through adaptive questioning.

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

RESPONSE FORMAT:
Respond with a JSON object. Either a question:
{
  "status": "question",
  "question": "Your question here",
  "category": "core_functionality|user_model|data_model|tech|scope|design|platform|integration|workflow",
  "recommended_answer": "Your recommendation here",
  "why_asking": "Brief explanation of why this matters"
}

Or when you have enough context:
{
  "status": "done",
  "summary": "Here's what I understand you want to build: ...",
  "assumptions": ["Any gaps I'm filling with reasonable defaults"],
  "confidence": "high|medium|low"
}

Do NOT ask more than 25 questions. If you've asked 20 and still have
gaps, wrap up and note assumptions.
```

### Conversation Flow (code side)

```python
async def grill_node(state, ui):
    idea = state["idea"]
    history = []
    max_questions = 25

    # Initialize the conversation
    history.append({
        "role": "user",
        "content": f"I want to build: {idea}"
    })

    for turn in range(max_questions):
        # Agent sees full history, produces next question or done
        response = await agent_turn(history)

        if response["status"] == "done":
            ui.show_summary(response["summary"])
            break

        # Display question to user, get answer
        answer = ui.grill_question(
            question=response["question"],
            recommended=response["recommended_answer"],
            category=response["category"],
            why=response["why_asking"],
            turn=turn + 1,
        )

        if answer.lower() == "done":
            # User wants to stop — agent fills gaps with assumptions
            history.append({"role": "user", "content": "I'm done answering questions. Fill in any gaps with your best judgment."})
            final = await agent_turn(history)
            break

        # Add Q&A to history
        history.append({"role": "assistant", "content": json.dumps(response)})
        history.append({"role": "user", "content": answer})

    # Compile spec from full conversation
    spec = await compile_spec(idea, history)
    return spec
```

### Auto-Approve Mode

In auto-approve mode, the loop still runs but the user's answer is always
the recommended_answer. The agent still adapts — if its recommendation for
Q1 implies certain things about Q2, it adjusts Q2 accordingly.

```python
if auto_approve:
    answer = response["recommended_answer"]
    ui.info(f"  Auto: {answer}")
else:
    answer = ui.grill_question(...)
```

---

## What Changes in the UI

### Current
```
Q1 [scope]: What are the must-have features?
  Recommended: Task CRUD, team workspaces
  Your answer (Enter to accept):
```

### Proposed
```
━━━ Grill (1/~15) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  What are the must-have features for the MVP?

  Why I'm asking: Starting broad to understand core scope.
  My recommendation: Task CRUD, assignment, due dates, status tracking

  Your answer (Enter to accept, 'done' to finish):
  > _
```

The addition of "why I'm asking" gives the user context for each question.
The turn counter shows `1/~15` (approximate, since the total adapts).

---

## What Changes in the Spec Output

The compiled spec adds an `assumptions` field documenting where the agent
filled in gaps:

```json
{
  "app_name": "TaskFlow",
  "idea": "...",
  "decisions": [...],
  "assumptions": [
    "Assumed email/password auth since no auth preference stated",
    "Assumed PostgreSQL since data model has relational structure",
    "Assumed responsive web (no native mobile) since platform was 'web'"
  ],
  "confidence": "high",
  "core_features": [...],
  ...
}
```

This makes the spec honest about what was decided vs. what was inferred.

---

## Implementation Approach

### Option A: Agent SDK Conversation Mode

Use the Claude Agent SDK's multi-turn conversation capability. The agent
session stays open across all questions, maintaining full context.

Pros: True conversation, agent remembers everything naturally.
Cons: Long-running agent session, potential for context window issues
on very long conversations.

### Option B: Stateless Turns with History Injection

Each question is a separate agent call, but the full Q&A history is
injected into the prompt each time.

Pros: No long-running session, easier to debug.
Cons: Redundant token usage (re-sending full history each turn),
potential for context window issues on very long conversations.

### Recommendation: Option B

Stateless turns with history injection. It's simpler, more debuggable,
and the history won't exceed a few thousand tokens even with 25 questions.
Each turn is independent — if one fails, you don't lose the session. And
it's the same pattern the rest of the pipeline uses.

---

## Applies To

Both **Kindle** and **Graft**. The Grill stage is architecturally identical
in both — the only difference is the system prompt context:

- **Kindle:** "You're interrogating about a new app to build from scratch"
- **Graft:** "You're interrogating about a feature to add to an existing
  codebase" (with the Discover stage's output as additional context)

The conversation engine, UI interaction, and spec compilation are shared.
Could be extracted into a shared library, or just kept in sync across both.

---

## Migration Path

1. Rewrite `grill.py` in Kindle with the new conversational model
2. Port to Graft's `grill.py` with the adjusted system prompt
3. Update UI to show "why I'm asking" and adaptive turn counter
4. Update tests
5. Update design docs

The spec output format is backward-compatible — we're adding fields
(assumptions, confidence), not removing any.
