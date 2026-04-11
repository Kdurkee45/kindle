# Kindle — Design Document

## Overview

An AI-powered pipeline for building complete applications from a short
human prompt. You describe what you want in a sentence. The machine asks
questions, researches the technical landscape, architects the system,
builds it in parallel, validates quality, and delivers a working project.

**The zero-to-one factory.** Part of a trilogy:
- **Kindle** — build from nothing (ignite)
- **Graft** — add features to existing code (grow)
- **Hone** — optimize existing code (sharpen)

Core philosophy: **You never write a PRD. You answer questions.
The machine does the rest.**

---

## How It Works

You say: _"Build me a real-time collaborative task manager with team workspaces."_

The machine:
1. **Grills you** — focused questions to understand scope, priorities, constraints
2. **Researches** — technology landscape, libraries, patterns, prior art
3. **Architects** — tech stack, directory structure, data model, parallel dev tasks
4. **Builds** — multiple agents implement in parallel, each scoped to non-overlapping directories
5. **Validates** — deterministic quality checks + AI review + anti-slop detection
6. **Packages** — working project directory, ready to run or deploy

---

## Pipeline Stages

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   GRILL ──► RESEARCH ──► ARCHITECT ──► DEV ──► QA ──► PACKAGE │
│     │                                   ↑      │              │
│  [human]                                └──────┘              │
│                                        (self-heal)            │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Grill:      interrogate the human to build a complete feature spec
Research:   gather technical context — libraries, patterns, prior art
Architect:  design the system and split work into parallel dev tasks
Dev:        multiple agents build in parallel, tests after each unit
QA:         deterministic checks + AI review + anti-slop audit
Package:    deliver a working, runnable project
```

Six stages. Six agent personas. One human touchpoint (Grill).

---

### Stage 1: GRILL

**Agent persona:** Principal Product Interrogator
**Purpose:** Turn a vague idea into a precise, buildable specification
through focused questioning.

This is the human-in-the-loop stage. Instead of asking the user to write
a PRD, the machine asks one question at a time with a recommended answer.
The user can accept the recommendation, override it, or skip it.

How it works:
- Start with the user's prompt (e.g., "a task manager with teams")
- Ask clarifying questions one at a time, depth-first:
  - Core functionality: what are the must-have features?
  - User model: who uses this? Auth required?
  - Data model: what are the key entities and relationships?
  - Tech preferences: any stack constraints? (React, Python, etc.)
  - Scope boundaries: what's explicitly out of scope?
  - Design preferences: minimal? polished? dashboard-heavy?
- Each question comes with a recommended answer based on the context so far
- The user can type "done" at any point to stop and proceed with defaults
- Self-regulating depth — the interrogator decides when it has enough

Output: `feature_spec.json` + `grill_transcript.md`

The spec is structured data — not a wall of text. It includes:
- Core features (prioritized)
- User stories with acceptance criteria
- Data model sketch
- Tech constraints
- Scope boundaries
- Design direction

**Human involvement:** Required. This IS the input mechanism. But it's
low-effort — answer 5-15 questions, most with one-word overrides of
sensible defaults.

**Auto-approve mode:** For fully autonomous runs, the Grill uses all
recommended answers automatically. Good for demos, testing, or when you
genuinely don't care about the details.

---

### Stage 2: RESEARCH

**Agent persona:** Principal Research Engineer
**Purpose:** Gather technical context needed to make good architecture
decisions.

What it does:
- Read the spec from Grill
- Research the technology landscape for the chosen stack
- Identify relevant libraries, frameworks, and APIs
- Find patterns and best practices for the app type
- Surface potential pitfalls and edge cases
- Check for existing solutions / prior art

Output: `research_report.md`

The Research agent is deliberately separated from the Architect to avoid
premature optimization. Research gathers facts. Architecture makes
decisions. Mixing them leads to the AI picking a framework before
understanding the full picture.

**Human involvement:** None.

---

### Stage 3: ARCHITECT

**Agent persona:** Principal Solution Architect
**Purpose:** Design the system and split the work into parallel dev tasks.

What it does:
- Read the spec and research report
- Choose the tech stack (with justification)
- Design the high-level architecture
- Define the directory structure
- Design the data model
- Define the API surface (if applicable)
- Split the work into parallel dev tasks with non-overlapping directory scopes

Output: `architecture.md` + `dev_tasks.json`

Each dev task includes:
- Task ID, title, description
- Directory scope (non-overlapping — enables parallel execution)
- Dependencies on other tasks (if any)
- Acceptance criteria
- Pattern references (how it should be built)

The non-overlapping directory constraint is critical for parallel dev.
Task A owns `src/components/`, Task B owns `src/api/`, Task C owns
`src/services/`. No conflicts, no merge issues.

**Human involvement:** None. But an optional `--review-arch` flag could
pause here for human review of the architecture before building starts.

---

### Stage 4: DEV (Parallel Build)

**Agent persona:** Principal Software Engineer (one per task)
**Purpose:** Build the application. Multiple agents work in parallel,
each scoped to their assigned directory.

How it works:
- Topological sort of dev tasks by dependencies
- Launch agents in parallel (bounded by concurrency semaphore)
- Each agent:
  1. Reads the spec, architecture, and its task description
  2. Builds everything in its directory scope
  3. Writes complete, working code — no placeholders, no TODOs
  4. Writes tests for what it builds
  5. Ensures its code compiles/lints clean

Key constraints:
- **Write tests as you build.** Every dev task includes writing tests
  for the code it produces. This is non-negotiable — the QA stage needs
  tests to validate against.
- **No placeholders.** Every file must be complete and functional.
  "TODO: implement later" is a build failure.
- **Stay in scope.** Each agent only writes files in its assigned
  directory. Cross-cutting concerns (shared types, utilities) get their
  own task.
- **Follow the architecture.** The architecture document is the source
  of truth. Don't improvise.

Concurrency control:
- Default: 4 concurrent agents (configurable via `--concurrency`)
- Semaphore prevents API rate limiting
- Each agent runs independently — no shared state during execution

**Human involvement:** None.

---

### Stage 5: QA (Quality Assurance)

**Agent persona:** Principal Quality Engineer + VP of Product (dual audit)
**Purpose:** Validate the build from both technical and product
perspectives.

Two sub-stages that run in sequence:

#### 5A: Technical QA

- Run the test suite (written by Dev agents)
- Run linter (auto-detected for the stack)
- Run type checker (if applicable)
- Run formatter check
- Check for missing files referenced in architecture
- Check for broken imports / unresolved dependencies
- Verify the project builds / compiles

If technical QA fails → loop back to Dev for fixes (self-healing).
Max retries: 10 (configurable via `--qa-retries`).

#### 5B: Product Audit (Anti-Slop)

- Read every source file
- Compare against the spec — does the app do what was asked?
- Check for AI-generated laziness:
  - Placeholder text ("Lorem ipsum", "Example content")
  - Default framework boilerplate that wasn't customized
  - Generic copy that doesn't match the app's purpose
  - Unused code / dead imports
  - Inconsistent design (mismatched styles, mixed patterns)
- Check for user-facing quality:
  - Error handling (not just console.log)
  - Input validation
  - Reasonable defaults
  - Responsive design (if web)

If product audit finds issues → loop back to Dev for fixes.
Max retries: 10 (configurable via `--cpo-retries`).

Output: `qa_report.md` + `product_audit.md`

**Human involvement:** None.

---

### Stage 6: PACKAGE

**Agent persona:** Principal DevOps Engineer
**Purpose:** Ensure the project is runnable and well-documented.

What it does:
- Verify all dependencies are installed and locked
- Ensure the project runs (`npm start`, `python -m app`, etc.)
- Generate a README.md with:
  - What the app does
  - How to run it
  - Tech stack
  - Project structure
  - Environment variables needed
- Set up a basic Dockerfile (if applicable)
- Initialize git repo with a clean first commit

Output: A complete, runnable project directory.

The Package stage does NOT deploy. Deployment is a separate concern —
the user decides where and how to deploy. Kindle produces a project,
not a hosted app.

**Human involvement:** None. The output is the project directory.

---

## Human-in-the-Loop Summary

| Stage     | Human Role              | Gate Type  | Can Skip?          |
|-----------|-------------------------|------------|--------------------|
| Grill     | Answer questions        | Required   | Yes (--auto-approve uses defaults) |
| Research  | None                    | —          | —                  |
| Architect | None (optional review)  | Optional   | Yes (--review-arch) |
| Dev       | None                    | —          | —                  |
| QA        | None (self-healing)     | —          | —                  |
| Package   | None                    | —          | —                  |

The Grill is the only required human touchpoint. Everything else is
autonomous with self-healing loops.

---

## Key Differences from Nerdery

| Aspect              | Nerdery (original)          | Kindle (new)                  |
|---------------------|-----------------------------|-------------------------------|
| Input model         | User writes app idea        | Grill interrogation           |
| Spec generation     | AI writes PRD from idea     | Spec emerges from Q&A         |
| Human gates         | Spec review + CPO review    | Grill only (+ optional arch)  |
| Test strategy       | Tests optional              | Tests mandatory (written during Dev) |
| Slop detection      | CPO stage                   | Integrated into QA            |
| Output              | Deployed app (Netlify)      | Runnable project directory    |
| Deploy              | Built-in Netlify deploy     | Not included (separate concern) |
| Cost estimation     | None                        | Shown before build starts     |
| Quality checks      | Basic lint/test in QA       | Full lint + format + type check + tests |
| Self-healing        | QA ↔ Dev loop               | Technical QA ↔ Dev + Product Audit ↔ Dev |
| Concurrency         | Parallel dev                | Same, with semaphore control  |

---

## Key Differences from Graft

| Aspect              | Graft (features)            | Kindle (greenfield)           |
|---------------------|-----------------------------|-------------------------------|
| Starting point      | Existing codebase           | Nothing                       |
| Discovery stage     | Yes (understand existing)   | No (nothing to discover)      |
| Architecture        | Fits existing patterns      | Designs from scratch          |
| Dev model           | Sequential (cross-cutting)  | Parallel (scoped directories) |
| Test strategy       | Rely on existing tests      | Write tests during build      |
| Safety model        | Revert on existing test failure | Revert on new test failure  |
| Output              | PR on existing repo         | New project directory         |

---

## State Shape

```python
class KindleState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────
    idea: str                             # The user's initial prompt
    project_id: str
    project_dir: str                      # Working directory for the build

    # ── User preferences (from CLI flags) ───────────────────
    stack_preference: str                 # e.g. "react", "nextjs", "python"
    auto_approve: bool
    max_concurrent_agents: int
    max_qa_retries: int
    max_cpo_retries: int

    # ── Stage artifacts ─────────────────────────────────────
    feature_spec: dict                    # From Grill
    grill_transcript: str                 # From Grill
    research_report: str                  # From Research
    architecture: str                     # From Architect
    dev_tasks: list[dict]                 # From Architect
    qa_report: str                        # From QA
    product_audit: str                    # From QA
    package_readme: str                   # From Package

    # ── Quality tracking ────────────────────────────────────
    qa_passed: bool
    cpo_passed: bool
    qa_retries: int
    cpo_retries: int

    # ── Settings ────────────────────────────────────────────
    model: str
    max_agent_turns: int

    # ── Pipeline state ──────────────────────────────────────
    current_stage: str
```

---

## CLI Interface

```bash
# Basic — describe what you want
kindle "a real-time collaborative task manager with team workspaces"

# With stack preference
kindle "an API for managing restaurant reservations" --stack fastapi

# Fully autonomous (auto-answer Grill questions)
kindle "a personal finance tracker" --auto-approve

# Control concurrency
kindle "an e-commerce store" --concurrency 6

# Review architecture before building
kindle "a social media dashboard" --review-arch

# Set output directory
kindle "a CLI tool for managing dotfiles" --output ~/projects/dotfiles

# Verbose output
kindle "a weather app" -v
```

---

## Lessons from Hone and Graft

### From Hone:
1. **Tests first.** Hone taught us that test-writing should be integral,
   not an afterthought. Kindle's Dev agents write tests alongside code.
2. **Verification must match CI rigor.** QA runs lint, format check,
   type check, AND tests. No gaps between what QA checks and what a
   real CI pipeline would catch.
3. **Use the project's own tools.** QA uses the project's own python/node
   interpreter, not Kindle's. This ensures verification matches reality.
4. **Cost estimation helps.** Show estimated cost before building starts.
5. **Self-healing loops need limits.** 10 retries, then move on.

### From Graft:
1. **Grill > PRD.** Questions with recommended answers are better than
   asking the user to write a spec. Lower effort, higher quality.
2. **Research before architecture.** Gathering facts before making
   decisions produces better architecture.
3. **Pattern references matter.** Dev agents need to know HOW to build,
   not just WHAT. Architecture should include pattern references.

### From Nerdery:
1. **Parallel dev works.** Non-overlapping directory scopes enable
   concurrent agent execution without conflicts.
2. **Slop detection is valuable.** The CPO audit catches AI laziness
   that technical QA misses. Keep it.
3. **Don't deploy.** Deployment is opinionated and fragile. Deliver
   a runnable project, let the user deploy how they want.

---

## Open Questions

1. **Should Kindle output a git repo with atomic commits?** Each dev
   task could be a separate commit, making the build history reviewable.
   Or just deliver a flat directory with everything in it.

2. **Integration testing.** Individual dev tasks write their own tests,
   but who writes the integration tests that verify tasks work together?
   Options: (a) the QA agent writes integration tests, (b) the Architect
   stage defines integration test specs, (c) skip for V1.

3. **Dependency installation.** Should the Dev agents install dependencies
   as they go, or should the Package stage handle all dependency resolution?
   Early installation means dev agents can verify their code actually runs.
   Late installation means fewer conflicts between parallel agents.

4. **Frontend scaffolding.** For web apps, should Kindle use a framework
   CLI (create-react-app, create-next-app, etc.) to bootstrap, or build
   from scratch? Scaffolding gives you the right structure and config but
   also gives you boilerplate the CPO audit will flag.

5. **Multi-language projects.** Some apps need both a frontend and a
   backend (React + FastAPI). How should the Architect split that? Two
   services with separate dependency manifests? A monorepo layout?
