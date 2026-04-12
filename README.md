# Kindle

AI-powered application factory. Describe what you want in a sentence, answer a
few questions, and get a complete, runnable project with tests, documentation,
and git history.

## How It Works

```
Grill → Research → Architect → Dev → QA → Package
```

**6 stages, 6 AI agents, 1 runnable project.**

1. **Grill** — The machine asks you focused questions with recommended answers.
   Your spec emerges from the conversation, not from a blank page. Type "done"
   at any point to proceed with defaults.

2. **Research** — Investigates the technology landscape, libraries, patterns,
   and prior art relevant to your app.

3. **Architect** — Designs the tech stack, directory structure, data model, and
   splits work into parallel dev tasks with non-overlapping directory scopes.

4. **Dev** — Multiple agents build simultaneously, each in their own directory.
   Tests are written alongside code. Concurrency controlled by semaphore.

5. **QA** — Two-part validation:
   - **Technical QA**: tests, lint, format, type check + integration tests
   - **Product Audit**: anti-slop detection (no placeholders, no boilerplate)
   - Self-healing loop back to Dev if issues found (up to 10 retries).

6. **Package** — Ensures the project runs, generates a README, initializes git
   with atomic commits (one per dev task), and delivers a complete project.

**You never write a PRD.** You answer questions. The machine builds.

## Installation

```bash
# Clone and install
git clone https://github.com/Kdurkee45/kindle.git
cd kindle
uv sync

# Set up API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Requires:
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

## Usage

```bash
# Basic — describe what you want
kindle "a real-time collaborative task manager with team workspaces"

# With stack preference
kindle "an API for managing restaurant reservations" --stack fastapi

# Fully autonomous (auto-answer Grill questions with defaults)
kindle "a personal finance tracker" --auto-approve

# Control concurrency
kindle "an e-commerce store" --concurrency 6

# Review architecture before building
kindle "a social media dashboard" --review-arch

# Set output directory
kindle "a CLI tool for managing dotfiles" --output ~/projects/dotfiles

# Verbose output
kindle "a weather app" -v

# Resume a previous session
kindle resume ~/.kindle/projects/proj_XXXXX --from dev

# List past sessions
kindle list
```

## Safety Model

1. **Tests written during build** — Dev agents write tests alongside code,
   not as an afterthought. QA writes integration tests after.
2. **Self-healing QA loop** — if tests fail or quality issues are found,
   QA loops back to Dev for fixes (up to 10 retries).
3. **Anti-slop product audit** — detects AI-generated laziness: placeholder
   text, default boilerplate, generic copy, unused code.
4. **Atomic git history** — each dev task is a separate commit, reviewable
   and individually revertable.
5. **Scaffold then strip** — framework CLIs (create-next-app, etc.) provide
   correct infrastructure, boilerplate is removed before Dev begins.

## Configuration

Settings via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Required. Claude API key. | — |
| `KINDLE_MODEL` | Claude model to use. | `claude-opus-4-20250514` |
| `KINDLE_MAX_TURNS` | Max agent turns per stage. | `50` |

## Project Structure

```
src/kindle/
├── cli.py              # Typer CLI (build, resume, list)
├── config.py           # Settings from env/dotenv
├── state.py            # LangGraph typed state
├── graph.py            # Pipeline state machine
├── agent.py            # Claude Agent SDK wrapper
├── artifacts.py        # Persistent artifact storage
├── ui.py               # Rich terminal UI
└── stages/
    ├── grill.py        # Stage 1: focused Q&A
    ├── research.py     # Stage 2: tech landscape
    ├── architect.py    # Stage 3: system design + task split
    ├── dev.py          # Stage 4: parallel build
    ├── qa.py           # Stage 5: technical QA + product audit
    └── package.py      # Stage 6: package + deliver
```

## Related Projects

The trilogy: **Kindle** (ignite) → **Graft** (grow) → **Hone** (sharpen)

- **[Graft](https://github.com/Kdurkee45/graft)** — Add features to existing codebases
- **[Hone](https://github.com/Kdurkee45/hone)** — Optimize existing code quality

## License

TBD
