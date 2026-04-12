"""QA stage — technical validation + product audit with self-healing loop.

Two sub-stages run in sequence:
  5A: Technical QA — tests, lint, format, type check
  5B: Product Audit — anti-slop detection, spec compliance

If either fails, loops back to Dev for fixes (up to max retries).
"""

from __future__ import annotations

import json

from kindle.agent import run_agent
from kindle.artifacts import mark_stage_complete, save_artifact, workspace_path
from kindle.state import KindleState
from kindle.ui import UI

TECHNICAL_QA_SYSTEM_PROMPT = """\
You are a Principal Quality Engineer. Your job is to validate the built
application from a technical perspective.

## Pre-step: Install Dependencies
First, install all project dependencies. Look for:
- pyproject.toml / requirements.txt → use pip/uv
- package.json → use npm/yarn/pnpm
- go.mod → use go mod download
Install them in the working directory.

## Then Run Checks

1. **Test Suite** — Run the test suite (pytest, jest, go test, etc.)
2. **Linter** — Run the linter (ruff, eslint, golint, etc.)
3. **Formatter** — Check formatting (ruff format --check, prettier --check, etc.)
4. **Type Checker** — Run type checking (mypy, tsc --noEmit, etc.)
5. **Missing Files** — Check architecture for files that should exist but don't
6. **Broken Imports** — Check for unresolved imports/dependencies
7. **Build Verification** — Verify the project builds/compiles

IMPORTANT: Use the project's own Python/Node interpreter, NOT the system one.
Look for a virtualenv (`.venv/bin/python`, `node_modules/.bin/*`) and use it.

## Integration Tests
After running existing tests, write integration tests that verify cross-boundary
interactions (e.g., component → API → database). Write them in a test file
appropriate for the project's test framework.

## Output

Write `qa_report.md` to the working directory with:
- Each check: PASS or FAIL with details
- List of all issues found
- Overall verdict: PASS or FAIL

If ALL checks pass, the verdict is PASS. If ANY check fails, the verdict is FAIL.

Write the file to the current working directory.
"""

PRODUCT_AUDIT_SYSTEM_PROMPT = """\
You are a VP of Product performing an anti-slop audit. Read EVERY source file
in the project and compare against the feature spec.

## Check For

1. **Spec Compliance** — Does the app do what was asked? Check every feature.
2. **Placeholder Text** — "Lorem ipsum", "Example content", "TODO", "FIXME"
3. **Boilerplate** — Default framework content that wasn't customized
4. **Generic Copy** — Text that doesn't match the app's purpose
5. **Dead Code** — Unused imports, unreachable code, empty handlers
6. **Error Handling** — Not just console.log / print. Proper error messages.
7. **Input Validation** — Forms and APIs validate input properly
8. **Reasonable Defaults** — Config values make sense, not just "example"
9. **Consistency** — Consistent naming, styling, patterns throughout

## Output

Write `product_audit.md` to the working directory with:
- Each check category: PASS or FAIL with specific file:line references
- List of all slop found
- Overall verdict: PASS or FAIL

Write the file to the current working directory.
"""

DEV_FIX_SYSTEM_PROMPT = """\
You are a Principal Software Engineer. The QA stage found issues in the project.
Fix ALL of them. Do not introduce new issues.

Read the QA report carefully. Fix every issue mentioned. Ensure tests pass
after your fixes.
"""


def _find_workspace_python(ws: Path) -> str:
    """Find the project's own Python interpreter, falling back to system python."""
    candidates = [
        ws / ".venv" / "bin" / "python",
        ws / ".venv" / "bin" / "python3",
        ws / "venv" / "bin" / "python",
        ws / "venv" / "bin" / "python3",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "python3"


def _parse_verdict(report: str) -> bool:
    """Parse a QA/audit report and return True if the overall verdict is PASS.

    Handles two verdict formats:
      1. Report contains PASS and no FAIL after the last "VERDICT" marker.
      2. A line matching "verdict.*pass" (case-insensitive).
    """
    if not report:
        return False
    upper = report.upper()
    if "PASS" in upper and "FAIL" not in upper.split("VERDICT")[-1]:
        return True
    if "verdict" in report.lower():
        for line in report.lower().split("\n"):
            if "verdict" in line and "pass" in line:
                return True
    return False


async def _run_technical_qa(state: KindleState, ui: UI, ws, project_dir: str) -> tuple[str, bool]:
    """Run technical QA sub-stage and return (report_text, passed)."""
    ui.info("Running Technical QA...")
    workspace_python = _find_workspace_python(ws)
    ui.stage_log("qa", f"Using Python: {workspace_python}")

    architecture = state.get("architecture", "")
    feature_spec = state.get("feature_spec", {})

    tech_prompt = (
        f"Run technical QA on the project in the working directory.\n\n"
        f"ARCHITECTURE:\n{architecture}\n\n"
        f"FEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}\n\n"
        f"Project Python: {workspace_python}\n\n"
        f"Install dependencies first, then run all checks. "
        f"Write qa_report.md to the working directory."
    )

    await run_agent(
        persona="Principal Quality Engineer",
        system_prompt=TECHNICAL_QA_SYSTEM_PROMPT,
        user_prompt=tech_prompt,
        cwd=str(ws),
        project_dir=project_dir,
        stage="qa_technical",
        ui=ui,
        model=state.get("model"),
        max_turns=state.get("max_agent_turns", 50),
    )

    qa_report_path = ws / "qa_report.md"
    qa_report = qa_report_path.read_text() if qa_report_path.exists() else ""
    save_artifact(project_dir, "qa_report.md", qa_report)

    qa_passed = _parse_verdict(qa_report)
    return qa_report, qa_passed


async def _run_product_audit(state: KindleState, ui: UI, ws, project_dir: str) -> tuple[str, bool]:
    """Run product audit sub-stage and return (report_text, passed)."""
    ui.info("Running Product Audit (anti-slop detection)...")
    feature_spec = state.get("feature_spec", {})

    audit_prompt = (
        f"Perform a product audit on the project in the working directory.\n\n"
        f"FEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}\n\n"
        f"Read every source file. Check for slop, placeholders, spec compliance. "
        f"Write product_audit.md to the working directory."
    )

    await run_agent(
        persona="VP of Product (Anti-Slop Auditor)",
        system_prompt=PRODUCT_AUDIT_SYSTEM_PROMPT,
        user_prompt=audit_prompt,
        cwd=str(ws),
        project_dir=project_dir,
        stage="qa_product_audit",
        ui=ui,
        model=state.get("model"),
        max_turns=state.get("max_agent_turns", 50),
    )

    audit_path = ws / "product_audit.md"
    product_audit = audit_path.read_text() if audit_path.exists() else ""
    save_artifact(project_dir, "product_audit.md", product_audit)

    cpo_passed = _parse_verdict(product_audit)
    return product_audit, cpo_passed


async def _run_self_heal(state: KindleState, ui: UI, ws, project_dir: str, fix_report: str) -> None:
    """Run a dev-fix agent to address issues found by QA."""
    ui.info("Running self-healing dev fix...")
    architecture = state.get("architecture", "")
    feature_spec = state.get("feature_spec", {})

    fix_prompt = (
        f"Fix the issues found in the QA report.\n\n"
        f"QA REPORT:\n{fix_report}\n\n"
        f"ARCHITECTURE:\n{architecture}\n\n"
        f"FEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}\n\n"
        f"Fix every issue. Ensure the project passes all checks."
    )

    await run_agent(
        persona="Principal Software Engineer (QA Fix)",
        system_prompt=DEV_FIX_SYSTEM_PROMPT,
        user_prompt=fix_prompt,
        cwd=str(ws),
        project_dir=project_dir,
        stage="qa_fix",
        ui=ui,
        model=state.get("model"),
        max_turns=state.get("max_agent_turns", 50),
    )


async def qa_node(state: KindleState, ui: UI) -> dict:
    """LangGraph node: run Technical QA + Product Audit."""
    ui.stage_start("qa")
    project_dir = state["project_dir"]
    ws = workspace_path(project_dir)
    qa_retries = state.get("qa_retries", 0)
    cpo_retries = state.get("cpo_retries", 0)

    # ── 5A: Technical QA ──────────────────────────────────────────
    qa_report, qa_passed = await _run_technical_qa(state, ui, ws, project_dir)
    if not qa_passed:
        ui.info(f"Technical QA FAILED (attempt {qa_retries + 1})")
    else:
        ui.info("Technical QA PASSED ✓")

    # ── 5B: Product Audit ─────────────────────────────────────────
    product_audit, cpo_passed = "", False
    if qa_passed:
        product_audit, cpo_passed = await _run_product_audit(state, ui, ws, project_dir)
        if not cpo_passed:
            ui.info(f"Product Audit FAILED (attempt {cpo_retries + 1})")
        else:
            ui.info("Product Audit PASSED ✓")

    # ── Self-heal if needed ───────────────────────────────────────
    if not qa_passed:
        qa_retries += 1
        await _run_self_heal(state, ui, ws, project_dir, qa_report)
    elif not cpo_passed:
        cpo_retries += 1
        await _run_self_heal(state, ui, ws, project_dir, product_audit)

    if qa_passed and cpo_passed:
        mark_stage_complete(project_dir, "qa")

    ui.stage_done("qa")

    return {
        "qa_report": qa_report,
        "product_audit": product_audit,
        "qa_passed": qa_passed,
        "cpo_passed": cpo_passed,
        "qa_retries": qa_retries,
        "cpo_retries": cpo_retries,
        "current_stage": "qa",
    }


def qa_router(state: KindleState) -> str:
    """Route after QA: loop back to dev for fixes or proceed to package."""
    qa_passed = state.get("qa_passed", False)
    cpo_passed = state.get("cpo_passed", False)
    qa_retries = state.get("qa_retries", 0)
    cpo_retries = state.get("cpo_retries", 0)
    max_qa = state.get("max_qa_retries", 10)
    max_cpo = state.get("max_cpo_retries", 10)

    if qa_passed and cpo_passed:
        return "package"

    # Check retry limits
    if not qa_passed and qa_retries >= max_qa:
        return "package"  # Give up on QA fixes
    if not cpo_passed and cpo_retries >= max_cpo:
        return "package"  # Give up on product audit fixes

    return "qa"  # Self-heal: re-run QA after fixes
