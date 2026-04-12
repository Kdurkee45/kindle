"""Tests for kindle.stages.qa — technical QA + product audit with self-healing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kindle.stages.qa import (
    DEV_FIX_SYSTEM_PROMPT,
    TECHNICAL_QA_SYSTEM_PROMPT,
    _find_workspace_python,
    _parse_verdict,
    qa_node,
    qa_router,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_FEATURE_SPEC = {
    "app_name": "TaskFlow",
    "idea": "a task management app",
    "core_features": ["task CRUD", "auth"],
    "tech_constraints": ["React frontend"],
}

SAMPLE_ARCHITECTURE = """\
# Architecture

## Tech Stack
- React 18.x frontend
- Express 4.x backend
- PostgreSQL database
"""

QA_REPORT_PASS = """\
# QA Report

## Test Suite: PASS
All 42 tests passing.

## Linter: PASS
No warnings.

## Overall Verdict: PASS
"""

QA_REPORT_FAIL = """\
# QA Report

## Test Suite: FAIL
3 tests failing in test_auth.py.

## Linter: PASS
No warnings.

## Overall Verdict: FAIL
"""

PRODUCT_AUDIT_PASS = """\
# Product Audit

## Spec Compliance: PASS
## Placeholder Text: PASS
## Overall Verdict: PASS
"""

PRODUCT_AUDIT_FAIL = """\
# Product Audit

## Spec Compliance: PASS
## Placeholder Text: FAIL
Found "Lorem ipsum" in src/components/Hero.tsx:12

## Overall Verdict: FAIL
"""


def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal KindleState dict pointing at *tmp_path* as project_dir."""
    project_dir = tmp_path / "project"
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)
    # metadata.json needed by mark_stage_complete
    meta = {"project_id": "kindle_test1234", "stages_completed": []}
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    state: dict = {
        "project_dir": str(project_dir),
        "idea": "a task management app",
        "feature_spec": SAMPLE_FEATURE_SPEC,
        "architecture": SAMPLE_ARCHITECTURE,
    }
    state.update(overrides)
    return state


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods qa_node actually calls."""
    ui = MagicMock()
    ui.auto_approve = False
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.stage_log = MagicMock()
    return ui


# ---------------------------------------------------------------------------
# qa_router — pure function, no async, no I/O
# ---------------------------------------------------------------------------


class TestQaRouter:
    """qa_router should map state to 'package' or 'qa'."""

    def test_both_passed_routes_to_package(self) -> None:
        """When both QA and CPO pass, proceed to packaging."""
        state = {"qa_passed": True, "cpo_passed": True, "qa_retries": 0, "cpo_retries": 0}
        assert qa_router(state) == "package"

    def test_qa_retries_exceeded_routes_to_package(self) -> None:
        """When QA retries hit the limit, give up and package anyway."""
        state = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 10,
            "cpo_retries": 0,
            "max_qa_retries": 10,
        }
        assert qa_router(state) == "package"

    def test_cpo_retries_exceeded_routes_to_package(self) -> None:
        """When CPO retries hit the limit, give up and package anyway."""
        state = {
            "qa_passed": True,
            "cpo_passed": False,
            "qa_retries": 0,
            "cpo_retries": 5,
            "max_cpo_retries": 5,
        }
        assert qa_router(state) == "package"

    def test_neither_passed_routes_to_qa(self) -> None:
        """When checks fail but retries remain, loop back to QA."""
        state = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 2,
            "cpo_retries": 0,
            "max_qa_retries": 10,
            "max_cpo_retries": 10,
        }
        assert qa_router(state) == "qa"

    def test_qa_passed_cpo_failed_with_retries_remaining(self) -> None:
        """QA passed but CPO failed with retries left → loop back to qa."""
        state = {
            "qa_passed": True,
            "cpo_passed": False,
            "qa_retries": 0,
            "cpo_retries": 1,
            "max_cpo_retries": 10,
        }
        assert qa_router(state) == "qa"

    def test_defaults_used_when_keys_missing(self) -> None:
        """Defaults are False/0/10 when state keys are missing."""
        # Empty state: qa_passed=False, cpo_passed=False, retries=0, max=10
        # → retries (0) < max (10), so loop back to "qa"
        assert qa_router({}) == "qa"

    def test_qa_retries_at_exactly_max(self) -> None:
        """qa_retries == max_qa_retries triggers 'package' (>=, not >)."""
        state = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 3,
            "max_qa_retries": 3,
        }
        assert qa_router(state) == "package"

    def test_qa_retries_one_below_max_routes_to_qa(self) -> None:
        """qa_retries one below max still loops."""
        state = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 2,
            "max_qa_retries": 3,
        }
        assert qa_router(state) == "qa"

    def test_custom_max_retries_respected(self) -> None:
        """Custom max_qa_retries and max_cpo_retries are respected."""
        state = {
            "qa_passed": False,
            "cpo_passed": False,
            "qa_retries": 1,
            "max_qa_retries": 1,
        }
        assert qa_router(state) == "package"


# ---------------------------------------------------------------------------
# Verdict detection logic (tested indirectly through qa_node)
# ---------------------------------------------------------------------------


class TestVerdictDetection:
    """Test the verdict parsing logic from QA report / product audit strings.

    The verdict parsing is inline in qa_node. We test it by running qa_node
    with mock agents that write various report formats to the workspace.
    """

    @pytest.mark.asyncio
    async def test_verdict_pass_uppercase(self, tmp_path: Path) -> None:
        """'VERDICT: PASS' in report → qa_passed=True."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text("## Results\nAll checks pass.\n\n## Overall Verdict: PASS\n")
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text("## Results\nNo slop found.\n\n## Overall Verdict: PASS\n")
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is True

    @pytest.mark.asyncio
    async def test_verdict_pass_lowercase(self, tmp_path: Path) -> None:
        """'verdict: pass' in report → qa_passed=True (case-insensitive path)."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text("all checks done.\noverall verdict: pass\n")
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text("overall verdict: pass\n")
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is True

    @pytest.mark.asyncio
    async def test_verdict_pass_mixed_case(self, tmp_path: Path) -> None:
        """'Verdict: Pass' in mixed case → qa_passed=True."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text("## Summary\nOverall Verdict: Pass\n")
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text("Overall Verdict: Pass\n")
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is True

    @pytest.mark.asyncio
    async def test_verdict_fail_detected(self, tmp_path: Path) -> None:
        """'Overall Verdict: FAIL' → qa_passed=False."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is False

    @pytest.mark.asyncio
    async def test_empty_report_means_fail(self, tmp_path: Path) -> None:
        """If qa_report.md is missing/empty → qa_passed=False."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            # Do NOT write qa_report.md
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is False
        assert result["qa_report"] == ""

    @pytest.mark.asyncio
    async def test_pass_in_body_but_fail_in_verdict(self, tmp_path: Path) -> None:
        """Individual checks PASS, but overall verdict is FAIL → qa_passed=False.

        The verdict logic splits on 'VERDICT' and checks the text *after* it.
        """
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        report = "## Linter: PASS\n## Formatter: PASS\n## Tests: FAIL\n\n## Overall Verdict: FAIL\n"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(report)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is False

    @pytest.mark.asyncio
    async def test_verdict_with_bold_markdown(self, tmp_path: Path) -> None:
        """'**Verdict**: PASS' (bold markdown) still detected."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text("## Summary\n**Overall Verdict**: PASS\n")
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text("**Overall Verdict**: PASS\n")
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is True

    @pytest.mark.asyncio
    async def test_no_verdict_keyword_all_pass_still_detected(self, tmp_path: Path) -> None:
        """Report with PASS but no VERDICT keyword — primary check sees PASS."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                # Has PASS but no VERDICT keyword, and no FAIL → primary check succeeds
                (ws / "qa_report.md").write_text("## All checks\nResult: PASS\nEverything looks good.\n")
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text("Result: PASS\n")
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        # Primary check: "PASS" in upper AND no "FAIL" after "VERDICT" split
        # (when there is no VERDICT, split returns the whole string, but
        # there is no FAIL in it → qa_passed = True)
        assert result["qa_passed"] is True


# ---------------------------------------------------------------------------
# _parse_verdict — direct unit tests for the verdict parsing helper
# ---------------------------------------------------------------------------


class TestParseVerdict:
    """Direct unit tests for _parse_verdict covering all branches."""

    @pytest.mark.parametrize(
        ("report", "expected"),
        [
            pytest.param("", False, id="empty_string"),
            pytest.param("All checks: PASS\n\nOverall Verdict: PASS\n", True, id="uppercase_pass"),
            pytest.param("## Tests: FAIL\n\n## Overall Verdict: FAIL\n", False, id="uppercase_fail"),
            pytest.param(
                "Overall Verdict: Pass\nSome FAIL note at the end",
                True,
                id="line_level_verdict_pass_overrides_trailing_fail",
            ),
            pytest.param(
                "Section PASS ok\nVerdict: Pass\nFAIL residue",
                True,
                id="verdict_pass_line_with_fail_elsewhere",
            ),
            pytest.param(
                "No verdict keyword, just FAIL",
                False,
                id="fail_without_verdict_keyword",
            ),
            pytest.param(
                "Result: PASS\nNo issues found.",
                True,
                id="pass_without_verdict_keyword",
            ),
        ],
    )
    def test_parse_verdict(self, report: str, expected: bool) -> None:
        assert _parse_verdict(report) is expected


# ---------------------------------------------------------------------------
# _find_workspace_python — filesystem probing
# ---------------------------------------------------------------------------


class TestFindWorkspacePython:
    """Test Python interpreter discovery with mock filesystem."""

    def test_finds_venv_python(self, tmp_path: Path) -> None:
        """Prefer .venv/bin/python when it exists."""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").touch()

        result = _find_workspace_python(tmp_path)
        assert result == str(venv_bin / "python")

    def test_finds_venv_python3(self, tmp_path: Path) -> None:
        """Falls back to .venv/bin/python3 when python does not exist."""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python3").touch()

        result = _find_workspace_python(tmp_path)
        assert result == str(venv_bin / "python3")

    def test_finds_plain_venv_python(self, tmp_path: Path) -> None:
        """Falls back to venv/bin/python (no dot prefix)."""
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").touch()

        result = _find_workspace_python(tmp_path)
        assert result == str(venv_bin / "python")

    def test_finds_plain_venv_python3(self, tmp_path: Path) -> None:
        """Falls back to venv/bin/python3 (no dot prefix)."""
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python3").touch()

        result = _find_workspace_python(tmp_path)
        assert result == str(venv_bin / "python3")

    def test_fallback_to_system_python3(self, tmp_path: Path) -> None:
        """When no venv found, falls back to 'python3'."""
        result = _find_workspace_python(tmp_path)
        assert result == "python3"

    def test_priority_order_dot_venv_first(self, tmp_path: Path) -> None:
        """When both .venv and venv exist, .venv/bin/python wins."""
        for prefix in (".venv", "venv"):
            venv_bin = tmp_path / prefix / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").touch()
            (venv_bin / "python3").touch()

        result = _find_workspace_python(tmp_path)
        assert result == str(tmp_path / ".venv" / "bin" / "python")

    def test_dot_venv_python3_before_venv_python(self, tmp_path: Path) -> None:
        """.venv/bin/python3 is preferred over venv/bin/python."""
        dot_venv_bin = tmp_path / ".venv" / "bin"
        dot_venv_bin.mkdir(parents=True)
        (dot_venv_bin / "python3").touch()

        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").touch()

        result = _find_workspace_python(tmp_path)
        assert result == str(dot_venv_bin / "python3")


# ---------------------------------------------------------------------------
# Technical QA phase (5A)
# ---------------------------------------------------------------------------


class TestTechnicalQaPhase:
    """Tests for the Technical QA sub-stage of qa_node."""

    @pytest.mark.asyncio
    async def test_run_agent_called_for_qa_technical(self, tmp_path: Path) -> None:
        """run_agent is called with stage='qa_technical'."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        calls = []

        async def fake_run_agent(**kwargs):
            calls.append(kwargs)
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        tech_calls = [c for c in calls if c.get("stage") == "qa_technical"]
        assert len(tech_calls) == 1
        assert TECHNICAL_QA_SYSTEM_PROMPT in tech_calls[0]["system_prompt"]

    @pytest.mark.asyncio
    async def test_qa_report_saved_as_artifact(self, tmp_path: Path) -> None:
        """qa_report.md content is persisted to artifacts/."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_PASS)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_report"] == QA_REPORT_PASS
        artifact_path = Path(state["project_dir"]) / "artifacts" / "qa_report.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == QA_REPORT_PASS

    @pytest.mark.asyncio
    async def test_qa_report_contains_architecture_in_prompt(self, tmp_path: Path) -> None:
        """The tech QA prompt includes the architecture for context."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        captured_prompt = None

        async def fake_run_agent(**kwargs):
            nonlocal captured_prompt
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                captured_prompt = kwargs.get("user_prompt", "")
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        assert SAMPLE_ARCHITECTURE in captured_prompt

    @pytest.mark.asyncio
    async def test_workspace_python_logged(self, tmp_path: Path) -> None:
        """The discovered Python interpreter is logged via ui.stage_log."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        ui.stage_log.assert_called()
        log_args = ui.stage_log.call_args_list
        python_logs = [c for c in log_args if "Python" in str(c)]
        assert len(python_logs) >= 1


# ---------------------------------------------------------------------------
# Product Audit phase (5B) — only runs when qa_passed=True
# ---------------------------------------------------------------------------


class TestProductAuditPhase:
    """Tests for the Product Audit sub-stage of qa_node."""

    @pytest.mark.asyncio
    async def test_product_audit_runs_when_qa_passes(self, tmp_path: Path) -> None:
        """Product audit agent is called only when QA passes."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        stages_called = []

        async def fake_run_agent(**kwargs):
            stages_called.append(kwargs.get("stage"))
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_PASS)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert "qa_product_audit" in stages_called
        assert result["cpo_passed"] is True

    @pytest.mark.asyncio
    async def test_product_audit_skipped_when_qa_fails(self, tmp_path: Path) -> None:
        """When Technical QA fails, product audit is skipped entirely."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        stages_called = []

        async def fake_run_agent(**kwargs):
            stages_called.append(kwargs.get("stage"))
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert "qa_product_audit" not in stages_called
        assert result["cpo_passed"] is False
        assert result["product_audit"] == ""

    @pytest.mark.asyncio
    async def test_product_audit_saved_as_artifact(self, tmp_path: Path) -> None:
        """product_audit.md is persisted to artifacts/."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_PASS)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["product_audit"] == PRODUCT_AUDIT_PASS
        artifact_path = Path(state["project_dir"]) / "artifacts" / "product_audit.md"
        assert artifact_path.exists()

    @pytest.mark.asyncio
    async def test_product_audit_fail_detected(self, tmp_path: Path) -> None:
        """Product audit FAIL verdict → cpo_passed=False."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is True
        assert result["cpo_passed"] is False

    @pytest.mark.asyncio
    async def test_product_audit_empty_report_means_fail(self, tmp_path: Path) -> None:
        """If product_audit.md is not written → cpo_passed=False."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            # Don't write product_audit.md
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["cpo_passed"] is False
        assert result["product_audit"] == ""


# ---------------------------------------------------------------------------
# Self-healing — dev fix agent called on failures
# ---------------------------------------------------------------------------


class TestSelfHealing:
    """When QA or CPO fails, a dev fix agent should be invoked."""

    @pytest.mark.asyncio
    async def test_fix_agent_called_on_qa_failure(self, tmp_path: Path) -> None:
        """When Technical QA fails, a fix agent is dispatched."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        stages_called = []

        async def fake_run_agent(**kwargs):
            stages_called.append(kwargs.get("stage"))
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert "qa_fix" in stages_called
        assert result["qa_retries"] == 1

    @pytest.mark.asyncio
    async def test_fix_agent_called_on_cpo_failure(self, tmp_path: Path) -> None:
        """When CPO audit fails (but QA passes), fix agent is dispatched."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        stages_called = []

        async def fake_run_agent(**kwargs):
            stages_called.append(kwargs.get("stage"))
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert "qa_fix" in stages_called
        assert result["cpo_retries"] == 1

    @pytest.mark.asyncio
    async def test_fix_agent_receives_qa_report_in_prompt(self, tmp_path: Path) -> None:
        """The fix agent gets the QA report in its prompt for context."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        fix_prompt = None

        async def fake_run_agent(**kwargs):
            nonlocal fix_prompt
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            elif kwargs.get("stage") == "qa_fix":
                fix_prompt = kwargs.get("user_prompt", "")
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        assert fix_prompt is not None
        assert "FAIL" in fix_prompt
        assert "test_auth.py" in fix_prompt

    @pytest.mark.asyncio
    async def test_fix_agent_uses_dev_fix_system_prompt(self, tmp_path: Path) -> None:
        """The fix agent uses DEV_FIX_SYSTEM_PROMPT."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        fix_system_prompt = None

        async def fake_run_agent(**kwargs):
            nonlocal fix_system_prompt
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            elif kwargs.get("stage") == "qa_fix":
                fix_system_prompt = kwargs.get("system_prompt", "")
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        assert fix_system_prompt == DEV_FIX_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_no_fix_agent_when_both_pass(self, tmp_path: Path) -> None:
        """When both QA and CPO pass, no fix agent is called."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        stages_called = []

        async def fake_run_agent(**kwargs):
            stages_called.append(kwargs.get("stage"))
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_PASS)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        assert "qa_fix" not in stages_called

    @pytest.mark.asyncio
    async def test_qa_retries_incremented_on_qa_failure(self, tmp_path: Path) -> None:
        """qa_retries is incremented by 1 on technical QA failure."""
        state = _make_state(tmp_path, qa_retries=3)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_retries"] == 4

    @pytest.mark.asyncio
    async def test_cpo_retries_incremented_on_cpo_failure(self, tmp_path: Path) -> None:
        """cpo_retries is incremented by 1 on product audit failure."""
        state = _make_state(tmp_path, cpo_retries=2)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["cpo_retries"] == 3


# ---------------------------------------------------------------------------
# State return — verify all returned keys
# ---------------------------------------------------------------------------


class TestStateReturn:
    """Verify qa_node returns all expected state keys."""

    @pytest.mark.asyncio
    async def test_all_state_keys_present_on_full_pass(self, tmp_path: Path) -> None:
        """All 7 state keys are present when both QA and CPO pass."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_PASS)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        expected_keys = {
            "qa_report",
            "product_audit",
            "qa_passed",
            "cpo_passed",
            "qa_retries",
            "cpo_retries",
            "current_stage",
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_current_stage_is_qa(self, tmp_path: Path) -> None:
        """current_stage is always 'qa' after qa_node."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["current_stage"] == "qa"

    @pytest.mark.asyncio
    async def test_state_keys_on_failure(self, tmp_path: Path) -> None:
        """All state keys are present even on failure path."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        assert result["qa_passed"] is False
        assert result["cpo_passed"] is False
        assert result["qa_retries"] == 1
        assert result["cpo_retries"] == 0
        assert result["qa_report"] == QA_REPORT_FAIL
        assert result["product_audit"] == ""
        assert result["current_stage"] == "qa"

    @pytest.mark.asyncio
    async def test_retries_preserved_from_state(self, tmp_path: Path) -> None:
        """Retry counters carry forward from input state."""
        state = _make_state(tmp_path, qa_retries=5, cpo_retries=3)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_PASS)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        # Retries should NOT increment when both pass
        assert result["qa_retries"] == 5
        assert result["cpo_retries"] == 3


# ---------------------------------------------------------------------------
# Stage lifecycle — UI calls and mark_stage_complete
# ---------------------------------------------------------------------------


class TestStageLifecycle:
    """Verify UI lifecycle calls and stage completion marking."""

    @pytest.mark.asyncio
    async def test_stage_start_and_done_called(self, tmp_path: Path) -> None:
        """ui.stage_start('qa') and ui.stage_done('qa') are always called."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        ui.stage_start.assert_called_once_with("qa")
        ui.stage_done.assert_called_once_with("qa")

    @pytest.mark.asyncio
    async def test_stage_marked_complete_on_full_pass(self, tmp_path: Path) -> None:
        """mark_stage_complete is called when both QA and CPO pass."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_PASS)
            elif kwargs.get("stage") == "qa_product_audit":
                (ws / "product_audit.md").write_text(PRODUCT_AUDIT_PASS)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            result = await qa_node(state, ui)

        # Check that metadata reflects stage completion
        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "qa" in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_stage_not_marked_complete_on_failure(self, tmp_path: Path) -> None:
        """mark_stage_complete is NOT called when QA fails."""
        state = _make_state(tmp_path)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        async def fake_run_agent(**kwargs):
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        meta_path = Path(state["project_dir"]) / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert "qa" not in meta["stages_completed"]

    @pytest.mark.asyncio
    async def test_model_passed_through_to_agent(self, tmp_path: Path) -> None:
        """The model preference is forwarded to run_agent."""
        state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        captured_models = []

        async def fake_run_agent(**kwargs):
            captured_models.append(kwargs.get("model"))
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        assert all(m == "claude-sonnet-4-20250514" for m in captured_models)

    @pytest.mark.asyncio
    async def test_max_agent_turns_forwarded(self, tmp_path: Path) -> None:
        """max_agent_turns from state is forwarded to run_agent."""
        state = _make_state(tmp_path, max_agent_turns=25)
        ui = _make_ui()
        ws = Path(state["project_dir"]) / "workspace"

        captured_turns = []

        async def fake_run_agent(**kwargs):
            captured_turns.append(kwargs.get("max_turns"))
            ws.mkdir(parents=True, exist_ok=True)
            if kwargs.get("stage") == "qa_technical":
                (ws / "qa_report.md").write_text(QA_REPORT_FAIL)
            return MagicMock()

        with patch("kindle.stages.qa.run_agent", AsyncMock(side_effect=fake_run_agent)):
            await qa_node(state, ui)

        assert all(t == 25 for t in captured_turns)
