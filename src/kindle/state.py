"""Pipeline state — typed dict that flows through every LangGraph node."""

from __future__ import annotations

from typing import Annotated, TypedDict


def _replace(a: str, b: str) -> str:
    """Reducer that always takes the latest value."""
    return b


def _replace_bool(a: bool, b: bool) -> bool:
    return b


def _replace_int(a: int, b: int) -> int:
    return b


def _replace_list(a: list, b: list) -> list:
    return b


def _replace_dict(a: dict, b: dict) -> dict:
    return b


class KindleState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────
    idea: Annotated[str, _replace]
    project_id: Annotated[str, _replace]
    project_dir: Annotated[str, _replace]  # Working directory for the build

    # ── User preferences (from CLI flags) ───────────────────
    stack_preference: Annotated[str, _replace]  # e.g. "react", "nextjs", "python"
    auto_approve: Annotated[bool, _replace_bool]
    review_arch: Annotated[bool, _replace_bool]
    max_concurrent_agents: Annotated[int, _replace_int]
    max_qa_retries: Annotated[int, _replace_int]
    max_cpo_retries: Annotated[int, _replace_int]

    # ── Stage artifacts ─────────────────────────────────────
    feature_spec: Annotated[dict, _replace_dict]  # From Grill
    grill_transcript: Annotated[str, _replace]  # From Grill
    research_report: Annotated[str, _replace]  # From Research
    architecture: Annotated[str, _replace]  # From Architect
    dev_tasks: Annotated[list[dict], _replace_list]  # From Architect
    qa_report: Annotated[str, _replace]  # From QA
    product_audit: Annotated[str, _replace]  # From QA
    package_readme: Annotated[str, _replace]  # From Package

    # ── Quality tracking ────────────────────────────────────
    qa_passed: Annotated[bool, _replace_bool]
    cpo_passed: Annotated[bool, _replace_bool]
    qa_retries: Annotated[int, _replace_int]
    cpo_retries: Annotated[int, _replace_int]

    # ── Settings ────────────────────────────────────────────
    model: Annotated[str, _replace]
    max_agent_turns: Annotated[int, _replace_int]

    # ── Pipeline state ──────────────────────────────────────
    current_stage: Annotated[str, _replace]
