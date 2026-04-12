"""Tests for kindle.state — Pipeline state TypedDict and reducer functions."""

from __future__ import annotations

import typing
from typing import get_type_hints

import pytest

from kindle.state import (
    KindleState,
    _replace,
    _replace_bool,
    _replace_dict,
    _replace_int,
    _replace_list,
)

# ---------------------------------------------------------------------------
# _replace (str reducer)
# ---------------------------------------------------------------------------


class TestReplace:
    """The _replace reducer always returns its second argument (overwrite)."""

    def test_returns_second_argument(self) -> None:
        assert _replace("old", "new") == "new"

    def test_ignores_first_argument(self) -> None:
        assert _replace("anything", "winner") == "winner"

    def test_identical_values(self) -> None:
        assert _replace("same", "same") == "same"

    def test_empty_strings(self) -> None:
        assert _replace("", "") == ""
        assert _replace("non-empty", "") == ""
        assert _replace("", "non-empty") == "non-empty"

    def test_multiline_strings(self) -> None:
        old = "line1\nline2"
        new = "line3\nline4\nline5"
        assert _replace(old, new) == new

    def test_unicode_strings(self) -> None:
        assert _replace("héllo", "wörld") == "wörld"

    def test_whitespace_only(self) -> None:
        assert _replace("content", "   ") == "   "


# ---------------------------------------------------------------------------
# _replace_bool (bool reducer)
# ---------------------------------------------------------------------------


class TestReplaceBool:
    """The _replace_bool reducer always returns its second argument."""

    def test_true_overwrites_false(self) -> None:
        assert _replace_bool(False, True) is True

    def test_false_overwrites_true(self) -> None:
        assert _replace_bool(True, False) is False

    def test_true_overwrites_true(self) -> None:
        assert _replace_bool(True, True) is True

    def test_false_overwrites_false(self) -> None:
        assert _replace_bool(False, False) is False


# ---------------------------------------------------------------------------
# _replace_int (int reducer)
# ---------------------------------------------------------------------------


class TestReplaceInt:
    """The _replace_int reducer always returns its second argument."""

    def test_returns_second_argument(self) -> None:
        assert _replace_int(1, 2) == 2

    def test_zero_overwrites(self) -> None:
        assert _replace_int(42, 0) == 0

    def test_negative_values(self) -> None:
        assert _replace_int(-1, -99) == -99

    def test_large_values(self) -> None:
        assert _replace_int(0, 10**9) == 10**9

    def test_same_values(self) -> None:
        assert _replace_int(7, 7) == 7


# ---------------------------------------------------------------------------
# _replace_list (list reducer)
# ---------------------------------------------------------------------------


class TestReplaceList:
    """The _replace_list reducer always returns its second argument."""

    def test_returns_second_list(self) -> None:
        assert _replace_list([1, 2], [3, 4]) == [3, 4]

    def test_empty_list_overwrites(self) -> None:
        assert _replace_list([1, 2, 3], []) == []

    def test_non_empty_overwrites_empty(self) -> None:
        assert _replace_list([], [1]) == [1]

    def test_both_empty(self) -> None:
        assert _replace_list([], []) == []

    def test_returns_exact_reference(self) -> None:
        """The reducer should return the exact same list object, not a copy."""
        new = [1, 2, 3]
        result = _replace_list([], new)
        assert result is new

    def test_nested_lists(self) -> None:
        old = [[1], [2]]
        new = [[3, 4], [5, 6]]
        assert _replace_list(old, new) == [[3, 4], [5, 6]]

    def test_mixed_types_in_list(self) -> None:
        assert _replace_list(["a"], [1, "b", None]) == [1, "b", None]


# ---------------------------------------------------------------------------
# _replace_dict (dict reducer)
# ---------------------------------------------------------------------------


class TestReplaceDict:
    """The _replace_dict reducer always returns its second argument."""

    def test_returns_second_dict(self) -> None:
        assert _replace_dict({"a": 1}, {"b": 2}) == {"b": 2}

    def test_empty_dict_overwrites(self) -> None:
        assert _replace_dict({"key": "val"}, {}) == {}

    def test_non_empty_overwrites_empty(self) -> None:
        assert _replace_dict({}, {"key": "val"}) == {"key": "val"}

    def test_both_empty(self) -> None:
        assert _replace_dict({}, {}) == {}

    def test_returns_exact_reference(self) -> None:
        """The reducer should return the exact same dict object, not a copy."""
        new = {"x": 1}
        result = _replace_dict({}, new)
        assert result is new

    def test_nested_dicts(self) -> None:
        old = {"outer": {"inner": 1}}
        new = {"outer": {"inner": 2, "extra": 3}}
        assert _replace_dict(old, new) == {"outer": {"inner": 2, "extra": 3}}

    def test_overlapping_keys_fully_replaced(self) -> None:
        """Overwrite semantics means the old dict is discarded entirely, not merged."""
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 99}
        result = _replace_dict(old, new)
        assert result == {"a": 99}
        assert "b" not in result
        assert "c" not in result


# ---------------------------------------------------------------------------
# KindleState TypedDict — structural checks
# ---------------------------------------------------------------------------


class TestKindleStateStructure:
    """Verify KindleState TypedDict shape and metadata."""

    def test_is_typed_dict(self) -> None:
        """KindleState must be a TypedDict subclass."""
        assert issubclass(KindleState, dict)

    def test_total_false_allows_partial_instantiation(self) -> None:
        """total=False means all fields are optional — an empty dict is valid."""
        state: KindleState = KindleState()  # type: ignore[typeddict-item]
        assert isinstance(state, dict)
        assert len(state) == 0

    def test_single_field_instantiation(self) -> None:
        state = KindleState(idea="build something")
        assert state["idea"] == "build something"
        assert len(state) == 1

    def test_multiple_fields_instantiation(self) -> None:
        state = KindleState(
            idea="test app",
            project_id="proj-001",
            auto_approve=True,
            max_concurrent_agents=8,
        )
        assert state["idea"] == "test app"
        assert state["project_id"] == "proj-001"
        assert state["auto_approve"] is True
        assert state["max_concurrent_agents"] == 8

    def test_expected_fields_exist(self) -> None:
        """All documented state fields must appear in the type hints."""
        hints = get_type_hints(KindleState, include_extras=True)
        expected_fields = {
            "idea",
            "project_id",
            "project_dir",
            "stack_preference",
            "auto_approve",
            "review_arch",
            "max_concurrent_agents",
            "max_qa_retries",
            "max_cpo_retries",
            "feature_spec",
            "grill_transcript",
            "research_report",
            "architecture",
            "dev_tasks",
            "qa_report",
            "product_audit",
            "package_readme",
            "qa_passed",
            "cpo_passed",
            "qa_retries",
            "cpo_retries",
            "model",
            "max_agent_turns",
            "current_stage",
        }
        assert set(hints.keys()) == expected_fields


# ---------------------------------------------------------------------------
# Annotated reducer metadata — verify the correct reducer is wired up
# ---------------------------------------------------------------------------


class TestReducerAnnotations:
    """Each KindleState field should carry the correct reducer in its Annotated metadata."""

    @pytest.fixture
    def hints(self) -> dict[str, type]:
        return get_type_hints(KindleState, include_extras=True)

    @staticmethod
    def _get_reducer(annotated_type: type) -> object:
        """Extract the reducer function from an Annotated type."""
        metadata = typing.get_args(annotated_type)
        # Annotated[T, reducer] → args == (T, reducer)
        assert len(metadata) >= 2, f"Expected Annotated type, got {annotated_type}"
        return metadata[1]

    @pytest.mark.parametrize(
        "field",
        [
            "idea",
            "project_id",
            "project_dir",
            "stack_preference",
            "grill_transcript",
            "research_report",
            "architecture",
            "qa_report",
            "product_audit",
            "package_readme",
            "model",
            "current_stage",
        ],
    )
    def test_str_fields_use_replace(self, hints: dict[str, type], field: str) -> None:
        assert self._get_reducer(hints[field]) is _replace

    @pytest.mark.parametrize(
        "field",
        ["auto_approve", "review_arch", "qa_passed", "cpo_passed"],
    )
    def test_bool_fields_use_replace_bool(self, hints: dict[str, type], field: str) -> None:
        assert self._get_reducer(hints[field]) is _replace_bool

    @pytest.mark.parametrize(
        "field",
        [
            "max_concurrent_agents",
            "max_qa_retries",
            "max_cpo_retries",
            "qa_retries",
            "cpo_retries",
            "max_agent_turns",
        ],
    )
    def test_int_fields_use_replace_int(self, hints: dict[str, type], field: str) -> None:
        assert self._get_reducer(hints[field]) is _replace_int

    def test_dev_tasks_uses_replace_list(self, hints: dict[str, type]) -> None:
        assert self._get_reducer(hints["dev_tasks"]) is _replace_list

    def test_feature_spec_uses_replace_dict(self, hints: dict[str, type]) -> None:
        assert self._get_reducer(hints["feature_spec"]) is _replace_dict
