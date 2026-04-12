"""Tests for kindle.state — reducer functions and KindleState TypedDict."""

from __future__ import annotations

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
# _replace  (str reducer)
# ---------------------------------------------------------------------------


class TestReplace:
    """Tests for the _replace(a, b) → b string reducer."""

    def test_returns_second_argument(self):
        assert _replace("old", "new") == "new"

    def test_ignores_first_argument(self):
        assert _replace("anything", "latest") == "latest"

    def test_identical_values(self):
        assert _replace("same", "same") == "same"

    def test_empty_strings(self):
        assert _replace("", "") == ""

    def test_first_empty_second_nonempty(self):
        assert _replace("", "value") == "value"

    def test_first_nonempty_second_empty(self):
        assert _replace("value", "") == ""

    def test_multiline_strings(self):
        multi = "line1\nline2\nline3"
        assert _replace("old", multi) == multi

    def test_unicode_strings(self):
        assert _replace("hello", "こんにちは") == "こんにちは"


# ---------------------------------------------------------------------------
# _replace_bool
# ---------------------------------------------------------------------------


class TestReplaceBool:
    """Tests for the _replace_bool(a, b) → b boolean reducer."""

    def test_returns_true_when_b_is_true(self):
        assert _replace_bool(False, True) is True

    def test_returns_false_when_b_is_false(self):
        assert _replace_bool(True, False) is False

    def test_both_true(self):
        assert _replace_bool(True, True) is True

    def test_both_false(self):
        assert _replace_bool(False, False) is False

    def test_ignores_first_argument(self):
        """Regardless of `a`, result always equals `b`."""
        for a in (True, False):
            for b in (True, False):
                assert _replace_bool(a, b) is b


# ---------------------------------------------------------------------------
# _replace_int
# ---------------------------------------------------------------------------


class TestReplaceInt:
    """Tests for the _replace_int(a, b) → b integer reducer."""

    def test_returns_second_argument(self):
        assert _replace_int(1, 2) == 2

    def test_ignores_first_argument(self):
        assert _replace_int(999, 0) == 0

    def test_negative_values(self):
        assert _replace_int(-1, -42) == -42

    def test_zero_values(self):
        assert _replace_int(0, 0) == 0

    def test_large_values(self):
        big = 10**18
        assert _replace_int(0, big) == big

    def test_identical_values(self):
        assert _replace_int(7, 7) == 7


# ---------------------------------------------------------------------------
# _replace_list
# ---------------------------------------------------------------------------


class TestReplaceList:
    """Tests for the _replace_list(a, b) → b list reducer."""

    def test_returns_second_list(self):
        assert _replace_list([1, 2], [3, 4]) == [3, 4]

    def test_ignores_first_list(self):
        assert _replace_list(["old"], ["new"]) == ["new"]

    def test_empty_lists(self):
        assert _replace_list([], []) == []

    def test_first_empty_second_nonempty(self):
        assert _replace_list([], [1]) == [1]

    def test_first_nonempty_second_empty(self):
        assert _replace_list([1, 2, 3], []) == []

    def test_nested_lists(self):
        nested = [[1, 2], [3, 4]]
        assert _replace_list([], nested) == nested

    def test_returns_exact_object_identity(self):
        """The reducer should return the `b` object itself, not a copy."""
        b = [1, 2, 3]
        result = _replace_list([], b)
        assert result is b

    def test_mixed_type_list(self):
        mixed = [1, "two", 3.0, None]
        assert _replace_list([], mixed) == mixed


# ---------------------------------------------------------------------------
# _replace_dict
# ---------------------------------------------------------------------------


class TestReplaceDict:
    """Tests for the _replace_dict(a, b) → b dict reducer."""

    def test_returns_second_dict(self):
        assert _replace_dict({"a": 1}, {"b": 2}) == {"b": 2}

    def test_ignores_first_dict(self):
        assert _replace_dict({"old": True}, {"new": False}) == {"new": False}

    def test_empty_dicts(self):
        assert _replace_dict({}, {}) == {}

    def test_first_empty_second_nonempty(self):
        assert _replace_dict({}, {"key": "val"}) == {"key": "val"}

    def test_first_nonempty_second_empty(self):
        assert _replace_dict({"key": "val"}, {}) == {}

    def test_nested_dicts(self):
        nested = {"outer": {"inner": 42}}
        assert _replace_dict({}, nested) == nested

    def test_returns_exact_object_identity(self):
        """The reducer should return the `b` object itself, not a copy."""
        b = {"x": 1}
        result = _replace_dict({}, b)
        assert result is b

    def test_overlapping_keys_returns_b(self):
        """When both dicts share keys, the result is purely `b`."""
        a = {"shared": "old", "only_a": 1}
        b = {"shared": "new", "only_b": 2}
        result = _replace_dict(a, b)
        assert result == b
        assert "only_a" not in result


# ---------------------------------------------------------------------------
# KindleState TypedDict
# ---------------------------------------------------------------------------


class TestKindleState:
    """Tests for the KindleState TypedDict (total=False)."""

    def test_empty_instantiation(self):
        """total=False means all fields are optional — empty dict is valid."""
        state: KindleState = {}  # type: ignore[typeddict-item]
        assert isinstance(state, dict)
        assert len(state) == 0

    def test_single_field(self):
        """Can create a state with only the 'idea' field."""
        state: KindleState = {"idea": "build a CLI tool"}
        assert state["idea"] == "build a CLI tool"

    def test_multiple_fields(self):
        """Can create a state with several fields of different types."""
        state: KindleState = {
            "idea": "weather app",
            "project_id": "proj-123",
            "auto_approve": True,
            "max_concurrent_agents": 4,
            "dev_tasks": [{"name": "setup"}],
            "feature_spec": {"title": "Weather"},
        }
        assert state["idea"] == "weather app"
        assert state["project_id"] == "proj-123"
        assert state["auto_approve"] is True
        assert state["max_concurrent_agents"] == 4
        assert state["dev_tasks"] == [{"name": "setup"}]
        assert state["feature_spec"] == {"title": "Weather"}

    def test_quality_tracking_fields(self):
        """Quality tracking fields accept expected types."""
        state: KindleState = {
            "qa_passed": False,
            "cpo_passed": True,
            "qa_retries": 3,
            "cpo_retries": 0,
        }
        assert state["qa_passed"] is False
        assert state["cpo_passed"] is True
        assert state["qa_retries"] == 3
        assert state["cpo_retries"] == 0

    def test_settings_fields(self):
        """Settings fields (model, max_agent_turns) accept expected types."""
        state: KindleState = {
            "model": "claude-opus-4-20250514",
            "max_agent_turns": 50,
        }
        assert state["model"] == "claude-opus-4-20250514"
        assert state["max_agent_turns"] == 50

    def test_pipeline_state_field(self):
        state: KindleState = {"current_stage": "architect"}
        assert state["current_stage"] == "architect"

    def test_state_is_mutable_dict(self):
        """KindleState instances are plain dicts — mutable by design."""
        state: KindleState = {"idea": "first"}
        state["idea"] = "second"
        assert state["idea"] == "second"

    def test_all_known_keys_are_present_in_annotations(self):
        """Verify the expected set of field names on KindleState."""
        expected_keys = {
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
        actual_keys = set(KindleState.__annotations__.keys())
        assert actual_keys == expected_keys


# ---------------------------------------------------------------------------
# Reducer annotations on KindleState fields
# ---------------------------------------------------------------------------


class TestReducerAnnotations:
    """Verify each KindleState field is annotated with the correct reducer."""

    def test_str_fields_use_replace(self):
        str_fields = [
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
        ]
        for field in str_fields:
            annotation = KindleState.__annotations__[field]
            # Annotated types expose __metadata__
            assert hasattr(annotation, "__metadata__"), f"{field} is not Annotated"
            assert annotation.__metadata__[0] is _replace, (
                f"{field} should use _replace"
            )

    def test_bool_fields_use_replace_bool(self):
        bool_fields = ["auto_approve", "review_arch", "qa_passed", "cpo_passed"]
        for field in bool_fields:
            annotation = KindleState.__annotations__[field]
            assert hasattr(annotation, "__metadata__"), f"{field} is not Annotated"
            assert annotation.__metadata__[0] is _replace_bool, (
                f"{field} should use _replace_bool"
            )

    def test_int_fields_use_replace_int(self):
        int_fields = [
            "max_concurrent_agents",
            "max_qa_retries",
            "max_cpo_retries",
            "qa_retries",
            "cpo_retries",
            "max_agent_turns",
        ]
        for field in int_fields:
            annotation = KindleState.__annotations__[field]
            assert hasattr(annotation, "__metadata__"), f"{field} is not Annotated"
            assert annotation.__metadata__[0] is _replace_int, (
                f"{field} should use _replace_int"
            )

    def test_list_fields_use_replace_list(self):
        list_fields = ["dev_tasks"]
        for field in list_fields:
            annotation = KindleState.__annotations__[field]
            assert hasattr(annotation, "__metadata__"), f"{field} is not Annotated"
            assert annotation.__metadata__[0] is _replace_list, (
                f"{field} should use _replace_list"
            )

    def test_dict_fields_use_replace_dict(self):
        dict_fields = ["feature_spec"]
        for field in dict_fields:
            annotation = KindleState.__annotations__[field]
            assert hasattr(annotation, "__metadata__"), f"{field} is not Annotated"
            assert annotation.__metadata__[0] is _replace_dict, (
                f"{field} should use _replace_dict"
            )
