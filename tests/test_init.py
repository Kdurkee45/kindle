"""Smoke tests for package __init__.py modules."""


def test_import_kindle() -> None:
    """Importing the top-level kindle package should succeed and expose the docstring."""
    import kindle

    assert kindle.__doc__ is not None
    assert "Kindle" in kindle.__doc__


def test_import_kindle_stages() -> None:
    """Importing the kindle.stages subpackage should succeed and expose the docstring."""
    import kindle.stages

    assert kindle.stages.__doc__ is not None
    assert "stages" in kindle.stages.__doc__


def test_kindle_stages_reachable_from_parent() -> None:
    """The stages subpackage should be reachable via attribute access on the kindle package."""
    import kindle
    import kindle.stages

    assert hasattr(kindle, "stages")
    assert kindle.stages is not None
