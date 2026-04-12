"""Tests for kindle.config — Settings dataclass and environment loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kindle.config import Settings, _find_env_file


# ---------------------------------------------------------------------------
# _find_env_file
# ---------------------------------------------------------------------------


class TestFindEnvFile:
    """Tests for the .env file discovery function."""

    def test_returns_none_when_no_env_file_exists(self, tmp_path, monkeypatch):
        """No candidates exist → None."""
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))
        assert _find_env_file() is None

    def test_finds_env_in_cwd(self, tmp_path, monkeypatch):
        """First candidate: .env in the current working directory."""
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=test")
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        result = _find_env_file()
        assert result is not None
        assert result == env_file

    def test_finds_env_in_parent_of_cwd(self, tmp_path, monkeypatch):
        """Second candidate: .env in the parent of cwd."""
        child = tmp_path / "subdir"
        child.mkdir()
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=test")
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: child))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        result = _find_env_file()
        assert result is not None
        assert result == env_file

    def test_finds_env_in_home_kindle_dir(self, tmp_path, monkeypatch):
        """Third candidate: ~/.kindle/.env."""
        kindle_dir = tmp_path / ".kindle"
        kindle_dir.mkdir()
        env_file = kindle_dir / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=test")

        # cwd and its parent have no .env
        cwd = tmp_path / "a" / "b"
        cwd.mkdir(parents=True)
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: cwd))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        result = _find_env_file()
        assert result is not None
        assert result == env_file

    def test_cwd_env_takes_precedence_over_parent(self, tmp_path, monkeypatch):
        """cwd/.env should be preferred over cwd/../.env."""
        child = tmp_path / "subdir"
        child.mkdir()
        parent_env = tmp_path / ".env"
        parent_env.write_text("parent")
        child_env = child / ".env"
        child_env.write_text("child")

        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: child))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        result = _find_env_file()
        assert result == child_env

    def test_parent_env_takes_precedence_over_home(self, tmp_path, monkeypatch):
        """cwd/../.env should be preferred over ~/.kindle/.env."""
        child = tmp_path / "subdir"
        child.mkdir()
        parent_env = tmp_path / ".env"
        parent_env.write_text("parent")

        kindle_dir = tmp_path / "fakehome" / ".kindle"
        kindle_dir.mkdir(parents=True)
        home_env = kindle_dir / ".env"
        home_env.write_text("home")

        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: child))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        result = _find_env_file()
        assert result == parent_env


# ---------------------------------------------------------------------------
# Settings dataclass defaults
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    """Test the dataclass field defaults (no env loading)."""

    def test_required_api_key(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.anthropic_api_key == "sk-test"

    def test_default_model(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.model == "claude-opus-4-20250514"

    def test_default_max_agent_turns(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.max_agent_turns == 50

    def test_default_max_concurrent_agents(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.max_concurrent_agents == 4

    def test_default_max_qa_retries(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.max_qa_retries == 10

    def test_default_max_cpo_retries(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.max_cpo_retries == 10

    def test_default_projects_root(self):
        s = Settings(anthropic_api_key="sk-test")
        assert s.projects_root == Path.home() / ".kindle" / "projects"

    def test_frozen_dataclass_rejects_mutation(self):
        s = Settings(anthropic_api_key="sk-test")
        with pytest.raises(AttributeError):
            s.model = "other-model"  # type: ignore[misc]

    def test_custom_values_override_defaults(self):
        s = Settings(
            anthropic_api_key="sk-custom",
            model="claude-sonnet-4-20250514",
            max_agent_turns=10,
            max_concurrent_agents=2,
            max_qa_retries=3,
            max_cpo_retries=5,
            projects_root=Path("/tmp/custom"),
        )
        assert s.anthropic_api_key == "sk-custom"
        assert s.model == "claude-sonnet-4-20250514"
        assert s.max_agent_turns == 10
        assert s.max_concurrent_agents == 2
        assert s.max_qa_retries == 3
        assert s.max_cpo_retries == 5
        assert s.projects_root == Path("/tmp/custom")


# ---------------------------------------------------------------------------
# Settings.load()
# ---------------------------------------------------------------------------


class TestSettingsLoad:
    """Tests for Settings.load() which reads from the environment."""

    def test_load_with_all_env_vars(self, monkeypatch, tmp_path):
        """All environment variables provided → all fields populated."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-all-set")
        monkeypatch.setenv("KINDLE_MODEL", "claude-sonnet-4-20250514")
        monkeypatch.setenv("KINDLE_MAX_TURNS", "25")
        monkeypatch.setenv("KINDLE_MAX_CONCURRENT", "8")
        monkeypatch.setenv("KINDLE_MAX_QA_RETRIES", "5")
        monkeypatch.setenv("KINDLE_MAX_CPO_RETRIES", "7")

        # Prevent .env file discovery from interfering
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        s = Settings.load()

        assert s.anthropic_api_key == "sk-all-set"
        assert s.model == "claude-sonnet-4-20250514"
        assert s.max_agent_turns == 25
        assert s.max_concurrent_agents == 8
        assert s.max_qa_retries == 5
        assert s.max_cpo_retries == 7

    def test_load_with_defaults_when_optional_vars_missing(self, monkeypatch, tmp_path):
        """Only ANTHROPIC_API_KEY set → optional fields get defaults."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-minimal")
        # Ensure optional vars are NOT set
        monkeypatch.delenv("KINDLE_MODEL", raising=False)
        monkeypatch.delenv("KINDLE_MAX_TURNS", raising=False)
        monkeypatch.delenv("KINDLE_MAX_CONCURRENT", raising=False)
        monkeypatch.delenv("KINDLE_MAX_QA_RETRIES", raising=False)
        monkeypatch.delenv("KINDLE_MAX_CPO_RETRIES", raising=False)

        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        s = Settings.load()

        assert s.anthropic_api_key == "sk-minimal"
        assert s.model == "claude-opus-4-20250514"
        assert s.max_agent_turns == 50
        assert s.max_concurrent_agents == 4
        assert s.max_qa_retries == 10
        assert s.max_cpo_retries == 10

    def test_load_raises_system_exit_when_api_key_missing(self, monkeypatch, tmp_path):
        """Missing ANTHROPIC_API_KEY → SystemExit with helpful message."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY not set"):
            Settings.load()

    def test_load_raises_system_exit_when_api_key_empty(self, monkeypatch, tmp_path):
        """Empty ANTHROPIC_API_KEY → same SystemExit as missing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY not set"):
            Settings.load()

    def test_load_reads_dotenv_file(self, monkeypatch, tmp_path):
        """Settings.load() should discover and load a .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-from-dotenv\nKINDLE_MODEL=claude-haiku\n")

        # Ensure the env var is NOT already set in the process
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("KINDLE_MODEL", raising=False)

        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        s = Settings.load()

        assert s.anthropic_api_key == "sk-from-dotenv"
        assert s.model == "claude-haiku"

    def test_load_env_var_overrides_dotenv(self, monkeypatch, tmp_path):
        """Process-level env vars should take precedence over .env values.

        python-dotenv's load_dotenv does NOT override existing env vars by
        default, so a pre-existing ANTHROPIC_API_KEY should win.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-from-file\n")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        s = Settings.load()
        assert s.anthropic_api_key == "sk-from-shell"

    def test_load_integer_parsing_from_env(self, monkeypatch, tmp_path):
        """Numeric env vars are correctly parsed as ints."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-int-test")
        monkeypatch.setenv("KINDLE_MAX_TURNS", "100")
        monkeypatch.setenv("KINDLE_MAX_CONCURRENT", "16")
        monkeypatch.setenv("KINDLE_MAX_QA_RETRIES", "0")
        monkeypatch.setenv("KINDLE_MAX_CPO_RETRIES", "1")

        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        s = Settings.load()
        assert s.max_agent_turns == 100
        assert s.max_concurrent_agents == 16
        assert s.max_qa_retries == 0
        assert s.max_cpo_retries == 1

    def test_load_invalid_integer_raises_value_error(self, monkeypatch, tmp_path):
        """Non-numeric value for an int field → ValueError from int()."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bad-int")
        monkeypatch.setenv("KINDLE_MAX_TURNS", "not-a-number")

        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        with pytest.raises(ValueError):
            Settings.load()

    def test_load_calls_find_env_file(self, monkeypatch, tmp_path):
        """Verify load() delegates to _find_env_file() for discovery."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-spy")
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        with patch("kindle.config._find_env_file", return_value=None) as mock_find:
            Settings.load()
            mock_find.assert_called_once()

    def test_load_does_not_call_load_dotenv_when_no_env_file(self, monkeypatch, tmp_path):
        """When _find_env_file returns None, load_dotenv should not be called."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-no-file")
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "fakehome"))

        with patch("kindle.config.load_dotenv") as mock_load:
            Settings.load()
            mock_load.assert_not_called()
