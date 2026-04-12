"""Tests for kindle.config — Settings dataclass and .env discovery."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kindle.config import Settings, _find_env_file

# ---------------------------------------------------------------------------
# _find_env_file
# ---------------------------------------------------------------------------


class TestFindEnvFile:
    """Tests for the .env file search across three candidate directories."""

    def test_returns_cwd_env_when_present(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.touch()
        with patch("kindle.config.Path.cwd", return_value=tmp_path):
            result = _find_env_file()
        assert result == env_file

    def test_returns_parent_env_when_cwd_missing(self, tmp_path: Path) -> None:
        child = tmp_path / "sub"
        child.mkdir()
        parent_env = tmp_path / ".env"
        parent_env.touch()
        with patch("kindle.config.Path.cwd", return_value=child):
            result = _find_env_file()
        assert result == parent_env

    def test_returns_home_kindle_env_when_others_missing(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "home"
        kindle_dir = fake_home / ".kindle"
        kindle_dir.mkdir(parents=True)
        home_env = kindle_dir / ".env"
        home_env.touch()
        cwd = tmp_path / "empty"
        cwd.mkdir()
        with (
            patch("kindle.config.Path.cwd", return_value=cwd),
            patch("kindle.config.Path.home", return_value=fake_home),
        ):
            result = _find_env_file()
        assert result == home_env

    def test_returns_none_when_no_env_file_exists(self, tmp_path: Path) -> None:
        cwd = tmp_path / "empty"
        cwd.mkdir()
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        with (
            patch("kindle.config.Path.cwd", return_value=cwd),
            patch("kindle.config.Path.home", return_value=fake_home),
        ):
            result = _find_env_file()
        assert result is None

    def test_prefers_cwd_over_parent_and_home(self, tmp_path: Path) -> None:
        """When .env exists in all three locations, cwd wins."""
        cwd = tmp_path / "sub"
        cwd.mkdir()
        (cwd / ".env").touch()
        (tmp_path / ".env").touch()  # parent
        fake_home = tmp_path / "home"
        kindle_dir = fake_home / ".kindle"
        kindle_dir.mkdir(parents=True)
        (kindle_dir / ".env").touch()
        with (
            patch("kindle.config.Path.cwd", return_value=cwd),
            patch("kindle.config.Path.home", return_value=fake_home),
        ):
            result = _find_env_file()
        assert result == cwd / ".env"

    def test_prefers_parent_over_home(self, tmp_path: Path) -> None:
        """When .env is missing in cwd but present in parent and home, parent wins."""
        cwd = tmp_path / "sub"
        cwd.mkdir()
        (tmp_path / ".env").touch()  # parent
        fake_home = tmp_path / "home"
        kindle_dir = fake_home / ".kindle"
        kindle_dir.mkdir(parents=True)
        (kindle_dir / ".env").touch()
        with (
            patch("kindle.config.Path.cwd", return_value=cwd),
            patch("kindle.config.Path.home", return_value=fake_home),
        ):
            result = _find_env_file()
        assert result == tmp_path / ".env"


# ---------------------------------------------------------------------------
# Settings.load — successful cases
# ---------------------------------------------------------------------------


class TestSettingsLoad:
    """Tests for Settings.load() with valid environment configurations."""

    @patch("kindle.config.load_dotenv")
    @patch("kindle.config._find_env_file", return_value=None)
    def test_load_with_only_api_key_uses_defaults(self, mock_find: MagicMock, mock_dotenv: MagicMock) -> None:
        env = {"ANTHROPIC_API_KEY": "sk-test-key-123"}
        with patch.dict("os.environ", env, clear=True):
            settings = Settings.load()
        assert settings.anthropic_api_key == "sk-test-key-123"
        assert settings.model == "claude-opus-4-20250514"
        assert settings.max_agent_turns == 50
        assert settings.max_concurrent_agents == 4
        assert settings.max_qa_retries == 10
        assert settings.max_cpo_retries == 10

    @patch("kindle.config.load_dotenv")
    @patch("kindle.config._find_env_file", return_value=None)
    def test_load_with_all_env_vars_set(self, mock_find: MagicMock, mock_dotenv: MagicMock) -> None:
        env = {
            "ANTHROPIC_API_KEY": "sk-custom-key",
            "KINDLE_MODEL": "claude-sonnet-4-20250514",
            "KINDLE_MAX_TURNS": "100",
            "KINDLE_MAX_CONCURRENT": "8",
            "KINDLE_MAX_QA_RETRIES": "5",
            "KINDLE_MAX_CPO_RETRIES": "3",
        }
        with patch.dict("os.environ", env, clear=True):
            settings = Settings.load()
        assert settings.anthropic_api_key == "sk-custom-key"
        assert settings.model == "claude-sonnet-4-20250514"
        assert settings.max_agent_turns == 100
        assert settings.max_concurrent_agents == 8
        assert settings.max_qa_retries == 5
        assert settings.max_cpo_retries == 3

    @patch("kindle.config.load_dotenv")
    @patch("kindle.config._find_env_file", return_value=None)
    def test_load_partial_overrides(self, mock_find: MagicMock, mock_dotenv: MagicMock) -> None:
        """Only some optional vars set; rest use defaults."""
        env = {
            "ANTHROPIC_API_KEY": "sk-key",
            "KINDLE_MAX_TURNS": "25",
        }
        with patch.dict("os.environ", env, clear=True):
            settings = Settings.load()
        assert settings.max_agent_turns == 25
        assert settings.max_concurrent_agents == 4  # default
        assert settings.model == "claude-opus-4-20250514"  # default

    @patch("kindle.config.load_dotenv")
    def test_load_calls_load_dotenv_when_env_file_found(self, mock_dotenv: MagicMock, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        env_path.touch()
        with (
            patch("kindle.config._find_env_file", return_value=env_path),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-key"}, clear=True),
        ):
            Settings.load()
        mock_dotenv.assert_called_once_with(env_path)

    @patch("kindle.config.load_dotenv")
    @patch("kindle.config._find_env_file", return_value=None)
    def test_load_skips_load_dotenv_when_no_env_file(self, mock_find: MagicMock, mock_dotenv: MagicMock) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-key"}, clear=True):
            Settings.load()
        mock_dotenv.assert_not_called()


# ---------------------------------------------------------------------------
# Settings.load — error cases
# ---------------------------------------------------------------------------


class TestSettingsLoadErrors:
    """Tests for Settings.load() when required env vars are missing."""

    @patch("kindle.config.load_dotenv")
    @patch("kindle.config._find_env_file", return_value=None)
    def test_missing_api_key_raises_system_exit(self, mock_find: MagicMock, mock_dotenv: MagicMock) -> None:
        with patch.dict("os.environ", {}, clear=True), pytest.raises(SystemExit) as exc_info:
            Settings.load()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch("kindle.config.load_dotenv")
    @patch("kindle.config._find_env_file", return_value=None)
    def test_empty_api_key_raises_system_exit(self, mock_find: MagicMock, mock_dotenv: MagicMock) -> None:
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            Settings.load()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch("kindle.config.load_dotenv")
    @patch("kindle.config._find_env_file", return_value=None)
    def test_system_exit_message_is_helpful(self, mock_find: MagicMock, mock_dotenv: MagicMock) -> None:
        with patch.dict("os.environ", {}, clear=True), pytest.raises(SystemExit) as exc_info:
            Settings.load()
        msg = str(exc_info.value)
        assert ".env" in msg
        assert "export" in msg or "shell" in msg


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------


class TestSettingsImmutability:
    """The Settings dataclass is frozen; attributes must not be reassignable."""

    def test_cannot_set_api_key(self) -> None:
        settings = Settings(anthropic_api_key="sk-key")
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.anthropic_api_key = "sk-other"  # type: ignore[misc]

    def test_cannot_set_model(self) -> None:
        settings = Settings(anthropic_api_key="sk-key")
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.model = "other-model"  # type: ignore[misc]

    def test_cannot_set_max_agent_turns(self) -> None:
        settings = Settings(anthropic_api_key="sk-key")
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.max_agent_turns = 999  # type: ignore[misc]

    def test_cannot_delete_attribute(self) -> None:
        settings = Settings(anthropic_api_key="sk-key")
        with pytest.raises(dataclasses.FrozenInstanceError):
            del settings.anthropic_api_key  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Settings dataclass defaults (direct construction, not via load)
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    """Verify default field values when constructing Settings directly."""

    def test_default_model(self) -> None:
        s = Settings(anthropic_api_key="sk-key")
        assert s.model == "claude-opus-4-20250514"

    def test_default_max_agent_turns(self) -> None:
        s = Settings(anthropic_api_key="sk-key")
        assert s.max_agent_turns == 50

    def test_default_max_concurrent_agents(self) -> None:
        s = Settings(anthropic_api_key="sk-key")
        assert s.max_concurrent_agents == 4

    def test_default_max_qa_retries(self) -> None:
        s = Settings(anthropic_api_key="sk-key")
        assert s.max_qa_retries == 10

    def test_default_max_cpo_retries(self) -> None:
        s = Settings(anthropic_api_key="sk-key")
        assert s.max_cpo_retries == 10

    def test_default_projects_root(self) -> None:
        s = Settings(anthropic_api_key="sk-key")
        assert s.projects_root == Path.home() / ".kindle" / "projects"

    def test_projects_root_is_path_instance(self) -> None:
        s = Settings(anthropic_api_key="sk-key")
        assert isinstance(s.projects_root, Path)
