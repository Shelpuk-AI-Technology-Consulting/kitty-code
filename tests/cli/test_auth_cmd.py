"""Tests for kitty auth openai command."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.profiles.store import ProfileStore

# Module path for patching
_MOD = "kitty.cli.auth_cmd"


@contextmanager
def _mock_tty():
    """Context manager to mock TTY state as True."""
    with patch("sys.stdin.isatty", return_value=True):
        yield


@pytest.fixture()
def store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(path=tmp_path / "profiles.json")


@pytest.fixture()
def cred_store(tmp_path: Path) -> CredentialStore:
    return CredentialStore(backends=[FileBackend(path=tmp_path / "creds.json")])


def _make_oauth_session() -> OAuthSession:
    """Return a valid-looking OAuthSession for testing."""
    return OAuthSession(
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        access_token="at_test",
        refresh_token="rt_test",
        id_token="id_test",
        api_key="sk-test-openai-key",
        access_token_expires_at=9999999999.0,
        api_key_expires_at=9999999999.0,
        _file_path=None,
    )


def _mock_questionary_text(responses):
    """Create a mock for questionary.text().ask() returning given responses.

    questionary.text("label", default="x").ask() is a chain of calls:
      1. questionary.text("label", default="x") -> question object
      2. .ask() -> string response

    We mock questionary.text to return an object whose ask() returns the next response.
    """
    call_count = 0

    def _text_factory(*args, **kwargs):
        nonlocal call_count
        idx = call_count
        call_count += 1
        mock_q = MagicMock()
        mock_q.ask.return_value = responses[idx]
        return mock_q

    return MagicMock(side_effect=_text_factory)


def _mock_questionary_confirm(response=True):
    """Create a mock for questionary.confirm().ask()."""
    mock_q = MagicMock()
    mock_q.ask.return_value = response
    return MagicMock(return_value=mock_q)


class TestAuthOpenaiProfileCreation:
    def test_auth_cmd_creates_profile_with_openai_subscription_provider(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """Profile is saved with provider=openai_subscription after successful OAuth."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_flow", new=AsyncMock(return_value=session)),
            patch(f"{_MOD}.questionary.text", _mock_questionary_text(["gpt-5.3-codex", "my-openai-profile"])),
            patch(f"{_MOD}.questionary.confirm", _mock_questionary_confirm(True)),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=MagicMock(valid=True))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        assert profile is not None
        assert profile.provider == "openai_subscription"
        assert profile.model == "gpt-5.3-codex"
        assert profile.name == "my-openai-profile"

        # Verify saved to store
        loaded = store.get("my-openai-profile")
        assert loaded is not None
        assert loaded.provider == "openai_subscription"

    def test_auth_cmd_stores_oauth_session_in_file(
        self, store: ProfileStore, cred_store: CredentialStore, tmp_path: Path
    ) -> None:
        """OAuth session file is created and credential store holds the path."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_flow", new=AsyncMock(return_value=session)),
            patch(f"{_MOD}.questionary.text", _mock_questionary_text(["gpt-5.3-codex", "openai-sub"])),
            patch(f"{_MOD}.questionary.confirm", _mock_questionary_confirm(True)),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=MagicMock(valid=True))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        # Session file path should be stored in credential store
        stored_path = cred_store.get(profile.auth_ref)
        assert stored_path is not None

        session_file = Path(stored_path)
        assert session_file.exists(), f"Session file not found at {session_file}"

        # Session file should contain valid JSON
        with open(session_file) as f:
            data = json.load(f)
        assert data["client_id"] == session.client_id
        assert data["api_key"] == session.api_key

    def test_auth_cmd_asks_for_model_and_profile_name(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """Command prompts for model name and profile name via questionary."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_flow", new=AsyncMock(return_value=session)),
            patch(f"{_MOD}.questionary.text", _mock_questionary_text(["custom-model", "my-profile"])),
            patch(f"{_MOD}.questionary.confirm", _mock_questionary_confirm(False)),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=MagicMock(valid=True))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        assert profile.model == "custom-model"
        assert profile.name == "my-profile"
        assert profile.is_default is False

    def test_auth_cmd_uses_default_model_and_profile_name(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """When user accepts defaults (presses Enter), gpt-5.3-codex and openai-sub are used.

        questionary.text with a default returns the default value when user presses Enter.
        """
        session = _make_oauth_session()

        # Simulate questionary returning defaults (as it does when user presses Enter)
        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_flow", new=AsyncMock(return_value=session)),
            patch(f"{_MOD}.questionary.text", _mock_questionary_text(["gpt-5.3-codex", "openai-sub"])),
            patch(f"{_MOD}.questionary.confirm", _mock_questionary_confirm(False)),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=MagicMock(valid=True))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        assert profile.model == "gpt-5.3-codex"
        assert profile.name == "openai-sub"

    def test_auth_cmd_raises_if_oauth_flow_fails(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """OAuth flow failure is printed and re-raised."""

        class OAuthFailure(Exception):
            pass

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_flow", new=AsyncMock(side_effect=OAuthFailure("Browser not available"))),
            patch(f"{_MOD}.print_error") as mock_error,
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            with pytest.raises(OAuthFailure):
                asyncio.run(run_auth_openai(store, cred_store))

        mock_error.assert_called()
        error_call = mock_error.call_args[0][0]
        assert "OpenAI OAuth flow failed" in error_call

    def test_auth_cmd_validation_failure_warns_but_creates_profile(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """Validation failure warns the user but the profile is still created."""
        session = _make_oauth_session()

        validation_result = MagicMock()
        validation_result.valid = False
        validation_result.reason = "Invalid API key"

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_flow", new=AsyncMock(return_value=session)),
            patch(f"{_MOD}.questionary.text", _mock_questionary_text(["gpt-5.3-codex", "test-profile"])),
            patch(f"{_MOD}.questionary.confirm", _mock_questionary_confirm(True)),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=validation_result)),
            patch(f"{_MOD}.print_warning") as mock_warn,
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        assert profile is not None
        assert store.get("test-profile") is not None
        mock_warn.assert_called()
        warn_call = mock_warn.call_args[0][0]
        assert "validation failed" in warn_call

    def test_auth_cmd_non_tty_raises(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """Non-TTY environment raises NonTTYError."""
        from kitty.tui.prompts import NonTTYError

        with patch("sys.stdin.isatty", return_value=False):
            from kitty.cli.auth_cmd import run_auth_openai

            with pytest.raises(NonTTYError):
                asyncio.run(run_auth_openai(store, cred_store))
