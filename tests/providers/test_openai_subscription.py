"""Tests for OpenAI subscription provider adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.providers.openai_subscription import (
    _CODEX_BACKEND_URL,
    OpenAISubscriptionAdapter,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def adapter() -> OpenAISubscriptionAdapter:
    return OpenAISubscriptionAdapter()


def _make_id_token(account_id: str | None = None) -> str:
    """Build a minimal JWT id_token with an optional chatgpt_account_id."""
    header = "eyJhbGciOiJIUzI1NiJ9"
    payload_dict: dict = {}
    if account_id is not None:
        payload_dict["https://api.openai.com/auth"] = {
            "chatgpt_account_id": account_id,
        }
    payload_json = json.dumps(payload_dict)
    import base64
    payload = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode()
    signature = "fake_sig"
    return f"{header}.{payload}.{signature}"


@pytest.fixture()
def fresh_session(tmp_path: Path) -> tuple[OAuthSession, Path]:
    """Create a fresh OAuthSession file (tokens not expired)."""
    now = time.time()
    id_token = _make_id_token("acct-1234")
    session = OAuthSession(
        client_id="app_test",
        access_token="at_fresh",
        refresh_token="rt_fresh",
        id_token=id_token,
        api_key=None,
        access_token_expires_at=now + 3600,
        api_key_expires_at=now + 3600,
        _file_path=str(tmp_path / "oauth_session.json"),
    )
    session.save()
    return session, Path(session._file_path)


@pytest.fixture()
def cc_request(fresh_session: tuple[OAuthSession, Path]) -> dict:
    """A typical cc_request with _resolved_key pointing to the session file."""
    _, session_path = fresh_session
    return {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
        "tools": [
            {
                "function": {
                    "name": "bash",
                    "description": "Run a shell command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
                "type": "function",
            }
        ],
        "_resolved_key": str(session_path),
        "_provider_config": {},
    }


@pytest.fixture()
def responses_body() -> dict:
    """A typical Responses API request body."""
    return {
        "model": "gpt-5.4",
        "instructions": "You are helpful.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        "stream": True,
        "tools": [
            {
                "type": "function",
                "name": "bash",
                "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                "strict": True,  # Should be stripped
            }
        ],
    }


# ── Properties ──────────────────────────────────────────────────────────────

class TestProperties:
    def test_provider_type(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.provider_type == "openai_subscription"

    def test_use_custom_transport(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.use_custom_transport is True

    def test_default_base_url_is_codex_backend(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.default_base_url == _CODEX_BACKEND_URL
        assert "chatgpt.com" in adapter.default_base_url

    def test_normalize_model_name_strips_prefix(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.normalize_model_name("openai/gpt-5.4") == "gpt-5.4"

    def test_normalize_model_name_no_prefix(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.normalize_model_name("gpt-5.4") == "gpt-5.4"


# ── Helpers ──────────────────────────────────────────────────────────────

class TestExtractAccountId:
    def test_extracts_account_id_from_jwt(self) -> None:
        token = _make_id_token("my-acct-123")
        assert OpenAISubscriptionAdapter._extract_account_id(token) == "my-acct-123"

    def test_returns_none_if_missing(self) -> None:
        token = _make_id_token(None)
        assert OpenAISubscriptionAdapter._extract_account_id(token) is None

    def test_returns_none_on_garbage(self) -> None:
        assert OpenAISubscriptionAdapter._extract_account_id("not-a-jwt") is None


class TestPrepareResponsesBody:
    def test_strips_strict_from_tools(self) -> None:
        body = {
            "tools": [
                {"type": "function", "name": "bash", "strict": True, "parameters": {}},
            ],
        }
        result = OpenAISubscriptionAdapter._prepare_responses_body(body)
        assert "strict" not in result["tools"][0]

    def test_sets_store_false(self) -> None:
        body = {}
        result = OpenAISubscriptionAdapter._prepare_responses_body(body)
        assert result["store"] is False

    def test_sets_stream_true(self) -> None:
        body = {}
        result = OpenAISubscriptionAdapter._prepare_responses_body(body)
        assert result["stream"] is True


class TestCcToResponses:
    def test_converts_system_message_to_instructions(self, adapter: OpenAISubscriptionAdapter) -> None:
        cc = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = adapter._cc_to_responses(cc)
        assert result["instructions"] == "Be helpful."
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"

    def test_converts_tools(self, adapter: OpenAISubscriptionAdapter) -> None:
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
        result = adapter._cc_to_responses(cc)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "bash"
        assert "function" not in result["tools"][0]

    def test_does_not_include_max_output_tokens(self, adapter: OpenAISubscriptionAdapter) -> None:
        """The Codex backend rejects max_output_tokens with 400."""
        cc = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 4096,
        }
        result = adapter._cc_to_responses(cc)
        assert "max_output_tokens" not in result
        assert "max_tokens" not in result


# ── map_error ─────────────────────────────────────────────────────────────

class TestMapError:
    def test_map_error_429_rate_limited(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(429, {"error": {"message": "Rate limited"}})
        assert "rate limited" in str(error).lower()
        assert "Rate limited" in str(error)

    def test_map_error_400_includes_status(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(400, {"error": {"message": "Bad request"}})
        assert "400" in str(error)
        assert "Bad request" in str(error)

    def test_map_error_401_mentions_reauth(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(401, {"error": {"message": "Unauthorized", "code": "invalid_api_key"}})
        assert "re-authenticate" in str(error)
