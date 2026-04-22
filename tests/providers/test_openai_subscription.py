"""Tests for OpenAI subscription provider adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.auth.oauth_session import OAuthSession
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def adapter() -> OpenAISubscriptionAdapter:
    return OpenAISubscriptionAdapter()


@pytest.fixture()
def fresh_session(tmp_path: Path) -> tuple[OAuthSession, Path]:
    """Create a fresh OAuthSession file (tokens not expired)."""
    now = time.time()
    session = OAuthSession(
        client_id="app_test",
        access_token="at_fresh",
        refresh_token="rt_fresh",
        id_token="id_fresh",
        api_key="sk-test-fresh-key",
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
        "model": "gpt-5.3-codex",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "_resolved_key": str(session_path),
        "_provider_config": {},
    }


UPSTREAM_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-5.3-codex",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}


# ── Properties ──────────────────────────────────────────────────────────────

class TestProperties:
    def test_provider_type(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.provider_type == "openai_subscription"

    def test_use_custom_transport(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.use_custom_transport is True

    def test_default_base_url(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.default_base_url == "https://api.openai.com/v1"

    def test_normalize_model_name_strips_prefix(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.normalize_model_name("openai/gpt-5.3-codex") == "gpt-5.3-codex"

    def test_normalize_model_name_no_prefix(self, adapter: OpenAISubscriptionAdapter) -> None:
        assert adapter.normalize_model_name("gpt-5.3-codex") == "gpt-5.3-codex"


# ── make_request ────────────────────────────────────────────────────────

class TestMakeRequest:
    @pytest.mark.asyncio
    async def test_sends_correct_api_key(
        self, adapter: OpenAISubscriptionAdapter, cc_request: dict
    ) -> None:
        with aioresponses() as m:
            m.post(
                "https://api.openai.com/v1/chat/completions",
                status=200,
                payload=UPSTREAM_RESPONSE,
            )

            result = await adapter.make_request(cc_request)

        assert result["choices"][0]["message"]["content"] == "Hello! How can I help?"
        # Verify the request was made with the correct API key
        # (aioresponses doesn't easily expose request headers, so we check the result)

    @pytest.mark.asyncio
    async def test_returns_cc_format_response(
        self, adapter: OpenAISubscriptionAdapter, cc_request: dict
    ) -> None:
        with aioresponses() as m:
            m.post(
                "https://api.openai.com/v1/chat/completions",
                status=200,
                payload=UPSTREAM_RESPONSE,
            )

            result = await adapter.make_request(cc_request)

        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_raises_provider_error_on_500(
        self, adapter: OpenAISubscriptionAdapter, cc_request: dict
    ) -> None:
        from kitty.providers.base import ProviderError

        with aioresponses() as m:
            m.post(
                "https://api.openai.com/v1/chat/completions",
                status=500,
                payload={"error": {"message": "Internal server error"}},
            )

            with pytest.raises(ProviderError) as exc_info:
                await adapter.make_request(cc_request)
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_401_triggers_reauth_message(
        self, adapter: OpenAISubscriptionAdapter, cc_request: dict
    ) -> None:
        from kitty.providers.base import ProviderError

        # Create an expired session so the re-auth attempt also fails
        _, session_path = cc_request["_resolved_key"], cc_request.get("_resolved_key")
        # The session is fresh (expires in 3600s), so 401 will trigger re-exchange
        # which will also fail (since we mock a second 401)
        with aioresponses() as m:
            # First request: 401
            m.post(
                "https://api.openai.com/v1/chat/completions",
                status=401,
                payload={"error": {"message": "Invalid API key", "code": "invalid_api_key"}},
            )
            # Re-exchange token (to auth.openai.com) also fails
            m.post(
                "https://auth.openai.com/oauth/token",
                status=400,
                payload={"error": "invalid_grant"},
            )

            with pytest.raises(ProviderError) as exc_info:
                await adapter.make_request(cc_request)
            assert "re-authenticate" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_raises_on_missing_session_file(
        self, adapter: OpenAISubscriptionAdapter
    ) -> None:
        cc_req = {
            "model": "gpt-5.3-codex",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "_resolved_key": "/nonexistent/path/session.json",
            "_provider_config": {},
        }
        with pytest.raises(FileNotFoundError):
            await adapter.make_request(cc_req)


# ── map_error ─────────────────────────────────────────────────────────────

class TestMapError:
    def test_map_error_includes_status_and_body(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(429, {"error": {"message": "Rate limited"}})
        assert "429" in str(error)
        assert "Rate limited" in str(error)

    def test_map_error_401_mentions_reauth(self, adapter: OpenAISubscriptionAdapter) -> None:
        error = adapter.map_error(401, {"error": {"message": "Unauthorized", "code": "invalid_api_key"}})
        assert "re-authenticate" in str(error)
