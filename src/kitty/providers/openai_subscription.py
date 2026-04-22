"""OpenAI ChatGPT subscription provider — OAuth-authenticated upstream.

This provider authenticates via OpenAI's Codex OAuth flow, which produces a
standard OpenAI API key (via token exchange). The key is billed against the
user's ChatGPT subscription quota, not per-token.

Token lifecycle (refresh, re-exchange) is handled by OAuthSession in kitty.auth.oauth_session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Awaitable

import aiohttp

from kitty.auth.oauth_session import OAuthSession, OAuthRefreshFailed
from kitty.providers.base import ProviderAdapter, ProviderError
from kitty.providers.openai import OpenAIAdapter

logger = logging.getLogger(__name__)

__all__ = ["OpenAISubscriptionAdapter"]


class OpenAISubscriptionAdapter(OpenAIAdapter):
    """OpenAI ChatGPT subscription adapter.

    Uses OAuth (Codex flow) to obtain and refresh an API key. The exchanged
    API key works against the standard OpenAI API at api.openai.com.
    """

    provider_type = "openai_subscription"

    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    # OAuth sessions keyed by resolved file path (cache per profile)
    _session_cache: dict[str, OAuthSession] = {}

    # The bridge server injects _resolved_key into cc_request for custom-transport providers.
    # For openai_subscription, _resolved_key is the path to the OAuthSession JSON file.

    # ── Custom transport ─────────────────────────────────────────────────

    @property
    def use_custom_transport(self) -> bool:
        return True

    async def make_request(self, cc_request: dict) -> dict:
        """Handle a non-streaming request with token refresh."""
        session_file = Path(cc_request["_resolved_key"])
        session = OAuthSession.load(session_file)

        async with aiohttp.ClientSession() as http:
            api_key = await session.get_valid_api_key(http)

        # Build and send the actual OpenAI request using the parent class logic
        upstream_body = self.translate_to_upstream(cc_request)
        upstream_headers = self.build_upstream_headers(api_key)
        url = self.build_base_url(cc_request.get("_provider_config", {}))
        model = cc_request.get("model", "")
        path = self.get_upstream_path(model)
        full_url = f"{url.rstrip('/')}{path}"

        async with aiohttp.ClientSession() as http:
            async with http.post(
                full_url,
                json=upstream_body,
                headers=upstream_headers,
            ) as resp:
                status = resp.status
                try:
                    body = await resp.json()
                except Exception:
                    body = {}

                if status == 401:
                    # API key may have been invalidated — force re-exchange and retry
                    session.api_key_expires_at = 0.0
                    try:
                        async with aiohttp.ClientSession() as http2:
                            api_key = await session.get_valid_api_key(http2)
                            # Retry with fresh key inside the same session
                            upstream_headers = self.build_upstream_headers(api_key)
                            async with http2.post(
                                full_url, json=upstream_body, headers=upstream_headers
                            ) as retry_resp:
                                retry_status = retry_resp.status
                                try:
                                    retry_body = await retry_resp.json()
                                except Exception:
                                    retry_body = {}
                                if retry_status >= 400:
                                    raise self.map_error(retry_status, retry_body)
                                return self.translate_from_upstream(retry_body)
                    except OAuthRefreshFailed as exc:
                        raise ProviderError(
                            f"OpenAI subscription token refresh failed: {exc}. "
                            "Please re-authenticate with 'kitty auth openai'."
                        ) from exc
                        return self.translate_from_upstream(retry_body)

                if status >= 400:
                    raise self.map_error(status, body)

                return self.translate_from_upstream(body)

    async def stream_request(
        self,
        cc_request: dict,
        write: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Handle a streaming request with token refresh."""
        session_file = Path(cc_request["_resolved_key"])
        session = OAuthSession.load(session_file)

        async with aiohttp.ClientSession() as http:
            api_key = await session.get_valid_api_key(http)

        upstream_body = self.translate_to_upstream(cc_request)
        upstream_headers = self.build_upstream_headers(api_key)
        url = self.build_base_url(cc_request.get("_provider_config", {}))
        model = cc_request.get("model", "")
        path = self.get_upstream_path(model)
        full_url = f"{url.rstrip('/')}{path}"

        async with aiohttp.ClientSession() as http:
            async with http.post(
                full_url,
                json=upstream_body,
                headers=upstream_headers,
            ) as resp:
                if resp.status == 401:
                    session.api_key_expires_at = 0.0
                    async with aiohttp.ClientSession() as http2:
                        api_key = await session.get_valid_api_key(http2)
                    upstream_headers = self.build_upstream_headers(api_key)
                    resp.release()
                    resp = await http2.post(
                        full_url, json=upstream_body, headers=upstream_headers
                    )

                if resp.status >= 400:
                    body = await resp.json()
                    raise self.map_error(resp.status, body)

                async for chunk in resp.content.iter_any():
                    if chunk:
                        await write(chunk)

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_obj = body.get("error", body)
        if isinstance(error_obj, dict):
            msg = error_obj.get("message", str(error_obj))
        else:
            msg = str(error_obj)
        code = body.get("error", {}).get("code") if isinstance(body.get("error"), dict) else None
        if code == "invalid_api_key" or status_code == 401:
            return ProviderError(
                f"OpenAI subscription API key invalid or expired. "
                f"Please re-authenticate with 'kitty auth openai'. Details: {msg}"
            )
        return ProviderError(f"OpenAI subscription error {status_code}: {msg}")
