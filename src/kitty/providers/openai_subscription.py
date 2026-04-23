"""OpenAI ChatGPT subscription provider — Codex backend via OAuth.

This provider authenticates via OpenAI's Codex OAuth flow, which produces
an access_token JWT that works against the Codex backend at
chatgpt.com/backend-api/codex/responses (Responses API format).

The access_token JWT is NOT a valid Bearer token for api.openai.com — it
only works with the ChatGPT backend.  The token exchange (id_token → API key)
is attempted but treated as best-effort; for org accounts without a Platform
API org mapping, the exchange fails and we fall back to using the access_token
directly against the Codex backend.

Token lifecycle (refresh, re-exchange) is handled by OAuthSession in
kitty.auth.oauth_session.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

import aiohttp

from kitty.auth.oauth_session import OAuthSession, OAuthRefreshFailed
from kitty.providers.base import ProviderError

# Avoid circular import — only need the parent class methods
from kitty.providers.openai import OpenAIAdapter

__all__ = ["OpenAISubscriptionAdapter"]

logger = logging.getLogger(__name__)

# Codex backend endpoint for ChatGPT subscription access
_CODEX_BACKEND_URL = "https://chatgpt.com/backend-api/codex/responses"
_CODEX_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)


def _convert_content_types(items: list) -> list:
    """Ensure content types match the role in Responses API format.

    The Codex backend validates content types strictly:
    - User messages must use ``input_text``
    - Assistant messages must use ``output_text``
    """
    converted = []
    for item in items:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        new_item = dict(item)
        role = new_item.get("role", "")
        content = new_item.get("content")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if role == "assistant" and part_type == "input_text":
                        new_parts.append({**part, "type": "output_text"})
                    elif role == "user" and part_type == "output_text":
                        new_parts.append({**part, "type": "input_text"})
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            new_item["content"] = new_parts
        converted.append(new_item)
    return converted


class OpenAISubscriptionAdapter(OpenAIAdapter):
    """OpenAI ChatGPT subscription adapter using the Codex backend.

    Uses OAuth (Codex PKCE flow) to obtain an access_token JWT.  Routes
    requests to chatgpt.com/backend-api/codex/responses (Responses API).
    """

    provider_type = "openai_subscription"

    @property
    def requires_oauth(self) -> bool:
        return True

    @property
    def default_base_url(self) -> str:
        # NOTE: This is the *upstream* URL for the Codex backend, not
        # api.openai.com.  The JWT from Codex OAuth only works here.
        return _CODEX_BACKEND_URL

    @property
    def use_custom_transport(self) -> bool:
        return True

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_account_id(id_token: str) -> str | None:
        """Extract chatgpt_account_id from the JWT payload (no sig verify)."""
        try:
            payload_b64 = id_token.split(".")[1]
            # Add padding (modulo to avoid over-padding when already aligned)
            missing = (4 - len(payload_b64) % 4) % 4
            payload_b64 += "=" * missing
            payload = json.loads(__import__("base64").urlsafe_b64decode(payload_b64))
            auth_ns = payload.get("https://api.openai.com/auth", {})
            return auth_ns.get("chatgpt_account_id")
        except Exception:
            return None

    def _build_codex_headers(
        self, access_token: str, id_token: str
    ) -> dict[str, str]:
        """Build headers for the Codex backend request."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "session_id": uuid.uuid4().hex,
            # Mimic Codex CLI browser-like headers for Cloudflare bypass
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Codex-CLI/0.1.0 Chrome/136.0.0.0 "
                "Safari/537.36"
            ),
            "Origin": "https://chatgpt.com",
            "Referer": "https://chatgpt.com/",
        }
        # NOTE: Do NOT set originator: codex_cli_rs — it triggers strict
        # tool validation that only allows Codex CLI's built-in tools.
        account_id = self._extract_account_id(id_token)
        if account_id:
            headers["chatgpt-account-id"] = account_id
        return headers

    @staticmethod
    def _load_session(cc_request: dict) -> OAuthSession:
        """Load the OAuthSession from the resolved key file path."""
        key = cc_request.get("_resolved_key")
        if not key:
            raise ProviderError("Missing OAuth session file path (_resolved_key)")
        session_file = Path(key)
        if not session_file.exists():
            raise ProviderError(f"OAuth session file not found: {session_file}")
        return OAuthSession.load(session_file)

    _ALLOWED_RESPONSES_PARAMS = frozenset({
        "model", "stream", "store", "instructions", "input",
        "tools", "tool_choice", "parallel_tool_calls", "include",
        "reasoning",
    })

    @staticmethod
    def _prepare_responses_body(original_body: dict) -> dict:
        """Clean a Responses API request for the Codex backend.

        The Codex backend uses strict parameter validation — it rejects
        any parameter not in the Codex CLI's allowlist (e.g.
        ``max_output_tokens``, ``temperature``, ``strict`` on tools).
        This method builds a clean body from only the allowed fields.
        """
        # Log dropped parameters so users understand why settings don't apply
        dropped = set(original_body) - OpenAISubscriptionAdapter._ALLOWED_RESPONSES_PARAMS
        if dropped:
            logger.debug("Codex backend unsupported parameters (dropped): %s", sorted(dropped))

        # Only pass parameters that the Codex backend accepts.
        # Additional parameters cause 400 "Unsupported parameter: X".
        body: dict = {
            "model": original_body.get("model", "gpt-5.4"),
            "stream": True,
            "store": False,
        }

        if original_body.get("instructions"):
            body["instructions"] = original_body["instructions"]
        if original_body.get("input"):
            # Convert input_text → output_text in content parts
            # (Codex backend only supports output_text and refusal)
            body["input"] = _convert_content_types(original_body["input"])
        if original_body.get("tools"):
            # Strip ``strict`` — the Codex backend rejects it on tools
            tools = []
            for tool in original_body["tools"]:
                t = {k: v for k, v in tool.items() if k != "strict"}
                tools.append(t)
            body["tools"] = tools
        if original_body.get("tool_choice"):
            body["tool_choice"] = original_body["tool_choice"]
        if original_body.get("parallel_tool_calls") is not None:
            body["parallel_tool_calls"] = original_body["parallel_tool_calls"]
        if original_body.get("include"):
            body["include"] = original_body["include"]

        # Reasoning effort: pass through from original body or inject from metadata
        if original_body.get("reasoning"):
            body["reasoning"] = original_body["reasoning"]
        elif original_body.get("_reasoning_effort") and original_body["_reasoning_effort"] != "none":
            body["reasoning"] = {"effort": original_body["_reasoning_effort"]}

        return body

    # ── Custom transport ──────────────────────────────────────────────────

    async def make_request(self, cc_request: dict) -> dict:
        """Handle a non-streaming request via the Codex backend.

        The Codex backend requires streaming — ``stream: false`` is rejected
        with ``"Stream must be set to true"``.  We always send with
        ``stream: true`` and collect all SSE events into a single response.
        """
        session = self._load_session(cc_request)

        async with aiohttp.ClientSession() as http:
            try:
                access_token = await session.get_valid_api_key(http)
            except OAuthRefreshFailed as exc:
                raise ProviderError(
                    f"Authentication refresh failed. "
                    f"Please re-authenticate with 'kitty auth openai'. Details: {exc}"
                )

            original_body = cc_request.get("_original_body")
            if original_body:
                resp_body = self._prepare_responses_body(original_body)
                # Inject reasoning effort from normalized metadata if not already present
                if "reasoning" not in resp_body:
                    effort = cc_request.get("_reasoning_effort")
                    if effort and effort != "none":
                        resp_body["reasoning"] = {"effort": effort}
            else:
                resp_body = self._cc_to_responses(cc_request)

            # Codex backend requires streaming
            resp_body["stream"] = True

            headers = self._build_codex_headers(access_token, session.id_token)

            async with http.post(
                _CODEX_BACKEND_URL, json=resp_body, headers=headers,
                timeout=_CODEX_TIMEOUT,
            ) as resp:
                if resp.status >= 400:
                    try:
                        body = await resp.json()
                    except Exception:
                        body = {}
                    raise self.map_error(resp.status, body)
                # Read the full streamed response
                raw = await resp.read()
                return self._parse_sse_to_response(raw)

    async def stream_request(
        self,
        cc_request: dict,
        write: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Handle a streaming request via the Codex backend.

        If ``_original_body`` is present (Responses API path), forwards the
        Responses API request directly and passes SSE events through.
        Otherwise converts from CC to Responses API format.
        """
        session = self._load_session(cc_request)

        async with aiohttp.ClientSession() as http:
            try:
                access_token = await session.get_valid_api_key(http)
            except OAuthRefreshFailed as exc:
                raise ProviderError(
                    f"Authentication refresh failed. "
                    f"Please re-authenticate with 'kitty auth openai'. Details: {exc}"
                )

            original_body = cc_request.get("_original_body")
            if original_body:
                resp_body = self._prepare_responses_body(original_body)
                # Inject reasoning effort from normalized metadata if not already present
                if "reasoning" not in resp_body:
                    effort = cc_request.get("_reasoning_effort")
                    if effort and effort != "none":
                        resp_body["reasoning"] = {"effort": effort}
            else:
                resp_body = self._cc_to_responses(cc_request)

            # Debug: log the full request body for diagnosis
            logger.debug(
                "Codex backend request: %s",
                json.dumps(resp_body, indent=2, ensure_ascii=False)[:3000],
            )

            headers = self._build_codex_headers(access_token, session.id_token)

            async with http.post(
                _CODEX_BACKEND_URL, json=resp_body, headers=headers,
                timeout=_CODEX_TIMEOUT,
            ) as resp:
                if resp.status >= 400:
                    raw = await resp.text()
                    body = {}
                    try:
                        body = json.loads(raw)
                    except Exception:
                        pass
                    # Log raw response at DEBUG level for diagnosis (may contain
                    # sensitive tokens/PII — don't leak to client).
                    logger.debug("Codex backend error %d: %s", resp.status, raw[:500])
                    if not body:
                        body = {"error": {"message": f"Codex backend returned {resp.status}"}}
                    raise self.map_error(resp.status, body)
                async for chunk in resp.content.iter_any():
                    if chunk:
                        # Strip UTF-8 BOM that some responses include
                        cleaned = chunk.replace(b"\xef\xbb\xbf", b"")
                        if cleaned:
                            await write(cleaned)

    def _cc_to_responses(self, cc_request: dict) -> dict:
        """Convert a CC (Chat Completions) request to Responses API format.

        This is used when the client sends a Chat Completions request
        (e.g. via /v1/chat/completions) to an openai_subscription profile.
        """
        # CC params that have no Responses API equivalent
        _cc_only_params = frozenset({
            "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "frequency_penalty", "presence_penalty", "logprobs",
            "top_logprobs", "response_format", "stop", "n",
            "stream_options", "seed", "logit_bias",
        })
        dropped = set(cc_request) & _cc_only_params
        if dropped:
            logger.debug("CC parameters with no Codex equivalent (dropped): %s", sorted(dropped))

        messages = cc_request.get("messages", [])
        instructions = ""
        input_items = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                instructions += (content + "\n") if instructions else content

            elif role == "assistant":
                # Assistant message — may contain text and/or tool_calls
                item: dict = {
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                }
                if content:
                    item["content"].append(
                        {"type": "output_text", "text": str(content)}
                    )
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                    })
                if item["content"]:
                    input_items.append(item)

            elif role == "user":
                if content:
                    input_items.append({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": str(content)}],
                    })

            elif role == "tool":
                # Tool result → function_call_output
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": str(content),
                })

        body: dict = {
            "model": cc_request.get("model", "gpt-5.4"),
            "input": input_items,
            "stream": True,
            "store": False,
        }
        if instructions:
            body["instructions"] = instructions.strip()

        # Map tools from CC format to Responses API format
        cc_tools = cc_request.get("tools", [])
        if cc_tools:
            resp_tools = []
            for tool in cc_tools:
                func = tool.get("function", {})
                resp_tools.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            body["tools"] = resp_tools

        if cc_request.get("tool_choice"):
            body["tool_choice"] = cc_request["tool_choice"]

        # Inject reasoning effort from normalized metadata
        effort = cc_request.get("_reasoning_effort")
        if effort and effort != "none":
            body["reasoning"] = {"effort": effort}

        # NOTE: Do NOT include max_output_tokens or temperature —
        # the Codex backend rejects them with 400 "Unsupported parameter".

        return body

    @staticmethod
    def _parse_sse_to_response(raw: bytes) -> dict:
        """Parse a full SSE stream from the Codex backend into a CC response.

        Extracts text content, tool calls, usage, and model info from the
        Responses API SSE events.
        """
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_args: dict[str, str] = {}  # call_id → accumulated args
        model = ""
        finish_reason = "stop"
        usage: dict = {}

        for line in raw.decode("utf-8", errors="replace").split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    text_parts.append(delta)

            elif event_type == "response.function_call_arguments.delta":
                call_id = event.get("call_id", "")
                delta = event.get("delta", "")
                tool_args[call_id] = tool_args.get(call_id, "") + delta

            elif event_type == "response.function_call_arguments.done":
                # Some backends send the complete arguments in a done event
                call_id = event.get("call_id", "")
                args = event.get("arguments", "")
                if call_id and args and not tool_args.get(call_id):
                    tool_args[call_id] = args

            elif event_type == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    idx = len(tool_calls)
                    tool_calls.append({
                        "index": idx,
                        "id": item.get("call_id", f"call_{idx}"),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": "",
                        },
                    })

            elif event_type == "response.output_item.done":
                # Some backends include full item data in the done event
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", "")
                    # Check if we already registered this call from .added
                    existing = next(
                        (tc for tc in tool_calls if tc["id"] == call_id), None
                    )
                    if existing is None:
                        idx = len(tool_calls)
                        tool_calls.append({
                            "index": idx,
                            "id": call_id or f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", ""),
                            },
                        })
                        if call_id and item.get("arguments"):
                            tool_args[call_id] = item["arguments"]
                    elif call_id and item.get("arguments"):
                        # Update args if not yet captured via delta events
                        if not tool_args.get(call_id):
                            tool_args[call_id] = item["arguments"]

            elif event_type == "response.completed":
                resp_data = event.get("response", {})
                model = resp_data.get("model", model)
                resp_usage = resp_data.get("usage", {})
                if resp_usage:
                    usage = resp_usage
                status = resp_data.get("status", "completed")
                if status == "incomplete":
                    finish_reason = "length"

                # Check for tool calls embedded in completed output items
                for item in resp_data.get("output", []):
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        existing = next(
                            (tc for tc in tool_calls if tc["id"] == call_id), None
                        )
                        if existing is None:
                            idx = len(tool_calls)
                            tool_calls.append({
                                "index": idx,
                                "id": call_id or f"call_{idx}",
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", ""),
                                },
                            })
                            if call_id and item.get("arguments"):
                                tool_args[call_id] = item["arguments"]
                        elif call_id and item.get("arguments"):
                            if not tool_args.get(call_id):
                                tool_args[call_id] = item["arguments"]

            else:
                # Log unknown event types for debugging
                logger.debug("Unknown Codex SSE event type: %s", event_type)

        # Finalize tool call arguments
        for tc in tool_calls:
            call_id = tc["id"]
            tc["function"]["arguments"] = tool_args.get(call_id, "")

        # Debug: warn if tool calls have empty arguments (helps diagnose parsing issues)
        if tool_calls:
            for tc in tool_calls:
                if not tc["function"]["arguments"]:
                    logger.warning(
                        "Tool call '%s' (id=%s) has empty arguments after SSE parsing",
                        tc["function"]["name"], tc["id"],
                    )
                    # Dump raw SSE for diagnosis
                    import tempfile
                    try:
                        dump_path = Path(tempfile.gettempdir()) / "kitty_codex_sse_dump.txt"
                        dump_path.write_bytes(raw)
                        logger.warning("Raw SSE dumped to %s", dump_path)
                    except Exception:
                        pass

        content = "".join(text_parts)
        message: dict = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_obj = body.get("error", body)
        if isinstance(error_obj, dict):
            msg = error_obj.get("message", str(error_obj))
        else:
            msg = str(error_obj)
        if status_code == 401:
            return ProviderError(
                f"OpenAI subscription auth failed. "
                f"Please re-authenticate with 'kitty auth openai'. Details: {msg}"
            )
        if status_code == 429:
            return ProviderError(
                f"OpenAI subscription rate limited: {msg}"
            )
        if status_code == 403:
            return ProviderError(
                f"OpenAI subscription access denied: {msg}"
            )
        if status_code >= 500:
            return ProviderError(
                f"OpenAI subscription server error {status_code}: {msg}"
            )
        return ProviderError(f"OpenAI subscription error {status_code}: {msg}")
