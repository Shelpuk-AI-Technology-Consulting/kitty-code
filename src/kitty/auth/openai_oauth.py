"""OpenAI OAuth flow orchestrator using OpenID Connect / OAuth 2.0 authorization code flow.

This module implements the OpenAI Codex CLI OAuth flow as documented at
https://auth.openai.com/oauth/authorize with PKCE (RFC 7636) support.

The full flow:
  1. Generate PKCE code_verifier + code_challenge (S256)
  2. Start a local callback server on port 1455 (fallback 1456-1459)
  3. Open the authorization URL in the user's browser
  4. Wait for the browser to redirect to localhost with the auth code
  5. Exchange the auth code for tokens (authorization_code grant)
  6. Exchange the id_token for an OpenAI API key (token-exchange grant)
  7. Return an OAuthSession wrapping the tokens and API key
"""

from __future__ import annotations

import asyncio
import secrets
import urllib.parse
import webbrowser
from typing import TYPE_CHECKING

import aiohttp

from kitty.auth.oauth_session import (
    ID_TOKEN_TYPE,
    OAuthSession,
    TOKEN_EXCHANGE_GRANT,
)
from kitty.auth.pkce import generate_code_challenge, generate_code_verifier

if TYPE_CHECKING:
    from asyncio import Future

# ── OpenAI OAuth endpoints ───────────────────────────────────────────────
OAUTH_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"

# OpenAI Codex CLI OAuth client ID
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Redirect URI registered for the local callback server
REDIRECT_URI = "http://localhost:1455/auth/callback"

# OAuth scope requested (space-separated, will be URL-encoded in the URL)
SCOPE = "openid profile email offline_access"

# Callback server port range
_CALLBACK_PORT_START = 1455
_CALLBACK_PORT_END = 1459

# OAuth flow timeout (5 minutes)
OAUTH_TIMEOUT_SECONDS = 300


# ── Exceptions ───────────────────────────────────────────────────────────
class OAuthPortConflictError(Exception):
    """Raised when none of the callback ports (1455-1459) are available."""

    pass


class OAuthTimeoutError(Exception):
    """Raised when the OAuth callback is not received within the timeout period."""

    pass


class OAuthAuthorizationError(Exception):
    """Raised when the authorization server returns an error."""

    def __init__(self, error: str, error_description: str | None = None) -> None:
        self.error = error
        self.error_description = error_description or ""
        super().__init__(
            f"{error}: {error_description}" if error_description else error
        )


# ── URL builder ──────────────────────────────────────────────────────────
def build_auth_url(code_verifier: str, state: str | None = None) -> str:
    """Build the OpenAI authorization URL with PKCE parameters.

    Args:
        code_verifier: The PKCE code verifier string.
        state: Optional CSRF state token. If not provided, a random one is
               generated.

    Returns:
        The full authorization URL string.
    """
    code_challenge = generate_code_challenge(code_verifier)
    if state is None:
        state = secrets.token_urlsafe(32)

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"


# ── Callback server ──────────────────────────────────────────────────────
async def _start_callback_server(
    state: str,
    code_future: "Future[str | OAuthAuthorizationError]",
    host: str,
    port: int,
) -> aiohttp.web.Application:
    """Create an aiohttp web application that handles the OAuth callback.

    Routes:
        GET /              -- Landing page with a link to the auth URL
        GET /auth/callback -- OAuth redirect endpoint; validates state and
                              resolves *code_future* with either the auth
                              code (success) or an OAuthAuthorizationError
                              (error or state mismatch).

    Args:
        state: The expected state parameter (used for CSRF protection).
        code_future: A Future that will be resolved with the auth code or
                     an OAuthAuthorizationError when the callback is received.
        host: Host to bind to.
        port: Port to bind to.

    Returns:
        A configured aiohttp.web.Application.
    """
    app = aiohttp.web.Application()

    async def handle_index(request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.Response(
            text="<html><body><p>Waiting for OpenAI authorization...</p>"
            "<p>You can close this tab.</p></body></html>",
            content_type="text/html",
        )

    async def handle_callback(request: aiohttp.web.Request) -> aiohttp.web.Response:
        received_state = request.query.get("state", "")
        if received_state != state:
            code_future.set_result(
                OAuthAuthorizationError(
                    "state_mismatch",
                    f"State mismatch: expected {state!r}, got {received_state!r}",
                )
            )
            return aiohttp.web.Response(
                status=400,
                text="<html><body><p>State mismatch. Please try again.</p></body></html>",
                content_type="text/html",
            )

        # Check for OAuth error response
        error = request.query.get("error")
        if error:
            error_description = request.query.get("error_description", "")
            oauth_error = OAuthAuthorizationError(error, error_description)
            code_future.set_result(oauth_error)
            return aiohttp.web.Response(
                status=400,
                text=f"<html><body><p>Authorization error: {error}</p>"
                f"<p>{error_description}</p></body></html>",
                content_type="text/html",
            )

        code = request.query.get("code", "")
        code_future.set_result(code)
        return aiohttp.web.Response(
            status=200,
            text="<html><body><p>Authorization successful!</p>"
            "<p>You can close this tab and return to the terminal.</p></body></html>",
            content_type="text/html",
        )

    app.router.add_get("/", handle_index)
    app.router.add_get("/auth/callback", handle_callback)

    return app


# ── Token exchange ───────────────────────────────────────────────────────
async def _exchange_code_for_tokens(
    code: str,
    code_verifier: str,
    client_id: str,
    http: aiohttp.ClientSession,
) -> dict:
    """Exchange an authorization code for tokens.

    Args:
        code: The authorization code from the OAuth callback.
        code_verifier: The PKCE code verifier.
        client_id: The OAuth client ID.
        http: an aiohttp ClientSession for making the request.

    Returns:
        Parsed JSON response from the token endpoint.
    """
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    async with http.post(OAUTH_TOKEN_URL, data=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _exchange_id_token_for_api_key(
    id_token: str,
    client_id: str,
    http: aiohttp.ClientSession,
) -> str:
    """Exchange an id_token for an OpenAI API key via token-exchange grant.

    Args:
        id_token: The OpenID Connect id_token.
        client_id: The OAuth client ID.
        http: An aiohttp ClientSession for making the request.

    Returns:
        The exchanged OpenAI API key string.
    """
    payload = {
        "grant_type": TOKEN_EXCHANGE_GRANT,
        "requested_token": "openai-api-key",
        "subject_token": id_token,
        "subject_token_type": ID_TOKEN_TYPE,
        "client_id": client_id,
    }
    async with http.post(OAUTH_TOKEN_URL, data=payload) as resp:
        resp.raise_for_status()
        result = await resp.json()
    return result["openai_api_key"]


# ── Main orchestrator ────────────────────────────────────────────────────
async def run_oauth_flow(http: aiohttp.ClientSession | None = None) -> OAuthSession:
    """Run the full OpenAI OAuth flow.

    This coroutine:
      1. Generates a PKCE code verifier and state
      2. Builds the authorization URL
      3. Starts the local callback server (trying ports 1455-1459)
      4. Opens the authorization URL in the default browser
      5. Waits for the callback (with a 5-minute timeout)
      6. Exchanges the auth code for tokens
      7. Exchanges the id_token for an API key
      8. Returns an OAuthSession wrapping all credentials

    Args:
        http: Optional aiohttp ClientSession. If not provided, a new one
              is created and closed internally.

    Returns:
        An OAuthSession with access_token, refresh_token, id_token,
        api_key, and expiry timestamps.

    Raises:
        OAuthPortConflictError: If none of the callback ports are available.
        OAuthTimeoutError: If the callback is not received within 5 minutes.
        OAuthAuthorizationError: If the authorization server returns an error.
    """
    # Generate PKCE + state
    code_verifier = generate_code_verifier()
    state = secrets.token_urlsafe(32)

    # Build auth URL (pass the state so the URL and callback server match)
    auth_url = build_auth_url(code_verifier, state=state)

    # Prepare the Future that the callback server will resolve
    loop = asyncio.get_running_loop()
    code_future: "Future[str | OAuthAuthorizationError]" = loop.create_future()

    # Start callback server (try ports 1455-1459)
    runner: aiohttp.web.AppRunner | None = None
    bound_port: int | None = None
    for port in range(_CALLBACK_PORT_START, _CALLBACK_PORT_END + 1):
        try:
            app = await _start_callback_server(state, code_future, "127.0.0.1", port)
            runner = aiohttp.web.AppRunner(app)
            await runner.setup()
            site = aiohttp.web.TCPSite(runner, "127.0.0.1", port)
            await site.start()
            bound_port = port
            break
        except OSError:
            # Port in use — clean up runner and try next port
            if runner is not None:
                await runner.cleanup()
                runner = None
            continue

    if runner is None or bound_port is None:
        raise OAuthPortConflictError(
            f"None of the callback ports {_CALLBACK_PORT_START}-{_CALLBACK_PORT_END} are available"
        )

    try:
        # Open browser
        webbrowser.open(auth_url)

        # Wait for callback with timeout
        try:
            result = await asyncio.wait_for(
                code_future, timeout=OAUTH_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            raise OAuthTimeoutError(
                f"No OAuth callback received within {OAUTH_TIMEOUT_SECONDS} seconds"
            )

        # Handle OAuth errors returned via the callback
        if isinstance(result, OAuthAuthorizationError):
            raise result

        code = result

    finally:
        # Shut down the callback server
        await runner.cleanup()

    # Exchange code for tokens
    close_http = http is None
    http = http or aiohttp.ClientSession()
    try:
        tokens = await _exchange_code_for_tokens(code, code_verifier, CLIENT_ID, http)
        id_token = tokens["id_token"]

        # Exchange id_token for API key
        api_key = await _exchange_id_token_for_api_key(id_token, CLIENT_ID, http)
        tokens["api_key"] = api_key
    finally:
        if close_http:
            await http.close()

    # Build and return OAuthSession
    session = OAuthSession.from_token_response(tokens, CLIENT_ID)
    return session
