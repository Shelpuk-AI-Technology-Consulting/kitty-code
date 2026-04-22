"""CLI command: kitty auth openai — run the OAuth flow for ChatGPT subscription."""

from __future__ import annotations

import uuid
from pathlib import Path

import questionary

from kitty.auth.oauth_session import OAuthSession
from kitty.auth.openai_oauth import run_oauth_flow
from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore
from kitty.tui.display import print_error, print_section, print_status, print_warning
from kitty.tui.prompts import check_tty

__all__ = ["run_auth_openai"]


async def run_auth_openai(profile_store: ProfileStore, cred_store: CredentialStore) -> Profile:
    """Run the OpenAI OAuth flow and create a profile.

    Steps:
    1. Print "Opening browser for ChatGPT login..."
    2. Call run_oauth_flow() -> OAuthSession
    3. Generate UUIDv4 auth_ref
    4. OAuthSession.create_session_file(session, auth_ref, config_dir)
       where config_dir is the parent directory of the profile store path
    5. cred_store.set(auth_ref, str(session_file_path))
    6. Prompt for model name (default: gpt-5.3-codex) using questionary
    7. Prompt for profile name (default: openai-sub) using questionary
    8. Ask if should be set as default
    9. Create Profile(provider='openai_subscription', model=model, auth_ref=auth_ref)
    10. Validate API key via validate_api_key(provider, api_key, {})
    11. Save profile via profile_store.save()
    12. Print success message
    13. Return the profile

    Args:
        profile_store: Profile store to save the new profile.
        cred_store: Credential store to store the session file path.

    Returns:
        The created and saved Profile.

    Raises:
        NonTTYError: If not running in an interactive terminal.
    """
    check_tty()
    print_section("kitty auth openai")

    # Step 1: Print opening browser message
    print_status("Opening browser for ChatGPT login...")

    # Step 2: Run OAuth flow
    try:
        session = await run_oauth_flow()
    except Exception as exc:
        print_error(f"OpenAI OAuth flow failed: {exc}")
        raise

    # Step 3: Generate UUIDv4 auth_ref
    auth_ref = str(uuid.uuid4())

    # Step 4: Create session file
    config_dir = profile_store._path.parent  # type: ignore[attr-defined]
    session = OAuthSession.create_session_file(session, auth_ref, config_dir)

    # Step 5: Store session file path in credential store
    session_file_path = Path(session._file_path)  # type: ignore[assignment]
    cred_store.set(auth_ref, str(session_file_path))

    # Step 6: Prompt for model name
    model = questionary.text("Model name: ", default="gpt-5.3-codex").ask()
    if not model or not model.strip():
        print_error("Model name cannot be empty")
        raise ValueError("Model name cannot be empty")
    model = model.strip()

    # Step 7: Prompt for profile name
    profile_name = questionary.text("Profile name: ", default="openai-sub").ask()
    if not profile_name or not profile_name.strip():
        profile_name = "openai-sub"
    profile_name = profile_name.strip().lower()

    # Step 8: Ask if should be set as default
    result = questionary.confirm("Set as default profile?", default=True).ask()
    is_default = result if result is not None else True

    # Step 9: Create profile
    profile = Profile(
        name=profile_name,
        provider="openai_subscription",  # type: ignore[arg-type]
        model=model,
        auth_ref=auth_ref,
        is_default=is_default,
    )

    # Step 10: Validate API key
    try:
        from kitty.providers.registry import get_provider
        from kitty.validation import validate_api_key

        provider = get_provider("openai_subscription")
        validation_result = await validate_api_key(provider, str(session_file_path), {})
        if not validation_result.valid:
            print_warning(f"API key validation failed: {validation_result.reason}")
        elif validation_result.warning:
            print_warning(validation_result.warning)
    except Exception as exc:
        print_warning(f"API key validation skipped: {exc}")

    # Step 11: Save profile
    profile_store.save(profile)

    # Step 12: Print success message
    print_status(f"Profile {profile.name!r} created successfully")

    # Step 13: Return the profile
    return profile
