from __future__ import annotations


def is_cloudflare_block(status: int, body: str) -> bool:
    if status != 403:
        return False
    lower = body.lower()
    return any(
        sig in lower
        for sig in ("cf-mitigated", "_cf_chl_opt", "cf-browser-verification", "cloudflare")
    )
