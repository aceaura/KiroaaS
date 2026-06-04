"""
Redirect chat off runtime.kiro.dev for accounts that have no profileArn.

runtime.kiro.dev requires a profileArn on EVERY generateAssistantResponse
request (it 400s with "profileArn is required for this request." otherwise).
Free-tier AWS Builder ID accounts never carry a profileArn — and they cannot
fetch one, because ListAvailableProfiles 403s for Builder ID
("AWS Builder ID is not supported for this operation"). So the
profile_arn_autofetch extension can't help them and every chat request dies.

The old q.{region}.amazonaws.com endpoint does NOT require a profileArn — it
accepts the same Builder ID token and request shape (verified: empty-profileArn
chat returns 429-throttle there, not 400). So for an account with no profileArn
we redirect the chat host (api_host) to q.amazonaws.com, mirroring how
control_plane_host.py already redirects q_host.

Scope: ONLY accounts with an empty profileArn are redirected. Paid accounts
(Kiro Pro / Q Developer Pro) carry a profileArn and keep using runtime.kiro.dev
untouched — they may have access to runtime-only models (e.g. glm-5) that the
old endpoint doesn't serve, so we must not move them.

Caveat: the q.amazonaws.com endpoint generally serves only the Claude-family
models. A redirected free account can chat with Claude but may still 4xx on
runtime-only third-party models (glm-5, minimax, etc). That is an AWS-side
limitation, not something this redirect can work around.

kiro/ stays zero-diff; this is installed from app_entry.py. Companion to
control_plane_host.py (q_host) and profile_arn_autofetch.py (profileArn backfill).
"""

import re

from loguru import logger


_installed = False
_original_api_host_descriptor = None

# runtime.{region}.kiro.dev  ->  q.{region}.amazonaws.com
_RUNTIME_HOST_RE = re.compile(r"^https://runtime\.(?P<region>[^/]+?)\.kiro\.dev/?$")
_Q_HOST_TEMPLATE = "https://q.{region}.amazonaws.com"


def _fallback_api_host(self):
    """Property getter: rewrite a runtime api_host to its q.amazonaws equivalent,
    but ONLY when this account has no profileArn.

    Reads the already-resolved self._api_host so all of upstream's region
    resolution still applies; only the host shape is swapped. Accounts that
    carry a profileArn (paid) and non-runtime hosts pass through unchanged.

    Evaluated lazily on every access: by the time chat reads api_host, account
    initialization has already run get_access_token (and thus profile_arn
    autofetch), so _profile_arn is in its final state. If a profileArn is later
    backfilled, this getter self-corrects to runtime on the next access.
    """
    host = self._api_host
    if not getattr(self, "_profile_arn", None):
        match = _RUNTIME_HOST_RE.match(host or "")
        if match:
            return _Q_HOST_TEMPLATE.format(region=match.group("region"))
    return host


def install_chat_host_fallback() -> None:
    """Redirect chat (api_host) to q.amazonaws for accounts lacking a profileArn."""
    global _installed, _original_api_host_descriptor
    if _installed:
        return

    from kiro.auth import KiroAuthManager

    # api_host is a read-only property on the class; swap the descriptor.
    _original_api_host_descriptor = KiroAuthManager.__dict__["api_host"]
    KiroAuthManager.api_host = property(_fallback_api_host)

    _installed = True
    logger.info(
        "Installed chat-host fallback (api_host -> q.amazonaws when profileArn absent)"
    )


def uninstall_chat_host_fallback() -> None:
    """Restore the original api_host property. Intended for unit tests."""
    global _installed, _original_api_host_descriptor

    if _original_api_host_descriptor is not None:
        from kiro.auth import KiroAuthManager

        KiroAuthManager.api_host = _original_api_host_descriptor

    _original_api_host_descriptor = None
    _installed = False
