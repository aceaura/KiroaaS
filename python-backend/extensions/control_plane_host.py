"""
Redirect control-plane operations off runtime.kiro.dev to q.amazonaws.com.

This backend's config points BOTH api_host and q_host at
runtime.{region}.kiro.dev (KIRO_API_HOST_TEMPLATE / KIRO_Q_HOST_TEMPLATE).
But runtime.kiro.dev only serves the chat operation (GenerateAssistantResponse).
Every control-plane operation 400s there with UnknownOperationException:

    GetUsageLimits        runtime → 400   |  q.amazonaws → 200
    ListAvailableModels   runtime → 404   |  q.amazonaws → 200
    /mcp                  runtime → 400   |  q.amazonaws → 200

q_host is consumed only by ListAvailableModels (account_manager) and /mcp
(mcp_tools). Both want q.amazonaws.com, so we redirect the q_host property
to https://q.{region}.amazonaws.com — api_host (chat) is left untouched.

A second problem: account_manager._is_runtime_endpoint() gates model fetching.
When it sees a runtime api_host it skips ListAvailableModels entirely and serves
a static FALLBACK_MODELS list, so accounts never see their real models. Once
q_host is redirected to a working endpoint, fetching is valid for every account,
so we force _is_runtime_endpoint() to False. If a fetch still fails, the existing
except-clause falls back to FALLBACK_MODELS — same as today, no regression.

kiro/ stays zero-diff; this is installed from app_entry.py.
"""

import re

from loguru import logger


_installed = False
_original_q_host_descriptor = None
_original_is_runtime_endpoint = None

# runtime.{region}.kiro.dev  ->  q.{region}.amazonaws.com
_RUNTIME_HOST_RE = re.compile(r"^https://runtime\.(?P<region>[^/]+?)\.kiro\.dev/?$")
_Q_HOST_TEMPLATE = "https://q.{region}.amazonaws.com"


def to_q_amazonaws_host(host: str) -> str:
    """Rewrite a runtime.kiro.dev host to its q.amazonaws control-plane equivalent.

    A pure function, usable without installing the redirect below. Callers that
    issue control-plane operations need it either way: those operations 400/404
    on runtime.kiro.dev (see the module docstring), and the q_host property is
    only rewritten when install_control_plane_host_redirect() has run — which is
    not the case when the gateway is started via main.py.

    Non-runtime hosts (an already-redirected q.amazonaws host, or any custom
    endpoint) pass through unchanged, so applying this twice is a no-op.
    """
    match = _RUNTIME_HOST_RE.match(host or "")
    if match:
        return _Q_HOST_TEMPLATE.format(region=match.group("region"))
    return host


def _redirected_q_host(self):
    """Property getter: rewrite a runtime q_host to its q.amazonaws equivalent.

    Reads the already-resolved self._q_host so all of upstream's region
    resolution still applies; only the host shape is swapped. Non-runtime
    hosts (old q.amazonaws endpoint) pass through unchanged.
    """
    return to_q_amazonaws_host(self._q_host)


def install_control_plane_host_redirect() -> None:
    """Redirect q_host to q.amazonaws and re-enable dynamic model fetching."""
    global _installed, _original_q_host_descriptor, _original_is_runtime_endpoint
    if _installed:
        return

    from kiro.auth import KiroAuthManager
    import kiro.account_manager as account_manager

    # q_host is a read-only property on the class; swap the descriptor.
    _original_q_host_descriptor = KiroAuthManager.__dict__["q_host"]
    KiroAuthManager.q_host = property(_redirected_q_host)

    # Force dynamic model fetching on (q_host now serves ListAvailableModels).
    _original_is_runtime_endpoint = account_manager._is_runtime_endpoint
    account_manager._is_runtime_endpoint = lambda auth_manager: False

    _installed = True
    logger.info(
        "Installed control-plane host redirect (q_host -> q.amazonaws, dynamic models on)"
    )


def uninstall_control_plane_host_redirect() -> None:
    """Restore the original q_host property and _is_runtime_endpoint. For tests."""
    global _installed, _original_q_host_descriptor, _original_is_runtime_endpoint

    if _original_q_host_descriptor is not None:
        from kiro.auth import KiroAuthManager

        KiroAuthManager.q_host = _original_q_host_descriptor

    if _original_is_runtime_endpoint is not None:
        import kiro.account_manager as account_manager

        account_manager._is_runtime_endpoint = _original_is_runtime_endpoint

    _original_q_host_descriptor = None
    _original_is_runtime_endpoint = None
    _installed = False
