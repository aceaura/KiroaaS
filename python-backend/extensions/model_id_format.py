"""
Serve dash-format Claude model IDs to Claude-family clients on /v1/models.

The gateway normalizes every model ID to a dotted form (claude-sonnet-4.6,
claude-3.7-sonnet) and the /v1/models endpoint returns exactly that. But some
clients — notably Claude Code and Claude Desktop — only recognize the dashed
form (claude-sonnet-4-6, claude-3-7-sonnet) when they read a model list, so the
dotted IDs never show up as selectable.

This extension rewrites the /v1/models response on the way out, but ONLY for
requests that look like they came from a Claude-family client (any request
header value contains "claude", e.g. User-Agent: claude-cli/...). Other clients
keep seeing the dotted form, so nothing they relied on changes.

Why a middleware and not a route wrapper: FastAPI compiles each route's handler
into the app when the @router.get decorator runs, which happens during
`from main import app`. By the time app_entry installs runtime extensions the
/v1/models route is already wired to the original get_models, so swapping the
module attribute has no effect (same import-order trap that makes role_widening
install before import). A response middleware is independent of that ordering
and post-processes the already-serialized JSON.

Safety: rewriting the listed IDs to the dashed form does not break model
resolution. When a client later sends a dashed ID back, kiro's
normalize_model_name() converts dash→dot before lookup, so both forms resolve to
the same model. kiro/ stays zero-diff; this is installed from app_entry.py.
"""

import json

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_MODELS_PATH = "/v1/models"


def _is_claude_client(request) -> bool:
    """True if any request header (except Authorization) contains "claude".

    Authorization is skipped so we never inspect the API key and never match on
    a secret that happens to contain the substring. The check is deliberately
    permissive: a false positive only swaps dotted IDs for dashed ones, which
    still resolve correctly, so erring toward "looks like Claude" is harmless.
    """
    for name, value in request.headers.items():
        if name.lower() == "authorization":
            continue
        if value and "claude" in value.lower():
            return True
    return False


def _dashify_claude_id(model_id: str) -> str:
    """Convert a dotted Claude model ID to its dashed form.

    claude-sonnet-4.6 -> claude-sonnet-4-6
    claude-3.7-sonnet -> claude-3-7-sonnet
    claude-opus-4     -> claude-opus-4  (no dot, unchanged)

    Non-Claude IDs (e.g. "auto") are returned untouched. Claude IDs only carry
    dots inside their version, so a blanket dot→dash replacement is safe.
    """
    if not model_id or "claude" not in model_id.lower():
        return model_id
    return model_id.replace(".", "-")


def _rewrite_models_body(body: bytes) -> bytes:
    """Return a /v1/models JSON body with Claude IDs dashed, or the input on no-op.

    Any parse problem returns the original bytes unchanged — the response must
    still be delivered even if its shape is unexpected.
    """
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return body

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return body

    changed = False
    for item in data:
        if not isinstance(item, dict):
            continue
        original = item.get("id")
        dashed = _dashify_claude_id(original)
        if dashed != original:
            item["id"] = dashed
            changed = True

    if not changed:
        return body
    return json.dumps(payload).encode("utf-8")


async def _read_response_body(response) -> bytes:
    body = b""
    async for chunk in response.body_iterator:
        body += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
    return body


def _rebuild_response(response, body: bytes) -> Response:
    """Rebuild a Response from collected bytes, letting Starlette recompute length.

    content-length/content-type are dropped from the copied headers so the new
    Response derives a correct length and we re-assert JSON; all other headers
    (CORS, etc.) are preserved.
    """
    headers = dict(response.headers)
    headers.pop("content-length", None)
    headers.pop("content-type", None)
    return Response(
        content=body,
        status_code=response.status_code,
        headers=headers,
        media_type="application/json",
    )


class ClaudeModelIdFormatMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        if (
            request.method == "GET"
            and request.url.path == _MODELS_PATH
            and response.status_code == 200
            and _is_claude_client(request)
        ):
            body = await _read_response_body(response)
            return _rebuild_response(response, _rewrite_models_body(body))

        return response


_installed_apps = []


def install_model_id_format(app) -> None:
    """Add the Claude model-ID dash-format middleware to the given app.

    Must run before the app starts serving (app_entry installs it before
    uvicorn.run), while the middleware stack is still mutable.
    """
    if app in _installed_apps:
        return
    app.add_middleware(ClaudeModelIdFormatMiddleware)
    _installed_apps.append(app)
    logger.info("Installed Claude model-ID dash-format middleware (/v1/models)")


def uninstall_model_id_format(app) -> None:
    """Remove the middleware from the app and force a stack rebuild. For tests."""
    if app not in _installed_apps:
        return
    app.user_middleware = [
        m for m in app.user_middleware if m.cls is not ClaudeModelIdFormatMiddleware
    ]
    app.middleware_stack = None
    _installed_apps.remove(app)
