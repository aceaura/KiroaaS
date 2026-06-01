"""
Runtime widening of AnthropicMessage.role to accept inline 'system'.

Newer Claude Code clients send role="system" inline in the messages array
(carrying <system-reminder>, currentDate, etc.). Upstream's Pydantic model
declares role as Literal["user", "assistant"], so the request 422s before
reaching converters_core.normalize_message_roles() — which already folds
unknown roles into 'user'.

Rather than patch upstream models_anthropic.py (keeping the backend
zero-diff against kiro-gateway-origin), we widen the Literal at runtime
and rebuild the affected models so the validator picks up the change.

TIMING: install_role_widening() MUST run before the FastAPI route modules
are imported. FastAPI compiles and caches each route's request-body
validator when the @router.post decorator runs (at import time). Widening
afterwards updates the model but NOT the route's already-cached validator,
so HTTP requests would still 422 even though direct model instantiation
succeeds. app_entry.py installs this before `from main import app`.
"""

from typing import Literal

from loguru import logger


_installed = False


def install_role_widening() -> None:
    global _installed
    if _installed:
        return

    from kiro.models_anthropic import (
        AnthropicCountTokensRequest,
        AnthropicMessage,
        AnthropicMessagesRequest,
    )

    # Pydantic FieldInfo.annotation is typed `type[Any] | None`; assigning a
    # Literal special form is exactly the runtime widening we want, but the
    # type checker can't model it — so ignore the spurious assignment error.
    AnthropicMessage.model_fields["role"].annotation = Literal[  # type: ignore[assignment]
        "user", "assistant", "system"
    ]
    AnthropicMessage.model_rebuild(force=True)
    AnthropicMessagesRequest.model_rebuild(force=True)
    AnthropicCountTokensRequest.model_rebuild(force=True)

    _installed = True
    logger.info("Widened AnthropicMessage.role to accept inline 'system'")


def uninstall_role_widening() -> None:
    """Restore the original Literal. Intended for unit tests."""
    global _installed

    from kiro.models_anthropic import (
        AnthropicCountTokensRequest,
        AnthropicMessage,
        AnthropicMessagesRequest,
    )

    AnthropicMessage.model_fields["role"].annotation = Literal["user", "assistant"]  # type: ignore[assignment]
    AnthropicMessage.model_rebuild(force=True)
    AnthropicMessagesRequest.model_rebuild(force=True)
    AnthropicCountTokensRequest.model_rebuild(force=True)

    _installed = False
