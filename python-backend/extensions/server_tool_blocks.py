"""
Runtime widening of AnthropicMessage.content to accept server-side tool blocks.

Anthropic's server-side tools (web_search, web_fetch, code_execution, ...) emit
two extra content-block types that clients such as Claude Code replay back into
the conversation history:

  - ``server_tool_use``         — the server-tool invocation
  - ``web_search_tool_result``  — the results payload for web_search

Upstream's ``ContentBlock`` union in kiro/models_anthropic.py only declares the
six client-side block types, so any request whose history carries a server-tool
block 422s at request-body validation — before converters_core ever sees it.
(The user observes this as a 422 with a long list of literal_error/missing
entries, one per failed union member.)

Rather than patch upstream models_anthropic.py (keeping the backend zero-diff
against kiro-gateway-origin), we define the missing block models here and widen
``AnthropicMessage.content`` at runtime, rebuilding the affected models so the
validator picks up the change.

The converter side is already safe: convert_anthropic_content_to_text only
reads ``type == "text"`` blocks, extract_tool_uses_from_anthropic_content only
reads ``tool_use``, and extract_tool_results_from_anthropic_content only reads
``tool_result``. Unknown blocks are silently ignored, so accepting them for
validation neither drops the text context nor leaks a server-tool call as a
real tool_call.

TIMING: install_server_tool_blocks() MUST run before the FastAPI route modules
are imported. FastAPI compiles and caches each route's request-body validator
when the @router.post decorator runs (at import time). Widening afterwards
updates the model but NOT the route's already-cached validator, so HTTP
requests would still 422 even though direct model instantiation succeeds.
app_entry.py installs this before `from main import app`. See role_widening.py
for the same constraint.
"""

from typing import Any, Dict, List, Literal, Union

from loguru import logger
from pydantic import BaseModel, Field


_installed = False


class ServerToolUseContentBlock(BaseModel):
    """
    Server-side tool use block (Anthropic server tools).

    Emitted by Anthropic server-side tools such as web_search, web_fetch and
    code_execution. Accepted for validation; the converter only extracts the
    accompanying text blocks, so the raw server-tool call is ignored.
    """

    type: Literal["server_tool_use"] = "server_tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class WebSearchToolResultContentBlock(BaseModel):
    """
    Web search tool result block (Anthropic server tool).

    Paired with a ServerToolUseContentBlock, this carries the search results
    returned by Anthropic's server-side web_search tool. The ``content`` field
    is a list of provider-specific result objects (or an error object), so it is
    kept loosely typed. Accepted for validation; not forwarded to Kiro.
    """

    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: str
    content: Union[List[Dict[str, Any]], Dict[str, Any], str, None] = None

    model_config = {"extra": "allow"}


class UnknownContentBlock(BaseModel):
    """
    Forward-compatible fallback for unrecognized content blocks.

    Anthropic periodically ships new server-side tool block types (e.g.
    code_execution_tool_result, mcp_tool_use). Rather than 422 on every new
    block, we accept any block that carries a ``type`` string and preserve its
    extra fields. The converter ignores blocks it does not explicitly handle,
    so unknown blocks are dropped safely instead of breaking the request.

    MUST be the last member of the widened union: as a catch-all with only a
    ``type: str`` field it would otherwise shadow the specific typed blocks
    under Pydantic union resolution.
    """

    type: str

    model_config = {"extra": "allow"}


def install_server_tool_blocks() -> None:
    global _installed
    if _installed:
        return

    from kiro.models_anthropic import (
        AnthropicCountTokensRequest,
        AnthropicMessage,
        AnthropicMessagesRequest,
        TextContentBlock,
        ThinkingContentBlock,
        ImageContentBlock,
        ToolUseContentBlock,
        ToolResultContentBlock,
        ToolReferenceContentBlock,
    )

    # Rebuild the content-block union: all upstream members first, then the
    # new server-tool blocks, then the catch-all fallback last (so it does not
    # greedily shadow the typed blocks).
    widened_block = Union[
        TextContentBlock,
        ThinkingContentBlock,
        ImageContentBlock,
        ToolUseContentBlock,
        ToolResultContentBlock,
        ToolReferenceContentBlock,
        ServerToolUseContentBlock,
        WebSearchToolResultContentBlock,
        UnknownContentBlock,
    ]

    # Pydantic FieldInfo.annotation is typed `type[Any] | None`; assigning a
    # Union special form is exactly the runtime widening we want, but the type
    # checker can't model it — so ignore the spurious assignment error.
    AnthropicMessage.model_fields["content"].annotation = Union[  # type: ignore[assignment]
        str, List[widened_block]
    ]
    AnthropicMessage.model_rebuild(force=True)
    AnthropicMessagesRequest.model_rebuild(force=True)
    AnthropicCountTokensRequest.model_rebuild(force=True)

    _installed = True
    logger.info(
        "Widened AnthropicMessage.content to accept server-side tool blocks "
        "(server_tool_use, web_search_tool_result, + unknown fallback)"
    )


def uninstall_server_tool_blocks() -> None:
    """Restore the original content union. Intended for unit tests."""
    global _installed

    from kiro.models_anthropic import (
        AnthropicCountTokensRequest,
        AnthropicMessage,
        AnthropicMessagesRequest,
        ContentBlock,
    )

    AnthropicMessage.model_fields["content"].annotation = Union[  # type: ignore[assignment]
        str, List[ContentBlock]
    ]
    AnthropicMessage.model_rebuild(force=True)
    AnthropicMessagesRequest.model_rebuild(force=True)
    AnthropicCountTokensRequest.model_rebuild(force=True)

    _installed = False
