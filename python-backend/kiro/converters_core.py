# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Core converters for transforming API formats to Kiro format.

This module contains shared logic used by both OpenAI and Anthropic converters:
- Text content extraction from various formats
- Message merging and processing
- Kiro payload building
- Tool processing and sanitization

The core layer provides a unified interface that API-specific adapters use
to convert their formats to Kiro API format.
"""

import base64
import ipaddress
import json
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx
from loguru import logger

from kiro.config import (
    TOOL_DESCRIPTION_MAX_LENGTH,
    FAKE_REASONING_ENABLED,
    FAKE_REASONING_MAX_TOKENS,
    FAKE_REASONING_BUDGET_CAP,
    KIRO_MAX_PAYLOAD_BYTES,
    AUTO_TRIM_PAYLOAD,
    NATIVE_EFFORT_ENABLED,
    NATIVE_EFFORT_DEFAULT,
    NATIVE_EFFORT_NONE_ON_DISABLED,
    NATIVE_EFFORT_SUPPRESS_TAGS,
    FETCH_IMAGE_URLS,
    FETCH_IMAGE_URL_TIMEOUT,
    FETCH_IMAGE_URL_MAX_BYTES,
    FETCH_IMAGE_URL_MAX_COUNT,
    FETCH_IMAGE_URL_ALLOWED_HOSTS,
    FETCH_IMAGE_URL_ALLOWED_IP_RANGES,
)
from kiro.effort_schema import (
    NATIVE_EFFORT_FIELD,
    lookup_effort_schema,
    resolve_effort_decision,
)
from kiro.request_audit import RequestAudit
from kiro.payload_guards import check_payload_size, trim_payload_to_limit


# ==================================================================================================
# SSRF-Safe Image URL Fetching
# ==================================================================================================

def _parse_allowed_ip_networks() -> List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
    """
    Parses the operator-configured CIDR allowlist once, at import time.

    Invalid entries are logged and skipped rather than raising, so one typo in
    the environment cannot prevent the gateway from starting. Validation lives
    here because config.py has no logger.

    Returns:
        List of parsed networks; empty when nothing is configured
    """
    networks: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []
    for cidr in FETCH_IMAGE_URL_ALLOWED_IP_RANGES:
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError as e:
            logger.warning(f"Ignoring invalid FETCH_IMAGE_URL_ALLOWED_IP_RANGES entry {cidr!r}: {e}")
    if networks:
        logger.info(
            "Image URL fetching permits extra IP ranges: "
            + ", ".join(str(n) for n in networks)
        )
    return networks


_ALLOWED_IP_NETWORKS = _parse_allowed_ip_networks()


def _is_blocked_ip(ip_str: str) -> bool:
    """
    Checks whether an IP address must never be reached by a server-side fetch.

    Blocks private, loopback, link-local (including the 169.254.169.254 cloud
    metadata address), multicast, reserved, and unspecified ranges for both
    IPv4 and IPv6.

    An address inside FETCH_IMAGE_URL_ALLOWED_IP_RANGES is permitted even when
    it would otherwise be blocked: that is an explicit operator override, used
    for example to allow a fake-ip proxy's 198.18.0.0/15 placeholders.

    Args:
        ip_str: IP address as a string (e.g. "127.0.0.1")

    Returns:
        True if the address is in a blocked range
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Unparseable address cannot be proven safe: fail closed.
        return True

    for network in _ALLOWED_IP_NETWORKS:
        if ip.version == network.version and ip in network:
            return False

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_and_validate_host(hostname: str) -> None:
    """
    Resolves a hostname and raises if any resolved address is blocked.

    Validates at the resolved-IP level rather than by string-matching the
    hostname, so a name like "internal.example.com" that happens to resolve
    to a private address is still caught.

    A hostname listed in FETCH_IMAGE_URL_ALLOWED_HOSTS skips resolution and the
    IP check entirely. That is an explicit operator trust decision, needed for
    fake-ip proxy setups where every name resolves to a placeholder address.
    Only the IP check is skipped: the caller still enforces the scheme
    allowlist, the redirect ban and the size limits.

    Args:
        hostname: Hostname to resolve and validate

    Raises:
        ValueError: If DNS resolution fails, or if any resolved address
            falls in a blocked range (see _is_blocked_ip)
    """
    # "host." and "host" are the same name in DNS.
    normalized = hostname.strip().lower().rstrip(".")
    if normalized in FETCH_IMAGE_URL_ALLOWED_HOSTS:
        logger.debug(f"Host {normalized} is allowlisted; skipping resolved-IP check")
        return

    try:
        addrs = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for {hostname}: {e}")

    for _family, _type, _proto, _canonname, sockaddr in addrs:
        ip_str = sockaddr[0]
        if _is_blocked_ip(ip_str):
            raise ValueError(f"Resolved address {ip_str} for {hostname} is not allowed")


async def fetch_image_as_base64(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetches a client-supplied image URL server-side and returns base64 data.

    Kiro API only accepts inline base64 images, so a URL-based image must be
    fetched and re-encoded before it reaches the payload. Because the URL
    comes from an authenticated but not necessarily trusted caller, SSRF
    protection is unconditional: scheme allowlist, DNS-resolved IP blocking
    against private/loopback/link-local/reserved ranges, and no redirect
    following (a redirect to a blocked address must not be a bypass).

    Never raises. A bad image URL should not fail the whole request, so every
    rejection path logs a warning and returns None instead.

    Args:
        url: Client-supplied image URL (http:// or https://)

    Returns:
        {"media_type": "image/jpeg", "data": "<base64>"} on success, or None
        if the URL is rejected, unreachable, too large, or fetch fails
    """
    try:
        parsed = urlparse(url)
        scheme, hostname = parsed.scheme, parsed.hostname
    except ValueError as e:
        # urlparse rejects some malformed input outright, e.g. "http://[bad"
        logger.warning(f"Rejected malformed image URL: {e}")
        return None

    if scheme not in ("http", "https") or not hostname:
        logger.warning(f"Rejected image URL scheme/host: {url[:80]}")
        return None

    try:
        _resolve_and_validate_host(hostname)
    except ValueError as e:
        logger.warning(f"Rejected image URL: {e}")
        return None

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(FETCH_IMAGE_URL_TIMEOUT),
            follow_redirects=False,
        ) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    logger.warning(f"Image URL fetch got HTTP {response.status_code}: {url[:80]}")
                    return None

                # A malformed Content-Length must not raise: the streaming
                # counter below enforces the real limit regardless.
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_bytes = int(content_length)
                    except (TypeError, ValueError):
                        declared_bytes = None
                    if declared_bytes is not None and declared_bytes > FETCH_IMAGE_URL_MAX_BYTES:
                        logger.warning(f"Image URL exceeds size limit: {url[:80]}")
                        return None

                media_type = response.headers.get("content-type", "image/jpeg").split(";")[0].strip()

                chunks: List[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > FETCH_IMAGE_URL_MAX_BYTES:
                        logger.warning(f"Image URL exceeded size limit while streaming: {url[:80]}")
                        return None
                    chunks.append(chunk)

                data = base64.b64encode(b"".join(chunks)).decode("ascii")
                return {"media_type": media_type, "data": data}
    except (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError) as e:
        logger.warning(f"Image URL fetch failed: {e}")
        return None


@dataclass
class ImageFetchBudget:
    """
    Bounds server-side image URL fetching for the scope it is shared across.

    Two protections, both needed because the URLs come from a caller that holds
    the proxy key but is not otherwise trusted:

    - Deduplication: a URL is fetched at most once. Conversation history is
      re-converted on every turn, so without this the same image is re-fetched
      for the whole life of a conversation.
    - Count cap: at most max_count fetches. One request listing N URLs would
      otherwise cost N sequential outbound requests, letting a single caller
      occupy a worker for minutes.

    A failed fetch is cached as None, so a broken URL is attempted once and
    consumes exactly one unit of budget instead of being retried per mention.

    Attributes:
        max_count: Maximum fetches allowed; defaults to FETCH_IMAGE_URL_MAX_COUNT
        cache: URL to fetch result (None for a URL that failed)
        fetch_count: Fetches attempted so far
    """
    max_count: int = field(default_factory=lambda: FETCH_IMAGE_URL_MAX_COUNT)
    cache: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=dict)
    fetch_count: int = 0
    _cap_reported: bool = False

    async def fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetches an image URL unless it is cached or the budget is exhausted.

        Args:
            url: Client-supplied image URL

        Returns:
            Image dict as returned by fetch_image_as_base64, or None if the
            fetch failed, was previously known to fail, or was refused
        """
        if url in self.cache:
            return self.cache[url]

        if self.fetch_count >= self.max_count:
            if not self._cap_reported:
                self._cap_reported = True
                logger.warning(
                    f"Image URL fetch limit reached ({self.max_count}); "
                    f"skipping remaining URL images in this request"
                )
            return None

        self.fetch_count += 1
        result = await fetch_image_as_base64(url)
        self.cache[url] = result
        return result


# ==================================================================================================
# Data Classes for Unified Message Format
# ==================================================================================================

@dataclass
class ThinkingConfig:
    """
    Unified thinking configuration for reasoning control.

    Attributes:
        enabled: Whether reasoning may be requested
        budget_tokens: Explicit numeric token budget supplied by the client
        effort: Qualitative effort tier supplied by the client
    """
    enabled: bool = True
    budget_tokens: Optional[int] = None
    effort: Optional[str] = None


@dataclass
class UnifiedMessage:
    """
    Unified message format used internally by converters.
    
    This format is API-agnostic and can be created from both OpenAI and Anthropic formats.
    Serves as the canonical representation for all message data before conversion to Kiro API.
    
    Attributes:
        role: Message role (user, assistant, system)
        content: Text content or list of content blocks
        tool_calls: List of tool calls (for assistant messages)
        tool_results: List of tool results (for user messages with tool responses)
        images: List of images in unified format (for multimodal user messages)
                Format: [{"media_type": "image/jpeg", "data": "base64..."}]
    """
    role: str
    content: Any = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None


@dataclass
class UnifiedTool:
    """
    Unified tool format used internally by converters.
    
    Attributes:
        name: Tool name
        description: Tool description
        input_schema: JSON Schema for tool parameters
    """
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


@dataclass
class KiroPayloadResult:
    """
    Result of building Kiro payload.
    
    Attributes:
        payload: The complete Kiro API payload
        tool_documentation: Documentation for tools with long descriptions (to add to system prompt)
    """
    payload: Dict[str, Any]
    tool_documentation: str = ""


@dataclass(frozen=True)
class ToolChoicePolicy:
    """Normalized tool-selection policy shared by both client protocols."""

    mode: str = "auto"
    tool_name: Optional[str] = None

    @property
    def is_strict(self) -> bool:
        """Return whether the response must be buffered and validated."""
        return self.mode != "auto"

    def filter_tools(self, tools: Optional[List[UnifiedTool]]) -> Optional[List[UnifiedTool]]:
        """Return the tools that may be sent upstream for this policy."""
        if self.mode == "none":
            return None
        if self.mode == "named":
            return [tool for tool in tools or [] if tool.name == self.tool_name]
        return tools

    def build_directive(self) -> str:
        """Build the prompt directive used because Kiro has no tool-choice field."""
        if self.mode == "none":
            return (
                "\n\n[Tool Policy] For THIS response you must NOT call any tool. "
                "Reply with plain content only."
            )
        if self.mode == "required":
            return (
                "\n\n[Tool Policy] For THIS response you MUST call at least one tool. "
                "Do not reply with text only."
            )
        if self.mode == "named" and self.tool_name:
            return (
                f"\n\n[Tool Policy] For THIS response you MUST call the tool named "
                f"'{self.tool_name}'. Do not call any other tool and do not reply with text only."
            )
        return ""


def parse_tool_choice_policy(
    tool_choice: Any,
    tools: Optional[List[UnifiedTool]],
    protocol: str,
) -> ToolChoicePolicy:
    """Validate and normalize an OpenAI or Anthropic tool-choice value.

    Args:
        tool_choice: Raw tool-choice value from the client request.
        tools: Tools using their original client-visible names.
        protocol: Either ``openai`` or ``anthropic``.

    Raises:
        ValueError: If the value is malformed or cannot be satisfied.
    """
    if protocol not in ("openai", "anthropic"):
        raise ValueError(f"Unsupported tool_choice protocol: {protocol}")

    raw = tool_choice
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(exclude_none=True)
    elif hasattr(raw, "dict") and not isinstance(raw, dict):
        raw = raw.dict(exclude_none=True)

    mode = "auto"
    tool_name: Optional[str] = None

    if raw is None:
        pass
    elif protocol == "openai" and isinstance(raw, str):
        if raw not in ("auto", "none", "required"):
            raise ValueError(f"Invalid OpenAI tool_choice value: {raw!r}")
        mode = raw
    elif protocol == "openai" and isinstance(raw, dict):
        function = raw.get("function")
        if raw.get("type") != "function" or not isinstance(function, dict):
            raise ValueError("OpenAI named tool_choice must contain type='function' and a function object")
        tool_name = function.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("OpenAI named tool_choice requires a non-empty function.name")
        mode = "named"
    elif protocol == "anthropic" and isinstance(raw, dict):
        choice_type = raw.get("type")
        if choice_type not in ("auto", "any", "none", "tool"):
            raise ValueError(f"Invalid Anthropic tool_choice type: {choice_type!r}")
        if choice_type == "tool":
            tool_name = raw.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("Anthropic tool_choice type='tool' requires a non-empty name")
            mode = "named"
        else:
            if "name" in raw:
                raise ValueError(f"Anthropic tool_choice type={choice_type!r} must not include name")
            mode = "required" if choice_type == "any" else choice_type
    else:
        raise ValueError(f"Invalid {protocol} tool_choice structure")

    available_names = {tool.name for tool in tools or []}
    if mode == "required" and not available_names:
        raise ValueError("tool_choice requires at least one tool, but no tools were provided")
    if mode == "named" and tool_name not in available_names:
        raise ValueError(f"tool_choice references unknown tool {tool_name!r}")

    return ToolChoicePolicy(mode=mode, tool_name=tool_name)


# ==================================================================================================
# Text Content Extraction
# ==================================================================================================

def extract_text_content(content: Any) -> str:
    """
    Extracts text content from various formats.
    
    Supports multiple content formats used by different APIs:
    - String: "Hello, world!"
    - List of content blocks: [{"type": "text", "text": "Hello"}]
    - None: empty message
    
    Args:
        content: Content in any supported format
    
    Returns:
        Extracted text or empty string
    
    Example:
        >>> extract_text_content("Hello")
        'Hello'
        >>> extract_text_content([{"type": "text", "text": "World"}])
        'World'
        >>> extract_text_content(None)
        ''
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Skip image and tool_reference blocks - they're handled separately
                if item.get("type") in ("image", "image_url", "tool_reference"):
                    continue
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(item["text"])
            elif hasattr(item, "text"):
                # Handle Pydantic models like TextContentBlock
                text_parts.append(getattr(item, "text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content)


async def extract_images_from_content(
    content: Any,
    budget: Optional["ImageFetchBudget"] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts images from message content in unified format.
    
    Supports multiple image formats used by different APIs:
    
    OpenAI format (image_url with data URL):
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
    
    Anthropic format (image with source):
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/..."}}
    
    URL-based images (either format) are fetched server-side and inlined as
    base64 when FETCH_IMAGE_URLS is enabled (the default) — see
    fetch_image_as_base64. When disabled, they are skipped with a warning,
    same as before this behavior existed.
    
    This is a coroutine because fetching a URL-based image requires an
    outbound HTTP request.
    
    Args:
        content: Content in any supported format (usually a list of content blocks)
        budget: Shared ImageFetchBudget giving request-wide deduplication and a
            request-wide fetch cap. When omitted, a fresh budget scoped to this
            single call is used, so the call is still bounded but shares no
            state with the rest of the request.
    
    Returns:
        List of images in unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
        Empty list if no images found or content is not a list.
    
    Example:
        >>> import asyncio
        >>> asyncio.run(extract_images_from_content([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}}]))
        [{'media_type': 'image/png', 'data': 'abc123'}]
    """
    images: List[Dict[str, Any]] = []
    
    if not isinstance(content, list):
        return images
    
    if budget is None:
        budget = ImageFetchBudget()
    
    for item in content:
        # Handle both dict and Pydantic model objects
        if isinstance(item, dict):
            item_type = item.get("type")
        elif hasattr(item, "type"):
            item_type = item.type
        else:
            continue
        
        # OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        if item_type == "image_url":
            if isinstance(item, dict):
                image_url_obj = item.get("image_url", {})
            else:
                image_url_obj = getattr(item, "image_url", {})
            
            if isinstance(image_url_obj, dict):
                url = image_url_obj.get("url", "")
            elif hasattr(image_url_obj, "url"):
                url = image_url_obj.url
            else:
                url = ""
            
            if url.startswith("data:"):
                # Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
                try:
                    header, data = url.split(",", 1)
                    # Extract media type from "data:image/jpeg;base64"
                    media_part = header.split(";")[0]  # "data:image/jpeg"
                    media_type = media_part.replace("data:", "")  # "image/jpeg"
                    
                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse image data URL: {e}")
            elif url.startswith("http"):
                if FETCH_IMAGE_URLS:
                    fetched = await budget.fetch(url)
                    if fetched:
                        images.append(fetched)
                else:
                    logger.warning(f"URL-based images are not supported by Kiro API, skipping: {url[:80]}...")
        
        # Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        elif item_type == "image":
            source = item.get("source", {}) if isinstance(item, dict) else getattr(item, "source", None)
            
            if source is None:
                continue
            
            if isinstance(source, dict):
                source_type = source.get("type")
                
                if source_type == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    
                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                elif source_type == "url":
                    # URL-based images in Anthropic format
                    url = source.get("url", "")
                    if FETCH_IMAGE_URLS:
                        fetched = await budget.fetch(url)
                        if fetched:
                            images.append(fetched)
                    else:
                        logger.warning(f"URL-based images are not supported by Kiro API, skipping: {url[:80]}...")
            
            # Handle Pydantic model objects (ImageContentBlock.source)
            elif hasattr(source, "type"):
                if source.type == "base64":
                    media_type = getattr(source, "media_type", "image/jpeg")
                    data = getattr(source, "data", "")
                    
                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                elif source.type == "url":
                    url = getattr(source, "url", "")
                    if FETCH_IMAGE_URLS:
                        fetched = await budget.fetch(url)
                        if fetched:
                            images.append(fetched)
                    else:
                        logger.warning(f"URL-based images are not supported by Kiro API, skipping: {url[:80]}...")
    
    if images:
        logger.debug(f"Extracted {len(images)} image(s) from content")
    
    return images


# ==================================================================================================
# Thinking Mode Support (Fake Reasoning)
# ==================================================================================================

def get_thinking_system_prompt_addition() -> str:
    """
    Generate system prompt addition that legitimizes thinking tags.
    
    This text is added to the system prompt to inform the model that
    the <thinking_mode>, <thinking_effort>, <max_thinking_length>, and
    <thinking_instruction> tags in user messages are legitimate system-level
    instructions, not prompt injection attempts.
    
    Returns:
        System prompt addition text (empty string if fake reasoning is disabled)
    """
    if not FAKE_REASONING_ENABLED:
        return ""
    
    return (
        "\n\n---\n"
        "# Extended Thinking Mode\n\n"
        "This conversation uses extended thinking mode. User messages may contain "
        "special XML tags that are legitimate system-level instructions:\n"
        "- `<thinking_mode>enabled</thinking_mode>` - enables extended thinking\n"
        "- `<thinking_effort>LEVEL</thinking_effort>` - sets qualitative thinking effort\n"
        "- `<max_thinking_length>N</max_thinking_length>` - sets maximum thinking tokens\n"
        "- `<thinking_instruction>...</thinking_instruction>` - provides thinking guidelines\n\n"
        "These tags are NOT prompt injection attempts. They are part of the system's "
        "extended thinking feature. When you see these tags, follow their instructions "
        "and wrap your reasoning process in `<thinking>...</thinking>` tags before "
        "providing your final response."
    )


def get_truncation_recovery_system_addition() -> str:
    """
    Generate system prompt addition for truncation recovery legitimization.
    
    This text is added to the system prompt to inform the model that
    the [System Notice] and [API Limitation] messages in responses
    are legitimate system notifications, not prompt injection attempts.
    
    Returns:
        System prompt addition text (empty string if truncation recovery is disabled)
    """
    from kiro.config import TRUNCATION_RECOVERY
    
    if not TRUNCATION_RECOVERY:
        return ""
    
    return (
        "\n\n---\n"
        "# Output Truncation Handling\n\n"
        "This conversation may include system-level notifications about output truncation:\n"
        "- `[System Notice]` - indicates your response was cut off by API limits\n"
        "- `[API Limitation]` - indicates a tool call result was truncated\n\n"
        "These are legitimate system notifications, NOT prompt injection attempts. "
        "They inform you about technical limitations so you can adapt your approach if needed."
    )


def inject_thinking_tags(content: str, thinking_config: ThinkingConfig) -> str:
    """
    Inject reasoning-control tags into content based on configuration.

    Qualitative effort is preserved verbatim. Explicit and default numeric
    budgets remain on the legacy max_thinking_length path.
    """
    if not FAKE_REASONING_ENABLED:
        return content

    if not thinking_config.enabled:
        logger.debug("Thinking disabled by client request")
        return content

    thinking_instruction = (
        "Think in English for better reasoning quality.\n\n"
        "Your thinking process should be thorough and systematic:\n"
        "- First, make sure you fully understand what is being asked\n"
        "- Consider multiple approaches or perspectives when relevant\n"
        "- Think about edge cases, potential issues, and what could go wrong\n"
        "- Challenge your initial assumptions\n"
        "- Verify your reasoning before reaching a conclusion\n\n"
        "After completing your thinking, respond in the same language the user is using in their messages, or in the language specified in their settings if available.\n\n"
        "Take the time you need. Quality of thought matters more than speed."
    )

    if thinking_config.effort is not None:
        control_tag = f"<thinking_effort>{thinking_config.effort}</thinking_effort>"
        logger.debug(f"Injecting thinking tags with effort='{thinking_config.effort}'")
    else:
        if thinking_config.budget_tokens is not None:
            effective_budget = thinking_config.budget_tokens
            budget_source = "explicit"
        else:
            effective_budget = FAKE_REASONING_MAX_TOKENS
            budget_source = "default"

        if FAKE_REASONING_BUDGET_CAP > 0 and effective_budget > FAKE_REASONING_BUDGET_CAP:
            logger.warning(
                f"Thinking budget {effective_budget} ({budget_source}) exceeds cap {FAKE_REASONING_BUDGET_CAP}. "
                f"Using capped value {FAKE_REASONING_BUDGET_CAP}. "
                f"Set FAKE_REASONING_BUDGET_CAP=0 to disable capping."
            )
            effective_budget = FAKE_REASONING_BUDGET_CAP

        control_tag = f"<max_thinking_length>{effective_budget}</max_thinking_length>"
        logger.debug(f"Injecting thinking tags with {budget_source} budget={effective_budget}")

    thinking_prefix = (
        "<thinking_mode>enabled</thinking_mode>\n"
        f"{control_tag}\n"
        f"<thinking_instruction>{thinking_instruction}</thinking_instruction>\n\n"
    )

    return thinking_prefix + content


# ==================================================================================================
# JSON Schema Sanitization
# ==================================================================================================

def sanitize_json_schema(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitizes JSON Schema from fields that Kiro API doesn't accept.
    
    Kiro API returns 400 "Improperly formed request" error if:
    - required is an empty array []
    - additionalProperties is present in schema
    
    This function recursively processes the schema and removes problematic fields.
    
    Args:
        schema: JSON Schema to sanitize
    
    Returns:
        Sanitized copy of schema
    """
    if not schema:
        return {}
    
    result = {}
    
    for key, value in schema.items():
        # Skip empty required arrays
        if key == "required" and isinstance(value, list) and len(value) == 0:
            continue
        
        # Skip additionalProperties - Kiro API doesn't support it
        if key == "additionalProperties":
            continue
        
        # Recursively process nested objects
        if key == "properties" and isinstance(value, dict):
            result[key] = {
                prop_name: sanitize_json_schema(prop_value) if isinstance(prop_value, dict) else prop_value
                for prop_name, prop_value in value.items()
            }
        elif isinstance(value, dict):
            result[key] = sanitize_json_schema(value)
        elif isinstance(value, list):
            # Process lists (e.g., anyOf, oneOf)
            result[key] = [
                sanitize_json_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result


# ==================================================================================================
# Tool Processing
# ==================================================================================================

def process_tools_with_long_descriptions(
    tools: Optional[List[UnifiedTool]]
) -> Tuple[Optional[List[UnifiedTool]], str]:
    """
    Processes tools with long descriptions.
    
    Kiro API has a limit on description length in toolSpecification.
    If description exceeds the limit, full description is moved to system prompt,
    and a reference to documentation remains in the tool.
    
    Args:
        tools: List of tools in unified format
    
    Returns:
        Tuple of:
        - List of tools with processed descriptions (or None if tools is empty)
        - String with documentation to add to system prompt (empty if all descriptions are short)
    """
    if not tools:
        return None, ""
    
    # If limit is disabled (0), return tools unchanged
    if TOOL_DESCRIPTION_MAX_LENGTH <= 0:
        return tools, ""
    
    tool_documentation_parts = []
    processed_tools = []
    
    for tool in tools:
        description = tool.description or ""
        
        if len(description) <= TOOL_DESCRIPTION_MAX_LENGTH:
            # Description is short - leave as is
            processed_tools.append(tool)
        else:
            # Description is too long - move to system prompt
            logger.debug(
                f"Tool '{tool.name}' has long description ({len(description)} chars > {TOOL_DESCRIPTION_MAX_LENGTH}), "
                f"moving to system prompt"
            )
            
            # Create documentation for system prompt
            tool_documentation_parts.append(f"## Tool: {tool.name}\n\n{description}")
            
            # Create copy of tool with reference description
            reference_description = f"[Full documentation in system prompt under '## Tool: {tool.name}']"
            
            processed_tool = UnifiedTool(
                name=tool.name,
                description=reference_description,
                input_schema=tool.input_schema
            )
            processed_tools.append(processed_tool)
    
    # Form final documentation
    tool_documentation = ""
    if tool_documentation_parts:
        tool_documentation = (
            "\n\n---\n"
            "# Tool Documentation\n"
            "The following tools have detailed documentation that couldn't fit in the tool definition.\n\n"
            + "\n\n---\n\n".join(tool_documentation_parts)
        )
    
    return processed_tools if processed_tools else None, tool_documentation


def validate_tool_names(tools: Optional[List[UnifiedTool]]) -> None:
    """
    Validates tool names against Kiro API 64-character limit.
    
    Logs WARNING for each problematic tool and raises ValueError
    with complete list of violations.
    
    Args:
        tools: List of tools to validate
    
    Raises:
        ValueError: If any tool name exceeds 64 characters
    
    Example:
        >>> validate_tool_names([UnifiedTool(name="short_name", description="test")])
        # No error
        >>> validate_tool_names([UnifiedTool(name="a" * 70, description="test")])
        # Raises ValueError with detailed message
    """
    if not tools:
        return
    
    problematic_tools = []
    for tool in tools:
        if len(tool.name) > 64:
            problematic_tools.append((tool.name, len(tool.name)))
    
    if problematic_tools:
        # Build detailed error message for client (no logging here - routes will log)
        tool_list = "\n".join([
            f"  - '{name}' ({length} characters)"
            for name, length in problematic_tools
        ])
        
        raise ValueError(
            f"Tool name(s) exceed Kiro API limit of 64 characters:\n"
            f"{tool_list}\n\n"
            f"Solution: Use shorter tool names (max 64 characters).\n"
            f"Example: 'get_user_data' instead of 'get_authenticated_user_profile_data_with_extended_information_about_it'"
        )


def convert_tools_to_kiro_format(tools: Optional[List[UnifiedTool]]) -> List[Dict[str, Any]]:
    """
    Converts unified tools to Kiro API format.
    
    Args:
        tools: List of tools in unified format
    
    Returns:
        List of tools in Kiro toolSpecification format
    """
    if not tools:
        return []
    
    kiro_tools = []
    for tool in tools:
        # Sanitize parameters from fields that Kiro API doesn't accept
        sanitized_params = sanitize_json_schema(tool.input_schema)
        
        # Kiro API requires non-empty description
        description = tool.description
        if not description or not description.strip():
            description = f"Tool: {tool.name}"
            logger.debug(f"Tool '{tool.name}' has empty description, using placeholder")
        
        kiro_tools.append({
            "toolSpecification": {
                "name": tool.name,
                "description": description,
                "inputSchema": {"json": sanitized_params}
            }
        })
    
    return kiro_tools


# ==================================================================================================
# Image Conversion to Kiro Format
# ==================================================================================================

def convert_images_to_kiro_format(images: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Converts unified images to Kiro API format.
    
    Unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
    Kiro format: [{"format": "jpeg", "source": {"bytes": "base64..."}}]
    
    IMPORTANT: Images must be placed directly in userInputMessage.images,
    NOT in userInputMessageContext.images. This matches the native Kiro IDE format.
    
    Also handles the case where data contains a full data URL (data:image/jpeg;base64,...)
    by stripping the prefix and extracting pure base64.
    
    Args:
        images: List of images in unified format
    
    Returns:
        List of images in Kiro format, ready for userInputMessage.images
    
    Example:
        >>> convert_images_to_kiro_format([{"media_type": "image/png", "data": "abc123"}])
        [{'format': 'png', 'source': {'bytes': 'abc123'}}]
    """
    if not images:
        return []
    
    kiro_images = []
    for img in images:
        media_type = img.get("media_type", "image/jpeg")
        data = img.get("data", "")
        
        if not data:
            logger.warning("Skipping image with empty data")
            continue
        
        # Strip data URL prefix if present (some clients send "data:image/jpeg;base64,..." in data field)
        # Kiro API expects pure base64 without the prefix
        if data.startswith("data:"):
            try:
                header, actual_data = data.split(",", 1)
                # Extract media type from header if present
                media_part = header.split(";")[0]  # "data:image/jpeg"
                extracted_media_type = media_part.replace("data:", "")
                if extracted_media_type:
                    media_type = extracted_media_type
                data = actual_data
                logger.debug(f"Stripped data URL prefix, extracted media_type: {media_type}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse data URL prefix: {e}")
        
        # Extract format from media_type: "image/jpeg" -> "jpeg"
        format_str = media_type.split("/")[-1] if "/" in media_type else media_type
        
        kiro_images.append({
            "format": format_str,
            "source": {
                "bytes": data
            }
        })
    
    if kiro_images:
        logger.debug(f"Converted {len(kiro_images)} image(s) to Kiro format")
    
    return kiro_images


# ==================================================================================================
# Tool Results and Tool Uses Extraction
# ==================================================================================================

def convert_tool_results_to_kiro_format(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts unified tool results to Kiro API format.
    
    Unified format: {"type": "tool_result", "tool_use_id": "...", "content": "..."}
    Kiro format: {"content": [{"text": "..."}], "status": "success", "toolUseId": "..."}
    
    Args:
        tool_results: List of tool results in unified format
    
    Returns:
        List of tool results in Kiro format
    """
    kiro_results = []
    for tr in tool_results:
        content = tr.get("content", "")
        if isinstance(content, str):
            content_text = content
        else:
            content_text = extract_text_content(content)
        
        # Ensure content is not empty - Kiro API requires non-empty content
        if not content_text:
            content_text = "(empty result)"
        
        kiro_results.append({
            "content": [{"text": content_text}],
            "status": "success",
            "toolUseId": tr.get("tool_use_id", "")
        })
    
    return kiro_results


def extract_tool_results_from_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool results from message content.
    
    Looks for content blocks with type="tool_result" and converts them
    to Kiro API format.
    
    Args:
        content: Message content (can be a list of content blocks)
    
    Returns:
        List of tool results in Kiro format
    """
    tool_results = []
    
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_results.append({
                    "content": [{"text": extract_text_content(item.get("content", "")) or "(empty result)"}],
                    "status": "success",
                    "toolUseId": item.get("tool_use_id", "")
                })
    
    return tool_results


def coerce_tool_input_to_dict(raw: Any) -> Dict[str, Any]:
    """
    Coerces a tool-call input/arguments value to a dict for the Kiro API.

    Kiro requires tool inputs to be JSON objects, but OpenAI-format
    ``arguments`` arrive as raw strings and bridged GPT conversations can
    carry strings that are not valid JSON at all (e.g. ``":"``). Valid
    JSON objects are parsed as-is; anything else degrades to an empty dict
    with a warning instead of raising and failing the whole request.

    Args:
        raw: Tool input value of any type (dict, str, other)

    Returns:
        A dict suitable for the Kiro API
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                f"Tool input string is not valid JSON, coercing to empty dict: {stripped[:80]!r}"
            )
            return {}
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            f"Tool input JSON is not an object, coercing to empty dict: {stripped[:80]!r}"
        )
        return {}
    if raw is not None:
        logger.warning(
            f"Tool input is not an object, coercing to empty dict: {type(raw).__name__}"
        )
    return {}


def extract_tool_uses_from_message(
    content: Any,
    tool_calls: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts tool uses from assistant message.
    
    Looks for tool calls in both:
    - tool_calls field (OpenAI format)
    - content blocks with type="tool_use" (Anthropic format)
    
    Args:
        content: Message content
        tool_calls: List of tool calls (OpenAI format)
    
    Returns:
        List of tool uses in Kiro format
    """
    tool_uses = []
    
    # From tool_calls field (OpenAI format or unified format from Anthropic)
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                arguments = func.get("arguments", "{}")
                # Handle both string (OpenAI) and dict (Anthropic unified) formats;
                # malformed strings degrade to {} instead of raising
                input_data = coerce_tool_input_to_dict(arguments)
                tool_uses.append({
                    "name": func.get("name", ""),
                    "input": input_data,
                    "toolUseId": tc.get("id", "")
                })

    # From content blocks (Anthropic format)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                tool_uses.append({
                    "name": item.get("name", ""),
                    "input": coerce_tool_input_to_dict(item.get("input", {})),
                    "toolUseId": item.get("id", "")
                })

    return tool_uses


# ==================================================================================================
# Tool Content to Text Conversion (for stripping when no tools defined)
# ==================================================================================================

def tool_calls_to_text(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Converts tool_calls to human-readable text representation.
    
    This is used when stripping tool content from messages (when no tools are defined).
    Instead of losing the context, we convert tool calls to text so the model
    can still understand what happened in the conversation.
    
    Args:
        tool_calls: List of tool calls in unified format
    
    Returns:
        Text representation of tool calls
    
    Example:
        >>> tool_calls_to_text([{"id": "call_123", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}])
        '[Tool: bash] (call_123)\\n{"command": "ls"}'
    """
    if not tool_calls:
        return ""
    
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        arguments = func.get("arguments", "{}")
        tool_id = tc.get("id", "")
        
        # Format: [Tool: name] (id)\narguments
        if tool_id:
            parts.append(f"[Tool: {name} ({tool_id})]\n{arguments}")
        else:
            parts.append(f"[Tool: {name}]\n{arguments}")
    
    return "\n\n".join(parts)


def tool_results_to_text(tool_results: List[Dict[str, Any]]) -> str:
    """
    Converts tool_results to human-readable text representation.
    
    This is used when stripping tool content from messages (when no tools are defined).
    Instead of losing the context, we convert tool results to text so the model
    can still understand what happened in the conversation.
    
    Args:
        tool_results: List of tool results in unified format
    
    Returns:
        Text representation of tool results
    
    Example:
        >>> tool_results_to_text([{"tool_use_id": "call_123", "content": "file1.txt\\nfile2.txt"}])
        '[Tool Result] (call_123)\\nfile1.txt\\nfile2.txt'
    """
    if not tool_results:
        return ""
    
    parts = []
    for tr in tool_results:
        content = tr.get("content", "")
        tool_use_id = tr.get("tool_use_id", "")
        
        if isinstance(content, str):
            content_text = content
        else:
            content_text = extract_text_content(content)
        
        # Use placeholder if content is empty
        if not content_text:
            content_text = "(empty result)"
        
        # Format: [Tool Result] (id)\ncontent
        if tool_use_id:
            parts.append(f"[Tool Result ({tool_use_id})]\n{content_text}")
        else:
            parts.append(f"[Tool Result]\n{content_text}")
    
    return "\n\n".join(parts)


# ==================================================================================================
# Message Merging
# ==================================================================================================

def strip_all_tool_content(messages: List[UnifiedMessage]) -> Tuple[List[UnifiedMessage], bool]:
    """
    Strips ALL tool-related content from messages, converting it to text representation.
    
    This is used when no tools are defined in the request. Kiro API rejects
    requests that have toolResults but no tools defined.
    
    Instead of simply removing tool content, this function converts tool_calls
    and tool_results to human-readable text, preserving the context for
    summarization and other use cases.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        Tuple of:
        - List of messages with tool content converted to text
        - Boolean indicating whether any tool content was converted
    """
    if not messages:
        return [], False
    
    result = []
    total_tool_calls_stripped = 0
    total_tool_results_stripped = 0
    
    for msg in messages:
        # Check if this message has any tool content
        has_tool_calls = bool(msg.tool_calls)
        has_tool_results = bool(msg.tool_results)
        
        if has_tool_calls or has_tool_results:
            if has_tool_calls:
                total_tool_calls_stripped += len(msg.tool_calls)
            if has_tool_results:
                total_tool_results_stripped += len(msg.tool_results)
            
            # Start with existing text content
            existing_content = extract_text_content(msg.content)
            content_parts = []
            
            if existing_content:
                content_parts.append(existing_content)
            
            # Convert tool_calls to text (for assistant messages)
            if has_tool_calls:
                tool_text = tool_calls_to_text(msg.tool_calls)
                if tool_text:
                    content_parts.append(tool_text)
            
            # Convert tool_results to text (for user messages)
            if has_tool_results:
                result_text = tool_results_to_text(msg.tool_results)
                if result_text:
                    content_parts.append(result_text)
            
            # Join all parts with double newline
            content = "\n\n".join(content_parts) if content_parts else "(empty placeholder)"
            
            # Create a copy of the message without tool content but with text representation
            # IMPORTANT: Preserve images from the original message (e.g., screenshots from MCP tools)
            cleaned_msg = UnifiedMessage(
                role=msg.role,
                content=content,
                tool_calls=None,
                tool_results=None,
                images=msg.images
            )
            result.append(cleaned_msg)
        else:
            result.append(msg)
    
    had_tool_content = total_tool_calls_stripped > 0 or total_tool_results_stripped > 0
    
    # Log summary once (DEBUG level - this is normal for clients like Cline/Roo/Cursor)
    if had_tool_content:
        logger.debug(
            f"Converted tool content to text (no tools defined): "
            f"{total_tool_calls_stripped} tool_calls, {total_tool_results_stripped} tool_results"
        )
    
    return result, had_tool_content


def ensure_assistant_before_tool_results(messages: List[UnifiedMessage]) -> Tuple[List[UnifiedMessage], bool]:
    """
    Ensures that messages with tool_results have a preceding assistant message with tool_calls.
    
    Kiro API requires that when toolResults are present, there must be a preceding
    assistantResponseMessage with toolUses. Some clients (like Cline/Roo/Cursor) may send
    truncated conversations where the assistant message is missing.
    
    Since we don't know the original tool name and arguments when the assistant message
    is missing, we cannot create a valid synthetic assistant message. Instead, we convert
    the tool_results to text representation and append to the message content, preserving
    the context for the model while avoiding Kiro API rejection.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        Tuple of:
        - List of messages with orphaned tool_results converted to text
        - Boolean indicating whether any tool_results were converted (used to skip thinking tag injection)
    """
    if not messages:
        return [], False
    
    result = []
    converted_any_tool_results = False
    
    for msg in messages:
        # Check if this message has tool_results
        if msg.tool_results:
            # Check if the previous message is an assistant with tool_calls
            has_preceding_assistant = (
                result and
                result[-1].role == "assistant" and
                result[-1].tool_calls
            )
            
            if not has_preceding_assistant:
                # We cannot create a valid synthetic assistant message because we don't know
                # the original tool name and arguments. Kiro API validates tool names.
                # Convert tool_results to text to preserve context for the model.
                logger.debug(
                    f"Converting {len(msg.tool_results)} orphaned tool_results to text "
                    f"(no preceding assistant message with tool_calls). "
                    f"Tool IDs: {[tr.get('tool_use_id', 'unknown') for tr in msg.tool_results]}"
                )
                
                # Convert tool_results to text representation
                tool_results_text = tool_results_to_text(msg.tool_results)
                
                # Append to existing content
                original_content = extract_text_content(msg.content) or ""
                if original_content and tool_results_text:
                    new_content = f"{original_content}\n\n{tool_results_text}"
                elif tool_results_text:
                    new_content = tool_results_text
                else:
                    new_content = original_content
                
                # Create a copy of the message with tool_results converted to text
                cleaned_msg = UnifiedMessage(
                    role=msg.role,
                    content=new_content,
                    tool_calls=msg.tool_calls,
                    tool_results=None,  # Remove orphaned tool_results (now in text)
                    images=msg.images
                )
                result.append(cleaned_msg)
                converted_any_tool_results = True
                continue
        
        result.append(msg)
    
    return result, converted_any_tool_results


def merge_adjacent_messages(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Merges adjacent messages with the same role.
    
    Kiro API does not accept multiple consecutive messages from the same role.
    This function merges such messages into one.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        List of messages with merged adjacent messages
    """
    if not messages:
        return []
    
    merged = []
    # Statistics for summary logging
    merge_counts = {"user": 0, "assistant": 0}
    total_tool_calls_merged = 0
    total_tool_results_merged = 0
    
    for msg in messages:
        if not merged:
            merged.append(msg)
            continue
        
        last = merged[-1]
        if msg.role == last.role:
            # Merge content
            if isinstance(last.content, list) and isinstance(msg.content, list):
                last.content = last.content + msg.content
            elif isinstance(last.content, list):
                last.content = last.content + [{"type": "text", "text": extract_text_content(msg.content)}]
            elif isinstance(msg.content, list):
                last.content = [{"type": "text", "text": extract_text_content(last.content)}] + msg.content
            else:
                last_text = extract_text_content(last.content)
                current_text = extract_text_content(msg.content)
                last.content = f"{last_text}\n{current_text}"
            
            # Merge tool_calls for assistant messages
            if msg.role == "assistant" and msg.tool_calls:
                if last.tool_calls is None:
                    last.tool_calls = []
                last.tool_calls = list(last.tool_calls) + list(msg.tool_calls)
                total_tool_calls_merged += len(msg.tool_calls)
            
            # Merge tool_results for user messages
            if msg.role == "user" and msg.tool_results:
                if last.tool_results is None:
                    last.tool_results = []
                last.tool_results = list(last.tool_results) + list(msg.tool_results)
                total_tool_results_merged += len(msg.tool_results)
            
            # Count merges by role
            if msg.role in merge_counts:
                merge_counts[msg.role] += 1
        else:
            merged.append(msg)
    
    # Log summary if any merges occurred
    total_merges = sum(merge_counts.values())
    if total_merges > 0:
        parts = []
        for role, count in merge_counts.items():
            if count > 0:
                parts.append(f"{count} {role}")
        merge_summary = ", ".join(parts)
        
        extras = []
        if total_tool_calls_merged > 0:
            extras.append(f"{total_tool_calls_merged} tool_calls")
        if total_tool_results_merged > 0:
            extras.append(f"{total_tool_results_merged} tool_results")
        
        if extras:
            logger.debug(f"Merged {total_merges} adjacent messages ({merge_summary}), including {', '.join(extras)}")
        else:
            logger.debug(f"Merged {total_merges} adjacent messages ({merge_summary})")
    
    return merged


def ensure_first_message_is_user(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Ensures that the first message in the conversation is from user role.
    
    Kiro API requires conversations to start with a user message. If the first
    message is from assistant (or any other non-user role), we prepend a minimal
    synthetic user message.
    
    This matches LiteLLM behavior for Anthropic API compatibility and fixes
    issue #60 where conversations starting with assistant messages cause
    "Improperly formed request" errors.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        List of messages with guaranteed user-first order
    
    Example:
        >>> messages = [
        ...     UnifiedMessage(role="assistant", content="Hello"),
        ...     UnifiedMessage(role="user", content="Hi")
        ... ]
        >>> result = ensure_first_message_is_user(messages)
        >>> result[0].role
        'user'
        >>> result[0].content
        '(empty placeholder)'
    """
    if not messages:
        return messages
    
    if messages[0].role != "user":
        logger.debug(
            f"First message is '{messages[0].role}', prepending synthetic user message "
            f"(Kiro API requires conversations to start with user)"
        )
        # Create minimal synthetic user message (matches LiteLLM behavior)
        # Using "(empty placeholder)" as minimal valid content to avoid disrupting conversation context
        synthetic_user = UnifiedMessage(
            role="user",
            content="(empty placeholder)"
        )
        
        return [synthetic_user] + messages
    
    return messages


def normalize_message_roles(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Normalizes unknown message roles to 'user'.
    
    Kiro API only supports 'user' and 'assistant' roles in history.
    Any other role (e.g., 'developer', 'system') is converted to 'user'
    to maintain compatibility.
    
    This normalization MUST happen before ensure_alternating_roles()
    to ensure consecutive messages with unknown roles are properly detected
    and synthetic assistant messages are inserted between them.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        List of messages with normalized roles
    
    Example:
        >>> messages = [
        ...     UnifiedMessage(role="developer", content="Context 1"),
        ...     UnifiedMessage(role="developer", content="Context 2"),
        ...     UnifiedMessage(role="user", content="Question")
        ... ]
        >>> result = normalize_message_roles(messages)
        >>> [msg.role for msg in result]
        ['user', 'user', 'user']
    """
    if not messages:
        return messages
    
    normalized = []
    converted_count = 0
    
    for msg in messages:
        if msg.role not in ("user", "assistant"):
            logger.debug(f"Normalizing role '{msg.role}' to 'user'")
            normalized_msg = UnifiedMessage(
                role="user",
                content=msg.content,
                tool_calls=msg.tool_calls,
                tool_results=msg.tool_results,
                images=msg.images
            )
            normalized.append(normalized_msg)
            converted_count += 1
        else:
            normalized.append(msg)
    
    if converted_count > 0:
        logger.debug(f"Normalized {converted_count} message(s) with unknown roles to 'user'")
    
    return normalized


def repair_unpaired_tool_uses(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Synthesizes toolResults for toolUses that never received one.

    Kiro rejects a request with 400 REQUEST_BODY_INVALID when an assistant
    toolUse has no matching toolResult. Proxies in front of the gateway
    (notably cc-switch translating Codex /responses items) can drop tool
    messages whose output is pure media, leaving orphaned function calls in
    the history. Once such items enter a session every later request fails
    identically, so the gateway guarantees the pairing invariant itself.

    Only missing results are synthesized; existing results (including
    duplicates) are left untouched.

    Args:
        messages: Unified messages after role normalization/alternation

    Returns:
        The same list (mutated in place) with synthetic tool_results appended
        to the user message that follows the owning assistant message
    """
    repaired = 0
    for idx, msg in enumerate(messages):
        if msg.role != "assistant" or not msg.tool_calls:
            continue

        target_idx = idx + 1
        if target_idx >= len(messages) or messages[target_idx].role != "user":
            synthetic = UnifiedMessage(role="user", content="", tool_results=[])
            messages.insert(target_idx, synthetic)

        target = messages[target_idx]
        adjacent_result_ids = {
            result.get("tool_use_id")
            for result in target.tool_results or []
            if result.get("tool_use_id")
        }
        missing = [
            tool_call.get("id")
            for tool_call in msg.tool_calls
            if tool_call.get("id") and tool_call.get("id") not in adjacent_result_ids
        ]
        if not missing:
            continue

        if target.tool_results is None:
            target.tool_results = []
        for tool_use_id in missing:
            target.tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": (
                    "[gateway: tool result was not delivered by the client; "
                    "the tool likely produced an image or other media that was "
                    "moved into an adjacent user message.]"
                ),
            })
            repaired += 1

    if repaired:
        logger.info(f"Repaired {repaired} unpaired tool_use(s) with synthetic results")
    return messages


def ensure_alternating_roles(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Ensures alternating user/assistant roles by inserting synthetic assistant messages.
    
    Kiro API requires alternating userInputMessage and assistantResponseMessage.
    When consecutive user messages are detected, synthetic assistant messages
    with "(empty placeholder)" placeholder are inserted between them to maintain alternation.
    
    This fixes multiple unknown roles (converted to user)
    create consecutive userInputMessage entries that violate Kiro API requirements.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        List of messages with synthetic assistant messages inserted where needed
    
    Example:
        >>> messages = [
        ...     UnifiedMessage(role="user", content="First"),
        ...     UnifiedMessage(role="user", content="Second"),
        ...     UnifiedMessage(role="user", content="Third")
        ... ]
        >>> result = ensure_alternating_roles(messages)
        >>> len(result)
        5  # 3 user + 2 synthetic assistant
        >>> result[1].role
        'assistant'
        >>> result[1].content
        '(empty placeholder)'
    """
    if not messages or len(messages) < 2:
        return messages
    
    result = [messages[0]]
    synthetic_count = 0
    
    for msg in messages[1:]:
        prev_role = result[-1].role
        
        # If both current and previous are user → insert synthetic assistant
        if msg.role == "user" and prev_role == "user":
            synthetic_assistant = UnifiedMessage(
                role="assistant",
                content="(empty placeholder)"  # Consistent with build_kiro_history() placeholder
            )
            result.append(synthetic_assistant)
            synthetic_count += 1
        
        result.append(msg)
    
    if synthetic_count > 0:
        logger.debug(f"Inserted {synthetic_count} synthetic assistant message(s) to ensure alternation")
    
    return result


# ==================================================================================================
# Kiro History Building
# ==================================================================================================

async def build_kiro_history(
    messages: List[UnifiedMessage],
    model_id: str,
    budget: Optional["ImageFetchBudget"] = None,
) -> List[Dict[str, Any]]:
    """
    Builds history array for Kiro API from unified messages.
    
    Kiro API expects alternating userInputMessage and assistantResponseMessage.
    This function converts unified format to Kiro format.
    
    All messages should have 'user' or 'assistant' roles at this point,
    as unknown roles are normalized earlier in the pipeline by normalize_message_roles().
    
    Args:
        messages: List of messages in unified format (with normalized roles)
        model_id: Internal Kiro model ID
        budget: Shared ImageFetchBudget passed down to image extraction, so a
            URL image repeated across history turns is fetched only once
    
    Returns:
        List of dictionaries for history field in Kiro API
    """
    history = []
    
    # Own a budget when none was shared, so dedup and the cap span the whole
    # history instead of resetting on every message.
    if budget is None:
        budget = ImageFetchBudget()
    
    for msg in messages:
        if msg.role == "user":
            content = extract_text_content(msg.content)

            # Process tool_results - convert to Kiro format if present
            if msg.tool_results:
                kiro_tool_results = convert_tool_results_to_kiro_format(msg.tool_results)
            else:
                # Try to extract from content (already in Kiro format)
                kiro_tool_results = extract_tool_results_from_content(msg.content)

            # Kiro accepts empty content when the message carries toolResults,
            # so pure tool-result turns send "" instead of a visible placeholder.
            if not content:
                content = "" if kiro_tool_results else "(empty placeholder)"

            user_input = {
                "content": content,
                "modelId": model_id,
                "origin": "AI_EDITOR",
            }

            # Process images - extract from message or content
            # IMPORTANT: images go directly into userInputMessage, NOT into userInputMessageContext
            # This matches the native Kiro IDE format
            images = msg.images or await extract_images_from_content(msg.content, budget)
            if images:
                kiro_images = convert_images_to_kiro_format(images)
                if kiro_images:
                    user_input["images"] = kiro_images

            if kiro_tool_results:
                user_input["userInputMessageContext"] = {"toolResults": kiro_tool_results}

            history.append({"userInputMessage": user_input})
            
        elif msg.role == "assistant":
            content = extract_text_content(msg.content)

            # Process tool_calls
            tool_uses = extract_tool_uses_from_message(msg.content, msg.tool_calls)

            # Kiro rejects a missing content key but accepts an empty string, so
            # pure tool-call turns send "" instead of a visible placeholder.
            if not content:
                content = "" if tool_uses else "(empty placeholder)"

            assistant_response = {"content": content}
            if tool_uses:
                assistant_response["toolUses"] = tool_uses

            history.append({"assistantResponseMessage": assistant_response})
    
    return history


# ==================================================================================================
# Main Payload Building
# ==================================================================================================

def build_tool_choice_directive(
    tool_choice: Any,
    tools: Optional[List[UnifiedTool]] = None,
    protocol: Optional[str] = None,
) -> str:
    """Build a tool-choice directive, with strict validation when possible.

    ``tools`` and ``protocol`` are optional for compatibility with callers that
    only need the historical best-effort directive behavior.
    """
    if protocol:
        return parse_tool_choice_policy(tool_choice, tools, protocol).build_directive()
    if not tool_choice or tool_choice == "auto":
        return ""
    if tool_choice in ("required", "any"):
        return ToolChoicePolicy(mode="required").build_directive()
    if tool_choice == "none":
        return ToolChoicePolicy(mode="none").build_directive()
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in ("required", "any"):
            return ToolChoicePolicy(mode="required").build_directive()
        if choice_type == "none":
            return ToolChoicePolicy(mode="none").build_directive()
        if choice_type == "tool":
            name = tool_choice.get("name")
        elif choice_type == "function" and isinstance(tool_choice.get("function"), dict):
            name = tool_choice["function"].get("name")
        else:
            name = None
        if name:
            return ToolChoicePolicy(mode="named", tool_name=name).build_directive()
    return ""


async def build_kiro_payload(
    messages: List[UnifiedMessage],
    system_prompt: str,
    model_id: str,
    tools: Optional[List[UnifiedTool]],
    conversation_id: str,
    profile_arn: str,
    thinking_config: ThinkingConfig,
    native_thinking: Optional[Dict[str, Any]] = None,
    request_audit: Optional[RequestAudit] = None,
    budget: Optional["ImageFetchBudget"] = None,
) -> KiroPayloadResult:
    """
    Builds complete payload for Kiro API from unified data.
    
    This is the main function that assembles the Kiro API payload from
    API-agnostic unified message and tool formats.
    
    Args:
        messages: List of messages in unified format (without system messages)
        system_prompt: Already extracted system prompt
        model_id: Internal Kiro model ID
        tools: List of tools in unified format (or None)
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN
        thinking_config: Thinking configuration from API adapter
        native_thinking: Anthropic adaptive thinking dictionary to forward verbatim
        request_audit: Optional request audit state shared with the response stream
        budget: Shared ImageFetchBudget covering both history and the current
            message, so one request cannot exceed the URL fetch cap and repeated
            URLs are fetched once

    Returns:
        KiroPayloadResult with payload and tool documentation
    
    Raises:
        ValueError: If there are no messages to send
    """
    # One budget for the whole payload: history and the current message share
    # the fetch cap and the dedup cache.
    if budget is None:
        budget = ImageFetchBudget()
    
    # Process tools with long descriptions
    processed_tools, tool_documentation = process_tools_with_long_descriptions(tools)
    
    # Validate tool names against Kiro API 64-character limit
    validate_tool_names(processed_tools)
    
    # Add tool documentation to system prompt if present
    full_system_prompt = system_prompt
    if tool_documentation:
        full_system_prompt = full_system_prompt + tool_documentation if full_system_prompt else tool_documentation.strip()

    if not thinking_config.enabled:
        requested_effort = "none" if NATIVE_EFFORT_NONE_ON_DISABLED else None
        default_effort = None
    else:
        requested_effort = thinking_config.effort
        default_effort = NATIVE_EFFORT_DEFAULT
    effort_decision = resolve_effort_decision(
        model_id, requested_effort, default_tier=default_effort
    )
    if request_audit is not None:
        request_audit.record_effort(model_id, effort_decision)
    else:
        logger.info(
            "effort_decision "
            f"model={model_id} "
            f"requested={effort_decision.requested or 'none'} "
            f"adopted={effort_decision.adopted or 'none'} "
            f"field={effort_decision.field or 'none'} "
            f"outcome={effort_decision.outcome} "
            f"clamped={str(effort_decision.clamped).lower()} "
            f"reason={effort_decision.reason}"
        )

    native_fragment: Dict[str, Any] = (
        dict(effort_decision.fragment) if effort_decision.fragment is not None else {}
    )
    native_thinking_requested = native_thinking is not None and NATIVE_EFFORT_ENABLED
    if native_thinking_requested:
        if lookup_effort_schema(model_id) is not None:
            native_fragment["thinking"] = dict(native_thinking)
        else:
            logger.debug(
                f"Model '{model_id}' has no native thinking channel; "
                "omitting adaptive thinking instead of fabricating a budget"
            )

    suppress_tags = (
        bool(native_fragment) or native_thinking_requested
    ) and NATIVE_EFFORT_SUPPRESS_TAGS

    # Add thinking mode legitimization to system prompt if enabled
    if not suppress_tags:
        thinking_system_addition = get_thinking_system_prompt_addition()
        if thinking_system_addition:
            full_system_prompt = full_system_prompt + thinking_system_addition if full_system_prompt else thinking_system_addition.strip()
    
    # Add truncation recovery legitimization to system prompt if enabled
    truncation_system_addition = get_truncation_recovery_system_addition()
    if truncation_system_addition:
        full_system_prompt = full_system_prompt + truncation_system_addition if full_system_prompt else truncation_system_addition.strip()
    
    declared_tool_names = {tool.name for tool in processed_tools or []}
    has_undeclared_history_tool = any(
        tool_call.get("function", {}).get("name") not in declared_tool_names
        for message in messages
        for tool_call in message.tool_calls or []
    )
    if not processed_tools or has_undeclared_history_tool:
        messages_without_tools, had_tool_content = strip_all_tool_content(messages)
        messages_with_assistants = messages_without_tools
        converted_tool_results = had_tool_content
    else:
        # Ensure assistant messages exist before tool_results (Kiro API requirement)
        # Also returns flag if any tool_results were converted (to skip thinking tag injection)
        messages_with_assistants, converted_tool_results = ensure_assistant_before_tool_results(messages)
    
    # Merge adjacent messages with the same role
    merged_messages = merge_adjacent_messages(messages_with_assistants)
    
    # Ensure first message is from user (Kiro API requirement, fixes issue #60)
    merged_messages = ensure_first_message_is_user(merged_messages)
    
    # Normalize unknown roles to 'user' (fixes issue #64)
    # This must happen BEFORE ensure_alternating_roles() so that consecutive
    # messages with unknown roles (e.g., 'developer') are properly detected
    merged_messages = normalize_message_roles(merged_messages)
    
    # Ensure alternating user/assistant roles (fixes issue #64)
    # Insert synthetic assistant messages between consecutive user messages
    merged_messages = ensure_alternating_roles(merged_messages)

    # Guarantee every assistant toolUse has a matching toolResult. Client-side
    # proxies can drop pure-media tool messages (see repair_unpaired_tool_uses),
    # and Kiro rejects the whole request otherwise.
    merged_messages = repair_unpaired_tool_uses(merged_messages)

    if not merged_messages:
        raise ValueError("No messages to send")
    
    # Build history (all messages except the last one)
    history_messages = merged_messages[:-1] if len(merged_messages) > 1 else []
    
    # If there's a system prompt, add it to the first user message in history
    if full_system_prompt and history_messages:
        first_msg = history_messages[0]
        if first_msg.role == "user":
            original_content = extract_text_content(first_msg.content)
            first_msg.content = f"{full_system_prompt}\n\n{original_content}"
    
    history = await build_kiro_history(history_messages, model_id, budget)
    
    # Current message (the last one)
    current_message = merged_messages[-1]
    current_content = extract_text_content(current_message.content)
    
    # If system prompt exists but history is empty - add to current message
    if full_system_prompt and not history:
        current_content = f"{full_system_prompt}\n\n{current_content}"
    
    # If current message is assistant, need to add it to history
    # and create user message placeholder
    if current_message.role == "assistant":
        history.append({
            "assistantResponseMessage": {
                "content": current_content
            }
        })
        current_content = "(empty placeholder)"
    
    # If content is empty and the message carries tool results, Kiro accepts
    # an empty string; only use the visible placeholder for truly empty turns.
    if not current_content:
        has_tool_results = bool(current_message.tool_results) or bool(
            extract_tool_results_from_content(current_message.content)
        )
        current_content = "" if has_tool_results else "(empty placeholder)"
    
    # Process images in current message - extract from message or content
    # IMPORTANT: images go directly into userInputMessage, NOT into userInputMessageContext
    # This matches the native Kiro IDE format
    images = current_message.images or await extract_images_from_content(current_message.content, budget)
    kiro_images = None
    if images:
        kiro_images = convert_images_to_kiro_format(images)
        if kiro_images:
            logger.debug(f"Added {len(kiro_images)} image(s) to current message")
    
    # Build user_input_context for tools and toolResults only (NOT images)
    user_input_context: Dict[str, Any] = {}
    
    # Add tools if present
    kiro_tools = convert_tools_to_kiro_format(processed_tools)
    if kiro_tools:
        user_input_context["tools"] = kiro_tools
    
    # Process tool_results in current message - convert to Kiro format if present
    if current_message.tool_results:
        # Convert unified format to Kiro format
        kiro_tool_results = convert_tool_results_to_kiro_format(current_message.tool_results)
        if kiro_tool_results:
            user_input_context["toolResults"] = kiro_tool_results
    else:
        # Try to extract from content (already in Kiro format)
        tool_results = extract_tool_results_from_content(current_message.content)
        if tool_results:
            user_input_context["toolResults"] = tool_results
    
    # Inject thinking tags if enabled (only for the current/last user message)
    if current_message.role == "user":
        if suppress_tags:
            logger.debug("Skipping thinking tag injection for native thinking or effort")
        else:
            current_content = inject_thinking_tags(current_content, thinking_config)
    
    # Build userInputMessage
    user_input_message = {
        "content": current_content,
        "modelId": model_id,
        "origin": "AI_EDITOR",
    }
    
    # Add images directly to userInputMessage (NOT to userInputMessageContext)
    if kiro_images:
        user_input_message["images"] = kiro_images
    
    # Add user_input_context if present (contains tools and toolResults only)
    if user_input_context:
        user_input_message["userInputMessageContext"] = user_input_context
    
    # Assemble final payload
    payload = {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": conversation_id,
            "currentMessage": {
                "userInputMessage": user_input_message
            }
        }
    }
    
    # Add history only if not empty
    if history:
        payload["conversationState"]["history"] = history
    
    # Add profileArn
    if profile_arn:
        payload["profileArn"] = profile_arn

    if native_fragment:
        payload[NATIVE_EFFORT_FIELD] = native_fragment

    # Payload size guard — auto-trim if enabled
    if AUTO_TRIM_PAYLOAD:
        payload_size = check_payload_size(payload)
        if payload_size > KIRO_MAX_PAYLOAD_BYTES:
            stats = trim_payload_to_limit(payload, KIRO_MAX_PAYLOAD_BYTES)
            logger.info(
                f"Trimmed conversation history: {stats.original_entries} -> {stats.final_entries} messages "
                f"({stats.original_bytes} -> {stats.final_bytes} bytes)"
            )

    return KiroPayloadResult(payload=payload, tool_documentation=tool_documentation)