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
Converters for transforming Anthropic Messages API format to Kiro format.

This module is an adapter layer that converts Anthropic-specific formats
to the unified format used by converters_core.py.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from kiro.config import (
    EFFORT_FALLBACK,
    EFFORT_ORDER,
    GPT_EDIT_RECOVERY,
    HIDDEN_MODELS,
    MODEL_ALIASES,
    NATIVE_EFFORT_DEFAULT_ANTHROPIC,
)
from kiro.model_resolver import get_model_id_for_kiro
from kiro.effort_schema import (
    detect_effort_level,
    effort_from_budget,
    resolve_first_token_timeout,
)
from kiro.models_anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessage,
    AnthropicTool,
)
from kiro.converters_core import (
    UnifiedMessage,
    UnifiedTool,
    ThinkingConfig,
    ToolChoicePolicy,
    ImageFetchBudget,
    build_kiro_payload,
    parse_tool_choice_policy,
    coerce_tool_input_to_dict,
    extract_text_content,
    extract_images_from_content,
)
from kiro.request_audit import RequestAudit


def convert_anthropic_content_to_text(content: Any) -> str:
    """
    Extracts text content from Anthropic message content.

    Anthropic content can be:
    - String: "Hello, world!"
    - List of content blocks: [{"type": "text", "text": "Hello"}]

    Args:
        content: Anthropic message content

    Returns:
        Extracted text content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)
        return "".join(text_parts)

    return str(content) if content else ""


EDIT_TOOL_MISMATCH = "String to replace not found in file"


GPT_EDIT_RECOVERY_POLICY = (
    "\n\n---\n"
    "# Claude Code Edit Recovery\n\n"
    "When using Claude Code's Edit tool, copy `old_string` exactly from the "
    "latest contents returned by Read. Keep each replacement small and unique. "
    "If an Edit tool result says `String to replace not found in file`, do not "
    "repeat that Edit call or its arguments: first Read the same target file "
    "again, then create a new Edit from the latest text.\n"
)


GPT_EDIT_RECOVERY_NOTICE = (
    "\n\n[Edit Recovery Notice] The previous Edit failed because its old_string "
    "did not match the current file. Do not repeat the same Edit parameters. "
    "Read the target file now, then retry with a small, unique old_string "
    "copied exactly from that latest Read result.\n"
)


def is_gpt_model(model: Any) -> bool:
    """Return whether an Anthropic request targets a GPT model.

    Args:
        model: Client-supplied model name.

    Returns:
        True when the name identifies a GPT model.
    """
    if not isinstance(model, str):
        return False
    normalized = model.strip().lower()
    return normalized.startswith(("gpt-", "gpt_"))


def _block_value(block: Any, key: str, default: Any = None) -> Any:
    """Read a field from a content block held as either a dict or a model."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def has_unrecovered_edit_mismatch(messages: Any) -> bool:
    """Detect a latest, unrecovered Claude Code Edit replacement failure.

    The gateway cannot inspect the local workspace. It only uses the Anthropic
    tool-use/tool-result history and deliberately treats the latest tool event
    as authoritative: a later assistant tool call (for example Read) clears the
    pending failure, so the recovery notice is emitted at most once per failure.

    Args:
        messages: Anthropic request messages.

    Returns:
        True when the most recent tool event is an unrecovered Edit mismatch.
    """
    if not isinstance(messages, list):
        return False

    pending_edit_ids: Set[str] = set()
    latest_tool_event_is_edit_mismatch = False

    for message in messages:
        role = _block_value(message, "role", "")
        content = _block_value(message, "content", None)
        blocks = content if isinstance(content, list) else []

        if role == "assistant":
            assistant_called_tool = False
            for block in blocks:
                if _block_value(block, "type") != "tool_use":
                    continue
                assistant_called_tool = True
                latest_tool_event_is_edit_mismatch = False
                tool_id = _block_value(block, "id")
                tool_name = str(_block_value(block, "name", ""))
                if tool_id and tool_name.lower() == "edit":
                    pending_edit_ids.add(tool_id)

            # Any assistant tool call means the model has started a new step;
            # do not carry a prior failure past it.
            if assistant_called_tool:
                continue

        if role != "user":
            continue

        for block in blocks:
            if _block_value(block, "type") != "tool_result":
                continue
            latest_tool_event_is_edit_mismatch = False
            tool_id = _block_value(block, "tool_use_id")
            if tool_id not in pending_edit_ids:
                continue
            pending_edit_ids.discard(tool_id)
            result_text = convert_anthropic_content_to_text(
                _block_value(block, "content", "")
            )
            latest_tool_event_is_edit_mismatch = EDIT_TOOL_MISMATCH in result_text

    return latest_tool_event_is_edit_mismatch


def build_gpt_edit_recovery_directive(model: Any, messages: Any) -> str:
    """Build the GPT-only Edit policy and one-shot recovery notice.

    Args:
        model: Client-supplied model name.
        messages: Anthropic request messages.

    Returns:
        Directive text to append to the system prompt, or an empty string when
        the feature is disabled or the request does not target a GPT model.
    """
    if not GPT_EDIT_RECOVERY:
        return ""
    if not is_gpt_model(model):
        return ""
    directive = GPT_EDIT_RECOVERY_POLICY
    if has_unrecovered_edit_mismatch(messages):
        directive += GPT_EDIT_RECOVERY_NOTICE
    return directive


def extract_system_prompt(system: Any) -> str:
    """
    Extracts system prompt text from Anthropic system field.

    Anthropic API supports system in two formats:
    1. String: "You are helpful"
    2. List of content blocks: [{"type": "text", "text": "...", "cache_control": {...}}]

    The second format is used for prompt caching with cache_control.
    We extract only the text, ignoring cache_control (not supported by Kiro).

    Args:
        system: System prompt in string or list format

    Returns:
        Extracted system prompt as string
    """
    if system is None:
        return ""

    if isinstance(system, str):
        return system

    if isinstance(system, list):
        text_parts = []
        for block in system:
            if isinstance(block, dict):
                # Handle {"type": "text", "text": "...", "cache_control": {...}}
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                # Handle Pydantic model
                text_parts.append(getattr(block, "text", ""))
        return "\n".join(text_parts)

    return str(system)


def extract_tool_results_from_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool results from Anthropic message content.

    Looks for content blocks with type="tool_result".

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of tool results in unified format
    """
    tool_results = []

    if not isinstance(content, list):
        return tool_results

    for block in content:
        block_type = None
        tool_use_id = None
        result_content = ""

        if isinstance(block, dict):
            block_type = block.get("type")
            tool_use_id = block.get("tool_use_id")
            result_content = block.get("content", "")
        elif hasattr(block, "type"):
            block_type = block.type
            tool_use_id = getattr(block, "tool_use_id", None)
            result_content = getattr(block, "content", "")

        if block_type == "tool_result" and tool_use_id:
            # Convert content to text if it's a list
            if isinstance(result_content, list):
                result_content = extract_text_content(result_content)
            elif not isinstance(result_content, str):
                result_content = str(result_content) if result_content else ""

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content or "(empty result)",
                }
            )

    return tool_results


async def extract_images_from_tool_results(
    content: Any,
    budget: Optional[ImageFetchBudget] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts images from tool_result content blocks.

    Tool results in Anthropic format can contain images (e.g., screenshots from browser tools).
    This function extracts those images so they can be passed to the model.

    Args:
        content: Anthropic message content (list of content blocks)
        budget: Shared ImageFetchBudget, so URL images inside tool results count
            against the same request-wide fetch cap as everything else

    Returns:
        List of images in unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
    """
    images: List[Dict[str, Any]] = []

    if not isinstance(content, list):
        return images

    for block in content:
        block_type = None
        result_content = None

        if isinstance(block, dict):
            block_type = block.get("type")
            result_content = block.get("content")
        elif hasattr(block, "type"):
            block_type = block.type
            result_content = getattr(block, "content", None)

        if block_type == "tool_result" and isinstance(result_content, list):
            # Extract images from the tool_result's content
            tool_result_images = await extract_images_from_content(result_content, budget)
            images.extend(tool_result_images)

    if images:
        logger.debug(f"Extracted {len(images)} image(s) from tool_result content")

    return images

    return tool_results


def extract_tool_uses_from_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool uses from Anthropic assistant message content.

    Looks for content blocks with type="tool_use".

    Args:
        content: Anthropic message content (list of content blocks)

    Returns:
        List of tool calls in unified format
    """
    tool_calls = []

    if not isinstance(content, list):
        return tool_calls

    for block in content:
        block_type = None
        tool_id = None
        tool_name = None
        tool_input = {}

        if isinstance(block, dict):
            block_type = block.get("type")
            tool_id = block.get("id")
            tool_name = block.get("name")
            tool_input = block.get("input", {})
        elif hasattr(block, "type"):
            block_type = block.type
            tool_id = getattr(block, "id", None)
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", {})

        if block_type == "tool_use" and tool_id and tool_name:
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        # Raw-dict paths bypass Pydantic coercion, so normalize
                        # string inputs here as well.
                        "arguments": coerce_tool_input_to_dict(tool_input),
                    },
                }
            )

    return tool_calls


async def convert_anthropic_messages(
    messages: List[AnthropicMessage],
    budget: Optional[ImageFetchBudget] = None,
) -> List[UnifiedMessage]:
    """
    Converts Anthropic messages to unified format.

    Handles:
    - Text content (string or list of text blocks)
    - Tool use blocks (assistant messages)
    - Tool result blocks (user messages)

    Args:
        messages: List of Anthropic messages
        budget: Shared ImageFetchBudget, so URL images across all messages count
            against one request-wide fetch cap and repeats are fetched once

    Returns:
        List of messages in unified format
    """

    unified_messages = []
    total_tool_calls = 0
    total_tool_results = 0
    total_images = 0

    for msg in messages:
        role = msg.role
        content = msg.content

        # Extract text content
        text_content = convert_anthropic_content_to_text(content)

        # Extract tool-related data and images based on role
        tool_calls = None
        tool_results = None
        images = None

        if role == "assistant":
            # Assistant messages may contain tool_use blocks
            tool_calls = extract_tool_uses_from_anthropic_content(content)
            if tool_calls:
                total_tool_calls += len(tool_calls)

        elif role == "user":
            # User messages may contain tool_result blocks and images
            tool_results = extract_tool_results_from_anthropic_content(content)
            if tool_results:
                total_tool_results += len(tool_results)

            # Extract images from user messages (both top-level and inside tool_results)
            images = await extract_images_from_content(content, budget)

            # Also extract images from inside tool_result content blocks
            # (e.g., screenshots returned by browser MCP tools)
            tool_result_images = await extract_images_from_tool_results(content, budget)
            if tool_result_images:
                if images:
                    images.extend(tool_result_images)
                else:
                    images = tool_result_images

            if images:
                total_images += len(images)

        unified_msg = UnifiedMessage(
            role=role,
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            images=images if images else None,
        )
        unified_messages.append(unified_msg)

    # Log summary if any tool content or images were found
    if total_tool_calls > 0 or total_tool_results > 0 or total_images > 0:
        logger.debug(
            f"Converted {len(messages)} Anthropic messages: "
            f"{total_tool_calls} tool_calls, {total_tool_results} tool_results, {total_images} images"
        )

    return unified_messages


def convert_anthropic_tools(
    tools: Optional[List[AnthropicTool]],
) -> Optional[List[UnifiedTool]]:
    """
    Converts Anthropic tools to unified format.

    Args:
        tools: List of Anthropic tools

    Returns:
        List of tools in unified format, or None if no tools
    """
    if not tools:
        return None

    unified_tools = []
    for tool in tools:
        # Handle both dict and Pydantic model
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description")
            input_schema = tool.get("input_schema", {})
        else:
            name = tool.name
            description = tool.description
            input_schema = tool.input_schema

        unified_tools.append(
            UnifiedTool(name=name, description=description, input_schema=input_schema)
        )

    return unified_tools if unified_tools else None


def resolve_anthropic_tool_choice(
    request: AnthropicMessagesRequest,
) -> Tuple[ToolChoicePolicy, Optional[List[UnifiedTool]], Set[str]]:
    """Resolve policy, upstream tools, and client-visible allowed names."""
    unified_tools = convert_anthropic_tools(request.tools)
    policy = parse_tool_choice_policy(request.tool_choice, unified_tools, "anthropic")
    selected_tools = policy.filter_tools(unified_tools)
    allowed_names = {tool.name for tool in selected_tools or []}
    return policy, selected_tools, allowed_names


def _normalize_effort(value: Any) -> Optional[str]:
    """Normalize a client-supplied effort value for schema lookup."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _thinking_config_from_effort(effort: Optional[str], source: str) -> Optional[ThinkingConfig]:
    """Convert a normalized effort value into a ThinkingConfig."""
    if effort is None:
        return None
    if effort == "none":
        logger.debug(f"Anthropic {source}='none' -> explicit disable")
        return ThinkingConfig(enabled=False)
    if effort not in EFFORT_ORDER:
        logger.warning(
            f"Unknown {source}='{effort}', defaulting to '{EFFORT_FALLBACK}'"
        )
        return ThinkingConfig(effort=EFFORT_FALLBACK)
    logger.debug(f"Extracted thinking config from Anthropic: source='{source}', effort='{effort}'")
    return ThinkingConfig(effort=effort)


def _effort_scan_texts(request: AnthropicMessagesRequest) -> List[str]:
    """Collect system and user-authored text for effort-level detection.

    Tool-result content is excluded: tool output is data, not instruction, and
    must not inject tier directives.
    """
    texts: List[str] = []
    system_text = extract_system_prompt(request.system)
    if system_text:
        texts.append(system_text)
    for msg in request.messages:
        if msg.role != "user":
            continue
        if isinstance(msg.content, str):
            texts.append(msg.content)
        elif isinstance(msg.content, list):
            for block in msg.content:
                block_type = (
                    block.get("type")
                    if isinstance(block, dict)
                    else getattr(block, "type", None)
                )
                if block_type == "text":
                    text = (
                        block.get("text", "")
                        if isinstance(block, dict)
                        else getattr(block, "text", "")
                    )
                    if text:
                        texts.append(text)
    return texts


def extract_thinking_config_from_anthropic(request: AnthropicMessagesRequest) -> ThinkingConfig:
    """Resolve Anthropic thinking settings without fabricating adaptive budgets.

    Priority is explicit disable, explicit numeric budget, adaptive thinking,
    output_config.effort, reasoning_effort, an effort=N directive in context
    text, then the legacy default path.
    """
    thinking = request.thinking if isinstance(request.thinking, dict) else None

    if thinking is not None and thinking.get("type") == "disabled":
        return ThinkingConfig(enabled=False)

    if thinking is not None:
        raw_budget = thinking.get("budget_tokens")
        if isinstance(raw_budget, int) and not isinstance(raw_budget, bool) and raw_budget > 0:
            derived_tier = effort_from_budget(raw_budget)
            logger.debug(
                "Extracted thinking config from Anthropic: "
                f"source='thinking.budget_tokens', budget={raw_budget}, "
                f"derived_tier='{derived_tier}'"
            )
            return ThinkingConfig(budget_tokens=raw_budget, effort=derived_tier)

    if thinking is not None and thinking.get("type") == "adaptive":
        effort_config = _thinking_config_from_effort(
            _normalize_effort(getattr(request.output_config, "effort", None)),
            "output_config.effort",
        )
        if effort_config is not None:
            return effort_config
        effort_config = _thinking_config_from_effort(
            _normalize_effort(request.reasoning_effort),
            "reasoning_effort",
        )
        if effort_config is not None:
            return effort_config
        detected = detect_effort_level(_effort_scan_texts(request))
        if detected is not None:
            return ThinkingConfig(effort=detected)
        return ThinkingConfig()

    effort_config = _thinking_config_from_effort(
        _normalize_effort(getattr(request.output_config, "effort", None)),
        "output_config.effort",
    )
    if effort_config is not None:
        return effort_config

    effort_config = _thinking_config_from_effort(
        _normalize_effort(request.reasoning_effort),
        "reasoning_effort",
    )
    if effort_config is not None:
        return effort_config

    detected = detect_effort_level(_effort_scan_texts(request))
    if detected is not None:
        return ThinkingConfig(effort=detected)

    if thinking is not None and thinking.get("type") == "enabled":
        return ThinkingConfig()

    return ThinkingConfig()


def resolve_anthropic_first_token_timeout(request: AnthropicMessagesRequest) -> float:
    """Resolve the first-byte wait for this request's effective effort tier."""
    model_id = get_model_id_for_kiro(request.model, HIDDEN_MODELS, MODEL_ALIASES)
    thinking_config = extract_thinking_config_from_anthropic(request)
    return resolve_first_token_timeout(
        model_id,
        thinking_config.effort,
        default_tier=NATIVE_EFFORT_DEFAULT_ANTHROPIC if thinking_config.enabled else None,
    )


async def anthropic_to_kiro(
    request: AnthropicMessagesRequest,
    conversation_id: str,
    profile_arn: str,
    request_audit: Optional[RequestAudit] = None,
) -> dict:
    """
    Converts Anthropic Messages API request to Kiro API payload.

    This is the main entry point for Anthropic → Kiro conversion.

    Key differences from OpenAI:
    - System prompt is a separate field (not in messages)
    - Content can be string or list of content blocks
    - Tool format uses input_schema instead of parameters

    Args:
        request: Anthropic MessagesRequest
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN
        request_audit: Optional request audit state shared with the response stream

    Returns:
        Payload dictionary for POST request to Kiro API

    Raises:
        ValueError: If there are no messages to send
    """
    # One budget for the request: message conversion and payload building both
    # extract images, so they must share the cap and the dedup cache.
    image_budget = ImageFetchBudget()

    # Convert messages to unified format
    unified_messages = await convert_anthropic_messages(request.messages, image_budget)

    # Resolve against original client-visible names before extension aliasing.
    tool_choice_policy, unified_tools, _ = resolve_anthropic_tool_choice(request)

    # System prompt is already separate in Anthropic format!
    # It can be a string or list of content blocks (for prompt caching)
    system_prompt = extract_system_prompt(request.system)

    tool_choice_directive = tool_choice_policy.build_directive()
    if tool_choice_directive:
        system_prompt = system_prompt + tool_choice_directive if system_prompt else tool_choice_directive.strip()

    # Opt-in via GPT_EDIT_RECOVERY. Anthropic path only; OpenAI conversion is untouched.
    edit_recovery_directive = build_gpt_edit_recovery_directive(request.model, request.messages)
    if edit_recovery_directive:
        system_prompt = (
            system_prompt + edit_recovery_directive
            if system_prompt
            else edit_recovery_directive.strip()
        )

    # Get model ID for Kiro API (normalizes + resolves hidden models)
    # Pass-through principle: we normalize and send to Kiro, Kiro decides if valid
    model_id = get_model_id_for_kiro(request.model, HIDDEN_MODELS, MODEL_ALIASES)

    # Extract thinking configuration from thinking parameter
    thinking_config = extract_thinking_config_from_anthropic(request)
    native_thinking = (
        dict(request.thinking)
        if isinstance(request.thinking, dict) and request.thinking.get("type") == "adaptive"
        else None
    )

    logger.debug(
        f"Converting Anthropic request: model={request.model} -> {model_id}, "
        f"messages={len(unified_messages)}, tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}, "
        f"thinking_enabled={thinking_config.enabled}, thinking_budget={thinking_config.budget_tokens}, "
        f"thinking_effort={thinking_config.effort}"
    )

    # Use core function to build payload
    result = await build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=thinking_config,
        native_thinking=native_thinking,
        default_effort=NATIVE_EFFORT_DEFAULT_ANTHROPIC,
        request_audit=request_audit,
        budget=image_budget,
    )

    return result.payload
