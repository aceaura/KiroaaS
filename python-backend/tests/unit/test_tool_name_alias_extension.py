"""
Tests for the local tool-name aliasing extension.

The extension is installed at runtime by app_entry.py so the upstream kiro/
tree can stay sync-friendly.
"""

from extensions.tool_name_alias import (
    alias_for_tool_name,
    install_tool_name_aliasing,
    original_for_tool_name,
    uninstall_tool_name_aliasing,
)
import pytest


LONG_TOOL_NAME = (
    "mcp__plugin_everything-claude-code_github__create_pull_request_review"
)


@pytest.fixture(autouse=True)
def clean_tool_name_aliasing():
    uninstall_tool_name_aliasing()
    yield
    uninstall_tool_name_aliasing()


def test_alias_is_short_stable_and_reversible():
    first = alias_for_tool_name(LONG_TOOL_NAME)
    second = alias_for_tool_name(LONG_TOOL_NAME)

    assert first == second
    assert len(first) <= 64
    assert first != LONG_TOOL_NAME
    assert original_for_tool_name(first) == LONG_TOOL_NAME


def test_build_payload_sends_alias_to_kiro():
    install_tool_name_aliasing()

    from kiro.converters_core import (
        ThinkingConfig,
        UnifiedMessage,
        UnifiedTool,
        build_kiro_payload,
    )

    result = build_kiro_payload(
        messages=[UnifiedMessage(role="user", content="Call the tool")],
        system_prompt="",
        model_id="test-model",
        tools=[
            UnifiedTool(
                name=LONG_TOOL_NAME,
                description="Create a review",
                input_schema={"type": "object", "properties": {}},
            )
        ],
        conversation_id="test-conversation",
        profile_arn="",
        thinking_config=ThinkingConfig(enabled=False),
    )

    context = result.payload["conversationState"]["currentMessage"][
        "userInputMessage"
    ]["userInputMessageContext"]
    spec_name = context["tools"][0]["toolSpecification"]["name"]

    assert spec_name != LONG_TOOL_NAME
    assert len(spec_name) <= 64
    assert original_for_tool_name(spec_name) == LONG_TOOL_NAME


def test_alias_avoids_existing_short_tool_name():
    predicted_alias = alias_for_tool_name(LONG_TOOL_NAME)
    uninstall_tool_name_aliasing()
    install_tool_name_aliasing()

    from kiro.converters_core import (
        ThinkingConfig,
        UnifiedMessage,
        UnifiedTool,
        build_kiro_payload,
    )

    result = build_kiro_payload(
        messages=[UnifiedMessage(role="user", content="Call the tool")],
        system_prompt="",
        model_id="test-model",
        tools=[
            UnifiedTool(
                name=predicted_alias,
                description="Already short",
                input_schema={"type": "object", "properties": {}},
            ),
            UnifiedTool(
                name=LONG_TOOL_NAME,
                description="Create a review",
                input_schema={"type": "object", "properties": {}},
            ),
        ],
        conversation_id="test-conversation",
        profile_arn="",
        thinking_config=ThinkingConfig(enabled=False),
    )

    context = result.payload["conversationState"]["currentMessage"][
        "userInputMessage"
    ]["userInputMessageContext"]
    spec_names = [
        tool["toolSpecification"]["name"] for tool in context["tools"]
    ]

    assert spec_names[0] == predicted_alias
    assert spec_names[1] != predicted_alias
    assert len(spec_names[1]) <= 64
    assert original_for_tool_name(spec_names[1]) == LONG_TOOL_NAME


def test_history_tool_uses_are_aliased_for_kiro():
    install_tool_name_aliasing()

    from kiro.converters_core import (
        ThinkingConfig,
        UnifiedMessage,
        UnifiedTool,
        build_kiro_payload,
    )

    result = build_kiro_payload(
        messages=[
            UnifiedMessage(role="user", content="Use it"),
            UnifiedMessage(
                role="assistant",
                content="Calling tool",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": LONG_TOOL_NAME,
                            "arguments": "{}",
                        },
                    }
                ],
            ),
            UnifiedMessage(role="user", content="Continue"),
        ],
        system_prompt="",
        model_id="test-model",
        tools=[
            UnifiedTool(
                name=LONG_TOOL_NAME,
                description="Create a review",
                input_schema={"type": "object", "properties": {}},
            )
        ],
        conversation_id="test-conversation",
        profile_arn="",
        thinking_config=ThinkingConfig(enabled=False),
    )

    history = result.payload["conversationState"]["history"]
    tool_use_name = history[1]["assistantResponseMessage"]["toolUses"][0]["name"]

    assert tool_use_name != LONG_TOOL_NAME
    assert len(tool_use_name) <= 64
    assert original_for_tool_name(tool_use_name) == LONG_TOOL_NAME


def test_kiro_tool_calls_are_restored_for_client():
    install_tool_name_aliasing()

    from kiro.parsers import parse_bracket_tool_calls

    alias = alias_for_tool_name(LONG_TOOL_NAME)
    calls = parse_bracket_tool_calls(f'[Called {alias} with args: {{"x": 1}}]')

    assert calls[0]["function"]["name"] == LONG_TOOL_NAME
