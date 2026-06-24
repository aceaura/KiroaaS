"""
Tests for the local billing-header strip extension.

Claude Code (2.1.x) prepends an ``x-anthropic-billing-header: ...`` line as the
first system block. extract_system_prompt() joins all system blocks into one
string, so that line lands at the front of the forwarded prompt and poisons
upstream prompt-cache prefixes. The extension wraps extract_system_prompt() at
runtime to drop the leading attribution line.
"""

import pytest

import kiro.converters_anthropic as converters_anthropic
from extensions.billing_header_strip import (
    install_billing_header_strip,
    strip_billing_attribution,
    uninstall_billing_header_strip,
)


@pytest.fixture(autouse=True)
def clean_billing_strip(monkeypatch):
    # Default-on for most tests; individual tests override as needed.
    monkeypatch.delenv("STRIP_BILLING_HEADER", raising=False)
    uninstall_billing_header_strip()
    yield
    uninstall_billing_header_strip()


# A realistic captured Claude Code system field: billing block first, then the
# actual system prompt blocks (cache_control is ignored by extract_system_prompt).
def _system_with_billing(header: str):
    return [
        {"type": "text", "text": header},
        {
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_helper_strips_version_only_header():
    text = (
        "x-anthropic-billing-header: cc_version=2.1.186.004; cc_entrypoint=claude-vscode;\n"
        "You are Claude Code."
    )
    assert strip_billing_attribution(text) == "You are Claude Code."


def test_helper_strips_cch_variant():
    text = (
        "x-anthropic-billing-header: cc_version=2.1.153.d02; cc_entrypoint=sdk-cli; cch=a1b2c;\n"
        "You are Claude Code."
    )
    assert strip_billing_attribution(text) == "You are Claude Code."


def test_helper_passthrough_without_header():
    text = "You are a helpful assistant.\nSecond line."
    assert strip_billing_attribution(text) == text


def test_helper_does_not_strip_when_not_leading():
    # The header is only stripped when it leads the prompt (matches upstream's
    # first-block behavior); a mid-prompt occurrence is left untouched.
    text = "Real prompt line.\nx-anthropic-billing-header: cc_version=1;"
    assert strip_billing_attribution(text) == text


def test_billing_block_stripped_after_install():
    install_billing_header_strip()

    system = _system_with_billing(
        "x-anthropic-billing-header: cc_version=2.1.186.004; cc_entrypoint=claude-vscode;"
    )
    result = converters_anthropic.extract_system_prompt(system)

    assert "x-anthropic-billing-header" not in result
    assert result == "You are Claude Code, Anthropic's official CLI for Claude."


def test_cch_token_stripped_after_install():
    install_billing_header_strip()

    system = _system_with_billing(
        "x-anthropic-billing-header: cc_version=2.1.153.d02; cc_entrypoint=sdk-cli; cch=9f3e1;"
    )
    result = converters_anthropic.extract_system_prompt(system)

    assert "cch=" not in result
    assert "x-anthropic-billing-header" not in result


def test_plain_string_system_unaffected_in_content():
    install_billing_header_strip()

    result = converters_anthropic.extract_system_prompt("You are helpful")

    assert result == "You are helpful"


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("STRIP_BILLING_HEADER", "false")
    install_billing_header_strip()

    header = "x-anthropic-billing-header: cc_version=2.1.186.004; cc_entrypoint=cli;"
    result = converters_anthropic.extract_system_prompt(_system_with_billing(header))

    # Gate is off: the line is preserved (extract_system_prompt joins with "\n").
    assert "x-anthropic-billing-header" in result


def test_uninstall_restores_original():
    install_billing_header_strip()
    header = "x-anthropic-billing-header: cc_version=2.1.186.004; cc_entrypoint=cli;"
    assert "x-anthropic-billing-header" not in converters_anthropic.extract_system_prompt(
        _system_with_billing(header)
    )

    uninstall_billing_header_strip()

    # Back to upstream behavior: the billing line survives the join.
    assert "x-anthropic-billing-header" in converters_anthropic.extract_system_prompt(
        _system_with_billing(header)
    )
