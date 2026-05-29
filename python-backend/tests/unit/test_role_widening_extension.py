"""
Tests for the local role-widening extension.

Newer Claude Code clients embed role="system" inline in the messages
array. The extension widens AnthropicMessage.role at runtime so the
request reaches converters_core.normalize_message_roles() (which folds
unknown roles into 'user') instead of 422'ing at Pydantic validation.
"""

import pytest
from pydantic import ValidationError

from extensions.role_widening import (
    install_role_widening,
    uninstall_role_widening,
)


@pytest.fixture(autouse=True)
def clean_role_widening():
    uninstall_role_widening()
    yield
    uninstall_role_widening()


def _request(role: str):
    from kiro.models_anthropic import AnthropicMessagesRequest

    return AnthropicMessagesRequest(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": role, "content": "context blob"},
        ],
    )


def test_inline_system_role_rejected_before_install():
    with pytest.raises(ValidationError):
        _request("system")


def test_inline_system_role_accepted_after_install():
    install_role_widening()

    req = _request("system")

    assert [m.role for m in req.messages] == ["user", "system"]


def test_other_invalid_roles_still_rejected_after_install():
    install_role_widening()

    with pytest.raises(ValidationError):
        _request("developer")


def test_uninstall_restores_strict_validation():
    install_role_widening()
    _request("system")  # sanity check that it was installed

    uninstall_role_widening()

    with pytest.raises(ValidationError):
        _request("system")
