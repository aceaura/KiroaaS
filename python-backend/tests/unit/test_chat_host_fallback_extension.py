# -*- coding: utf-8 -*-

"""
Tests for the chat-host fallback extension.

runtime.kiro.dev requires a profileArn on every chat request. Free-tier
Builder ID accounts have no profileArn and cannot fetch one, so the extension
redirects KiroAuthManager.api_host (chat) to q.{region}.amazonaws.com — but
ONLY for accounts that lack a profileArn. Accounts that carry a profileArn
(paid) keep using runtime.kiro.dev untouched.
"""

import pytest

from kiro.auth import KiroAuthManager
from extensions.chat_host_fallback import (
    install_chat_host_fallback,
    uninstall_chat_host_fallback,
)


@pytest.fixture(autouse=True)
def clean_fallback():
    uninstall_chat_host_fallback()
    install_chat_host_fallback()
    yield
    uninstall_chat_host_fallback()


def _manager(profile_arn=None, region="us-east-1"):
    return KiroAuthManager(refresh_token="rt", profile_arn=profile_arn, region=region)


def test_api_host_redirected_when_no_profile_arn():
    manager = _manager(profile_arn=None)
    # Sanity: upstream config resolves api_host to runtime.kiro.dev.
    assert manager._api_host == "https://runtime.us-east-1.kiro.dev"
    # No profileArn → chat must fall back to q.amazonaws.com.
    assert manager.api_host == "https://q.us-east-1.amazonaws.com"


def test_api_host_untouched_when_profile_arn_present():
    manager = _manager(profile_arn="arn:aws:codewhisperer:us-east-1:123:profile/abc")
    # Paid account with a profileArn keeps using runtime.kiro.dev.
    assert manager.api_host == "https://runtime.us-east-1.kiro.dev"


def test_region_preserved_in_redirect():
    manager = _manager(profile_arn=None, region="eu-west-1")
    assert manager.api_host == "https://q.eu-west-1.amazonaws.com"


def test_non_runtime_api_host_passes_through():
    manager = _manager(profile_arn=None)
    # An already-q api_host (old endpoint) must not be rewritten.
    manager._api_host = "https://q.us-east-1.amazonaws.com"
    assert manager.api_host == "https://q.us-east-1.amazonaws.com"


def test_self_corrects_when_profile_arn_backfilled():
    manager = _manager(profile_arn=None)
    assert manager.api_host == "https://q.us-east-1.amazonaws.com"
    # Lazy getter: once a profileArn is backfilled, chat returns to runtime.
    manager._profile_arn = "arn:aws:codewhisperer:us-east-1:123:profile/abc"
    assert manager.api_host == "https://runtime.us-east-1.kiro.dev"


def test_uninstall_restores_api_host():
    manager = _manager(profile_arn=None)
    assert manager.api_host == "https://q.us-east-1.amazonaws.com"

    uninstall_chat_host_fallback()

    # Original property restored: api_host reflects raw _api_host again.
    assert manager.api_host == "https://runtime.us-east-1.kiro.dev"
